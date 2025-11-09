import ccxt.async_support as ccxt
import pandas as pd
import time
import logging
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
import os
import asyncio
# Import backtest module for strategy validation
import backtest  # Assuming backtest.py is in same dir
# Load API credentials from .env file
load_dotenv()
API_KEY = os.getenv('API_KEY')
API_SECRET = os.getenv('API_SECRET')
if not API_KEY or not API_SECRET:
    raise EnvironmentError("API_KEY and API_SECRET must be in .env file")
# =============================================================================
# LIVE TRADING CONFIGURATION (Trader Variables)
# =============================================================================
LIVE_CONFIG = {
    'candle_timeframe': '1s',
    'short_window': 10,             # Will be overridden by backtest if successful
    'long_window': 50,             # Will be overridden by backtest if successful
    'position_size_pct': 0.2,     # RECOMMEND: Reduce from 0.2 for safety
    'max_trade_usd': 1000.0,       # Maximum USD to risk per trade
    'stop_loss_pct': 0.1,         # Tightened to 0.01 for volatility in 1s timeframe
    'symbol': 'BTC/USDT',          # Trading pair
    'base_currency': 'BTC',
    'quote_currency': 'USDT',
    'taker_fee_pct': 0.001,         # 0.1% default; adjust for your tier
    'paper_trading': os.getenv('PAPER_TRADING', 'True').lower() == 'true',
}
# =============================================================================
# BACKTEST CONFIGURATION (Backtest Variables)
# =============================================================================
# API rate limits (Binance: ~1200 req/min).
BACKTEST_CONFIG = {
    'data_limit': 5000,  # 1000 ~17min data 5000 (~1.4hr data)
    'top_combos_to_display': 10,   # Number of top combinations to print
}
# =============================================================================
# SHARED/CORE SETUP (Not specific to backtest or live)
# =============================================================================
# Setup logging to file and console with timestamp format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[
        logging.FileHandler("hf_bot.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
# Initialize Binance spot exchange with rate limiting (async)
exchange = ccxt.binance({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'enableRateLimit': True,
    'options': {'defaultType': 'spot'},
    # 'verbose': True  # Logs raw requests
})
# Log trading mode (paper or real)
if LIVE_CONFIG['paper_trading']:
    logger.info("PAPER TRADING MODE - NO REAL ORDERS")
    logger.info(
        "Safety First: Test paper mode for 1hr, expect 100+ trades with 1s + 0.1% fees + tight stops = high churn/costs.")
else:
    logger.warning("REAL TRADING MODE - ORDERS WILL EXECUTE!")
# Global position state dictionary
position = {
    'in_position': False,
    'entry_price': 0.0,
    'amount': 0.0,
    'highest_price': 0.0,
    'entry_fee_usd': 0.0  # NEW: Track entry fee for accurate P/L
}
# Global for candles
df_candles = pd.DataFrame()
# Global current price
current_price = 0.0
last_candle_ts = 0
# NEW: Cache for balance fetches to reduce API calls
last_balance_fetch = 0
cached_bal = {'usdt': 0.0, 'btc': 0.0}
BALANCE_CACHE_REFRESH_SEC = 30  # Refresh every 30 seconds


async def get_real_balances():
    """
    Fetch real USDT and BTC balances from Binance (always real).
    """
    try:
        bal = await exchange.fetch_balance()
        return {
            'usdt': bal['total'].get(LIVE_CONFIG['quote_currency'], 0.0),
            'btc': bal['total'].get(LIVE_CONFIG['base_currency'], 0.0)
        }
    except Exception as e:
        logger.error(f"Real balance fetch error: {e}")
        return {'usdt': 0.0, 'btc': 0.0}


async def get_balance():
    """
    Get USDT balance from exchange (always real, for trading decisions).
    """
    global last_balance_fetch, cached_bal
    now = time.time()
    if now - last_balance_fetch > BALANCE_CACHE_REFRESH_SEC:
        try:
            bal = await exchange.fetch_balance()
            cached_bal['usdt'] = bal['total'].get(
                LIVE_CONFIG['quote_currency'], 0.0)
            cached_bal['btc'] = bal['total'].get(
                LIVE_CONFIG['base_currency'], 0.0)
            last_balance_fetch = now
            logger.debug(
                f"Balance cache refreshed at {datetime.now().strftime('%H:%M:%S')}")
        except Exception as e:
            logger.error(f"Balance fetch error: {e}")
            # Return cached or 0
            pass
    return cached_bal['usdt']


def calculate_position_size(usdt_balance, price):
    """
    Calculate BTC amount based on risk rules.
    """
    usd = min(usdt_balance *
              LIVE_CONFIG['position_size_pct'], LIVE_CONFIG['max_trade_usd'])
    amount = usd / price
    return round(amount, 6)


async def place_market_order(side, amount):
    """
    Place or simulate market order.
    """
    if LIVE_CONFIG['paper_trading']:
        logger.info(
            f"[PAPER] {side.upper()} {amount:.6f} {LIVE_CONFIG['base_currency']} @ market")
        return {'id': f"paper_{int(time.time())}", 'status': 'closed'}
    try:
        order = await exchange.create_order(
            LIVE_CONFIG['symbol'], 'market', side, amount)
        logger.info(
            f"REAL {side.upper()} {amount:.6f} {LIVE_CONFIG['base_currency']} | ID: {order['id']}")
        return order
    except Exception as e:
        logger.error(f"Order failed: {e}")
        return None


def calculate_fees(side, amount, price):
    """
    NEW: Calculate approximate taker fees in USD.
    Buy: Fee on USDT (quote).
    Sell: Fee on BTC (base), converted to USD.
    """
    fee_pct = LIVE_CONFIG['taker_fee_pct']
    if side == 'buy':
        return amount * price * fee_pct  # Fee deducted from USDT
    else:  # sell
        return amount * price * fee_pct  # Fee deducted from BTC, valued in USD


async def check_trailing_stop(price):
    """
    Check and enforce trailing stop-loss.
    """
    global position
    if not position['in_position']:
        return False
    if price > position['highest_price']:
        position['highest_price'] = price
        logger.info(f"New high: ${price:,.2f}")
    stop = position['highest_price'] * (1 - LIVE_CONFIG['stop_loss_pct'])
    if price <= stop:
        logger.warning(f"TRAIL STOP HIT @ ${price:,.2f}")
        order = await place_market_order('sell', position['amount'])
        if order:
            # NEW: Adjust P/L for round-trip fees
            entry_fee = position['entry_fee_usd']
            exit_fee = calculate_fees('sell', position['amount'], price)
            gross_pnl = (price - position['entry_price']) * position['amount']
            pnl = gross_pnl - entry_fee - exit_fee
            logger.info(
                f"Closed on stop | Gross P/L: {gross_pnl:+.2f} | Fees: {entry_fee + exit_fee:.2f} | Net P/L: {pnl:+.2f} {LIVE_CONFIG['quote_currency']}")
            position.update(
                {'in_position': False, 'entry_price': 0, 'amount': 0, 'highest_price': 0, 'entry_fee_usd': 0})
        return True
    return False


async def run_signal_strategy():
    """
    Execute MA crossover strategy logic on new closed candle.
    """
    global position, df_candles
    if len(df_candles) < LIVE_CONFIG['long_window']:
        logger.warning(f"Insufficient data: {len(df_candles)} candles")
        return
    close = df_candles['close']
    price = close.iloc[-1]
    short_ma = close.rolling(LIVE_CONFIG['short_window']).mean()
    long_ma = close.rolling(LIVE_CONFIG['long_window']).mean()
    curr_s, prev_s = short_ma.iloc[-1], short_ma.iloc[-2]
    curr_l, prev_l = long_ma.iloc[-1], long_ma.iloc[-2]
    if position['in_position']:
        if prev_s >= prev_l and curr_s < curr_l:
            # Death cross exit
            order = await place_market_order('sell', position['amount'])
            if order:
                # NEW: Adjust P/L for round-trip fees
                entry_fee = position['entry_fee_usd']
                exit_fee = calculate_fees('sell', position['amount'], price)
                gross_pnl = (
                    price - position['entry_price']) * position['amount']
                pnl = gross_pnl - entry_fee - exit_fee
                logger.info(
                    f"Death Cross exit | Gross P/L: {gross_pnl:+.2f} | Fees: {entry_fee + exit_fee:.2f} | Net P/L: {pnl:+.2f} {LIVE_CONFIG['quote_currency']}")
                position.update(
                    {'in_position': False, 'entry_price': 0, 'amount': 0, 'highest_price': 0, 'entry_fee_usd': 0})
        else:
            logger.info("Holding - trailing stop active")
        return
    if prev_s <= prev_l and curr_s > curr_l:
        usdt = await get_balance()
        if usdt < 10:
            logger.warning("Low balance")
            return
        amount = calculate_position_size(usdt, price)
        if amount < 0.0001:
            return
        order = await place_market_order('buy', amount)
        if order:
            # NEW: Track entry fee for later P/L
            entry_fee = calculate_fees('buy', amount, price)
            position.update({
                'in_position': True,
                'entry_price': price,
                'amount': amount,
                'highest_price': price,
                'entry_fee_usd': entry_fee
            })
            logger.info(
                f"BUY {amount:.6f} {LIVE_CONFIG['base_currency']} @ ${price:,.2f} | Est. Fee: {entry_fee:.2f} {LIVE_CONFIG['quote_currency']}")


async def candle_poller():
    """
    Poller for new closed candles.
    """
    global df_candles, last_candle_ts
    symbol = LIVE_CONFIG['symbol']
    timeframe = LIVE_CONFIG['candle_timeframe']
    logger.info("Starting candle poller...")
    cycle_count = 0  # Initialize cycle counter for heartbeat
    # Initial load
    try:
        ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, limit=300)
        df_candles = pd.DataFrame(
            ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df_candles['timestamp'] = pd.to_datetime(
            df_candles['timestamp'], unit='ms')
        df_candles.set_index('timestamp', inplace=True)
        last_candle_ts = int(df_candles.index[-1].timestamp())
        logger.info(
            f"Loaded {len(df_candles)} historical candles up to {df_candles.index[-1]}")
    except Exception as e:
        logger.error(f"Failed to fetch initial candles: {e}")
        return
    while True:
        try:
            await asyncio.sleep(1)  # Poll every 1s
            cycle_count += 1
            if cycle_count % 60 == 0:  # Log every ~60s
                logger.info(
                    f"Candle poller alive: {len(df_candles)} candles, last TS {pd.to_datetime(last_candle_ts, unit='s')}")
            new_ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, limit=2)
            if len(new_ohlcv) < 2:
                continue
            new_df = pd.DataFrame(
                new_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            new_df['timestamp'] = pd.to_datetime(
                new_df['timestamp'], unit='ms')
            new_df.set_index('timestamp', inplace=True)
            new_ts = int(new_df.index[-1].timestamp())
            if new_ts > last_candle_ts:
                # New candle closed
                new_row = new_df.iloc[-1]
                df_candles = pd.concat([df_candles, new_row.to_frame().T])
                df_candles = df_candles.tail(300)
                last_candle_ts = new_ts
                await run_signal_strategy()
        except Exception as e:
            logger.error(f"Candle poller error: {e}")
            await asyncio.sleep(1)


async def price_poller():
    """
    Poller for real-time price updates (for trailing stop).
    """
    global current_price
    symbol = LIVE_CONFIG['symbol']
    logger.info("Starting price poller...")
    price_cycle = 0  # Initialize cycle counter for heartbeat
    while True:
        try:
            ticker = await exchange.fetch_ticker(symbol)
            current_price = ticker['last']
            if position['in_position']:
                await check_trailing_stop(current_price)
            price_cycle += 1
            if price_cycle % 120 == 0:  # Log every ~60s
                logger.info(
                    f"Price poller alive: Current ${current_price:,.2f}")
            await asyncio.sleep(0.5)  # Poll every 0.5s
        except Exception as e:
            logger.error(f"Price poller error: {e}")
            await asyncio.sleep(1)


async def main():
    logger.info("HF MA BOT STARTED")
    logger.info(f"{LIVE_CONFIG['candle_timeframe']} | {LIVE_CONFIG['short_window']}/{LIVE_CONFIG['long_window']} | {LIVE_CONFIG['position_size_pct']*100}% risk | ${LIVE_CONFIG['max_trade_usd']} cap | {LIVE_CONFIG['stop_loss_pct']*100}% trail | Fee: {LIVE_CONFIG['taker_fee_pct']*100}% | Polling for real-time data | Safety: 1s + 0.1% fees + tight stops = high churn/costs")
    if not LIVE_CONFIG['paper_trading']:
        confirm = input("\nREAL TRADING! Type 'YES' to continue: ")
        if confirm != "YES":
            return
    print("\nRunning quick backtest...")
    try:
        # Unpack config as kwargs to match run_hf_backtest signature
        bt = backtest.run_hf_backtest(limit=BACKTEST_CONFIG['data_limit'])
        if not bt.empty:
            LIVE_CONFIG['short_window'] = int(
                bt.iloc[0]['short'])  # FIX: Cast to int
            LIVE_CONFIG['long_window'] = int(
                bt.iloc[0]['long'])   # FIX: Cast to int
            logger.info(
                f"Backtest override: Using MA {LIVE_CONFIG['short_window']}/{LIVE_CONFIG['long_window']} ({bt.iloc[0]['return_%']}%)")
            print(bt.head(BACKTEST_CONFIG['top_combos_to_display']).to_string(
                index=False))
            print(
                f"\nTop combo: {bt.iloc[0]['short']}/{bt.iloc[0]['long']} → {bt.iloc[0]['return_%']}%")
        else:
            logger.warning(
                "Backtest returned empty results; using LIVE_CONFIG defaults")
    except Exception as e:
        logger.error(f"Backtest failed: {e}. Using LIVE_CONFIG defaults")
        bt = pd.DataFrame()
    logger.warning(
        "Backtest Validation: backtest.py likely optimizes returns, but for 1s, add slippage/sim fees to avoid overfit.")
    # Log initial real balances
    real_bal = await get_real_balances()
    logger.info(
        f"Initial Real Balances - {LIVE_CONFIG['quote_currency']}: {real_bal['usdt']:,.2f} | {LIVE_CONFIG['base_currency']}: {real_bal['btc']:.6f}")
    # Initialize current price and position if existing BTC balance
    try:
        ticker = await exchange.fetch_ticker(LIVE_CONFIG['symbol'])
        global current_price
        current_price = ticker['last']
        if real_bal['btc'] > 0.0001:
            position.update({
                'in_position': True,
                'entry_price': current_price,
                'amount': real_bal['btc'],
                'highest_price': current_price,
                'entry_fee_usd': 0.0  # No fee for existing position
            })
            logger.info(
                f"Initialized existing position: {real_bal['btc']:.6f} {LIVE_CONFIG['base_currency']} @ ${current_price:,.2f} (for active management)")
    except Exception as e:
        logger.error(f"Failed to fetch initial ticker for position init: {e}")
    logger.info("Bot LIVE with Polling. Ctrl+C to stop.\n")
    # Start poller tasks
    candle_task = asyncio.create_task(candle_poller())
    price_task = asyncio.create_task(price_poller())
    try:
        await asyncio.gather(candle_task, price_task)
    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    finally:
        await exchange.close()
if __name__ == "__main__":
    asyncio.run(main())
