import ccxt.async_support as ccxt
import pandas as pd
import time
import logging
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
import os
import asyncio
# Import backtest module for strategy validation (assumes spot long-only compatible)
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
    'candle_timeframe': '1h',
    'short_window': 12,             # Start with EMA 12 (common for 1h)
    'long_window': 50,              # EMA 50
    'position_size_pct': 0.15,      # 15% of available USDT balance per trade
    'max_trade_usd': 500.0,         # Maximum USD notional per trade
    'take_profit_pct': 0.05,        # Increased to 5% for longer timeframe
    'trailing_stop_pct': 0.03,      # Increased to 3% trail for 1h
    'symbol': 'BTC/USDT',           # Trading pair for spot market
    'base_currency': 'BTC',
    'quote_currency': 'USDT',
    'taker_fee_pct': 0.001,         # 0.1% taker fee (adjust for your tier)
    'market_type': 'spot',          # Fixed to spot for long-only trading
    'paper_trading': os.getenv('PAPER_TRADING', 'True').lower() == 'true',
    'adx_period': 14,               # ADX period for trend filter
    'adx_threshold': 25,            # ADX > 25 for strong trend
    'rsi_period': 14,               # RSI period
    'rsi_overbought': 70,           # Avoid buy if RSI > 70
    'rsi_oversold': 30,             # Additional close if RSI < 30 (optional)
}
# =============================================================================
# BACKTEST CONFIGURATION (Backtest Variables)
# =============================================================================
BACKTEST_CONFIG = {
    'data_limit': 2000,             # ~83 days of 1h data for backtest
    'top_combos_to_display': 10,    # Number of top MA window combinations to display
}
# =============================================================================
# SHARED/CORE SETUP
# =============================================================================
# Setup logging to file and console with timestamps
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[
        logging.FileHandler("spot_ma_bot.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
# Initialize exchange for spot trading
symbol = LIVE_CONFIG['symbol']
exchange = ccxt.binance({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'enableRateLimit': True,
    'options': {'defaultType': 'spot'},
    # 'verbose': True  # Uncomment for raw API request logs
})
LIVE_CONFIG['symbol'] = symbol
# Log trading mode
if LIVE_CONFIG['paper_trading']:
    logger.info("PAPER TRADING MODE - NO REAL ORDERS PLACED")
    logger.info(
        "Safety: Test in paper mode first; fewer trades on 1h but monitor fees/stops.")
else:
    logger.warning("REAL TRADING MODE - ORDERS WILL EXECUTE ON BINANCE SPOT!")
logger.info(
    f"SPOT LONG-ONLY MODE: {symbol} (EMA crossover with ADX/RSI filters)")
# Global position state (long-only)
position = {
    'type': 'none',                 # 'long' or 'none'
    'size': 0.0,                    # Positive BTC amount for long
    'entry_price': 0.0,
    'highest_price': 0.0,           # Tracks high for trailing stop
    'entry_fee_usd': 0.0
}
# Global data structures
df_candles = pd.DataFrame()
current_price = 0.0
last_candle_ts = 0
# Cache for balance fetches to reduce API calls
last_balance_fetch = 0
cached_bal = {'usdt': 0.0, 'btc': 0.0}
BALANCE_CACHE_REFRESH_SEC = 30  # Refresh every 30 seconds


async def get_real_balances():
    """
    Fetch real balances for spot: free USDT and total BTC holdings.
    """
    try:
        bal = await exchange.fetch_balance()
        quote_free = bal[LIVE_CONFIG['quote_currency']]['free']
        base_total = bal[LIVE_CONFIG['base_currency']]['total']
        return {'usdt': quote_free, 'btc': base_total}
    except Exception as e:
        logger.error(f"Balance fetch error: {e}")
        return {'usdt': 0.0, 'btc': 0.0}


async def log_balance_after_trade():
    """
    Log estimated total balance in USDT after a trade (colored for console).
    """
    global current_price
    try:
        real_bal = await get_real_balances()
        estimated_usdt = real_bal['usdt'] + real_bal['btc'] * current_price
        color_msg = f"\033[92mEstimated Balance: {estimated_usdt:,.2f} {LIVE_CONFIG['quote_currency']} (USDT: {real_bal['usdt']:,.2f}, {LIVE_CONFIG['base_currency']}: {real_bal['btc']:.6f} @ ${current_price:,.2f})\033[0m"
        logger.info(color_msg)
    except Exception as e:
        logger.error(f"Failed to log balance after trade: {e}")


async def get_position_info():
    """
    Get current spot position: treat BTC holdings as long position.
    """
    try:
        bal = await exchange.fetch_balance()
        size = bal[LIVE_CONFIG['base_currency']]['total']
        if size > 0.0001:
            # Approx entry
            return {'size': size, 'side': 'long', 'entry_price': current_price}
        return {'size': 0, 'side': None, 'entry_price': 0}
    except Exception as e:
        logger.error(f"Position info fetch error: {e}")
        return {'size': 0, 'side': None, 'entry_price': 0}


async def get_balance():
    """
    Get cached free USDT balance (for position sizing).
    """
    global last_balance_fetch, cached_bal
    now = time.time()
    if now - last_balance_fetch > BALANCE_CACHE_REFRESH_SEC:
        try:
            real_bal = await get_real_balances()
            cached_bal['usdt'] = real_bal['usdt']
            cached_bal['btc'] = real_bal['btc']
            last_balance_fetch = now
            logger.debug(
                f"Balance cache refreshed at {datetime.now().strftime('%H:%M:%S')}")
        except Exception as e:
            logger.error(f"Balance fetch error: {e}")
    return cached_bal['usdt']


def calculate_position_size(usdt_balance, price):
    """
    Calculate BTC amount based on risk rules: min(pct of balance, max USD) / price.
    """
    usd = min(usdt_balance *
              LIVE_CONFIG['position_size_pct'], LIVE_CONFIG['max_trade_usd'])
    amount = usd / price
    return round(amount, 6)


async def place_market_order(side, amount):
    """
    Place or simulate market order on spot (buy/sell BTC with USDT).
    Amount is positive BTC quantity.
    """
    if LIVE_CONFIG['paper_trading']:
        logger.info(
            f"[PAPER] {side.upper()} {amount:.6f} {LIVE_CONFIG['base_currency']} @ market")
        return {'id': f"paper_{int(time.time())}", 'status': 'closed'}
    # Fetch fresh balance before order
    real_bal = await get_real_balances()
    required_usd = amount * current_price
    fee_usd = required_usd * LIVE_CONFIG['taker_fee_pct']
    if side == 'buy':
        if real_bal['usdt'] < required_usd + fee_usd:
            logger.error(
                f"Insufficient USDT for BUY {amount:.6f} BTC @ ${current_price:,.2f}: Need {required_usd + fee_usd:,.2f}, Have {real_bal['usdt']:,.2f}")
            return None
        logger.info(
            f"Balance OK for BUY: USDT {real_bal['usdt']:,.2f} >= {required_usd + fee_usd:,.2f}")
    elif side == 'sell':
        if real_bal['btc'] < amount:
            logger.error(
                f"Insufficient BTC for SELL {amount:.6f} BTC @ ${current_price:,.2f}: Need {amount:.6f}, Have {real_bal['btc']:.6f}")
            return None
        logger.info(
            f"Balance OK for SELL: BTC {real_bal['btc']:.6f} >= {amount:.6f}")
    try:
        order = await exchange.create_order(LIVE_CONFIG['symbol'], 'market', side, amount)
        logger.info(
            f"REAL {side.upper()} {amount:.6f} {LIVE_CONFIG['base_currency']} | Order ID: {order['id']}")
        return order
    except Exception as e:
        logger.error(f"Order failed: {e}")
        return None


def calculate_fees(side, amount, price):
    """
    Calculate approximate taker fees in USD based on notional value.
    """
    notional = amount * price
    return notional * LIVE_CONFIG['taker_fee_pct']


def ema(series, period):
    """
    Calculate Exponential Moving Average (EMA).
    """
    return series.ewm(span=period, adjust=False).mean()


def rsi(series, period):
    """
    Calculate Relative Strength Index (RSI).
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_adx(high, low, close, period):
    """
    Calculate Average Directional Index (ADX) for trend strength.
    """
    tr = pd.DataFrame({
        'hl': high - low,
        'hc': abs(high - close.shift()),
        'lc': abs(low - close.shift())
    }).max(axis=1)
    atr = tr.rolling(period).mean()
    up = close.diff().clip(lower=0)
    down = -close.diff().clip(upper=0)
    plus_di = 100 * (up.ewm(span=period).mean() / atr)
    minus_di = 100 * (down.ewm(span=period).mean() / atr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(span=period).mean()
    return adx


async def check_trailing_stop(price):
    """
    Check trailing stop-loss and fixed take-profit for long position.
    Updates highest price and triggers close if hit.
    """
    global position
    if position['type'] != 'long' or position['size'] <= 0:
        return False
    # Update trailing high
    if price > position['highest_price']:
        position['highest_price'] = price
    # Calculate levels
    stop_price = position['highest_price'] * \
        (1 - LIVE_CONFIG['trailing_stop_pct'])
    tp_price = position['entry_price'] * (1 + LIVE_CONFIG['take_profit_pct'])
    # Check stop-loss first (protective)
    if price <= stop_price:
        logger.warning(f"TRAILING STOP HIT (LONG) @ ${price:,.2f}")
        order = await place_market_order('sell', position['size'])
        if order:
            entry_fee = position['entry_fee_usd']
            exit_fee = calculate_fees('sell', position['size'], price)
            gross_pnl = (price - position['entry_price']) * position['size']
            pnl = gross_pnl - entry_fee - exit_fee
            logger.info(
                f"Long closed on TRAILING STOP | Gross P/L: {gross_pnl:+.2f} | Fees: {entry_fee + exit_fee:.2f} | Net P/L: {pnl:+.2f} {LIVE_CONFIG['quote_currency']}")
            position = {
                'type': 'none',
                'size': 0.0,
                'entry_price': 0.0,
                'highest_price': 0.0,
                'entry_fee_usd': 0.0
            }
            await log_balance_after_trade()
        return True
    # Check take-profit
    elif price >= tp_price:
        logger.warning(f"TAKE PROFIT HIT (LONG) @ ${price:,.2f}")
        order = await place_market_order('sell', position['size'])
        if order:
            entry_fee = position['entry_fee_usd']
            exit_fee = calculate_fees('sell', position['size'], price)
            gross_pnl = (price - position['entry_price']) * position['size']
            pnl = gross_pnl - entry_fee - exit_fee
            logger.info(
                f"Long closed on TAKE PROFIT | Gross P/L: {gross_pnl:+.2f} | Fees: {entry_fee + exit_fee:.2f} | Net P/L: {pnl:+.2f} {LIVE_CONFIG['quote_currency']}")
            position = {
                'type': 'none',
                'size': 0.0,
                'entry_price': 0.0,
                'highest_price': 0.0,
                'entry_fee_usd': 0.0
            }
            await log_balance_after_trade()
        return True
    return False


async def run_signal_strategy():
    """
    Execute EMA crossover strategy for spot long-only with ADX and RSI filters.
    - Bull cross (short EMA > long EMA) + ADX > threshold + RSI < overbought: Open long if flat.
    - Bear cross (short EMA < long EMA) or RSI > overbought: Close long if open.
    - No cross: Hold if in position.
    """
    global position, df_candles
    if len(df_candles) < max(LIVE_CONFIG['long_window'], LIVE_CONFIG['rsi_period'], LIVE_CONFIG['adx_period']):
        logger.warning(f"Insufficient data: {len(df_candles)} candles")
        return
    close = df_candles['close']
    high = df_candles['high']
    low = df_candles['low']
    price = close.iloc[-1]
    short_ema = ema(close, LIVE_CONFIG['short_window'])
    long_ema = ema(close, LIVE_CONFIG['long_window'])
    curr_s, prev_s = short_ema.iloc[-1], short_ema.iloc[-2]
    curr_l, prev_l = long_ema.iloc[-1], long_ema.iloc[-2]
    bull_cross = (prev_s <= prev_l) and (curr_s > curr_l)
    bear_cross = (prev_s >= prev_l) and (curr_s < curr_l)
    adx = calculate_adx(high, low, close, LIVE_CONFIG['adx_period']).iloc[-1]
    rsi_val = rsi(close, LIVE_CONFIG['rsi_period']).iloc[-1]
    adx_ok = adx > LIVE_CONFIG['adx_threshold']
    rsi_buy_ok = rsi_val < LIVE_CONFIG['rsi_overbought']
    # Additional close condition
    rsi_sell_ok = rsi_val > LIVE_CONFIG['rsi_overbought']
    if bull_cross and adx_ok and rsi_buy_ok:
        if position['type'] == 'none':
            usdt = await get_balance()
            if usdt < 10:
                logger.warning("Low USDT balance for entry")
                return
            amount = calculate_position_size(usdt, price)
            if amount < 0.0001:
                logger.debug("Calculated position too small")
                return
            order = await place_market_order('buy', amount)
            if order:
                entry_fee = calculate_fees('buy', amount, price)
                position.update({
                    'type': 'long',
                    'size': amount,
                    'entry_price': price,
                    'highest_price': price,
                    'entry_fee_usd': entry_fee
                })
                logger.info(
                    f"OPEN LONG {amount:.6f} {LIVE_CONFIG['base_currency']} @ ${price:,.2f} | Est. Fee: {entry_fee:.2f} {LIVE_CONFIG['quote_currency']} | ADX: {adx:.2f} | RSI: {rsi_val:.2f}")
                await log_balance_after_trade()
    elif bear_cross or rsi_sell_ok:
        if position['type'] == 'long':
            # Sync size from real balance before close
            pos_info = await get_position_info()
            if pos_info['size'] > 0:
                position['size'] = pos_info['size']
            order = await place_market_order('sell', position['size'])
            if order:
                entry_fee = position['entry_fee_usd']
                exit_fee = calculate_fees('sell', position['size'], price)
                gross_pnl = (
                    price - position['entry_price']) * position['size']
                pnl = gross_pnl - entry_fee - exit_fee
                logger.info(
                    f"CLOSE LONG @ ${price:,.2f} | Gross P/L: {gross_pnl:+.2f} | Fees: {entry_fee + exit_fee:.2f} | Net P/L: {pnl:+.2f} {LIVE_CONFIG['quote_currency']} | ADX: {adx:.2f} | RSI: {rsi_val:.2f}")
                position = {
                    'type': 'none',
                    'size': 0.0,
                    'entry_price': 0.0,
                    'highest_price': 0.0,
                    'entry_fee_usd': 0.0
                }
                await log_balance_after_trade()
    else:
        # No crossover: hold position or stay flat
        if position['type'] == 'long':
            logger.debug(
                f"Holding LONG {position['size']:.6f} BTC - trailing active | ADX: {adx:.2f} | RSI: {rsi_val:.2f}")
        else:
            logger.debug("No signal - remaining flat")


async def candle_poller():
    """
    Poll for new closed candles and trigger strategy on update.
    For 1h, poll every 60s; maintain ~100 candles (~4 days).
    """
    global df_candles, last_candle_ts
    timeframe = LIVE_CONFIG['candle_timeframe']
    logger.info("Starting candle poller...")
    cycle_count = 0
    # Initial load
    try:
        ohlcv = await exchange.fetch_ohlcv(LIVE_CONFIG['symbol'], timeframe, limit=100)
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
            await asyncio.sleep(60)  # Poll every 60 seconds for 1h candles
            cycle_count += 1
            if cycle_count % 60 == 0:
                logger.info(
                    f"Candle poller alive: {len(df_candles)} candles, last {pd.to_datetime(last_candle_ts, unit='s')}")
            new_ohlcv = await exchange.fetch_ohlcv(LIVE_CONFIG['symbol'], timeframe, limit=2)
            if len(new_ohlcv) < 2:
                continue
            new_df = pd.DataFrame(new_ohlcv, columns=[
                                  'timestamp', 'open', 'high', 'low', 'close', 'volume'])
            new_df['timestamp'] = pd.to_datetime(
                new_df['timestamp'], unit='ms')
            new_df.set_index('timestamp', inplace=True)
            new_ts = int(new_df.index[-1].timestamp())
            if new_ts > last_candle_ts:
                # Append new closed candle
                new_row = new_df.iloc[-1]
                df_candles = pd.concat([df_candles, new_row.to_frame().T])
                df_candles = df_candles.tail(100)  # Keep recent history
                last_candle_ts = new_ts
                await run_signal_strategy()
        except Exception as e:
            logger.error(f"Candle poller error: {e}")
            await asyncio.sleep(60)


async def price_poller():
    """
    Poll real-time price for trailing stop checks (every 30s for 1h strategy).
    """
    global current_price
    logger.info("Starting price poller...")
    price_cycle = 0
    while True:
        try:
            ticker = await exchange.fetch_ticker(LIVE_CONFIG['symbol'])
            current_price = ticker['last']
            # Check trailing for active long
            if position['type'] == 'long':
                await check_trailing_stop(current_price)
            price_cycle += 1
            if price_cycle % 120 == 0:  # Log every ~60m
                logger.info(
                    f"Price poller alive: ${current_price:,.2f} | Position: {position['type']} {position['size']:.6f} BTC")
            await asyncio.sleep(30)  # Poll every 30s for 1h
        except Exception as e:
            logger.error(f"Price poller error: {e}")
            await asyncio.sleep(30)


async def main():
    """
    Main entry: Run backtest with ADX filter, init, and start pollers.
    """
    logger.info(
        "SPOT LONG-ONLY EMA CROSSOVER BOT STARTED (with ADX/RSI filters)")
    logger.info(f"{LIVE_CONFIG['candle_timeframe']} candles | EMA {LIVE_CONFIG['short_window']}/{LIVE_CONFIG['long_window']} | {LIVE_CONFIG['position_size_pct']*100}% risk | ${LIVE_CONFIG['max_trade_usd']} cap | {LIVE_CONFIG['trailing_stop_pct']*100}% trail stop | {LIVE_CONFIG['take_profit_pct']*100}% TP | {LIVE_CONFIG['taker_fee_pct']*100}% fee | ADX > {LIVE_CONFIG['adx_threshold']} | RSI < {LIVE_CONFIG['rsi_overbought']} for entry")
    if not LIVE_CONFIG['paper_trading']:
        confirm = input(
            "\nREAL TRADING CONFIRMATION! Type 'YES' to continue: ")
        if confirm != "YES":
            logger.info("Confirmation failed - exiting.")
            return
    print("\nRunning quick backtest for EMA optimization...")
    try:
        bt = backtest.run_hf_backtest(
            limit=BACKTEST_CONFIG['data_limit'], use_adx_filter=True)
        if not bt.empty:
            LIVE_CONFIG['short_window'] = int(bt.iloc[0]['short'])
            LIVE_CONFIG['long_window'] = int(bt.iloc[0]['long'])
            logger.info(
                f"Backtest override: Using EMA {LIVE_CONFIG['short_window']}/{LIVE_CONFIG['long_window']} ({bt.iloc[0]['return_%']}%)")
            print(bt.head(BACKTEST_CONFIG['top_combos_to_display']).to_string(
                index=False))
            print(
                f"\nTop combo: {bt.iloc[0]['short']}/{bt.iloc[0]['long']} → {bt.iloc[0]['return_%']}% return")
        else:
            logger.warning("Backtest returned empty; using config defaults")
    except Exception as e:
        logger.error(f"Backtest failed: {e}. Using config defaults")
        bt = pd.DataFrame()
    # logger.warning(
        # "Backtest Note: Optimized for spot with fees/slippage; review for realism.")
    # Log initial balances
    real_bal = await get_real_balances()
    logger.info(
        f"Initial Balances - {LIVE_CONFIG['quote_currency']}: {real_bal['usdt']:,.2f} | {LIVE_CONFIG['base_currency']}: {real_bal['btc']:.6f}")
    # Init current price and sync existing position
    try:
        ticker = await exchange.fetch_ticker(LIVE_CONFIG['symbol'])
        global current_price
        current_price = ticker['last']
        pos_info = await get_position_info()
        if pos_info['size'] > 0.0001:
            position.update({
                'type': 'long',
                'size': pos_info['size'],
                'entry_price': pos_info['entry_price'],
                'highest_price': current_price,
                'entry_fee_usd': 0.0  # Unknown for existing
            })
            logger.info(
                f"Initialized existing LONG: {position['size']:.6f} {LIVE_CONFIG['base_currency']} @ ${current_price:,.2f} (entry ~${pos_info['entry_price']:,.2f})")
    except Exception as e:
        logger.error(f"Initial ticker/position init error: {e}")
    logger.info("Bot LIVE with pollers running. Ctrl+C to stop.\n")
    # Start async tasks
    candle_task = asyncio.create_task(candle_poller())
    price_task = asyncio.create_task(price_poller())
    try:
        await asyncio.gather(candle_task, price_task)
    except KeyboardInterrupt:
        logger.info("Bot stopped by user.")
    finally:
        await exchange.close()
if __name__ == "__main__":
    asyncio.run(main())
