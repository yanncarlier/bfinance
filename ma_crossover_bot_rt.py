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
    'position_size_pct': 0.3,     # RECOMMEND: Reduce from 0.2 for safety
    'max_trade_usd': 1000.0,       # Maximum USD to risk per trade
    'stop_loss_pct': 0.01,         # Tightened to 0.01 for volatility in 1s timeframe
    # Trading pair (adjusted internally for futures)
    'symbol': 'BTC/USDT',
    'base_currency': 'BTC',
    'quote_currency': 'USDT',
    'taker_fee_pct': 0.001,         # 0.1% default; adjust for your tier
    # NEW: 'spot' or 'futures' (use 'futures' for long/short)
    # Keep as 'futures' if you want shorting; change to 'spot' for long-only (no leverage possible)
    'market_type': 'spot',
    # NEW: Leverage for futures (ignored for spot) — set to 1 to disable amplification
    'leverage': 1,
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
# Initialize exchange based on market type
market_type = LIVE_CONFIG['market_type']
symbol = LIVE_CONFIG['symbol']
if market_type == 'futures':
    symbol = symbol.replace('/', '')  # e.g., 'BTCUSDT'
    exchange = ccxt.binance({
        'apiKey': API_KEY,
        'secret': API_SECRET,
        'enableRateLimit': True,
        'options': {'defaultType': 'future'},
        # 'verbose': True  # Logs raw requests
    })
else:
    exchange = ccxt.binance({
        'apiKey': API_KEY,
        'secret': API_SECRET,
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'},
        # 'verbose': True  # Logs raw requests
    })
LIVE_CONFIG['symbol'] = symbol  # Update for consistency
# Log trading mode (paper or real)
if LIVE_CONFIG['paper_trading']:
    logger.info("PAPER TRADING MODE - NO REAL ORDERS")
    logger.info(
        "Safety First: Test paper mode for 1hr, expect 100+ trades with 1s + 0.1% fees + tight stops = high churn/costs.")
else:
    logger.warning("REAL TRADING MODE - ORDERS WILL EXECUTE!")
# Log market type
if market_type == 'futures':
    logger.info(
        f"FUTURES MODE: {symbol} with {LIVE_CONFIG['leverage']}x leverage (long/short enabled)")
else:
    logger.info(
        f"SPOT MODE: {symbol} (long-only; shorts limited to selling holdings)")
# Global position state dictionary - UPDATED for long/short
position = {
    'type': 'none',                # NEW: 'long', 'short', or 'none'
    'size': 0.0,                   # Positive for long, negative for short
    'entry_price': 0.0,
    'highest_price': 0.0,          # For long trailing stop
    'lowest_price': float('inf'),  # NEW: For short trailing target
    'entry_fee_usd': 0.0
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
    Fetch real quote (USDT) free balance and base position size (always real).
    For spot: free USDT and total BTC.
    For futures: free USDT and open position contracts (positive regardless of side).
    """
    try:
        bal = await exchange.fetch_balance()
        quote_free = bal[LIVE_CONFIG['quote_currency']]['free']
        if market_type == 'spot':
            base_total = bal[LIVE_CONFIG['base_currency']]['total']
            return {'usdt': quote_free, 'btc': base_total}
        else:
            # For futures, get from positions
            positions = await exchange.fetch_positions([LIVE_CONFIG['symbol']])
            base_pos = 0.0
            for pos in positions:
                if pos['symbol'] == LIVE_CONFIG['symbol'] and pos['contracts'] > 0:
                    base_pos = pos['contracts']
                    break
            return {'usdt': quote_free, 'btc': base_pos}
    except Exception as e:
        logger.error(f"Real balance fetch error: {e}")
        return {'usdt': 0.0, 'btc': 0.0}


async def log_balance_after_trade():
    """
    Log estimated total balance in USDT after a trade.
    """
    global current_price
    try:
        real_bal = await get_real_balances()
        estimated_usdt = real_bal['usdt'] + real_bal['btc'] * current_price
        # Use ANSI escape codes for green color on console (file will have codes, but readable)
        color_msg = f"\033[92mEstimated Balance: {estimated_usdt:,.2f} {LIVE_CONFIG['quote_currency']} (USDT: {real_bal['usdt']:,.2f}, {LIVE_CONFIG['base_currency']}: {real_bal['btc']:.6f} @ ${current_price:,.2f})\033[0m"
        logger.info(color_msg)
    except Exception as e:
        logger.error(f"Failed to log balance after trade: {e}")


async def get_position_info():
    """
    NEW: Get detailed position info (size, side, entry_price).
    For spot: Treat holdings as 'long'.
    """
    try:
        if market_type == 'spot':
            bal = await exchange.fetch_balance()
            size = bal[LIVE_CONFIG['base_currency']]['total']
            if size > 0.0001:
                # Approx entry
                return {'size': size, 'side': 'long', 'entry_price': current_price}
            return {'size': 0, 'side': None, 'entry_price': 0}
        else:
            positions = await exchange.fetch_positions([LIVE_CONFIG['symbol']])
            for pos in positions:
                if pos['contracts'] > 0:
                    return {
                        'size': pos['contracts'],
                        'side': pos['side'].lower(),
                        'entry_price': pos.get('entryPrice', current_price)
                    }
            return {'size': 0, 'side': None, 'entry_price': 0}
    except Exception as e:
        logger.error(f"Position info fetch error: {e}")
        return {'size': 0, 'side': None, 'entry_price': 0}


async def get_balance():
    """
    Get free USDT balance from exchange (always real, for trading decisions).
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
            # Return cached or 0
            pass
    return cached_bal['usdt']


def calculate_position_size(usdt_balance, price):
    """
    Calculate base amount (BTC) based on risk rules.
    """
    usd = min(usdt_balance *
              LIVE_CONFIG['position_size_pct'], LIVE_CONFIG['max_trade_usd'])
    amount = usd / price
    return round(amount, 6)


async def place_market_order(side, amount):
    """
    Place or simulate market order.
    For futures: amount in base currency (BTC); side 'buy'/'sell'.
    """
    if LIVE_CONFIG['paper_trading']:
        logger.info(
            f"[PAPER] {side.upper()} {abs(amount):.6f} {LIVE_CONFIG['base_currency']} @ market")
        return {'id': f"paper_{int(time.time())}", 'status': 'closed'}
    # Fetch fresh balance before placing order
    real_bal = await get_real_balances()
    required_usd = abs(amount) * current_price
    fee_usd = required_usd * LIVE_CONFIG['taker_fee_pct']
    if side == 'buy':
        if real_bal['usdt'] < required_usd + fee_usd:
            logger.error(
                f"Insufficient {LIVE_CONFIG['quote_currency']} for BUY {abs(amount):.6f} {LIVE_CONFIG['base_currency']} @ ${current_price:,.2f}: Need {required_usd + fee_usd:,.2f}, Have {real_bal['usdt']:,.2f}")
            return None
        logger.info(
            f"Balance check OK for BUY: USDT {real_bal['usdt']:,.2f} >= {required_usd + fee_usd:,.2f}")
    elif side == 'sell':
        if market_type == 'spot' and real_bal['btc'] < abs(amount):
            logger.error(
                f"Insufficient {LIVE_CONFIG['base_currency']} for SELL {abs(amount):.6f} {LIVE_CONFIG['base_currency']} @ ${current_price:,.2f}: Need {abs(amount):.6f}, Have {real_bal['btc']:.6f}")
            return None
        logger.info(
            f"Balance check OK for SELL: BTC {real_bal['btc']:.6f} >= {abs(amount):.6f}")
    try:
        order = await exchange.create_order(
            LIVE_CONFIG['symbol'], 'market', side, abs(amount))
        logger.info(
            f"REAL {side.upper()} {abs(amount):.6f} {LIVE_CONFIG['base_currency']} | ID: {order['id']}")
        return order
    except Exception as e:
        logger.error(f"Order failed: {e}")
        return None


def calculate_fees(side, amount, price):
    """
    Calculate approximate taker fees in USD (notional-based).
    """
    fee_pct = LIVE_CONFIG['taker_fee_pct']
    notional = abs(amount) * price
    return notional * fee_pct


async def check_trailing_stop(price):
    """
    Check and enforce trailing stop-loss for long positions.
    """
    global position
    if position['type'] != 'long' or position['size'] <= 0:
        return False
    if price > position['highest_price']:
        position['highest_price'] = price
    stop = position['highest_price'] * (1 - LIVE_CONFIG['stop_loss_pct'])
    if price <= stop:
        logger.warning(f"TRAIL STOP HIT (LONG) @ ${price:,.2f}")
        order = await place_market_order('sell', position['size'])
        if order:
            entry_fee = position['entry_fee_usd']
            exit_fee = calculate_fees('sell', position['size'], price)
            gross_pnl = (price - position['entry_price']) * position['size']
            pnl = gross_pnl - entry_fee - exit_fee
            logger.info(
                f"Long closed on stop | Gross P/L: {gross_pnl:+.2f} | Fees: {entry_fee + exit_fee:.2f} | Net P/L: {pnl:+.2f} {LIVE_CONFIG['quote_currency']}")
            position = {
                'type': 'none',
                'size': 0.0,
                'entry_price': 0.0,
                'highest_price': 0.0,
                'lowest_price': float('inf'),
                'entry_fee_usd': 0.0
            }
            await log_balance_after_trade()
        return True
    return False


async def check_trailing_short(price):
    """
    NEW: Check and enforce trailing target for short positions (trail up from low).
    """
    global position
    if position['type'] != 'short' or abs(position['size']) <= 0:
        return False
    if price < position['lowest_price']:
        position['lowest_price'] = price
    target = position['lowest_price'] * (1 + LIVE_CONFIG['stop_loss_pct'])
    if price >= target:
        logger.warning(f"TRAIL TARGET HIT (SHORT) @ ${price:,.2f}")
        # Buy to cover
        order = await place_market_order('buy', abs(position['size']))
        if order:
            entry_fee = position['entry_fee_usd']
            exit_fee = calculate_fees('buy', abs(position['size']), price)
            gross_pnl = (position['entry_price'] -
                         price) * abs(position['size'])
            pnl = gross_pnl - entry_fee - exit_fee
            logger.info(
                f"Short closed on target | Gross P/L: {gross_pnl:+.2f} | Fees: {entry_fee + exit_fee:.2f} | Net P/L: {pnl:+.2f} {LIVE_CONFIG['quote_currency']}")
            position = {
                'type': 'none',
                'size': 0.0,
                'entry_price': 0.0,
                'highest_price': 0.0,
                'lowest_price': float('inf'),
                'entry_fee_usd': 0.0
            }
            await log_balance_after_trade()
        return True
    return False


async def run_signal_strategy():
    """
    UPDATED: Execute MA crossover strategy for long/short.
    Golden Cross (bull): Close short → Open long.
    Death Cross (bear): Close long → Open short.
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
    bull_cross = prev_s <= prev_l and curr_s > curr_l
    bear_cross = prev_s >= prev_l and curr_s < curr_l
    # if bull_cross:
    #     # Close short if open
    #     if position['type'] == 'short':
    #         order = await place_market_order('buy', abs(position['size']))
    #         if order:
    #             entry_fee = position['entry_fee_usd']
    #             exit_fee = calculate_fees('buy', abs(position['size']), price)
    #             gross_pnl = (position['entry_price'] -
    #                          price) * abs(position['size'])
    #             pnl = gross_pnl - entry_fee - exit_fee
    #             logger.info(
    #                 f"Short exit on bull cross | Gross P/L: {gross_pnl:+.2f} | Fees: {entry_fee + exit_fee:.2f} | Net P/L: {pnl:+.2f} {LIVE_CONFIG['quote_currency']}")
    #             position = {
    #                 'type': 'none',
    #                 'size': 0.0,
    #                 'entry_price': 0.0,
    #                 'highest_price': 0.0,
    #                 'lowest_price': float('inf'),
    #                 'entry_fee_usd': 0.0
    #             }
    #     # Open long if none
    #     # Spot can't force open if no BTC, but futures can
    #     if position['type'] == 'none' and market_type != 'spot':
    #         usdt = await get_balance()
    #         if usdt < 10:
    #             logger.warning("Low balance")
    #             return
    #         amount = calculate_position_size(usdt, price)
    #         if amount < 0.0001:
    #             return
    #         order = await place_market_order('buy', amount)
    #         if order:
    #             entry_fee = calculate_fees('buy', amount, price)
    #             position.update({
    #                 'type': 'long',
    #                 'size': amount,
    #                 'entry_price': price,
    #                 'highest_price': price,
    #                 'lowest_price': float('inf'),
    #                 'entry_fee_usd': entry_fee
    #             })
    #             logger.info(
    #                 f"LONG {amount:.6f} {LIVE_CONFIG['base_currency']} @ ${price:,.2f} | Est. Fee: {entry_fee:.2f} {LIVE_CONFIG['quote_currency']}")
    #     elif position['type'] == 'none' and market_type == 'spot':
    #         logger.info("Bull signal ignored (spot mode: no position to open)")
    if bull_cross:
        # Close short if open (spot can't have one, but safe to check)
        if position['type'] == 'short':
            # Update position size from real balance before closing
            pos_info = await get_position_info()
            if pos_info['size'] > 0:
                position['size'] = pos_info['size']
            order = await place_market_order('buy', abs(position['size']))
            if order:
                entry_fee = position['entry_fee_usd']
                exit_fee = calculate_fees('buy', abs(position['size']), price)
                gross_pnl = (position['entry_price'] -
                             price) * abs(position['size'])
                pnl = gross_pnl - entry_fee - exit_fee
                logger.info(
                    f"Short exit on bull cross | Gross P/L: {gross_pnl:+.2f} | Fees: {entry_fee + exit_fee:.2f} | Net P/L: {pnl:+.2f} {LIVE_CONFIG['quote_currency']}")
                position = {
                    'type': 'none',
                    'size': 0.0,
                    'entry_price': 0.0,
                    'highest_price': 0.0,
                    'lowest_price': float('inf'),
                    'entry_fee_usd': 0.0
                }
                await log_balance_after_trade()
        # Open long if none (works in spot OR futures)
        if position['type'] == 'none':
            usdt = await get_balance()
            if usdt < 10:
                logger.warning("Low balance")
                return
            amount = calculate_position_size(usdt, price)
            if amount < 0.0001:
                return
            order = await place_market_order('buy', amount)
            if order:
                entry_fee = calculate_fees('buy', amount, price)
                position.update({
                    'type': 'long',
                    'size': amount,
                    'entry_price': price,
                    'highest_price': price,
                    'lowest_price': float('inf'),
                    'entry_fee_usd': entry_fee
                })
                logger.info(
                    f"LONG {amount:.6f} {LIVE_CONFIG['base_currency']} @ ${price:,.2f} | Est. Fee: {entry_fee:.2f} {LIVE_CONFIG['quote_currency']}")
                await log_balance_after_trade()
        # Remove any old "elif position['type'] == 'none' and market_type == 'spot':" block here—it was the bug causing ignores.
        ###################################################################
    elif bear_cross:
        # Close long if open
        if position['type'] == 'long':
            # Update position size from real balance before closing
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
                    f"Long exit on bear cross | Gross P/L: {gross_pnl:+.2f} | Fees: {entry_fee + exit_fee:.2f} | Net P/L: {pnl:+.2f} {LIVE_CONFIG['quote_currency']}")
                position = {
                    'type': 'none',
                    'size': 0.0,
                    'entry_price': 0.0,
                    'highest_price': 0.0,
                    'lowest_price': float('inf'),
                    'entry_fee_usd': 0.0
                }
                await log_balance_after_trade()
        # Open short if none
        if position['type'] == 'none' and market_type == 'futures':  # Spot can't short
            usdt = await get_balance()
            if usdt < 10:
                logger.warning("Low balance")
                return
            amount = calculate_position_size(usdt, price)
            if amount < 0.0001:
                return
            order = await place_market_order('sell', amount)
            if order:
                entry_fee = calculate_fees('sell', amount, price)
                position.update({
                    'type': 'short',
                    'size': -amount,
                    'entry_price': price,
                    'highest_price': 0.0,
                    'lowest_price': price,
                    'entry_fee_usd': entry_fee
                })
                logger.info(
                    f"SHORT {amount:.6f} {LIVE_CONFIG['base_currency']} @ ${price:,.2f} | Est. Fee: {entry_fee:.2f} {LIVE_CONFIG['quote_currency']}")
                await log_balance_after_trade()
        elif position['type'] == 'none' and market_type == 'spot':
            logger.info("Bear signal ignored (spot mode: no shorting)")
    else:
        # No cross: Hold if in position
        if position['type'] != 'none':
            logger.debug(
                f"Holding {position['type'].upper()} - trailing active")
        else:
            logger.debug("No signal - flat")


async def candle_poller():
    """
    Poller for new closed candles.
    """
    global df_candles, last_candle_ts
    timeframe = LIVE_CONFIG['candle_timeframe']
    logger.info("Starting candle poller...")
    cycle_count = 0  # Initialize cycle counter for heartbeat
    # Initial load
    try:
        ohlcv = await exchange.fetch_ohlcv(LIVE_CONFIG['symbol'], timeframe, limit=300)
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
            new_ohlcv = await exchange.fetch_ohlcv(LIVE_CONFIG['symbol'], timeframe, limit=2)
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
    UPDATED: Poller for real-time price updates (for trailing stops/targets).
    """
    global current_price
    logger.info("Starting price poller...")
    price_cycle = 0  # Initialize cycle counter for heartbeat
    while True:
        try:
            ticker = await exchange.fetch_ticker(LIVE_CONFIG['symbol'])
            current_price = ticker['last']
            if position['type'] == 'long':
                await check_trailing_stop(current_price)
            elif position['type'] == 'short':
                await check_trailing_short(current_price)
            price_cycle += 1
            if price_cycle % 120 == 0:  # Log every ~60s
                logger.info(
                    f"Price poller alive: Current ${current_price:,.2f} | Position: {position['type']} {position['size']:.6f}")
            await asyncio.sleep(0.5)  # Poll every 0.5s
        except Exception as e:
            logger.error(f"Price poller error: {e}")
            await asyncio.sleep(1)


async def main():
    # Set leverage for futures
    if market_type == 'futures' and not LIVE_CONFIG['paper_trading']:
        try:
            await exchange.set_leverage(LIVE_CONFIG['leverage'], LIVE_CONFIG['symbol'])
            logger.info(f"Leverage set to {LIVE_CONFIG['leverage']}x")
        except Exception as e:
            logger.error(f"Failed to set leverage: {e}")
    logger.info("HF MA BOT STARTED (Long/Short)")
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
    # Initialize current price and position if existing
    try:
        ticker = await exchange.fetch_ticker(LIVE_CONFIG['symbol'])
        global current_price
        current_price = ticker['last']
        pos_info = await get_position_info()
        if pos_info['size'] > 0.0001:
            side = pos_info['side']
            size = pos_info['size'] if side == 'long' else -pos_info['size']
            entry = pos_info['entry_price']
            position.update({
                'type': side,
                'size': size,
                'entry_price': entry,
                'highest_price': current_price if side == 'long' else 0.0,
                'lowest_price': current_price if side == 'short' else float('inf'),
                'entry_fee_usd': 0.0  # No fee for existing
            })
            logger.info(
                f"Initialized existing {side.upper()} position: {abs(size):.6f} {LIVE_CONFIG['base_currency']} @ ${current_price:,.2f} (entry ~${entry:,.2f})")
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
