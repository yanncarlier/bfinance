import ccxt.async_support as ccxt
import pandas as pd
import time
import logging
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
import os
import asyncio
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
    # 'candle_timeframe': '1h',
    # 'short_window': 12,             # Start with EMA 12 (common for 1h)
    # 'long_window': 50,              # EMA 50
    'candle_timeframe': '2h',
    'short_window': 20,
    'long_window': 95,
    'adx_threshold': 25,  # Higher for longer TF
    'position_size_pct': 0.15,      # % of available USDT balance per trade
    'max_trade_usd': 500.0,         # Maximum USD notional per trade
    'take_profit_pct': 0.05,        # Increased to 5% for longer timeframe
    'trailing_stop_pct': 0.03,      # Increased to 3% trail for 1h
    'symbol': 'BTC/USDT',           # Trading pair for spot market
    'base_currency': 'BTC',
    'quote_currency': 'USDT',
    'taker_fee_pct': 0.001,         # 0.1% taker fee (adjust for your tier)
    # Simulated slippage (0.05% for market orders)
    'slippage_pct': 0.0005,
    'market_type': 'spot',          # Fixed to spot for long-only trading
    'paper_trading': os.getenv('PAPER_TRADING', 'True').lower() == 'true',
    'adx_period': 14,               # ADX period for trend filter
    # 'adx_threshold': 20,            # ADX > 20 for strong trend
    'rsi_period': 14,               # RSI period
    'rsi_overbought': 70,           # Avoid buy if RSI > 70
    'rsi_oversold': 30,             # Additional close if RSI < 30 (optional)
    'candle_poller': 120,           # Poll candles every 120 seconds
    'volume_ma_period': 20,         # Period for volume moving average
    'volume_multiplier': 1.2,       # Volume must be > this x MA for confirmation
    # Start date for historical data load (YYYY-MM-DD)
    'start_date': '2024-01-01',
    # End date dynamically set to today (YYYY-MM-DD)
    'end_date': pd.Timestamp.today().strftime('%Y-%m-%d'),
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
    logger.info("PAPER MODE: No real orders")
else:
    logger.warning("LIVE MODE: Real orders on Binance spot")
logger.info(f"SPOT LONG-ONLY: {symbol} (EMA x-over + ADX/RSI filters)")
# Global position state (long-only)
position = {
    'type': 'none',                 # 'long' or 'none'
    'size': 0.0,                    # Positive BTC amount for long
    'entry_price': 0.0,
    'highest_price': 0.0,           # Tracks high for trailing stop
    'entry_fee_usd': 0.0,
    'entry_slip_usd': 0.0
}
# Global data structures
df_candles = pd.DataFrame()
current_price = 0.0
last_candle_ts = 0
# Cache for balance fetches to reduce API calls
last_balance_fetch = 0
cached_bal = {'usdt': 0.0, 'btc': 0.0}
BALANCE_CACHE_REFRESH_SEC = 30  # Refresh every 30 seconds
# =============================================================================
# COLOR HELPER FUNCTIONS
# =============================================================================


def color_pct(pct):
    """Green for positive, red for negative percentage."""
    if pct > 0:
        return f"\033[32m{pct:.3f}\033[0m"
    elif pct < 0:
        return f"\033[31m{pct:.3f}\033[0m"
    else:
        return f"{pct:.3f}"


def color_value(val, fmt=":.2f"):
    """Green for positive, red for negative value."""
    if val > 0:
        return f"\033[32m{val:{fmt}}\033[0m"
    elif val < 0:
        return f"\033[31m{val:{fmt}}\033[0m"
    else:
        return f"{val:{fmt}}"
# =============================================================================
# CORE FUNCTIONS
# =============================================================================


async def fetch_ohlcv_period(symbol, timeframe, start_date, end_date, limit_per_call=1000):
    """
    Fetch OHLCV data from Binance for a specific time period (async).
    Args:
        symbol (str): Trading pair (e.g., 'BTC/USDT')
        timeframe (str): Candle interval (e.g., '5m')
        start_date (str): Start date (YYYY-MM-DD)
        end_date (str): End date (YYYY-MM-DD)
        limit_per_call (int): Candles per API call (default 1000)
    Returns:
        pd.DataFrame: OHLCV data indexed by timestamp within the period, or empty on failure
    """
    try:
        since_ts = int(pd.to_datetime(start_date).timestamp() * 1000)
        until_ts = int(pd.to_datetime(end_date).timestamp() * 1000)
        all_candles = []
        current_since = since_ts
        while current_since < until_ts:
            raw = await exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=limit_per_call)
            if not raw:
                break
            all_candles.extend(raw)
            # Start next fetch after last candle
            current_since = raw[-1][0] + 1
            if current_since >= until_ts:
                break
        if not all_candles:
            return pd.DataFrame()
        df = pd.DataFrame(
            all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        # Filter to the exact period (inclusive)
        df = df[(df.index >= pd.to_datetime(start_date))
                & (df.index <= pd.to_datetime(end_date))]
        # Remove duplicates if any (unlikely but safe)
        df = df[~df.index.duplicated(keep='first')]
        return df.sort_index()
    except Exception as e:
        logger.error(f"Fetch period failed: {e}")
        return pd.DataFrame()


async def get_real_balances():
    """Fetch real balances for spot: free USDT and total BTC holdings."""
    try:
        bal = await exchange.fetch_balance()
        quote_free = bal[LIVE_CONFIG['quote_currency']]['free']
        base_total = bal[LIVE_CONFIG['base_currency']]['total']
        return {'usdt': quote_free, 'btc': base_total}
    except Exception as e:
        logger.error(f"Bal fetch err: {e}")
        return {'usdt': 0.0, 'btc': 0.0}


async def log_balance_after_trade():
    """Log estimated total balance in USDT after a trade (colored)."""
    global current_price
    try:
        real_bal = await get_real_balances()
        estimated_usdt = real_bal['usdt'] + real_bal['btc'] * current_price
        color_msg = f"\033[92mEst Bal: {estimated_usdt:,.2f} {LIVE_CONFIG['quote_currency']} (USDT: {real_bal['usdt']:,.2f}, {LIVE_CONFIG['base_currency']}: {real_bal['btc']:.6f} @ ${current_price:,.2f})\033[0m"
        logger.info(color_msg)
    except Exception as e:
        logger.error(f"Bal log err: {e}")


async def get_position_info():
    """Get current spot position: treat BTC holdings as long position."""
    try:
        bal = await exchange.fetch_balance()
        size = bal[LIVE_CONFIG['base_currency']]['total']
        if size > 0.0001:
            return {'size': size, 'side': 'long', 'entry_price': current_price}
        return {'size': 0, 'side': None, 'entry_price': 0}
    except Exception as e:
        logger.error(f"Pos fetch err: {e}")
        return {'size': 0, 'side': None, 'entry_price': 0}


async def get_balance():
    """Get cached free USDT balance (for position sizing)."""
    global last_balance_fetch, cached_bal
    now = time.time()
    if now - last_balance_fetch > BALANCE_CACHE_REFRESH_SEC:
        try:
            real_bal = await get_real_balances()
            cached_bal['usdt'] = real_bal['usdt']
            cached_bal['btc'] = real_bal['btc']
            last_balance_fetch = now
        except Exception as e:
            logger.error(f"Bal cache err: {e}")
    return cached_bal['usdt']


def calculate_position_size(usdt_balance, price):
    """Calculate BTC amount based on risk rules."""
    usd = min(usdt_balance *
              LIVE_CONFIG['position_size_pct'], LIVE_CONFIG['max_trade_usd'])
    amount = usd / price
    return round(amount, 6)


async def place_market_order(side, amount):
    """Place or simulate market order on spot."""
    if LIVE_CONFIG['paper_trading']:
        logger.info(
            f"[PAPER] {side.upper()} {amount:.6f} {LIVE_CONFIG['base_currency']} @ mkt")
        return {'id': f"paper_{int(time.time())}", 'status': 'closed'}
    real_bal = await get_real_balances()
    required_usd = amount * current_price
    fee_usd = required_usd * LIVE_CONFIG['taker_fee_pct']
    if side == 'buy':
        if real_bal['usdt'] < required_usd + fee_usd:
            logger.error(
                f"Insuf USDT: Need {required_usd + fee_usd:,.2f} for {amount:.6f} BTC")
            return None
        logger.info(
            f"BUY OK: USDT {real_bal['usdt']:,.2f} >= {required_usd + fee_usd:,.2f}")
    elif side == 'sell':
        if real_bal['btc'] < amount:
            logger.error(
                f"Insuf BTC: Need {amount:.6f}, have {real_bal['btc']:.6f}")
            return None
        logger.info(f"SELL OK: BTC {real_bal['btc']:.6f} >= {amount:.6f}")
    try:
        order = await exchange.create_order(LIVE_CONFIG['symbol'], 'market', side, amount)
        logger.info(
            f"REAL {side.upper()} {amount:.6f} {LIVE_CONFIG['base_currency']} | ID: {order['id']}")
        return order
    except Exception as e:
        logger.error(f"Order err: {e}")
        return None


def calculate_fees(side, amount, price):
    """Calculate taker fees in USD."""
    notional = amount * price
    return notional * LIVE_CONFIG['taker_fee_pct']


def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def rsi(series, period):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_adx(high, low, close, period):
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
    """Check trailing stop-loss and take-profit."""
    global position
    if position['type'] != 'long' or position['size'] <= 0:
        return False
    if price > position['highest_price']:
        position['highest_price'] = price
    stop_price = position['highest_price'] * \
        (1 - LIVE_CONFIG['trailing_stop_pct'])
    tp_price = position['entry_price'] * (1 + LIVE_CONFIG['take_profit_pct'])
    if price <= stop_price:
        logger.warning(f"TRAIL STOP HIT @ ${price:,.2f}")
        order = await place_market_order('sell', position['size'])
        if order:
            entry_fee = position['entry_fee_usd']
            entry_slip = position['entry_slip_usd']
            exit_fee = calculate_fees('sell', position['size'], price)
            exit_slip = position['size'] * price * LIVE_CONFIG['slippage_pct']
            gross_pnl = position['size'] * (price - position['entry_price'])
            net_pnl = gross_pnl - entry_fee - entry_slip - exit_fee - exit_slip
            gross_col = color_value(gross_pnl, "+:.2f")
            pnl_col = color_value(net_pnl, "+:.2f")
            total_cost = entry_fee + entry_slip + exit_fee + exit_slip
            logger.info(
                f"LONG CLOSE (TRAIL) | Gross: {gross_col} | Costs: {total_cost:.2f} (Fees:{entry_fee + exit_fee:.2f}, Slip:{entry_slip + exit_slip:.2f}) | Net: {pnl_col} {LIVE_CONFIG['quote_currency']}")
            position = {'type': 'none', 'size': 0.0, 'entry_price': 0.0,
                        'highest_price': 0.0, 'entry_fee_usd': 0.0, 'entry_slip_usd': 0.0}
            await log_balance_after_trade()
        return True
    elif price >= tp_price:
        logger.warning(f"TAKE PROFIT HIT @ ${price:,.2f}")
        order = await place_market_order('sell', position['size'])
        if order:
            entry_fee = position['entry_fee_usd']
            entry_slip = position['entry_slip_usd']
            exit_fee = calculate_fees('sell', position['size'], price)
            exit_slip = position['size'] * price * LIVE_CONFIG['slippage_pct']
            gross_pnl = position['size'] * (price - position['entry_price'])
            net_pnl = gross_pnl - entry_fee - entry_slip - exit_fee - exit_slip
            gross_col = color_value(gross_pnl, "+:.2f")
            pnl_col = color_value(net_pnl, "+:.2f")
            total_cost = entry_fee + entry_slip + exit_fee + exit_slip
            logger.info(
                f"LONG CLOSE (TP) | Gross: {gross_col} | Costs: {total_cost:.2f} (Fees:{entry_fee + exit_fee:.2f}, Slip:{entry_slip + exit_slip:.2f}) | Net: {pnl_col} {LIVE_CONFIG['quote_currency']}")
            position = {'type': 'none', 'size': 0.0, 'entry_price': 0.0,
                        'highest_price': 0.0, 'entry_fee_usd': 0.0, 'entry_slip_usd': 0.0}
            await log_balance_after_trade()
        return True
    return False


async def run_signal_strategy():
    """Execute EMA crossover with ADX/RSI filters."""
    global position, df_candles
    if len(df_candles) < max(LIVE_CONFIG['long_window'], LIVE_CONFIG['rsi_period'], LIVE_CONFIG['adx_period'], LIVE_CONFIG['volume_ma_period']):
        logger.warning(f"Insuf data: {len(df_candles)} candles")
        return
    close = df_candles['close']
    high = df_candles['high']
    low = df_candles['low']
    volume = df_candles['volume']
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
    rsi_sell_ok = rsi_val > LIVE_CONFIG['rsi_overbought']
    # Volume filter: Current volume > multiplier x MA(period)
    vol_ma = volume.rolling(LIVE_CONFIG['volume_ma_period']).mean().iloc[-1]
    vol_ok = volume.iloc[-1] > vol_ma * LIVE_CONFIG['volume_multiplier']
    logger.info(
        f"SIG CHK | BullX: {'Y' if bull_cross else 'N'} | ADX:{adx:.1f}({'>25' if adx_ok else '<25'}) | RSI:{rsi_val:.1f}(<70:{'Y' if rsi_buy_ok else 'N'}) | VolOK:{'Y' if vol_ok else 'N'} | Pos:{position['type']} | Prc:${price:,.0f}"
    )
    if bull_cross and adx_ok and rsi_buy_ok and vol_ok and position['type'] == 'none':
        usdt = await get_balance()
        if usdt < 10:
            logger.warning("Low USDT for entry")
            return
        amount = calculate_position_size(usdt, price)
        if amount < 0.0001:
            return
        order = await place_market_order('buy', amount)
        if order:
            entry_fee = calculate_fees('buy', amount, price)
            entry_slip = amount * price * LIVE_CONFIG['slippage_pct']
            position.update({
                'type': 'long', 'size': amount, 'entry_price': price,
                'highest_price': price, 'entry_fee_usd': entry_fee, 'entry_slip_usd': entry_slip
            })
            logger.info(
                f"OPEN LONG {amount:.6f} {LIVE_CONFIG['base_currency']} @ ${price:,.2f} | Fee:{entry_fee:.2f} | Slip:{entry_slip:.2f} | ADX:{adx:.2f} | RSI:{rsi_val:.2f} | Vol:{volume.iloc[-1]:,.0f}(>{vol_ma * LIVE_CONFIG['volume_multiplier']:.0f})")
            await log_balance_after_trade()
    elif (bear_cross or rsi_sell_ok) and position['type'] == 'long':
        pos_info = await get_position_info()
        if pos_info['size'] > 0:
            position['size'] = pos_info['size']
        order = await place_market_order('sell', position['size'])
        if order:
            entry_fee = position['entry_fee_usd']
            entry_slip = position['entry_slip_usd']
            exit_fee = calculate_fees('sell', position['size'], price)
            exit_slip = position['size'] * price * LIVE_CONFIG['slippage_pct']
            gross_pnl = position['size'] * (price - position['entry_price'])
            net_pnl = gross_pnl - entry_fee - entry_slip - exit_fee - exit_slip
            gross_col = color_value(gross_pnl, "+:.2f")
            pnl_col = color_value(net_pnl, "+:.2f")
            total_cost = entry_fee + entry_slip + exit_fee + exit_slip
            logger.info(
                f"CLOSE LONG @ ${price:,.2f} | Gross:{gross_col} | Costs:{total_cost:.2f} (Fees:{entry_fee + exit_fee:.2f}, Slip:{entry_slip + exit_slip:.2f}) | Net:{pnl_col} {LIVE_CONFIG['quote_currency']} | ADX:{adx:.2f} | RSI:{rsi_val:.2f}")
            position = {'type': 'none', 'size': 0.0, 'entry_price': 0.0,
                        'highest_price': 0.0, 'entry_fee_usd': 0.0, 'entry_slip_usd': 0.0}
            await log_balance_after_trade()


async def candle_poller():
    global df_candles, last_candle_ts
    timeframe = LIVE_CONFIG['candle_timeframe']
    logger.info(
        f"Loading historical candles from {LIVE_CONFIG['start_date']} to {LIVE_CONFIG['end_date']}")
    logger.info("Candle poller started")
    try:
        df_candles = await fetch_ohlcv_period(LIVE_CONFIG['symbol'], timeframe, LIVE_CONFIG['start_date'], LIVE_CONFIG['end_date'])
        if df_candles.empty:
            logger.error("Failed to load historical candles")
            return
        last_candle_ts = int(df_candles.index[-1].timestamp())
        logger.info(
            f"Loaded {len(df_candles)} candles from {df_candles.index[0]} to {df_candles.index[-1]}")
    except Exception as e:
        logger.error(f"Init candles err: {e}")
        return
    while True:
        try:
            await asyncio.sleep(LIVE_CONFIG['candle_poller'])
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
                new_row = new_df.iloc[-1]
                df_candles = pd.concat(
                    [df_candles, new_row.to_frame().T]).tail(100)
                last_candle_ts = new_ts
                await run_signal_strategy()
        except Exception as e:
            logger.error(f"Candle poller err: {e}")
            await asyncio.sleep(LIVE_CONFIG['candle_poller'])


async def price_poller():
    global current_price
    logger.info("Price poller started")
    while True:
        try:
            ticker = await exchange.fetch_ticker(LIVE_CONFIG['symbol'])
            current_price = ticker['last']
            if position['type'] == 'long':
                await check_trailing_stop(current_price)
            await asyncio.sleep(30)
        except Exception as e:
            logger.error(f"Price poller err: {e}")
            await asyncio.sleep(30)


async def main():
    logger.info("SPOT EMA BOT STARTED (long-only + ADX/RSI)")
    logger.info(f"TF:{LIVE_CONFIG['candle_timeframe']} | EMA:{LIVE_CONFIG['short_window']}/{LIVE_CONFIG['long_window']} | Risk:{LIVE_CONFIG['position_size_pct']*100}% | Max:${LIVE_CONFIG['max_trade_usd']} | TP:{LIVE_CONFIG['take_profit_pct']*100}% | Trail:{LIVE_CONFIG['trailing_stop_pct']*100}% | ADX>{LIVE_CONFIG['adx_threshold']} | RSI<{LIVE_CONFIG['rsi_overbought']} | Vol>{LIVE_CONFIG['volume_multiplier']}xMA({LIVE_CONFIG['volume_ma_period']})")
    if not LIVE_CONFIG['paper_trading']:
        confirm = input("\nLIVE MODE! Type 'YES' to continue: ")
        if confirm != "YES":
            logger.info("Confirm failed - exit")
            return
    real_bal = await get_real_balances()
    logger.info(
        f"Init Bal: {LIVE_CONFIG['quote_currency']}:{real_bal['usdt']:,.2f} | {LIVE_CONFIG['base_currency']}:{real_bal['btc']:.6f}")
    try:
        ticker = await exchange.fetch_ticker(LIVE_CONFIG['symbol'])
        global current_price
        current_price = ticker['last']
        pos_info = await get_position_info()
        if pos_info['size'] > 0.0001:
            position.update({
                'type': 'long', 'size': pos_info['size'], 'entry_price': pos_info['entry_price'],
                'highest_price': current_price, 'entry_fee_usd': 0.0, 'entry_slip_usd': 0.0
            })
            logger.info(
                f"Resume LONG: {position['size']:.6f} {LIVE_CONFIG['base_currency']} @ ${current_price:,.2f}")
    except Exception as e:
        logger.error(f"Init err: {e}")
    logger.info("Bot live. Ctrl+C to stop.\n")
    await asyncio.gather(candle_poller(), price_poller())
if __name__ == "__main__":
    asyncio.run(main())
