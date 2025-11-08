import ccxt
import pandas as pd
import time
import logging
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
import os
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
    'candle_timeframe': '5m',
    'short_window': 7,  # Will be overridden by backtest if successful
    'long_window': 30,  # Will be overridden by backtest if successful
    'position_size_pct': 0.02,
    'max_trade_usd': 1000.0,
    'stop_loss_pct': 0.05,
    'schedule_minutes': 1,
    'symbol': 'BTC/USDT',
    'base_currency': 'BTC',
    'quote_currency': 'USDT',
    'init_usdt': 10000.0,
    # FIX: Load from .env, default True
    'paper_trading': os.getenv('PAPER_TRADING', 'True').lower() == 'true',
}
# =============================================================================
# BACKTEST CONFIGURATION (Backtest Variables)
# =============================================================================
BACKTEST_CONFIG = {
    'data_limit': 1000,  # UPGRADE: ~3.5 days for more signals
    'top_combos_to_display': 10,
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
# Initialize Binance spot exchange with rate limiting
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
else:
    logger.warning("REAL TRADING MODE - ORDERS WILL EXECUTE!")
# FIX: Simulated balance for paper trading (starts at init_usdt, updated on trades)
simulated_balance = {'usdt': LIVE_CONFIG['init_usdt']}
# Global position state dictionary
position = {
    'in_position': False,
    'entry_price': 0.0,
    'amount': 0.0,
    'highest_price': 0.0
}
def fetch_ohlcv(symbol, timeframe, limit, retries=3):
    limit = int(limit)  # FIX: Ensure int for API
    for attempt in range(retries):
        try:
            raw = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(
                raw, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            logger.info(f"Fetched {len(df)} candles successfully")
            return df
        except Exception as e:
            logger.error(f"OHLCV fetch attempt {attempt+1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                return None
def get_balance():
    """
    Get USDT balance from exchange or simulation.
    """
    if LIVE_CONFIG['paper_trading']:
        return simulated_balance['usdt']
    try:
        bal = exchange.fetch_balance()
        return bal['total'].get(LIVE_CONFIG['quote_currency'], 0.0)
    except Exception as e:
        logger.error(f"Balance error: {e}")
        return 0.0
# FIX: New helper for paper trades
def update_simulated_balance(side, amount, price):
    """
    Update simulated USDT balance on paper trades.
    """
    global simulated_balance
    usd_value = amount * price
    if side == 'buy':
        simulated_balance['usdt'] -= usd_value
    elif side == 'sell':
        simulated_balance['usdt'] += usd_value
    logger.info(f"[SIM] Balance updated: USDT={simulated_balance['usdt']:.2f}")
def calculate_position_size(usdt_balance, price):
    """
    Calculate BTC amount based on risk rules.
    """
    usd = min(usdt_balance *
              LIVE_CONFIG['position_size_pct'], LIVE_CONFIG['max_trade_usd'])
    amount = usd / price
    return round(amount, 6)
def place_market_order(side, amount):
    """
    Place or simulate market order.
    """
    if LIVE_CONFIG['paper_trading']:
        logger.info(
            f"[PAPER] {side.upper()} {amount:.6f} {LIVE_CONFIG['base_currency']} @ market")
        # FIX: Update sim balance here (price fetched earlier in caller)
        return {'id': f"paper_{int(time.time())}", 'status': 'closed'}
    try:
        order = exchange.create_order(
            LIVE_CONFIG['symbol'], 'market', side, amount)
        logger.info(
            f"REAL {side.upper()} {amount:.6f} {LIVE_CONFIG['base_currency']} | ID: {order['id']}")
        return order
    except Exception as e:
        logger.error(f"Order failed: {e}")
        return None
def check_trailing_stop(price):
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
        order = place_market_order('sell', position['amount'])
        if order:
            pnl = (price - position['entry_price']) * position['amount']
            logger.info(
                f"Closed on stop | P/L: {pnl:+.2f} {LIVE_CONFIG['quote_currency']}")
            if LIVE_CONFIG['paper_trading']:
                update_simulated_balance('sell', position['amount'], price)
            position.update(
                {'in_position': False, 'entry_price': 0, 'amount': 0, 'highest_price': 0})
        return True
    return False
# Track last candle timestamp to avoid duplicates
last_candle_ts = 0
def run_strategy():
    """
    Execute MA crossover strategy logic.
    """
    global position, last_candle_ts
    limit = LIVE_CONFIG['long_window'] + 10
    df = fetch_ohlcv(LIVE_CONFIG['symbol'],
                     LIVE_CONFIG['candle_timeframe'], limit)
    if df is None or len(df) < LIVE_CONFIG['long_window']:
        logger.warning(
            f"Insufficient data: {len(df) if df is not None else 0} candles")
        return
    current_ts = df.index[-1].timestamp()
    if current_ts == last_candle_ts:
        return
    last_candle_ts = current_ts
    close = df['close']
    price = close.iloc[-1]
    short_ma = close.rolling(LIVE_CONFIG['short_window']).mean()
    long_ma = close.rolling(LIVE_CONFIG['long_window']).mean()
    curr_s, prev_s = short_ma.iloc[-1], short_ma.iloc[-2]
    curr_l, prev_l = long_ma.iloc[-1], long_ma.iloc[-2]
    logger.info(
        f"Price: ${price:,.2f} | MA{LIVE_CONFIG['short_window']}: {curr_s:,.2f} | MA{LIVE_CONFIG['long_window']}: {curr_l:,.2f}")
    if position['in_position']:
        if check_trailing_stop(price):
            return
        logger.info("Holding - trailing stop active")
        return
    if prev_s <= prev_l and curr_s > curr_l:
        usdt = get_balance()
        if usdt < 10:
            logger.warning("Low balance")
            return
        amount = calculate_position_size(usdt, price)
        if amount < 0.0001:
            return
        order = place_market_order('buy', amount)
        if order:
            if LIVE_CONFIG['paper_trading']:
                update_simulated_balance('buy', amount, price)
            position.update({
                'in_position': True,
                'entry_price': price,
                'amount': amount,
                'highest_price': price
            })
            logger.info(
                f"BUY {amount:.6f} {LIVE_CONFIG['base_currency']} @ ${price:,.2f}")
    elif position['in_position'] and prev_s >= prev_l and curr_s < curr_l:
        order = place_market_order('sell', position['amount'])
        if order:
            pnl = (price - position['entry_price']) * position['amount']
            logger.info(
                f"Death Cross exit | P/L: {pnl:+.2f} {LIVE_CONFIG['quote_currency']}")
            if LIVE_CONFIG['paper_trading']:
                update_simulated_balance('sell', position['amount'], price)
            position.update(
                {'in_position': False, 'entry_price': 0, 'amount': 0, 'highest_price': 0})
def main():
    logger.info("HF MA BOT STARTED")
    logger.info(f"{LIVE_CONFIG['candle_timeframe']} | {LIVE_CONFIG['short_window']}/{LIVE_CONFIG['long_window']} | {LIVE_CONFIG['position_size_pct']*100}% risk | ${LIVE_CONFIG['max_trade_usd']} cap | {LIVE_CONFIG['stop_loss_pct']*100}% trail")
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
    logger.info("Bot LIVE. Ctrl+C to stop.\n")
    cycle = 0
    while True:
        try:
            cycle += 1
            logger.info(
                f"--- CYCLE {cycle} @ {datetime.now().strftime('%H:%M:%S')} ---")
            start = time.time()
            run_strategy()
            usdt_balance = get_balance()
            logger.info(
                f"{LIVE_CONFIG['quote_currency']} Balance: {usdt_balance:,.2f}")
            elapsed = time.time() - start
            sleep = max(0, LIVE_CONFIG['schedule_minutes'] * 60 - elapsed)
            logger.info(f"Cycle done in {elapsed:.1f}s | Sleep {sleep:.0f}s")
            time.sleep(sleep)
        except KeyboardInterrupt:
            logger.info("Stopped by user.")
            break
        except Exception as e:
            logger.critical(f"Error: {e}", exc_info=True)
            time.sleep(30)
if __name__ == "__main__":
    main()
