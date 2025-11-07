"""
================================================================================
HIGH-FREQUENCY MOVING AVERAGE CROSSOVER BOT (BTC/USDT) - FULL STANDALONE
================================================================================
- 5-minute candles (true high-frequency)
- Ultra-responsive MAs: 7-period fast, 30-period slow
- 2% risk per trade, $1000 max, 5% trailing stop
- Paper trading by default (set PAPER_TRADING = False for real)
- 1-minute schedule (catches every new candle)
- Built-in backtester (run once to validate)
- Full logging, error recovery, Binance spot via CCXT
"""
# =============================================================================
# IMPORTS
# =============================================================================
# CCXT library for interacting with cryptocurrency exchanges like Binance
import ccxt
# Pandas for data manipulation, especially for handling OHLCV data as DataFrames
import pandas as pd
# Time module for handling delays, timestamps, and scheduling
import time
# Logging module for structured output to console and files for debugging and auditing
import logging
# NumPy for numerical computations, used in backtesting for arrays and cumulative sums
import numpy as np
# Datetime for formatting timestamps in logs
from datetime import datetime
# Dotenv for loading environment variables from .env file securely
from dotenv import load_dotenv
# OS module for accessing environment variables
import os
# Itertools.product for generating combinations of parameters in backtesting grid search
from itertools import product
# Concurrent.futures for parallel execution of backtest combinations to speed up processing
from concurrent.futures import ThreadPoolExecutor
# =============================================================================
# LOAD CREDENTIALS
# =============================================================================
# Load environment variables from .env file in the current directory
load_dotenv()
# Retrieve Binance API key from environment variables
API_KEY = os.getenv('API_KEY')
# Retrieve Binance API secret from environment variables
API_SECRET = os.getenv('API_SECRET')
# Check if credentials are present; raise an error if missing to prevent runtime failures
if not API_KEY or not API_SECRET:
    raise EnvironmentError("API_KEY and API_SECRET must be in .env file")
# =============================================================================
# CONFIGURATION - HIGH FREQUENCY OPTIMIZED
# =============================================================================
# Timeframe for candlestick data; '5m' means 5-minute intervals for high-frequency trading
TIMEFRAME = '5m'              # 5-minute candles
# Period for the short (fast) moving average; 7 periods on 5m timeframe ≈ 35 minutes
SHORT_WINDOW = 7              # ~35 min fast MA
# Period for the long (slow) moving average; 30 periods on 5m timeframe ≈ 2.5 hours
LONG_WINDOW = 30              # ~2.5 hour trend filter
# Percentage of available USDT balance to risk per trade (e.g., 0.02 = 2%)
POSITION_SIZE_PCT = 0.02      # 2% of USDT balance per trade
# Maximum USD amount to risk per trade, acting as a hard cap for position sizing
MAX_TRADE_USD = 1000.0        # Never risk more than $1,000
# Trailing stop-loss percentage; trails 5% below the highest price since entry
STOP_LOSS_PCT = 0.05          # 5% trailing stop
# Interval in minutes to check for signals; 1 minute ensures catching every new 5m candle
SCHEDULE_MINUTES = 1          # Check every 60 seconds
# Trading pair symbol on Binance (BTC quoted in USDT)
SYMBOL = 'BTC/USDT'
# Base currency (the asset being traded, BTC)
BASE_CURRENCY = 'BTC'
# Quote currency (the stablecoin used for balance and P/L calculations, USDT)
QUOTE_CURRENCY = 'USDT'
# Flag for paper trading mode; True simulates trades without real orders; set to False for live trading
PAPER_TRADING = True          # SET TO False FOR REAL TRADES
# =============================================================================
# LOGGING
# =============================================================================
# Configure basic logging with INFO level; logs to both file and console
logging.basicConfig(
    level=logging.INFO,  # INFO level shows key events; change to DEBUG for more details
    # Timestamp | Level | Message format
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[
        # File handler for persistent logs in 'hf_bot.log' with UTF-8 encoding
        logging.FileHandler("hf_bot.log", encoding='utf-8'),
        # Stream handler for real-time console output
        logging.StreamHandler()
    ]
)
# Get a logger instance named after the current module for organized logging
logger = logging.getLogger(__name__)
# =============================================================================
# EXCHANGE
# =============================================================================
# Initialize CCXT Binance exchange object with credentials and spot trading options
exchange = ccxt.binance({
    'apiKey': API_KEY,  # API key for authentication
    'secret': API_SECRET,  # API secret for authentication
    'enableRateLimit': True,  # Respect rate limits to avoid bans from Binance API
    # Use spot market (not futures or margin)
    'options': {'defaultType': 'spot'}
})
# Log the trading mode at startup for user awareness
if PAPER_TRADING:
    logger.info("PAPER TRADING MODE - NO REAL ORDERS")
else:
    logger.warning("REAL TRADING MODE - ORDERS WILL EXECUTE!")
# =============================================================================
# POSITION STATE
# =============================================================================
# Dictionary to track the current position state; acts as global memory for the bot
position = {
    'in_position': False,  # Boolean flag indicating if a position is currently open
    'entry_price': 0.0,  # Price at which the position was entered
    'amount': 0.0,  # Quantity of BTC held in the position
    'highest_price': 0.0  # Highest price reached since entry, used for trailing stop
}
# =============================================================================
# HELPERS
# =============================================================================
# Function to fetch Open-High-Low-Close-Volume (OHLCV) data from Binance


def fetch_ohlcv(symbol, timeframe, limit=100):
    try:
        # Fetch raw OHLCV data as list of lists
        raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(
            raw, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        # Convert millisecond timestamps to datetime objects
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        # Set timestamp as index for time-series operations
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        # Log error and return None to handle gracefully in calling functions
        logger.error(f"OHLCV fetch failed: {e}")
        return None
# Function to retrieve the available balance in the quote currency (USDT)


def get_balance():
    try:
        # Fetch full balance info from exchange
        bal = exchange.fetch_balance()
        # Return total balance for USDT (includes free + locked)
        return bal['total'].get(QUOTE_CURRENCY, 0.0)
        # Return total balance for BTC
    except Exception as e:
        # Log error and return 0 to avoid trading with invalid balance
        logger.error(f"Balance error: {e}")
        return 0.0
# Function to calculate the position size based on risk parameters


def calculate_position_size(usdt_balance, price):
    # Calculate USD to risk: min of % of balance or max cap
    usd = min(usdt_balance * POSITION_SIZE_PCT, MAX_TRADE_USD)
    # Convert USD to BTC amount
    amount = usd / price
    # Round to 6 decimals to match exchange precision requirements
    return round(amount, 6)
# Function to place a market order (buy or sell)


def place_market_order(side, amount):
    if PAPER_TRADING:
        # Simulate order in paper mode: log and return fake order dict
        logger.info(f"[PAPER] {side.upper()} {amount:.6f} BTC @ market")
        return {'id': f"paper_{int(time.time())}", 'status': 'closed'}
    try:
        # Create real market order via CCXT
        order = exchange.create_order(SYMBOL, 'market', side, amount)
        # Log successful real order execution
        logger.info(
            f"REAL {side.upper()} {amount:.6f} BTC | ID: {order['id']}")
        return order
    except Exception as e:
        # Log failure and return None
        logger.error(f"Order failed: {e}")
        return None
# Function to check and enforce the trailing stop-loss


def check_trailing_stop(price):
    global position  # Access global position state
    if not position['in_position']:
        return False  # No position to check
    if price > position['highest_price']:
        # Update peak price if current is higher
        position['highest_price'] = price
        logger.info(f"New high: ${price:,.2f}")
    # Calculate dynamic stop price: peak minus trailing percentage
    stop = position['highest_price'] * (1 - STOP_LOSS_PCT)
    if price <= stop:
        # Log stop hit and attempt to sell
        logger.warning(f"TRAIL STOP HIT @ ${price:,.2f}")
        order = place_market_order('sell', position['amount'])
        if order:
            # Calculate P/L for logging
            pnl = (price - position['entry_price']) * position['amount']
            logger.info(f"Closed on stop | P/L: {pnl:+.2f} USDT")
            # Reset position state
            position.update(
                {'in_position': False, 'entry_price': 0, 'amount': 0, 'highest_price': 0})
        return True
    return False
# =============================================================================
# BACKTEST ENGINE
# =============================================================================
# Function to backtest a single MA combination on historical data


def backtest_one(df, short_win, long_win,
                 init_usdt=10000, pos_pct=POSITION_SIZE_PCT,
                 max_usd=MAX_TRADE_USD, trail_pct=STOP_LOSS_PCT):
    # Extract close prices as NumPy array for efficiency
    close = df['close'].values
    n = len(close)
    # Compute moving averages using Pandas rolling mean
    short_ma = pd.Series(close).rolling(short_win).mean().values
    long_ma = pd.Series(close).rolling(long_win).mean().values
    # Initialize simulation variables
    cash = init_usdt
    btc = entry = peak = 0.0
    trades = []  # List to store trade P/L for metrics
    # Loop through data starting after longest MA window
    for i in range(max(short_win, long_win), n):
        price = close[i]
        if btc > 0:  # If in position
            if price > peak:
                peak = price  # Update peak
            stop = peak * (1 - trail_pct)
            if price <= stop:
                # Simulate stop-loss exit
                cash += btc * price
                trades.append({'pnl': btc * (price - entry)})
                btc = entry = peak = 0.0
                continue
            # Check for death cross early exit
            if short_ma[i-1] >= long_ma[i-1] and short_ma[i] < long_ma[i]:
                cash += btc * price
                trades.append({'pnl': btc * (price - entry)})
                btc = entry = peak = 0.0
                continue
        # Check for golden cross entry
        if btc == 0 and short_ma[i-1] <= long_ma[i-1] and short_ma[i] > long_ma[i]:
            usd = min(cash * pos_pct, max_usd)
            btc = round(usd / price, 6)
            cash -= btc * price
            entry = peak = price
    # Close any open position at the end of data
    if btc > 0:
        cash += btc * close[-1]
        trades.append({'pnl': btc * (close[-1] - entry)})
    # Calculate total P/L and return percentage
    pnl = cash - init_usdt
    ret = pnl / init_usdt * 100
    # Calculate win rate
    wins = sum(1 for t in trades if t['pnl'] > 0)
    winrate = wins / len(trades) * 100 if trades else 0
    # Calculate equity curve and max drawdown
    equity = np.cumsum([init_usdt] + [t['pnl'] for t in trades])
    dd = np.maximum.accumulate(equity) - equity
    max_dd = dd.max() / init_usdt * 100 if equity.size else 0
    return {
        'short': short_win, 'long': long_win,
        'return_%': round(ret, 3), 'trades': len(trades),
        'winrate_%': round(winrate, 1), 'max_dd_%': round(max_dd, 2),
        'final_usdt': round(cash, 2)
    }
# Function to run a grid search backtest over multiple MA combinations


def run_hf_backtest(limit=3000):
    # Fetch historical data for backtesting
    df = fetch_ohlcv(SYMBOL, '5m', limit)
    if df is None or len(df) < 90:
        # Return empty DataFrame if data fetch fails or insufficient
        return pd.DataFrame()
    # Define ranges for grid search: short 3-15 (step 2), long 20-80 (step 5)
    short_range = range(3, 16, 2)
    long_range = range(20, 81, 5)
    # Generate all combinations
    combos = list(product(short_range, long_range))
    results = []  # List to collect results
    # Worker function for parallel execution

    def worker(s, l):
        return backtest_one(df, s, l)
    # Use thread pool for parallel backtests (up to 12 workers)
    with ThreadPoolExecutor(max_workers=12) as ex:
        for s, l in combos:
            res = ex.submit(worker, s, l).result()
            if res:
                results.append(res)
    # Convert to DataFrame and sort by return descending
    out = pd.DataFrame(results).sort_values(
        'return_%', ascending=False).reset_index(drop=True)
    return out


# =============================================================================
# STRATEGY CORE
# =============================================================================
# Global variable to track the timestamp of the last processed candle to avoid duplicates
last_candle_ts = 0
# Core function that runs the strategy logic on each cycle


def run_strategy():
    global position, last_candle_ts  # Access global state
    # Fetch recent OHLCV data (enough for MAs + buffer)
    df = fetch_ohlcv(SYMBOL, TIMEFRAME, LONG_WINDOW + 10)
    if df is None or len(df) < LONG_WINDOW:
        return  # Skip if data insufficient
    # Get timestamp of latest candle
    current_ts = df.index[-1].timestamp()
    if current_ts == last_candle_ts:
        return  # Skip if same candle as last cycle (no new data)
    last_candle_ts = current_ts  # Update last timestamp
    # Extract close prices
    close = df['close']
    price = close.iloc[-1]  # Current price
    # Compute moving averages
    short_ma = close.rolling(SHORT_WINDOW).mean()
    long_ma = close.rolling(LONG_WINDOW).mean()
    # Get current and previous MA values for crossover detection
    curr_s, prev_s = short_ma.iloc[-1], short_ma.iloc[-2]
    curr_l, prev_l = long_ma.iloc[-1], long_ma.iloc[-2]
    # Log current market data
    logger.info(
        f"Price: ${price:,.2f} | MA{SHORT_WINDOW}: {curr_s:,.2f} | MA{LONG_WINDOW}: {curr_l:,.2f}")
    if position['in_position']:
        # If in position, check trailing stop first
        if check_trailing_stop(price):
            return  # Position closed, skip rest
        # Log holding status
        logger.info("Holding - trailing stop active")
        return
    # Check for Golden Cross (buy signal)
    if prev_s <= prev_l and curr_s > curr_l:
        usdt = get_balance()  # Get current balance
        if usdt < 10:
            # Skip if balance too low
            logger.warning("Low balance")
            return
        amount = calculate_position_size(usdt, price)  # Size position
        if amount < 0.0001:
            # Skip if amount too small (Binance minimum)
            return
        order = place_market_order('buy', amount)  # Execute buy
        if order:
            # Update position state on success
            position.update({
                'in_position': True,
                'entry_price': price,
                'amount': amount,
                'highest_price': price
            })
            logger.info(f"BUY {amount:.6f} BTC @ ${price:,.2f}")
    # Check for Death Cross (early sell signal if in position)
    elif position['in_position'] and prev_s >= prev_l and curr_s < curr_l:
        order = place_market_order('sell', position['amount'])  # Execute sell
        if order:
            # Calculate and log P/L
            pnl = (price - position['entry_price']) * position['amount']
            logger.info(f"Death Cross exit | P/L: {pnl:+.2f} USDT")
            # Reset position state
            position.update(
                {'in_position': False, 'entry_price': 0, 'amount': 0, 'highest_price': 0})
# =============================================================================
# MAIN LOOP
# =============================================================================
# Main entry function for the bot


def main():
    # Log startup messages
    logger.info("HF MA BOT STARTED")
    logger.info(
        f"5m | {SHORT_WINDOW}/{LONG_WINDOW} | 2% risk | $1k cap | 5% trail")
    if not PAPER_TRADING:
        # Safety prompt for real trading
        confirm = input("\nREAL TRADING! Type 'YES' to continue: ")
        if confirm != "YES":
            return  # Exit if not confirmed
    # Run optional backtest at startup for validation
    print("\nRunning quick backtest (last ~10 days)...")
    bt = run_hf_backtest()
    if not bt.empty:
        # Print top 5 results
        print(bt.head(5).to_string(index=False))
        # Highlight top combo
        print(
            f"\nTop combo: {bt.iloc[0]['short']}/{bt.iloc[0]['long']} → {bt.iloc[0]['return_%']}%")
    # Log that bot is live
    logger.info("Bot LIVE. Ctrl+C to stop.\n")
    cycle = 0  # Cycle counter
    while True:
        try:
            cycle += 1
            # Log cycle start with time
            logger.info(
                f"--- CYCLE {cycle} @ {datetime.now().strftime('%H:%M:%S')} ---")
            start = time.time()  # Time cycle duration
            run_strategy()  # Execute strategy
            # print balance
            usdt_balance = get_balance()
            logger.info(f"USDT Balance: {usdt_balance:,.2f}")
            elapsed = time.time() - start  # Calculate elapsed time
            # Calculate sleep to maintain schedule
            sleep = max(0, SCHEDULE_MINUTES * 60 - elapsed)
            logger.info(f"Cycle done in {elapsed:.1f}s | Sleep {sleep:.0f}s")
            time.sleep(sleep)  # Sleep until next cycle
        except KeyboardInterrupt:
            # Graceful exit on Ctrl+C
            logger.info("Stopped by user.")
            break
        except Exception as e:
            # Log critical errors and recover after 30s
            logger.critical(f"Error: {e}", exc_info=True)
            time.sleep(30)


# =============================================================================
# ENTRY POINT
# =============================================================================
# Standard Python idiom: run main() only if script is executed directly
if __name__ == "__main__":
    main()
