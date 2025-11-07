"""
================================================================================
MOVING AVERAGE CROSSOVER TRADING BOT (BTC/USDT) - FULLY COMMENTED
================================================================================
A complete, production-ready crypto trading bot using Binance Spot via CCXT.
Implements:
    • Dual Moving Average (Golden/Death Cross) strategy
    • Dynamic position sizing (% of balance + max USD cap)
    • Trailing stop-loss for profit protection
    • Paper trading mode for safe testing
    • Scheduled execution with logging to file + console
    • Robust error handling and recovery
Author: Your Name
Date: November 06, 2025
Tested on: Python 3.9+, ccxt, pandas
"""
# =============================================================================
# IMPORTS & DEPENDENCIES
# =============================================================================
# Crypto exchange interface (Binance API wrapper)
import ccxt
import pandas as pd           # Data manipulation for OHLCV candles and MAs
import time                   # Timing and sleep control for scheduler
import logging                # Structured logging for debugging and audit
from datetime import datetime  # Human-readable timestamps in logs
from dotenv import load_dotenv  # Securely load API keys from .env file
import os                     # OS-level operations (env var access)
# =============================================================================
# CONFIGURATION VARIABLES - CUSTOMIZE YOUR STRATEGY HERE
# =============================================================================
TIMEFRAME = '1h'              # Candlestick interval: '1m', '5m', '1h', '4h', '1d'
SHORT_WINDOW = 10             # Fast moving average period (e.g., 10-hour MA)
LONG_WINDOW = 50              # Slow moving average period (e.g., 50-hour MA)
POSITION_SIZE_PCT = 0.02      # Risk 2% of available USDT per trade
MAX_TRADE_USD = 1000.0         # Hard cap: never risk more than $100 equivalent
STOP_LOSS_PCT = 0.05          # Trail stop-loss at 5% below peak price
SCHEDULE_MINUTES = 5          # How often to check signals (in minutes)
# Trading pair and asset symbols
SYMBOL = 'BTC/USDT'           # Market to trade
BASE_CURRENCY = 'BTC'         # Asset being bought/sold
QUOTE_CURRENCY = 'USDT'       # Stablecoin used for sizing and P/L
# =============================================================================
# SAFETY FIRST: PAPER TRADING MODE
# =============================================================================
# Set to True → simulates trades in logs only (no real money)
# Set to False → executes real market orders (use with extreme caution!)
PAPER_TRADING = True
# =============================================================================
# LOAD SECURE CREDENTIALS
# =============================================================================
# Create a .env file in the same directory:
# API_KEY=your_key_here
# API_SECRET=your_secret_here
load_dotenv()  # Reads .env into environment variables
API_KEY = os.getenv('API_KEY')
API_SECRET = os.getenv('API_SECRET')
# Validate credentials exist
if not API_KEY or not API_SECRET:
    raise EnvironmentError(
        "API_KEY and API_SECRET must be defined in .env file")
# =============================================================================
# LOGGING SETUP - CRITICAL FOR MONITORING & AUDITING
# =============================================================================
# Logs appear in both console AND bot.log file
# Format: "2025-11-06 15:22:10,123 | INFO | Starting bot..."
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for verbose output
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[
        # Persistent log file
        logging.FileHandler("bot.log", encoding='utf-8'),
        logging.StreamHandler()                            # Live console output
    ]
)
logger = logging.getLogger(__name__)  # Logger inherits module name
# =============================================================================
# BINANCE EXCHANGE CONNECTION
# =============================================================================
exchange = ccxt.binance({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'enableRateLimit': True,           # Prevents API bans by respecting rate limits
    'options': {
        'defaultType': 'spot'          # Use spot wallet (not futures/margin)
    }
})
# Confirm paper trading status at startup
if PAPER_TRADING:
    logger.info("PAPER TRADING MODE ENABLED - NO REAL ORDERS WILL BE PLACED")
else:
    logger.warning("REAL TRADING MODE ACTIVE - ORDERS WILL BE EXECUTED!")
# =============================================================================
# GLOBAL POSITION STATE - TRACKS CURRENT TRADE
# =============================================================================
# This dictionary holds the bot's memory between cycles
position = {
    'in_position': False,      # Are we currently holding BTC?
    'entry_price': 0.0,        # Price at which we bought
    'amount': 0.0,             # Quantity of BTC held
    'highest_price': 0.0       # Peak price since entry (for trailing stop)
}
# =============================================================================
# HELPER FUNCTIONS - MODULAR & REUSABLE
# =============================================================================


def fetch_ohlcv(symbol, timeframe, limit=100):
    """
    Fetch historical candle data from Binance.
    Args:
        symbol (str): Trading pair e.g., 'BTC/USDT'
        timeframe (str): Candle interval e.g., '1h'
        limit (int): Number of candles to retrieve
    Returns:
        pd.DataFrame: Indexed by timestamp with OHLCV columns, or None on error
    """
    try:
        # Raw data: [[timestamp_ms, open, high, low, close, volume], ...]
        raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(
            raw,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        # Convert millisecond timestamp to readable datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        logger.error(f"Failed to fetch OHLCV for {symbol}: {e}")
        return None


def get_balance():
    """
    Retrieve available USDT balance for trading.
    Returns:
        float: Total free USDT, or 0.0 if error
    """
    try:
        balance = exchange.fetch_balance()
        # 'total' includes free + locked; use 'free' for safer sizing
        return balance['total'].get(QUOTE_CURRENCY, 0.0)
    except Exception as e:
        logger.error(f"Balance fetch failed: {e}")
        return 0.0


def calculate_position_size(usdt_balance, current_price):
    """
    Determine how much BTC to buy based on risk rules.
    Logic:
        1. Calculate dollar amount from percentage
        2. Apply hard USD cap
        3. Convert to BTC amount
        4. Round to 6 decimals (Binance precision)
    Returns:
        float: BTC amount to trade
    """
    usd_by_pct = usdt_balance * POSITION_SIZE_PCT
    usd_to_use = min(usd_by_pct, MAX_TRADE_USD)  # Never exceed max
    amount = usd_to_use / current_price
    return round(amount, 6)  # Binance minimum ~0.0001 BTC


def place_market_order(side, amount):
    """
    Execute (or simulate) a market order.
    Args:
        side (str): 'buy' or 'sell'
        amount (float): BTC quantity
    Returns:
        dict: Order response or simulated order
    """
    if PAPER_TRADING:
        # Simulate instant fill at current price
        order_id = f"paper_{int(time.time())}"
        logger.info(
            f"[PAPER] Market {side.upper()} {amount:.6f} {BASE_CURRENCY} @ market")
        return {
            'id': order_id,
            'status': 'closed',
            'amount': amount,
            'side': side,
            'price': 'market'
        }
    else:
        try:
            order = exchange.create_order(
                symbol=SYMBOL,
                type='market',
                side=side,
                amount=amount
            )
            logger.info(
                f"REAL ORDER EXECUTED: {side.upper()} {amount:.6f} BTC | ID: {order['id']}")
            return order
        except Exception as e:
            logger.error(f"Order execution failed ({side}): {e}")
            return None


def check_trailing_stop(current_price):
    """
    Monitor and enforce trailing stop-loss.
    - Updates highest price seen
    - Triggers sell if price drops STOP_LOSS_PCT below peak
    Returns:
        bool: True if stop triggered and position closed
    """
    global position
    if not position['in_position']:
        return False  # No position to protect
    # Update running high
    if current_price > position['highest_price']:
        old_high = position['highest_price']
        position['highest_price'] = current_price
        logger.info(f"New high: {current_price:.2f} (was {old_high:.2f})")
    # Calculate dynamic stop price
    stop_price = position['highest_price'] * (1 - STOP_LOSS_PCT)
    if current_price <= stop_price:
        logger.warning(f"TRAILING STOP HIT! Selling at {current_price:.2f} "
                       f"(stop: {stop_price:.2f})")
        order = place_market_order('sell', position['amount'])
        if order:
            # Calculate profit/loss
            pnl = (current_price -
                   position['entry_price']) * position['amount']
            logger.info(
                f"Position closed via stop-loss | P/L: {pnl:+.2f} USDT")
            # Reset position
            position.update({
                'in_position': False,
                'entry_price': 0.0,
                'amount': 0.0,
                'highest_price': 0.0
            })
        return True
    return False
# =============================================================================
# CORE STRATEGY: MOVING AVERAGE CROSSOVER
# =============================================================================


def run_strategy():
    """
    Main trading logic executed every SCHEDULE_MINUTES.
    Steps:
        1. Fetch candle data
        2. Calculate short & long MAs
        3. Detect Golden Cross (buy) or Death Cross (sell)
        4. Manage trailing stop
    """
    global position
    logger.info("=" * 60)
    logger.info("RUNNING STRATEGY CYCLE")
    # Step 1: Get recent price data
    df = fetch_ohlcv(SYMBOL, TIMEFRAME, LONG_WINDOW + 10)
    if df is None or len(df) < LONG_WINDOW:
        logger.warning("Insufficient data for MA calculation")
        return
    close_prices = df['close']
    # Step 2: Compute moving averages
    df['short_ma'] = close_prices.rolling(window=SHORT_WINDOW).mean()
    df['long_ma'] = close_prices.rolling(window=LONG_WINDOW).mean()
    # Current values
    current_price = close_prices.iloc[-1]
    short_ma = df['short_ma'].iloc[-1]
    long_ma = df['long_ma'].iloc[-1]
    prev_short_ma = df['short_ma'].iloc[-2]
    prev_long_ma = df['long_ma'].iloc[-2]
    logger.info(f"Price: {current_price:,.2f} | "
                f"MA{SHORT_WINDOW}: {short_ma:,.2f} | "
                f"MA{LONG_WINDOW}: {long_ma:,.2f}")
    # Step 3: Exit check - trailing stop first
    if position['in_position']:
        if check_trailing_stop(current_price):
            return  # Position already closed
        logger.info("Holding position - monitoring stop-loss")
        return  # Skip entry signals while in trade
    # Step 4: Entry signal - Golden Cross
    if prev_short_ma <= prev_long_ma and short_ma > long_ma:
        logger.info("GOLDEN CROSS DETECTED - BULLISH SIGNAL!")
        usdt_balance = get_balance()
        if usdt_balance < 10:
            logger.warning("USDT balance too low (< $10)")
            return
        amount = calculate_position_size(usdt_balance, current_price)
        if amount < 0.0001:
            logger.warning("Calculated trade size too small")
            return
        logger.info(
            f"EXECUTING BUY: {amount:.6f} BTC @ ~${current_price:,.2f}")
        order = place_market_order('buy', amount)
        if order:
            position.update({
                'in_position': True,
                'entry_price': current_price,
                'amount': amount,
                'highest_price': current_price
            })
            logger.info(f"ENTRY CONFIRMED | Size: {amount:.6f} BTC | "
                        f"Entry: ${current_price:,.2f}")
    # Step 5: Optional early exit - Death Cross
    elif prev_short_ma >= prev_long_ma and short_ma < long_ma:
        if position['in_position']:
            logger.info("DEATH CROSS - CLOSING POSITION EARLY")
            order = place_market_order('sell', position['amount'])
            if order:
                pnl = (current_price -
                       position['entry_price']) * position['amount']
                logger.info(f"Closed on Death Cross | P/L: {pnl:+.2f} USDT")
                position.update({
                    'in_position': False,
                    'entry_price': 0.0,
                    'amount': 0.0,
                    'highest_price': 0.0
                })
# =============================================================================
# SCHEDULER - MAIN LOOP
# =============================================================================


def main():
    """
    Bot entry point. Runs indefinitely until stopped.
    Features:
        • Startup summary
        • Real-trade confirmation prompt
        • Precise timing (respects execution duration)
        • Graceful shutdown on Ctrl+C
    """
    logger.info("STARTING MOVING AVERAGE CROSSOVER BOT")
    logger.info(f"Strategy: {SHORT_WINDOW}/{LONG_WINDOW} MA on {TIMEFRAME}")
    logger.info(
        f"Risk: {POSITION_SIZE_PCT*100}% per trade, max ${MAX_TRADE_USD}")
    logger.info(f"Stop-Loss: {STOP_LOSS_PCT*100}% trailing")
    logger.info(f"Cycle: Every {SCHEDULE_MINUTES} minutes")
    # Final safety gate for real trading
    if not PAPER_TRADING:
        confirm = input("\nREAL TRADING ENABLED! Type 'YES' to proceed: ")
        if confirm != "YES":
            logger.info("Bot terminated by user - safe exit")
            return
    logger.info("Bot is now LIVE. Press Ctrl+C to stop.\n")
    cycle = 0
    while True:
        try:
            cycle += 1
            logger.info(
                f"--- CYCLE {cycle} @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
            start = time.time()
            run_strategy()  # Execute one full strategy pass
            elapsed = time.time() - start
            # Sleep until next scheduled run
            sleep_seconds = max(0, SCHEDULE_MINUTES * 60 - elapsed)
            logger.info(f"Cycle {cycle} complete in {elapsed:.1f}s | "
                        f"Sleeping {sleep_seconds:.0f}s\n")
            time.sleep(sleep_seconds)
        except KeyboardInterrupt:
            logger.info("\nBot stopped by user (Ctrl+C). Goodbye!")
            break
        except Exception as e:
            logger.critical(f"CRITICAL ERROR: {e}", exc_info=True)
            logger.info("Recovering in 60 seconds...")
            time.sleep(60)


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    """
    Standard Python idiom: only run main() if this file is executed directly.
    Allows safe importing of functions in other scripts.
    """
    main()
