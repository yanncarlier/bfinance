"""
trading_bot_cli.py
==============
A fully automated BTC/USDT trading bot using CCXT and a **Moving Average Crossover** strategy.
Features:
- Live data from Binance (via CCXT) with yfinance fallback
- Buy when short MA crosses above long MA
- Sell when short MA crosses below long MA
- Risk-controlled position sizing (1% of USDT balance, max $100)
- 5% trailing stop-loss via limit order
- Paper trading mode (sandbox) by default
- Scheduled execution every 5 minutes
- Detailed logging to console
Author: You (with help from Grok)
Date: 2025-10-29
"""
# ──────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
import os                     # For environment variables and file paths
import time                   # For delays in the scheduler loop
import ccxt                   # Unified crypto exchange API (Binance, etc.)
# Data manipulation and analysis (OHLCV → DataFrame)
import pandas as pd
# Fallback data source (Yahoo Finance) if CCXT fails
import yfinance as yf
from dotenv import load_dotenv  # Load API keys from .env file securely
import schedule               # Lightweight job scheduler
import logging                # Structured logging (INFO, ERROR, etc.)
# ──────────────────────────────────────────────────────────────────────────────
# LOGGING CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
# Configure root logger to:
#   - Show INFO level and above
#   - Print timestamp, log level, and message
#   - Output to console only (file logging can be added later)
logging.basicConfig(
    level=logging.INFO,
    # e.g., "2025-10-29 12:34:56,789 - INFO - ..."
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)  # Get logger for this module
# ──────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT & CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
load_dotenv()  # Load variables from .env file (API_KEY, API_SECRET)
# API credentials — NEVER hardcode in production
API_KEY = os.getenv('API_KEY')
API_SECRET = os.getenv('API_SECRET')
# Trading pair and timeframe
SYMBOL = 'BTC/USDT'      # Trading pair on Binance (BTC priced in USDT)
TIMEFRAME = '1d'         # Daily candles (1d = 1 day)
# Strategy parameters
SHORT_WINDOW = 10        # Short-term moving average (10-day)
LONG_WINDOW = 30         # Long-term moving average (30-day)
# Risk management
POSITION_SIZE = 0.01     # Base position size factor (used with dynamic sizing)
STOP_LOSS_PCT = 0.05     # 5% stop-loss below entry price
# Exchange
EXCHANGE_ID = 'binance'  # Use Binance (supports testnet/sandbox)
# ──────────────────────────────────────────────────────────────────────────────
# TRADING BOT CLASS
# ──────────────────────────────────────────────────────────────────────────────


class TradingBot:
    """
    Core trading bot class.
    Handles data fetching, signal generation, and trade execution.
    Designed for scheduled runs (e.g., every 5 minutes).
    """

    def __init__(self):
        """
        Initialize the bot:
        - Set up CCXT exchange instance
        - Initialize position tracking
        """
        # Initialize CCXT exchange (Binance)
        self.exchange = getattr(ccxt, EXCHANGE_ID)({
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True,   # Prevent rate limit bans
            # PAPER TRADING: Use testnet (set False for LIVE)
            'sandbox': True,
        })
        # Current open position
        # Format: {'side': 'buy', 'amount': 0.001, 'entry_price': 60000.0}
        self.position = None
    # ──────────────────────────────────────────────────────────────────────────

    def fetch_data(self, period='100d'):
        """
        Fetch historical OHLCV (Open, High, Low, Close, Volume) data.
        Args:
            period (str): Time period, e.g., '100d' → last 100 days
        Returns:
            pd.DataFrame: DataFrame with 'close' column and datetime index
        Priority:
            1. CCXT (live exchange data)
            2. yfinance (fallback if CCXT fails)
        """
        try:
            # Parse period: '100d' → limit = 100
            limit = int(period[:-1]) if period.endswith('d') else 100
            # Fetch OHLCV from exchange
            ohlcv = self.exchange.fetch_ohlcv(
                SYMBOL, timeframe=TIMEFRAME, limit=limit
            )
            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            # Convert timestamp (milliseconds) to datetime and set as index
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            # Log warning and fallback to yfinance
            logger.warning(f"CCXT fetch failed: {e}. Using yfinance fallback.")
            ticker = 'BTC-USD'
            data = yf.download(ticker, period=period, progress=False)
            # Standardize column names to match CCXT format
            df = data[['Open', 'High', 'Low', 'Close', 'Volume']].rename(
                columns={'Close': 'close'}
            )
            return df
    # ──────────────────────────────────────────────────────────────────────────

    def generate_signal(self, df):
        """
        Generate trading signal using Moving Average Crossover.
        Buy Signal:  Short MA crosses above Long MA
        Sell Signal: Short MA crosses below Long MA
        Args:
            df (pd.DataFrame): OHLCV data with 'close' column
        Returns:
            str or None: 'buy', 'sell', or None (hold)
        """
        # Calculate rolling moving averages
        df['short_ma'] = df['close'].rolling(window=SHORT_WINDOW).mean()
        df['long_ma'] = df['close'].rolling(window=LONG_WINDOW).mean()
        # Get current and previous MA values
        current_short = df['short_ma'].iloc[-1]
        current_long = df['long_ma'].iloc[-1]
        prev_short = df['short_ma'].iloc[-2]
        prev_long = df['long_ma'].iloc[-2]
        # Bullish crossover: short MA moves from below to above long MA
        if current_short > current_long and prev_short <= prev_long:
            return 'buy'
        # Bearish crossover: short MA moves from above to below long MA
        elif current_short < current_long and prev_short >= prev_long:
            return 'sell'
        # No crossover → hold position
        return None
    # ──────────────────────────────────────────────────────────────────────────

    def execute_trade(self, signal):
        """
        Execute market buy or sell order based on signal.
        Args:
            signal (str): 'buy' or 'sell'
        Features:
        - Dynamic position sizing: 1% of USDT balance (max $100)
        - Market orders for instant execution
        - Stop-loss via limit order (5% below entry)
        - Updates internal position state
        """
        if not signal:
            return  # No signal → do nothing
        try:
            # Get current market price
            current_price = self.exchange.fetch_ticker(SYMBOL)['last']
            # Fetch account balance
            balance = self.exchange.fetch_balance()
            usdt_balance = balance['USDT']['free']  # Available USDT
            btc_balance = balance['BTC']['free']    # Available BTC
            # ── BUY LOGIC ──
            if (signal == 'buy' and
                self.position is None and
                    usdt_balance > 10):  # Minimum $10 to avoid dust
                # Risk 1% of balance, capped at $100
                amount_usdt = min(usdt_balance * 0.01, 100)
                # Convert to BTC amount, scaled by POSITION_SIZE
                amount = (amount_usdt / current_price) * POSITION_SIZE
                # Place market buy order
                order = self.exchange.create_market_buy_order(SYMBOL, amount)
                # Record position
                self.position = {
                    'side': 'buy',
                    'amount': amount,
                    'entry_price': current_price
                }
                # Log trade
                logger.info(
                    f"BUY: {amount:.6f} BTC at ${current_price:,.2f} | Order: {order}"
                )
                # ── STOP-LOSS ORDER ──
                stop_price = current_price * (1 - STOP_LOSS_PCT)
                self.exchange.create_stop_limit_order(
                    SYMBOL, 'sell', amount, stop_price, stop_price
                )
            # ── SELL LOGIC ──
            elif (signal == 'sell' and
                  self.position and
                  self.position['side'] == 'buy' and
                  btc_balance >= self.position['amount']):
                # Close full position
                order = self.exchange.create_market_sell_order(
                    SYMBOL, self.position['amount']
                )
                # Calculate profit/loss
                profit_loss = (
                    current_price - self.position['entry_price']
                ) * self.position['amount']
                # Log trade
                logger.info(
                    f"SELL: {self.position['amount']:.6f} BTC at ${current_price:,.2f} | "
                    f"P/L: ${profit_loss:,.2f}"
                )
                # Clear position
                self.position = None
        except Exception as e:
            # Log any error (network, API, balance, etc.)
            logger.error(f"Trade execution failed: {e}")
    # ──────────────────────────────────────────────────────────────────────────

    def run(self):
        """
        Main execution loop:
        1. Fetch latest market data
        2. Generate trading signal
        3. Execute trade if signal exists
        4. Log current state
        """
        df = self.fetch_data()
        signal = self.generate_signal(df)
        # Log current market and signal status
        logger.info(
            f"Current Price: ${df['close'].iloc[-1]:,.2f} | Signal: {signal or 'hold'}"
        )
        # Execute trade if signal is present
        self.execute_trade(signal)
# ──────────────────────────────────────────────────────────────────────────────
# SCHEDULER SETUP
# ──────────────────────────────────────────────────────────────────────────────


def job():
    """
    Wrapper function for scheduled execution.
    Instantiates a fresh bot instance and runs one cycle.
    """
    bot = TradingBot()
    bot.run()


# ──────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Start the bot when script is run directly.
    - Logs startup
    - Schedules job every 5 minutes
    - Runs once immediately
    - Enters infinite loop to keep scheduler alive
    """
    logger.info("Starting BTC Trading Bot (Paper Mode: True)")
    # Schedule the job to run every 5 minutes
    schedule.every(5).minutes.do(job)
    # Run once immediately on startup
    job()
    # Keep the script running indefinitely
    while True:
        schedule.run_pending()  # Check if any scheduled job is due
        time.sleep(1)            # Sleep 1 second to avoid CPU spin
