"""
trading_bot.py
==============
A live Bitcoin (BTC/USDT) trading bot using CCXT and a Moving Average Crossover strategy.
Features:
- Fetches OHLCV data from Binance (or fallback to yfinance)
- Generates buy/sell signals using short/long MA crossover
- Executes market orders with position sizing (1% risk)
- Implements stop-loss (5% below entry)
- Logs all actions to console + bot.log
- Persists position & trade history to bot_state.json
- Supports paper trading (sandbox) and live mode
Author: You (with help from Grok)
Date: 2025-10-29
"""
# ──────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
import os                     # For environment variables and file paths
import time                   # For timestamps and delays (if needed)
import ccxt                   # Unified crypto exchange API (Binance, etc.)
import pandas as pd           # Data manipulation (OHLCV → DataFrame)
import json                   # Save/load bot state to JSON
from dotenv import load_dotenv  # Load API keys from .env file
import logging                # Structured logging (INFO, ERROR, etc.)
# ──────────────────────────────────────────────────────────────────────────────
# LOGGING CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
# Configure logging to:
#   1. Write to bot.log (persistent file)
#   2. Print to console (real-time monitoring)
logging.basicConfig(
    level=logging.INFO,  # Show INFO and above (INFO, WARNING, ERROR)
    # Timestamp + level + msg
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bot.log"),      # Append logs to file
        logging.StreamHandler()              # Print to terminal
    ]
)
logger = logging.getLogger(__name__)  # Logger for this module
# ──────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT & CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
load_dotenv()  # Load variables from .env file (API_KEY, API_SECRET)
# API credentials (never hardcode in production!)
API_KEY = os.getenv('API_KEY')
API_SECRET = os.getenv('API_SECRET')
# Trading pair and timeframe
SYMBOL = 'BTC/USDT'      # Trading pair on Binance
TIMEFRAME = '1d'         # Daily candles (1d = 1 day)
# Strategy parameters
SHORT_WINDOW = 10        # Short-term moving average (e.g., 10-day)
LONG_WINDOW = 30         # Long-term moving average (e.g., 30-day)
# Risk management
POSITION_SIZE = 0.01     # Risk 1% of balance per trade (configurable)
STOP_LOSS_PCT = 0.05     # 5% stop-loss below entry price
# Exchange
EXCHANGE_ID = 'binance'  # Use Binance (supports testnet)
# ──────────────────────────────────────────────────────────────────────────────
# TRADING BOT CLASS
# ──────────────────────────────────────────────────────────────────────────────


class TradingBot:
    """
    Core trading bot class.
    Handles data fetching, signal generation, trade execution, and state persistence.
    """

    def __init__(self):
        """
        Initialize the bot:
        - Set up CCXT exchange instance
        - Initialize position and trade history
        - Load saved state (if any)
        """
        # Initialize CCXT exchange (Binance)
        self.exchange = getattr(ccxt, EXCHANGE_ID)({
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True,   # Respect exchange rate limits
            # PAPER TRADING: Use testnet (set False for LIVE)
            'sandbox': True,
        })
        # Current open position (None if flat)
        self.position = None
        # List of executed trades
        self.trades = []
        # Load previous state from disk
        self.load_state()
    # ──────────────────────────────────────────────────────────────────────────

    def fetch_data(self, period='100d'):
        """
        Fetch historical OHLCV data.
        Args:
            period (str): Period string, e.g., '100d' → last 100 days
        Returns:
            pd.DataFrame: DataFrame with 'close' column and datetime index
        Fallback: If CCXT fails (e.g., rate limit), use yfinance.
        """
        try:
            # Parse period: '100d' → limit = 100
            limit = int(period[:-1]) if period.endswith('d') else 100
            # Fetch OHLCV from exchange
            ohlcv = self.exchange.fetch_ohlcv(
                SYMBOL, timeframe=TIMEFRAME, limit=limit)
            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            # Convert timestamp (ms) to datetime and set as index
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            # Log warning and fallback to yfinance
            logger.warning(f"CCXT fetch failed: {e}. Using yfinance.")
            import yfinance as yf
            ticker = 'BTC-USD'
            data = yf.download(ticker, period=period, progress=False)
            # Standardize column names and return
            return data[['Open', 'High', 'Low', 'Close', 'Volume']].rename(
                columns={'Close': 'close'}
            )
    # ──────────────────────────────────────────────────────────────────────────

    def generate_signal(self, df):
        """
        Generate trading signal using MA crossover.
        Buy:  Short MA crosses above Long MA
        Sell: Short MA crosses below Long MA
        Args:
            df (pd.DataFrame): OHLCV data with 'close' column
        Returns:
            str or None: 'buy', 'sell', or None
        """
        # Calculate moving averages
        df['short_ma'] = df['close'].rolling(window=SHORT_WINDOW).mean()
        df['long_ma'] = df['close'].rolling(window=LONG_WINDOW).mean()
        # Get current and previous MA values
        current_short = df['short_ma'].iloc[-1]
        current_long = df['long_ma'].iloc[-1]
        prev_short = df['short_ma'].iloc[-2]
        prev_long = df['long_ma'].iloc[-2]
        # Bullish crossover: short MA crosses above long MA
        if current_short > current_long and prev_short <= prev_long:
            return 'buy'
        # Bearish crossover: short MA crosses below long MA
        elif current_short < current_long and prev_short >= prev_long:
            return 'sell'
        # No crossover
        return None
    # ──────────────────────────────────────────────────────────────────────────

    def execute_trade(self, signal):
        """
        Execute buy or sell order based on signal.
        Args:
            signal (str): 'buy' or 'sell'
        Features:
        - Position sizing: 1% of USDT balance (max $100)
        - Market orders
        - Stop-loss via limit order
        - Updates position and trade log
        """
        if not signal:
            return  # No action
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
                    usdt_balance > 10):  # Minimum $10 to trade
                # Risk 1% of balance, cap at $100
                amount_usdt = min(usdt_balance * 0.01, 100)
                # Convert to BTC amount
                amount = amount_usdt / current_price
                # Place market buy order
                order = self.exchange.create_market_buy_order(SYMBOL, amount)
                # Record position
                self.position = {
                    'side': 'buy',
                    'amount': amount,
                    'entry_price': current_price
                }
                # Log trade
                self.trades.append({
                    'time': pd.Timestamp.now(),
                    'action': 'BUY',
                    'price': current_price,
                    'amount': amount
                })
                logger.info(f"BUY: {amount:.6f} BTC at ${current_price:,.2f}")
                # ── STOP-LOSS ──
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
                # Calculate P/L
                profit_loss = (
                    current_price - self.position['entry_price']
                ) * self.position['amount']
                # Log trade
                self.trades.append({
                    'time': pd.Timestamp.now(),
                    'action': 'SELL',
                    'price': current_price,
                    'profit': profit_loss
                })
                logger.info(
                    f"SELL: {self.position['amount']:.6f} BTC at ${current_price:,.2f} | "
                    f"P/L: ${profit_loss:,.2f}"
                )
                # Close position
                self.position = None
        except Exception as e:
            # Log any error (API, network, etc.)
            logger.error(f"Trade execution failed: {e}")
        finally:
            # Always save state after trade attempt
            self.save_state()
    # ──────────────────────────────────────────────────────────────────────────

    def run(self):
        """
        Main bot loop:
        1. Fetch data
        2. Generate signal
        3. Execute trade (if any)
        4. Log status
        """
        df = self.fetch_data()
        signal = self.generate_signal(df)
        # Log current market state
        logger.info(
            f"Price: ${df['close'].iloc[-1]:,.2f} | "
            f"Signal: {signal or 'hold'}"
        )
        # Execute if signal exists
        self.execute_trade(signal)
    # ──────────────────────────────────────────────────────────────────────────

    def save_state(self):
        """
        Persist bot state to disk (bot_state.json).
        Ensures continuity after restart.
        """
        state = {
            'position': self.position,
            'trades': self.trades
        }
        with open('bot_state.json', 'w') as f:
            # `default=str` handles non-serializable objects (e.g., Timestamp)
            json.dump(state, f, default=str, indent=2)
    # ──────────────────────────────────────────────────────────────────────────

    def load_state(self):
        """
        Load previous bot state from disk.
        Restores position and trade history on startup.
        """
        try:
            with open('bot_state.json', 'r') as f:
                state = json.load(f)
                self.position = state.get('position')
                self.trades = state.get('trades', [])
        except FileNotFoundError:
            # First run: no state file yet
            pass
        except Exception as e:
            logger.warning(f"Failed to load state: {e}")
