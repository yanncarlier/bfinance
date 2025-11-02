"""
trading_bot_dashboard.py
============
A live Streamlit dashboard for monitoring and controlling a BTC/USDT trading bot.
Features:
- Real-time price, moving averages, and trading signals
- Live P/L and position tracking
- Interactive Plotly chart with buy/sell markers
- Trade history table
- Bot log viewer
- Manual signal trigger
- Live / Paper trading toggle
- Configurable MA periods and refresh rate
- Persistent bot state via bot_state.json
- Safe log reading with error handling
Author: You (with help from Grok)
Date: 2025-10-29
"""
# ──────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
import streamlit as st                 # Web dashboard framework
# Data manipulation (trade history, signals)
import pandas as pd
import plotly.graph_objects as go      # Interactive charts
import numpy as np                     # For np.where in signal logic
from datetime import datetime          # Timestamp formatting
import time                            # For refresh loop delay
import json                            # For potential future state handling
# Import the trading bot class
# Assumes trading_bot.py is in the same directory
from trading_bot import TradingBot
# """
# Note: If trading_bot.py is in a subdirectory, use:
#     from bfinance.trading_bot import TradingBot
# """
# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BTC Trading Bot Dashboard",  # Browser tab title
    layout="wide"                            # Use full width for better layout
)
# """
# Configures the Streamlit app:
# - Title appears in browser tab
# - 'wide' layout maximizes horizontal space for charts/metrics
# """
st.title("Live BTC Trading Bot Dashboard")
# """
# Main dashboard title with red circle emoji for "live" status.
# Appears at the top of the page.
# """
# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR: USER CONTROLS
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.header("Bot Settings")
# """
# Creates a collapsible sidebar section titled "Bot Settings".
# All controls go here for clean UI.
# """
live_mode = st.sidebar.checkbox(
    "Live Trading (Disable Sandbox)",
    value=False
)
# """
# Toggle between:
# - Paper trading (sandbox = True) → safe, no real money
# - Live trading (sandbox = False) → real funds at risk
# Default: OFF (safe)
# """
ma_short = st.sidebar.number_input(
    "Short MA", min_value=5, max_value=50, value=10
)
# """
# Short-term moving average period.
# - Min 5, Max 50 to prevent extreme values
# - Default: 10 days
# """
ma_long = st.sidebar.number_input(
    "Long MA", min_value=20, max_value=200, value=30
)
# """
# Long-term moving average period.
# - Min 20 to ensure > short MA
# - Default: 30 days
# """
refresh_interval = st.sidebar.slider(
    "Refresh (seconds)", min_value=10, max_value=300, value=30
)
# """
# Auto-refresh interval for the dashboard.
# - Range: 10s to 5min
# - Default: 30 seconds
# """
if st.sidebar.button("Manual Run Signal"):
    st.session_state.manual_run = True
# """
# Button to manually trigger one bot cycle.
# - Sets a flag in session state
# - Bot runs immediately on next loop iteration
# """
# ──────────────────────────────────────────────────────────────────────────────
# BOT INITIALIZATION (CACHED)
# ──────────────────────────────────────────────────────────────────────────────


@st.cache_resource
def init_bot():
    """
    Initialize the trading bot with current sandbox setting.
    @st.cache_resource ensures:
    - Bot is created only once per app restart
    - Shared across sessions
    - Survives reruns
    """
    bot = TradingBot()
    # Override sandbox mode based on user toggle
    bot.exchange.sandbox = not live_mode
    return bot


bot = init_bot()
# """
# Create (or reuse) the bot instance.
# - First run: creates bot
# - Subsequent reruns: returns cached instance
# """
# ──────────────────────────────────────────────────────────────────────────────
# DATA FETCHING (CACHED)
# ──────────────────────────────────────────────────────────────────────────────


@st.cache_data(ttl=refresh_interval)
def get_data():
    """
    Fetch market data and calculate indicators.
    @st.cache_data with ttl=refresh_interval:
    - Refreshes data every N seconds
    - Prevents redundant API calls
    - Speeds up dashboard
    """
    # Fetch 100 days of OHLCV data from exchange
    df = bot.fetch_data(period='100d')
    # Calculate moving averages
    df['Short_MA'] = df['close'].rolling(ma_short).mean()
    df['Long_MA'] = df['close'].rolling(ma_long).mean()
    # Generate crossover signals — FIXED: Use .loc to avoid chained assignment
    df['Signal'] = 0
    df.loc[df.index[ma_short]:, 'Signal'] = np.where(
        df['Short_MA'][ma_short:] > df['Long_MA'][ma_short:], 1, 0
    )
    # Position = change in signal (1 = buy, -1 = sell)
    df['Position'] = df['Signal'].diff()
    return df


# ──────────────────────────────────────────────────────────────────────────────
# MAIN DISPLAY LOOP
# ──────────────────────────────────────────────────────────────────────────────
placeholder = st.empty()
# """
# Create a placeholder container.
# - All UI elements go inside
# - Updated in-place on each loop → no flickering
# """
# Load latest bot state from disk
bot.load_state()
# Check for manual run request
if 'manual_run' in st.session_state:
    bot.run()                     # Execute one full bot cycle
    del st.session_state.manual_run  # Clear flag
# ──────────────────────────────────────────────────────────────────────────────
# INFINITE REFRESH LOOP
# ──────────────────────────────────────────────────────────────────────────────
while True:
    # Fetch fresh data (cached with TTL)
    df = get_data()
    latest = df.iloc[-1]   # Most recent row
    prev = df.iloc[-2]     # Previous row
    # Update entire dashboard inside placeholder
    with placeholder.container():
        # ── METRICS ROW ──────────────────────────────────────────────────────
        col1, col2, col3, col4 = st.columns(4)
        # """
        # Four equal-width columns for key metrics.
        # """
        col1.metric(
            "BTC Price",
            f"${latest['close']:,.2f}",
            f"{latest['close'] - prev['close']:,.0f}"
        )
        # """
        # Current price + 24h change.
        # - Delta shows absolute change
        # - Formatted with commas
        # """
        col2.metric("Short MA", f"{latest['Short_MA']:,.0f}")
        col3.metric("Long MA", f"{latest['Long_MA']:,.0f}")
        # """
        # Current moving average values.
        # - Rounded to nearest dollar
        # """
        current_signal = "BUY" if latest['Short_MA'] > latest['Long_MA'] else "SELL"
        col4.metric(
            "Signal",
            current_signal,
            "Bullish" if current_signal == "BUY" else "Bearish"
        )
        # """
        # Current trading signal based on MA crossover.
        # - Green = BUY, Red = SELL
        # """
        # ── CURRENT POSITION ────────────────────────────────────────────────
        st.subheader("Current Position")
        if bot.position:
            current_price = latest['close']
            # Absolute P/L
            pnl = (current_price -
                   bot.position['entry_price']) * bot.position['amount']
            # Percentage P/L
            pnl_pct = (
                pnl / (bot.position['entry_price'] * bot.position['amount'])) * 100
            st.success(
                f"**LONG {bot.position['amount']:.6f} BTC** @ ${bot.position['entry_price']:,.2f} | "
                f"P/L: ${pnl:,.2f} ({pnl_pct:+.2f}%) | "
                f"Mode: {'LIVE' if not bot.exchange.sandbox else 'PAPER'}"
            )
            """
            Success box (green) for open long position.
            Shows:
            - Amount in BTC
            - Entry price
            - P/L in $ and %
            - Trading mode
            """
        else:
            st.info("No open position")
            # """
            # Info box (blue) when flat.
            # """
        # ── INTERACTIVE CHART ───────────────────────────────────────────────
        chart_placeholder = st.empty()
        with chart_placeholder.container():
            fig = go.Figure()
            # Price line
            fig.add_trace(go.Scatter(
                x=df.index, y=df['close'],
                name='BTC Price', line=dict(width=2)
            ))
            # Moving averages
            fig.add_trace(go.Scatter(
                x=df.index, y=df['Short_MA'],
                name=f'{ma_short}d MA', line=dict(dash='dot')
            ))
            fig.add_trace(go.Scatter(
                x=df.index, y=df['Long_MA'],
                name=f'{ma_long}d MA', line=dict(dash='dot')
            ))
            # Buy signals (green triangles)
            buys = df[df['Position'] == 1]
            if not buys.empty:
                fig.add_trace(go.Scatter(
                    x=buys.index, y=buys['close'],
                    mode='markers', name='Buy Signal',
                    marker=dict(symbol='triangle-up', size=12, color='green')
                ))
            # Sell signals (red triangles)
            sells = df[df['Position'] == -1]
            if not sells.empty:
                fig.add_trace(go.Scatter(
                    x=sells.index, y=sells['close'],
                    mode='markers', name='Sell Signal',
                    marker=dict(symbol='triangle-down', size=12, color='red')
                ))
            # Chart styling
            fig.update_layout(
                title="BTC/USDT Live Chart",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                height=500
            )
            # Render chart with unique key to avoid duplicates
            st.plotly_chart(
                fig,
                use_container_width=True,
                # Dynamic key prevents Streamlit error
                key=f"chart_{int(time.time())}"
            )
        # ── TRADE HISTORY TABLE ─────────────────────────────────────────────
        st.subheader("Trade History")
        if bot.trades:
            trade_df = pd.DataFrame(bot.trades)
            trade_df['time'] = pd.to_datetime(
                trade_df['time'])  # Ensure datetime
            # Show last 20 trades, newest first
            st.dataframe(
                trade_df.sort_values('time', ascending=False).head(20),
                use_container_width=True
            )
        else:
            st.write("No trades executed yet.")
        # ── BOT LOG VIEWER ──────────────────────────────────────────────────
        st.subheader("Recent Bot Log")
        try:
            with open('bot.log', 'r') as f:
                logs = f.read().splitlines()
                # Show last 10 log lines
                recent_logs = logs[-10:] if len(logs) > 10 else logs
                st.code("\n".join(recent_logs))  # Monospaced code block
        except FileNotFoundError:
            st.warning("No bot.log found. Run the bot once to generate logs.")
        except Exception as e:
            st.error(f"Error reading log: {e}")
    # ── REFRESH & PERSISTENCE ───────────────────────────────────────────
    time.sleep(refresh_interval)  # Wait before next update
    bot.save_state()              # Save position/trades to disk
