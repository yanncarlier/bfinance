import ccxt
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
from itertools import product
from concurrent.futures import ThreadPoolExecutor
# Load environment variables from .env file
load_dotenv()
# Retrieve Binance API key and secret
API_KEY = os.getenv('API_KEY')
API_SECRET = os.getenv('API_SECRET')
# Ensure credentials are available
if not API_KEY or not API_SECRET:
    raise EnvironmentError("API_KEY and API_SECRET must be in .env file")
# =============================================================================
# BACKTEST CONFIGURATION (Backtest-Specific Variables)
# =============================================================================
BACKTEST_CONFIG = {
    # Candle timeframe (default; overridden in loop)
    'candle_timeframe': '1h',
    'symbol': 'BTC/USDT',          # Trading pair
    'init_usdt': 10000.0,          # Initial USDT balance for backtesting
    'data_limit': 5000,            # ~3.5 days of 1m data; increase for longer periods
    'top_combos_to_display': 10,   # Number of top combinations to print
}
# =============================================================================
# STRATEGY CONFIGURATION (Shared with Live Trading; Override as Needed)
# =============================================================================
# These are backtest defaults but can be overridden (e.g., from LIVE_CONFIG if imported)
STRATEGY_CONFIG = {
    # Risk percentage of balance per trade (match live)
    'position_size_pct': 0.15,
    # Maximum USD to risk per trade (match live)
    'max_trade_usd': 500.0,
    'take_profit_pct': 0.03,        # Take-profit percentage (match live)
    # Trailing stop-loss percentage (match live)
    'trailing_stop_pct': 0.02,
    'taker_fee_pct': 0.001,         # Binance spot taker fee (0.1%)
    # Simulated slippage (0.05% for market orders)
    'slippage_pct': 0.0005,
}
# =============================================================================
# CORE SETUP (Not specific to backtest)
# =============================================================================
# Initialize Binance exchange via CCXT (spot by default)
exchange = ccxt.binance({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'enableRateLimit': True,
    'options': {'defaultType': 'spot'}
})


def fetch_ohlcv(symbol, timeframe, limit):
    """
    Fetch OHLCV data from Binance.
    Args:
        symbol (str): Trading pair (e.g., 'BTC/USDT')
        timeframe (str): Candle interval (e.g., '5m')
        limit (int): Number of candles to retrieve
    Returns:
        pd.DataFrame: OHLCV data indexed by timestamp, or None on failure
    """
    try:
        # Use keyword arg for limit to avoid 'since' misinterpretation
        raw = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(
            raw, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        print(f"Fetch failed: {e}")  # Use print for backtest visibility
        return None


def calculate_adx(high, low, close, period=14):  # Optional trend filter
    """
    Calculate ADX for trend strength (>25 = trending).
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


def backtest_one(df, short_win, long_win,
                 init_usdt=BACKTEST_CONFIG['init_usdt'],
                 pos_pct=STRATEGY_CONFIG['position_size_pct'],
                 max_usd=STRATEGY_CONFIG['max_trade_usd'],
                 take_profit_pct=STRATEGY_CONFIG['take_profit_pct'],
                 trail_pct=STRATEGY_CONFIG['trailing_stop_pct'],
                 taker_fee_pct=STRATEGY_CONFIG['taker_fee_pct'],
                 slippage_pct=STRATEGY_CONFIG['slippage_pct'],
                 use_adx_filter=False):
    """
    Backtest a single MA combination on historical data (spot long-only with fees/slippage).
    Args:
        ... (as before)
        take_profit_pct (float): Fixed take-profit rate (e.g., 0.03 for 3%)
        taker_fee_pct (float): Taker fee rate (e.g., 0.001 for 0.1%)
        slippage_pct (float): Slippage rate (e.g., 0.0005 for 0.05%)
        use_adx_filter (bool): If True, only trade if ADX > 25
    Returns:
        dict: Performance metrics + buy_hold_ret for benchmark
    """
    close = df['close'].values
    high = df['high'].values  # For ADX
    low = df['low'].values
    n = len(close)
    short_ma = pd.Series(close).rolling(short_win).mean().values
    long_ma = pd.Series(close).rolling(long_win).mean().values
    if use_adx_filter:
        adx = calculate_adx(pd.Series(high), pd.Series(low),
                            pd.Series(close)).values
    cash = init_usdt
    btc = entry = peak = 0.0
    trades = []
    for i in range(max(short_win, long_win), n):
        price = close[i]
        adx_ok = not use_adx_filter or adx[i] > 25
        if btc > 0:
            if price > peak:
                peak = price
            stop = peak * (1 - trail_pct)
            tp = entry * (1 + take_profit_pct)
            if price <= stop:
                # Exit on stop: apply slippage and fee to proceeds
                proceeds = (btc * price) * (1 - slippage_pct) * \
                    (1 - taker_fee_pct)
                cash += proceeds
                gross_pnl = btc * (price - entry)
                # Approx net (ignores entry fee/slip)
                net_pnl = proceeds - (btc * entry)
                trades.append({'pnl': net_pnl})
                btc = entry = peak = 0.0
                continue
            elif price >= tp:
                # Exit on take-profit: apply slippage and fee
                proceeds = (btc * price) * (1 - slippage_pct) * \
                    (1 - taker_fee_pct)
                cash += proceeds
                gross_pnl = btc * (price - entry)
                net_pnl = proceeds - (btc * entry)
                trades.append({'pnl': net_pnl})
                btc = entry = peak = 0.0
                continue
            if short_ma[i-1] >= long_ma[i-1] and short_ma[i] < long_ma[i]:
                # Exit on bear cross: apply slippage and fee
                proceeds = (btc * price) * (1 - slippage_pct) * \
                    (1 - taker_fee_pct)
                cash += proceeds
                gross_pnl = btc * (price - entry)
                net_pnl = proceeds - (btc * entry)
                trades.append({'pnl': net_pnl})
                btc = entry = peak = 0.0
                continue
        if (btc == 0 and
            short_ma[i-1] <= long_ma[i-1] and short_ma[i] > long_ma[i] and
                adx_ok):
            # Entry on bull cross: apply slippage and fee
            intended_usd = min(cash * pos_pct, max_usd)
            if intended_usd < 10:  # Min trade size
                continue
            # Slippage reduces BTC received; fee increases cash outflow
            btc_received = (intended_usd / price) * (1 - slippage_pct)
            btc = round(btc_received, 6)
            cash -= intended_usd * (1 + taker_fee_pct)
            entry = peak = price
    if btc > 0:
        # Final exit at last price
        final_price = close[-1]
        proceeds = (btc * final_price) * \
            (1 - slippage_pct) * (1 - taker_fee_pct)
        cash += proceeds
        gross_pnl = btc * (final_price - entry)
        net_pnl = proceeds - (btc * entry)
        trades.append({'pnl': net_pnl})
    pnl = cash - init_usdt
    ret = pnl / init_usdt * 100
    wins = sum(1 for t in trades if t['pnl'] > 0)
    winrate = wins / len(trades) * 100 if trades else 0
    equity = np.cumsum([init_usdt] + [t['pnl'] for t in trades])
    dd = np.maximum.accumulate(equity) - equity
    max_dd = dd.max() / init_usdt * 100 if equity.size else 0
    # Buy-hold benchmark (no fees/slippage for pure HODL)
    buy_hold_ret = (close[-1] / close[0] - 1) * 100
    return {
        'short': int(short_win), 'long': int(long_win),
        'return_%': round(ret, 3), 'trades': len(trades),
        'winrate_%': round(winrate, 1), 'max_dd_%': round(max_dd, 2),
        'final_usdt': round(cash, 2),
        'buy_hold_%': round(buy_hold_ret, 3)  # Benchmark
    }


def color_pct(pct):
    """
    Color the percentage: green if positive, red if negative, default if zero.
    """
    if pct > 0:
        return f"\033[32m{pct:.3f}\033[0m"
    elif pct < 0:
        return f"\033[31m{pct:.3f}\033[0m"
    else:
        return f"{pct:.3f}"


def run_hf_backtest(
    limit=BACKTEST_CONFIG['data_limit'],
    symbol=BACKTEST_CONFIG['symbol'],
    timeframe=BACKTEST_CONFIG['candle_timeframe'],
    short_range=range(5, 21, 3),   # Short MA: 5-20 step 3
    long_range=range(25, 101, 10),  # Long MA: 25-100 step 10
    max_workers=12,
    use_adx_filter=False  # Set True for trend-filtered trades
):
    """
    Run grid search backtest over MA combinations (spot-optimized with fees/slippage).
    Returns:
        pd.DataFrame: Sorted results by return %
    """
    df = fetch_ohlcv(symbol, timeframe, limit)
    if df is None or len(df) < 90:
        print(
            f"Failed to fetch {limit} candles; got {len(df) if df is not None else 0}")
        return pd.DataFrame()
    print(
        f"Backtesting on {len(df)} {timeframe} candles ({df.index[0]} to {df.index[-1]})")
    combos = list(product(short_range, long_range))
    print(
        f"Testing {len(combos)} combos with {STRATEGY_CONFIG['taker_fee_pct']*100}% fees + {STRATEGY_CONFIG['slippage_pct']*100}% slippage...")
    results = []

    def worker(s, l):
        return backtest_one(df, s, l, use_adx_filter=use_adx_filter)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(worker, s, l) for s, l in combos]
        for future in futures:
            res = future.result()
            if res:
                results.append(res)
    out = pd.DataFrame(results).sort_values(
        'return_%', ascending=False).reset_index(drop=True)
    # Print benchmark with color
    if not out.empty:
        bh_colored = color_pct(out.iloc[0]['buy_hold_%'])
        print(
            f"Buy & Hold (no fees/slip): {bh_colored}% (over period)")
    return out


def print_colored_table(df):
    """
    Print the DataFrame with colored return_% and buy_hold_% columns.
    """
    # Print headers
    print(f"{'short':>5} {'long':>5} {'return_%':>10} {'trades':>7} {'winrate_%':>10} {'max_dd_%':>9} {'final_usdt':>11} {'buy_hold_%':>11}")
    # Print rows with colors
    for _, row in df.iterrows():
        ret_col = color_pct(row['return_%'])
        bh_col = color_pct(row['buy_hold_%'])
        print(f"{row['short']:>5} {row['long']:>5} {ret_col:>10} {row['trades']:>7} {row['winrate_%']:>10} {row['max_dd_%']:>9} {row['final_usdt']:>11} {bh_col:>11}")


def main():
    timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h',
                  '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']
    print("\nRunning spot-optimized backtest (with fees/slippage) across all timeframes...")
    for timeframe in timeframes:
        print(f"\n=== Testing timeframe: {timeframe} ===")
        # Set True to test ADX-filtered version
        bt = run_hf_backtest(timeframe=timeframe, use_adx_filter=False)
        if not bt.empty:
            print_colored_table(
                bt.head(BACKTEST_CONFIG['top_combos_to_display']))
            ret_colored = color_pct(bt.iloc[0]['return_%'])
            print(
                f"\nTop combo for {timeframe}: {bt.iloc[0]['short']}/{bt.iloc[0]['long']} → {ret_colored}% (net of fees/slippage)")


if __name__ == "__main__":
    main()
