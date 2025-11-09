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
    'candle_timeframe': '1s',      # Candle timeframe
    'symbol': 'BTC/USDT',          # Trading pair
    'init_usdt': 10000.0,          # Initial USDT balance for backtesting
    'data_limit': 1000,
    'top_combos_to_display': 10,   # Number of top combinations to print
}
# =============================================================================
# STRATEGY CONFIGURATION (Shared with Live Trading; Override as Needed)
# =============================================================================
# These are backtest defaults but can be overridden (e.g., from LIVE_CONFIG if imported)
STRATEGY_CONFIG = {
    'position_size_pct': 0.05,     # Risk percentage of balance per trade
    'max_trade_usd': 1000.0,       # Maximum USD to risk per trade
    'stop_loss_pct': 0.02,         # Trailing stop-loss percentage
}
# =============================================================================
# CORE SETUP (Not specific to backtest)
# =============================================================================
# Initialize Binance exchange via CCXT
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
        # FIX: Use keyword arg for limit to avoid 'since' misinterpretation
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
    Calculate ADX for trend strength ( >25 = trending).
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
                 init_usdt=BACKTEST_CONFIG['init_usdt'], pos_pct=STRATEGY_CONFIG['position_size_pct'],
                 max_usd=STRATEGY_CONFIG['max_trade_usd'], trail_pct=STRATEGY_CONFIG['stop_loss_pct'],
                 use_adx_filter=False):
    """
    Backtest a single MA combination on historical data.
    Args:
        ... (same as before)
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
        adx_ok = not use_adx_filter or adx[i] > 25  # FIX: Optional filter
        if btc > 0:
            if price > peak:
                peak = price
            stop = peak * (1 - trail_pct)
            if price <= stop:
                cash += btc * price
                trades.append({'pnl': btc * (price - entry)})
                btc = entry = peak = 0.0
                continue
            if short_ma[i-1] >= long_ma[i-1] and short_ma[i] < long_ma[i]:
                cash += btc * price
                trades.append({'pnl': btc * (price - entry)})
                btc = entry = peak = 0.0
                continue
        if (btc == 0 and
            short_ma[i-1] <= long_ma[i-1] and short_ma[i] > long_ma[i] and
                adx_ok):
            usd = min(cash * pos_pct, max_usd)
            btc = round(usd / price, 6)
            cash -= btc * price
            entry = peak = price
    if btc > 0:
        cash += btc * close[-1]
        trades.append({'pnl': btc * (close[-1] - entry)})
    pnl = cash - init_usdt
    ret = pnl / init_usdt * 100
    wins = sum(1 for t in trades if t['pnl'] > 0)
    winrate = wins / len(trades) * 100 if trades else 0
    equity = np.cumsum([init_usdt] + [t['pnl'] for t in trades])
    dd = np.maximum.accumulate(equity) - equity
    max_dd = dd.max() / init_usdt * 100 if equity.size else 0
    # FIX: Add buy-hold benchmark
    buy_hold_ret = (close[-1] / close[0] - 1) * 100
    return {
        # FIX: Ensure int from source
        'short': int(short_win), 'long': int(long_win),
        'return_%': round(ret, 3), 'trades': len(trades),
        'winrate_%': round(winrate, 1), 'max_dd_%': round(max_dd, 2),
        'final_usdt': round(cash, 2),
        'buy_hold_%': round(buy_hold_ret, 3)  # New: Benchmark
    }
def run_hf_backtest(
    limit=BACKTEST_CONFIG['data_limit'],
    symbol=BACKTEST_CONFIG['symbol'],
    timeframe=BACKTEST_CONFIG['candle_timeframe'],
    short_range=range(5, 21, 3),   # FIX: Wider, less aggressive (was 3-16/2)
    long_range=range(25, 101, 10),  # FIX: Sparser for quality (was 20-81/5)
    max_workers=12,
    use_adx_filter=False  # FIX: New param; set True for trend-only trades
):
    """
    Run grid search backtest over MA combinations.
    ... (same args)
    Returns:
        pd.DataFrame: Sorted results
    """
    df = fetch_ohlcv(symbol, timeframe, limit)
    if df is None or len(df) < 90:
        print(
            f"Failed to fetch {limit} candles; got {len(df) if df is not None else 0}")
        return pd.DataFrame()
    print(
        f"Backtesting on {len(df)} candles ({df.index[0]} to {df.index[-1]})")
    combos = list(product(short_range, long_range))
    print(f"Testing {len(combos)} combos...")
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
    # FIX: Print benchmark
    if not out.empty:
        print(f"Buy & Hold: {out.iloc[0]['buy_hold_%']:.3f}% (over period)")
    return out
def main():
    print("\nRunning backtest...")
    # Set True to test filtered version
    bt = run_hf_backtest(use_adx_filter=False)
    if not bt.empty:
        print(bt.head(BACKTEST_CONFIG['top_combos_to_display']).to_string(
            index=False))
        print(
            f"\nTop combo: {bt.iloc[0]['short']}/{bt.iloc[0]['long']} → {bt.iloc[0]['return_%']}%")
if __name__ == "__main__":
    main()
