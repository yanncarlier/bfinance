#!/usr/bin/env python
"""Paper-trading *replay* of ``ma_crossover_bot.py`` over historical candles.

The live bot only ever sees "now", so a short live session cannot demonstrate a
multi-week trend strategy. This replays the bot's own ``run_strategy`` /
``check_trailing_stop`` over a longer historical window against a simulated
wallet (via ``paper_run``), so we can see whether the configured strategy makes
money in paper mode over a longer period.

Usage (repo root)::

    MA_CANDLE_TIMEFRAME=4h MA_SHORT_WINDOW=20 MA_LONG_WINDOW=100 \
    MA_STOP_LOSS_PCT=0.15 MA_POSITION_SIZE_PCT=0.99 MA_MAX_TRADE_USD=100 \
    PAPER_TRADING=true uv run scripts/paper_replay.py --since 2024-06-17
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ma_crossover_bot as bot  # noqa: E402
import paper_run  # noqa: E402


def fetch_history(symbol: str, tf: str, since_iso: str) -> pd.DataFrame:
    since = int(pd.Timestamp(since_iso, tz="UTC").timestamp() * 1000)
    rows: list = []
    while True:
        batch = bot.exchange.fetch_ohlcv(symbol, tf, since=since, limit=1000)
        if not batch:
            break
        rows += batch
        since = batch[-1][0] + 1
        if len(batch) < 1000:
            break
    df = pd.DataFrame(
        rows, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("timestamp")
    return df[~df.index.duplicated(keep="last")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--since", default="2024-06-17")
    ap.add_argument("--usdt", type=float, default=100.0)
    ap.add_argument("--out", default="paper_replay_result.json")
    args = ap.parse_args()

    if not bot.LIVE_CONFIG["paper_trading"]:
        raise SystemExit("Refusing to run: PAPER_TRADING is not true.")

    tf = bot.LIVE_CONFIG["candle_timeframe"]
    symbol = bot.LIVE_CONFIG["symbol"]
    df = fetch_history(symbol, tf, args.since)
    print(
        f"[replay] {len(df)} {tf} candles {df.index[0]} -> {df.index[-1]} "
        f"| MA {bot.LIVE_CONFIG['short_window']}/{bot.LIVE_CONFIG['long_window']} "
        f"| trail {bot.LIVE_CONFIG['stop_loss_pct'] * 100:.0f}%"
    )

    wallet = paper_run.make_wallet(args.usdt)
    trades: list[dict] = []
    paper_run.install_simulated_account(wallet, trades)
    bot.BALANCE_CACHE_REFRESH_SEC = 0  # refresh every call during replay

    state = {"i": 0}

    def replay_fetch(_symbol, _timeframe, limit, _retries=3):
        end = state["i"] + 1
        return df.iloc[max(0, end - int(limit)) : end].copy()

    bot.fetch_ohlcv = replay_fetch
    bot.exchange.fetch_ticker = lambda _s: {"last": float(df["close"].iloc[state["i"]])}

    equity_curve = []
    start_i = bot.LIVE_CONFIG["long_window"] + 1
    for i in range(start_i, len(df)):
        state["i"] = i
        bot.run_strategy()
        equity_curve.append(wallet["usdt"] + wallet["btc"] * float(df["close"].iloc[i]))

    last = float(df["close"].iloc[-1])
    equity = wallet["usdt"] + wallet["btc"] * last
    eq = pd.Series(equity_curve, index=df.index[start_i:])
    dd = ((eq.cummax() - eq) / eq.cummax()).max()
    closed = [t for t in trades if t["pnl"] is not None]
    wins = [t for t in closed if t["pnl"] > 0]
    bh = args.usdt * last / float(df["close"].iloc[start_i])

    summary = {
        "period": f"{df.index[start_i]} -> {df.index[-1]}",
        "start_usdt": args.usdt,
        "final_equity": round(equity, 4),
        "return_pct": round((equity / args.usdt - 1) * 100, 2),
        "buy_hold_return_pct": round((bh / args.usdt - 1) * 100, 2),
        "orders": len(trades),
        "round_trips": len(closed),
        "win_rate_pct": round(len(wins) / len(closed) * 100, 1) if closed else None,
        "max_dd_pct": round(dd * 100, 2),
        "fees_paid": round(wallet["fees_paid"], 4),
        "still_holding": bool(wallet["btc"] > 0),
    }
    print("\n=== PAPER REPLAY SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    pd.DataFrame({"equity": eq}).to_csv(Path(args.out).with_suffix(".equity.csv"))
    Path(args.out).write_text(
        json.dumps({"summary": summary, "trades": trades}, indent=2), encoding="utf-8"
    )
    print(f"\n[replay] wrote {args.out}")


if __name__ == "__main__":
    main()
