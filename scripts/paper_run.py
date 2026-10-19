#!/usr/bin/env python
"""Run ``ma_crossover_bot.py`` against a *simulated* USDT wallet.

Why this exists
---------------
``ma_crossover_bot.py`` only *simulates orders* in paper mode, but it still
reads the **real** exchange balance for its sizing / "low balance" checks. With
an empty account that means it silently refuses to trade (see the
``USDT 0.00`` lines in ``hf_bot.log``). There is no simulated wallet anywhere in
the project, so paper mode cannot actually be exercised.

This wrapper injects a virtual account (default 100 USDT) by patching the
module-level ``exchange.fetch_balance`` and ``place_market_order`` used by the
bot. The bot's own signal logic (``run_strategy`` / ``check_trailing_stop`` /
``main``) runs unmodified -- we only replace the account and the fills.

It is intentionally additive: no bot code or config schema is changed.

Usage (from the repo root)::

    MA_CANDLE_TIMEFRAME=1m MA_SHORT_WINDOW=20 MA_LONG_WINDOW=100 \
    MA_POSITION_SIZE_PCT=0.99 MA_MAX_TRADE_USD=100 MA_STOP_LOSS_PCT=0.03 \
    PAPER_TRADING=true uv run scripts/paper_run.py --usdt 100 --seconds 3600
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

# config.py lives at the repo root; allow running directly from scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

import config  # noqa: E402
import ma_crossover_bot as bot  # noqa: E402

# Slippage is not part of the main bot's LIVE_CONFIG, but the shared .env key
# exists and the real-time bots use it, so honour it here too.
SLIPPAGE_PCT = config.get_float("SLIPPAGE_PCT", 0.0005)
FEE_PCT = bot.LIVE_CONFIG["taker_fee_pct"]
SYMBOL = bot.LIVE_CONFIG["symbol"]


def make_wallet(usdt: float, btc: float = 0.0) -> dict:
    return {"usdt": usdt, "btc": btc, "fees_paid": 0.0}


def install_simulated_account(wallet: dict, trades: list[dict]) -> None:
    """Patch the bot's exchange balance + order fills to hit ``wallet``."""
    if not bot.LIVE_CONFIG["paper_trading"]:
        raise SystemExit("Refusing to run: PAPER_TRADING is not true.")

    def _sim_fetch_balance(*_args, **_kwargs):
        # Shape mirrors ccxt: ['total'][code] and [code]['free'/'total'].
        return {
            "total": {"USDT": wallet["usdt"], "BTC": wallet["btc"]},
            "free": {"USDT": wallet["usdt"], "BTC": wallet["btc"]},
            "USDT": {"free": wallet["usdt"], "total": wallet["usdt"]},
            "BTC": {"free": wallet["btc"], "total": wallet["btc"]},
        }

    real_place = bot.place_market_order

    def _sim_place(side: str, amount: float):
        ticker = bot.exchange.fetch_ticker(SYMBOL)
        price = ticker["last"]
        fill = (
            price * (1 + SLIPPAGE_PCT) if side == "buy" else price * (1 - SLIPPAGE_PCT)
        )
        fee = amount * fill * FEE_PCT

        record = {
            "ts": time.time(),
            "side": side,
            "amount": amount,
            "mark_price": price,
            "fill_price": fill,
            "fee": fee,
            "pnl": None,
        }
        if side == "buy":
            wallet["usdt"] -= amount * fill + fee
            wallet["btc"] += amount
        else:
            entry = bot.position["entry_price"]
            entry_fee = bot.position["entry_fee_usd"]
            gross = (fill - entry) * amount
            record["pnl"] = gross - entry_fee - fee
            wallet["btc"] -= amount
            wallet["usdt"] += amount * fill - fee
        wallet["fees_paid"] += fee
        record["equity"] = wallet["usdt"] + wallet["btc"] * price
        trades.append(record)
        # Delegate logging to the bot's own paper-order path.
        return real_place(side, amount)

    bot.exchange.fetch_balance = _sim_fetch_balance
    bot.place_market_order = _sim_place


def disable_mismatched_backtest_override() -> None:
    """Stop the 2h backtest from overriding the live (shorter) MA windows.

    ``bot.main()`` optimises on ``BT_CANDLE_TIMEFRAME`` (2h) and then applies the
    winner to the live timeframe. On a 1m/1s feed that is a known mismatch (the
    bot even warns about it). Returning an empty frame makes it keep the live
    windows from the environment instead.
    """

    def _no_backtest(*_args, **_kwargs):
        print(
            "[paper_run] startup backtest disabled (2h params applied to a "
            "short feed is a known mismatch); using env MA windows."
        )
        return pd.DataFrame()

    bot.backtest.run_hf_backtest = _no_backtest


def summarize(wallet: dict, trades: list[dict], start_usdt: float) -> dict:
    try:
        last_price = bot.exchange.fetch_ticker(SYMBOL)["last"]
    except Exception:
        last_price = trades[-1]["mark_price"] if trades else 0.0
    equity = wallet["usdt"] + wallet["btc"] * last_price
    closed = [t for t in trades if t["pnl"] is not None]
    wins = [t for t in closed if t["pnl"] > 0]
    summary = {
        "start_usdt": start_usdt,
        "final_usdt_cash": round(wallet["usdt"], 4),
        "final_btc": round(wallet["btc"], 8),
        "last_price": last_price,
        "final_equity": round(equity, 4),
        "pnl_usdt": round(equity - start_usdt, 4),
        "return_pct": round((equity / start_usdt - 1) * 100, 4),
        "orders": len(trades),
        "round_trips": len(closed),
        "wins": len(wins),
        "losses": len(closed) - len(wins),
        "fees_paid": round(wallet["fees_paid"], 4),
        "win_rate_pct": round(len(wins) / len(closed) * 100, 1) if closed else None,
    }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--usdt", type=float, default=100.0, help="starting paper USDT")
    ap.add_argument("--btc", type=float, default=0.0, help="starting paper BTC")
    ap.add_argument("--seconds", type=float, default=3600.0, help="run duration")
    ap.add_argument("--out", default="paper_run_result.json", help="summary file")
    args = ap.parse_args()

    wallet = make_wallet(args.usdt, args.btc)
    trades: list[dict] = []
    install_simulated_account(wallet, trades)
    disable_mismatched_backtest_override()

    print(
        f"[paper_run] simulated wallet: {args.usdt:.2f} USDT, {args.btc:.6f} BTC | "
        f"duration {args.seconds:.0f}s | slippage {SLIPPAGE_PCT * 100:.3f}% | "
        f"fee {FEE_PCT * 100:.3f}%"
    )

    def _stop(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGALRM, _stop)
    signal.setitimer(signal.ITIMER_REAL, args.seconds)
    try:
        bot.main()  # the real bot loop; our patches swap in the paper account
    except KeyboardInterrupt:
        pass
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)

    summary = summarize(wallet, trades, args.usdt)
    print("\n=== PAPER RUN SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    Path(args.out).write_text(
        json.dumps({"summary": summary, "trades": trades}, indent=2), encoding="utf-8"
    )
    print(f"\n[paper_run] wrote {args.out}")


if __name__ == "__main__":
    main()
