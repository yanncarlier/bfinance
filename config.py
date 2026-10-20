"""Central configuration sourced from environment variables.

Every value that used to be hardcoded in the scripts is now read from the
process environment, falling back to the original hardcoded value as the
default. Copy ``.env.example`` to ``.env`` and edit it to override anything.

Variable naming:

* Shared / account-level settings use no prefix (for example ``SYMBOL``,
  ``TAKER_FEE_PCT``, ``PAPER_TRADING``).
* Settings that are identical across the backtest scripts are shared too
  (``INIT_USDT``, ``POSITION_SIZE_PCT``, ``MAX_TRADE_USD``,
  ``TAKE_PROFIT_PCT``, ``TRAILING_STOP_PCT``, ``TOP_COMBOS_TO_DISPLAY``).
* Everything else is prefixed with the script it belongs to:
  ``MA_`` (ma_crossover_bot.py), ``RT_`` / ``RT_TB_`` (the real-time bots in
  ``scripts/``), ``BT_`` (backtest.py), ``BT_TB_`` / ``BT_CT_`` / ``BT_CT_TB_``
  (the backtest variants in ``scripts/``).
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

# Load .env once, at import time, before any getter is called.
load_dotenv()

_TRUE_VALUES = {'1', 'true', 'yes', 'on'}
_FALSE_VALUES = {'0', 'false', 'no', 'off'}


def _raw(name: str) -> str | None:
    """Return the raw env value for ``name``, or None if unset/blank."""
    value = os.getenv(name)
    if value is None or value.strip() == '':
        return None
    return value


def is_set(name: str) -> bool:
    """Return True if env var ``name`` is present and non-blank."""
    return _raw(name) is not None


def get_str(name: str, default: str) -> str:
    """Return env var ``name`` as a string, or ``default`` if unset/blank."""
    value = _raw(name)
    return default if value is None else value


def get_int(name: str, default: int) -> int:
    """Return env var ``name`` as an int, or ``default`` if unset/blank."""
    value = _raw(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(
            f"Environment variable {name} must be an integer, got {value!r}"
        ) from exc


def get_float(name: str, default: float) -> float:
    """Return env var ``name`` as a float, or ``default`` if unset/blank."""
    value = _raw(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(
            f"Environment variable {name} must be a float, got {value!r}"
        ) from exc


def get_bool(name: str, default: bool) -> bool:
    """Return env var ``name`` as a bool, or ``default`` if unset/blank."""
    value = _raw(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError(
        f"Environment variable {name} must be a boolean, got {value!r}"
    )


def get_api_credentials() -> tuple[str, str]:
    """Return ``(API_KEY, API_SECRET)``, raising if either is unset."""
    api_key = os.getenv('API_KEY')
    api_secret = os.getenv('API_SECRET')
    if not api_key or not api_secret:
        raise EnvironmentError("API_KEY and API_SECRET must be in .env file")
    return api_key, api_secret
