"""Shared yfinance helpers for process-local ticker reuse."""

from __future__ import annotations

from functools import lru_cache

import yfinance as yf


def normalize_yf_ticker(ticker: str) -> str:
    """Normalize ticker symbols before using them as cache keys."""
    return str(ticker or "").strip().upper()


@lru_cache(maxsize=128)
def _cached_ticker(normalized_ticker: str):
    return yf.Ticker(normalized_ticker)


def get_yf_ticker(ticker: str, *, use_cache: bool = True):
    """Return a yfinance Ticker, reusing instances for repeated lookups by default."""
    normalized_ticker = normalize_yf_ticker(ticker)
    if use_cache:
        return _cached_ticker(normalized_ticker)
    return yf.Ticker(normalized_ticker)


def clear_yf_ticker_cache() -> None:
    _cached_ticker.cache_clear()
