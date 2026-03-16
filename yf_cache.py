"""Shared yfinance helpers for process-local ticker reuse."""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache

import pandas as pd
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


def _get_yf_attr(stock, attr_name: str):
    try:
        return getattr(stock, attr_name)
    except Exception:
        return None


def get_yf_mapping(stock, attr_name: str) -> dict:
    """Safely read a mapping-like yfinance attribute such as info/fast_info."""
    raw_value = _get_yf_attr(stock, attr_name)
    if raw_value is None:
        return {}
    if isinstance(raw_value, Mapping):
        try:
            return dict(raw_value)
        except Exception:
            return {}
    try:
        return dict(raw_value)
    except Exception:
        return {}


def get_yf_info(stock) -> dict:
    """Safely read stock.info as a plain dict."""
    return get_yf_mapping(stock, "info")


def get_yf_fast_info(stock) -> dict:
    """Safely read stock.fast_info as a plain dict."""
    return get_yf_mapping(stock, "fast_info")


def get_yf_frame(stock, attr_name: str) -> pd.DataFrame:
    """Safely read a yfinance DataFrame attribute, returning an empty frame on failure."""
    raw_value = _get_yf_attr(stock, attr_name)
    return raw_value if isinstance(raw_value, pd.DataFrame) else pd.DataFrame()


def clear_yf_ticker_cache() -> None:
    _cached_ticker.cache_clear()
