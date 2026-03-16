import pandas as pd

from data_adapter import DataAdapter
from engine import get_financial_data, get_financials
from yf_cache import clear_yf_ticker_cache, get_yf_fast_info, get_yf_frame, get_yf_info, get_yf_ticker, normalize_yf_ticker


class DummyTicker:
    def __init__(self, symbol: str):
        self.symbol = symbol


def test_normalize_yf_ticker():
    assert normalize_yf_ticker(" msft ") == "MSFT"
    assert normalize_yf_ticker("^tnx") == "^TNX"


def test_get_yf_ticker_reuses_cached_instances(monkeypatch):
    clear_yf_ticker_cache()
    calls = []

    def fake_ticker(symbol: str):
        calls.append(symbol)
        return DummyTicker(symbol)

    monkeypatch.setattr("yf_cache.yf.Ticker", fake_ticker)

    first = get_yf_ticker(" msft ")
    second = get_yf_ticker("MSFT")

    assert first is second
    assert first.symbol == "MSFT"
    assert calls == ["MSFT"]


def test_get_yf_ticker_can_bypass_cache(monkeypatch):
    clear_yf_ticker_cache()
    calls = []

    def fake_ticker(symbol: str):
        calls.append(symbol)
        return DummyTicker(symbol)

    monkeypatch.setattr("yf_cache.yf.Ticker", fake_ticker)

    first = get_yf_ticker("AAPL", use_cache=False)
    second = get_yf_ticker("AAPL", use_cache=False)

    assert first is not second
    assert calls == ["AAPL", "AAPL"]


def test_get_yf_info_handles_none_and_property_errors():
    class NoneInfoTicker:
        info = None

    class ErrorInfoTicker:
        @property
        def info(self):
            raise AttributeError("upstream broken")

    assert get_yf_info(NoneInfoTicker()) == {}
    assert get_yf_info(ErrorInfoTicker()) == {}


def test_get_yf_fast_info_converts_mapping_like_values():
    class MappingLike:
        def __iter__(self):
            yield ("lastPrice", 123.45)

    class FastInfoTicker:
        fast_info = MappingLike()

    assert get_yf_fast_info(FastInfoTicker()) == {"lastPrice": 123.45}


def test_get_yf_frame_handles_none_and_property_errors():
    expected = pd.DataFrame({"latest": [1]})

    class FrameTicker:
        quarterly_income_stmt = expected

    class NoneFrameTicker:
        quarterly_income_stmt = None

    class ErrorFrameTicker:
        @property
        def quarterly_income_stmt(self):
            raise RuntimeError("broken frame")

    assert get_yf_frame(FrameTicker(), "quarterly_income_stmt").equals(expected)
    assert get_yf_frame(NoneFrameTicker(), "quarterly_income_stmt").empty
    assert get_yf_frame(ErrorFrameTicker(), "quarterly_income_stmt").empty


def test_get_financials_bypasses_shared_ticker_cache(monkeypatch):
    calls = []

    class StatementTicker:
        income_stmt = pd.DataFrame({"latest": [1]})
        balance_sheet = pd.DataFrame({"latest": [1]})
        cashflow = pd.DataFrame({"latest": [1]})
        quarterly_cashflow = pd.DataFrame({"latest": [1]})

    def fake_get_yf_ticker(symbol: str, *, use_cache: bool = True):
        calls.append((symbol, use_cache))
        return StatementTicker()

    monkeypatch.setattr("engine.get_yf_ticker", fake_get_yf_ticker)

    income_stmt, balance_sheet, cash_flow, quarterly_cash_flow = get_financials("MSFT")

    assert not income_stmt.empty
    assert not balance_sheet.empty
    assert not cash_flow.empty
    assert not quarterly_cash_flow.empty
    assert calls == [("MSFT", False)]


def test_get_financial_data_bypasses_shared_ticker_cache_without_fmp(monkeypatch):
    calls = []

    class StatementTicker:
        quarterly_income_stmt = pd.DataFrame({"latest": [1]})
        quarterly_balance_sheet = pd.DataFrame({"latest": [1]})
        quarterly_cashflow = pd.DataFrame({"latest": [1]})

    def fake_get_yf_ticker(symbol: str, *, use_cache: bool = True):
        calls.append((symbol, use_cache))
        return StatementTicker()

    monkeypatch.setattr("engine.get_yf_ticker", fake_get_yf_ticker)

    income_stmt, balance_sheet, cash_flow, quarterly_cash_flow, data_source, warning = get_financial_data("MSFT", None)

    assert not income_stmt.empty
    assert not balance_sheet.empty
    assert not cash_flow.empty
    assert not quarterly_cash_flow.empty
    assert data_source == "Yahoo Finance"
    assert "yfinance" in warning
    assert calls == [("MSFT", False)]


def test_data_adapter_fetch_bypasses_shared_ticker_cache(monkeypatch):
    calls = []

    class StatementTicker:
        pass

    def fake_get_yf_ticker(symbol: str, *, use_cache: bool = True):
        calls.append((symbol, use_cache))
        return StatementTicker()

    monkeypatch.setattr("data_adapter.get_yf_ticker", fake_get_yf_ticker)
    monkeypatch.setattr(DataAdapter, "_fetch_price_and_shares", lambda self, stock: None)
    monkeypatch.setattr(DataAdapter, "_fetch_balance_sheet", lambda self, stock: None)
    monkeypatch.setattr(DataAdapter, "_fetch_cash_flow", lambda self, stock: None)
    monkeypatch.setattr(DataAdapter, "_fetch_income_statement", lambda self, stock: None)
    monkeypatch.setattr(DataAdapter, "_fetch_quarterly_history", lambda self, stock: None)
    monkeypatch.setattr(DataAdapter, "_fetch_analyst_revenue_estimates", lambda self, stock: None)
    monkeypatch.setattr(DataAdapter, "_calculate_suggested_assumptions", lambda self: None)

    snapshot = DataAdapter(" msft ").fetch()

    assert snapshot.ticker == " MSFT ".strip().upper()
    assert calls == [("MSFT", False)]
