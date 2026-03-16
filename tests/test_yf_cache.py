from yf_cache import clear_yf_ticker_cache, get_yf_ticker, normalize_yf_ticker


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
