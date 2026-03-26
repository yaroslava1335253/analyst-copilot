import pandas as pd

from engine import fetch_consensus_estimates


class EmptyConsensusTicker:
    info = None
    earnings_estimate = None
    revenue_estimate = None
    recommendations_summary = None


class PartialConsensusTicker:
    info = {
        "targetMeanPrice": 512.4,
        "numberOfAnalystOpinions": None,
        "recommendationKey": None,
    }
    earnings_estimate = pd.DataFrame(
        {"avg": [4.25]},
        index=["0q"],
    )
    revenue_estimate = pd.DataFrame(
        {"avg": [82_500_000_000], "numberOfAnalysts": [41]},
        index=["0q"],
    )
    recommendations_summary = None


class ExplodingInfoTicker:
    @property
    def info(self):
        raise AttributeError("'NoneType' object has no attribute 'get'")

    earnings_estimate = None
    revenue_estimate = None
    recommendations_summary = None


class YahooQueryFallbackTicker:
    def __init__(self, symbol: str):
        self.symbol = symbol

    @property
    def financial_data(self):
        return {
            self.symbol: {
                "targetMeanPrice": 594.62,
                "targetHighPrice": 650.0,
                "targetLowPrice": 500.0,
                "numberOfAnalystOpinions": 57,
            }
        }

    @property
    def earnings_trend(self):
        return {
            self.symbol: {
                "trend": [
                    {
                        "period": "0q",
                        "revenueEstimate": {"avg": 81_360_000_000},
                        "earningsEstimate": {"avg": 4.09},
                    },
                    {
                        "period": "0y",
                        "revenueEstimate": {"avg": 318_000_000_000},
                        "earningsEstimate": {"avg": 15.30},
                    },
                ]
            }
        }

    @property
    def calendar_events(self):
        return {self.symbol: {"earnings": {}}}

    @property
    def recommendation_trend(self):
        return pd.DataFrame(
            {
                "period": ["0m"],
                "strongBuy": [18],
                "buy": [21],
                "hold": [15],
                "sell": [3],
                "strongSell": [0],
            },
            index=pd.MultiIndex.from_tuples([(self.symbol, 0)], names=["symbol", "row"]),
        )


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class FakeErrorResponse(FakeResponse):
    def raise_for_status(self):
        return None


def test_fetch_consensus_estimates_handles_none_info(monkeypatch):
    monkeypatch.delenv("FMP_API_KEY", raising=False)
    monkeypatch.setattr("engine.get_yf_ticker", lambda ticker, use_cache=False: EmptyConsensusTicker())
    monkeypatch.setattr("engine.YQTicker", None)

    result = fetch_consensus_estimates("MSFT", "FY2026 Q1")

    assert "error" not in result
    assert result["next_quarter"]["revenue_estimate"] == "N/A"
    assert result["next_quarter"]["eps_estimate"] == "N/A"
    assert result["analyst_coverage"]["num_analysts"] is None
    assert result["warning"] == "Yahoo Finance did not return analyst consensus data for this run."


def test_fetch_consensus_estimates_uses_partial_data_without_error(monkeypatch):
    monkeypatch.delenv("FMP_API_KEY", raising=False)
    monkeypatch.setattr("engine.get_yf_ticker", lambda ticker, use_cache=False: PartialConsensusTicker())
    monkeypatch.setattr("engine.YQTicker", None)

    result = fetch_consensus_estimates("MSFT", "FY2026 Q1")

    assert "error" not in result
    assert result["next_quarter"]["revenue_estimate"] == "$82.50B"
    assert result["next_quarter"]["eps_estimate"] == "$4.25"
    assert result["analyst_coverage"]["num_analysts"] == 41
    assert "warning" not in result


def test_fetch_consensus_estimates_handles_info_property_error(monkeypatch):
    monkeypatch.delenv("FMP_API_KEY", raising=False)
    monkeypatch.setattr("engine.get_yf_ticker", lambda ticker, use_cache=False: ExplodingInfoTicker())
    monkeypatch.setattr("engine.YQTicker", None)

    result = fetch_consensus_estimates("AAPL", "FY2026 Q1")

    assert "error" not in result
    assert result["next_quarter"]["revenue_estimate"] == "N/A"
    assert result["warning"] == "Yahoo Finance did not return analyst consensus data for this run."


def test_fetch_consensus_estimates_falls_back_to_yahooquery(monkeypatch):
    monkeypatch.delenv("FMP_API_KEY", raising=False)
    monkeypatch.setattr("engine.get_yf_ticker", lambda ticker, use_cache=False: EmptyConsensusTicker())
    monkeypatch.setattr("engine.YQTicker", YahooQueryFallbackTicker)

    result = fetch_consensus_estimates("MSFT", "FY2026 Q1")

    assert "error" not in result
    assert result["next_quarter"]["revenue_estimate"] == "$81.36B"
    assert result["next_quarter"]["eps_estimate"] == "$4.09"
    assert result["full_year"]["revenue_estimate"] == "$318.00B"
    assert result["price_targets"]["average"] == "$594.62"
    assert result["analyst_coverage"]["num_analysts"] == 57
    assert result["source"] == "Yahoo Finance (yahooquery)"
    assert "warning" not in result


def test_fetch_consensus_estimates_uses_fmp_when_available(monkeypatch):
    def fake_requests_get(url, timeout=10):
        if "analyst-estimates" in url and "period=quarter" in url:
            return FakeResponse(
                [
                    {
                        "date": "2026-03-31",
                        "estimatedRevenueAvg": 81_360_000_000,
                        "estimatedEpsAvg": 4.09,
                        "numberAnalystsEstimatedRevenue": 57,
                    }
                ]
            )
        if "analyst-estimates" in url and "period=annual" in url:
            return FakeResponse(
                [
                    {
                        "date": "2026-06-30",
                        "estimatedRevenueAvg": 318_000_000_000,
                        "estimatedEpsAvg": 15.30,
                    }
                ]
            )
        if "price-target-consensus" in url:
            return FakeResponse(
                [
                    {
                        "targetConsensus": 594.62,
                        "targetHigh": 650.00,
                        "targetLow": 500.00,
                        "analystCount": 57,
                    }
                ]
            )
        if "grades-consensus" in url:
            return FakeResponse(
                [
                    {
                        "strongBuy": 18,
                        "buy": 21,
                        "hold": 15,
                        "sell": 3,
                        "strongSell": 0,
                    }
                ]
            )
        raise AssertionError(url)

    monkeypatch.setattr("engine.get_yf_ticker", lambda ticker, use_cache=False: EmptyConsensusTicker())
    monkeypatch.setattr("engine.requests.get", fake_requests_get)
    monkeypatch.setattr("engine.YQTicker", None)

    result = fetch_consensus_estimates("MSFT", "FY2026 Q1", fmp_api_key="test-key")

    assert "error" not in result
    assert result["next_quarter"]["revenue_estimate"] == "$81.36B"
    assert result["next_quarter"]["eps_estimate"] == "$4.09"
    assert result["full_year"]["revenue_estimate"] == "$318.00B"
    assert result["price_targets"]["average"] == "$594.62"
    assert result["analyst_coverage"]["num_analysts"] == 57
    assert result["analyst_coverage"]["buy_ratings"] == 39
    assert result["source"].startswith("Financial Modeling Prep")
    assert "warning" not in result


def test_fetch_consensus_estimates_keeps_fmp_targets_when_estimates_unavailable(monkeypatch):
    def fake_requests_get(url, timeout=10):
        if "analyst-estimates" in url:
            return FakeErrorResponse(
                {
                    "Error Message": "Premium Query Parameter: 'Special Endpoint : This value set for 'period' is not available under your current subscription",
                }
            )
        if "price-target-consensus" in url:
            return FakeResponse(
                [
                    {
                        "targetConsensus": 583.67,
                        "targetHigh": 675.00,
                        "targetLow": 392.00,
                    }
                ]
            )
        if "grades-consensus" in url:
            return FakeResponse(
                [
                    {
                        "strongBuy": 0,
                        "buy": 62,
                        "hold": 16,
                        "sell": 0,
                        "strongSell": 0,
                    }
                ]
            )
        raise AssertionError(url)

    monkeypatch.setattr("engine.get_yf_ticker", lambda ticker, use_cache=False: EmptyConsensusTicker())
    monkeypatch.setattr("engine.requests.get", fake_requests_get)
    monkeypatch.setattr("engine.YQTicker", None)

    result = fetch_consensus_estimates("MSFT", "FY2026 Q1", fmp_api_key="test-key")

    assert "error" not in result
    assert result["next_quarter"]["revenue_estimate"] == "N/A"
    assert result["next_quarter"]["eps_estimate"] == "N/A"
    assert result["price_targets"]["average"] == "$583.67"
    assert result["analyst_coverage"]["num_analysts"] == 78
    assert result["analyst_coverage"]["buy_ratings"] == 62
    assert result["analyst_coverage"]["hold_ratings"] == 16
    assert result["source"].startswith("Financial Modeling Prep")
    assert "warning" not in result


def test_fetch_consensus_estimates_preserves_per_section_source_attribution(monkeypatch):
    def fake_requests_get(url, timeout=10):
        if "analyst-estimates" in url:
            return FakeErrorResponse(
                {
                    "Error Message": "Premium Query Parameter: 'Special Endpoint : This value set for 'period' is not available under your current subscription",
                }
            )
        if "price-target-consensus" in url:
            return FakeResponse(
                [
                    {
                        "targetConsensus": 583.67,
                        "targetHigh": 675.00,
                        "targetLow": 392.00,
                    }
                ]
            )
        if "grades-consensus" in url:
            return FakeResponse(
                [
                    {
                        "strongBuy": 0,
                        "buy": 62,
                        "hold": 16,
                        "sell": 0,
                        "strongSell": 0,
                    }
                ]
            )
        raise AssertionError(url)

    monkeypatch.setattr("engine.get_yf_ticker", lambda ticker, use_cache=False: PartialConsensusTicker())
    monkeypatch.setattr("engine.requests.get", fake_requests_get)
    monkeypatch.setattr("engine.YQTicker", None)

    result = fetch_consensus_estimates("MSFT", "FY2026 Q1", fmp_api_key="test-key")

    assert result["source"] == "Financial Modeling Prep + Yahoo Finance (yfinance)"
    assert result["next_quarter"]["source"] == "Yahoo Finance (yfinance)"
    assert result["price_targets"]["source"] == "Financial Modeling Prep"
