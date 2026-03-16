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


def test_fetch_consensus_estimates_handles_none_info(monkeypatch):
    monkeypatch.setattr("engine.get_yf_ticker", lambda ticker, use_cache=False: EmptyConsensusTicker())

    result = fetch_consensus_estimates("MSFT", "FY2026 Q1")

    assert "error" not in result
    assert result["next_quarter"]["revenue_estimate"] == "N/A"
    assert result["next_quarter"]["eps_estimate"] == "N/A"
    assert result["analyst_coverage"]["num_analysts"] is None
    assert result["warning"] == "Yahoo Finance did not return analyst consensus data for this run."


def test_fetch_consensus_estimates_uses_partial_data_without_error(monkeypatch):
    monkeypatch.setattr("engine.get_yf_ticker", lambda ticker, use_cache=False: PartialConsensusTicker())

    result = fetch_consensus_estimates("MSFT", "FY2026 Q1")

    assert "error" not in result
    assert result["next_quarter"]["revenue_estimate"] == "$82.50B"
    assert result["next_quarter"]["eps_estimate"] == "$4.25"
    assert result["analyst_coverage"]["num_analysts"] == 41
    assert "warning" not in result


def test_fetch_consensus_estimates_handles_info_property_error(monkeypatch):
    monkeypatch.setattr("engine.get_yf_ticker", lambda ticker, use_cache=False: ExplodingInfoTicker())

    result = fetch_consensus_estimates("AAPL", "FY2026 Q1")

    assert "error" not in result
    assert result["next_quarter"]["revenue_estimate"] == "N/A"
    assert result["warning"] == "Yahoo Finance did not return analyst consensus data for this run."
