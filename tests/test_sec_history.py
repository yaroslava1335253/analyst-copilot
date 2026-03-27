import pytest
import pandas as pd

from engine import (
    _filter_source_diagnostics_for_window,
    _get_quarterly_income_history,
    _get_quarterly_income_history_sec,
    _merge_yahoo_sec_quarterly_income,
    analyze_quarterly_trends,
    _sec_backfill_annual_end_quarter_from_annual,
)


def _quarterly_income_df(metric_values: dict[str, dict[str, float]]) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            pd.Timestamp(date): {
                metric: value
                for metric, metric_series in metric_values.items()
                if date in metric_series
                for value in [metric_series[date]]
            }
            for date in sorted(
                {
                    date
                    for metric_series in metric_values.values()
                    for date in metric_series
                },
                reverse=True,
            )
        }
    )
    return frame


@pytest.mark.parametrize(
    ("quarterly", "annual", "expected_date", "expected_bucket_label", "expected_value"),
    [
        (
            {
                "2024-09-30": 58_000_000_000,
                "2024-06-30": 57_000_000_000,
                "2024-03-31": 55_000_000_000,
            },
            {"2024-12-31": 230_000_000_000},
            "2024-12-31",
            "2024-Q4",
            60_000_000_000,
        ),
        (
            {
                "2024-06-30": 58_000_000_000,
                "2024-03-31": 57_000_000_000,
                "2023-12-31": 55_000_000_000,
            },
            {"2024-09-30": 230_000_000_000},
            "2024-09-30",
            "2024-Q3",
            60_000_000_000,
        ),
        (
            {
                "2024-03-31": 61_858_000_000,
                "2023-12-31": 62_020_000_000,
                "2023-09-30": 56_517_000_000,
            },
            {"2024-06-30": 245_122_000_000},
            "2024-06-30",
            "2024-Q2",
            64_727_000_000,
        ),
        (
            {
                "2023-12-31": 58_000_000_000,
                "2023-09-30": 57_000_000_000,
                "2023-06-30": 55_000_000_000,
            },
            {"2024-03-31": 230_000_000_000},
            "2024-03-31",
            "2024-Q1",
            60_000_000_000,
        ),
        (
            {
                "2024-11-02": 58_000_000_000,
                "2024-08-03": 57_000_000_000,
                "2024-05-04": 55_000_000_000,
            },
            {"2025-02-01": 230_000_000_000},
            "2025-02-01",
            "2025-Q1",
            60_000_000_000,
        ),
    ],
)
def test_sec_backfill_derives_annual_end_quarter_for_multiple_fiscal_calendars(
    quarterly,
    annual,
    expected_date,
    expected_bucket_label,
    expected_value,
):
    series, derived = _sec_backfill_annual_end_quarter_from_annual(quarterly, annual)

    assert series[expected_date] == expected_value
    assert derived == [
        {
            "quarter": expected_bucket_label,
            "date": expected_date,
            "value": float(expected_value),
        }
    ]


def test_sec_backfill_requires_prior_three_quarters():
    quarterly = {
        "2024-03-31": 61_858_000_000,
        "2023-09-30": 56_517_000_000,
    }
    annual = {
        "2024-06-30": 245_122_000_000,
    }

    series, derived = _sec_backfill_annual_end_quarter_from_annual(quarterly, annual)

    assert "2024-06-30" not in series
    assert derived == []


def test_get_quarterly_income_history_sec_backfills_msft_style_annual_end(monkeypatch):
    monkeypatch.setattr("engine._load_sec_ticker_cik_map", lambda: {"MSFT": 789019})
    monkeypatch.setattr("engine._sec_get_json", lambda url, timeout=15: {"facts": {}})

    def fake_sec_extract_series(company_facts, concepts, unit_kind="USD", strict_duration=True, period="quarterly"):
        if period == "quarterly":
            if "Revenues" in concepts:
                return {
                    "2024-03-31": 61_858_000_000,
                    "2023-12-31": 62_020_000_000,
                    "2023-09-30": 56_517_000_000,
                }
            return {}

        if "Revenues" in concepts:
            return {"2024-06-30": 245_122_000_000}
        return {}

    monkeypatch.setattr("engine._sec_extract_series", fake_sec_extract_series)

    df = _get_quarterly_income_history_sec("MSFT", max_quarters=10)

    assert isinstance(df, pd.DataFrame)
    assert pd.Timestamp("2024-06-30") in df.columns
    assert df.loc["Total Revenue", pd.Timestamp("2024-06-30")] == 64_727_000_000
    assert df.attrs["sec_annual_end_backfilled"]["total_annual_end_derived"] == 1
    assert df.attrs["sec_annual_end_backfilled"]["quarters"] == ["2024-Q2"]


def test_merge_yahoo_sec_extends_only_after_clean_overlap_validation():
    yahoo_df = _quarterly_income_df(
        {
            "Total Revenue": {
                "2025-12-31": 81_273_000_000,
                "2025-09-30": 77_673_000_000,
                "2025-06-30": 76_441_000_000,
            }
        }
    )
    sec_df = _quarterly_income_df(
        {
            "Total Revenue": {
                "2025-12-31": 81_273_000_000,
                "2025-09-30": 77_673_000_000,
                "2025-06-30": 76_441_000_000,
                "2025-03-31": 70_066_000_000,
                "2024-12-31": 69_632_000_000,
            }
        }
    )

    merged_df, diagnostics = _merge_yahoo_sec_quarterly_income(yahoo_df, sec_df, max_quarters=10)

    assert [str(col)[:10] for col in merged_df.columns] == [
        "2025-12-31",
        "2025-09-30",
        "2025-06-30",
        "2025-03-31",
        "2024-12-31",
    ]
    assert merged_df.loc["Total Revenue", pd.Timestamp("2025-03-31")] == 70_066_000_000
    assert diagnostics["sec_overlap_points"] == 3
    assert diagnostics["sec_validation_passed"] is True
    assert diagnostics["sec_extension_applied"] is True
    assert diagnostics["mismatch_points"] == 0


def test_merge_yahoo_sec_keeps_yahoo_window_when_overlap_mismatches():
    yahoo_df = _quarterly_income_df(
        {
            "Total Revenue": {
                "2025-12-31": 81_273_000_000,
                "2025-09-30": 77_673_000_000,
                "2025-06-30": 76_441_000_000,
            }
        }
    )
    sec_df = _quarterly_income_df(
        {
            "Total Revenue": {
                "2025-12-31": 81_273_000_000,
                "2025-09-30": 70_000_000_000,
                "2025-06-30": 76_441_000_000,
                "2025-03-31": 70_066_000_000,
                "2024-12-31": 69_632_000_000,
            }
        }
    )

    merged_df, diagnostics = _merge_yahoo_sec_quarterly_income(yahoo_df, sec_df, max_quarters=10)

    assert [str(col)[:10] for col in merged_df.columns] == [
        "2025-12-31",
        "2025-09-30",
        "2025-06-30",
    ]
    assert pd.Timestamp("2025-03-31") not in merged_df.columns
    assert diagnostics["sec_overlap_points"] == 3
    assert diagnostics["sec_validation_passed"] is False
    assert diagnostics["sec_extension_applied"] is False
    assert diagnostics["mismatch_points"] == 1
    assert diagnostics["mismatch_samples"][0]["quarter"] == "2025-Q3"


def test_get_quarterly_income_history_labels_cross_check_without_extension(monkeypatch):
    yahoo_df = _quarterly_income_df(
        {
            "Total Revenue": {
                "2025-12-31": 81_273_000_000,
                "2025-09-30": 77_673_000_000,
            }
        }
    )
    sec_df = _quarterly_income_df(
        {
            "Total Revenue": {
                "2025-12-31": 81_273_000_000,
                "2025-09-30": 77_673_000_000,
            }
        }
    )
    sec_df.attrs["sec_annual_end_backfilled"] = {
        "total_annual_end_derived": 1,
        "quarters": ["2024-Q2"],
    }

    monkeypatch.setattr("engine._get_quarterly_income_history_sec", lambda ticker_symbol, max_quarters=20: sec_df)
    monkeypatch.setattr("engine.get_yf_ticker", lambda ticker_symbol, use_cache=False: object())
    monkeypatch.setattr("engine.get_yf_frame", lambda stock, field: yahoo_df)

    merged_df, source, diagnostics = _get_quarterly_income_history("MSFT", max_quarters=10)

    assert not merged_df.empty
    assert source == "Yahoo Finance (SEC cross-check)"
    assert diagnostics["sec_validation_passed"] is True
    assert diagnostics["sec_extension_applied"] is False
    assert "sec_annual_end_backfilled" not in diagnostics


def test_get_quarterly_income_history_labels_validated_extension(monkeypatch):
    yahoo_df = _quarterly_income_df(
        {
            "Total Revenue": {
                "2025-12-31": 81_273_000_000,
                "2025-09-30": 77_673_000_000,
            }
        }
    )
    sec_df = _quarterly_income_df(
        {
            "Total Revenue": {
                "2025-12-31": 81_273_000_000,
                "2025-09-30": 77_673_000_000,
                "2025-06-30": 76_441_000_000,
            }
        }
    )
    sec_df.attrs["sec_annual_end_backfilled"] = {
        "total_annual_end_derived": 3,
        "quarters": ["2025-Q2", "2024-Q2", "2023-Q2"],
    }

    monkeypatch.setattr("engine._get_quarterly_income_history_sec", lambda ticker_symbol, max_quarters=20: sec_df)
    monkeypatch.setattr("engine.get_yf_ticker", lambda ticker_symbol, use_cache=False: object())
    monkeypatch.setattr("engine.get_yf_frame", lambda stock, field: yahoo_df)

    merged_df, source, diagnostics = _get_quarterly_income_history("MSFT", max_quarters=10)

    assert [str(col)[:10] for col in merged_df.columns] == [
        "2025-12-31",
        "2025-09-30",
        "2025-06-30",
    ]
    assert source == "Yahoo + SEC (validated extension)"
    assert diagnostics["sec_extension_applied"] is True
    assert diagnostics["sec_extended_quarters"] == ["2025-Q2"]
    assert diagnostics["sec_annual_end_backfilled"]["total_annual_end_derived"] == 1
    assert diagnostics["sec_annual_end_backfilled"]["quarters"] == ["2025-Q2"]


def test_filter_source_diagnostics_for_window_keeps_only_visible_quarters():
    diagnostics = {
        "sec_overlap_points": 5,
        "sec_extension_applied": True,
        "sec_extended_quarters": ["2024-Q2", "2023-Q2", "2022-Q2", "2021-Q2"],
        "sec_annual_end_backfilled": {
            "total_annual_end_derived": 4,
            "quarters": ["2024-Q2", "2023-Q2", "2022-Q2", "2021-Q2"],
        },
    }

    filtered = _filter_source_diagnostics_for_window(
        diagnostics,
        [
            (2025, 4),
            (2025, 3),
            (2025, 2),
            (2025, 1),
            (2024, 4),
            (2024, 3),
            (2024, 2),
            (2024, 1),
            (2023, 4),
            (2023, 3),
        ],
    )

    assert filtered["sec_extended_quarters"] == ["2024-Q2"]
    assert filtered["sec_extension_applied_in_window"] is True
    assert filtered["sec_annual_end_backfilled"]["total_annual_end_derived"] == 1
    assert filtered["sec_annual_end_backfilled"]["quarters"] == ["2024-Q2"]


def test_analyze_quarterly_trends_trims_sec_backfill_diagnostics_to_requested_window(monkeypatch):
    quarter_range = pd.period_range(end="2025Q4", periods=20, freq="Q")
    quarter_dates = [pd.Timestamp(period.end_time.date()) for period in reversed(quarter_range)]
    income_df = pd.DataFrame(
        {
            quarter_date: {
                "Total Revenue": 100_000_000_000 - idx * 1_000_000_000,
                "Operating Income": 30_000_000_000 - idx * 500_000_000,
                "Basic EPS": 5.0 - idx * 0.05,
                "Diluted EPS": 5.0 - idx * 0.05,
            }
            for idx, quarter_date in enumerate(quarter_dates)
        }
    )
    diagnostics = {
        "sec_overlap_points": 5,
        "sec_extension_applied": True,
        "sec_extended_quarters": ["2024-Q2", "2023-Q2", "2022-Q2", "2021-Q2"],
        "sec_annual_end_backfilled": {
            "total_annual_end_derived": 4,
            "quarters": ["2024-Q2", "2023-Q2", "2022-Q2", "2021-Q2"],
        },
    }

    monkeypatch.setattr("engine.get_yf_ticker", lambda ticker_symbol: object())
    monkeypatch.setattr("engine.get_yf_info", lambda stock: {})
    monkeypatch.setattr("engine.get_yf_fast_info", lambda stock: {})
    monkeypatch.setattr(
        "engine._get_quarterly_income_history",
        lambda ticker_symbol, max_quarters=20: (income_df, "Yahoo + SEC (validated extension)", diagnostics),
    )
    monkeypatch.setattr(
        "engine.fetch_consensus_estimates",
        lambda ticker_symbol, next_quarter_label, include_qualitative=False, fmp_api_key=None: {},
    )

    result = analyze_quarterly_trends("MSFT", num_quarters=10)

    assert result["data_source"] == "Yahoo + SEC (validated extension)"
    filtered = result["historical_trends"]["source_diagnostics"]
    assert filtered["sec_extended_quarters"] == ["2024-Q2"]
    assert filtered["sec_annual_end_backfilled"]["total_annual_end_derived"] == 1
    assert filtered["sec_annual_end_backfilled"]["quarters"] == ["2024-Q2"]


def test_analyze_quarterly_trends_uses_company_fiscal_quarter_labels(monkeypatch):
    income_df = _quarterly_income_df(
        {
            "Total Revenue": {
                "2025-06-30": 70_000_000_000,
                "2025-03-31": 68_000_000_000,
                "2024-12-31": 66_000_000_000,
                "2024-09-30": 64_000_000_000,
                "2024-06-30": 62_000_000_000,
            },
            "Operating Income": {
                "2025-06-30": 30_000_000_000,
                "2025-03-31": 29_000_000_000,
                "2024-12-31": 28_000_000_000,
                "2024-09-30": 27_000_000_000,
                "2024-06-30": 26_000_000_000,
            },
            "Basic EPS": {
                "2025-06-30": 3.1,
                "2025-03-31": 3.0,
                "2024-12-31": 2.9,
                "2024-09-30": 2.8,
                "2024-06-30": 2.7,
            },
            "Diluted EPS": {
                "2025-06-30": 3.1,
                "2025-03-31": 3.0,
                "2024-12-31": 2.9,
                "2024-09-30": 2.8,
                "2024-06-30": 2.7,
            },
        }
    )
    captured = {}

    monkeypatch.setattr("engine.get_yf_ticker", lambda ticker_symbol: object())
    monkeypatch.setattr(
        "engine.get_yf_info",
        lambda stock: {
            "longName": "Microsoft Corporation",
            "lastFiscalYearEnd": int(pd.Timestamp("2025-06-30", tz="UTC").timestamp()),
        },
    )
    monkeypatch.setattr("engine.get_yf_fast_info", lambda stock: {})
    monkeypatch.setattr(
        "engine._get_quarterly_income_history",
        lambda ticker_symbol, max_quarters=20: (income_df, "Yahoo Finance", {}),
    )

    def _fake_consensus(ticker_symbol, next_quarter_label, include_qualitative=False, fmp_api_key=None):
        captured["next_quarter_label"] = next_quarter_label
        return {"next_quarter": {"quarter_label": f"{next_quarter_label} (Est.)"}}

    monkeypatch.setattr("engine.fetch_consensus_estimates", _fake_consensus)

    result = analyze_quarterly_trends("MSFT", num_quarters=5)

    assert result["fiscal_calendar"]["fiscal_year_end_month"] == 6
    assert result["fiscal_calendar"]["is_calendar_aligned"] is False
    assert "calendar Q2 is fiscal Q4" in result["fiscal_calendar"]["note"]
    assert result["historical_trends"]["most_recent_quarter"]["label"] == "FY2025 Q4"
    assert result["historical_trends"]["most_recent_quarter"]["calendar_label"] == "CY2025 Q2"
    assert result["historical_trends"]["quarterly_data"][0]["quarter"] == "FY2025 Q4"
    assert result["historical_trends"]["quarterly_data"][0]["calendar_quarter"] == 2
    assert result["historical_trends"]["quarterly_data"][1]["quarter"] == "FY2025 Q3"
    assert result["next_forecast_quarter"]["label"] == "FY2026 Q1"
    assert captured["next_quarter_label"] == "FY2026 Q1"
