import pytest
import pandas as pd

from engine import (
    _get_quarterly_income_history,
    _get_quarterly_income_history_sec,
    _merge_yahoo_sec_quarterly_income,
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
