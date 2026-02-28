# app.py
"""
Analyst Co-Pilot - v3.0
=======================
Clean, minimalistic financial analysis tool with 3 steps:
1. Historical Analysis & Growth Rates
2. Wall Street Consensus
3. AI Outlook
"""

import os
import re
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

# Load API keys from .env file (if exists)
load_dotenv()
try:
    if not os.environ.get("GEMINI_API_KEY"):
        secret_key = st.secrets.get("GEMINI_API_KEY")
        if secret_key:
            os.environ["GEMINI_API_KEY"] = str(secret_key)
except Exception:
    # st.secrets is not always available locally.
    pass
import pandas as pd
import altair as alt
import json
from engine import get_financials, run_structured_prompt, calculate_metrics, run_chat, analyze_quarterly_trends, generate_independent_forecast, get_latest_date_info, get_available_report_dates, calculate_comprehensive_analysis
from data_adapter import DataAdapter, DataQualityMetadata, NormalizedFinancialSnapshot
from dcf_engine import DCFEngine, DCFAssumptions
from dcf_ui_adapter import DCFUIAdapter
from sources import SOURCE_CATALOG

UI_CACHE_VERSION = 1
REPORT_DATES_CACHE_VERSION = "v5"
UI_CACHE_PATH = Path(__file__).resolve().parent / "data" / "user_ui_cache.json"
MAX_TICKER_LIBRARY_SIZE = 100
MAX_REPORT_DATE_CACHE_TICKERS = 300
TICKER_PATTERN = re.compile(r"^[A-Z0-9.\-]{1,10}$")
MAG7_TICKERS = ["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA"]
AVAILABLE_DATES_TIMEOUT_SECONDS = 8
FINANCIALS_TIMEOUT_SECONDS = 20
QUARTERLY_ANALYSIS_TIMEOUT_SECONDS = 25
SNAPSHOT_TIMEOUT_SECONDS = 12
SNAPSHOT_METADATA_FIELDS = [
    "price",
    "shares_outstanding",
    "market_cap",
    "total_debt",
    "average_total_debt",
    "cash_and_equivalents",
    "ttm_revenue",
    "ttm_operating_cash_flow",
    "ttm_capex",
    "ttm_fcf",
    "ttm_ebitda",
    "ttm_operating_income",
    "ttm_net_income",
    "effective_tax_rate",
    "beta",
    "suggested_wacc",
    "suggested_cost_of_equity",
    "suggested_cost_of_debt",
    "suggested_fcf_growth",
    "analyst_long_term_growth",
]


def _default_ui_cache() -> dict:
    return {
        "version": UI_CACHE_VERSION,
        "available_report_dates_version": REPORT_DATES_CACHE_VERSION,
        "ticker_library": MAG7_TICKERS.copy(),
        "last_selected_ticker": "MSFT",
        "available_report_dates": {},
        "results": {},
    }


def _normalize_ticker(ticker: str) -> str:
    if ticker is None:
        return ""
    return str(ticker).strip().upper()


def _is_valid_ticker_format(ticker: str) -> bool:
    return bool(TICKER_PATTERN.match(_normalize_ticker(ticker)))


def _normalize_ticker_library(raw_tickers) -> list:
    ordered = []
    for ticker in MAG7_TICKERS + (raw_tickers or []):
        t = _normalize_ticker(ticker)
        if not t or not _is_valid_ticker_format(t):
            continue
        if t not in ordered:
            ordered.append(t)
        if len(ordered) >= MAX_TICKER_LIBRARY_SIZE:
            break
    return ordered if ordered else MAG7_TICKERS.copy()


def _normalize_report_dates(raw_dates) -> list:
    normalized = []
    seen_values = set()
    if not isinstance(raw_dates, list):
        return normalized
    for entry in raw_dates:
        if not isinstance(entry, dict):
            continue
        value = str(entry.get("value", "")).strip()
        display = str(entry.get("display", "")).strip()
        if not value:
            continue
        if value in seen_values:
            continue
        seen_values.add(value)
        normalized.append({"display": display or value, "value": value})
    return normalized


def _normalize_available_report_dates_cache(raw_cache) -> dict:
    normalized = {}
    if not isinstance(raw_cache, dict):
        return normalized
    for raw_ticker, raw_dates in raw_cache.items():
        ticker = _normalize_ticker(raw_ticker)
        if not _is_valid_ticker_format(ticker):
            continue
        dates = _normalize_report_dates(raw_dates)
        if not dates:
            continue
        normalized[ticker] = dates
        if len(normalized) >= MAX_REPORT_DATE_CACHE_TICKERS:
            break
    return normalized


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    return str(value)


def _sanitize_ai_valuation_language(text: str) -> str:
    """
    Keep valuation wording assumption-aware in rendered AI text.
    Also normalizes older cached outputs produced before prompt updates.
    """
    if not isinstance(text, str):
        return text
    sanitized = text
    replacements = [
        (r"(?i)\bfundamental floor\b", "model-implied value under current assumptions"),
        (r"(?i)\bvaluation floor\b", "model-implied value under current assumptions"),
        (r"(?i)\bintrinsic floor\b", "model-implied value under current assumptions"),
        (r"(?i)\bhard floor\b", "assumption-sensitive downside case"),
    ]
    for pattern, replacement in replacements:
        sanitized = re.sub(pattern, replacement, sanitized)
    return sanitized


def load_ui_cache() -> dict:
    default = _default_ui_cache()
    if not UI_CACHE_PATH.exists():
        return default
    try:
        with UI_CACHE_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return default
        report_dates_version = str(data.get("available_report_dates_version", "")).strip()
        available_dates_cache = (
            _normalize_available_report_dates_cache(data.get("available_report_dates", {}))
            if report_dates_version == REPORT_DATES_CACHE_VERSION
            else {}
        )
        cache = {
            "version": data.get("version", UI_CACHE_VERSION),
            "available_report_dates_version": REPORT_DATES_CACHE_VERSION,
            "ticker_library": _normalize_ticker_library(data.get("ticker_library", [])),
            "last_selected_ticker": _normalize_ticker(data.get("last_selected_ticker", "MSFT")) or "MSFT",
            "available_report_dates": available_dates_cache,
            "results": data.get("results", {}) if isinstance(data.get("results", {}), dict) else {},
        }
        return cache
    except Exception:
        return default


def save_ui_cache(cache_obj: dict) -> None:
    try:
        UI_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": UI_CACHE_VERSION,
            "available_report_dates_version": REPORT_DATES_CACHE_VERSION,
            "ticker_library": _normalize_ticker_library(cache_obj.get("ticker_library", [])),
            "last_selected_ticker": _normalize_ticker(cache_obj.get("last_selected_ticker", "MSFT")) or "MSFT",
            "available_report_dates": _normalize_available_report_dates_cache(cache_obj.get("available_report_dates", {})),
            "results": _json_safe(cache_obj.get("results", {})),
        }
        tmp_path = UI_CACHE_PATH.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        tmp_path.replace(UI_CACHE_PATH)
    except Exception:
        # Persistence failure should never break the UI flow.
        pass


def build_context_key(ticker: str, end_date: str, num_quarters: int) -> str:
    return f"{_normalize_ticker(ticker)}|{end_date}|{int(num_quarters)}"


def _metadata_from_dict(raw: dict) -> DataQualityMetadata:
    if not isinstance(raw, dict):
        return DataQualityMetadata()
    return DataQualityMetadata(
        value=raw.get("value"),
        units=raw.get("units", "USD"),
        period_end=raw.get("period_end"),
        period_type=raw.get("period_type"),
        source_path=raw.get("source_path"),
        retrieved_at=raw.get("retrieved_at"),
        reliability_score=raw.get("reliability_score", 0),
        notes=raw.get("notes"),
        is_estimated=raw.get("is_estimated", False),
        fallback_reason=raw.get("fallback_reason"),
    )


def snapshot_from_dict(raw: dict) -> NormalizedFinancialSnapshot:
    base = raw if isinstance(raw, dict) else {}
    ticker = _normalize_ticker(base.get("ticker")) or "UNKNOWN"
    snapshot = NormalizedFinancialSnapshot(ticker)

    snapshot.retrieved_at = base.get("retrieved_at", snapshot.retrieved_at)
    snapshot.currency = base.get("currency", snapshot.currency)
    snapshot.company_name = base.get("company_name")
    snapshot.sector = base.get("sector")
    snapshot.industry = base.get("industry")
    snapshot.num_quarters_available = base.get("num_quarters_available", 0)
    snapshot.overall_quality_score = base.get("overall_quality_score", snapshot.overall_quality_score)
    snapshot.warnings = base.get("warnings", []) if isinstance(base.get("warnings", []), list) else []
    snapshot.errors = base.get("errors", []) if isinstance(base.get("errors", []), list) else []
    snapshot.wacc_components = base.get("wacc_components", {}) if isinstance(base.get("wacc_components", {}), dict) else {}
    snapshot.risk_free_rate = base.get("risk_free_rate")
    snapshot.rf_source = base.get("rf_source")
    snapshot.analyst_revenue_estimates = (
        base.get("analyst_revenue_estimates", [])
        if isinstance(base.get("analyst_revenue_estimates", []), list)
        else []
    )

    for field in SNAPSHOT_METADATA_FIELDS:
        setattr(snapshot, field, _metadata_from_dict(base.get(field, {})))

    return snapshot


def _upsert_ticker_in_library(ticker: str) -> None:
    t = _normalize_ticker(ticker)
    if not _is_valid_ticker_format(t):
        return

    cache = st.session_state.get("ui_cache", _default_ui_cache())
    library = _normalize_ticker_library(cache.get("ticker_library", []))
    if t not in library:
        library.append(t)
        library = _normalize_ticker_library(library)
        cache["ticker_library"] = library
        st.session_state.ui_cache = cache
        st.session_state.ticker_library = library
        save_ui_cache(cache)


def _get_persisted_report_dates(ticker: str) -> list:
    normalized_ticker = _normalize_ticker(ticker)
    if not _is_valid_ticker_format(normalized_ticker):
        return []

    cache = st.session_state.get("ui_cache", _default_ui_cache())
    dates_cache = _normalize_available_report_dates_cache(cache.get("available_report_dates", {}))
    return dates_cache.get(normalized_ticker, [])


def _persist_report_dates_for_ticker(ticker: str, dates: list) -> None:
    normalized_ticker = _normalize_ticker(ticker)
    if not _is_valid_ticker_format(normalized_ticker):
        return

    normalized_dates = _normalize_report_dates(dates)
    if not normalized_dates:
        return

    cache = st.session_state.get("ui_cache", _default_ui_cache())
    dates_cache = _normalize_available_report_dates_cache(cache.get("available_report_dates", {}))
    if normalized_ticker in dates_cache:
        dates_cache.pop(normalized_ticker, None)
    dates_cache[normalized_ticker] = normalized_dates

    while len(dates_cache) > MAX_REPORT_DATE_CACHE_TICKERS:
        oldest_ticker = next(iter(dates_cache))
        dates_cache.pop(oldest_ticker, None)

    cache["available_report_dates"] = dates_cache
    st.session_state.ui_cache = cache
    save_ui_cache(cache)


def _load_report_dates_for_ticker(ticker: str, force_refresh: bool = False) -> list:
    normalized_ticker = _normalize_ticker(ticker)
    if not _is_valid_ticker_format(normalized_ticker):
        return []

    persisted_dates = _get_persisted_report_dates(normalized_ticker)
    if persisted_dates and not force_refresh:
        return persisted_dates

    if force_refresh:
        fetched_dates = _call_with_timeout(
            get_available_report_dates,
            normalized_ticker,
            timeout_seconds=AVAILABLE_DATES_TIMEOUT_SECONDS,
            fallback=[],
        )
    else:
        fetched_dates = cached_available_dates(normalized_ticker)

    normalized_dates = _normalize_report_dates(fetched_dates)
    if normalized_dates:
        _persist_report_dates_for_ticker(normalized_ticker, normalized_dates)
        return normalized_dates

    return persisted_dates if persisted_dates else []


def _restore_cached_results_for_context(ticker: str, end_date: str, num_quarters: int) -> dict:
    restored = {"dcf": False, "ai": False}
    if not ticker or not end_date or num_quarters is None:
        return restored

    cache = st.session_state.get("ui_cache", _default_ui_cache())
    context_key = build_context_key(ticker, end_date, num_quarters)
    entry = cache.get("results", {}).get(context_key, {})
    if not isinstance(entry, dict):
        return restored

    dcf_entry = entry.get("dcf")
    if isinstance(dcf_entry, dict):
        engine_result = dcf_entry.get("engine_result")
        snapshot_dict = dcf_entry.get("snapshot")
        if isinstance(engine_result, dict) and isinstance(snapshot_dict, dict):
            try:
                snapshot = snapshot_from_dict(snapshot_dict)
                ui_adapter = DCFUIAdapter(engine_result, snapshot)
                st.session_state.dcf_ui_adapter = ui_adapter
                st.session_state.dcf_engine_result = engine_result
                st.session_state.dcf_snapshot = snapshot
                st.session_state.dcf_wacc = dcf_entry.get("dcf_wacc")
                st.session_state.dcf_fcf_growth = dcf_entry.get("dcf_fcf_growth")
                terminal_growth_from_cache = dcf_entry.get("dcf_terminal_growth")
                if terminal_growth_from_cache is None:
                    terminal_growth_from_cache = (engine_result.get("assumptions", {}).get("terminal_growth_rate") or 0.03) * 100
                st.session_state.dcf_terminal_growth = terminal_growth_from_cache
                st.session_state.dcf_terminal_scenario = dcf_entry.get("dcf_terminal_scenario")
                st.session_state.dcf_custom_multiple = dcf_entry.get("dcf_custom_multiple")
                restored["dcf"] = True
            except Exception:
                pass

    ai_entry = entry.get("ai_outlook")
    if isinstance(ai_entry, dict) and isinstance(ai_entry.get("independent_forecast"), dict):
        st.session_state.independent_forecast = ai_entry.get("independent_forecast")
        st.session_state.forecast_just_generated = False
        restored["ai"] = True

    if restored["dcf"] or restored["ai"]:
        st.session_state.last_restore_key = context_key

    return restored


def _persist_dcf_result_for_context(ticker: str, end_date: str, num_quarters: int) -> None:
    if not ticker or not end_date or num_quarters is None:
        return

    engine_result = st.session_state.get("dcf_engine_result")
    snapshot = st.session_state.get("dcf_snapshot")
    if not isinstance(engine_result, dict) or snapshot is None:
        return

    snapshot_dict = snapshot.to_dict() if hasattr(snapshot, "to_dict") else None
    if not isinstance(snapshot_dict, dict):
        return

    cache = st.session_state.get("ui_cache", _default_ui_cache())
    results = cache.setdefault("results", {})
    context_key = build_context_key(ticker, end_date, num_quarters)
    entry = results.get(context_key, {})
    if not isinstance(entry, dict):
        entry = {}

    entry.update({
        "ticker": _normalize_ticker(ticker),
        "end_date": end_date,
        "num_quarters": int(num_quarters),
        "updated_at": datetime.utcnow().isoformat(),
    })
    entry["dcf"] = {
        "dcf_wacc": st.session_state.get("dcf_wacc"),
        "dcf_fcf_growth": st.session_state.get("dcf_fcf_growth"),
        "dcf_terminal_growth": st.session_state.get("dcf_terminal_growth"),
        "dcf_terminal_scenario": st.session_state.get("dcf_terminal_scenario"),
        "dcf_custom_multiple": st.session_state.get("dcf_custom_multiple"),
        "engine_result": _json_safe(engine_result),
        "snapshot": _json_safe(snapshot_dict),
    }
    results[context_key] = entry
    cache["results"] = results
    cache["last_selected_ticker"] = _normalize_ticker(ticker)
    st.session_state.ui_cache = cache
    save_ui_cache(cache)


def _persist_ai_result_for_context(ticker: str, end_date: str, num_quarters: int) -> None:
    if not ticker or not end_date or num_quarters is None:
        return

    forecast = st.session_state.get("independent_forecast")
    if not isinstance(forecast, dict):
        return

    cache = st.session_state.get("ui_cache", _default_ui_cache())
    results = cache.setdefault("results", {})
    context_key = build_context_key(ticker, end_date, num_quarters)
    entry = results.get(context_key, {})
    if not isinstance(entry, dict):
        entry = {}

    entry.update({
        "ticker": _normalize_ticker(ticker),
        "end_date": end_date,
        "num_quarters": int(num_quarters),
        "updated_at": datetime.utcnow().isoformat(),
    })
    entry["ai_outlook"] = {
        "independent_forecast": _json_safe(forecast),
        "forecast_date": forecast.get("forecast_date"),
    }
    results[context_key] = entry
    cache["results"] = results
    cache["last_selected_ticker"] = _normalize_ticker(ticker)
    st.session_state.ui_cache = cache
    save_ui_cache(cache)


def _call_with_timeout(func, *args, timeout_seconds: int, fallback=None):
    """
    Prevent long-running upstream data calls from blocking the first paint forever.
    Returns fallback on timeout/error.
    """
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(func, *args)
    try:
        return future.result(timeout=timeout_seconds)
    except FuturesTimeoutError:
        return fallback
    except Exception:
        return fallback
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


# --- Cached API Functions ---
# These decorators cache results so API calls only happen once per input
# TTL (time-to-live) of 1 hour = 3600 seconds

@st.cache_data(ttl=3600, show_spinner=False)
def cached_quarterly_analysis(
    ticker: str,
    num_quarters: int = 8,
    end_date: str = None,
    history_source_version: str = "v11",
) -> dict:
    """Cached version of analyze_quarterly_trends to avoid API rate limits."""
    _ = history_source_version  # cache-key salt when quarterly history sourcing logic changes
    fallback = {
        "ticker": ticker,
        "analysis_date": datetime.now().strftime("%Y-%m-%d"),
        "historical_trends": {},
        "growth_rates": {},
        "projections": {},
        "consensus_estimates": {},
        "next_forecast_quarter": {},
        "errors": [f"Quarterly analysis timed out after {QUARTERLY_ANALYSIS_TIMEOUT_SECONDS}s."],
    }
    result = _call_with_timeout(
        analyze_quarterly_trends,
        ticker,
        num_quarters,
        end_date,
        timeout_seconds=QUARTERLY_ANALYSIS_TIMEOUT_SECONDS,
        fallback=fallback,
    )
    return result if isinstance(result, dict) else fallback

@st.cache_data(ttl=3600, show_spinner=False)
def cached_available_dates(ticker: str, history_source_version: str = "v11") -> list:
    """Cached wrapper for get_available_report_dates."""
    _ = history_source_version  # cache-key salt when available-date sourcing logic changes
    result = _call_with_timeout(
        get_available_report_dates,
        ticker,
        timeout_seconds=AVAILABLE_DATES_TIMEOUT_SECONDS,
        fallback=[],
    )
    return result if isinstance(result, list) else []

@st.cache_data(ttl=3600, show_spinner=False)
def cached_independent_forecast(ticker: str, quarterly_data_hash: str, company_name: str, dcf_data: dict = None) -> dict:
    """
    Cached version of generate_independent_forecast.
    quarterly_data_hash is used to bust cache if underlying data changes.
    """
    # We need to re-fetch the analysis since we can't cache complex dicts as keys
    analysis = cached_quarterly_analysis(ticker)
    return generate_independent_forecast(analysis, company_name, dcf_data)

@st.cache_data(ttl=3600, show_spinner=False)
def cached_financials(ticker: str) -> tuple:
    """Cached version of get_financials."""
    fallback = (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    result = _call_with_timeout(
        get_financials,
        ticker,
        timeout_seconds=FINANCIALS_TIMEOUT_SECONDS,
        fallback=fallback,
    )
    if not isinstance(result, tuple) or len(result) != 4:
        return fallback
    return result

@st.cache_data(ttl=3600, show_spinner=False)
def cached_latest_date_info(ticker: str) -> dict:
    """Cached wrapper for get_latest_date_info from engine."""
    return get_latest_date_info(ticker)

@st.cache_data(ttl=3600, show_spinner=False)
def cached_financial_snapshot(ticker: str, suggestion_algo_version: str = "v2"):
    """Cached wrapper to get financial snapshot for suggested assumptions."""
    _ = suggestion_algo_version  # cache-key salt for suggestion logic revisions
    try:
        adapter = DataAdapter(ticker)
        snapshot = _call_with_timeout(
            adapter.fetch,
            timeout_seconds=SNAPSHOT_TIMEOUT_SECONDS,
            fallback=None,
        )
        return snapshot
    except Exception:
        return None

def run_dcf_analysis(ticker: str, wacc: float = None, fcf_growth: float = None,
                     terminal_growth: float = None, terminal_scenario: str = "current",
                     custom_multiple: float = None) -> tuple:
    """Run DCF analysis with user-adjustable assumptions. Returns (ui_adapter, engine_result, snapshot)."""
    try:
        adapter = DataAdapter(ticker)
        snapshot = adapter.fetch()
        
        # Create assumptions with user overrides
        assumptions = DCFAssumptions()
        if wacc is not None:
            assumptions.wacc = wacc / 100.0  # Convert from percentage
        if fcf_growth is not None:
            assumptions.fcf_growth_rate = fcf_growth / 100.0  # Convert from percentage
        if terminal_growth is not None:
            assumptions.terminal_growth_rate = terminal_growth / 100.0  # Convert from percentage

        # Set terminal multiple scenario
        assumptions.terminal_multiple_scenario = terminal_scenario
        if terminal_scenario == "custom" and custom_multiple is not None:
            assumptions.exit_multiple = custom_multiple
        assumptions.cashflow_discount_policy = "adaptive"
        assumptions.proxy_risk_premium_pct = 1.0
        
        engine = DCFEngine(snapshot, assumptions)
        engine_result = engine.run()
        ui_adapter = DCFUIAdapter(engine_result, snapshot)
        return (ui_adapter, engine_result, snapshot)
    except Exception as e:
        return (None, {"success": False, "error": str(e), "errors": [str(e)]}, None)


def _show_dcf_details_page():
    """Render DCF details in a clear sequential flow."""
    st.markdown("---")
    st.subheader("DCF Calculation Details")

    if st.button("<- Back to Summary", key="back_from_details_top"):
        st.session_state.show_dcf_details = False
        st.rerun()

    ui_adapter = st.session_state.dcf_ui_adapter
    ui_data = ui_adapter.get_ui_data()
    engine_result = st.session_state.get("dcf_engine_result") or {}
    snapshot = st.session_state.get("dcf_snapshot")

    assumptions = ui_data.get("assumptions", {}) or {}
    yearly_projections = assumptions.get("yearly_projections", []) or []
    fcf_projections = ui_data.get("fcf_projections", []) or []
    trace_steps = engine_result.get("trace", []) if isinstance(engine_result, dict) else []

    forecast_years = int(assumptions.get("forecast_years") or 5)
    wacc_used = assumptions.get("wacc")
    terminal_growth = assumptions.get("terminal_growth_rate")
    tv_method = assumptions.get("terminal_value_method", "gordon_growth")

    ev = ui_data.get("enterprise_value")
    equity = ui_data.get("equity_value")
    price_per_share = ui_data.get("price_per_share")
    pv_fcf_sum = ui_data.get("pv_fcf_sum")
    pv_tv = ui_data.get("pv_terminal_value")
    terminal_value_yearN = ui_data.get("terminal_value_yearN")
    net_debt = ui_data.get("net_debt")
    shares_outstanding = ui_data.get("shares_outstanding")

    def _to_float(value):
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    def _fmt_money(value, decimals=2):
        num = _to_float(value)
        if num is None:
            return "N/A"
        abs_num = abs(num)
        if abs_num >= 1e12:
            return f"${num / 1e12:.{decimals}f}T"
        if abs_num >= 1e9:
            return f"${num / 1e9:.{decimals}f}B"
        if abs_num >= 1e6:
            return f"${num / 1e6:.{decimals}f}M"
        return f"${num:,.{decimals}f}"

    def _fmt_rate(value, decimals=1):
        num = _to_float(value)
        if num is None:
            return "N/A"
        return f"{num * 100:.{decimals}f}%"

    def _fmt_multiple(value, decimals=1):
        num = _to_float(value)
        if num is None:
            return "N/A"
        return f"{num:.{decimals}f}x"

    def _fmt_number(value, decimals=2):
        num = _to_float(value)
        if num is None:
            return "N/A"
        return f"{num:,.{decimals}f}"

    def _parse_reliability(value):
        if isinstance(value, str) and value.endswith("/100"):
            try:
                return int(value.split("/")[0])
            except ValueError:
                return None
        return None

    def _format_trace_chat_response(raw_text: str) -> str:
        """Normalize model output into a consistent, readable structure."""
        if not isinstance(raw_text, str) or not raw_text.strip():
            return (
                "**Answer:** Unable to generate a structured response.\n\n"
                "**Value:** N/A\n\n"
                "**Formula Path:** N/A\n\n"
                "**Source Path:** N/A"
            )

        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
            cleaned = cleaned.strip()

        parsed = None
        try:
            parsed = json.loads(cleaned)
        except Exception:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    parsed = json.loads(cleaned[start:end + 1])
                except Exception:
                    parsed = None

        if isinstance(parsed, dict):
            direct_answer = str(parsed.get("direct_answer") or parsed.get("answer") or "").strip()
            value_text = str(parsed.get("value") or "").strip()
            formula_text = str(parsed.get("formula_path") or parsed.get("formula") or "").strip()
            source_text = str(parsed.get("source_path") or parsed.get("source") or "").strip()
        else:
            lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
            direct_answer = lines[0] if lines else cleaned

            value_match = re.search(r"(?i)\bValue:\s*(.+)", cleaned)
            formula_match = re.search(r"(?i)\bFormula(?:\s*path)?:\s*(.+)", cleaned)
            source_match = re.search(r"(?i)\bSource(?:\s*path)?:\s*(.+)", cleaned)

            value_text = value_match.group(1).strip() if value_match else ""
            formula_text = formula_match.group(1).strip() if formula_match else ""
            source_text = source_match.group(1).strip() if source_match else ""

        if not direct_answer:
            direct_answer = "Direct answer not provided."
        if not value_text:
            value_text = "Not explicitly provided."
        if not formula_text:
            formula_text = "Not explicitly provided."
        if not source_text:
            source_text = "Not explicitly provided."

        return (
            f"**Answer:** {direct_answer}\n\n"
            f"**Value:** {value_text}\n\n"
            f"**Formula Path:** {formula_text}\n\n"
            f"**Source Path:** {source_text}"
        )

    def _sanitize_notice_text(text: str) -> str:
        """Remove decorative emoji/icons and normalize spacing for professional warning copy."""
        if not isinstance(text, str):
            return str(text)
        cleaned = text.strip()
        for marker in ["⚠️", "⚠", "🔴", "🟡", "🟠", "🟢", "✅", "❌", "📊", "📈", "📉", "ℹ️", "ℹ"]:
            cleaned = cleaned.replace(marker, "")
        cleaned = re.sub(r"^[\s\-\u2022]+", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _classify_notice(text: str, source_kind: str) -> str:
        """Classify message severity for grouped display."""
        if source_kind == "error":
            return "critical"

        lower = text.lower()
        critical_tokens = [
            "fatal",
            "invalid",
            "implausible",
            "framework mixing",
            "cannot calculate",
            "terminal growth",
            "dominates",
        ]
        review_tokens = [
            "very low",
            "very high",
            "disagree",
            "sensitive",
            "fallback",
            "approximate",
            "proxy",
            "elevated",
            "defaulted",
            "missing",
            "low ",
            "high ",
        ]

        if any(token in lower for token in critical_tokens):
            return "critical"
        if any(token in lower for token in review_tokens):
            return "review"
        return "info"

    def _estimate_quota_reset_text(error_text: str) -> str:
        """Estimate next usable time from rate-limit text; fall back to next daily reset estimate."""
        text = str(error_text or "")
        lower = text.lower()

        retry_seconds = None
        retry_patterns = [
            r"retry in\s+(\d+)\s*(?:seconds|secs|sec|s)\b",
            r"try again in\s+(\d+)\s*(?:seconds|secs|sec|s)\b",
            r"retry_delay[^\d]{0,20}(\d+)",
            r"seconds[\"']?\s*:\s*(\d+)",
        ]
        for pattern in retry_patterns:
            match = re.search(pattern, lower)
            if match:
                try:
                    retry_seconds = int(match.group(1))
                    break
                except Exception:
                    pass

        if retry_seconds is not None and retry_seconds > 0:
            retry_dt_utc = datetime.now(timezone.utc) + timedelta(seconds=retry_seconds)
            retry_local = retry_dt_utc.astimezone()
            return retry_local.strftime("%b %d, %Y %I:%M %p %Z")

        # Quota resets are provider-plan dependent; this is an explicit estimate.
        try:
            pt = ZoneInfo("America/Los_Angeles")
            now_pt = datetime.now(pt)
            next_reset_pt = (now_pt + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            return next_reset_pt.strftime("%b %d, %Y %I:%M %p PT (estimated)")
        except Exception:
            return "next daily quota reset (estimated)"

    def _format_chat_runtime_message(raw_text: str) -> str:
        """Create user-facing assistant status when API/chat call fails."""
        message = str(raw_text or "").strip()
        lower = message.lower()
        if lower.startswith("chat error:"):
            message = message.split(":", 1)[1].strip()
            lower = message.lower()
        elif lower.startswith("error:"):
            message = message.split(":", 1)[1].strip()
            lower = message.lower()

        quota_markers = [
            "resource_exhausted",
            "quota",
            "rate limit",
            "too many requests",
            "429",
        ]
        if any(marker in lower for marker in quota_markers):
            reset_text = _estimate_quota_reset_text(message)
            return (
                "**Assistant temporarily unavailable (API request limit reached).**\n\n"
                f"**Estimated retry/reset time:** {reset_text}\n\n"
                "The DCF tables and trace above are still available for manual verification."
            )

        if "api key" in lower:
            return (
                "**Assistant unavailable (API key missing or invalid).**\n\n"
                "Set `GEMINI_API_KEY` and rerun."
            )

        return (
            "**Assistant unavailable right now.**\n\n"
            f"**Details:** {message}\n\n"
            "You can still inspect the trace and source tables above."
        )

    st.caption("Sequential view of inputs, assumptions, calculations, and outputs.")

    col_h1, col_h2, col_h3, col_h4 = st.columns(4)
    with col_h1:
        st.metric(
            "Intrinsic Value / Share",
            f"${_to_float(price_per_share):.2f}" if _to_float(price_per_share) is not None else "N/A",
        )
    with col_h2:
        st.metric("Enterprise Value", _fmt_money(ev))
    with col_h3:
        st.metric("Equity Value", _fmt_money(equity))
    with col_h4:
        tv_ratio = ((_to_float(pv_tv) or 0.0) / (_to_float(ev) or 1.0) * 100.0) if _to_float(ev) else 0.0
        st.metric("PV(Terminal) as % EV", f"{tv_ratio:.1f}%")

    st.markdown("### 1. Input Data")
    input_order = {
        "Current Price": 1,
        "Market Cap": 2,
        "Shares Outstanding": 3,
        "TTM Revenue": 4,
        "TTM Operating Income": 5,
        "TTM EBITDA": 6,
        "TTM Operating Cash Flow": 7,
        "TTM CapEx": 8,
        "TTM Free Cash Flow": 9,
        "Total Debt": 10,
        "Cash & Equivalents": 11,
    }

    input_rows_raw = ui_adapter.format_input_table()
    input_rows = sorted(input_rows_raw, key=lambda row: input_order.get(row.get("Item", ""), 999))

    input_table = []
    low_reliability_inputs = []
    for row in input_rows:
        score = _parse_reliability(row.get("Reliability", ""))
        if score is not None and score < 60:
            low_reliability_inputs.append(f"{row.get('Item', 'Unknown')} ({score}/100)")

        input_table.append(
            {
                "Item": row.get("Item", "N/A"),
                "Value": row.get("Value", "N/A"),
                "Period": row.get("Period", "N/A"),
                "Source": row.get("Source", "N/A"),
                "Reliability": row.get("Reliability", "N/A"),
                "Notes": row.get("Notes", "N/A"),
            }
        )

    st.dataframe(pd.DataFrame(input_table), use_container_width=True, hide_index=True)
    if low_reliability_inputs:
        st.warning("Low reliability inputs: " + ", ".join(low_reliability_inputs))

    st.markdown("### 2. Assumptions")
    assumption_rows = [
        {
            "Assumption": "Forecast Horizon",
            "Value": f"{forecast_years} years",
            "Where Used": "Projection length and discount periods",
        },
        {
            "Assumption": "WACC",
            "Value": _fmt_rate(wacc_used),
            "Where Used": "Discounting explicit cash flows and terminal value",
        },
        {
            "Assumption": "Near-Term Growth",
            "Value": _fmt_rate(assumptions.get("fcf_growth_rate")),
            "Where Used": "Starting point for growth schedule",
        },
        {
            "Assumption": "Terminal Growth (g)",
            "Value": _fmt_rate(terminal_growth),
            "Where Used": "Terminal value and spread checks",
        },
        {
            "Assumption": "Tax Rate",
            "Value": _fmt_rate(assumptions.get("tax_rate")),
            "Where Used": "NOPAT and FCFF calculations",
        },
        {
            "Assumption": "Terminal Method",
            "Value": str(tv_method).replace("_", " ").title(),
            "Where Used": "Primary terminal valuation branch",
        },
        {
            "Assumption": "Exit Multiple",
            "Value": _fmt_multiple(assumptions.get("exit_multiple")),
            "Where Used": "Exit-multiple terminal valuation",
        },
        {
            "Assumption": "Growth Schedule",
            "Value": assumptions.get("growth_schedule_method", "N/A"),
            "Where Used": "Shape of year-by-year growth path",
        },
    ]
    if snapshot:
        wacc_components = getattr(snapshot, "wacc_components", {}) or {}
        if wacc_components.get("wacc_mode_label"):
            assumption_rows.insert(2, {
                "Assumption": "WACC Mode",
                "Value": str(wacc_components.get("wacc_mode_label")),
                "Where Used": "Interpretation confidence for discount-rate construction",
            })
    st.dataframe(pd.DataFrame(assumption_rows), use_container_width=True, hide_index=True)

    if snapshot:
        wacc_components = getattr(snapshot, "wacc_components", {}) or {}
        if wacc_components:
            wacc_mode = wacc_components.get("wacc_mode", "")
            wacc_mode_label = wacc_components.get("wacc_mode_label", "Component-based WACC estimate")
            wacc_confidence = str(wacc_components.get("wacc_confidence", "n/a")).title()
            mode_explanations = {
                "full_wacc_observed_debt": "Debt cost is directly inferred from interest expense and debt basis.",
                "estimated_wacc_debt_fallback": "Debt cost uses a synthetic spread fallback where observed debt cost is unreliable.",
                "coe_only_proxy_no_debt": "No debt was detected, so discount rate collapses to cost of equity.",
                "coe_only_proxy_financial_sector": "Financial-sector guardrail is active; CoE proxy is used instead of standard WACC.",
            }
            st.markdown("#### WACC Trace")
            st.info(
                "Forward WACC is inherently estimated from observable inputs (it is not directly observable as a single live metric). "
                f"{wacc_mode_label} (confidence: {wacc_confidence}). "
                f"{mode_explanations.get(wacc_mode, 'Discount rate is built from component inputs with explicit source labels.')}"
            )

            rf = _to_float(wacc_components.get("risk_free_rate"))
            erp = _to_float(wacc_components.get("market_risk_premium"))
            beta = _to_float(wacc_components.get("beta"))
            cost_of_equity_rate = _to_float(wacc_components.get("cost_of_equity"))
            rd = _to_float(wacc_components.get("cost_of_debt_pre_tax"))
            rd_after_tax = _to_float(wacc_components.get("cost_of_debt_after_tax"))
            tax_rate = _to_float(wacc_components.get("tax_rate"))
            we = _to_float(wacc_components.get("equity_weight"))
            wd = _to_float(wacc_components.get("debt_weight"))

            rd_mode = wacc_components.get("rd_mode")
            rd_observed_raw = _to_float(wacc_components.get("rd_observed_raw"))
            debt_basis_value = _to_float(wacc_components.get("debt_basis_value"))
            equity_value_used = _to_float(wacc_components.get("equity_value_used"))
            total_debt = _to_float(wacc_components.get("total_debt"))
            ttm_interest_expense = _to_float(getattr(getattr(snapshot, "ttm_interest_expense", None), "value", None))
            suggested_wacc_value = _to_float(getattr(getattr(snapshot, "suggested_wacc", None), "value", None))

            capm_formula = "Re = Rf + beta * ERP"
            if all(v is not None for v in [rf, erp, beta, cost_of_equity_rate]):
                capm_formula = (
                    f"Re = {rf*100:.2f}% + {beta:.2f} * {erp*100:.2f}% = {cost_of_equity_rate*100:.2f}%"
                )

            if rd_mode == "observed":
                rd_formula = "Rd = Interest Expense / Debt Basis"
                if all(v is not None for v in [ttm_interest_expense, debt_basis_value, rd_observed_raw]):
                    rd_formula = (
                        f"Rd = {_fmt_money(ttm_interest_expense)} / {_fmt_money(debt_basis_value)}"
                        f" = {rd_observed_raw*100:.2f}%"
                    )
            elif rd_mode == "synthetic_fallback":
                rd_formula = "Rd = Rf + synthetic spread fallback"
                if all(v is not None for v in [rf, rd]):
                    spread = rd - rf
                    rd_formula = f"Rd = {rf*100:.2f}% + {spread*100:.2f}% = {rd*100:.2f}%"
            elif rd_mode == "no_debt":
                rd_formula = "No debt detected -> Rd branch is not used"
            else:
                rd_formula = "Rd path unavailable"

            rd_after_tax_formula = "Rd_after_tax = Rd * (1 - T)"
            if all(v is not None for v in [rd, tax_rate, rd_after_tax]):
                rd_after_tax_formula = (
                    f"Rd_after_tax = {rd*100:.2f}% * (1 - {tax_rate*100:.2f}%) = {rd_after_tax*100:.2f}%"
                )

            weight_formula = "We = E / (E + D), Wd = D / (E + D)"
            if all(v is not None for v in [equity_value_used, total_debt]) and (equity_value_used + total_debt) > 0:
                weight_formula = (
                    f"We = {_fmt_money(equity_value_used)} / ({_fmt_money(equity_value_used)} + {_fmt_money(total_debt)})"
                    f" = {((we or 0.0) * 100):.2f}%, "
                    f"Wd = {_fmt_money(total_debt)} / ({_fmt_money(equity_value_used)} + {_fmt_money(total_debt)})"
                    f" = {((wd or 0.0) * 100):.2f}%"
                )

            if wacc_mode in {"coe_only_proxy_no_debt", "coe_only_proxy_financial_sector"}:
                final_formula = "Discount rate = Re (CoE proxy mode)"
                if cost_of_equity_rate is not None and suggested_wacc_value is not None:
                    final_formula = f"Discount rate = Re = {cost_of_equity_rate*100:.2f}%"
            else:
                final_formula = "WACC = We * Re + Wd * Rd_after_tax"
                if all(v is not None for v in [we, cost_of_equity_rate, wd, rd_after_tax, suggested_wacc_value]):
                    final_formula = (
                        f"WACC = {we*100:.2f}% * {cost_of_equity_rate*100:.2f}% + {wd*100:.2f}% * {rd_after_tax*100:.2f}%"
                        f" = {suggested_wacc_value*100:.2f}%"
                    )

            wacc_trace_rows = [
                {
                    "Step": "1. CAPM Inputs",
                    "Formula / Calculation": (
                        f"Rf={_fmt_rate(rf)}, ERP={_fmt_rate(erp)}, beta={_fmt_number(beta, 2)}"
                    ),
                    "Result": "Inputs loaded",
                    "Source": f"{wacc_components.get('rf_source', '10Y Treasury')} + Damodaran + Yahoo",
                },
                {
                    "Step": "2. Cost of Equity",
                    "Formula / Calculation": capm_formula,
                    "Result": _fmt_rate(cost_of_equity_rate),
                    "Source": wacc_components.get("cost_of_equity_source", "CAPM"),
                },
                {
                    "Step": "3. Cost of Debt (pre-tax)",
                    "Formula / Calculation": rd_formula,
                    "Result": _fmt_rate(rd),
                    "Source": wacc_components.get("debt_cost_source", "N/A"),
                },
                {
                    "Step": "4. Cost of Debt (after-tax)",
                    "Formula / Calculation": rd_after_tax_formula,
                    "Result": _fmt_rate(rd_after_tax),
                    "Source": "Tax-adjusted debt cost",
                },
                {
                    "Step": "5. Capital Structure Weights",
                    "Formula / Calculation": weight_formula,
                    "Result": f"We={_fmt_rate(we)}, Wd={_fmt_rate(wd)}",
                    "Source": wacc_components.get("weights_source", "N/A"),
                },
                {
                    "Step": "6. Final Discount Rate",
                    "Formula / Calculation": final_formula,
                    "Result": _fmt_rate(suggested_wacc_value),
                    "Source": wacc_mode_label,
                },
            ]
            st.dataframe(pd.DataFrame(wacc_trace_rows), use_container_width=True, hide_index=True)

            wacc_rows = [
                {"Component": "WACC Mode", "Value": str(wacc_mode_label), "Source": "Method classification"},
                {"Component": "WACC Confidence", "Value": wacc_confidence, "Source": "Data quality and fallback diagnostics"},
                {"Component": "Debt Basis", "Value": _fmt_money(debt_basis_value), "Source": wacc_components.get("debt_basis_source", "N/A")},
                {"Component": "Tax Rate (T)", "Value": _fmt_rate(tax_rate), "Source": "Income statement / fallback"},
            ]
            st.markdown("#### WACC Component Sources")
            st.dataframe(pd.DataFrame(wacc_rows), use_container_width=True, hide_index=True)

            rd_guardrail_note = wacc_components.get("rd_guardrail_note")
            if rd_guardrail_note:
                st.caption(f"Rd guardrail: {rd_guardrail_note}")
        else:
            beta = _to_float(getattr(getattr(snapshot, "beta", None), "value", None))
            risk_free_rate = _to_float(getattr(snapshot, "risk_free_rate", None))
            market_risk_premium = 0.05
            rf_source_raw = getattr(snapshot, "rf_source", None)
            rf_source = str(rf_source_raw).strip() if rf_source_raw is not None else ""
            legacy_rf_fallback = False
            if risk_free_rate is None:
                risk_free_rate = 0.045
                legacy_rf_fallback = True
            if not rf_source or rf_source.lower() in {"none", "n/a", "na"}:
                rf_source = (
                    "Fallback default (legacy cache snapshot)"
                    if legacy_rf_fallback
                    else "10-Year Treasury"
                )

            capm_rows = [
                {"Input": "Risk-Free Rate (Rf)", "Value": _fmt_rate(risk_free_rate), "Source": rf_source},
                {"Input": "Equity Risk Premium (ERP)", "Value": _fmt_rate(market_risk_premium), "Source": "Damodaran implied ERP"},
                {"Input": "Beta (beta)", "Value": _fmt_number(beta, 2), "Source": "Yahoo Finance (5Y monthly)"},
            ]

            st.markdown("#### Discount Rate Inputs")
            st.dataframe(pd.DataFrame(capm_rows), use_container_width=True, hide_index=True)
            if legacy_rf_fallback:
                st.caption("This run came from older cached data missing Rf metadata. Using a 4.5% fallback here; rerun DCF to restore full WACC trace.")

    st.markdown("### 3. Forecast and Present Value")
    fcff_method = assumptions.get("fcff_method")
    fcff_reliability = assumptions.get("fcff_reliability")
    fcff_method_labels = {
        "proper_fcff": "Proper FCFF",
        "approx_unlevered": "Approximate Unlevered FCFF",
        "unlevered_proxy": "Approximate Unlevered FCFF",
        "levered_proxy": "Levered FCF Proxy",
    }
    method_label = fcff_method_labels.get(fcff_method, "Unknown")

    if fcff_reliability is not None:
        st.caption(
            "Driver chain: Revenue -> EBIT -> NOPAT -> Reinvestment -> FCFF. "
            f"FCFF basis: {method_label} ({fcff_reliability}/100 reliability)."
        )
    else:
        st.caption(
            "Driver chain: Revenue -> EBIT -> NOPAT -> Reinvestment -> FCFF. "
            f"FCFF basis: {method_label}."
        )

    growth_summary_rows = [
        {"Parameter": "Near-Term Growth", "Value": _fmt_rate(assumptions.get("near_term_growth_rate"))},
        {"Parameter": "Effective Near-Term Growth", "Value": _fmt_rate(assumptions.get("effective_near_term_growth_rate"))},
        {"Parameter": "Analyst Long-Term Growth Anchor", "Value": _fmt_rate(assumptions.get("analyst_long_term_growth_rate"))},
        {"Parameter": "Terminal Growth", "Value": _fmt_rate(assumptions.get("stable_growth_rate"))},
        {"Parameter": "Current ROIC", "Value": _fmt_rate(assumptions.get("base_roic"))},
        {"Parameter": "Terminal ROIC", "Value": _fmt_rate(assumptions.get("terminal_roic"))},
        {"Parameter": "Terminal Reinvestment Rate", "Value": _fmt_rate(assumptions.get("terminal_reinvestment_rate"))},
    ]
    st.dataframe(pd.DataFrame(growth_summary_rows), use_container_width=True, hide_index=True)

    projection_rows = []
    if yearly_projections:
        for proj in yearly_projections:
            year = proj.get("year")
            projection_rows.append(
                {
                    "Year": f"Y{int(year)}" if isinstance(year, (int, float)) else "N/A",
                    "Revenue": _fmt_money(proj.get("revenue")),
                    "Growth": _fmt_rate(proj.get("revenue_growth")),
                    "EBIT Margin": _fmt_rate(proj.get("ebit_margin")),
                    "EBIT": _fmt_money(proj.get("ebit")),
                    "NOPAT": _fmt_money(proj.get("nopat")),
                    "Reinvestment": _fmt_money(proj.get("reinvestment")),
                    "Reinvestment Rate": _fmt_rate(proj.get("reinvestment_rate")),
                    "FCFF": _fmt_money(proj.get("fcff")),
                    "PV(FCFF)": _fmt_money(proj.get("pv_fcff")),
                    "Revenue Source": proj.get("revenue_source", "driver_growth"),
                    "FCFF Source": proj.get("fcf_source", "driver_model"),
                }
            )
    elif fcf_projections:
        for proj in fcf_projections:
            year = proj.get("year")
            projection_rows.append(
                {
                    "Year": f"Y{int(year)}" if isinstance(year, (int, float)) else "N/A",
                    "FCFF": _fmt_money(proj.get("fcff", proj.get("fcf"))),
                    "Discount Factor": _fmt_number(proj.get("discount_factor"), 4),
                    "PV(FCFF)": _fmt_money(proj.get("pv")),
                }
            )

    if projection_rows:
        st.dataframe(pd.DataFrame(projection_rows), use_container_width=True, hide_index=True)
    else:
        st.warning("No forecast rows are available for this run.")

    st.markdown("### 4. Terminal Value")
    st.caption(f"Primary terminal method: {str(tv_method).replace('_', ' ').title()}")

    discount_factor = None
    if _to_float(wacc_used) is not None and forecast_years > 0:
        discount_factor = 1 / ((1 + float(wacc_used)) ** forecast_years)

    terminal_rows = []
    if tv_method == "exit_multiple":
        terminal_ebitda = assumptions.get("terminal_year_ebitda") or assumptions.get("projected_terminal_ebitda")
        terminal_rows = [
            {
                "Step": "Terminal EBITDA (Year N)",
                "Formula": "Projected EBITDA in final forecast year",
                "Value": _fmt_money(terminal_ebitda),
            },
            {
                "Step": "Exit Multiple",
                "Formula": "Selected EV/EBITDA terminal multiple",
                "Value": _fmt_multiple(assumptions.get("exit_multiple")),
            },
            {
                "Step": "Terminal Value (Year N)",
                "Formula": "Terminal EBITDA * Exit Multiple",
                "Value": _fmt_money(terminal_value_yearN),
            },
            {
                "Step": "Discount Factor",
                "Formula": "1 / (1 + WACC)^N",
                "Value": _fmt_number(discount_factor, 4) if discount_factor is not None else "N/A",
            },
            {
                "Step": "PV(Terminal Value)",
                "Formula": "Terminal Value * Discount Factor",
                "Value": _fmt_money(pv_tv),
            },
        ]
    else:
        terminal_fcff = assumptions.get("terminal_year_fcff")
        if terminal_fcff is None and yearly_projections:
            terminal_fcff = yearly_projections[-1].get("fcff")
        if terminal_fcff is None and fcf_projections:
            terminal_fcff = fcf_projections[-1].get("fcff", fcf_projections[-1].get("fcf"))

        terminal_fcff_n1 = None
        if _to_float(terminal_fcff) is not None and _to_float(terminal_growth) is not None:
            terminal_fcff_n1 = float(terminal_fcff) * (1 + float(terminal_growth))

        terminal_rows = [
            {
                "Step": "FCFF (Year N)",
                "Formula": "Projected FCFF in final forecast year",
                "Value": _fmt_money(terminal_fcff),
            },
            {
                "Step": "Terminal Growth (g)",
                "Formula": "Long-run growth assumption",
                "Value": _fmt_rate(terminal_growth),
            },
            {
                "Step": "FCFF (Year N+1)",
                "Formula": "FCFF_N * (1 + g)",
                "Value": _fmt_money(terminal_fcff_n1),
            },
            {
                "Step": "Terminal Value (Year N)",
                "Formula": "FCFF_(N+1) / (WACC - g)",
                "Value": _fmt_money(terminal_value_yearN),
            },
            {
                "Step": "Discount Factor",
                "Formula": "1 / (1 + WACC)^N",
                "Value": _fmt_number(discount_factor, 4) if discount_factor is not None else "N/A",
            },
            {
                "Step": "PV(Terminal Value)",
                "Formula": "Terminal Value * Discount Factor",
                "Value": _fmt_money(pv_tv),
            },
        ]

    st.dataframe(pd.DataFrame(terminal_rows), use_container_width=True, hide_index=True)

    st.markdown("#### Terminal Method Cross-Check")
    cross_check_rows = [
        {
            "Method": "Gordon Growth",
            "TV (Year N)": _fmt_money(assumptions.get("tv_gordon_growth")),
            "PV(TV)": _fmt_money(assumptions.get("pv_tv_gordon_growth")),
            "Implied Price": (
                f"${_to_float(assumptions.get('price_gordon_growth')):.2f}"
                if _to_float(assumptions.get("price_gordon_growth")) is not None
                else "N/A"
            ),
            "Used in EV": "Yes" if tv_method == "gordon_growth" else "No",
        },
        {
            "Method": "Exit Multiple",
            "TV (Year N)": _fmt_money(assumptions.get("tv_exit_multiple")),
            "PV(TV)": _fmt_money(assumptions.get("pv_tv_exit_multiple")),
            "Implied Price": (
                f"${_to_float(assumptions.get('price_exit_multiple')):.2f}"
                if _to_float(assumptions.get("price_exit_multiple")) is not None
                else "N/A"
            ),
            "Used in EV": "Yes" if tv_method == "exit_multiple" else "No",
        },
    ]
    st.dataframe(pd.DataFrame(cross_check_rows), use_container_width=True, hide_index=True)

    exit_scenarios = assumptions.get("exit_multiple_scenarios", []) or []
    if exit_scenarios:
        st.markdown("#### Exit Multiple Scenarios")
        scenario_rows = []
        for scenario in exit_scenarios:
            scenario_rows.append(
                {
                    "Scenario": scenario.get("name", "N/A"),
                    "Multiple": _fmt_multiple(scenario.get("multiple")),
                    "Implied Price": (
                        f"${_to_float(scenario.get('price')):.0f}"
                        if _to_float(scenario.get("price")) is not None
                        else "N/A"
                    ),
                    "Required FCFF/EBITDA": _fmt_rate(scenario.get("required_fcff_ebitda"), 0),
                    "Gap vs Forecast": scenario.get("gap", "N/A"),
                    "Status": scenario.get("status", "N/A"),
                }
            )
        st.dataframe(pd.DataFrame(scenario_rows), use_container_width=True, hide_index=True)

    st.markdown("### 5. Enterprise to Equity Bridge")
    bridge_rows = [
        {
            "Step": "PV of Explicit FCFF",
            "Formula": "Sum of discounted yearly FCFF",
            "Value": _fmt_money(pv_fcf_sum),
        },
        {
            "Step": "PV of Terminal Value",
            "Formula": "Discounted terminal value",
            "Value": _fmt_money(pv_tv),
        },
        {
            "Step": "Enterprise Value",
            "Formula": "PV(FCFF) + PV(TV)",
            "Value": _fmt_money(ev),
        },
        {
            "Step": "Net Debt",
            "Formula": "Total Debt - Cash and Equivalents",
            "Value": _fmt_money(net_debt),
        },
        {
            "Step": "Equity Value",
            "Formula": "Enterprise Value - Net Debt",
            "Value": _fmt_money(equity),
        },
        {
            "Step": "Intrinsic Value / Share",
            "Formula": "Equity Value / Shares Outstanding",
            "Value": f"${_to_float(price_per_share):.2f}" if _to_float(price_per_share) is not None else "N/A",
        },
    ]
    st.dataframe(pd.DataFrame(bridge_rows), use_container_width=True, hide_index=True)

    calc_ev = (_to_float(pv_fcf_sum) or 0.0) + (_to_float(pv_tv) or 0.0)
    calc_equity = (_to_float(ev) or 0.0) - (_to_float(net_debt) or 0.0)
    calc_price = calc_equity / _to_float(shares_outstanding) if _to_float(shares_outstanding) else None

    ev_delta = (_to_float(ev) - calc_ev) if _to_float(ev) is not None else None
    equity_delta = (_to_float(equity) - calc_equity) if _to_float(equity) is not None else None
    price_delta = (
        (_to_float(price_per_share) - calc_price)
        if (_to_float(price_per_share) is not None and calc_price is not None)
        else None
    )

    def _delta_status(delta, scale):
        if delta is None:
            return "N/A"
        tolerance = max(1.0, abs(scale) * 0.001)
        return "PASS" if abs(delta) <= tolerance else "CHECK"

    reconciliation_rows = [
        {
            "Check": "EV = PV(FCFF) + PV(TV)",
            "Computed": _fmt_money(calc_ev),
            "Reported": _fmt_money(ev),
            "Delta": _fmt_money(ev_delta),
            "Status": _delta_status(ev_delta, _to_float(ev) or calc_ev),
        },
        {
            "Check": "Equity = EV - Net Debt",
            "Computed": _fmt_money(calc_equity),
            "Reported": _fmt_money(equity),
            "Delta": _fmt_money(equity_delta),
            "Status": _delta_status(equity_delta, _to_float(equity) or calc_equity),
        },
        {
            "Check": "Price = Equity / Shares",
            "Computed": f"${calc_price:.4f}" if calc_price is not None else "N/A",
            "Reported": f"${_to_float(price_per_share):.4f}" if _to_float(price_per_share) is not None else "N/A",
            "Delta": f"${price_delta:.4f}" if price_delta is not None else "N/A",
            "Status": _delta_status(price_delta, _to_float(price_per_share) or (calc_price or 0.0)),
        },
    ]
    st.markdown("#### Reconciliation Checks")
    st.dataframe(pd.DataFrame(reconciliation_rows), use_container_width=True, hide_index=True)

    st.markdown("### 6. Warnings, Diagnostics, and Trace")
    errors = [err for err in ui_data.get("errors", []) if err]
    warnings = [warn for warn in ui_data.get("warnings", []) if warn]
    diagnostics = [diag for diag in ui_data.get("diagnostics", []) if diag]
    grouped_notices = {"critical": [], "review": [], "info": []}
    seen_notices = set()

    for message in errors:
        cleaned = _sanitize_notice_text(message)
        key = ("critical", cleaned)
        if cleaned and key not in seen_notices:
            grouped_notices["critical"].append(cleaned)
            seen_notices.add(key)

    for message in warnings:
        cleaned = _sanitize_notice_text(message)
        severity = _classify_notice(cleaned, "warning")
        key = (severity, cleaned)
        if cleaned and key not in seen_notices:
            grouped_notices[severity].append(cleaned)
            seen_notices.add(key)

    for message in diagnostics:
        cleaned = _sanitize_notice_text(message)
        severity = _classify_notice(cleaned, "diagnostic")
        key = (severity, cleaned)
        if cleaned and key not in seen_notices:
            grouped_notices[severity].append(cleaned)
            seen_notices.add(key)

    total_notices = sum(len(items) for items in grouped_notices.values())
    if total_notices > 0:
        c_count = len(grouped_notices["critical"])
        r_count = len(grouped_notices["review"])
        i_count = len(grouped_notices["info"])
        st.caption(f"Organized by severity: Critical {c_count} | Review {r_count} | Info {i_count}")

        if grouped_notices["critical"]:
            st.error("Critical Checks")
            for idx, item in enumerate(grouped_notices["critical"], start=1):
                st.markdown(f"{idx}. {item}")

        if grouped_notices["review"]:
            st.warning("Review Items")
            for idx, item in enumerate(grouped_notices["review"], start=1):
                st.markdown(f"{idx}. {item}")

        if grouped_notices["info"]:
            st.info("Model Notes")
            for idx, item in enumerate(grouped_notices["info"], start=1):
                st.markdown(f"{idx}. {item}")
    else:
        st.caption("No warnings or diagnostics were returned for this run.")

    if trace_steps:
        trace_rows = []
        for idx, step in enumerate(trace_steps, start=1):
            output = step.get("output")
            output_units = step.get("output_units")
            if isinstance(output, (int, float)) and output_units == "USD":
                output_text = _fmt_money(output)
            elif isinstance(output, (int, float)):
                output_text = _fmt_number(output, 4)
            else:
                output_text = str(output) if output is not None else "N/A"

            inputs = step.get("inputs", {})
            input_keys = ", ".join(inputs.keys()) if isinstance(inputs, dict) and inputs else "N/A"
            trace_rows.append(
                {
                    "Step #": idx,
                    "Name": _sanitize_notice_text(step.get("name", "N/A") or "N/A"),
                    "Formula": step.get("formula", "N/A") or "N/A",
                    "Input Keys": input_keys,
                    "Output": output_text,
                    "Notes": _sanitize_notice_text(step.get("notes", "N/A") or "N/A"),
                }
            )
        st.dataframe(pd.DataFrame(trace_rows), use_container_width=True, hide_index=True)
    else:
        st.caption("No trace steps available.")

    trace_payload = {
        "inputs": {k: str(v) for k, v in ui_data.get("inputs", {}).items()},
        "assumptions": assumptions,
        "results": {
            "enterprise_value": ev,
            "equity_value": equity,
            "price_per_share": price_per_share,
            "pv_fcf_sum": pv_fcf_sum,
            "pv_terminal_value": pv_tv,
        },
        "trace_steps": trace_steps,
    }
    trace_json = json.dumps(trace_payload, indent=2, default=str)
    st.download_button(
        label="Download Trace (JSON)",
        data=trace_json,
        file_name=f"{st.session_state.get('ticker', 'valuation')}_dcf_trace.json",
        mime="application/json",
    )

    st.markdown("---")
    st.markdown(
        '<div class="qa-chatbot-hero"><span class="qa-chatbot-pill">Q&A Chatbot</span></div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Ask where any number came from. Answers are grounded in this run's inputs, assumptions, "
        "WACC components, projections, and trace steps."
    )

    details_ticker = _normalize_ticker(
        st.session_state.get("ticker") or getattr(snapshot, "ticker", "UNKNOWN")
    ) or "UNKNOWN"
    details_chat_key = f"dcf_details_chat_history_{details_ticker}"
    details_chat_input_key = f"dcf_details_chat_input_{details_ticker}"

    if details_chat_key not in st.session_state or not isinstance(st.session_state[details_chat_key], list):
        st.session_state[details_chat_key] = []

    chat_history = st.session_state[details_chat_key]

    col_chat_meta, col_chat_clear = st.columns([4, 1])
    with col_chat_meta:
        st.caption("Assistant responses can make mistakes. Verify against the trace and source columns.")
    with col_chat_clear:
        if st.button("Clear Q&A", key=f"clear_{details_chat_key}"):
            st.session_state[details_chat_key] = []
            st.rerun()

    for msg in chat_history[-10:]:
        role = msg.get("role", "assistant")
        content = msg.get("content", "")
        with st.chat_message(role):
            st.markdown(content)

    user_question = ""
    submitted = False
    with st.form(key=f"dcf_details_chat_form_{details_ticker}", clear_on_submit=True):
        user_question = st.text_input(
            "Ask a clarification question",
            placeholder="Example: Why is PV(TV) 47.8% of EV? Where does Rd come from?",
            key=details_chat_input_key,
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Ask")

    if submitted and user_question:
        context_packet = {
            "ticker": details_ticker,
            "company_name": getattr(snapshot, "company_name", None) if snapshot else None,
            "inputs_table": input_table,
            "assumptions_table": assumption_rows,
            "growth_summary": growth_summary_rows,
            "results": {
                "enterprise_value": ev,
                "equity_value": equity,
                "price_per_share": price_per_share,
                "pv_fcf_sum": pv_fcf_sum,
                "pv_terminal_value": pv_tv,
                "terminal_value_yearN": terminal_value_yearN,
                "net_debt": net_debt,
                "shares_outstanding": shares_outstanding,
                "implied_fcf_yield": ui_data.get("implied_fcf_yield"),
                "implied_ev_fcf": ui_data.get("implied_ev_fcf"),
                "discount_rate_used": assumptions.get("discount_rate_used"),
                "wacc_input": assumptions.get("wacc"),
                "terminal_growth_rate": assumptions.get("terminal_growth_rate"),
            },
            "wacc_components": getattr(snapshot, "wacc_components", {}) if snapshot else {},
            "projection_rows": projection_rows[:40],
            "terminal_rows": terminal_rows,
            "bridge_rows": bridge_rows,
            "reconciliation_rows": reconciliation_rows,
            "trace_steps": trace_steps[:120],
        }

        context_data = json.dumps(context_packet, indent=2, default=str)
        trace_prompt = (
            "User question: " + user_question + "\n\n"
            "Answer rules:\n"
            "1) Use only the provided DCF context.\n"
            "2) Return ONLY valid JSON (no markdown, no prose outside JSON) with this exact schema:\n"
            "{\n"
            "  \"direct_answer\": \"one-sentence direct answer\",\n"
            "  \"value\": \"specific numeric value(s) used\",\n"
            "  \"formula_path\": \"equation or calculation chain\",\n"
            "  \"source_path\": \"table/field/trace step names and source labels\"\n"
            "}\n"
            "3) If missing, be explicit and provide nearest available fields in source_path.\n"
            "4) Do not provide investment advice.\n"
        )

        prior_history = [m for m in chat_history if isinstance(m, dict) and m.get("role") in {"user", "assistant"}]

        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Tracing value..."):
                assistant_raw = run_chat(prior_history, trace_prompt, context_data)
                raw_lower = str(assistant_raw or "").lower()
                if raw_lower.startswith("chat error:") or raw_lower.startswith("error:"):
                    assistant_reply = _format_chat_runtime_message(assistant_raw)
                else:
                    assistant_reply = _format_trace_chat_response(assistant_raw)
            st.markdown(assistant_reply)

        st.session_state[details_chat_key].append({"role": "user", "content": user_question})
        st.session_state[details_chat_key].append({"role": "assistant", "content": assistant_reply})

    st.divider()
    if st.button("<- Back to Summary", key="back_from_details_bottom"):
        st.session_state.show_dcf_details = False
        st.rerun()


def _show_user_guide_page():
    """Render standalone user guide so the main analysis page stays uncluttered."""
    st.markdown("---")
    st.markdown(
        '<div class="section-header"><span class="step-badge">Guide</span><span class="section-title">User Guide</span></div>',
        unsafe_allow_html=True,
    )
    st.caption("How to run the workflow, interpret outputs, and avoid common mistakes.")

    if st.button("<- Back to Analysis", key="back_from_user_guide_top"):
        st.session_state.show_user_guide = False
        st.rerun()

    st.markdown(
        """
<div class="guide-shell">
  <div class="guide-hero">
    <div>
      <div class="guide-kicker">Instruction Manual</div>
      <div class="guide-hero-title">How to Use Analyst Co-Pilot</div>
      <div class="guide-hero-sub">Use this workflow to go from raw financials to assumption-based valuation decisions with full traceability.</div>
    </div>
    <div class="guide-hero-chip">Research Workflow</div>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    col_qs, col_pw = st.columns(2)
    with col_qs:
        st.markdown(
            """
<div class="guide-card">
  <div class="guide-card-title">Quick Start</div>
  <ol class="guide-list guide-list-numbered">
    <li>Open the sidebar and choose a ticker.</li>
    <li>Select ending report date and historical quarter count.</li>
    <li>Click <code>Load Data</code>.</li>
    <li>In <code>Step 02</code>, set assumptions and run <code>Run DCF Analysis</code>.</li>
    <li>Review <code>Step 01</code> through <code>Step 06</code> in order.</li>
    <li>Generate <code>Step 05</code> AI Synthesis after DCF is complete.</li>
  </ol>
</div>
            """,
            unsafe_allow_html=True,
        )
    with col_pw:
        st.markdown(
            """
<div class="guide-card">
  <div class="guide-card-title">Practical Workflow</div>
  <ol class="guide-list guide-list-numbered">
    <li>Start from suggested assumptions for a base case.</li>
    <li>Create bull/base/bear scenarios by changing one input at a time.</li>
    <li>Check whether the conclusion survives realistic ranges.</li>
    <li>Use <code>View DCF Details</code> to audit formulas and lineage.</li>
  </ol>
</div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
<div class="guide-card">
  <div class="guide-card-title">Step Map</div>
  <div class="guide-step-grid">
    <div class="guide-step-row"><span class="guide-step-pill">Step 01</span><div><div class="guide-step-name">Investment Verdict</div><div class="guide-step-desc">Primary valuation call from current assumptions.</div></div></div>
    <div class="guide-step-row"><span class="guide-step-pill">Step 02</span><div><div class="guide-step-name">Valuation Drivers</div><div class="guide-step-desc">Input controls, model reruns, and core valuation outputs.</div></div></div>
    <div class="guide-step-row"><span class="guide-step-pill">Step 03</span><div><div class="guide-step-name">Business Momentum</div><div class="guide-step-desc">Historical trend context for operating performance.</div></div></div>
    <div class="guide-step-row"><span class="guide-step-pill">Step 04</span><div><div class="guide-step-name">Street Context</div><div class="guide-step-desc">Analyst expectations and target distribution context.</div></div></div>
    <div class="guide-step-row"><span class="guide-step-pill">Step 05</span><div><div class="guide-step-name">AI Synthesis</div><div class="guide-step-desc">Narrative interpretation grounded in model + market inputs.</div></div></div>
    <div class="guide-step-row"><span class="guide-step-pill">Step 06</span><div><div class="guide-step-name">Sources & Methodology</div><div class="guide-step-desc">Reference links and method notes for verification.</div></div></div>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    col_rules, col_dq = st.columns(2)
    with col_rules:
        st.markdown(
            """
<div class="guide-card">
  <div class="guide-card-title">DCF Interpretation Rules</div>
  <ul class="guide-list">
    <li>Intrinsic value is model-implied under assumptions, not a guaranteed level.</li>
    <li>Terminal assumptions can dominate EV, so monitor <code>TV Dominance</code>.</li>
    <li>Gordon Growth is usually default for mature steady-state cases.</li>
    <li>Exit Multiple can be used adaptively for high-growth or punitive Gordon outcomes.</li>
    <li>Interpret results through sensitivity, not a single point estimate.</li>
  </ul>
</div>
            """,
            unsafe_allow_html=True,
        )
    with col_dq:
        st.markdown(
            """
<div class="guide-card">
  <div class="guide-card-title">Data Quality and Common Mistakes</div>
  <ul class="guide-list">
    <li>Treat <code>Data Quality</code> as a confidence lens, not pass/fail.</li>
    <li>Lower reliability can materially shift valuation conclusions.</li>
    <li>Avoid setting terminal growth too close to or above WACC.</li>
    <li>Do not compare outputs across dates without reloading context.</li>
  </ul>
</div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
<div class="guide-disclaimer-card">
  <div class="guide-disclaimer-title">Important Disclaimer</div>
  <div class="guide-disclaimer-text">This tool supports research workflows only. Outputs may contain mistakes and are highly assumption-dependent. It is not investment advice.</div>
</div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("<- Back to Analysis", key="back_from_user_guide_bottom"):
        st.session_state.show_user_guide = False
        st.rerun()
# --- App Configuration ---
st.set_page_config(
    page_title="Analyst Co-Pilot",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"  # Start collapsed for expanded view
)

# --- Design System CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ═══════════════════════════════════════════
       DESIGN TOKENS
    ═══════════════════════════════════════════ */
    :root {
      --clr-bg:             #f8fafc;
      --clr-surface:        #ffffff;
      --clr-sidebar-bg:     #0d1630;
      --clr-sidebar-text:   #e2e8f0;
      --clr-sidebar-muted:  #94a3b8;
      --clr-accent:         #2563eb;
      --clr-accent-hover:   #1d4ed8;
      --clr-success:        #10b981;
      --clr-success-bg:     #ecfdf5;
      --clr-success-text:   #065f46;
      --clr-danger:         #f43f5e;
      --clr-danger-bg:      #fff1f2;
      --clr-danger-text:    #9f1239;
      --clr-warn:           #f59e0b;
      --clr-warn-bg:        #fffbeb;
      --clr-warn-text:      #92400e;
      --clr-text-primary:   #0f172a;
      --clr-text-secondary: #475569;
      --clr-text-muted:     #94a3b8;
      --clr-border:         #e2e8f0;
      --clr-border-strong:  #cbd5e1;
      --shadow-sm:  0 1px 3px rgba(0,0,0,0.08);
      --shadow-md:  0 4px 12px rgba(0,0,0,0.10);
      --radius-sm: 4px;
      --radius-md: 8px;
      --radius-lg: 12px;
    }

    /* QW-1: Global background */
    .stApp {
        background: var(--clr-bg) !important;
    }

    /* QW-2: Dark navy sidebar */
    [data-testid="stSidebar"] {
        background: var(--clr-sidebar-bg) !important;
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: var(--clr-sidebar-text) !important;
    }
    /* Input container wrapper (this is what shows the white bg) */
    [data-testid="stSidebar"] [data-baseweb="input"],
    [data-testid="stSidebar"] [data-baseweb="base-input"],
    [data-testid="stSidebar"] .stTextInput > div > div {
        background: rgba(255,255,255,0.10) !important;
        border: 1px solid rgba(255,255,255,0.20) !important;
        border-radius: var(--radius-sm) !important;
    }
    /* Remove BaseWeb's red/blue focus outline; replace with accent glow */
    [data-testid="stSidebar"] [data-baseweb="input"]:focus-within,
    [data-testid="stSidebar"] .stTextInput > div > div:focus-within {
        border-color: var(--clr-accent) !important;
        box-shadow: 0 0 0 2px rgba(37,99,235,0.35) !important;
        outline: none !important;
    }
    /* Inner input element — transparent bg so container colour shows */
    [data-testid="stSidebar"] [data-baseweb="input"] input,
    [data-testid="stSidebar"] input {
        background: transparent !important;
        color: var(--clr-sidebar-text) !important;
        caret-color: var(--clr-sidebar-text) !important;
        border: none !important;
        box-shadow: none !important;
    }
    [data-testid="stSidebar"] [data-baseweb="input"] input::placeholder,
    [data-testid="stSidebar"] input::placeholder {
        color: var(--clr-sidebar-muted) !important;
        opacity: 1 !important;
    }
    /* "Press Enter to apply" helper text */
    [data-testid="stSidebar"] [data-testid="InputInstructions"],
    [data-testid="stSidebar"] .stTextInput small,
    [data-testid="stSidebar"] .stTextInput [class*="instructions"] {
        color: var(--clr-sidebar-muted) !important;
    }
    /* Selectbox */
    [data-testid="stSidebar"] [data-baseweb="select"],
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background: rgba(255,255,255,0.10) !important;
        border: 1px solid rgba(255,255,255,0.20) !important;
        border-radius: var(--radius-sm) !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] [data-baseweb="selected-option"],
    [data-testid="stSidebar"] [data-baseweb="select"] [data-baseweb="placeholder"],
    [data-testid="stSidebar"] [data-baseweb="select"] svg {
        color: var(--clr-sidebar-text) !important;
        fill: var(--clr-sidebar-text) !important;
    }
    [data-testid="stSidebar"] [data-testid="stSliderThumb"],
    [data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"] {
        background: var(--clr-accent) !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.1) !important;
    }
    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] small {
        color: var(--clr-sidebar-muted) !important;
    }
    [data-testid="stSidebar"] .stButton > button[kind="primary"],
    [data-testid="stSidebar"] .stButton > button[data-testid="baseButton-primary"] {
        background: var(--clr-accent) !important;
        border: none !important;
        color: #fff !important;
    }
    [data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
        background: var(--clr-accent-hover) !important;
    }
    [data-testid="stSidebar"] .stButton > button[kind="secondary"],
    [data-testid="stSidebar"] .stButton > button[data-testid="baseButton-secondary"] {
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        color: var(--clr-sidebar-text) !important;
    }
    [data-testid="stSidebar"] .stButton > button {
        color: var(--clr-sidebar-text) !important;
    }

    /* QW-3: Metric tile cards */
    [data-testid="metric-container"] {
        background: var(--clr-surface) !important;
        border: 1px solid var(--clr-border) !important;
        border-radius: var(--radius-md) !important;
        padding: 14px 16px !important;
        box-shadow: var(--shadow-sm) !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.68rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
        color: var(--clr-text-muted) !important;
    }
    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 1.35rem !important;
        font-weight: 600 !important;
        color: var(--clr-text-primary) !important;
    }
    [data-testid="stMetricDelta"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.78rem !important;
    }

    /* QW-4: Section divider upgrade */
    hr {
        border: none !important;
        border-top: 2px solid var(--clr-border) !important;
        margin: 2rem 0 !important;
    }

    /* QW-5: Typography hierarchy */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: var(--clr-text-primary) !important;
    }
    h1 { font-size: 1.5rem !important; font-weight: 700 !important; }
    h2 { font-size: 1.15rem !important; font-weight: 700 !important; text-transform: uppercase; letter-spacing: 0.04em; }
    h3 { font-size: 0.95rem !important; font-weight: 700 !important; }
    .stApp, .stApp p, .stApp div, .stApp label {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    .stApp [class*="material-symbols"],
    .stApp .material-symbols-outlined,
    .stApp .material-symbols-rounded {
        font-family: "Material Symbols Rounded", "Material Symbols Outlined", "Material Icons" !important;
        font-style: normal;
        line-height: 1;
    }
    .stApp [data-testid="stExpanderToggleIcon"],
    .stApp [data-testid="stExpanderToggleIcon"] * {
        font-family: "Material Symbols Rounded", "Material Symbols Outlined", "Material Icons" !important;
        font-style: normal !important;
        font-weight: 400 !important;
        line-height: 1 !important;
    }
    .stCaption { color: var(--clr-text-muted) !important; font-size: 0.78rem !important; }

    /* QW-6: Call-box token migration */
    .call-outperform {
        background: var(--clr-success-bg) !important;
        border: 1px solid var(--clr-success) !important;
        color: var(--clr-success-text) !important;
        padding: 1rem;
        border-radius: var(--radius-md);
        text-align: center;
    }
    .call-underperform {
        background: var(--clr-danger-bg) !important;
        border: 1px solid var(--clr-danger) !important;
        color: var(--clr-danger-text) !important;
        padding: 1rem;
        border-radius: var(--radius-md);
        text-align: center;
    }
    .call-inline {
        background: var(--clr-bg) !important;
        border: 1px solid var(--clr-border-strong) !important;
        color: var(--clr-text-secondary) !important;
        padding: 1rem;
        border-radius: var(--radius-md);
        text-align: center;
    }
    .call-label {
        font-size: 0.875rem;
        font-weight: 700;
        letter-spacing: 0.05em;
    }

    /* QW-7: Expander modernization */
    [data-testid="stExpander"] {
        background: var(--clr-surface) !important;
        border: 1px solid var(--clr-border) !important;
        border-radius: var(--radius-md) !important;
        margin-top: 1rem !important;
        box-shadow: var(--shadow-sm) !important;
    }
    [data-testid="stExpander"] summary {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        list-style: none;
        padding: 0.75rem 1rem !important;
        font-weight: 600;
        color: var(--clr-text-primary) !important;
        cursor: pointer;
        transition: background 0.15s ease;
    }
    [data-testid="stExpander"] summary:hover {
        background: #f8fafc !important;
    }
    [data-testid="stExpander"] summary:focus-visible {
        outline: 2px solid var(--clr-accent) !important;
        outline-offset: 2px;
    }
    [data-testid="stExpander"] [data-testid="stExpanderToggleIcon"] {
        color: var(--clr-accent) !important;
    }

    /* QW-8: DataFrame finance density */
    [data-testid="stDataFrame"] th {
        font-size: 0.68rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        color: var(--clr-text-muted) !important;
        background: var(--clr-bg) !important;
        padding: 6px 10px !important;
    }
    [data-testid="stDataFrame"] td {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.8rem !important;
        padding: 5px 10px !important;
        color: var(--clr-text-primary) !important;
    }
    [data-testid="stDataFrame"] tr:hover td {
        background: #eff6ff !important;
    }

    /* QW-9: Buttons + alert boxes */
    .stButton > button {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        border-radius: var(--radius-sm) !important;
        transition: background 0.15s ease !important;
    }
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="baseButton-primary"] {
        background: var(--clr-accent) !important;
        border: none !important;
        color: #fff !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: var(--clr-accent-hover) !important;
    }
    [data-testid="stAlert"] {
        border-radius: var(--radius-md) !important;
    }

    /* QW-10: Column gap */
    [data-testid="column"] { padding: 0 8px !important; }
    .stMarkdown br { display: block; margin: 0.25rem 0; }

    /* ═══════════════════════════════════════════
       PHASE 2 UTILITY CLASSES
    ═══════════════════════════════════════════ */

    /* Section header with step badge */
    .section-header {
        display: flex; align-items: center; gap: 12px;
        margin-bottom: 1.25rem; padding: 0.5rem 0;
    }
    .step-badge {
        font-size: .65rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: .1em; color: var(--clr-accent); background: #eff6ff;
        border: 1px solid #bfdbfe; padding: 3px 8px; border-radius: 4px;
    }
    .section-title { font-size: 1.05rem; font-weight: 700; color: var(--clr-text-primary); }

    /* Unified badge system */
    .badge {
        display: inline-block; padding: 3px 10px; border-radius: var(--radius-sm);
        font-size: .7rem; font-weight: 700; letter-spacing: .06em; text-transform: uppercase;
    }
    .badge-pass  { background: var(--clr-success-bg); color: var(--clr-success-text); border: 1px solid var(--clr-success); }
    .badge-warn  { background: var(--clr-warn-bg);    color: var(--clr-warn-text);    border: 1px solid var(--clr-warn); }
    .badge-fail  { background: var(--clr-danger-bg);  color: var(--clr-danger-text);  border: 1px solid var(--clr-danger); }
    .badge-neutral-plain { background: var(--clr-bg); color: var(--clr-text-secondary); border: 1px solid var(--clr-border-strong); }

    /* Prominent disclaimer banner */
    .disclaimer-banner {
        background: var(--clr-warn-bg);
        border: 1px solid #fcd34d;
        border-left: 4px solid var(--clr-warn);
        border-radius: var(--radius-md);
        padding: 10px 12px;
        margin-bottom: 12px;
    }
    .disclaimer-title {
        font-size: .7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .08em;
        color: var(--clr-warn-text);
        margin-bottom: 4px;
    }
    .disclaimer-text {
        font-size: .8rem;
        color: #78350f;
        line-height: 1.35;
    }

    /* Stance cards */
    .stance-card {
        background: var(--clr-surface); border: 1px solid var(--clr-border);
        border-radius: var(--radius-md); padding: 14px 16px; border-left-width: 3px;
        border-left-style: solid;
    }
    .stance-card-bull { border-left-color: var(--clr-success) !important; }
    .stance-card-bear { border-left-color: var(--clr-danger) !important; }
    .stance-card-neut { border-left-color: var(--clr-text-muted) !important; }

    /* Chips */
    .input-chip {
        background: #eff6ff; border: 1px solid #bfdbfe; color: var(--clr-text-secondary);
        padding: 3px 10px; border-radius: 12px; font-size: .78rem;
        font-family: 'JetBrains Mono', monospace; display: inline-block;
    }
    .param-chip {
        background: var(--clr-bg); border: 1px solid var(--clr-border);
        color: var(--clr-text-secondary); padding: 3px 10px; border-radius: 12px;
        font-size: .78rem; display: inline-block;
    }

    /* Subsection header */
    .subsection-header {
        font-size: .85rem; font-weight: 700; color: var(--clr-text-primary);
        text-transform: uppercase; letter-spacing: .06em;
        padding-bottom: .5rem; border-bottom: 1px solid var(--clr-border);
        margin: 1.5rem 0 .75rem 0;
    }

    /* Sidebar data badge */
    .sidebar-data-badge {
        background: rgba(16,185,129,.12); border: 1px solid rgba(16,185,129,.3);
        border-radius: var(--radius-sm); padding: 6px 12px; text-align: center;
        font-size: .78rem; font-weight: 600; color: #6ee7b7; letter-spacing: .02em;
        margin: 8px 0;
    }

    /* Spacers */
    .spacer-sm { height: 1rem; }
    .spacer-md { height: 1.75rem; }

    /* App top bar */
    .app-topbar {
        display: flex; align-items: center; justify-content: space-between;
        padding: 12px 0; border-bottom: 2px solid var(--clr-border); margin-bottom: 1.5rem;
    }
    .app-wordmark {
        font-size: 1.1rem; font-weight: 700; color: var(--clr-text-primary); letter-spacing: -.01em;
    }
    .app-version {
        font-size: .6rem; font-weight: 700; text-transform: uppercase; letter-spacing: .08em;
        color: var(--clr-accent); background: #eff6ff; border: 1px solid #bfdbfe;
        padding: 2px 6px; border-radius: 3px; margin-left: 8px; vertical-align: middle;
    }
    .app-tagline { font-size: .78rem; color: var(--clr-text-muted); }

    /* Hero KPI strip */
    .hero-strip {
        background: var(--clr-surface); border: 1px solid var(--clr-border);
        border-radius: var(--radius-lg); padding: 16px 24px;
        display: flex; align-items: center; margin-bottom: 1.25rem;
        box-shadow: var(--shadow-sm);
    }
    .hero-item { flex: 1; padding: 0 20px; border-right: 1px solid var(--clr-border); }
    .hero-item:first-child { padding-left: 0; }
    .hero-label {
        font-size: .65rem; font-weight: 600; text-transform: uppercase;
        letter-spacing: .07em; color: var(--clr-text-muted); margin-bottom: 4px;
    }
    .hero-value {
        font-size: 1.3rem; font-weight: 700; font-family: 'JetBrains Mono', monospace;
        color: var(--clr-text-primary);
    }
    .hero-divider { width: 1px; background: var(--clr-border); height: 48px; margin: 0 24px; flex-shrink: 0; }
    .hero-ticker { text-align: right; flex-shrink: 0; }
    .hero-ticker-symbol {
        display: block; font-size: 1.8rem; font-weight: 800; color: var(--clr-accent);
        font-family: 'JetBrains Mono', monospace; letter-spacing: -.02em;
    }
    .hero-ticker-source {
        font-size: .65rem; color: var(--clr-text-muted); text-transform: uppercase; letter-spacing: .06em;
    }

    /* Report navigation + decision strip */
    .report-nav {
        display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 1.25rem;
        padding: 10px 12px; border: 1px solid var(--clr-border);
        border-radius: var(--radius-md); background: var(--clr-surface);
        box-shadow: var(--shadow-sm);
    }
    .report-nav a {
        font-size: .72rem; font-weight: 700; letter-spacing: .05em;
        text-transform: uppercase; color: var(--clr-text-secondary);
        text-decoration: none; padding: 6px 10px; border-radius: var(--radius-sm);
        border: 1px solid var(--clr-border);
    }
    .report-nav a:hover {
        color: var(--clr-accent); border-color: #bfdbfe; background: #eff6ff;
    }
    .report-nav a.is-active {
        color: var(--clr-accent);
        border-color: #93c5fd;
        background: #dbeafe;
        box-shadow: inset 0 0 0 1px #bfdbfe;
    }
    .floating-toc-wrap {
        position: fixed;
        right: 18px;
        top: 86px;
        z-index: 999;
        display: flex;
        flex-direction: row-reverse;
        align-items: flex-start;
        gap: 8px;
        opacity: 0;
        transform: translateX(12px);
        pointer-events: none;
        transition: opacity 0.18s ease, transform 0.18s ease;
    }
    .floating-toc-wrap.is-visible {
        opacity: 1;
        transform: translateX(0);
        pointer-events: auto;
    }
    .floating-toc-toggle {
        width: 36px;
        height: 36px;
        border: 1px solid var(--clr-border);
        border-radius: 999px;
        background: var(--clr-surface);
        color: var(--clr-text-secondary);
        font-size: 1rem;
        font-weight: 700;
        line-height: 1;
        cursor: pointer;
        box-shadow: var(--shadow-sm);
    }
    .floating-toc-toggle:hover {
        color: var(--clr-accent);
        border-color: #bfdbfe;
        background: #eff6ff;
    }
    .floating-toc-wrap.is-expanded .floating-toc-toggle {
        color: var(--clr-accent);
        border-color: #93c5fd;
        background: #dbeafe;
    }
    .floating-toc {
        width: 196px;
        background: var(--clr-surface);
        border: 1px solid var(--clr-border);
        border-radius: var(--radius-md);
        box-shadow: var(--shadow-md);
        padding: 10px;
        display: none;
        flex-direction: column;
        gap: 6px;
    }
    .floating-toc-wrap.is-expanded .floating-toc {
        display: flex;
    }
    .floating-toc-title {
        font-size: .62rem;
        font-weight: 700;
        color: var(--clr-text-muted);
        text-transform: uppercase;
        letter-spacing: .08em;
        margin-bottom: 2px;
    }
    .floating-toc a {
        font-size: .66rem;
        font-weight: 700;
        letter-spacing: .05em;
        text-transform: uppercase;
        color: var(--clr-text-secondary);
        text-decoration: none;
        border: 1px solid var(--clr-border);
        border-radius: var(--radius-sm);
        background: var(--clr-bg);
        padding: 6px 8px;
    }
    .floating-toc a:hover {
        color: var(--clr-accent);
        border-color: #bfdbfe;
        background: #eff6ff;
    }
    .floating-toc a.is-active {
        color: var(--clr-accent);
        border-color: #93c5fd;
        background: #dbeafe;
        box-shadow: inset 0 0 0 1px #bfdbfe;
    }
    .decision-strip {
        background: var(--clr-surface); border: 1px solid var(--clr-border);
        border-radius: var(--radius-md); box-shadow: var(--shadow-sm);
        padding: 12px 14px; margin-bottom: 12px;
    }
    .decision-grid {
        display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 10px;
    }
    .decision-tile-label {
        font-size: .62rem; font-weight: 700; letter-spacing: .08em;
        text-transform: uppercase; color: var(--clr-text-muted);
    }
    .decision-tile-value {
        font-size: 1.05rem; font-weight: 700; color: var(--clr-text-primary);
        font-family: 'JetBrains Mono', monospace;
    }
    .confidence-strip {
        display: flex; flex-wrap: wrap; gap: 6px;
        font-size: .72rem; color: var(--clr-text-secondary);
    }
    .confidence-pill {
        background: var(--clr-bg); border: 1px solid var(--clr-border);
        border-radius: 999px; padding: 4px 10px;
    }
    .qa-chatbot-hero {
        margin: 2px 0 8px 0;
    }
    .qa-chatbot-pill {
        display: inline-block;
        padding: 10px 14px;
        border-radius: 10px;
        background: #1d4ed8;
        border: 1px solid #1e40af;
        color: #ffffff;
        font-weight: 700;
        font-size: 1.02rem;
        letter-spacing: .02em;
        box-shadow: 0 2px 10px rgba(29, 78, 216, 0.25);
    }

    /* User guide page */
    .guide-shell {
        display: flex;
        flex-direction: column;
        gap: 12px;
        margin: 10px 0 12px 0;
    }
    .guide-hero {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 16px;
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
        border: 1px solid #1d4ed8;
        border-radius: var(--radius-lg);
        padding: 16px 18px;
        color: #e2e8f0;
        box-shadow: var(--shadow-md);
    }
    .guide-kicker {
        font-size: .62rem;
        font-weight: 700;
        letter-spacing: .10em;
        text-transform: uppercase;
        color: #93c5fd;
        margin-bottom: 5px;
    }
    .guide-hero-title {
        font-size: 1.2rem;
        font-weight: 700;
        letter-spacing: -.01em;
        color: #f8fafc;
        margin-bottom: 5px;
    }
    .guide-hero-sub {
        font-size: .83rem;
        line-height: 1.4;
        color: #cbd5e1;
        max-width: 820px;
    }
    .guide-hero-chip {
        border: 1px solid rgba(147, 197, 253, 0.45);
        background: rgba(148, 163, 184, 0.15);
        color: #dbeafe;
        border-radius: 999px;
        padding: 6px 10px;
        font-size: .65rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .06em;
        white-space: nowrap;
    }
    .guide-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 12px;
    }
    .guide-card {
        background: var(--clr-surface);
        border: 1px solid var(--clr-border);
        border-radius: var(--radius-md);
        padding: 14px 16px;
        box-shadow: var(--shadow-sm);
    }
    .guide-card-title {
        font-size: .72rem;
        font-weight: 700;
        color: var(--clr-accent);
        text-transform: uppercase;
        letter-spacing: .08em;
        margin-bottom: 10px;
    }
    .guide-list {
        margin: 0;
        padding-left: 1.1rem;
        color: var(--clr-text-secondary);
        font-size: .84rem;
        line-height: 1.45;
    }
    .guide-list li {
        margin-bottom: 6px;
    }
    .guide-list-numbered li::marker {
        color: var(--clr-accent);
        font-weight: 700;
    }
    .guide-step-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 10px;
    }
    .guide-step-row {
        display: grid;
        grid-template-columns: auto 1fr;
        gap: 10px;
        align-items: start;
        border: 1px solid var(--clr-border);
        background: #f8fafc;
        border-radius: var(--radius-sm);
        padding: 9px 10px;
    }
    .guide-step-pill {
        display: inline-block;
        font-size: .62rem;
        font-weight: 700;
        letter-spacing: .08em;
        text-transform: uppercase;
        color: var(--clr-accent);
        background: #dbeafe;
        border: 1px solid #bfdbfe;
        border-radius: 999px;
        padding: 3px 8px;
        line-height: 1.4;
    }
    .guide-step-name {
        font-size: .82rem;
        font-weight: 700;
        color: var(--clr-text-primary);
        margin-bottom: 2px;
    }
    .guide-step-desc {
        font-size: .78rem;
        color: var(--clr-text-secondary);
        line-height: 1.35;
    }
    .guide-disclaimer-card {
        background: #fff7ed;
        border: 1px solid #fdba74;
        border-left: 4px solid #f97316;
        border-radius: var(--radius-md);
        padding: 12px 14px;
    }
    .guide-disclaimer-title {
        font-size: .66rem;
        font-weight: 700;
        letter-spacing: .08em;
        text-transform: uppercase;
        color: #9a3412;
        margin-bottom: 4px;
    }
    .guide-disclaimer-text {
        font-size: .82rem;
        color: #7c2d12;
        line-height: 1.35;
    }

    @media (max-width: 1024px) {
        .decision-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
        .floating-toc-wrap { right: 10px; }
        .floating-toc { width: 176px; }
        .guide-step-grid { grid-template-columns: 1fr; }
    }
    @media (max-width: 640px) {
        .decision-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        .hero-strip { flex-wrap: wrap; padding: 14px; }
        .hero-item {
            min-width: calc(50% - 10px);
            border-right: none;
            border-bottom: 1px solid var(--clr-border);
            padding: 0 8px 8px 0;
            margin-bottom: 8px;
        }
        .hero-divider { display: none; }
        .hero-ticker { width: 100%; text-align: left; }
        .floating-toc-wrap { display: none; }
        .guide-grid { grid-template-columns: 1fr; }
        .guide-hero { flex-direction: column; }
        .guide-hero-chip { align-self: flex-start; }
        .guide-step-row { grid-template-columns: 1fr; gap: 6px; }
    }

    /* Step progress indicator */
    .step-progress {
        display: flex; align-items: center; justify-content: center;
        gap: 0; margin-bottom: 1.5rem; padding: 12px 0;
    }
    .step-pill {
        display: flex; align-items: center; gap: 8px; padding: 6px 16px; border-radius: 20px;
        font-size: .72rem; font-weight: 700; letter-spacing: .05em; text-transform: uppercase;
    }
    .step-pill-active  { background: #eff6ff; border: 1.5px solid var(--clr-accent); color: var(--clr-accent); }
    .step-pill-inactive { background: var(--clr-bg); border: 1.5px solid var(--clr-border); color: var(--clr-text-muted); }
    .step-num {
        width: 20px; height: 20px; border-radius: 50%;
        display: flex; align-items: center; justify-content: center; font-size: .65rem; font-weight: 800;
    }
    .step-num-active   { background: var(--clr-accent); color: #fff; }
    .step-num-inactive { background: var(--clr-border); color: var(--clr-text-muted); }
    .step-connector         { flex: 1; height: 2px; background: var(--clr-border); max-width: 40px; }
    .step-connector-active  { background: var(--clr-accent); }

    /* Sidebar brand */
    .sidebar-brand {
        padding: 16px 0 20px 0;
        border-bottom: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 16px;
    }
    .sidebar-brand-logo {
        width: 32px; height: 32px; background: var(--clr-accent); border-radius: 6px;
        display: inline-flex; align-items: center; justify-content: center;
        font-size: .85rem; font-weight: 800; color: #fff;
    }
    .sidebar-brand-name {
        font-size: .95rem; font-weight: 700; color: var(--clr-sidebar-text);
        margin-left: 10px; vertical-align: middle;
    }
    .sidebar-brand-sub {
        font-size: .65rem; color: var(--clr-sidebar-muted); text-transform: uppercase;
        letter-spacing: .08em; margin-top: 6px;
    }
    .sidebar-section-label {
        font-size: .6rem; font-weight: 700; text-transform: uppercase; letter-spacing: .1em;
        color: var(--clr-sidebar-muted); margin-bottom: 8px;
    }

    /* Fix chart legend overlap */
    [data-testid="stVegaLiteChart"] { margin-bottom: 2rem; }

    /* Hide streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    /* Hide the default sidebar collapse control */
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapseButton"],
    button[kind="header"],
    [data-testid="baseButton-header"] {
        display: none !important;
        visibility: hidden !important;
        width: 0 !important;
        height: 0 !important;
        opacity: 0 !important;
        pointer-events: none !important;
    }

    /* Fixed sidebar width */
    [data-testid="stSidebar"] > div:first-child {
        width: 300px !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Initialize Session State ---
if 'financials' not in st.session_state:
    st.session_state.financials = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'quarterly_analysis' not in st.session_state:
    st.session_state.quarterly_analysis = None
if 'independent_forecast' not in st.session_state:
    st.session_state.independent_forecast = None
if 'api_key_set' not in st.session_state:
    st.session_state.api_key_set = False
if 'dcf_ui_adapter' not in st.session_state:
    st.session_state.dcf_ui_adapter = None
if 'dcf_engine_result' not in st.session_state:
    st.session_state.dcf_engine_result = None
if 'dcf_snapshot' not in st.session_state:
    st.session_state.dcf_snapshot = None
if 'show_dcf_details' not in st.session_state:
    st.session_state.show_dcf_details = False
if 'show_user_guide' not in st.session_state:
    st.session_state.show_user_guide = False
if 'forecast_just_generated' not in st.session_state:
    st.session_state.forecast_just_generated = False
if 'ai_outlook_error' not in st.session_state:
    st.session_state.ai_outlook_error = None
if 'ui_cache' not in st.session_state:
    st.session_state.ui_cache = load_ui_cache()
if st.session_state.ui_cache.get("available_report_dates_version") != REPORT_DATES_CACHE_VERSION:
    st.session_state.ui_cache["available_report_dates_version"] = REPORT_DATES_CACHE_VERSION
    st.session_state.ui_cache["available_report_dates"] = {}
    save_ui_cache(st.session_state.ui_cache)
if 'ticker_library' not in st.session_state:
    st.session_state.ticker_library = _normalize_ticker_library(st.session_state.ui_cache.get("ticker_library", []))
if 'last_restore_key' not in st.session_state:
    st.session_state.last_restore_key = None
if 'cache_restore_notice' not in st.session_state:
    st.session_state.cache_restore_notice = ""
if 'ticker_dropdown' not in st.session_state:
    last_selected = _normalize_ticker(st.session_state.ui_cache.get("last_selected_ticker", "MSFT"))
    library = st.session_state.ticker_library
    st.session_state.ticker_dropdown = last_selected if last_selected in library else library[0]
if 'pending_ticker_dropdown' not in st.session_state:
    st.session_state.pending_ticker_dropdown = None
if 'assumption_suggestions_loaded' not in st.session_state:
    st.session_state.assumption_suggestions_loaded = False
if 'assumption_suggestions_ticker' not in st.session_state:
    st.session_state.assumption_suggestions_ticker = None
if 'config_num_quarters' not in st.session_state:
    existing_num_quarters = st.session_state.get("num_quarters", 8)
    if not isinstance(existing_num_quarters, int):
        existing_num_quarters = 8
    st.session_state.config_num_quarters = min(20, max(8, existing_num_quarters))
if 'pending_config_num_quarters' not in st.session_state:
    st.session_state.pending_config_num_quarters = None
if 'momentum_display_quarters' not in st.session_state:
    st.session_state.momentum_display_quarters = 8
if 'pending_momentum_display_quarters' not in st.session_state:
    st.session_state.pending_momentum_display_quarters = None

# --- Helper Functions ---
def reset_analysis():
    st.session_state.quarterly_analysis = None
    st.session_state.independent_forecast = None
    st.session_state.forecast_just_generated = False
    st.session_state.ai_outlook_error = None
    st.session_state.cache_restore_notice = ""
    st.session_state.last_restore_key = None
    # Reset DCF assumptions so they get re-calculated for new ticker
    st.session_state.dcf_wacc = None
    st.session_state.dcf_fcf_growth = None
    st.session_state.dcf_terminal_growth = None
    st.session_state.dcf_ui_adapter = None
    st.session_state.dcf_engine_result = None
    st.session_state.dcf_snapshot = None
    st.session_state.assumption_suggestions_loaded = False
    st.session_state.assumption_suggestions_ticker = None
    configured_quarters = st.session_state.get("config_num_quarters", 8)
    if not isinstance(configured_quarters, int):
        configured_quarters = 8
    st.session_state.momentum_display_quarters = min(20, max(8, configured_quarters))
    st.session_state.pending_config_num_quarters = None
    st.session_state.pending_momentum_display_quarters = None

def display_stock_call(call: str):
    """Displays the stock call with clean styling."""
    call_lower = call.lower() if call else ""
    
    if "outperform" in call_lower or "above" in call_lower or "buy" in call_lower:
        st.markdown("""
            <div class="call-outperform">
                <div style="font-size: 1.5rem; margin-bottom: 4px;">▲</div>
                <div class="call-label">OUTPERFORM</div>
            </div>
        """, unsafe_allow_html=True)
    elif "underperform" in call_lower or "below" in call_lower or "sell" in call_lower:
        st.markdown("""
            <div class="call-underperform">
                <div style="font-size: 1.5rem; margin-bottom: 4px;">▼</div>
                <div class="call-label">UNDERPERFORM</div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class="call-inline">
                <div style="font-size: 1.5rem; margin-bottom: 4px;">◆</div>
                <div class="call-label">IN-LINE</div>
            </div>
        """, unsafe_allow_html=True)

def parse_price_value(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).replace(",", "").strip()
    match = re.search(r"[-+]?\d*\.?\d+", text)
    return float(match.group(0)) if match else None

def is_quota_or_rate_limit_error(error_text: str) -> bool:
    text = str(error_text or "").lower()
    markers = ["resource_exhausted", "quota", "rate limit", "too many requests", "429"]
    return any(marker in text for marker in markers)

def estimate_api_retry_time(error_text: str) -> str:
    text = str(error_text or "").lower()
    retry_seconds = None
    retry_patterns = [
        r"retry in\s+(\d+)\s*(?:seconds|secs|sec|s)\b",
        r"try again in\s+(\d+)\s*(?:seconds|secs|sec|s)\b",
        r"retry_delay[^\d]{0,20}(\d+)",
        r"seconds[\"']?\s*:\s*(\d+)",
    ]
    for pattern in retry_patterns:
        match = re.search(pattern, text)
        if match:
            try:
                retry_seconds = int(match.group(1))
                break
            except Exception:
                pass

    if retry_seconds is not None and retry_seconds > 0:
        retry_dt_utc = datetime.now(timezone.utc) + timedelta(seconds=retry_seconds)
        return retry_dt_utc.astimezone().strftime("%b %d, %Y %I:%M %p %Z")

    # Provider quota reset timing is plan-dependent; this is an explicit estimate fallback.
    try:
        pt = ZoneInfo("America/Los_Angeles")
        now_pt = datetime.now(pt)
        next_reset_pt = (now_pt + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return next_reset_pt.strftime("%b %d, %Y %I:%M %p PT (estimated)")
    except Exception:
        return "next daily reset window (estimated)"

def get_valuation_verdict(upside_pct: float):
    """Map upside/downside to verdict label, style class, and short rationale."""
    if upside_pct > 25:
        return "Significantly Undervalued", "badge-pass", "DCF >25% above market"
    if upside_pct > 10:
        return "Modestly Undervalued", "badge-pass", "DCF 10-25% above market"
    if upside_pct < -25:
        return "Significantly Overvalued", "badge-fail", "DCF >25% below market"
    if upside_pct < -10:
        return "Modestly Overvalued", "badge-fail", "DCF 10-25% below market"
    return "Fairly Valued", "badge-neutral-plain", "DCF within +/-10% of market"

# --- Sidebar Toggle State ---
if "sidebar_visible" not in st.session_state:
    st.session_state.sidebar_visible = True

# Floating button to open sidebar when closed
if not st.session_state.sidebar_visible:
    st.markdown("""
        <style>
            [data-testid="stSidebar"] { display: none !important; }
        </style>
    """, unsafe_allow_html=True)
    
    # Show a floating open button in the corner
    if st.button("☰ Menu", key="open_sidebar"):
        st.session_state.sidebar_visible = True
        st.rerun()

# --- Sidebar ---
with st.sidebar:
    # Close button at top of sidebar
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("✕", key="close_sidebar", help="Close sidebar"):
            st.session_state.sidebar_visible = False
            st.rerun()
    
    # Sidebar brand header
    st.markdown("""
        <div class="sidebar-brand">
            <div>
                <span class="sidebar-brand-logo">AC</span>
                <span class="sidebar-brand-name">Analyst Co-Pilot</span>
            </div>
            <div class="sidebar-brand-sub">Equity Research</div>
        </div>
        <div class="sidebar-section-label">Configuration</div>
    """, unsafe_allow_html=True)

    # Auto-load API key from .env (no UI shown)
    env_api_key = os.environ.get("GEMINI_API_KEY", "")
    if env_api_key:
        st.session_state.api_key_set = True

    # Persistent ticker picker: MAG7 defaults + learned custom tickers
    st.session_state.ticker_library = _normalize_ticker_library(
        st.session_state.ui_cache.get("ticker_library", st.session_state.ticker_library)
    )
    ticker_options = st.session_state.ticker_library

    # Apply deferred widget updates before widget instantiation (Streamlit-safe).
    pending_dropdown = _normalize_ticker(st.session_state.get("pending_ticker_dropdown"))
    if pending_dropdown and pending_dropdown in ticker_options:
        st.session_state.ticker_dropdown = pending_dropdown
    st.session_state.pending_ticker_dropdown = None

    if st.session_state.ticker_dropdown not in ticker_options:
        st.session_state.ticker_dropdown = ticker_options[0]

    selected_ticker = st.selectbox("Stock Ticker", options=ticker_options, key="ticker_dropdown")

    ticker = _normalize_ticker(selected_ticker)
    ticker_valid = _is_valid_ticker_format(ticker)

    # Date-fetch state
    if "available_dates" not in st.session_state:
        st.session_state.available_dates = []
    if "available_dates_ticker" not in st.session_state:
        st.session_state.available_dates_ticker = None
    if "selected_end_date" not in st.session_state:
        st.session_state.selected_end_date = None
    if "report_dates_hint" not in st.session_state:
        st.session_state.report_dates_hint = ""

    # Load report-date state whenever ticker changes.
    if ticker_valid and ticker != st.session_state.available_dates_ticker:
        st.session_state.available_dates_ticker = ticker
        persisted_dates = _get_persisted_report_dates(ticker)
        if persisted_dates:
            dates = persisted_dates
        else:
            with st.spinner(f"Loading available reports for {ticker}..."):
                dates = _load_report_dates_for_ticker(ticker)
        st.session_state.available_dates = dates
        if st.session_state.available_dates:
            st.session_state.selected_end_date = st.session_state.available_dates[0]["value"]
            st.session_state.report_dates_hint = ""
        else:
            st.session_state.selected_end_date = None
            st.session_state.report_dates_hint = (
                f"No report dates returned within {AVAILABLE_DATES_TIMEOUT_SECONDS}s. "
                "Use Refresh Available Dates to try again."
            )
    elif not ticker_valid:
        st.session_state.available_dates_ticker = None
        st.session_state.available_dates = []
        st.session_state.selected_end_date = None
        st.session_state.report_dates_hint = ""

    available_dates = st.session_state.available_dates
    selected_end_date = st.session_state.selected_end_date

    if available_dates:
        latest_date = available_dates[0]["display"]
        st.markdown(f"""
            <div class="sidebar-data-badge">Latest Data: {latest_date}</div>
        """, unsafe_allow_html=True)
    elif ticker_valid:
        st.caption(st.session_state.report_dates_hint or "No report dates loaded yet.")
    
    # Analysis Period selection
    st.markdown("**Analysis Period**")
    pending_config_quarters = st.session_state.get("pending_config_num_quarters")
    if isinstance(pending_config_quarters, int):
        st.session_state.config_num_quarters = min(20, max(8, pending_config_quarters))
    st.session_state.pending_config_num_quarters = None
    num_quarters = st.slider(
        "Historical Quarters",
        min_value=8,
        max_value=20,
        key="config_num_quarters",
        help="How many quarters of historical data to analyze (minimum 8 for trend visibility)"
    )
    
    # Single selectbox for ending report date - only shows ACTUAL available dates
    if available_dates:
        date_options = [d["display"] for d in available_dates]
        date_values = [d["value"] for d in available_dates]
        default_idx = 0
        if st.session_state.selected_end_date in date_values:
            default_idx = date_values.index(st.session_state.selected_end_date)
        else:
            st.session_state.selected_end_date = date_values[0]

        selected_display = st.selectbox(
            "Select Ending Report",
            options=date_options,
            index=default_idx,
            help=f"Select the most recent quarter to include. {len(available_dates)} reports available."
        )

        # Get the corresponding ISO date value
        selected_idx = date_options.index(selected_display)
        selected_end_date = date_values[selected_idx]
        st.session_state.selected_end_date = selected_end_date
    else:
        st.selectbox(
            "Select Ending Report",
            options=["No report dates available"],
            disabled=True
        )
        selected_end_date = None

    if ticker_valid:
        if st.button(
            "Refresh Available Dates",
            key="refresh_available_reports",
            help="Manually re-fetch quarter-end report dates for this ticker."
        ):
            with st.spinner(f"Refreshing available reports for {ticker}..."):
                refreshed_dates = _load_report_dates_for_ticker(ticker, force_refresh=True)
            st.session_state.available_dates_ticker = ticker
            st.session_state.available_dates = refreshed_dates
            if refreshed_dates:
                st.session_state.selected_end_date = refreshed_dates[0]["value"]
                st.session_state.report_dates_hint = ""
            else:
                st.session_state.selected_end_date = None
                st.session_state.report_dates_hint = (
                    f"No report dates returned within {AVAILABLE_DATES_TIMEOUT_SECONDS}s. "
                    "Please try Refresh Available Dates again."
                )
            st.rerun()

    # Best-effort immediate restore for same active ticker/context
    active_ticker = _normalize_ticker(st.session_state.get("ticker", ""))
    if ticker_valid and selected_end_date:
        context_key = build_context_key(ticker, selected_end_date, num_quarters)
        loaded_end_date = st.session_state.get("end_date")
        loaded_num_quarters = st.session_state.get("num_quarters")
        same_loaded_context = (
            active_ticker == ticker
            and loaded_end_date == selected_end_date
            and loaded_num_quarters == num_quarters
        )
        can_restore_now = st.session_state.quarterly_analysis is None or same_loaded_context
        if can_restore_now and st.session_state.get("last_restore_key") != context_key:
            restored = _restore_cached_results_for_context(ticker, selected_end_date, num_quarters)
            if restored["dcf"] or restored["ai"]:
                restored_parts = []
                if restored["dcf"]:
                    restored_parts.append("DCF")
                if restored["ai"]:
                    restored_parts.append("AI outlook")
                st.session_state.cache_restore_notice = f"Restored cached {' + '.join(restored_parts)} for {ticker} ({selected_end_date}, {num_quarters}Q)."
            else:
                st.session_state.cache_restore_notice = ""
                st.session_state.last_restore_key = None
        elif st.session_state.get("last_restore_key") == context_key:
            # no-op; already restored for this exact context in current session
            pass
    else:
        st.session_state.cache_restore_notice = ""
        st.session_state.last_restore_key = None
    
    if st.button("Load Data", type="primary", use_container_width=True):
        if not st.session_state.api_key_set:
            st.error("API key not found. Add `GEMINI_API_KEY=your_key` to `.env` file and restart.")
        elif not ticker_valid:
            st.warning("Please enter/select a valid ticker symbol.")
        else:
            if not selected_end_date:
                with st.spinner(f"Fetching available reports for {ticker}..."):
                    dates = _load_report_dates_for_ticker(ticker)
                st.session_state.available_dates = dates if isinstance(dates, list) else []
                if st.session_state.available_dates:
                    st.session_state.selected_end_date = st.session_state.available_dates[0]["value"]
                    selected_end_date = st.session_state.selected_end_date
                    st.session_state.report_dates_hint = ""
                else:
                    st.session_state.selected_end_date = None
                    selected_end_date = None
                    st.session_state.report_dates_hint = (
                        f"No report dates returned within {AVAILABLE_DATES_TIMEOUT_SECONDS}s. "
                        "Use Refresh Available Dates to try again."
                    )

            if not selected_end_date:
                st.warning("Please select a valid ending report date first.")
            else:
                with st.spinner(f"Loading {ticker}..."):
                    inc, bal, cf, qcf = cached_financials(ticker)
                    if not inc.empty:
                        st.session_state.financials = {"income": inc, "balance": bal, "cashflow": cf, "quarterly_cashflow": qcf}
                        st.session_state.ticker = ticker
                        st.session_state.metrics = calculate_metrics(inc, bal)
                        st.session_state.num_quarters = num_quarters
                        st.session_state.end_date = selected_end_date
                        reset_analysis()

                        # Persist last selected ticker
                        cache = st.session_state.get("ui_cache", _default_ui_cache())
                        cache["last_selected_ticker"] = ticker
                        st.session_state.ui_cache = cache
                        save_ui_cache(cache)

                        # Auto-run quarterly analysis (cached) with user-selected end date
                        analysis = cached_quarterly_analysis(ticker, num_quarters, selected_end_date)
                        st.session_state.quarterly_analysis = analysis
                        # Calculate comprehensive analysis (DuPont + DCF)
                        quarterly_data = analysis.get("historical_trends", {}).get("quarterly_data", [])
                        st.session_state.momentum_display_quarters = min(
                            max(1, len(quarterly_data)),
                            max(8, num_quarters)
                        ) if quarterly_data else max(8, num_quarters)
                        st.session_state.comprehensive_analysis = calculate_comprehensive_analysis(
                            inc,
                            bal,
                            quarterly_data,
                            ticker,
                            cf,
                            qcf
                        )

                        # Restore cached per-context DCF / AI outputs (if available)
                        restored = _restore_cached_results_for_context(ticker, selected_end_date, num_quarters)
                        if restored["dcf"] or restored["ai"]:
                            restored_parts = []
                            if restored["dcf"]:
                                restored_parts.append("DCF")
                            if restored["ai"]:
                                restored_parts.append("AI outlook")
                            restore_message = f"Restored cached {' + '.join(restored_parts)} for {ticker} ({selected_end_date}, {num_quarters}Q)."
                            st.session_state.cache_restore_notice = restore_message
                        else:
                            st.session_state.cache_restore_notice = ""
                            st.session_state.last_restore_key = None
                        
                        # Show what was loaded
                        most_recent = analysis.get("historical_trends", {}).get("most_recent_quarter", {})
                        next_q = analysis.get("next_forecast_quarter", {})
                        if most_recent.get("label"):
                            st.success(f"Loaded {ticker} through {most_recent.get('label')}")
                            if next_q.get("label"):
                                st.info(f"Forecasting: {next_q.get('label')}")
                        else:
                            st.success(f"Loaded {ticker}")
                    else:
                        st.error("Failed to fetch data.")

    st.divider()
    st.markdown('<div class="sidebar-section-label">Help</div>', unsafe_allow_html=True)
    if st.button("Open User Guide", use_container_width=True, key="open_user_guide_sidebar"):
        st.session_state.show_user_guide = True
        st.rerun()
    
    # Clear cache button
    st.divider()
    if st.button("🔄 Clear Cache", help="Force refresh API data"):
        st.cache_data.clear()
        reset_analysis()
        st.success("Cache cleared! Click 'Load Data' to fetch fresh data.")



# --- Main Interface ---
st.markdown("""
<div class="app-topbar">
    <div>
        <span class="app-wordmark">Analyst Co-Pilot</span>
        <span class="app-version">Beta</span>
    </div>
    <div class="app-tagline">AI-powered equity research assistant</div>
</div>
""", unsafe_allow_html=True)

_guide_spacer, _guide_col = st.columns([6, 1])
with _guide_col:
    if st.button("User Guide", key="open_user_guide_header", use_container_width=True):
        st.session_state.show_user_guide = True
        st.rerun()

if st.session_state.get("show_user_guide"):
    _show_user_guide_page()
    st.stop()

if st.session_state.quarterly_analysis:
    analysis = st.session_state.quarterly_analysis
    ticker = st.session_state.ticker
    most_recent = analysis.get("historical_trends", {}).get("most_recent_quarter", {})
    next_forecast = analysis.get("next_forecast_quarter", {})
    data_source = analysis.get("data_source", "Unknown")
    warning = analysis.get("warning")
    hist_data = analysis.get("historical_trends", {}).get("quarterly_data", [])
    data_coverage = analysis.get("historical_trends", {}).get("data_coverage", {})
    source_diagnostics = analysis.get("historical_trends", {}).get("source_diagnostics", {})
    growth_summary = analysis.get("growth_rates", {}).get("summary", {})
    seasonality_info = analysis.get("growth_rates", {}).get("seasonality", {})
    growth_detail = analysis.get("growth_rates", {}).get("detailed", [])
    comp_analysis = st.session_state.get("comprehensive_analysis", {})
    consensus = analysis.get("consensus_estimates", {})
    next_forecast_label = next_forecast.get("label", "Next Quarter")
    dcf_ui = st.session_state.get("dcf_ui_adapter")

    st.markdown("""
<div class="disclaimer-banner">
  <div class="disclaimer-title">Disclaimer</div>
  <div class="disclaimer-text">AI-generated outputs may contain mistakes. This is not investment advice. Results are highly dependent on assumptions and input data quality.</div>
</div>
    """, unsafe_allow_html=True)

    # Top context strip
    _market_data = analysis.get("market_data", {})
    _price = _market_data.get("current_price")
    _mcap = _market_data.get("market_cap")
    _pe = _market_data.get("pe_ratio")
    _as_of = most_recent.get("label", "—")
    _price_str = f"${_price:,.2f}" if _price else "—"
    _mcap_str = f"${_mcap/1e9:.1f}B" if _mcap else "—"
    _show_pe = _pe is not None and _pe > 0
    _pe_str = f"{_pe:.1f}x" if _show_pe else None

    _hero_tiles = [
        ("Price", _price_str, None),
        ("Market Cap", _mcap_str, None),
    ]
    if _show_pe:
        _hero_tiles.append(("P/E Ratio", _pe_str, None))
    _hero_tiles.append(("As Of", _as_of, "font-size:1rem"))

    _hero_items_html = []
    for idx, (label, value, value_style) in enumerate(_hero_tiles):
        style = ' style="border-right:none"' if idx == len(_hero_tiles) - 1 else ""
        value_style_html = f' style="{value_style}"' if value_style else ""
        _hero_items_html.append(
            f'<div class="hero-item"{style}><div class="hero-label">{label}</div>'
            f'<div class="hero-value"{value_style_html}>{value}</div></div>'
        )
    _hero_items_html = "".join(_hero_items_html)

    st.markdown(f"""
<div class="hero-strip">
  {_hero_items_html}
  <div class="hero-divider"></div>
  <div class="hero-ticker">
    <span class="hero-ticker-symbol">{ticker}</span>
    <span class="hero-ticker-source">via yFinance</span>
  </div>
</div>
    """, unsafe_allow_html=True)

    st.markdown("""
<div id="report-nav-primary" class="report-nav">
  <a href="#verdict">Verdict</a>
  <a href="#valuation">Valuation Drivers</a>
  <a href="#momentum">Business Momentum</a>
  <a href="#consensus">Street Context</a>
  <a href="#outlook">AI Synthesis</a>
  <a href="#sources">Sources</a>
</div>
    """, unsafe_allow_html=True)

    st.markdown("""
<div id="floating-toc-wrap" class="floating-toc-wrap" aria-label="Quick navigation">
  <button id="floating-toc-toggle" class="floating-toc-toggle" type="button" aria-label="Toggle quick nav" aria-expanded="false" aria-controls="floating-toc">☰</button>
  <div id="floating-toc" class="floating-toc" aria-label="Report table of contents">
    <div class="floating-toc-title">Quick Nav</div>
    <a href="#verdict">Verdict</a>
    <a href="#valuation">Valuation</a>
    <a href="#momentum">Momentum</a>
    <a href="#consensus">Street</a>
    <a href="#outlook">AI View</a>
    <a href="#sources">Sources</a>
  </div>
</div>
    """, unsafe_allow_html=True)

    components.html(
        """
<script>
(function () {
  const p = window.parent;
  const d = p.document;
  const bindingKey = "__acpTocBindings";
  const sectionIds = ["verdict", "valuation", "momentum", "consensus", "outlook", "sources"];
  const navIds = ["report-nav-primary", "floating-toc"];

  const cleanup = () => {
    const prev = p[bindingKey];
    if (!prev) return;

    if (prev.scrollTargets && prev.onScroll) {
      prev.scrollTargets.forEach((target) => {
        try { target.removeEventListener("scroll", prev.onScroll); } catch (_) {}
      });
    } else if (prev.scroller && prev.onScroll) {
      try { prev.scroller.removeEventListener("scroll", prev.onScroll); } catch (_) {}
    }

    if (prev.onResize) {
      try { p.removeEventListener("resize", prev.onResize); } catch (_) {}
    }
    if (prev.toggleButton && prev.onToggle) {
      try { prev.toggleButton.removeEventListener("click", prev.onToggle); } catch (_) {}
    }
    if (prev.navLinks && prev.onLinkClick) {
      prev.navLinks.forEach((link) => {
        try { link.removeEventListener("click", prev.onLinkClick); } catch (_) {}
      });
    }
    if (prev.observer) {
      try { prev.observer.disconnect(); } catch (_) {}
    }
    if (prev.timer) {
      try { p.clearTimeout(prev.timer); } catch (_) {}
    }
    if (prev.rafId) {
      try { p.cancelAnimationFrame(prev.rafId); } catch (_) {}
    }
  };

  cleanup();

  const setActive = (sectionId) => {
    navIds.forEach((navId) => {
      const nav = d.getElementById(navId);
      if (!nav) return;
      const links = nav.querySelectorAll('a[href^="#"]');
      links.forEach((link) => {
        const target = link.getAttribute("href");
        const isActive = target === "#" + sectionId;
        link.classList.toggle("is-active", isActive);
        if (isActive) {
          link.setAttribute("aria-current", "true");
        } else {
          link.removeAttribute("aria-current");
        }
      });
    });
  };

  const getActiveSection = () => {
    const threshold = 160;
    let active = sectionIds[0];
    for (const id of sectionIds) {
      const el = d.getElementById(id);
      if (!el) continue;
      if (el.getBoundingClientRect().top <= threshold) {
        active = id;
      } else {
        break;
      }
    }
    return active;
  };

  const getScroller = () => {
    return (
      d.querySelector('[data-testid="stAppViewContainer"]') ||
      d.querySelector("section.main") ||
      p
    );
  };

  const init = (attempt) => {
    const primary = d.getElementById("report-nav-primary");
    const floatingWrap = d.getElementById("floating-toc-wrap");
    const floating = d.getElementById("floating-toc");
    const floatingToggle = d.getElementById("floating-toc-toggle");
    if (!primary || !floatingWrap || !floating || !floatingToggle) {
      if (attempt < 40) {
        const timer = p.setTimeout(() => init(attempt + 1), 80);
        p[bindingKey] = { ...(p[bindingKey] || {}), timer };
      }
      return;
    }

    const scroller = getScroller();
    let rafId = null;

    const setExpanded = (expanded) => {
      floatingWrap.classList.toggle("is-expanded", expanded);
      floatingToggle.setAttribute("aria-expanded", expanded ? "true" : "false");
    };

    const update = () => {
      const rect = primary.getBoundingClientRect();
      const shouldShow = rect.bottom <= 0;
      floatingWrap.classList.toggle("is-visible", shouldShow);
      if (!shouldShow) {
        setExpanded(false);
      }
      setActive(getActiveSection());
    };

    const scheduleUpdate = () => {
      if (rafId !== null) return;
      rafId = p.requestAnimationFrame(() => {
        rafId = null;
        update();
      });
      if (p[bindingKey]) {
        p[bindingKey].rafId = rafId;
      }
    };

    const onScroll = () => scheduleUpdate();
    const onResize = () => scheduleUpdate();
    const onToggle = () => {
      const isExpanded = floatingWrap.classList.contains("is-expanded");
      setExpanded(!isExpanded);
    };
    const onLinkClick = () => setExpanded(false);
    const navLinks = Array.from(floating.querySelectorAll('a[href^="#"]'));

    const scrollTargets = Array.from(new Set([scroller, p]));
    scrollTargets.forEach((target) => {
      target.addEventListener("scroll", onScroll, { passive: true });
    });
    p.addEventListener("resize", onResize);
    floatingToggle.addEventListener("click", onToggle);
    navLinks.forEach((link) => {
      link.addEventListener("click", onLinkClick);
    });

    let observer = null;
    if ("IntersectionObserver" in p) {
      observer = new p.IntersectionObserver(
        () => scheduleUpdate(),
        {
          root: scroller === p ? null : scroller,
          threshold: [0],
        }
      );
      observer.observe(primary);
    }

    p[bindingKey] = {
      scroller,
      scrollTargets,
      onScroll,
      onResize,
      toggleButton: floatingToggle,
      onToggle,
      navLinks,
      onLinkClick,
      observer,
      rafId: null,
      timer: null,
    };

    update();
  };

  init(0);
})();
</script>
        """,
        height=0,
        width=0,
    )

    # Show DCF Details page if requested
    if st.session_state.get("show_dcf_details") and dcf_ui:
        _show_dcf_details_page()
        st.stop()

    # SECTION A: Investment Verdict
    st.markdown('<div id="verdict"></div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="section-header"><span class="step-badge">Step 01</span><span class="section-title">Investment Verdict</span></div>', unsafe_allow_html=True)
    st.caption("Primary call first, then supporting evidence.")

    dcf_ui_data = dcf_ui.get_ui_data() if dcf_ui else None
    if dcf_ui_data:
        if not dcf_ui_data.get("success"):
            st.error("DCF analysis failed. Review assumptions and rerun.")
            for err in dcf_ui_data.get("errors", []):
                st.error(f"• {err}")
        else:
            current_price = dcf_ui_data.get("current_price", 0)
            intrinsic = dcf_ui_data.get("price_per_share", 0)
            data_quality = dcf_ui_data.get("data_quality_score", 0)
            assumptions = dcf_ui_data.get("assumptions", {})
            tv_dominance = assumptions.get("tv_dominance_pct", 0)
            price_exit = assumptions.get("price_exit_multiple")
            price_gordon = assumptions.get("price_gordon_growth")
            terminal_method = assumptions.get("terminal_value_method", "gordon_growth")
            high_growth_profile = bool(assumptions.get("high_growth_company", False))
            cashflow_regime = assumptions.get("cashflow_regime")
            cashflow_confidence = assumptions.get("cashflow_confidence")
            discount_rate_used = assumptions.get("discount_rate_used")
            discount_rate_input = assumptions.get("discount_rate_input")
            discount_rate_label = assumptions.get("discount_rate_label")

            upside_downside = None
            verdict_label = "Pending"
            verdict_badge = "badge-neutral-plain"
            verdict_hint = "Run DCF to generate a valuation verdict."
            if current_price and current_price > 0 and intrinsic and intrinsic > 0:
                upside_downside = ((intrinsic - current_price) / current_price * 100)
                verdict_label, verdict_badge, verdict_hint = get_valuation_verdict(upside_downside)

            terminal_method_label = "Exit Multiple" if terminal_method == "exit_multiple" else "Gordon Growth"
            if terminal_method == "exit_multiple" and high_growth_profile:
                terminal_method_context = "High-Growth Adaptive"
            elif terminal_method == "exit_multiple":
                terminal_method_context = "Adaptive Fallback"
            else:
                terminal_method_context = "Default"

            cashflow_regime_label = "FCFF"
            if cashflow_regime == "approx_unlevered":
                cashflow_regime_label = "Approx FCFF Proxy"
            elif cashflow_regime == "levered_proxy":
                cashflow_regime_label = "Levered Proxy"
            confidence_label = (cashflow_confidence or "n/a").capitalize()
            if discount_rate_used is not None:
                if discount_rate_input is not None and abs(discount_rate_used - discount_rate_input) > 1e-9:
                    discount_rate_text = (
                        f"{discount_rate_label or 'Discount rate'}: {discount_rate_used*100:.1f}% "
                        f"(input {discount_rate_input*100:.1f}%)"
                    )
                else:
                    discount_rate_text = f"{discount_rate_label or 'Discount rate'}: {discount_rate_used*100:.1f}%"
            else:
                discount_rate_text = discount_rate_label or "Discount rate: N/A"

            current_price_text = f"${current_price:.2f}" if current_price else "—"
            intrinsic_text = f"${intrinsic:.2f}" if intrinsic else "—"
            upside_text = f"{upside_downside:+.1f}%" if upside_downside is not None else "—"
            st.markdown(f"""
<div class="decision-strip">
  <div class="decision-grid">
    <div><div class="decision-tile-label">Ticker</div><div class="decision-tile-value">{ticker}</div></div>
    <div><div class="decision-tile-label">Current Price</div><div class="decision-tile-value">{current_price_text}</div></div>
    <div>
      <div class="decision-tile-label">Intrinsic Value</div>
      <div class="decision-tile-value">{intrinsic_text}</div>
    </div>
    <div><div class="decision-tile-label">Upside/Downside</div><div class="decision-tile-value">{upside_text}</div></div>
    <div><div class="decision-tile-label">Verdict</div><span class="badge {verdict_badge}">{verdict_label}</span></div>
  </div>
</div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
<div class="confidence-strip">
  <span class="confidence-pill">Data Quality: {data_quality:.0f}/100</span>
  <span class="confidence-pill">TV Dominance: {tv_dominance:.0f}%</span>
  <span class="confidence-pill">Terminal Method: {terminal_method_label} ({terminal_method_context})</span>
  <span class="confidence-pill">Cash Flow Regime: {cashflow_regime_label} ({confidence_label})</span>
  <span class="confidence-pill">{discount_rate_text}</span>
</div>
            """, unsafe_allow_html=True)
            st.caption(verdict_hint)
            if cashflow_regime in {"approx_unlevered", "levered_proxy"}:
                st.warning(
                    "Proxy cash-flow mode is active. Intrinsic value is still produced, but should be treated as lower-confidence "
                    "than a clean FCFF run."
                )
            if not dcf_ui_data.get("data_sufficient"):
                st.warning("Insufficient data quality: interpretation confidence is reduced.")
    else:
        st.info("Run DCF Analysis in Step 02 to generate the investment verdict.")

    # SECTION B: Valuation Drivers
    st.markdown('<div id="valuation"></div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="section-header"><span class="step-badge">Step 02</span><span class="section-title">Valuation Drivers</span></div>', unsafe_allow_html=True)
    st.caption("Tune assumptions, rerun the model, and review core valuation outputs.")

    snapshot_for_suggestions = None
    suggestions_ready = (
        st.session_state.get("assumption_suggestions_loaded", False)
        and st.session_state.get("assumption_suggestions_ticker") == ticker
    )
    if suggestions_ready:
        snapshot_for_suggestions = cached_financial_snapshot(
            ticker,
            suggestion_algo_version="v3_forward_consensus",
        )
    else:
        st.caption("Suggested WACC/FCF inputs are optional and loaded on demand for faster first paint.")
        if st.button(
            "Load Suggested Assumptions",
            key=f"load_suggested_assumptions_{ticker}",
            use_container_width=False,
        ):
            st.session_state.assumption_suggestions_loaded = True
            st.session_state.assumption_suggestions_ticker = ticker
            st.rerun()

    suggested_wacc = 9.0
    suggested_fcf_growth = 8.0
    suggested_fcf_reliability = None
    suggested_fcf_period_type = None
    if snapshot_for_suggestions:
        if snapshot_for_suggestions.suggested_wacc.value:
            suggested_wacc = round(snapshot_for_suggestions.suggested_wacc.value * 100, 1)
        if snapshot_for_suggestions.suggested_fcf_growth.value is not None:
            suggested_fcf_growth = round(snapshot_for_suggestions.suggested_fcf_growth.value * 100, 1)
            suggested_fcf_reliability = snapshot_for_suggestions.suggested_fcf_growth.reliability_score
            suggested_fcf_period_type = snapshot_for_suggestions.suggested_fcf_growth.period_type

    stored_wacc = st.session_state.get("dcf_wacc")
    stored_fcf_growth = st.session_state.get("dcf_fcf_growth")
    stored_terminal_growth = st.session_state.get("dcf_terminal_growth")
    default_wacc = stored_wacc if stored_wacc is not None else suggested_wacc
    if stored_fcf_growth is not None:
        default_fcf_growth = stored_fcf_growth
    else:
        # Keep defaults conservative if suggestion quality is low.
        if suggested_fcf_reliability is not None and suggested_fcf_reliability < 65:
            default_fcf_growth = 8.0
        else:
            default_fcf_growth = suggested_fcf_growth
    default_fcf_growth = max(0.0, min(25.0, default_fcf_growth))
    default_terminal_growth = stored_terminal_growth if stored_terminal_growth is not None else 3.0

    col_wacc, col_growth, col_terminal = st.columns(3)
    with col_wacc:
        user_wacc = st.slider(
            "WACC / Discount Rate (%)",
            min_value=5.0,
            max_value=15.0,
            value=default_wacc,
            step=0.1,
            format="%.1f",
            key=f"wacc_slider_{ticker}",
            help=(
                "Forward discount-rate assumption used in DCF. "
                "View DCF Details to see full component trace (Re, Rd, weights, and sources)."
            )
        )
        if snapshot_for_suggestions and snapshot_for_suggestions.suggested_wacc.value:
            beta_val = snapshot_for_suggestions.beta.value
            rf_source = getattr(snapshot_for_suggestions, "rf_source", "^TNX")
            wacc_components = getattr(snapshot_for_suggestions, "wacc_components", {}) or {}
            we = wacc_components.get("equity_weight")
            wd = wacc_components.get("debt_weight")
            cost_of_equity_rate = wacc_components.get("cost_of_equity")
            rd = wacc_components.get("cost_of_debt_pre_tax")
            tax_rate = wacc_components.get("tax_rate")
            beta_text = f", beta {beta_val:.2f}" if beta_val else ""
            if all(v is not None for v in [we, wd, cost_of_equity_rate, rd, tax_rate]):
                inputs_line = (
                    f"We {we*100:.0f}% × Re {cost_of_equity_rate*100:.1f}% + "
                    f"Wd {wd*100:.0f}% × Rd {rd*100:.1f}% × (1-T {tax_rate*100:.1f}%) | "
                    f"Rf {rf_source}{beta_text}"
                )
            else:
                inputs_line = f"Rf {rf_source}{beta_text}"
            st.caption(f"Suggested WACC: {suggested_wacc:.1f}% | Inputs: {inputs_line}")

    with col_growth:
        user_fcf_growth = st.slider(
            "FCF Growth Rate (%)",
            min_value=0.0,
            max_value=25.0,
            value=default_fcf_growth,
            step=0.1,
            format="%.1f",
            key=f"fcf_growth_slider_{ticker}",
            help="Annual free-cash-flow growth for projection period."
        )
        if snapshot_for_suggestions and snapshot_for_suggestions.suggested_fcf_growth.value is not None:
            source_label = "fallback"
            if suggested_fcf_period_type == "forward_analyst_consensus":
                source_label = "analyst consensus"
            elif suggested_fcf_period_type == "forward_analyst_long_term":
                source_label = "analyst LT only"
            elif suggested_fcf_period_type == "trailing_historical":
                source_label = "trailing historical"
            elif suggested_fcf_period_type == "calculated_fallback":
                source_label = "historical fallback"

            quality_text = f", quality {suggested_fcf_reliability}/100" if suggested_fcf_reliability is not None else ""
            st.caption(f"Suggested: {suggested_fcf_growth:.1f}% ({source_label}{quality_text})")
            if stored_fcf_growth is None and suggested_fcf_reliability is not None and suggested_fcf_reliability < 65:
                st.caption("Default set to 8.0% because suggestion quality is low.")

    with col_terminal:
        terminal_growth_min = 0.0
        terminal_growth_max = max(0.5, min(6.0, round(user_wacc - 0.5, 1)))
        terminal_growth_key = f"terminal_growth_slider_{ticker}"
        if terminal_growth_key in st.session_state:
            st.session_state[terminal_growth_key] = min(
                max(st.session_state[terminal_growth_key], terminal_growth_min),
                terminal_growth_max
            )
        terminal_growth_default = min(max(default_terminal_growth, terminal_growth_min), terminal_growth_max)
        user_terminal_growth = st.slider(
            "Terminal Growth g (%)",
            min_value=terminal_growth_min,
            max_value=terminal_growth_max,
            value=terminal_growth_default,
            step=0.1,
            key=terminal_growth_key,
            help="Perpetual growth rate used in Gordon Growth terminal value."
        )
        st.caption("Fallback/default: 3.0% (if not set)")

    col_run, col_details = st.columns([1, 1])
    with col_run:
        if st.button("Run DCF Analysis", type="primary", key="run_dcf"):
            with st.spinner("Running DCF analysis with full verification..."):
                st.session_state.dcf_wacc = user_wacc
                st.session_state.dcf_fcf_growth = user_fcf_growth
                st.session_state.dcf_terminal_growth = user_terminal_growth
                st.session_state.dcf_terminal_scenario = "current"
                st.session_state.dcf_custom_multiple = None
                ui_adapter_result, engine_result, snapshot = run_dcf_analysis(
                    ticker,
                    user_wacc,
                    user_fcf_growth,
                    terminal_growth=user_terminal_growth,
                    terminal_scenario="current",
                    custom_multiple=None,
                )
                st.session_state.dcf_ui_adapter = ui_adapter_result
                st.session_state.dcf_engine_result = engine_result
                st.session_state.dcf_snapshot = snapshot
                _persist_dcf_result_for_context(
                    ticker=ticker,
                    end_date=st.session_state.get("end_date") or st.session_state.get("selected_end_date"),
                    num_quarters=st.session_state.get("num_quarters"),
                )
                st.rerun()

    with col_details:
        if st.session_state.get("dcf_ui_adapter"):
            if st.button("View DCF Details →", key="view_details"):
                st.session_state.show_dcf_details = True
                st.rerun()

    dcf_ui = st.session_state.get("dcf_ui_adapter")
    dcf_ui_data = dcf_ui.get_ui_data() if dcf_ui else None
    if dcf_ui_data:
        if not dcf_ui_data.get("success"):
            st.error("DCF analysis failed.")
            for err in dcf_ui_data.get("errors", []):
                st.error(f"• {err}")
        else:
            current_price = dcf_ui_data.get("current_price", 0)
            intrinsic = dcf_ui_data.get("price_per_share", 0)
            upside_downside = ((intrinsic - current_price) / current_price * 100) if (current_price and current_price > 0 and intrinsic and intrinsic > 0) else None

            col_ev, col_equity, col_intrinsic, col_quality = st.columns(4)
            with col_ev:
                ev = dcf_ui_data.get("enterprise_value", 0)
                st.metric("Enterprise Value", f"${ev/1e9:.1f}B" if ev >= 1e9 else (f"${ev/1e6:.1f}M" if ev >= 1e6 else "—"))
            with col_equity:
                equity = dcf_ui_data.get("equity_value", 0)
                st.metric("Equity Value", f"${equity/1e9:.1f}B" if equity >= 1e9 else (f"${equity/1e6:.1f}M" if equity >= 1e6 else "—"))
            with col_intrinsic:
                st.metric("Intrinsic Value/Share", f"${intrinsic:.2f}" if intrinsic else "—", delta=f"{upside_downside:+.1f}%" if upside_downside is not None else None)
            with col_quality:
                st.metric("Data Quality", f"{dcf_ui_data.get('data_quality_score', 0):.0f}/100")

            st.markdown("**Valuation Bridge**")
            st.dataframe(pd.DataFrame(dcf_ui.format_bridge_table()), use_container_width=True, hide_index=True)

            st.markdown("**Key Assumptions**")
            st.dataframe(pd.DataFrame(dcf_ui.format_assumptions_table()), use_container_width=True, hide_index=True)

            assumptions = dcf_ui_data.get("assumptions", {})
            price_exit = assumptions.get("price_exit_multiple")
            price_gordon = assumptions.get("price_gordon_growth")
            terminal_method = assumptions.get("terminal_value_method", "gordon_growth")
            high_growth_profile = bool(assumptions.get("high_growth_company", False))
            if terminal_method == "exit_multiple":
                if high_growth_profile:
                    st.caption("Terminal method in use: Exit Multiple (adaptive method for high-growth profiles).")
                else:
                    st.caption("Terminal method in use: Exit Multiple (adaptive fallback when Gordon is overly punitive).")
            else:
                st.caption("Terminal method in use: Gordon Growth (default for mature and steady-state valuation).")

            if price_exit and price_gordon:
                st.caption(f"Method outputs: Exit Multiple ${price_exit:.2f} | Gordon Growth ${price_gordon:.2f}.")

            with st.expander("Deep DCF Detail", expanded=False, icon="🧮"):
                st.caption("For full traceability use 'View DCF Details'.")
                st.markdown("**Current Financial Position (TTM)**")
                col_d1, col_d2, col_d3 = st.columns(3)
                with col_d1:
                    current_price_metric = dcf_ui_data["inputs"].get("current_price")
                    shares_metric = dcf_ui_data["inputs"].get("shares_outstanding")
                    market_cap_metric = dcf_ui_data["inputs"].get("market_cap")
                    st.caption(f"Price: {current_price_metric.formatted()}")
                    st.caption(f"Shares: {shares_metric.formatted()}")
                    st.caption(f"Market Cap: {market_cap_metric.formatted()}")
                with col_d2:
                    rev = dcf_ui_data["inputs"].get("ttm_revenue")
                    op_income = dcf_ui_data["inputs"].get("ttm_operating_income")
                    ebitda = dcf_ui_data["inputs"].get("ttm_ebitda")
                    st.caption(f"Revenue: {rev.formatted()}")
                    st.caption(f"Op Income: {op_income.formatted()}")
                    st.caption(f"EBITDA: {ebitda.formatted()}")
                with col_d3:
                    cfo = dcf_ui_data["inputs"].get("ttm_operating_cash_flow")
                    capex = dcf_ui_data["inputs"].get("ttm_capex")
                    debt = dcf_ui_data["inputs"].get("total_debt")
                    cash = dcf_ui_data["inputs"].get("cash")
                    st.caption(f"Oper. CF: {cfo.formatted()}")
                    st.caption(f"CapEx: {capex.formatted()}")
                    st.caption(f"Total Debt: {debt.formatted()}")
                    st.caption(f"Cash: {cash.formatted()}")

                projections = dcf_ui_data.get("fcf_projections", [])
                if projections:
                    st.markdown("**5-Year FCF Projection**")
                    proj_table = []
                    for proj in projections:
                        proj_table.append({
                            "Year": f"Year {proj.get('year', 0)}",
                            "FCF": f"${proj.get('fcf', 0)/1e9:.1f}B",
                            "PV(FCF)": f"${proj.get('pv', 0)/1e9:.1f}B"
                        })
                    st.dataframe(pd.DataFrame(proj_table), use_container_width=True, hide_index=True)
    else:
        st.info("No DCF output yet. Set assumptions and run the model.")

    # SECTION C: Business Momentum
    st.markdown('<div id="momentum"></div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="section-header"><span class="step-badge">Step 03</span><span class="section-title">Business Momentum</span></div>', unsafe_allow_html=True)
    st.caption(f"Source: {data_source}")

    col_h1, col_h2, col_h3, col_h4 = st.columns(4)
    with col_h1:
        avg_rev = growth_summary.get("avg_revenue_yoy")
        st.metric("Avg Revenue Growth (YoY)", f"{avg_rev:.1f}%" if avg_rev is not None else "N/A")
    with col_h2:
        avg_eps = growth_summary.get("avg_eps_yoy")
        st.metric("Avg EPS Growth (YoY)", f"{avg_eps:.1f}%" if avg_eps is not None else "N/A")
    with col_h3:
        quarters_with_revenue = growth_summary.get("quarters_with_revenue")
        quarters_with_eps = growth_summary.get("quarters_with_eps")
        quarters_plotted = 0
        if isinstance(quarters_with_revenue, int):
            quarters_plotted = max(quarters_plotted, quarters_with_revenue)
        if isinstance(quarters_with_eps, int):
            quarters_plotted = max(quarters_plotted, quarters_with_eps)
        if not quarters_plotted:
            fallback_samples = growth_summary.get("samples_used")
            quarters_plotted = fallback_samples if isinstance(fallback_samples, int) else 0
        st.metric("Quarters Plotted", quarters_plotted if quarters_plotted else "N/A")
    with col_h4:
        seasonality = seasonality_info.get("pattern", "N/A")
        st.metric("Seasonality Pattern", seasonality)

    total_quarters = growth_summary.get("samples_used")
    revenue_yoy_pairs = growth_summary.get("revenue_yoy_pairs")
    eps_yoy_pairs = growth_summary.get("eps_yoy_pairs")
    coverage_parts = []
    if isinstance(quarters_with_revenue, int) and isinstance(total_quarters, int):
        coverage_parts.append(f"Revenue points: {quarters_with_revenue}/{total_quarters}")
    if isinstance(quarters_with_eps, int) and isinstance(total_quarters, int):
        coverage_parts.append(f"EPS points: {quarters_with_eps}/{total_quarters}")
    if isinstance(revenue_yoy_pairs, int):
        coverage_parts.append(f"Revenue YoY pairs: {revenue_yoy_pairs}")
    if isinstance(eps_yoy_pairs, int):
        coverage_parts.append(f"EPS YoY pairs: {eps_yoy_pairs}")
    mismatch_points = source_diagnostics.get("mismatch_points")
    if isinstance(mismatch_points, int) and mismatch_points > 0:
        coverage_parts.append(f"Cross-source mismatches: {mismatch_points} (Yahoo-priority)")
    yahoo_collisions = source_diagnostics.get("yahoo_date_collisions_collapsed")
    sec_collisions = source_diagnostics.get("sec_date_collisions_collapsed")
    collision_total = 0
    if isinstance(yahoo_collisions, int):
        collision_total += yahoo_collisions
    if isinstance(sec_collisions, int):
        collision_total += sec_collisions
    if collision_total > 0:
        coverage_parts.append(f"Quarter-date collisions normalized: {collision_total}")
    sec_error = source_diagnostics.get("sec_error")
    if isinstance(sec_error, str) and sec_error.strip():
        sec_error_preview = sec_error.strip()
        if len(sec_error_preview) > 120:
            sec_error_preview = sec_error_preview[:117] + "..."
        coverage_parts.append(f"SEC unavailable: {sec_error_preview}")
    sec_q4_backfilled = source_diagnostics.get("sec_q4_backfilled", {})
    if isinstance(sec_q4_backfilled, dict):
        total_q4_derived = sec_q4_backfilled.get("total_q4_derived")
        if isinstance(total_q4_derived, int) and total_q4_derived > 0:
            derived_quarters = sec_q4_backfilled.get("quarters", []) if isinstance(sec_q4_backfilled.get("quarters", []), list) else []
            preview = ", ".join(derived_quarters[:3]) if derived_quarters else ""
            if len(derived_quarters) > 3:
                preview += ", ..."
            coverage_parts.append(
                f"SEC Q4 backfilled: {total_q4_derived}" + (f" ({preview})" if preview else "")
            )
    missing_revenue_quarters = data_coverage.get("missing_revenue_quarters", [])
    missing_eps_quarters = data_coverage.get("missing_eps_quarters", [])
    missing_report_quarters = data_coverage.get("missing_report_quarters", [])
    if missing_report_quarters:
        missing_report_display = ", ".join(missing_report_quarters[:3])
        if len(missing_report_quarters) > 3:
            missing_report_display += ", ..."
        coverage_parts.append(f"Missing report quarters: {missing_report_display}")
    if missing_revenue_quarters:
        missing_rev_display = ", ".join(missing_revenue_quarters[:3])
        if len(missing_revenue_quarters) > 3:
            missing_rev_display += ", ..."
        coverage_parts.append(f"Missing revenue: {missing_rev_display}")
    if missing_eps_quarters:
        missing_eps_display = ", ".join(missing_eps_quarters[:3])
        if len(missing_eps_quarters) > 3:
            missing_eps_display += ", ..."
        coverage_parts.append(f"Missing EPS: {missing_eps_display}")
    if coverage_parts:
        st.caption(" | ".join(coverage_parts))
    seasonality_reason = seasonality_info.get("reason")
    if seasonality_reason:
        st.caption(f"Seasonality method: {seasonality_reason}")

    if hist_data:
        loaded_quarters = len(hist_data)
        configured_quarters = st.session_state.get("num_quarters", loaded_quarters)
        if not isinstance(configured_quarters, int):
            configured_quarters = loaded_quarters
        available_dates = st.session_state.get("available_dates", []) or []
        selected_end_date = st.session_state.get("selected_end_date")
        context_available_quarters = len(available_dates)
        if available_dates and selected_end_date:
            date_values = [d.get("value") for d in available_dates if isinstance(d, dict)]
            if selected_end_date in date_values:
                selected_idx = date_values.index(selected_end_date)
                context_available_quarters = max(1, len(date_values) - selected_idx)

        source_total_quarters = 0
        if isinstance(source_diagnostics, dict):
            source_total_quarters = (
                source_diagnostics.get("merged_quarters")
                or source_diagnostics.get("yahoo_quarters")
                or source_diagnostics.get("sec_quarters")
                or 0
            )
        if not isinstance(source_total_quarters, int):
            source_total_quarters = 0

        if source_total_quarters > 0:
            max_context_quarters = min(context_available_quarters, source_total_quarters) if context_available_quarters else source_total_quarters
        else:
            max_context_quarters = context_available_quarters
        max_available_quarters = min(20, max(loaded_quarters, max_context_quarters or loaded_quarters))
        min_display_quarters = 4 if max_available_quarters >= 4 else 1

        pending_display_quarters = st.session_state.get("pending_momentum_display_quarters")
        if isinstance(pending_display_quarters, int):
            clamped_pending = min(
                max_available_quarters,
                max(min_display_quarters, pending_display_quarters),
            )
            st.session_state.momentum_display_quarters = clamped_pending
        st.session_state.pending_momentum_display_quarters = None

        current_display_quarters = st.session_state.get("momentum_display_quarters", configured_quarters)
        if not isinstance(current_display_quarters, int):
            current_display_quarters = configured_quarters
        current_display_quarters = min(max_available_quarters, max(min_display_quarters, current_display_quarters))
        st.session_state.momentum_display_quarters = current_display_quarters

        trend_col, rail_col = st.columns([6, 1], gap="medium")

        requested_quarters = current_display_quarters
        with rail_col:
            st.markdown("**Quarter Rail**")
            st.caption(f"Loaded: {loaded_quarters} | Max: {max_available_quarters}")
            if available_dates and selected_end_date:
                date_values = [d.get("value") for d in available_dates if isinstance(d, dict)]
                if selected_end_date in date_values:
                    selected_idx = date_values.index(selected_end_date)
                    if selected_idx > 0:
                        st.caption(f"Anchor end-date limits visible history to {context_available_quarters} quarter(s).")
            st.slider(
                "Display quarters",
                min_value=min_display_quarters,
                max_value=max_available_quarters,
                key="momentum_display_quarters",
                label_visibility="collapsed",
                help="Move the rail to request more history. If you move above loaded data, additional quarters are fetched automatically.",
            )
            requested_quarters = int(st.session_state.get("momentum_display_quarters", current_display_quarters))
            requested_quarters = min(max_available_quarters, max(min_display_quarters, requested_quarters))
            if requested_quarters > loaded_quarters:
                st.caption(f"Auto-loading +{requested_quarters - loaded_quarters} quarter(s)")
            else:
                st.caption("Using currently loaded history.")

        if requested_quarters > loaded_quarters:
            end_date_for_reload = st.session_state.get("end_date") or st.session_state.get("selected_end_date")
            with st.spinner(f"Loading up to {requested_quarters} quarters for {ticker}..."):
                expanded_analysis = cached_quarterly_analysis(ticker, requested_quarters, end_date_for_reload)
            expanded_hist_data = expanded_analysis.get("historical_trends", {}).get("quarterly_data", [])
            if len(expanded_hist_data) > loaded_quarters:
                st.session_state.quarterly_analysis = expanded_analysis
                st.session_state.num_quarters = requested_quarters
                st.session_state.pending_config_num_quarters = requested_quarters
                st.session_state.pending_momentum_display_quarters = min(requested_quarters, len(expanded_hist_data))

                financials = st.session_state.get("financials", {}) if isinstance(st.session_state.get("financials", {}), dict) else {}
                inc = financials.get("income", pd.DataFrame())
                bal = financials.get("balance", pd.DataFrame())
                cf = financials.get("cashflow", pd.DataFrame())
                qcf = financials.get("quarterly_cashflow", pd.DataFrame())
                st.session_state.comprehensive_analysis = calculate_comprehensive_analysis(
                    inc,
                    bal,
                    expanded_hist_data,
                    ticker,
                    cf,
                    qcf
                )
                st.rerun()

            st.session_state.pending_config_num_quarters = max(8, loaded_quarters)
            st.session_state.pending_momentum_display_quarters = loaded_quarters
            st.info("No additional quarters were returned for this ticker and selected ending report.")
            st.rerun()

        display_quarters = min(requested_quarters, loaded_quarters)
        hist_window = hist_data[:display_quarters]
        df_hist = pd.DataFrame(hist_window)
        quarter_order = list(reversed(df_hist["quarter"].tolist())) if not df_hist.empty and "quarter" in df_hist.columns else []
        if not df_hist.empty:
            df_hist = df_hist.iloc[::-1].reset_index(drop=True)

        def render_trend_chart(chart_source: pd.DataFrame, value_col: str, y_title: str, color: str):
            chart_data = chart_source[["quarter", value_col]].copy()
            if chart_data.empty or not chart_data[value_col].notna().any():
                st.caption("No data available for this trend.")
                return

            line = (
                alt.Chart(chart_data)
                .mark_line(point=True, strokeWidth=3, color=color)
                .encode(
                    x=alt.X(
                        "quarter:N",
                        sort=quarter_order if quarter_order else None,
                        axis=alt.Axis(title=None, labelAngle=-45, labelOverlap=False),
                    ),
                    y=alt.Y(f"{value_col}:Q", title=y_title),
                    tooltip=[
                        alt.Tooltip("quarter:N", title="Quarter"),
                        alt.Tooltip(f"{value_col}:Q", title=y_title, format=",.2f"),
                    ],
                )
            )
            st.altair_chart(line, use_container_width=True)

        with trend_col:
            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                st.caption("Revenue Trend")
                render_trend_chart(df_hist, "revenue", "Revenue (USD)", "#1f77b4")
            with col_chart2:
                st.caption("EPS Trend")
                render_trend_chart(df_hist, "eps", "EPS", "#1f77b4")

    st.markdown("**Fundamental Drivers (DuPont)**")
    dupont = comp_analysis.get("dupont", {}) if comp_analysis else {}
    if dupont:
        col_d1, col_d2, col_d3, col_d4 = st.columns(4)
        with col_d1:
            st.metric("ROE", f"{dupont.get('roe', 0):.1f}%")
        with col_d2:
            st.metric("Profit Margin", f"{dupont.get('net_profit_margin', 0):.1f}%")
        with col_d3:
            st.metric("Asset Turnover", f"{dupont.get('asset_turnover', 0):.2f}x")
        with col_d4:
            st.metric("Leverage (EM)", f"{dupont.get('equity_multiplier', 0):.2f}x")
    else:
        st.info("DuPont data unavailable.")

    with st.expander("Quarterly Raw Detail", expanded=False, icon="📋"):
        if hist_data:
            df_display = pd.DataFrame(hist_data).set_index("quarter")
            if "revenue" in df_display.columns:
                df_display["Revenue"] = df_display["revenue"].apply(
                    lambda x: f"${x/1e9:.2f}B" if x and x > 1e9 else (f"${x/1e6:.1f}M" if x else "N/A")
                )
            if "eps" in df_display.columns:
                df_display["EPS"] = df_display["eps"].apply(lambda x: f"${x:.2f}" if x else "N/A")
            cols_to_show = [c for c in ["Revenue", "EPS"] if c in df_display.columns]
            if cols_to_show:
                st.dataframe(df_display[cols_to_show], use_container_width=True)

        if growth_detail:
            st.caption("Growth Rates")
            df_growth = pd.DataFrame(growth_detail).set_index("quarter")
            for col in df_growth.columns:
                df_growth[col] = df_growth[col].apply(lambda x: f"{x:.1f}%" if x is not None else "—")
            st.dataframe(df_growth, use_container_width=True)

    # SECTION D: Street Context
    st.markdown('<div id="consensus"></div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="section-header"><span class="step-badge">Step 04</span><span class="section-title">Street Context</span></div>', unsafe_allow_html=True)

    consensus_citations = []
    qual_sources = []
    if consensus.get("error"):
        st.error(consensus["error"])
    elif consensus:
        next_q = consensus.get("next_quarter", {})
        coverage = consensus.get("analyst_coverage", {})
        targets = consensus.get("price_targets", {})
        consensus_citations = consensus.get("citations", [])
        qual_sources = consensus.get("qualitative_sources", [])

        quarter_label = next_q.get("quarter_label") or next_forecast_label
        st.markdown(f"**{quarter_label} Estimates**")
        col_c1, col_c2, col_c3, col_c4 = st.columns(4)
        with col_c1:
            st.metric("Revenue", next_q.get("revenue_estimate", "N/A"))
        with col_c2:
            st.metric("EPS", next_q.get("eps_estimate", "N/A"))
        with col_c3:
            st.metric("Analysts", coverage.get("num_analysts", "N/A"))
        with col_c4:
            buy = coverage.get("buy_ratings", 0) or 0
            hold = coverage.get("hold_ratings", 0) or 0
            sell = coverage.get("sell_ratings", 0) or 0
            total = buy + hold + sell
            st.metric("Buy/Hold/Sell", f"{buy}/{hold}/{sell}" if total > 0 else "N/A")

        estimate_sources = []
        if next_q.get("source"):
            estimate_sources.append(next_q.get("source"))
        if coverage.get("source") and coverage.get("source") not in estimate_sources:
            estimate_sources.append(coverage.get("source"))
        if estimate_sources:
            st.caption(f"Source: {', '.join(estimate_sources)}")

        if targets:
            col_t1, col_t2, col_t3 = st.columns(3)
            with col_t1:
                st.metric("Price Target (Low)", targets.get("low", "N/A"))
            with col_t2:
                st.metric("Price Target (Avg)", targets.get("average", "N/A"))
            with col_t3:
                st.metric("Price Target (High)", targets.get("high", "N/A"))
            if targets.get("source"):
                st.caption(f"Price target source: {targets.get('source')}")

        market_data = analysis.get("market_data", {})
        shares_outstanding = market_data.get("shares_outstanding")
        current_market_cap = market_data.get("market_cap")
        if targets and shares_outstanding:
            avg_pt = parse_price_value(targets.get("average"))
            high_pt = parse_price_value(targets.get("high"))
            low_pt = parse_price_value(targets.get("low"))

            def format_mcap(value):
                if value is None:
                    return "N/A"
                return f"${value/1e9:.1f}B"

            def format_delta(value):
                if value is None or not current_market_cap:
                    return None
                delta_pct = (value - current_market_cap) / current_market_cap * 100
                return f"{delta_pct:+.0f}%"

            col_i1, col_i2, col_i3 = st.columns(3)
            with col_i1:
                implied_low = low_pt * shares_outstanding if low_pt is not None else None
                st.metric("Implied Value (Low PT)", format_mcap(implied_low), delta=format_delta(implied_low))
            with col_i2:
                implied_avg = avg_pt * shares_outstanding if avg_pt is not None else None
                st.metric("Implied Value (Avg PT)", format_mcap(implied_avg), delta=format_delta(implied_avg))
            with col_i3:
                implied_high = high_pt * shares_outstanding if high_pt is not None else None
                st.metric("Implied Value (High PT)", format_mcap(implied_high), delta=format_delta(implied_high))
        elif targets and not shares_outstanding:
            st.caption("Implied total value unavailable (shares outstanding missing).")

        qualitative = consensus.get("qualitative_summary")
        if qualitative:
            st.markdown(f"**Analyst View:** {qualitative}")
        else:
            buy = coverage.get("buy_ratings", 0) or 0
            sell = coverage.get("sell_ratings", 0) or 0
            total = (coverage.get("buy_ratings", 0) or 0) + (coverage.get("hold_ratings", 0) or 0) + (coverage.get("sell_ratings", 0) or 0)
            if total > 0 and targets:
                avg_pt_val = parse_price_value(targets.get("average"))
                current_price = market_data.get("current_price", 0)
                if avg_pt_val and current_price:
                    upside = ((avg_pt_val - current_price) / current_price) * 100
                    direction = "bullish" if upside > 5 else ("neutral" if upside > -5 else "cautious")
                    st.markdown(f"**Analyst View:** Consensus is {direction} with {buy} buy ratings vs {sell} sell, targeting {upside:+.0f}% from current levels.")
    else:
        st.info("No consensus data available.")

    # SECTION E: AI Synthesis
    st.markdown('<div id="outlook"></div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="section-header"><span class="step-badge">Step 05</span><span class="section-title">AI Synthesis</span></div>', unsafe_allow_html=True)

    dcf_ui = st.session_state.get("dcf_ui_adapter")
    dcf_data_for_forecast = dcf_ui.get_ui_data() if dcf_ui else None

    if not dcf_ui:
        st.warning("Run DCF Analysis first for a more complete synthesis.")
    st.caption("Combines valuation, consensus, and historical momentum into a multi-horizon view.")

    has_existing_forecast = bool(st.session_state.get("independent_forecast"))
    outlook_button_label = "Regenerate Multi-Horizon Outlook" if has_existing_forecast else "Generate Multi-Horizon Outlook"
    if st.button(outlook_button_label, type="primary", key="generate_outlook"):
        with st.spinner("Analyzing data and generating multi-horizon outlook..."):
            dcf_hash = str(hash(str(dcf_data_for_forecast.get("price_per_share", 0)))) if dcf_data_for_forecast else ""
            # If a forecast already exists, include a nonce so rerun doesn't get stuck on a cached response.
            rerun_nonce = str(datetime.utcnow().timestamp()) if has_existing_forecast else ""
            data_hash = str(hash(str(st.session_state.quarterly_analysis.get("analysis_date", "")) + dcf_hash + rerun_nonce))
            forecast = cached_independent_forecast(
                ticker,
                data_hash,
                company_name=ticker,
                dcf_data=dcf_data_for_forecast
            )
            if isinstance(forecast, dict) and forecast.get("error"):
                st.session_state.ai_outlook_error = str(forecast.get("error"))
                if not st.session_state.get("independent_forecast"):
                    st.session_state.independent_forecast = forecast
                st.session_state.forecast_just_generated = False
            else:
                st.session_state.ai_outlook_error = None
                st.session_state.independent_forecast = forecast
                st.session_state.forecast_just_generated = True
                _persist_ai_result_for_context(
                    ticker=ticker,
                    end_date=st.session_state.get("end_date") or st.session_state.get("selected_end_date"),
                    num_quarters=st.session_state.get("num_quarters"),
                )
            st.rerun()

    ai_outlook_error = st.session_state.get("ai_outlook_error")
    if ai_outlook_error:
        if is_quota_or_rate_limit_error(ai_outlook_error):
            retry_text = estimate_api_retry_time(ai_outlook_error)
            st.warning(
                "AI Synthesis is temporarily unavailable because API request quota is exhausted.\n\n"
                f"Estimated retry/reset time: {retry_text}"
            )
        else:
            st.warning("AI Synthesis is temporarily unavailable. Please try again shortly.")
        with st.expander("AI Synthesis Error Details", expanded=False):
            st.code(str(ai_outlook_error))

    if st.session_state.independent_forecast:
        forecast = st.session_state.independent_forecast
        if forecast.get("error"):
            if not ai_outlook_error:
                if is_quota_or_rate_limit_error(forecast["error"]):
                    retry_text = estimate_api_retry_time(forecast["error"])
                    st.warning(
                        "AI Synthesis is temporarily unavailable because API request quota is exhausted.\n\n"
                        f"Estimated retry/reset time: {retry_text}"
                    )
                else:
                    st.warning("AI Synthesis is temporarily unavailable. Please try again shortly.")
                with st.expander("AI Synthesis Error Details", expanded=False):
                    st.code(str(forecast["error"]))
        else:
            extracted = forecast.get("extracted_forecast") or {}
            full_analysis = _sanitize_ai_valuation_language(forecast.get("full_analysis", "") or "")
            full_analysis = full_analysis.replace("$", "\\$")
            has_extracted = extracted and (extracted.get("short_term_stance") or extracted.get("fundamental_outlook"))
            expanded_default = bool(st.session_state.get("forecast_just_generated", False))

            if has_extracted:
                col_s1, col_s2, col_s3 = st.columns(3)
                with col_s1:
                    short_stance = extracted.get("short_term_stance", "Neutral")
                    short_emoji = {"Bullish": "📈", "Neutral": "➡️", "Bearish": "📉"}.get(short_stance, "➡️")
                    _sc1 = "stance-card-bull" if short_stance == "Bullish" else "stance-card-bear" if short_stance == "Bearish" else "stance-card-neut"
                    st.markdown(f"""
<div class="stance-card {_sc1}">
  <div style="font-size:11px; color:var(--clr-text-muted);">SHORT-TERM (0-12m)</div>
  <div style="font-size:18px; font-weight:600;">{short_emoji} {short_stance}</div>
</div>
                    """, unsafe_allow_html=True)
                with col_s2:
                    fund_outlook = extracted.get("fundamental_outlook", "Stable")
                    fund_emoji = {"Strong": "💪", "Stable": "➡️", "Weakening": "⚠️"}.get(fund_outlook, "➡️")
                    _sc2 = "stance-card-bull" if fund_outlook == "Strong" else "stance-card-neut" if fund_outlook == "Stable" else "stance-card-bear"
                    st.markdown(f"""
<div class="stance-card {_sc2}">
  <div style="font-size:11px; color:var(--clr-text-muted);">FUNDAMENTALS</div>
  <div style="font-size:18px; font-weight:600;">{fund_emoji} {fund_outlook}</div>
</div>
                    """, unsafe_allow_html=True)
                with col_s3:
                    stock_outlook = extracted.get("stock_outlook", "Neutral")
                    stock_emoji = {"Bullish": "📈", "Neutral": "➡️", "Bearish": "📉"}.get(stock_outlook, "➡️")
                    conv_level = extracted.get("fundamental_conviction", "Medium")
                    conv_badge = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}.get(conv_level, "🟡")
                    _sc3 = "stance-card-bull" if stock_outlook == "Bullish" else "stance-card-bear" if stock_outlook == "Bearish" else "stance-card-neut"
                    st.markdown(f"""
<div class="stance-card {_sc3}">
  <div style="font-size:11px; color:var(--clr-text-muted);">STOCK OUTLOOK {conv_badge}</div>
  <div style="font-size:18px; font-weight:600;">{stock_emoji} {stock_outlook}</div>
</div>
                    """, unsafe_allow_html=True)

                key_conditional = _sanitize_ai_valuation_language(extracted.get("key_conditional", ""))
                if key_conditional and "null" not in str(key_conditional).lower():
                    st.info(f"**Key Conditional:** {key_conditional}")

                evidence_gaps = extracted.get("evidence_gaps", [])
                if evidence_gaps:
                    sanitized_gaps = [
                        _sanitize_ai_valuation_language(g) for g in evidence_gaps
                        if g and "null" not in str(g).lower()
                    ]
                    gaps_text = " • ".join(sanitized_gaps)
                    if gaps_text:
                        st.caption(f"Evidence gaps: {gaps_text}")

                with st.expander("Full Analysis & Final Assessment", expanded=expanded_default, icon="📄"):
                    st.markdown(full_analysis.strip())
            else:
                if full_analysis:
                    with st.expander("Full Analysis & Final Assessment", expanded=expanded_default, icon="📄"):
                        st.markdown(full_analysis.strip())
                else:
                    st.warning("No analysis generated. Please try again.")

            st.caption(f"Generated: {forecast.get('forecast_date', 'Unknown')}")
            st.session_state.forecast_just_generated = False

    # SECTION F: Sources & Methodology
    st.markdown('<div id="sources"></div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="section-header"><span class="step-badge">Step 06</span><span class="section-title">Sources & Methodology</span></div>', unsafe_allow_html=True)
    st.caption("Reference material and citations for all report sections.")

    with st.expander("Methodology", expanded=False, icon="📚"):
        st.markdown("Core data sources and method notes used in this report:")
        ticker_for_url = st.session_state.get("ticker", "{ticker}")
        for src in SOURCE_CATALOG.values():
            url = src["url"].replace("{ticker}", ticker_for_url)
            st.markdown(
                f"**[{src['id']}]** **{src['label']}** — {src['description']}  \n"
                f"*Method: {src['method']}*  \n"
                f"[{url}]({url})"
            )

    with st.expander("Citations", expanded=False, icon="🔗"):
        if consensus_citations:
            for cite in consensus_citations:
                url = cite.get("url", "")
                if url:
                    st.markdown(f"- [{cite.get('source_name', 'Source')}]({url}) — {cite.get('data_type', '')}")
        else:
            st.markdown(f"- [Yahoo Finance](https://finance.yahoo.com/quote/{ticker}/analysis) — EPS & Revenue estimates, analyst ratings")

        if qual_sources:
            st.markdown("**Analyst Commentary**")
            for source in qual_sources[:5]:
                st.markdown(f"- _{source.get('headline', '')}_ ({source.get('source', '')}, {source.get('date', '')})")

else:
    st.info("Enter a ticker and click 'Load Data' to begin analysis.")
