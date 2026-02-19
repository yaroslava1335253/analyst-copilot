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
from datetime import datetime
from dotenv import load_dotenv

# Load API keys from .env file (if exists)
load_dotenv()
import pandas as pd
import json
from engine import get_financials, run_structured_prompt, calculate_metrics, run_chat, analyze_quarterly_trends, generate_independent_forecast, get_latest_date_info, get_available_report_dates, calculate_comprehensive_analysis
from data_adapter import DataAdapter, DataQualityMetadata, NormalizedFinancialSnapshot
from dcf_engine import DCFEngine, DCFAssumptions
from dcf_ui_adapter import DCFUIAdapter
from sources import SOURCE_CATALOG

UI_CACHE_VERSION = 1
UI_CACHE_PATH = Path(__file__).resolve().parent / "data" / "user_ui_cache.json"
MAX_TICKER_LIBRARY_SIZE = 100
TICKER_PATTERN = re.compile(r"^[A-Z0-9.\-]{1,10}$")
MAG7_TICKERS = ["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA"]
SNAPSHOT_METADATA_FIELDS = [
    "price",
    "shares_outstanding",
    "market_cap",
    "total_debt",
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
    "suggested_fcf_growth",
]


def _default_ui_cache() -> dict:
    return {
        "version": UI_CACHE_VERSION,
        "ticker_library": MAG7_TICKERS.copy(),
        "last_selected_ticker": "MSFT",
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


def load_ui_cache() -> dict:
    default = _default_ui_cache()
    if not UI_CACHE_PATH.exists():
        return default
    try:
        with UI_CACHE_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return default
        cache = {
            "version": data.get("version", UI_CACHE_VERSION),
            "ticker_library": _normalize_ticker_library(data.get("ticker_library", [])),
            "last_selected_ticker": _normalize_ticker(data.get("last_selected_ticker", "MSFT")) or "MSFT",
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
            "ticker_library": _normalize_ticker_library(cache_obj.get("ticker_library", [])),
            "last_selected_ticker": _normalize_ticker(cache_obj.get("last_selected_ticker", "MSFT")) or "MSFT",
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

# --- Cached API Functions ---
# These decorators cache results so API calls only happen once per input
# TTL (time-to-live) of 1 hour = 3600 seconds

@st.cache_data(ttl=3600, show_spinner=False)
def cached_quarterly_analysis(ticker: str, num_quarters: int = 8, end_date: str = None) -> dict:
    """Cached version of analyze_quarterly_trends to avoid API rate limits."""
    return analyze_quarterly_trends(ticker, num_quarters, end_date)

@st.cache_data(ttl=3600, show_spinner=False)
def cached_available_dates(ticker: str) -> list:
    """Cached wrapper for get_available_report_dates."""
    return get_available_report_dates(ticker)

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
    return get_financials(ticker)

@st.cache_data(ttl=3600, show_spinner=False)
def cached_latest_date_info(ticker: str) -> dict:
    """Cached wrapper for get_latest_date_info from engine."""
    return get_latest_date_info(ticker)

@st.cache_data(ttl=3600, show_spinner=False)
def cached_financial_snapshot(ticker: str):
    """Cached wrapper to get financial snapshot for suggested assumptions."""
    try:
        adapter = DataAdapter(ticker)
        snapshot = adapter.fetch()
        return snapshot
    except Exception:
        return None

def run_dcf_analysis(ticker: str, wacc: float = None, fcf_growth: float = None, 
                     terminal_scenario: str = "current", custom_multiple: float = None) -> tuple:
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
        
        # Set terminal multiple scenario
        assumptions.terminal_multiple_scenario = terminal_scenario
        if terminal_scenario == "custom" and custom_multiple is not None:
            assumptions.exit_multiple = custom_multiple
        
        engine = DCFEngine(snapshot, assumptions)
        engine_result = engine.run()
        ui_adapter = DCFUIAdapter(engine_result, snapshot)
        return (ui_adapter, engine_result, snapshot)
    except Exception as e:
        return (None, {"success": False, "error": str(e), "errors": [str(e)]}, None)


def _show_dcf_details_page():
    """Render the full DCF Details page with spreadsheet-like layout.
    
    This page shows complete calculation breakdown with all inputs, assumptions,
    and intermediate steps for full audit traceability.
    """
    st.markdown("---")
    st.subheader("DCF Valuation Details")
    
    # Back button
    if st.button("← Back to Summary", key="back_from_details_top"):
        st.session_state.show_dcf_details = False
        st.rerun()
    
    ui_adapter = st.session_state.dcf_ui_adapter
    ui_data = ui_adapter.get_ui_data()
    engine_result = st.session_state.dcf_engine_result
    
    st.caption("Complete calculation breakdown with all inputs, assumptions, and intermediate steps")
    
    # INPUTS TABLE
    st.markdown("### Input Data")
    st.caption("All financial data fetched from yfinance with quality assessment")
    
    inputs_table = ui_adapter.format_input_table()
    df_inputs = pd.DataFrame(inputs_table)
    st.dataframe(df_inputs, use_container_width=True, hide_index=True)
    
    with st.expander("Input Data Legend", icon="ℹ️"):
        st.markdown("""
        - **Value**: Actual number or "—" if missing/zero
        - **Units**: USD (currency), shares (count), % (percentage), x (multiple)
        - **Period**: TTM (trailing 12 months), annual, quarterly
        - **Source**: yfinance field path for reproducibility
        - **Reliability**: Score out of 100 (higher = more confidence)
        - **Notes**: Fallback reasons (e.g., "quarterly unavailable, used annual")
        """)
    
    # ASSUMPTIONS TABLE
    st.markdown("### Valuation Assumptions")
    assumptions_table = ui_adapter.format_assumptions_table()
    df_assumptions = pd.DataFrame(assumptions_table)
    st.dataframe(df_assumptions, use_container_width=True, hide_index=True)
    
    # CAPM / WACC CALCULATION DETAILS
    st.markdown("### CAPM & WACC Calculation")
    st.caption("Cost of equity estimation using Capital Asset Pricing Model")
    
    snapshot = st.session_state.get('dcf_snapshot')
    if snapshot:
        # CAPM inputs
        beta = snapshot.beta.value
        
        # Get dynamic risk-free rate from snapshot (fetched from ^TNX)
        RISK_FREE_RATE = getattr(snapshot, 'risk_free_rate', 0.045)
        RF_SOURCE = getattr(snapshot, 'rf_source', '10-Year Treasury (^TNX)')
        MARKET_RISK_PREMIUM = 0.05  # Damodaran implied ERP (5.0%)
        DAMODARAN_DATE = "January 2026"
        
        col_capm1, col_capm2, col_capm3 = st.columns(3)
        
        with col_capm1:
            st.metric("Risk-Free Rate (Rf)¹¹", f"{RISK_FREE_RATE*100:.1f}%")
            st.caption(RF_SOURCE)

        with col_capm2:
            st.metric("Market Risk Premium (ERP)¹²", f"{MARKET_RISK_PREMIUM*100:.1f}%")
            st.caption(f"Damodaran Implied ERP ({DAMODARAN_DATE})")

        with col_capm3:
            if beta:
                st.metric("Beta (β)¹⁰", f"{beta:.2f}")
                st.caption("Yahoo Finance (5Y monthly)")
            else:
                st.metric("Beta (β)¹⁰", "N/A")
                st.caption("Not available")
        
        # Show CAPM calculation
        if beta:
            cost_of_equity = RISK_FREE_RATE + beta * MARKET_RISK_PREMIUM
            st.markdown("---")
            
            col_formula, col_result = st.columns([2, 1])
            with col_formula:
                st.markdown(f"""
                **CAPM Formula:**
                ```
                Cost of Equity = Rf + β × (Rm - Rf)
                              = {RISK_FREE_RATE*100:.1f}% + {beta:.2f} × {MARKET_RISK_PREMIUM*100:.1f}%
                              = {cost_of_equity*100:.1f}%
                ```
                """)
            
            with col_result:
                st.metric("Cost of Equity", f"{cost_of_equity*100:.1f}%")
                st.caption("CAPM result")
            
            # Compare to used WACC
            used_wacc = ui_data.get('assumptions', {}).get('wacc', 0.09)
            
            if abs(cost_of_equity - used_wacc) > 0.005:
                delta_wacc = (used_wacc - cost_of_equity) * 100
                st.info(f"**Used WACC: {used_wacc*100:.1f}%** — User adjusted by {delta_wacc:+.1f}% from CAPM suggestion")
            else:
                st.success(f"**Used WACC: {used_wacc*100:.1f}%** — Matches CAPM cost of equity")
        
        # Sources and methodology
        with st.expander("CAPM Sources & Methodology (Sources [9–12])", icon="📚"):
            st.markdown(f"""
            **Data Sources (all traceable):**
            - **Beta ({beta:.2f})**: Yahoo Finance — 5-year monthly returns vs S&P 500
            - **Risk-Free Rate ({RISK_FREE_RATE*100:.1f}%)**: 10-year U.S. Treasury yield
              - Source: [FRED DGS10](https://fred.stlouisfed.org/series/DGS10) (Feb 2026 ~4.5%)
            - **Implied Equity Risk Premium ({MARKET_RISK_PREMIUM*100:.1f}%)**: Damodaran, NYU Stern ({DAMODARAN_DATE})
              - Source: [pages.stern.nyu.edu/~adamodar/](https://pages.stern.nyu.edu/~adamodar/)
              - Using *implied* ERP (forward-looking), not historical average
            
            **CAPM Calculation:**
            ```
            Cost of Equity = Rf + β × ERP
                           = {RISK_FREE_RATE*100:.1f}% + {beta:.2f} × {MARKET_RISK_PREMIUM*100:.1f}%
                           = {cost_of_equity*100:.1f}%
            ```
            
            **Simplifications:**
            - This model uses Cost of Equity as a proxy for WACC
            - For companies with significant debt, a full WACC calculation would include:
              - Cost of Debt × (1 - Tax Rate) × Debt Weight
              - Cost of Equity × Equity Weight
            - Large-cap tech companies typically have minimal debt, so Cost of Equity ≈ WACC
            
            **Beta Interpretation:**
            - β = 1.0: Moves with the market
            - β > 1.0: More volatile than market (higher risk, higher expected return)
            - β < 1.0: Less volatile than market (lower risk, lower expected return)
            """)
    else:
        st.warning("Snapshot data not available for CAPM details.")
    
    # 5-YEAR FCF PROJECTION
    # Get FCFF method for the header badge
    fcff_method_for_header = ui_data.get('assumptions', {}).get('fcff_method', 'unknown')
    fcff_reliability_for_header = ui_data.get('assumptions', {}).get('fcff_reliability', 0)
    
    fcff_header_badges = {
        'proper_fcff': ('✅ Proper FCFF', '95%' if fcff_reliability_for_header >= 90 else f'{fcff_reliability_for_header}%'),
        'approx_unlevered': ('📊 Approx Unlevered', f'{fcff_reliability_for_header}%' if fcff_reliability_for_header else '70%'),
        'unlevered_proxy': ('📊 Approx Unlevered', f'{fcff_reliability_for_header}%' if fcff_reliability_for_header else '70%'),
        'levered_proxy': ('⚠️ Levered Proxy', f'{fcff_reliability_for_header}%' if fcff_reliability_for_header else '50%')
    }
    badge_label, badge_reliability = fcff_header_badges.get(fcff_method_for_header, ('Unknown', ''))
    
    assumptions = ui_data.get('assumptions', {})
    
    # Get horizon info
    forecast_years = assumptions.get('forecast_years', 5)
    display_years = assumptions.get('display_years', 5)
    is_large_cap = assumptions.get('is_large_cap', False)
    horizon_reason = assumptions.get('horizon_reason', 'standard')
    
    # Dynamic header based on horizon
    horizon_label = f"{forecast_years}-Year Projection"
    
    st.markdown(f"### {horizon_label} & Present Value")
    
    projections = ui_data.get("fcf_projections", [])
    use_driver_model = assumptions.get('use_driver_model', False)
    yearly_projections = assumptions.get('yearly_projections', [])
    
    if projections:
        # Check if we have driver-based projections (textbook DCF)
        if use_driver_model and yearly_projections:
            # ===== DRIVER-BASED PROJECTION TABLE (TEXTBOOK DCF) =====
            analyst_anchors_used = assumptions.get('analyst_fcf_anchors_used', False)
            if analyst_anchors_used:
                st.caption(
                    "📚 **Textbook DCF**: Revenue → EBIT → NOPAT → Reinvestment → FCFF  \n"
                    "FCF Years 1–3: analyst revenue consensus¹ × TTM FCF margin⁴  |  "
                    "FCF Years 4–10: FCFF driver model²  |  Growth rates: Damodaran fade⁶"
                )
            else:
                st.caption(
                    "📚 **Textbook DCF**: Revenue → EBIT → NOPAT → Reinvestment → FCFF  \n"
                    "FCF: FCFF driver model² (no analyst estimates available)  |  "
                    "Growth rates: Damodaran smooth fade⁶"
                )
            
            # Show growth fade and ROIC info
            near_term_g = assumptions.get('near_term_growth_rate', 0)
            stable_g = assumptions.get('stable_growth_rate', 0)
            current_roic = assumptions.get('base_roic', 0)
            terminal_roic = assumptions.get('terminal_roic', 0)
            industry_roic = assumptions.get('industry_roic', 0)
            terminal_reinv_rate = assumptions.get('terminal_reinvestment_rate', 0)
            
            st.info(
                f"**Growth Fade**⁶: {near_term_g:.1%} → Year {forecast_years}: {stable_g:.1%} (g_perp)⁸  \n"
                f"**ROIC Fade**⁷: Current {current_roic:.1%} → Terminal {terminal_roic:.1%} (industry: {industry_roic:.1%})  \n"
                f"**Terminal Reinvestment**⁶: {terminal_reinv_rate:.1%} = g_perp / ROIC_terminal"
            )
            
            # Full driver table (show display_years, but compute all)
            proj_table = []
            display_projs = yearly_projections[:display_years] if len(yearly_projections) > display_years else yearly_projections
            for proj in display_projs:
                fcf_src = proj.get('fcf_source', 'driver_model')
                src_badge = "[¹]" if fcf_src == "analyst_revenue_estimate" else "[²]"
                proj_table.append({
                    "Year": f"Y{proj.get('year', 0)}",
                    "Revenue³": f"${proj.get('revenue', 0)/1e9:.1f}B",
                    "Growth⁶": f"{proj.get('revenue_growth', 0):.1%}",
                    "EBIT Margin⁵": f"{proj.get('ebit_margin', 0):.1%}",
                    "EBIT": f"${proj.get('ebit', 0)/1e9:.1f}B",
                    "NOPAT": f"${proj.get('nopat', 0)/1e9:.1f}B",
                    "Reinvest⁶": f"${proj.get('reinvestment', 0)/1e9:.1f}B",
                    "Reinv Rate⁶": f"{proj.get('reinvestment_rate', 0):.0%}",
                    "FCFF⁴": f"${proj.get('fcff', 0)/1e9:.1f}B {src_badge}",
                    "PV(FCFF)": f"${proj.get('pv_fcff', 0)/1e9:.1f}B"
                })
            
            df_proj = pd.DataFrame(proj_table)
            st.dataframe(df_proj, use_container_width=True, hide_index=True)
            
            # Show years 6-10 in expander if 10-year horizon
            if forecast_years == 10 and len(yearly_projections) > 5:
                with st.expander(f"Years 6–10 (Fade to Terminal)", icon="📈"):
                    fade_table = []
                    for proj in yearly_projections[5:]:
                        fcf_src = proj.get('fcf_source', 'driver_model')
                        src_badge = "[¹]" if fcf_src == "analyst_revenue_estimate" else "[²]"
                        fade_table.append({
                            "Year": f"Y{proj.get('year', 0)}",
                            "Revenue³": f"${proj.get('revenue', 0)/1e9:.1f}B",
                            "Growth⁶": f"{proj.get('revenue_growth', 0):.1%}",
                            "NOPAT": f"${proj.get('nopat', 0)/1e9:.1f}B",
                            "Reinv Rate⁶": f"{proj.get('reinvestment_rate', 0):.0%}",
                            "FCFF⁴": f"${proj.get('fcff', 0)/1e9:.1f}B {src_badge}",
                            "PV(FCFF)": f"${proj.get('pv_fcff', 0)/1e9:.1f}B"
                        })
                    st.dataframe(pd.DataFrame(fade_table), use_container_width=True, hide_index=True)
            
            # Show formulas below
            with st.expander("Driver Formulas", icon="📐"):
                st.markdown("""
                | Driver | Formula |
                |--------|---------|
                | Revenue | `Revenue_t = Revenue_{t-1} × (1 + g_t)` |
                | EBIT | `EBIT_t = Revenue_t × EBIT_margin` |
                | NOPAT | `NOPAT_t = EBIT_t × (1 - tax_rate)` |
                | Reinvestment | `Reinvestment_t = ΔRevenue_t / Sales_to_Capital` |
                | FCFF | `FCFF_t = NOPAT_t - Reinvestment_t` |
                | PV(FCFF) | `PV_t = FCFF_t / (1 + WACC)^t` |
                """)
                
        else:
            # ===== LEGACY SIMPLE GROWTH TABLE =====
            proj_table = []
            for i, proj in enumerate(projections):
                growth_str = "—"
                if i > 0 and projections[i-1].get('fcf', 0) > 0:
                    growth = ((proj.get('fcf', 0) / projections[i-1].get('fcf', 1)) - 1) * 100
                    growth_str = f"{growth:.1f}%"
                proj_table.append({
                    "Year": f"Year {proj.get('year', 0)}",
                    "FCF ($B)": f"${proj.get('fcf', 0)/1e9:.1f}",
                    "Growth": growth_str,
                    "Discount Factor": f"{proj.get('discount_factor', 0):.4f}",
                    "PV(FCF) ($B)": f"${proj.get('pv', 0)/1e9:.1f}",
                    "Formula": "FCF_t / (1+WACC)^t"
                })
            df_proj = pd.DataFrame(proj_table)
            st.dataframe(df_proj, use_container_width=True, hide_index=True)
        
        pv_sum = sum([p.get('pv', 0) for p in projections])
    
    # TERMINAL VALUE CALCULATION
    st.markdown("### Terminal Value Calculation⁸ ¹³")
    tv_yearN = ui_data.get('terminal_value_yearN', 0)
    pv_tv = ui_data.get('pv_terminal_value', 0)
    assumptions = ui_data.get('assumptions', {})
    ev = ui_data.get('enterprise_value', 1)
    
    tv_method = assumptions.get('terminal_value_method', 'exit_multiple')
    
    # Show method-specific calculation details
    if tv_method == "exit_multiple":
        # Exit Multiple Method with current-anchored default
        exit_multiple = assumptions.get('exit_multiple', 15)
        current_ev_ebitda = assumptions.get('current_ev_ebitda')
        industry_ev_ebitda = assumptions.get('industry_ev_ebitda')
        damodaran_industry = assumptions.get('damodaran_industry', 'N/A')
        yf_industry = assumptions.get('yf_industry', 'N/A')
        is_exact_match = assumptions.get('is_exact_industry_match', False)
        multiple_source = assumptions.get('terminal_multiple_source', 'unknown')
        
        st.markdown("#### Exit Multiple Method (EV/EBITDA)")
        
        # Terminal Multiple Transparency Section
        st.markdown("##### Terminal Multiple Analysis")
        
        # Key metrics in columns
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            if current_ev_ebitda is not None:
                st.metric("Current EV/EBITDA", f"{current_ev_ebitda:.1f}x")
                st.caption("Company trading multiple")
            else:
                st.metric("Current EV/EBITDA", "N/A")
                st.caption("Cannot compute")
        with col_m2:
            if industry_ev_ebitda is not None:
                st.metric("Industry EV/EBITDA", f"{industry_ev_ebitda:.1f}x")
                st.caption(f"Damodaran: {damodaran_industry[:25]}...")
            else:
                st.metric("Industry EV/EBITDA", "N/A")
                st.caption("Not applicable")
        with col_m3:
            # Show terminal multiple with scenario info
            terminal_scenario = assumptions.get('terminal_multiple_scenario', 'current')
            rerating_pct = assumptions.get('terminal_multiple_rerating_pct')
            
            scenario_labels = {
                'current': 'Current',
                'industry': 'Industry', 
                'blended': 'Blended',
                'custom': 'Custom'
            }
            scenario_label = scenario_labels.get(terminal_scenario, terminal_scenario)
            
            if rerating_pct is not None and abs(rerating_pct) > 0.5:
                st.metric("Terminal EV/EBITDA (Used)", f"{exit_multiple:.1f}x", 
                         delta=f"{rerating_pct:+.1f}% rerating")
                st.caption(f"Scenario: {scenario_label}")
            else:
                st.metric("Terminal EV/EBITDA (Used)", f"{exit_multiple:.1f}x")
                st.caption(f"Scenario: {scenario_label} (no rerating)")
        
        # Show cash conversion analysis - BOTH TTM and Terminal
        observed_fcff_ebitda_ttm = assumptions.get('observed_fcff_ebitda_ttm')
        terminal_fcff_ebitda = assumptions.get('terminal_year_fcff_ebitda')  # From Year N projection
        required_fcff_ebitda = assumptions.get('required_fcff_ebitda_for_exit')
        consistent_multiple = assumptions.get('consistent_exit_multiple')
        
        if observed_fcff_ebitda_ttm or terminal_fcff_ebitda:
            st.markdown("##### Cash Conversion Analysis")
            col_cc1, col_cc2, col_cc3, col_cc4 = st.columns(4)
            
            with col_cc1:
                if observed_fcff_ebitda_ttm:
                    st.metric("TTM FCFF/EBITDA", f"{observed_fcff_ebitda_ttm*100:.0f}%")
                    st.caption("Current (today)")
                else:
                    st.metric("TTM FCFF/EBITDA", "N/A")
            
            with col_cc2:
                if terminal_fcff_ebitda:
                    st.metric("Terminal FCFF/EBITDA", f"{terminal_fcff_ebitda*100:.0f}%")
                    st.caption(f"Year {assumptions.get('forecast_years', 10)} forecast")
                else:
                    st.metric("Terminal FCFF/EBITDA", "N/A")
                    st.caption("Not projected")
            
            with col_cc3:
                if consistent_multiple:
                    st.metric("Consistent Multiple", f"{consistent_multiple:.1f}x")
                    st.caption("From economics")
                else:
                    st.metric("Consistent Multiple", "N/A")
            
            with col_cc4:
                if required_fcff_ebitda:
                    # Color-code based on plausibility
                    comparison_conversion = terminal_fcff_ebitda if terminal_fcff_ebitda else observed_fcff_ebitda_ttm
                    if required_fcff_ebitda > 0.85:
                        st.metric("Required", f"{required_fcff_ebitda*100:.0f}% 🔴")
                        st.caption("Implausible (>85%)")
                    elif comparison_conversion and required_fcff_ebitda > comparison_conversion + 0.15:
                        st.metric("Required", f"{required_fcff_ebitda*100:.0f}% ⚠️")
                        st.caption("Above forecast")
                    else:
                        st.metric("Required", f"{required_fcff_ebitda*100:.0f}% ✓")
                        st.caption("Reasonable")
                else:
                    st.metric("Required", "N/A")
        
        # Industry classification
        match_badge = "✅ Exact Match" if is_exact_match else "⚡ Approximate Match"
        
        with st.expander("Industry Classification & Methodology", icon="📚"):
            st.markdown(f"""
            **Industry Mapping:**
            - Yahoo Finance Industry: `{yf_industry}`
            - Damodaran Industry: `{damodaran_industry}` ({match_badge})
            
            📊 [View Damodaran Industry Multiples](https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/vebitda.html) | *Data as of January 2026*
            
            ---
            
            **Terminal Multiple Scenarios:**
            - **Current**: Uses company's current EV/EBITDA (no rerating assumption)
            - **Industry**: Uses Damodaran industry median
            - **Blended**: 70% current + 30% industry (partial mean reversion)
            - **Custom**: User-specified multiple
            
            **Cash Conversion Diagnostic:**
            - Required FCFF/EBITDA = Exit Multiple × (WACC − g)
            - If required >> observed, your multiple implies implausible efficiency gains
            - >85% cash conversion is economically unrealistic for most businesses
            """)
        
        # Calculation breakdown
        st.markdown("#### Calculation Steps")
        
        # Get TTM EBITDA and project to Year N
        ttm_ebitda_raw = ui_data.get('inputs', {}).get('ttm_ebitda', 0)
        # Handle FinancialMetric object, dict, or raw value
        if hasattr(ttm_ebitda_raw, 'value'):
            ttm_ebitda = ttm_ebitda_raw.value or 0
        elif isinstance(ttm_ebitda_raw, dict):
            ttm_ebitda = ttm_ebitda_raw.get('value', 0) or 0
        else:
            ttm_ebitda = ttm_ebitda_raw or 0
        
        fcf_growth = assumptions.get('fcf_growth_rate', 0.08)
        forecast_years = assumptions.get('forecast_years', 5)
        wacc = assumptions.get('wacc', 0.08)
        
        year_n_ebitda = ttm_ebitda * ((1 + fcf_growth) ** forecast_years) if ttm_ebitda else 0
        
        # Build step 4 description based on multiple source
        if multiple_source == "current":
            step4_formula = f"Current EV/EBITDA (no rerating)"
        elif multiple_source == "industry_fallback":
            step4_formula = f"Industry fallback: {damodaran_industry}"
        else:
            step4_formula = f"Current company trading multiple"
        
        calc_table = [
            {"Step": "1. TTM EBITDA", "Formula": "From financial statements", "Value": f"${ttm_ebitda/1e9:.2f}B" if ttm_ebitda else "N/A"},
            {"Step": "2. EBITDA Growth Rate", "Formula": "Using FCF growth as proxy", "Value": f"{fcf_growth*100:.1f}%"},
            {"Step": f"3. Year {forecast_years} EBITDA", "Formula": f"TTM EBITDA × (1 + g)^{forecast_years}", "Value": f"${year_n_ebitda/1e9:.2f}B" if year_n_ebitda else "N/A"},
            {"Step": "4. Terminal Multiple", "Formula": step4_formula, "Value": f"{exit_multiple}x EV/EBITDA"},
            {"Step": f"5. Terminal Value (Y{forecast_years})", "Formula": f"Year {forecast_years} EBITDA × Terminal Multiple", "Value": f"${tv_yearN/1e9:.1f}B"},
            {"Step": "6. Discount Factor", "Formula": f"1 / (1 + WACC)^{forecast_years}", "Value": f"{1 / (1 + wacc)**forecast_years:.4f}"},
            {"Step": "7. PV(Terminal Value)", "Formula": "Terminal Value × Discount Factor", "Value": f"${pv_tv/1e9:.1f}B"},
        ]
        df_calc = pd.DataFrame(calc_table)
        st.dataframe(df_calc, use_container_width=True, hide_index=True)
        
    else:
        # Gordon Growth Method
        st.markdown("##### Gordon Growth Model")
        
        terminal_growth = assumptions.get('terminal_growth_rate', 0.03)
        wacc = assumptions.get('wacc', 0.08)
        forecast_years = assumptions.get('forecast_years', 5)
        
        # Get Year N FCF from projections
        year_n_fcf = projections[-1].get('fcf', 0) if projections else 0
        terminal_fcf = year_n_fcf * (1 + terminal_growth)
        n_plus_1 = forecast_years + 1
        
        calc_table = [
            {"Step": f"1. Year {forecast_years} FCF", "Formula": f"From {forecast_years}-year projection", "Value": f"${year_n_fcf/1e9:.2f}B"},
            {"Step": "2. Terminal Growth Rate (g)", "Formula": "Long-term GDP growth proxy", "Value": f"{terminal_growth*100:.1f}%"},
            {"Step": f"3. Terminal FCF (Year {n_plus_1})", "Formula": f"Year {forecast_years} FCF × (1 + g)", "Value": f"${terminal_fcf/1e9:.2f}B"},
            {"Step": "4. WACC", "Formula": "Weighted average cost of capital", "Value": f"{wacc*100:.1f}%"},
            {"Step": "5. Terminal Value", "Formula": "Terminal FCF / (WACC - g)", "Value": f"${tv_yearN/1e9:.1f}B"},
            {"Step": "6. Discount Factor", "Formula": f"1 / (1 + WACC)^{forecast_years}", "Value": f"{1 / (1 + wacc)**forecast_years:.4f}"},
            {"Step": "7. PV(Terminal Value)", "Formula": "Terminal Value × Discount Factor", "Value": f"${pv_tv/1e9:.1f}B"},
        ]
        df_calc = pd.DataFrame(calc_table)
        st.dataframe(df_calc, use_container_width=True, hide_index=True)
        
        st.info(f"""
        **Gordon Growth Formula:** TV = FCF_({n_plus_1}) / (WACC - g) = ${terminal_fcf/1e9:.2f}B / ({wacc*100:.1f}% - {terminal_growth*100:.1f}%) = ${tv_yearN/1e9:.1f}B
        """)
    
    # ===== DUAL TV CROSS-CHECK SECTION =====
    # ===== SECTION 1: INTRINSIC VALUE (GORDON GROWTH) =====
    st.markdown("#### Intrinsic Value (Gordon Growth)")
    
    # Get all inputs for calculation trace
    pv_tv_exit = assumptions.get('pv_tv_exit_multiple')
    pv_tv_gordon = assumptions.get('pv_tv_gordon_growth')
    tv_exit_raw = assumptions.get('tv_exit_multiple')  # Undiscounted TV
    tv_gordon_raw = assumptions.get('tv_gordon_growth')  # Undiscounted TV
    price_exit = assumptions.get('price_exit_multiple')
    price_gordon = assumptions.get('price_gordon_growth')
    
    # Get underlying assumptions for trace
    exit_multiple_used = assumptions.get('exit_multiple', 0)
    terminal_growth = assumptions.get('terminal_growth_rate', 0.03)
    fcf_growth = assumptions.get('fcf_growth_rate', 0.10)
    wacc_used = assumptions.get('wacc', 0.10)
    forecast_years = assumptions.get('forecast_years', 5)
    tv_method_used = assumptions.get('terminal_value_method', 'exit_multiple')
    
    # Get TTM EBITDA and FCF from snapshot stored in session_state
    ttm_ebitda = None
    ttm_fcf = None
    if snapshot and hasattr(snapshot, 'ttm_ebitda') and snapshot.ttm_ebitda:
        ttm_ebitda = snapshot.ttm_ebitda.value
    if snapshot and hasattr(snapshot, 'ttm_fcf') and snapshot.ttm_fcf:
        ttm_fcf = snapshot.ttm_fcf.value
    
    if pv_tv_gordon:
        # Get TTM FCFF from the proper FCFF calculation
        ttm_fcff = assumptions.get('ttm_fcff')
        fcff_to_use = ttm_fcff if ttm_fcff else ttm_fcf
        
        # Get ACTUAL Year N FCFF from driver-based projections
        fcf_projections = ui_data.get("fcf_projections", [])
        if fcf_projections:
            year_n_fcf_actual = fcf_projections[-1].get('fcf', 0) or fcf_projections[-1].get('fcff', 0)
        else:
            year_n_fcf_actual = None
        
        # Calculate terminal FCF from the actual Year N value
        if year_n_fcf_actual and terminal_growth:
            terminal_fcf_actual = year_n_fcf_actual * (1 + terminal_growth)
        elif tv_gordon_raw and (wacc_used - terminal_growth) > 0:
            terminal_fcf_actual = tv_gordon_raw * (wacc_used - terminal_growth)
            year_n_fcf_actual = terminal_fcf_actual / (1 + terminal_growth) if terminal_growth else terminal_fcf_actual
        else:
            terminal_fcf_actual = None

        n_plus_1 = forecast_years + 1
        
        # Determine if we're using proper FCFF
        using_proper_fcff = ttm_fcff is not None and ttm_fcff > 0
        fcff_label = "FCFF" if using_proper_fcff else "FCF"
        
        # Calculate TV share of EV
        pv_fcf_sum = assumptions.get('pv_fcf_sum', 0)
        ev_gordon = assumptions.get('ev_gordon_growth', 0)
        tv_share_pct = (pv_tv_gordon / ev_gordon * 100) if ev_gordon and ev_gordon > 0 else 0
        
        # Calculate discount factor
        discount_factor = (1 + wacc_used) ** forecast_years
        
        # ===== HEADER ROW: Inputs (left chips) + Outputs (right stack) =====
        col_inputs, col_outputs = st.columns([3, 2])
        
        with col_inputs:
            # Compact input chips
            if year_n_fcf_actual:
                st.markdown(f"""
<div style="display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px;">
    <span class="input-chip">{fcff_label}₁₀ = <b>${year_n_fcf_actual/1e9:.2f}B</b></span>
    <span class="input-chip">g = <b>{terminal_growth*100:.1f}%</b></span>
    <span class="input-chip">WACC = <b>{wacc_used*100:.1f}%</b></span>
    <span class="input-chip">n = <b>{forecast_years} years</b></span>
</div>
                """, unsafe_allow_html=True)
        
        with col_outputs:
            # Output stack - key metrics as HTML to prevent truncation
            st.markdown(f"""
<div style="display: flex; gap: 24px; justify-content: flex-end;">
    <div style="text-align: center;">
        <div style="font-size: 12px; color: #666;">Intrinsic Value</div>
        <div style="font-size: 20px; font-weight: 600;">{f"${price_gordon:.2f}" if price_gordon else "N/A"}</div>
    </div>
    <div style="text-align: center;">
        <div style="font-size: 12px; color: #666;">PV(TV)</div>
        <div style="font-size: 20px; font-weight: 600;">${pv_tv_gordon/1e9:.1f}B</div>
    </div>
    <div style="text-align: center;">
        <div style="font-size: 12px; color: #666;">TV Share</div>
        <div style="font-size: 20px; font-weight: 600;">{tv_share_pct:.1f}%</div>
    </div>
</div>
            """, unsafe_allow_html=True)
        
        # ===== CALCULATION TRACE: Clean 3-step flow =====
        if year_n_fcf_actual and terminal_fcf_actual and tv_gordon_raw:
            wacc_minus_g = wacc_used - terminal_growth
            
            with st.expander("Calculation Details", expanded=False):
                st.markdown(f"""
| Step | Calculation | Result |
|:-----|:------------|-------:|
| 1. Terminal {fcff_label} (Year {n_plus_1}) | {fcff_label}₁₀ × (1 + g) = ${year_n_fcf_actual/1e9:.2f}B × {1 + terminal_growth:.2f} | ${terminal_fcf_actual/1e9:.2f}B |
| 2. Terminal Value | {fcff_label}₁₁ / (WACC − g) = ${terminal_fcf_actual/1e9:.2f}B / {wacc_minus_g*100:.1f}% | ${tv_gordon_raw/1e9:.1f}B |
| 3. Discount Factor | (1 + {wacc_used*100:.1f}%)^{forecast_years} | {discount_factor:.4f} |
| 4. PV(TV) | TV / Discount Factor = ${tv_gordon_raw/1e9:.1f}B / {discount_factor:.4f} | ${pv_tv_gordon/1e9:.1f}B |
                """)
        
        # Warning if WACC - g is too tight
        wacc_minus_g = wacc_used - terminal_growth
        if wacc_minus_g < 0.045:
            st.error(f"""
            🔴 **Gordon Growth Extremely Sensitive**: WACC - g = {wacc_minus_g*100:.1f}% (<4.5%)
            
            Small changes in growth or WACC cause massive swings in terminal value.
            """)
    else:
        st.warning("Gordon Growth terminal value not available.")
    
    # ===== SECTION 2: EXIT MULTIPLE CROSS-CHECK (STRESS TEST) =====
    st.markdown("---")
    st.markdown("#### Exit Multiple Cross-Check (Stress Test)")
    st.caption("⚠️ **These are stress tests, NOT base cases.** Today's market multiple embeds growth expectations and option value. A mature terminal state should trade at a LOWER multiple than today. If 'Required Conversion' shows >100%, that's expected — it means forcing today's multiple onto a mature business is unrealistic. The informative question: How much multiple compression does Gordon Growth imply?")
    
    # Get scenario data from engine
    exit_scenarios = assumptions.get('exit_multiple_scenarios', [])
    wacc_minus_g = wacc_used - terminal_growth
    terminal_fcff_ebitda = assumptions.get('terminal_year_fcff_ebitda')
    consistent_multiple = assumptions.get('consistent_exit_multiple')
    
    if exit_scenarios:
        # Get Year N EBITDA for context
        fcf_projections_exit = ui_data.get("fcf_projections", [])
        year_n_ebitda = None
        if fcf_projections_exit and len(fcf_projections_exit) > 0:
            year_n_proj = fcf_projections_exit[-1]
            year_n_ebitda = year_n_proj.get('ebitda', 0) or assumptions.get('projected_terminal_ebitda', 0)
        
        # ===== SHARED ASSUMPTIONS AS CHIPS =====
        forecasted_str = f"{terminal_fcff_ebitda:.0%}" if terminal_fcff_ebitda else "N/A"
        st.markdown(f"""
<div style="display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 16px;">
    <span class="param-chip">Spread (WACC − g) = <b>{wacc_minus_g*100:.1f}%</b></span>
    <span class="param-chip">Terminal Cash Conversion = <b>{forecasted_str}</b> <span style="font-size:11px; color:var(--clr-text-muted);">(Year {forecast_years} forecast)</span></span>
    <span class="param-chip">Terminal Year = <b>{forecast_years}</b></span>
</div>
        """, unsafe_allow_html=True)
        
        # ===== COMPACT ROW-CARD TABLE =====
        # Header row
        st.markdown("""
<div style="display: grid; grid-template-columns: 2fr 1fr 1fr 1.5fr 1fr; gap: 8px; font-size: 12px; color: #666; padding: 4px 8px; border-bottom: 1px solid #ddd;">
    <span>Scenario</span>
    <span style="text-align: right;">Multiple</span>
    <span style="text-align: right;">Implied Price</span>
    <span style="text-align: right;">Required Conversion <span title="The FCFF/EBITDA ratio needed at terminal year to justify this multiple. Compare to the forecast ratio to assess plausibility." style="cursor: help; color: #888;">ⓘ</span></span>
    <span style="text-align: right;">Gap vs Forecast</span>
</div>
        """, unsafe_allow_html=True)
        
        # Show each scenario as a compact row
        for scenario in exit_scenarios:
            status = scenario.get('status', 'N/A')
            required_conv = scenario.get('required_fcff_ebitda', 0)
            price = scenario.get('price')
            multiple = scenario.get('multiple', 0)
            name = scenario.get('name', 'Unknown')
            source = scenario.get('source', '')
            gap = scenario.get('gap', '')
            is_industry = scenario.get('is_industry', False)
            
            # Status pill styling
            status_styles = {
                'PASS': ('✅', 'var(--clr-success-bg)', 'var(--clr-success-text)'),
                'WARN': ('⚠️', 'var(--clr-warn-bg)', 'var(--clr-warn-text)'),
                'FAIL': ('🔴', 'var(--clr-danger-bg)', 'var(--clr-danger-text)')
            }
            icon, bg_color, text_color = status_styles.get(status, ('❓', 'var(--clr-bg)', 'var(--clr-text-secondary)'))
            
            # One-line explanation based on status
            if status == 'PASS':
                explanation = f"Plausible terminal economics"
            elif status == 'WARN':
                explanation = f"Elevated but possible for asset-light"
            else:
                if required_conv > 1.0:
                    explanation = f"Inconsistent with steady-state economics"
                else:
                    explanation = f"Requires unusually high conversion"
            
            # Source citation with explanation for Model-implied
            if is_industry:
                source_cite = "(Damodaran industry avg — ref only)"
            elif 'Implied by Gordon' in name or 'Model-implied' in name:
                source_cite = "= TV_gordon / EBITDA₁₀ (baseline from forecast)"
            elif source:
                source_cite = f"({source})"
            else:
                source_cite = "(current trading)"
            
            # Gap display
            gap_display = gap if gap else "—"
            if terminal_fcff_ebitda and required_conv:
                gap_pct = ((required_conv / terminal_fcff_ebitda) - 1) * 100 if terminal_fcff_ebitda > 0 else 0
                gap_display = f"{gap_pct:+.0f}%"
            
            # Row card
            price_str = f"${price:.0f}" if price else "N/A"
            st.markdown(f"""
<div style="display: grid; grid-template-columns: 2fr 1fr 1fr 1.5fr 1fr; gap: 8px; align-items: center; padding: 10px 8px; margin: 4px 0; background: var(--clr-surface); border-radius: 6px; border-left: 3px solid {bg_color}; box-shadow: var(--shadow-sm);">
    <div>
        <span style="font-weight: 600;">{name}</span><br/>
        <span style="font-size: 11px; color: #666;">{source_cite}</span>
    </div>
    <span style="text-align: right; font-weight: 600;">{multiple:.1f}x</span>
    <span style="text-align: right; font-weight: 600;">{price_str}</span>
    <span style="text-align: right; font-weight: 600;">{required_conv*100:.0f}%</span>
    <span style="text-align: right; color: {'var(--clr-danger)' if gap_display.startswith('+') and float(gap_display[:-1]) > 50 else 'var(--clr-text-muted)'};">{gap_display}</span>
</div>
<div style="font-size: 13px; color: var(--clr-text-secondary); margin-left: 12px; margin-bottom: 8px;">{icon} {explanation}</div>
            """, unsafe_allow_html=True)
        
        # Model-implied multiple callout
        if consistent_multiple and tv_method_used == 'gordon_growth':
            st.info(f"Model-Implied Multiple: {consistent_multiple:.1f}x — the EV/EBITDA directly implied by TV_gordon / EBITDA₁₀. Multiples above this require higher terminal cash conversion than the forecast.")
    else:
        # Fallback: show simple metrics if scenario data not computed but we have multiples
        implied_gordon_ev_ebitda = assumptions.get('implied_gordon_ev_ebitda') or assumptions.get('consistent_exit_multiple')
        current_mult = assumptions.get('current_ev_ebitda') or exit_multiple_used
        industry_mult = assumptions.get('industry_ev_ebitda')
        
        if implied_gordon_ev_ebitda or current_mult or industry_mult:
            # We have data - show it without the warning
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                if implied_gordon_ev_ebitda:
                    st.metric("Model-Implied", f"{implied_gordon_ev_ebitda:.1f}x")
                    st.caption("TV_gordon / EBITDA₁₀")
                else:
                    st.metric("Model-Implied", "N/A")
            with col_s2:
                if current_mult:
                    st.metric("Current Trading", f"{current_mult:.1f}x")
                    st.caption("Market EV/EBITDA")
                else:
                    st.metric("Current Trading", "N/A")
            with col_s3:
                if industry_mult:
                    st.metric("Industry (ref)", f"{industry_mult:.1f}x")
                    st.caption("Damodaran")
                else:
                    st.metric("Industry (ref)", "N/A")
        else:
            # Truly no data available
            st.warning("Exit multiple data not available. Check that EBITDA is present.")
    
    if not pv_tv_gordon:
        has_exit_tv = pv_tv_exit is not None
        has_gordon_tv = pv_tv_gordon is not None
        has_terminal_ebitda = assumptions.get('projected_terminal_ebitda') is not None and assumptions.get('projected_terminal_ebitda') > 0
        has_exit_multiple = exit_multiple_used is not None and exit_multiple_used > 0

        st.warning("Terminal value cross-check is partially unavailable.")
        st.markdown(
            f"- Exit multiple cross-check available: {'✅ yes' if has_exit_tv else '❌ no'}\n"
            f"- Gordon Growth available: {'✅ yes' if has_gordon_tv else '❌ no'}"
        )

        if not has_exit_tv:
            reasons = assumptions.get('exit_multiple_unavailable_reasons') or []
            if not reasons:
                if not has_exit_multiple:
                    reasons.append("missing terminal exit multiple")
                if not has_terminal_ebitda:
                    reasons.append(f"missing projected terminal EBITDA (Year {forecast_years})")
            if not reasons:
                reasons.append("exit-multiple terminal value not computed for this run")
            st.caption("Exit multiple unavailable because: " + ", ".join(reasons) + ".")
    
    # Summary metrics row
    st.markdown("#### Summary")
    
    if tv_method == "exit_multiple":
        # Show 4 columns for exit multiple method with current/industry/terminal
        col_tv1, col_tv2, col_tv3, col_tv4 = st.columns(4)
        with col_tv1:
            current_mult = assumptions.get('current_ev_ebitda')
            if current_mult:
                st.metric("Current EV/EBITDA", f"{current_mult:.1f}x")
                st.caption("DEFAULT (no rerating)")
            else:
                st.metric("Current EV/EBITDA", "N/A")
                st.caption("Cannot compute")
        with col_tv2:
            industry_mult = assumptions.get('industry_ev_ebitda')
            if industry_mult:
                st.metric("Industry EV/EBITDA", f"{industry_mult:.1f}x")
                st.caption("Damodaran (ref only)")
            else:
                st.metric("Industry EV/EBITDA", "N/A")
                st.caption("Not applicable")
        with col_tv3:
            st.metric("Terminal EV/EBITDA", f"{assumptions.get('exit_multiple', 'N/A')}x")
            st.caption("Used in DCF")
        with col_tv4:
            st.metric("PV(Terminal Value)", f"${pv_tv/1e9:.1f}B")
            tv_ratio = (pv_tv / ev) * 100 if ev > 0 else 0
            st.caption(f"{tv_ratio:.1f}% of EV")
    else:
        # Gordon Growth - original 3-column layout
        col_tv1, col_tv2, col_tv3 = st.columns(3)
        forecast_years_summary = assumptions.get('forecast_years', 5)
        with col_tv1:
            st.metric(f"Terminal Value (Year {forecast_years_summary})", f"${tv_yearN/1e9:.1f}B")
            st.caption("Method: Gordon Growth")
        with col_tv2:
            wacc = assumptions.get('wacc', 0.08)
            st.metric("Discount Factor", f"{1 / (1 + wacc)**forecast_years_summary:.4f}")
            st.caption(f"1 / (1 + {wacc*100:.1f}%)^{forecast_years_summary}")
        with col_tv3:
            st.metric("PV(Terminal Value)", f"${pv_tv/1e9:.1f}B")
            tv_ratio = (pv_tv / ev) * 100 if ev > 0 else 0
            st.caption(f"{tv_ratio:.1f}% of EV")
    
    # VALUATION BRIDGE
    st.markdown("### Valuation")
    st.caption("From enterprise value through to intrinsic value per share")
    bridge_table = ui_adapter.format_bridge_table()
    df_bridge = pd.DataFrame(bridge_table)
    st.dataframe(df_bridge, use_container_width=True, hide_index=True)
    
    # TRACE JSON EXPORT
    st.markdown("### 📥 Export Trace")
    trace_data = {
        "inputs": {k: str(v) for k, v in ui_data.get("inputs", {}).items()},
        "assumptions": ui_data.get("assumptions", {}),
        "results": {
            "enterprise_value": ui_data.get("enterprise_value"),
            "equity_value": ui_data.get("equity_value"),
            "price_per_share": ui_data.get("price_per_share"),
            "pv_fcf_sum": ui_data.get("pv_fcf_sum"),
            "pv_terminal_value": ui_data.get("pv_terminal_value")
        },
        "trace_steps": engine_result.get("trace", []) if engine_result else []
    }
    trace_json = json.dumps(trace_data, indent=2, default=str)
    st.download_button(
        label="Download Trace (JSON)",
        data=trace_json,
        file_name=f"{st.session_state.get('ticker', 'valuation')}_dcf_trace.json",
        mime="application/json"
    )
    
    # Back to summary
    st.divider()
    if st.button("← Back to Summary", key="back_from_details_bottom"):
        st.session_state.show_dcf_details = False
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

    @media (max-width: 1024px) {
        .decision-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
        .floating-toc-wrap { right: 10px; }
        .floating-toc { width: 176px; }
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
if 'forecast_just_generated' not in st.session_state:
    st.session_state.forecast_just_generated = False
if 'ui_cache' not in st.session_state:
    st.session_state.ui_cache = load_ui_cache()
if 'ticker_library' not in st.session_state:
    st.session_state.ticker_library = _normalize_ticker_library(st.session_state.ui_cache.get("ticker_library", []))
if 'last_restore_key' not in st.session_state:
    st.session_state.last_restore_key = None
if 'cache_restore_notice' not in st.session_state:
    st.session_state.cache_restore_notice = ""
if 'custom_ticker_input' not in st.session_state:
    st.session_state.custom_ticker_input = ""
if 'ticker_dropdown' not in st.session_state:
    last_selected = _normalize_ticker(st.session_state.ui_cache.get("last_selected_ticker", "MSFT"))
    library = st.session_state.ticker_library
    st.session_state.ticker_dropdown = last_selected if last_selected in library else library[0]
if 'pending_ticker_dropdown' not in st.session_state:
    st.session_state.pending_ticker_dropdown = None
if 'clear_custom_ticker_input' not in st.session_state:
    st.session_state.clear_custom_ticker_input = False

# --- Helper Functions ---
def reset_analysis():
    st.session_state.quarterly_analysis = None
    st.session_state.independent_forecast = None
    st.session_state.forecast_just_generated = False
    st.session_state.cache_restore_notice = ""
    st.session_state.last_restore_key = None
    # Reset DCF assumptions so they get re-calculated for new ticker
    st.session_state.dcf_wacc = None
    st.session_state.dcf_fcf_growth = None
    st.session_state.dcf_ui_adapter = None
    st.session_state.dcf_engine_result = None
    st.session_state.dcf_snapshot = None

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
    if st.session_state.get("clear_custom_ticker_input"):
        st.session_state.custom_ticker_input = ""
        st.session_state.clear_custom_ticker_input = False

    if st.session_state.ticker_dropdown not in ticker_options:
        st.session_state.ticker_dropdown = ticker_options[0]

    selected_ticker = st.selectbox("Stock Ticker", options=ticker_options, key="ticker_dropdown")
    st.text_input("Custom Ticker (optional)", key="custom_ticker_input", placeholder="e.g. NFLX")
    custom_ticker = _normalize_ticker(st.session_state.custom_ticker_input)
    ticker = custom_ticker if custom_ticker else _normalize_ticker(selected_ticker)
    ticker_valid = _is_valid_ticker_format(ticker)
    if custom_ticker and not ticker_valid:
        st.warning("Custom ticker format invalid. Use letters/numbers, up to 10 chars.")

    # Date-fetch state
    if "available_dates" not in st.session_state:
        st.session_state.available_dates = []
    if "available_dates_ticker" not in st.session_state:
        st.session_state.available_dates_ticker = None
    if "selected_end_date" not in st.session_state:
        st.session_state.selected_end_date = None

    # Refresh cached dates when ticker changes
    if ticker_valid and ticker != st.session_state.available_dates_ticker:
        st.session_state.available_dates_ticker = ticker
        st.session_state.selected_end_date = None
        with st.spinner(f"Fetching available reports for {ticker}..."):
            st.session_state.available_dates = cached_available_dates(ticker)
            if st.session_state.available_dates:
                st.session_state.selected_end_date = st.session_state.available_dates[0]["value"]
            else:
                st.session_state.selected_end_date = None
    elif not ticker_valid:
        st.session_state.available_dates_ticker = None
        st.session_state.available_dates = []
        st.session_state.selected_end_date = None

    # Initial fetch for first load
    if ticker_valid and ticker and not st.session_state.available_dates:
        with st.spinner(f"Fetching available reports for {ticker}..."):
            st.session_state.available_dates = cached_available_dates(ticker)
            if st.session_state.available_dates:
                st.session_state.selected_end_date = st.session_state.available_dates[0]["value"]
            else:
                st.session_state.selected_end_date = None

    available_dates = st.session_state.available_dates
    selected_end_date = st.session_state.selected_end_date

    if available_dates:
        latest_date = available_dates[0]["display"]
        st.markdown(f"""
            <div class="sidebar-data-badge">Latest Data: {latest_date}</div>
        """, unsafe_allow_html=True)
    
    # Analysis Period selection
    st.markdown("**Analysis Period**")
    num_quarters = st.slider("Historical Quarters", min_value=8, max_value=20, value=8, help="How many quarters of historical data to analyze (minimum 8 for trend visibility)")
    
    # Single selectbox for ending report date - only shows ACTUAL available dates
    if available_dates:
        date_options = [d["display"] for d in available_dates]
        date_values = [d["value"] for d in available_dates]

        selected_display = st.selectbox(
            "Select Ending Report",
            options=date_options,
            index=0,  # Default to most recent
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
        elif not selected_end_date:
            st.warning("Please select a valid ticker and ending report.")
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

                    # Persist last selected ticker and add successful custom ticker to library
                    cache = st.session_state.get("ui_cache", _default_ui_cache())
                    cache["last_selected_ticker"] = ticker
                    st.session_state.ui_cache = cache
                    save_ui_cache(cache)
                    if custom_ticker and ticker == custom_ticker:
                        _upsert_ticker_in_library(ticker)
                        st.session_state.pending_ticker_dropdown = ticker
                        st.session_state.clear_custom_ticker_input = True

                    # Auto-run quarterly analysis (cached) with user-selected end date
                    analysis = cached_quarterly_analysis(ticker, num_quarters, selected_end_date)
                    st.session_state.quarterly_analysis = analysis
                    # Calculate comprehensive analysis (DuPont + DCF)
                    quarterly_data = analysis.get("historical_trends", {}).get("quarterly_data", [])
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

if st.session_state.quarterly_analysis:
    analysis = st.session_state.quarterly_analysis
    ticker = st.session_state.ticker
    most_recent = analysis.get("historical_trends", {}).get("most_recent_quarter", {})
    next_forecast = analysis.get("next_forecast_quarter", {})
    data_source = analysis.get("data_source", "Unknown")
    warning = analysis.get("warning")
    hist_data = analysis.get("historical_trends", {}).get("quarterly_data", [])
    growth_summary = analysis.get("growth_rates", {}).get("summary", {})
    growth_detail = analysis.get("growth_rates", {}).get("detailed", [])
    comp_analysis = st.session_state.get("comprehensive_analysis", {})
    consensus = analysis.get("consensus_estimates", {})
    next_forecast_label = next_forecast.get("label", "Next Quarter")
    dcf_ui = st.session_state.get("dcf_ui_adapter")

    if warning:
        st.warning(warning)
    if st.session_state.get("cache_restore_notice"):
        st.info(st.session_state.cache_restore_notice)

    # Top context strip
    _market_data = analysis.get("market_data", {})
    _price = _market_data.get("current_price")
    _mcap = _market_data.get("market_cap")
    _pe = _market_data.get("pe_ratio")
    _as_of = most_recent.get("label", "—")
    _price_str = f"${_price:,.2f}" if _price else "—"
    _mcap_str = f"${_mcap/1e9:.1f}B" if _mcap else "—"
    _pe_str = f"{_pe:.1f}x" if _pe else "—"
    st.markdown(f"""
<div class="hero-strip">
  <div class="hero-item"><div class="hero-label">Price</div><div class="hero-value">{_price_str}</div></div>
  <div class="hero-item"><div class="hero-label">Market Cap</div><div class="hero-value">{_mcap_str}</div></div>
  <div class="hero-item"><div class="hero-label">P/E Ratio</div><div class="hero-value">{_pe_str}</div></div>
  <div class="hero-item" style="border-right:none"><div class="hero-label">As Of</div><div class="hero-value" style="font-size:1rem">{_as_of}</div></div>
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

            upside_downside = None
            verdict_label = "Pending"
            verdict_badge = "badge-neutral-plain"
            verdict_hint = "Run DCF to generate a valuation verdict."
            if current_price and current_price > 0 and intrinsic and intrinsic > 0:
                upside_downside = ((intrinsic - current_price) / current_price * 100)
                verdict_label, verdict_badge, verdict_hint = get_valuation_verdict(upside_downside)

            divergence_pct = None
            if price_exit and price_gordon and abs(price_gordon) > 0.01:
                divergence_pct = ((price_exit - price_gordon) / price_gordon) * 100

            current_price_text = f"${current_price:.2f}" if current_price else "—"
            intrinsic_text = f"${intrinsic:.2f}" if intrinsic else "—"
            upside_text = f"{upside_downside:+.1f}%" if upside_downside is not None else "—"
            st.markdown(f"""
<div class="decision-strip">
  <div class="decision-grid">
    <div><div class="decision-tile-label">Ticker</div><div class="decision-tile-value">{ticker}</div></div>
    <div><div class="decision-tile-label">Current Price</div><div class="decision-tile-value">{current_price_text}</div></div>
    <div><div class="decision-tile-label">Intrinsic Value</div><div class="decision-tile-value">{intrinsic_text}</div></div>
    <div><div class="decision-tile-label">Upside/Downside</div><div class="decision-tile-value">{upside_text}</div></div>
    <div><div class="decision-tile-label">Verdict</div><span class="badge {verdict_badge}">{verdict_label}</span></div>
  </div>
</div>
            """, unsafe_allow_html=True)

            divergence_text = "n/a"
            if divergence_pct is not None:
                divergence_text = f"{divergence_pct:+.1f}%"
            st.markdown(f"""
<div class="confidence-strip">
  <span class="confidence-pill">Data Quality: {data_quality:.0f}/100</span>
  <span class="confidence-pill">TV Dominance: {tv_dominance:.0f}%</span>
  <span class="confidence-pill">TV Cross-check: {divergence_text}</span>
</div>
            """, unsafe_allow_html=True)
            st.caption(verdict_hint)
            if not dcf_ui_data.get("data_sufficient"):
                st.warning("Insufficient data quality: interpretation confidence is reduced.")
    else:
        st.info("Run DCF Analysis in Step 02 to generate the investment verdict.")

    # SECTION B: Valuation Drivers
    st.markdown('<div id="valuation"></div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="section-header"><span class="step-badge">Step 02</span><span class="section-title">Valuation Drivers</span></div>', unsafe_allow_html=True)
    st.caption("Tune assumptions, rerun the model, and review core valuation outputs.")

    snapshot_for_suggestions = cached_financial_snapshot(ticker)
    suggested_wacc = 9.0
    suggested_fcf_growth = 8.0
    if snapshot_for_suggestions:
        if snapshot_for_suggestions.suggested_wacc.value:
            suggested_wacc = round(snapshot_for_suggestions.suggested_wacc.value * 100, 1)
        if snapshot_for_suggestions.suggested_fcf_growth.value:
            suggested_fcf_growth = round(snapshot_for_suggestions.suggested_fcf_growth.value * 100, 1)

    stored_wacc = st.session_state.get("dcf_wacc")
    stored_fcf_growth = st.session_state.get("dcf_fcf_growth")
    default_wacc = stored_wacc if stored_wacc is not None else suggested_wacc
    default_fcf_growth = stored_fcf_growth if stored_fcf_growth is not None else suggested_fcf_growth

    col_wacc, col_growth = st.columns(2)
    with col_wacc:
        user_wacc = st.slider(
            "WACC (%)",
            min_value=5.0,
            max_value=15.0,
            value=default_wacc,
            step=0.5,
            key=f"wacc_slider_{ticker}",
            help="Weighted Average Cost of Capital (discount rate)."
        )
        if snapshot_for_suggestions and snapshot_for_suggestions.suggested_wacc.value:
            beta_val = snapshot_for_suggestions.beta.value
            rf_source = getattr(snapshot_for_suggestions, "rf_source", "^TNX")
            st.caption(f"Suggested: {suggested_wacc:.1f}% (CAPM β={beta_val:.2f}, Rf={rf_source})" if beta_val else f"Suggested: {suggested_wacc:.1f}%")

    with col_growth:
        user_fcf_growth = st.slider(
            "FCF Growth Rate (%)",
            min_value=0.0,
            max_value=25.0,
            value=default_fcf_growth,
            step=0.5,
            key=f"fcf_growth_slider_{ticker}",
            help="Annual free-cash-flow growth for projection period."
        )
        if snapshot_for_suggestions and snapshot_for_suggestions.suggested_fcf_growth.value:
            st.caption(f"Suggested: {suggested_fcf_growth:.1f}%")

    col_run, col_details = st.columns([1, 1])
    with col_run:
        if st.button("Run DCF Analysis", type="primary", key="run_dcf"):
            with st.spinner("Running DCF analysis with full verification..."):
                st.session_state.dcf_wacc = user_wacc
                st.session_state.dcf_fcf_growth = user_fcf_growth
                st.session_state.dcf_terminal_scenario = "current"
                st.session_state.dcf_custom_multiple = None
                ui_adapter_result, engine_result, snapshot = run_dcf_analysis(
                    ticker, user_wacc, user_fcf_growth, terminal_scenario="current", custom_multiple=None
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
            if price_exit and price_gordon and abs(price_gordon) > 0.01:
                diff_pct = ((price_exit - price_gordon) / price_gordon) * 100
                st.caption(f"TV cross-check: Exit Multiple ${price_exit:.2f} vs Gordon Growth ${price_gordon:.2f} ({diff_pct:+.1f}%).")
                if abs(diff_pct) > 30:
                    st.warning("Large divergence between TV methods indicates higher model uncertainty.")

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
        st.metric("Avg Revenue Growth (YoY)", f"{avg_rev:.1f}%" if avg_rev else "N/A")
    with col_h2:
        avg_eps = growth_summary.get("avg_eps_yoy")
        st.metric("Avg EPS Growth (YoY)", f"{avg_eps:.1f}%" if avg_eps else "N/A")
    with col_h3:
        st.metric("Quarters Analyzed", growth_summary.get("samples_used", "N/A"))
    with col_h4:
        if growth_detail and len(growth_detail) >= 4:
            q4_revenues = [g.get("revenue_qoq") for g in growth_detail if "Q4" in g.get("quarter", "") and g.get("revenue_qoq")]
            if q4_revenues:
                avg_q4 = sum(q4_revenues) / len(q4_revenues)
                seasonality = "Strong Q4" if avg_q4 > 5 else ("Weak Q4" if avg_q4 < -5 else "Stable")
            else:
                seasonality = "N/A"
        else:
            seasonality = "N/A"
        st.metric("Seasonality Pattern", seasonality)

    if hist_data:
        df_hist = pd.DataFrame(hist_data).set_index("quarter")
        display_quarters = st.session_state.get("num_quarters", 8)
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            st.caption("Revenue Trend")
            chart_df = df_hist[["revenue"]].head(display_quarters).iloc[::-1]
            chart_df.columns = ["Revenue"]
            chart_df = chart_df.dropna(how="all")
            if not chart_df.empty:
                st.line_chart(chart_df, height=200, use_container_width=True)
        with col_chart2:
            st.caption("EPS Trend")
            eps_df = df_hist[["eps"]].head(display_quarters).iloc[::-1]
            eps_df.columns = ["EPS"]
            eps_df = eps_df.dropna(how="all")
            if not eps_df.empty:
                st.line_chart(eps_df, height=200, use_container_width=True)

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

    if not st.session_state.independent_forecast:
        if not dcf_ui:
            st.warning("Run DCF Analysis first for a more complete synthesis.")
        st.caption("Combines valuation, consensus, and historical momentum into a multi-horizon view.")
        if st.button("Generate Multi-Horizon Outlook", type="primary"):
            with st.spinner("Analyzing data and generating multi-horizon outlook..."):
                dcf_hash = str(hash(str(dcf_data_for_forecast.get("price_per_share", 0)))) if dcf_data_for_forecast else ""
                data_hash = str(hash(str(st.session_state.quarterly_analysis.get("analysis_date", "")) + dcf_hash))
                forecast = cached_independent_forecast(
                    ticker,
                    data_hash,
                    company_name=ticker,
                    dcf_data=dcf_data_for_forecast
                )
                st.session_state.independent_forecast = forecast
                st.session_state.forecast_just_generated = True
                _persist_ai_result_for_context(
                    ticker=ticker,
                    end_date=st.session_state.get("end_date") or st.session_state.get("selected_end_date"),
                    num_quarters=st.session_state.get("num_quarters"),
                )
                st.rerun()

    if st.session_state.independent_forecast:
        forecast = st.session_state.independent_forecast
        if forecast.get("error"):
            st.error(forecast["error"])
        else:
            extracted = forecast.get("extracted_forecast") or {}
            full_analysis = (forecast.get("full_analysis", "") or "").replace("$", "\\$")
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

                key_conditional = extracted.get("key_conditional", "")
                if key_conditional and "null" not in str(key_conditional).lower():
                    st.info(f"**Key Conditional:** {key_conditional}")

                evidence_gaps = extracted.get("evidence_gaps", [])
                if evidence_gaps:
                    gaps_text = " • ".join([g for g in evidence_gaps if g and "null" not in str(g).lower()])
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
