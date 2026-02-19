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
from dotenv import load_dotenv

# Load API keys from .env file (if exists)
load_dotenv()
import pandas as pd
import json
from engine import get_financials, run_structured_prompt, calculate_metrics, run_chat, analyze_quarterly_trends, generate_independent_forecast, get_latest_date_info, get_available_report_dates, calculate_comprehensive_analysis
from data_adapter import DataAdapter
from dcf_engine import DCFEngine, DCFAssumptions
from dcf_ui_adapter import DCFUIAdapter
from sources import SOURCE_CATALOG

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
    
    with st.expander("ℹ️ Input Data Legend"):
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
        with st.expander("📚 CAPM Sources & Methodology (Sources [9–12])"):
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
        
        with st.expander("📚 Industry Classification & Methodology"):
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

# --- Helper Functions ---
def reset_analysis():
    st.session_state.quarterly_analysis = None
    st.session_state.independent_forecast = None
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

    ticker = st.text_input("Stock Ticker", value="MSFT").upper()

    # Date-fetch state
    if "available_dates" not in st.session_state:
        st.session_state.available_dates = []
    if "available_dates_ticker" not in st.session_state:
        st.session_state.available_dates_ticker = None
    if "selected_end_date" not in st.session_state:
        st.session_state.selected_end_date = None

    # Refresh cached dates when ticker changes
    if ticker != st.session_state.available_dates_ticker:
        st.session_state.available_dates_ticker = ticker
        st.session_state.selected_end_date = None
        with st.spinner(f"Fetching available reports for {ticker}..."):
            st.session_state.available_dates = cached_available_dates(ticker)
            if st.session_state.available_dates:
                st.session_state.selected_end_date = st.session_state.available_dates[0]["value"]
            else:
                st.session_state.selected_end_date = None

    # Initial fetch for first load
    if ticker and not st.session_state.available_dates:
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
    
    if st.button("Load Data", type="primary", use_container_width=True):
        if not st.session_state.api_key_set:
            st.error("API key not found. Add `GEMINI_API_KEY=your_key` to `.env` file and restart.")
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
    
    # Show data context
    most_recent = analysis.get("historical_trends", {}).get("most_recent_quarter", {})
    next_forecast = analysis.get("next_forecast_quarter", {})
    data_source = analysis.get("data_source", "Unknown")
    warning = analysis.get("warning")
    
    # Show warning if using limited data
    if warning:
        st.warning(warning)

    # LC-3: Hero KPI strip
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

    # LC-4: Step progress indicator
    _has_forecast = bool(st.session_state.get("independent_forecast"))
    _c2 = "step-connector-active" if _has_forecast else ""
    _p3 = "step-pill-active" if _has_forecast else "step-pill-inactive"
    _n3 = "step-num-active" if _has_forecast else "step-num-inactive"
    st.markdown(f"""
<div class="step-progress">
  <div class="step-pill step-pill-active"><span class="step-num step-num-active">01</span>Historical</div>
  <div class="step-connector step-connector-active"></div>
  <div class="step-pill step-pill-active"><span class="step-num step-num-active">02</span>Consensus</div>
  <div class="step-connector {_c2}"></div>
  <div class="step-pill {_p3}"><span class="step-num {_n3}">03</span>AI Outlook</div>
</div>
    """, unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════
    # STEP 1: Historical Analysis
    # ═══════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown('<div class="section-header"><span class="step-badge">Step 01</span><span class="section-title">Historical Analysis</span></div>', unsafe_allow_html=True)
    st.caption(f"Source: {data_source}")
    
    
    hist_data = analysis.get("historical_trends", {}).get("quarterly_data", [])
    growth_summary = analysis.get("growth_rates", {}).get("summary", {})
    growth_detail = analysis.get("growth_rates", {}).get("detailed", [])
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        avg_rev = growth_summary.get('avg_revenue_yoy')
        st.metric("Avg Revenue Growth (YoY)", f"{avg_rev:.1f}%" if avg_rev else "N/A")
    with col2:
        avg_eps = growth_summary.get('avg_eps_yoy')
        st.metric("Avg EPS Growth (YoY)", f"{avg_eps:.1f}%" if avg_eps else "N/A")
    with col3:
        st.metric("Quarters Analyzed", growth_summary.get('samples_used', 'N/A'))
    with col4:
        # Calculate seasonality indicator (Q4 vs other quarters)
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
    
    # Charts - ensure we show at least 8 quarters
    if hist_data:
        df_hist = pd.DataFrame(hist_data).set_index("quarter")
        
        # Get the number of quarters to display (use stored value or default to 8)
        display_quarters = st.session_state.get('num_quarters', 8)
        
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            st.caption("Revenue Trend")
            # Take up to display_quarters rows, reverse for chronological order, keep NaN for gaps
            chart_df = df_hist[["revenue"]].head(display_quarters).iloc[::-1]
            chart_df.columns = ["Revenue"]
            # Only drop rows where ALL values are NaN, not just any
            chart_df = chart_df.dropna(how='all')
            if not chart_df.empty:
                st.line_chart(chart_df, height=200, use_container_width=True)
        
        with col_chart2:
            st.caption("EPS Trend")
            eps_df = df_hist[["eps"]].head(display_quarters).iloc[::-1]
            eps_df.columns = ["EPS"]
            eps_df = eps_df.dropna(how='all')
            if not eps_df.empty:
                st.line_chart(eps_df, height=200, use_container_width=True)
    
    st.markdown('<div class="spacer-md"></div>', unsafe_allow_html=True)
    
    # ========== DUPONT & DCF ANALYSIS ==========
    comp_analysis = st.session_state.get('comprehensive_analysis', {})
    
    if comp_analysis:
        st.markdown('<h3 class="subsection-header">Fundamental Analysis</h3>', unsafe_allow_html=True)
        
        # DuPont Analysis section
        st.markdown("**DuPont Analysis**")
        dupont = comp_analysis.get("dupont", {})
        if dupont:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("ROE", f"{dupont.get('roe', 0):.1f}%")
            with col2:
                st.metric("Profit Margin", f"{dupont.get('net_profit_margin', 0):.1f}%")
            with col3:
                st.metric("Asset Turnover", f"{dupont.get('asset_turnover', 0):.2f}x")
            with col4:
                st.metric("Leverage (EM)", f"{dupont.get('equity_multiplier', 0):.2f}x")
        else:
            st.info("DuPont data unavailable")
        
        st.markdown("---")
        
        # ═══════════════════════════════════════════════════════════════
        # DCF Valuation (Verified Engine) - replaces old DCF section
        # ═══════════════════════════════════════════════════════════════
        st.markdown("**DCF Valuation**")
        
        # Get suggested assumptions from financial snapshot
        snapshot_for_suggestions = cached_financial_snapshot(ticker)
        
        # Extract suggested values with sources
        suggested_wacc = 9.0  # Default
        wacc_source = "Default"
        wacc_date = ""
        suggested_fcf_growth = 8.0  # Default
        fcf_growth_source = "Default"
        fcf_growth_date = ""
        
        if snapshot_for_suggestions:
            # Suggested WACC
            if snapshot_for_suggestions.suggested_wacc.value:
                suggested_wacc = round(snapshot_for_suggestions.suggested_wacc.value * 100, 1)
                wacc_source = snapshot_for_suggestions.suggested_wacc.source_path or "CAPM"
                wacc_notes = snapshot_for_suggestions.suggested_wacc.notes or ""
                # Extract key info for display
                if "Damodaran" in wacc_notes:
                    wacc_date = wacc_notes.split("Damodaran")[-1].strip("() ")
            
            # Suggested FCF Growth
            if snapshot_for_suggestions.suggested_fcf_growth.value:
                suggested_fcf_growth = round(snapshot_for_suggestions.suggested_fcf_growth.value * 100, 1)
                fcf_growth_source = snapshot_for_suggestions.suggested_fcf_growth.source_path or "Historical"
                fcf_growth_notes = snapshot_for_suggestions.suggested_fcf_growth.notes or ""
        
        # Use stored values if user has already adjusted for THIS ticker, otherwise use suggested
        stored_wacc = st.session_state.get('dcf_wacc')
        stored_fcf_growth = st.session_state.get('dcf_fcf_growth')
        
        # If no stored value (new ticker), use the suggested value
        default_wacc = stored_wacc if stored_wacc is not None else suggested_wacc
        default_fcf_growth = stored_fcf_growth if stored_fcf_growth is not None else suggested_fcf_growth
        
        # User-adjustable DCF assumptions
        st.markdown("##### Adjust Assumptions")
        
        col_wacc, col_growth = st.columns(2)
        
        with col_wacc:
            user_wacc = st.slider(
                "WACC (%)", 
                min_value=5.0, 
                max_value=15.0, 
                value=default_wacc, 
                step=0.5,
                key=f"wacc_slider_{ticker}",
                help="Weighted Average Cost of Capital - discount rate for future cash flows"
            )
            # Show suggestion source
            if snapshot_for_suggestions and snapshot_for_suggestions.suggested_wacc.value:
                st.caption(f"💡 Suggested: {suggested_wacc:.1f}%")
                beta_val = snapshot_for_suggestions.beta.value
                rf_source = getattr(snapshot_for_suggestions, 'rf_source', '^TNX')
                if beta_val:
                    st.caption(f"Source: CAPM (β={beta_val:.2f}, Rf={rf_source})")
        
        with col_growth:
            user_fcf_growth = st.slider(
                "FCF Growth Rate (%)", 
                min_value=0.0, 
                max_value=25.0, 
                value=default_fcf_growth, 
                step=0.5,
                key=f"fcf_growth_slider_{ticker}",
                help="Annual growth rate for Free Cash Flow projections"
            )
            # Show suggestion source
            if snapshot_for_suggestions and snapshot_for_suggestions.suggested_fcf_growth.value:
                st.caption(f"💡 Suggested: {suggested_fcf_growth:.1f}%")
                period_type = snapshot_for_suggestions.suggested_fcf_growth.period_type or ""
                source_path = snapshot_for_suggestions.suggested_fcf_growth.source_path or ""
                
                # Display source based on period_type
                if period_type == "trailing_historical":
                    st.caption(f"Source: Yahoo Finance trailing revenue growth")
                elif period_type == "calculated_fallback":
                    st.caption(f"Source: Calculated YoY × 0.7 (fallback)")
                elif source_path:
                    st.caption(f"Source: {source_path}")
        
        col_dcf_button, col_details_button = st.columns([1, 1])
        
        with col_dcf_button:
            if st.button("Run DCF Analysis", type="primary", key="run_dcf"):
                with st.spinner("Running DCF analysis with full verification..."):
                    st.session_state.dcf_wacc = user_wacc
                    st.session_state.dcf_fcf_growth = user_fcf_growth
                    st.session_state.dcf_terminal_scenario = "current"  # Always use current as baseline
                    st.session_state.dcf_custom_multiple = None
                    ui_adapter_result, engine_result, snapshot = run_dcf_analysis(
                        ticker, user_wacc, user_fcf_growth, 
                        terminal_scenario="current", 
                        custom_multiple=None
                    )
                    st.session_state.dcf_ui_adapter = ui_adapter_result
                    st.session_state.dcf_engine_result = engine_result
                    st.session_state.dcf_snapshot = snapshot
                    st.rerun()
        
        with col_details_button:
            if st.session_state.get('dcf_ui_adapter'):
                if st.button("View DCF Details →", key="view_details"):
                    st.session_state.show_dcf_details = True
                    st.rerun()
        
        # Show DCF Details page if requested
        if st.session_state.get('show_dcf_details') and st.session_state.get('dcf_ui_adapter'):
            _show_dcf_details_page()
            st.stop()  # Stop rendering main page
        
        # DCF Summary (if analysis was run)
        if st.session_state.get('dcf_ui_adapter'):
            ui_adapter = st.session_state.dcf_ui_adapter
            ui_data = ui_adapter.get_ui_data()
            
            if not ui_data.get("success"):
                st.error(f"❌ DCF Analysis Failed")
                for err in ui_data.get("errors", []):
                    st.error(f"  • {err}")
            else:
                # Summary Metrics
                col_ev, col_equity, col_per_share, col_quality = st.columns(4)
                
                with col_ev:
                    ev = ui_data.get('enterprise_value', 0)
                    display_ev = f"${ev/1e9:.1f}B" if ev >= 1e9 else (f"${ev/1e6:.1f}M" if ev >= 1e6 else "—")
                    st.metric("Enterprise Value", display_ev, help="PV(FCF 1-5) + PV(Terminal Value)")
                
                with col_equity:
                    equity = ui_data.get('equity_value', 0)
                    display_equity = f"${equity/1e9:.1f}B" if equity >= 1e9 else (f"${equity/1e6:.1f}M" if equity >= 1e6 else "—")
                    st.metric("Equity Value", display_equity, help="EV - Net Debt")
                
                with col_per_share:
                    per_share = ui_data.get('price_per_share', 0)
                    current_price = ui_data.get('current_price', 0)
                    display_per_share = f"${per_share:.2f}" if per_share > 0 else "—"
                    
                    if current_price and current_price > 0 and per_share and per_share > 0:
                        upside = ((per_share - current_price) / current_price * 100)
                        st.metric("Intrinsic Value/Share", display_per_share, delta=f"{upside:+.1f}%", help=f"vs current ${current_price:.2f}")
                    else:
                        st.metric("Intrinsic Value/Share", display_per_share, help="Equity Value ÷ Shares")
                
                with col_quality:
                    quality = ui_data.get('data_quality_score', 0)
                    st.metric("Data Quality", f"{quality:.0f}/100", help="Composite reliability")
                
                # Data sufficiency warning
                if not ui_data.get("data_sufficient"):
                    st.warning("⚠️ **Insufficient Data**: Some required data is missing or quality is below 60/100.")
                
                # General disclaimer
                st.caption("⚠️ AI valuations are highly sensitive to assumptions and data quality. This estimate is not financial advice and may be materially wrong.")
                
                # Current Financial Position
                st.markdown("**Current Financial Position (TTM)**")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Valuation**")
                    current_price_metric = ui_data["inputs"].get("current_price")
                    shares_metric = ui_data["inputs"].get("shares_outstanding")
                    market_cap_metric = ui_data["inputs"].get("market_cap")
                    st.caption(f"Price: {current_price_metric.formatted()}")
                    st.caption(f"Shares: {shares_metric.formatted()}")
                    st.caption(f"Market Cap: {market_cap_metric.formatted()}")
                
                with col2:
                    st.markdown("**Income Statement**")
                    rev = ui_data["inputs"].get("ttm_revenue")
                    op_income = ui_data["inputs"].get("ttm_operating_income")
                    ebitda = ui_data["inputs"].get("ttm_ebitda")
                    st.caption(f"Revenue: {rev.formatted()}")
                    st.caption(f"Op Income: {op_income.formatted()}")
                    st.caption(f"EBITDA: {ebitda.formatted()}")
                
                with col3:
                    st.markdown("**Cash Flow & Debt**")
                    cfo = ui_data["inputs"].get("ttm_operating_cash_flow")
                    capex = ui_data["inputs"].get("ttm_capex")
                    debt = ui_data["inputs"].get("total_debt")
                    cash = ui_data["inputs"].get("cash")
                    st.caption(f"Oper. CF: {cfo.formatted()}")
                    st.caption(f"CapEx: {capex.formatted()}")
                    st.caption(f"Total Debt: {debt.formatted()}")
                    st.caption(f"Cash: {cash.formatted()}")
                
                # 5-Year Projection
                st.markdown("**5-Year FCF Projection**")
                projections = ui_data.get("fcf_projections", [])
                if projections:
                    proj_table = []
                    for proj in projections:
                        proj_table.append({
                            "Year": f"Year {proj.get('year', 0)}",
                            "FCF": f"${proj.get('fcf', 0)/1e9:.1f}B",
                            "PV(FCF)": f"${proj.get('pv', 0)/1e9:.1f}B"
                        })
                    df_proj = pd.DataFrame(proj_table)
                    st.dataframe(df_proj, use_container_width=True, hide_index=True)
                
                # Valuation Bridge
                st.markdown("**Valuation**")
                bridge_table = ui_adapter.format_bridge_table()
                df_bridge = pd.DataFrame(bridge_table)
                st.dataframe(df_bridge, use_container_width=True, hide_index=True)
                
                # Key Assumptions
                st.markdown("**Key Assumptions**")
                assumptions_table = ui_adapter.format_assumptions_table()
                df_assumptions = pd.DataFrame(assumptions_table)
                st.dataframe(df_assumptions, use_container_width=True, hide_index=True)
                
                # Valuation vs Market with explicit bands and confidence flags
                st.markdown("**Valuation vs Market**")
                current_price = ui_data.get('current_price', 0)
                intrinsic = ui_data.get('price_per_share', 0)
                assumptions = ui_data.get('assumptions', {})
                
                if current_price and current_price > 0 and intrinsic and intrinsic > 0:
                    upside_downside = ((intrinsic - current_price) / current_price * 100)
                    
                    # Get confidence flags
                    tv_dominance = assumptions.get('tv_dominance_pct', 0)
                    data_quality = assumptions.get('data_quality_score', 0)
                    growth_proxy_warning = assumptions.get('growth_proxy_warning', False)
                    price_exit = assumptions.get('price_exit_multiple')
                    price_gordon = assumptions.get('price_gordon_growth')
                    
                    col_cmp1, col_cmp2 = st.columns(2)
                    with col_cmp1:
                        st.metric("Current Price", f"${current_price:.2f}")
                        st.metric("Intrinsic Value", f"${intrinsic:.2f}")
                        st.metric("Upside/Downside", f"{upside_downside:+.1f}%")
                    
                    with col_cmp2:
                        # Explicit verdict bands
                        # Bands: ±10% = Fair, 10-25% = Modest, >25% = Significant
                        
                        if upside_downside > 25:
                            st.success("📈 **SIGNIFICANTLY UNDERVALUED**")
                            st.caption("DCF >25% above market")
                        elif upside_downside > 10:
                            st.success("📈 **MODESTLY UNDERVALUED**")
                            st.caption("DCF 10-25% above market")
                        elif upside_downside < -25:
                            st.error("📉 **SIGNIFICANTLY OVERVALUED**")
                            st.caption("DCF >25% below market")
                        elif upside_downside < -10:
                            st.error("📉 **MODESTLY OVERVALUED**")
                            st.caption("DCF 10-25% below market")
                        else:
                            st.info("➡️ **FAIRLY VALUED**")
                            st.caption("DCF within ±10% of market")
                    
                    # Cross-check with dual TV methods
                    if price_exit and price_gordon and abs(price_exit - price_gordon) > 0.01:
                        diff_pct = ((price_exit - price_gordon) / price_gordon * 100)
                        st.caption(f"📊 Cross-check: Exit Multiple → ${price_exit:.2f} | Gordon Growth → ${price_gordon:.2f} ({diff_pct:+.1f}% diff)")
                        
                        if abs(diff_pct) > 30:
                            st.warning("⚠️ Large divergence between TV methods — high model uncertainty")
                else:
                    st.warning("Missing price data for comparison.")
    
    st.markdown('<div class="spacer-sm"></div>', unsafe_allow_html=True)

    # Detailed data in expander
    with st.expander("View detailed quarterly data", icon="📋"):
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
    
    # ───────────────────────────────────────────────────────────────
    # DCF DATA SOURCES EXPANDER
    # ───────────────────────────────────────────────────────────────
    with st.expander("📚 Data Sources & Methodology", expanded=False):
        st.markdown("All data sources used in the DCF analysis above:")
        ticker_for_url = st.session_state.get('ticker', '{ticker}')
        for key, src in SOURCE_CATALOG.items():
            url = src['url'].replace('{ticker}', ticker_for_url)
            st.markdown(
                f"**[{src['id']}]** **{src['label']}** — {src['description']}  \n"
                f"*Method: {src['method']}*  \n"
                f"[{url}]({url})"
            )

    # ═══════════════════════════════════════════════════════════════
    # STEP 2: Wall Street Consensus
    # ═══════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown('<div class="section-header"><span class="step-badge">Step 02</span><span class="section-title">Wall Street Consensus</span></div>', unsafe_allow_html=True)
    
    consensus = analysis.get("consensus_estimates", {})
    next_forecast_label = next_forecast.get("label", "Next Quarter")
    
    if consensus.get("error"):
        st.error(consensus["error"])
    elif consensus:
        next_q = consensus.get("next_quarter", {})
        full_year = consensus.get("full_year", {})
        coverage = consensus.get("analyst_coverage", {})
        targets = consensus.get("price_targets", {})
        
        # Next Quarter - use the forecast label from analysis
        quarter_label = next_q.get('quarter_label') or next_forecast_label
        st.markdown(f"**{quarter_label} Estimates**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Revenue", next_q.get("revenue_estimate", "N/A"))
        with col2:
            st.metric("EPS", next_q.get("eps_estimate", "N/A"))
        with col3:
            st.metric("Analysts", coverage.get("num_analysts", "N/A"))
        with col4:
            buy = coverage.get("buy_ratings", 0) or 0
            hold = coverage.get("hold_ratings", 0) or 0
            sell = coverage.get("sell_ratings", 0) or 0
            total = buy + hold + sell
            if total > 0:
                sentiment = f"{buy}/{hold}/{sell}"
            else:
                sentiment = "N/A"
            st.metric("Buy/Hold/Sell", sentiment)
        
        # Source attribution for estimates
        estimate_sources = []
        if next_q.get("source"):
            estimate_sources.append(next_q.get("source"))
        if coverage.get("source") and coverage.get("source") not in estimate_sources:
            estimate_sources.append(coverage.get("source"))
        if estimate_sources:
            st.caption(f"Source: {', '.join(estimate_sources)}")
        
        # Price targets
        if targets:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Price Target (Low)", targets.get("low", "N/A"))
            with col2:
                st.metric("Price Target (Avg)", targets.get("average", "N/A"))
            with col3:
                st.metric("Price Target (High)", targets.get("high", "N/A"))
            if targets.get("source"):
                st.caption(f"Source: {targets.get('source')}")
        
        # Implied total value (market cap) from price targets
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
            
            col1, col2, col3 = st.columns(3)
            with col1:
                implied_low = low_pt * shares_outstanding if low_pt is not None else None
                st.metric("Implied Value (Low PT)", format_mcap(implied_low), delta=format_delta(implied_low))
            with col2:
                implied_avg = avg_pt * shares_outstanding if avg_pt is not None else None
                st.metric("Implied Value (Avg PT)", format_mcap(implied_avg), delta=format_delta(implied_avg))
            with col3:
                implied_high = high_pt * shares_outstanding if high_pt is not None else None
                st.metric("Implied Value (High PT)", format_mcap(implied_high), delta=format_delta(implied_high))
        elif targets and not shares_outstanding:
            st.caption("Implied total value unavailable (shares outstanding missing from market data).")
        
        # Qualitative summary from analysts
        qualitative = consensus.get("qualitative_summary")
        qual_sources = consensus.get("qualitative_sources", [])
        if qualitative:
            st.markdown(f"**Analyst View:** {qualitative}")
        else:
            # Fallback: generate summary from ratings data
            buy = coverage.get("buy_ratings", 0) or 0
            hold = coverage.get("hold_ratings", 0) or 0
            sell = coverage.get("sell_ratings", 0) or 0
            total = buy + hold + sell
            if total > 0 and targets:
                avg_pt_val = parse_price_value(targets.get("average"))
                market_data = analysis.get("market_data", {})
                current_price = market_data.get("current_price", 0)
                if avg_pt_val and current_price:
                    upside = ((avg_pt_val - current_price) / current_price) * 100
                    direction = "bullish" if upside > 5 else ("neutral" if upside > -5 else "cautious")
                    st.markdown(f"**Analyst View:** Consensus is {direction} with {buy} buy ratings vs {sell} sell, targeting {upside:+.0f}% from current levels.")
        
        st.markdown('<div class="spacer-sm"></div>', unsafe_allow_html=True)
        
        # Sources
        with st.expander("Sources & Citations", icon="🔗"):
            citations = consensus.get("citations", [])
            if citations:
                for cite in citations:
                    url = cite.get("url", "")
                    if url:
                        st.markdown(f"- [{cite.get('source_name', 'Source')}]({url}) — {cite.get('data_type', '')}")
            else:
                st.markdown(f"""
**Data Sources:**
- [Yahoo Finance](https://finance.yahoo.com/quote/{ticker}/analysis) — EPS & Revenue Estimates, Analyst Ratings
                """)
            st.markdown("*See **📚 Data Sources & Methodology** in the DCF section above for full methodology citations.*")
            
            # AI-sourced analyst commentary
            if qual_sources:
                st.markdown("**Analyst Commentary:**")
                for s in qual_sources[:3]:
                    st.markdown(f"- _{s.get('headline', '')}_ ({s.get('source', '')}, {s.get('date', '')})")
    else:
        st.info("No consensus data available.")
    
    # ═══════════════════════════════════════════════════════════════
    # STEP 3: AI Outlook
    # ═══════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown('<div class="section-header"><span class="step-badge">Step 03</span><span class="section-title">AI Outlook — Multi-Horizon Analysis</span></div>', unsafe_allow_html=True)
    
    # Check if DCF analysis has been run
    dcf_ui = st.session_state.get('dcf_ui_adapter')
    dcf_data_for_forecast = None
    if dcf_ui:
        dcf_data_for_forecast = dcf_ui.get_ui_data()
    
    if not st.session_state.independent_forecast:
        if not dcf_ui:
            st.warning("Run DCF Analysis (Step 1) first for a more comprehensive synthesis.")
        st.caption("Synthesize DCF valuation, Wall Street consensus, and historical trends into short-term, mid-term, and long-term outlooks.")
        if st.button("Generate Multi-Horizon Outlook", type="primary"):
            with st.spinner("Analyzing data and generating multi-horizon outlook..."):
                # Create a hash of the quarterly data + DCF to use as cache key
                dcf_hash = str(hash(str(dcf_data_for_forecast.get('price_per_share', 0)))) if dcf_data_for_forecast else ""
                data_hash = str(hash(str(st.session_state.quarterly_analysis.get("analysis_date", "")) + dcf_hash))
                forecast = cached_independent_forecast(
                    ticker,
                    data_hash,
                    company_name=ticker,
                    dcf_data=dcf_data_for_forecast
                )
                st.session_state.independent_forecast = forecast
                st.rerun()
    
    if st.session_state.independent_forecast:
        forecast = st.session_state.independent_forecast
        
        if forecast.get("error"):
            st.error(forecast["error"])
        else:
            extracted = forecast.get("extracted_forecast") or {}
            full_analysis = forecast.get("full_analysis", "")
            
            # Escape dollar signs to prevent KaTeX math rendering
            full_analysis = full_analysis.replace("$", "\\$")
            
            # Check if we have extracted data
            has_extracted = extracted and (extracted.get("short_term_stance") or extracted.get("fundamental_outlook"))
            
            if has_extracted:
                # ===== NEW MULTI-HORIZON LAYOUT =====
                
                # Quick summary cards at top
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    short_stance = extracted.get("short_term_stance", "Neutral")
                    short_emoji = {"Bullish": "📈", "Neutral": "➡️", "Bearish": "📉"}.get(short_stance, "➡️")
                    _sc1 = "stance-card-bull" if short_stance == "Bullish" else "stance-card-bear" if short_stance == "Bearish" else "stance-card-neut"
                    st.markdown(f"""
<div class="stance-card {_sc1}">
    <div style="font-size: 11px; color: var(--clr-text-muted);">SHORT-TERM (0-12m)</div>
    <div style="font-size: 18px; font-weight: 600;">{short_emoji} {short_stance}</div>
</div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    fund_outlook = extracted.get("fundamental_outlook", "Stable")
                    fund_emoji = {"Strong": "💪", "Stable": "➡️", "Weakening": "⚠️"}.get(fund_outlook, "➡️")
                    _sc2 = "stance-card-bull" if fund_outlook == "Strong" else "stance-card-neut" if fund_outlook == "Stable" else "stance-card-bear"
                    st.markdown(f"""
<div class="stance-card {_sc2}">
    <div style="font-size: 11px; color: var(--clr-text-muted);">FUNDAMENTALS</div>
    <div style="font-size: 18px; font-weight: 600;">{fund_emoji} {fund_outlook}</div>
</div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    stock_outlook = extracted.get("stock_outlook", "Neutral")
                    stock_horizon = extracted.get("stock_outlook_horizon", "")
                    stock_emoji = {"Bullish": "📈", "Neutral": "➡️", "Bearish": "📉"}.get(stock_outlook, "➡️")
                    conv_level = extracted.get("fundamental_conviction", "Medium")
                    conv_badge = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}.get(conv_level, "🟡")
                    _sc3 = "stance-card-bull" if stock_outlook == "Bullish" else "stance-card-bear" if stock_outlook == "Bearish" else "stance-card-neut"
                    st.markdown(f"""
<div class="stance-card {_sc3}">
    <div style="font-size: 11px; color: var(--clr-text-muted);">STOCK OUTLOOK {conv_badge}</div>
    <div style="font-size: 18px; font-weight: 600;">{stock_emoji} {stock_outlook}</div>
</div>
                    """, unsafe_allow_html=True)
                
                st.markdown("")
                
                # Key Conditional (the main takeaway)
                key_conditional = extracted.get("key_conditional", "")
                if key_conditional and "null" not in str(key_conditional).lower():
                    st.info(f"💡 **Key Conditional:** {key_conditional}")
                
                # DCF Assessment chips
                wacc_assess = extracted.get("dcf_wacc_assessment", "")
                growth_assess = extracted.get("dcf_growth_assessment", "")
                conv_assess = extracted.get("terminal_conversion_assessment", "")
                
                if wacc_assess or growth_assess or conv_assess:
                    st.markdown("**DCF Assumption Check:**")
                    cols = st.columns(3)
                    with cols[0]:
                        if wacc_assess:
                            _bcls = "badge-pass" if wacc_assess == "reasonable" else "badge-warn" if wacc_assess == "conservative" else "badge-fail"
                            st.markdown(f'<span class="badge {_bcls}">WACC: {wacc_assess}</span>', unsafe_allow_html=True)
                    with cols[1]:
                        if growth_assess:
                            _bcls = "badge-pass" if growth_assess == "reasonable" else "badge-warn" if growth_assess == "conservative" else "badge-fail"
                            st.markdown(f'<span class="badge {_bcls}">Growth: {growth_assess}</span>', unsafe_allow_html=True)
                    with cols[2]:
                        if conv_assess:
                            _bcls = "badge-pass" if conv_assess == "achievable" else "badge-fail"
                            st.markdown(f'<span class="badge {_bcls}">Terminal Conv: {conv_assess}</span>', unsafe_allow_html=True)
                
                # Evidence Gaps
                evidence_gaps = extracted.get("evidence_gaps", [])
                if evidence_gaps and len(evidence_gaps) > 0:
                    gaps_text = " • ".join([g for g in evidence_gaps if g and "null" not in str(g).lower()])
                    if gaps_text:
                        st.caption(f"⚠️ **Evidence gaps:** {gaps_text}")
                
                # ===== FULL ANALYSIS IN EXPANDER (WITH FINAL ASSESSMENT) =====
                with st.expander("Full Analysis & Final Assessment", expanded=True, icon="📄"):
                    st.markdown(full_analysis.strip())
            
            else:
                # Fallback: show full analysis directly (extraction failed)
                st.markdown("**AI Multi-Horizon Analysis**")
                
                # Show full analysis in expander
                if full_analysis:
                    with st.expander("Full Analysis & Final Assessment", expanded=True, icon="📄"):
                        st.markdown(full_analysis.strip())
                else:
                    st.warning("No analysis generated. Please try again.")
            
            st.caption(f"Generated: {forecast.get('forecast_date', 'Unknown')}")

else:
    st.info("Enter a ticker and click 'Load Data' to begin analysis.")
