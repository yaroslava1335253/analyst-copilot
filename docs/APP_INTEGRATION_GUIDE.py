"""
Updated app.py integration for new DCF engine
==============================================

This file shows how to integrate the new DCFEngine into Streamlit app.py

Copy relevant sections into app.py to replace old calculate_comprehensive_analysis calls.
"""

import streamlit as st
import pandas as pd
import json
from data_adapter import DataAdapter
from dcf_engine import DCFEngine, DCFAssumptions


# Cache function to avoid re-fetching data
@st.cache_data(ttl=3600, show_spinner=False)
def cached_dcf_calculation(ticker: str, wacc_override: float = None, 
                          fcf_growth_override: float = None) -> dict:
    """
    Cached DCF calculation with new engine.
    Returns full result with trace, diagnostics, etc.
    """
    adapter = DataAdapter(ticker)
    snapshot = adapter.fetch()
    
    assumptions = DCFAssumptions(
        forecast_years=5,
        wacc=wacc_override / 100 if wacc_override else None,
        fcf_growth_rate=fcf_growth_override / 100 if fcf_growth_override else None,
        terminal_value_method="exit_multiple"
    )
    
    engine = DCFEngine(snapshot, assumptions)
    dcf_result = engine.run()
    
    return {
        "snapshot": snapshot,
        "dcf_result": dcf_result,
        "data_quality": {
            "overall_score": snapshot.overall_quality_score,
            "key_inputs": {
                "price": snapshot.price,
                "market_cap": snapshot.market_cap,
                "ttm_revenue": snapshot.ttm_revenue,
                "ttm_fcf": snapshot.ttm_fcf,
                "ttm_ebitda": snapshot.ttm_ebitda,
                "total_debt": snapshot.total_debt,
                "cash_and_equivalents": snapshot.cash_and_equivalents,
                "shares_outstanding": snapshot.shares_outstanding,
            },
            "warnings": snapshot.warnings,
            "errors": snapshot.errors
        }
    }


def display_data_quality_panel(full_result: dict):
    """Display data quality metrics and reliability scores."""
    data_quality = full_result.get("data_quality", {})
    
    with st.expander("📊 Data Quality Report", expanded=False):
        # Overall score
        score = data_quality.get("overall_score", 0)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if score >= 90:
                st.success(f"Quality: {score:.0f}/100 ✓")
            elif score >= 70:
                st.warning(f"Quality: {score:.0f}/100 ⚠️")
            else:
                st.error(f"Quality: {score:.0f}/100 ❌")
        
        with col2:
            st.caption(f"Data fetched at {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        
        # Key inputs quality
        st.subheader("Key Inputs Reliability")
        
        inputs = data_quality.get("key_inputs", {})
        quality_data = []
        
        for field_name, field_obj in inputs.items():
            if field_obj is None:
                continue
            
            reliability = field_obj.reliability_score if hasattr(field_obj, 'reliability_score') else 0
            value = field_obj.value if hasattr(field_obj, 'value') else None
            period = field_obj.period_type if hasattr(field_obj, 'period_type') else None
            is_estimated = field_obj.is_estimated if hasattr(field_obj, 'is_estimated') else False
            fallback = field_obj.fallback_reason if hasattr(field_obj, 'fallback_reason') else None
            
            # Format value
            if value is None:
                value_str = "N/A"
            elif isinstance(value, (int, float)) and value > 1e9:
                value_str = f"${value/1e9:.2f}B"
            elif isinstance(value, (int, float)):
                value_str = f"${value/1e6:.0f}M"
            else:
                value_str = str(value)
            
            # Reliability indicator
            if reliability >= 90:
                reliability_indicator = "✓"
            elif reliability >= 70:
                reliability_indicator = "⚠️"
            else:
                reliability_indicator = "❌"
            
            quality_data.append({
                "Field": field_name.replace("_", " ").title(),
                "Value": value_str,
                "Period": period or "N/A",
                "Reliability": f"{reliability}/100 {reliability_indicator}",
                "Notes": ("Estimated" if is_estimated else "") + 
                        (f"\nFallback: {fallback}" if fallback else "")
            })
        
        df_quality = pd.DataFrame(quality_data)
        st.dataframe(df_quality, use_container_width=True, hide_index=True)
        
        # Warnings & Errors
        warnings = data_quality.get("warnings", [])
        errors = data_quality.get("errors", [])
        
        if errors:
            st.error("**Errors:**")
            for error in errors:
                code = error.get("code", "ERROR") if isinstance(error, dict) else "ERROR"
                msg = error.get("message", str(error)) if isinstance(error, dict) else error
                st.write(f"- {code}: {msg}")
        
        if warnings:
            st.warning("**Data Fetch Warnings:**")
            for warning in warnings:
                code = warning.get("code", "WARN") if isinstance(warning, dict) else "WARN"
                msg = warning.get("message", str(warning)) if isinstance(warning, dict) else warning
                st.write(f"- {code}: {msg}")


def display_calculation_trace(full_result: dict):
    """Display detailed calculation trace for expert mode."""
    dcf_result = full_result.get("dcf_result", {})
    trace = dcf_result.get("trace", [])
    
    with st.expander("🔍 Calculation Trace (Expert Mode)", expanded=False):
        st.subheader("Full DCF Calculation Waterfall")
        
        if not trace:
            st.info("No trace available")
            return
        
        # Display as interactive steps
        tab_list, tab_json = st.tabs(["Step-by-Step", "JSON"])
        
        with tab_list:
            for i, step in enumerate(trace, 1):
                with st.container():
                    st.markdown(f"**Step {i}: {step.get('name', 'Unknown')}**")
                    
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        if step.get("formula"):
                            st.code(step["formula"], language="math")
                    
                    with col2:
                        if step.get("output") is not None:
                            output = step["output"]
                            units = step.get("output_units", "")
                            if isinstance(output, float):
                                if units == "USD" and abs(output) > 1e9:
                                    st.metric("Output", f"${output/1e9:.2f}B")
                                elif units == "rate":
                                    st.metric("Output", f"{output*100:.2f}%")
                                else:
                                    st.metric("Output", f"{output:,.0f} {units}")
                            else:
                                st.metric("Output", output)
                    
                    with col3:
                        st.caption(f"({step.get('output_units', '')})")
                    
                    if step.get("notes"):
                        st.caption(f"ℹ️ {step['notes']}")
                    
                    st.divider()
        
        with tab_json:
            # Show raw JSON
            trace_json = json.dumps(trace, indent=2, default=str)
            st.json(trace)
            
            # Download button
            st.download_button(
                label="Download Trace as JSON",
                data=trace_json,
                file_name=f"dcf_trace_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )


def display_dcf_results(full_result: dict):
    """Display main DCF results using new engine."""
    dcf = full_result.get("dcf_result", {})
    snapshot = full_result.get("snapshot", {})
    
    if not dcf.get("success"):
        st.error("DCF Valuation Failed")
        for error in dcf.get("errors", []):
            st.write(f"- {error}")
        return
    
    st.markdown("**DCF Valuation (FCFF Method with Exit Multiple Terminal Value)**")
    
    # Main results
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        ev = dcf.get("enterprise_value", 0)
        st.metric("Enterprise Value", f"${ev/1e9:.2f}B")
    
    with col2:
        eq_val = dcf.get("equity_value", 0)
        st.metric("Equity Value", f"${eq_val/1e9:.2f}B")
    
    with col3:
        price = dcf.get("price_per_share")
        if price:
            st.metric("Intrinsic Value/Share", f"${price:.2f}")
    
    with col4:
        net_debt = dcf.get("net_debt", 0)
        st.metric("Net Debt", f"${net_debt/1e9:.2f}B")
    
    # Key assumptions
    assumptions = dcf.get("assumptions", {})
    col_a, col_b, col_c, col_d = st.columns(4)
    
    with col_a:
        st.caption(f"WACC: {assumptions.get('wacc', 0)*100:.1f}%")
    with col_b:
        st.caption(f"FCF Growth: {assumptions.get('fcf_growth_rate', 0)*100:.1f}%")
    with col_c:
        st.caption(f"Terminal Method: {assumptions.get('terminal_value_method', 'N/A')}")
    with col_d:
        exit_mult = assumptions.get('exit_multiple')
        if exit_mult:
            st.caption(f"Exit Multiple: {exit_mult}x EBITDA")
    
    # FCF Projection
    projections = dcf.get("fcf_projections", [])
    if projections:
        with st.expander("💡 5-Year FCF Projection", expanded=True):
            proj_data = []
            for p in projections:
                proj_data.append({
                    "Year": f"Y{p['year']}",
                    "FCF ($B)": f"${p['fcf']/1e9:.2f}",
                    "Discount Factor": f"{p['discount_factor']:.4f}",
                    "Present Value ($B)": f"${p['pv']/1e9:.2f}"
                })
            
            df_proj = pd.DataFrame(proj_data)
            st.dataframe(df_proj, use_container_width=True, hide_index=True)
            
            # Summary stats
            pv_fcf = dcf.get("pv_fcf_sum", 0)
            pv_tv = dcf.get("pv_terminal_value", 0)
            total_ev = pv_fcf + pv_tv
            
            st.caption(f"**Waterfall:** PV(5Y FCF)=${pv_fcf/1e9:.2f}B + PV(TV)=${pv_tv/1e9:.2f}B = EV=${total_ev/1e9:.2f}B")
    
    # Reality Check (EV vs Market)
    sanity = dcf.get("sanity_checks", {})
    if "ev_vs_market_cap" in sanity:
        with st.expander("✓ Reality Check", expanded=True):
            mc_data = sanity["ev_vs_market_cap"]
            
            col_check1, col_check2, col_check3 = st.columns(3)
            
            with col_check1:
                st.metric("DCF Enterprise Value", f"${mc_data.get('dcf_ev_b', 0):.2f}B")
            
            with col_check2:
                st.metric("Current Market Cap", f"${mc_data.get('market_cap_b', 0):.2f}B")
            
            with col_check3:
                diff = mc_data.get("diff_pct", 0)
                if abs(diff) < 20:
                    st.success(f"Within ±20%: {diff:+.1f}%")
                elif abs(diff) < 50:
                    st.warning(f"{diff:+.1f}% different")
                else:
                    st.error(f"{diff:+.1f}% different (significant)")
            
            # Multiples
            if "ev_ebitda_multiple" in sanity:
                st.caption(f"EV/EBITDA: {sanity['ev_ebitda_multiple']:.1f}x")
            if "ev_revenue_multiple" in sanity:
                st.caption(f"EV/Revenue: {sanity['ev_revenue_multiple']:.1f}x")
    
    # Warnings from DCF
    if dcf.get("warnings"):
        st.warning("**DCF Warnings:**")
        for w in dcf["warnings"]:
            st.write(f"- {w}")


# ============================================================================
# USAGE IN app.py (replace old calculation code)
# ============================================================================

def example_app_integration():
    """Example of how to use new engine in Streamlit app."""
    
    st.title("Analyst Co-Pilot - DCF Valuation")
    
    # Sidebar config
    with st.sidebar:
        st.header("Configuration")
        
        ticker = st.text_input("Ticker Symbol", value="AAPL", key="ticker_input").upper()
        
        if st.button("Load Data & Calculate DCF"):
            st.session_state.ticker = ticker
    
    # Main area
    if "ticker" in st.session_state:
        ticker = st.session_state.ticker
        
        with st.spinner(f"Analyzing {ticker}..."):
            # Calculate using new engine
            full_result = cached_dcf_calculation(ticker)
            
            # Display panels
            display_data_quality_panel(full_result)
            
            st.divider()
            
            display_dcf_results(full_result)
            
            st.divider()
            
            # Allow adjustment of assumptions
            st.subheader("Sensitivity Analysis")
            
            col_wacc, col_growth = st.columns(2)
            
            with col_wacc:
                new_wacc = st.number_input(
                    "Override WACC (%)",
                    min_value=1.0,
                    max_value=20.0,
                    value=full_result["dcf_result"].get("assumptions", {}).get("wacc", 8) * 100,
                    step=0.1
                )
            
            with col_growth:
                new_growth = st.number_input(
                    "Override FCF Growth (%)",
                    min_value=0.0,
                    max_value=30.0,
                    value=full_result["dcf_result"].get("assumptions", {}).get("fcf_growth_rate", 8) * 100,
                    step=0.1
                )
            
            if st.button("Recalculate with New Assumptions"):
                # Clear cache and recalculate
                st.cache_data.clear()
                full_result = cached_dcf_calculation(ticker, wacc_override=new_wacc, 
                                                     fcf_growth_override=new_growth)
                st.rerun()
            
            st.divider()
            
            # Expert mode: show trace
            display_calculation_trace(full_result)


if __name__ == "__main__":
    example_app_integration()
