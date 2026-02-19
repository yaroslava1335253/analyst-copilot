"""
app.py MIGRATION GUIDE: From Old to New DCF Engine
===================================================

This file provides step-by-step instructions for integrating the new DCFEngine
into your existing Streamlit app.py.

You can do this gradually (Phase 1) or all at once (Phase 2).

OPTION 1: GRADUAL MIGRATION (Recommended - 1-2 hours)
=====================================================

This approach keeps your old code working while you add new features.

Step 1: Update imports at top of app.py
-------

OLD (line ~15):
    from engine import (
        get_financials, run_structured_prompt, calculate_metrics, run_chat,
        analyze_quarterly_trends, generate_independent_forecast,
        get_latest_date_info, get_available_report_dates,
        calculate_comprehensive_analysis
    )

NEW (replace with):
    from engine import (
        get_financials, run_structured_prompt, calculate_metrics, run_chat,
        analyze_quarterly_trends, generate_independent_forecast,
        get_latest_date_info, get_available_report_dates
        # ⬇️ ADD THESE:
    )
    from dcf_integration import calculate_dcf_with_traceability, format_dcf_for_ui
    from data_adapter import DataAdapter
    from dcf_engine import DCFEngine, DCFAssumptions

Step 2: Replace cached calculation function
----

OLD (lines ~30-40):
    @st.cache_data(ttl=3600, show_spinner=False)
    def cached_quarterly_analysis(ticker: str, num_quarters: int = 8, end_date: str = None) -> dict:
        return analyze_quarterly_trends(ticker, num_quarters, end_date)

ADD AFTER (NEW cached function):
    @st.cache_data(ttl=3600, show_spinner=False)
    def cached_dcf_calculation(ticker: str, wacc_override: float = None, 
                              fcf_growth_override: float = None) -> dict:
        """Calculate DCF using new engine (automatically caches)."""
        from dcf_integration import calculate_dcf_with_traceability
        return calculate_dcf_with_traceability(ticker, wacc_override, fcf_growth_override)

Step 3: Find where you call calculate_comprehensive_analysis()
---

FIND (likely around line 680-720):
    comp_analysis = calculate_comprehensive_analysis(...)

REPLACE with:
    # Use new DCF engine
    full_result = cached_dcf_calculation(ticker, wacc_override, fcf_growth_override)
    comp_analysis = {
        "dcf": format_dcf_for_ui(full_result).get("dcf", {}),
        "dupont": {},  # Keep empty for now
    }

Step 4: Update DCF display section
---

FIND (around line 520-650, the DCF Valuation section):
    with col_dcf:
        st.markdown("**DCF Valuation (FCF Method)**")
        dcf = comp_analysis.get("dcf", {})
        if dcf:
            # ... lots of code ...

REPLACE with:
    with col_dcf:
        st.markdown("**DCF Valuation (FCFF Method with Exit Multiple Terminal Value)**")
        
        # Import helper functions from APP_INTEGRATION_GUIDE.py
        from APP_INTEGRATION_GUIDE import (
            display_dcf_results,
            display_data_quality_panel,
            display_calculation_trace
        )
        
        # Display data quality (NEW FEATURE)
        display_data_quality_panel(full_result)
        
        # Display main results
        display_dcf_results(full_result)
        
        # Expert mode: show trace (NEW FEATURE)
        display_calculation_trace(full_result)

Step 5: Test the changes
---

1. In your terminal:
   cd /Users/user/Desktop/analyst_copilot
   streamlit run app.py

2. Enter a ticker (e.g., AAPL) and load data

3. Check that:
   ✓ DCF Valuation section displays
   ✓ Data Quality Report shows reliability scores
   ✓ Calculation Trace is available in expert mode
   ✓ No errors in terminal

4. Compare with old behavior:
   - Should show similar EV values
   - New: shows equity value explicitly
   - New: shows net debt breakdown
   - New: shows data quality scores

Step 6: Optional - Update Reality Check section
---

If you have a "Reality Check" section, update it:

OLD:
    st.metric("Reality Check", f"DCF: {diff_pct:+.1f}%")

NEW:
    with st.expander("✓ Reality Check", expanded=True):
        mc_data = sanity["ev_vs_market_cap"]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("DCF EV", f"${mc_data['dcf_ev_b']:.2f}B")
        with col2:
            st.metric("Market Cap", f"${mc_data['market_cap_b']:.2f}B")
        with col3:
            diff = mc_data["diff_pct"]
            if abs(diff) < 20:
                st.success(f"Within ±20%: {diff:+.1f}%")
            else:
                st.warning(f"{diff:+.1f}% different")

---

OPTION 2: COMPLETE REPLACEMENT (Faster - 30 minutes, if no other dependencies)
=============================================================================

If you want to replace the entire DCF section at once:

1. Back up your app.py:
   cp app.py app.py.backup

2. Copy the example app from APP_INTEGRATION_GUIDE.py:
   - Replace your entire main analysis section with example_app_integration()
   - Adjust sidebar/header as needed

3. Test thoroughly before removing old code

---

DETAILED CHANGES BY SECTION
============================

A. SIDEBAR CONFIGURATION
   ----------------------
   
   KEEP YOUR OLD SIDEBAR for:
   - Ticker input
   - API key
   - Date picker (if you have it)
   
   ADD NEW INPUT:
   # Optional: advanced assumptions
   with st.expander("⚙️ Advanced DCF Assumptions (Optional)"):
       override_wacc = st.number_input("WACC Override (%)", min_value=1, max_value=20, value=None)
       override_growth = st.number_input("Growth Override (%)", min_value=0, max_value=30, value=None)
   
   Then pass these to cached_dcf_calculation():
       full_result = cached_dcf_calculation(ticker, override_wacc, override_growth)

B. MAIN ANALYSIS SECTION
   ----------------------
   
   OLD FLOW:
   1. Fetch financials
   2. Calculate metrics
   3. Run comprehensive analysis
   4. Display results
   
   NEW FLOW:
   1. Fetch financials (OLD, unchanged)
   2. Calculate metrics (OLD, unchanged)
   3. Run NEW DCF engine via cached_dcf_calculation()
   4. Display results using new panel functions
   
   ✅ Old functions still available if needed (analyze_quarterly_trends, etc.)

C. DISPLAY SECTIONS
   -----------------
   
   STEP 1 (Historical Analysis): UNCHANGED
   - Keep your current historical analysis display
   - No changes needed
   
   STEP 2 (Wall Street Consensus): UNCHANGED
   - Keep your current consensus display
   - No changes needed
   
   STEP 3 (AI Outlook): UNCHANGED
   - Keep your current forecast display
   - No changes needed
   
   NEW: Add DCF Valuation Section
   - Add after or alongside existing sections
   - Use display_dcf_results() helper
   - Show data quality panel first
   - Show trace in expert mode

D. SANITY CHECK SECTION
   --------------------
   
   OLD:
   if sanity:
       st.caption("**Reality Check**")
       market_cap = sanity.get("current_market_cap_b")
       diff_pct = sanity.get("dcf_vs_market_diff_pct")
       if market_cap:
           st.caption(f"Current Market Cap: ${market_cap:.1f}B")
       if diff_pct is not None:
           if abs(diff_pct) > 50:
               st.warning(f"⚠️ DCF differs {diff_pct:+.0f}% from market")
           else:
               st.success(f"✓ DCF within {diff_pct:+.0f}% of market")
   
   NEW:
   # Extract from new result
   sanity = full_result["dcf_result"].get("sanity_checks", {})
   if "ev_vs_market_cap" in sanity:
       with st.expander("✓ Reality Check", expanded=True):
           mc = sanity["ev_vs_market_cap"]
           col1, col2, col3 = st.columns(3)
           with col1:
               st.metric("DCF EV", f"${mc['dcf_ev_b']:.2f}B")
           with col2:
               st.metric("Market Cap", f"${mc['market_cap_b']:.2f}B")
           with col3:
               diff = mc["diff_pct"]
               if abs(diff) < 20:
                   st.success(f"Within ±20%: {diff:+.1f}%")
               elif abs(diff) < 50:
                   st.warning(f"{diff:+.1f}% different")
               else:
                   st.error(f"{diff:+.1f}% different (significant)")

E. WARNINGS & ERRORS
   ------------------
   
   NEW: Show data quality warnings
   
   if full_result["dcf_result"].get("errors"):
       st.error("**DCF Errors:**")
       for e in full_result["dcf_result"]["errors"]:
           st.write(f"- {e}")
   
   if full_result["dcf_result"].get("warnings"):
       st.warning("**DCF Warnings:**")
       for w in full_result["dcf_result"]["warnings"]:
           st.write(f"- {w}")

---

COMMON ISSUES & SOLUTIONS
=========================

Issue 1: "ModuleNotFoundError: No module named 'data_adapter'"
Solution: Make sure data_adapter.py, dcf_engine.py, dcf_integration.py are in same directory as app.py

Issue 2: "KeyError: 'dcf_result' in full_result"
Solution: Check that cached_dcf_calculation returned successfully:
    if not full_result.get("dcf_result", {}).get("success"):
        st.error("DCF calculation failed")
        return

Issue 3: "AttributeError: 'NoneType' object has no attribute 'value'"
Solution: Check data quality before display:
    if snapshot.ttm_revenue and snapshot.ttm_revenue.value:
        st.metric("Revenue", f"${snapshot.ttm_revenue.value/1e9:.1f}B")

Issue 4: Cache not clearing on rerun
Solution: Use st.cache_data.clear() before recalculating:
    if st.button("Recalculate"):
        st.cache_data.clear()
        full_result = cached_dcf_calculation(ticker, wacc, growth)
        st.rerun()

Issue 5: "WACC over terminal growth error"
Solution: This is correct behavior—prevents infinite growth models.
    Make sure WACC > terminal_growth_rate (default 3%)
    Or let auto-assignment handle it (8% WACC > 3% growth is always valid)

---

VERIFICATION CHECKLIST
======================

After making changes, verify:

☐ App starts without errors: streamlit run app.py
☐ Can load a ticker (AAPL)
☐ Data Quality panel appears
☐ DCF Valuation shows:
  ☐ Enterprise Value
  ☐ Equity Value
  ☐ Price per Share
  ☐ Assumptions (WACC, Growth, Method)
  ☐ 5-Year FCF Projection
  ☐ Reality Check
☐ Calculation Trace available in expert mode
☐ Warnings appear if data quality low
☐ Can override WACC/Growth and recalculate
☐ Values are reasonable (not negative, inf, etc.)
☐ Old functions still work (analyze_quarterly_trends, etc.)

---

EXAMPLE MINIMAL INTEGRATION
============================

If you want the absolute minimum change, just replace the DCF display:

```python
# At top with other imports
from dcf_integration import calculate_dcf_with_traceability, format_dcf_for_ui

@st.cache_data(ttl=3600)
def get_dcf(ticker):
    return calculate_dcf_with_traceability(ticker)

# In main app, replace dcf display with:
if ticker:
    full_result = get_dcf(ticker)
    dcf_ui = format_dcf_for_ui(full_result)
    
    st.metric("Enterprise Value", f"${dcf_ui.get('enterprise_value_b', 0):.2f}B")
    st.metric("Equity Value", f"${dcf_ui.get('equity_value_b', 0):.2f}B")
    
    if dcf_ui.get('price_per_share'):
        st.metric("Price/Share", f"${dcf_ui['price_per_share']:.2f}")
```

That's it—you now have the new engine with no major refactoring.

---

SUPPORT
=======

If you run into issues:

1. Check IMPLEMENTATION_SUMMARY.md for overview
2. Check DCF_ARCHITECTURE.md for detailed API
3. Check APP_INTEGRATION_GUIDE.py for example code
4. Run quick_test.py to verify new engine works with yfinance
5. Run test_dcf_engine.py to check unit tests

All modules have detailed docstrings.
"""
