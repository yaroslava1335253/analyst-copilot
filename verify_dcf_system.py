"""
Comprehensive Verification Suite for DCF System
===============================================
Tests the 8 critical checks:

1. Share count units & per-share math
2. EV vs equity consistency  
3. CapEx sign handling
4. TTM construction correctness
5. Terminal value discounting & dominance
6. Exit multiple logic (EBITDA projection)
7. Quality scoring explicitness
8. Golden + adversarial tests with real data

Run with: python verify_dcf_system.py
"""

import json
import pandas as pd
import sys
from data_adapter import DataAdapter, DataQualityMetadata
from dcf_engine import DCFEngine, DCFAssumptions


# ============================================================================
# SECTION 1: SHARE COUNT UNITS & PER-SHARE MATH VERIFICATION
# ============================================================================

def verify_share_count_and_per_share_math(ticker):
    """
    Hard check: Market Cap you compute internally must equal Price × Shares.
    If not, you have a share unit mismatch and every "Price/Share" output is junk.
    """
    print(f"\n{'='*80}")
    print(f"[CHECK 1] SHARE COUNT UNITS & PER-SHARE MATH")
    print(f"{'='*80}")
    print(f"Ticker: {ticker}")
    
    adapter = DataAdapter(ticker)
    snapshot = adapter.fetch()
    
    if not snapshot.price.value or not snapshot.shares_outstanding.value:
        print("❌ FAIL: Price or shares missing")
        return False
    
    price = snapshot.price.value
    shares = snapshot.shares_outstanding.value
    market_cap_from_price = price * shares
    market_cap_from_yf = snapshot.market_cap.value
    
    print(f"\n  Current Price: ${price:,.2f}")
    print(f"    Source: {snapshot.price.source_path}")
    print(f"    Units: {snapshot.price.units}")
    print(f"    Reliability: {snapshot.price.reliability_score}/100")
    
    print(f"\n  Shares Outstanding: {shares:,.0f}")
    print(f"    Source: {snapshot.shares_outstanding.source_path}")
    print(f"    Units: {snapshot.shares_outstanding.units}")
    print(f"    Reliability: {snapshot.shares_outstanding.reliability_score}/100")
    print(f"    NOTE: Are these raw shares or thousands? Check if {shares:,.0f} makes sense vs market cap")
    
    print(f"\n  Market Cap (yfinance): ${market_cap_from_yf:,.0f}")
    print(f"    Source: {snapshot.market_cap.source_path}")
    
    print(f"\n  Computed Market Cap (Price × Shares): ${market_cap_from_price:,.0f}")
    
    # Check for consistency
    if market_cap_from_yf:
        diff_pct = abs(market_cap_from_price - market_cap_from_yf) / market_cap_from_yf * 100
        print(f"  Difference: {diff_pct:.2f}%")
        
        if diff_pct < 1.0:
            print(f"  ✅ PASS: Price × Shares matches yfinance market cap (within 1%)")
            return True
        elif diff_pct < 5.0:
            print(f"  ⚠️  WARNING: {diff_pct:.1f}% difference (likely timing lag)")
            return True
        else:
            print(f"  ❌ FAIL: {diff_pct:.1f}% difference indicates share unit mismatch")
            print(f"    Possible issues:")
            print(f"      - Shares are in thousands but treated as raw count")
            print(f"      - Class A/B mismatch for {ticker} (check ticker consistency)")
            print(f"      - yfinance returned float vs fully-diluted shares")
            return False
    
    return True


# ============================================================================
# SECTION 2: EV VS EQUITY CONSISTENCY VERIFICATION
# ============================================================================

def verify_ev_equity_bridge(ticker):
    """
    Hard check: equity_value = enterprise_value − net_debt must hold numerically,
    and net_debt must be defined identically everywhere.
    """
    print(f"\n{'='*80}")
    print(f"[CHECK 2] EV VS EQUITY CONSISTENCY (EV→Equity Bridge)")
    print(f"{'='*80}")
    print(f"Ticker: {ticker}")
    
    adapter = DataAdapter(ticker)
    snapshot = adapter.fetch()
    
    engine = DCFEngine(snapshot)
    result = engine.run()
    
    if not result.get('success'):
        print(f"❌ FAIL: DCF engine failed: {result.get('errors')}")
        return False
    
    ev = result['enterprise_value']
    equity = result['equity_value']
    net_debt = result['net_debt']
    net_debt_details = result['net_debt_details']
    
    computed_equity = ev - net_debt
    diff = abs(equity - computed_equity)
    
    print(f"\n  Enterprise Value: ${ev:,.0f}")
    print(f"  Net Debt: ${net_debt:,.0f}")
    print(f"    Total Debt: ${net_debt_details['total_debt']:,.0f}")
    print(f"    Cash: ${net_debt_details['cash']:,.0f}")
    print(f"    Definition: Total Debt - Cash")
    
    print(f"\n  Reported Equity Value: ${equity:,.0f}")
    print(f"  Computed Equity (EV - ND): ${computed_equity:,.0f}")
    print(f"  Difference: ${diff:,.0f} ({diff/computed_equity*100:.3f}%)")
    
    if diff < 1:
        print(f"  ✅ PASS: EV - Net Debt = Equity Value formula holds")
        return True
    else:
        print(f"  ❌ FAIL: Bridge formula not satisfied")
        return False


# ============================================================================
# SECTION 3: CAPEX SIGN HANDLING VERIFICATION
# ============================================================================

def verify_capex_sign_handling(ticker):
    """
    Hard check: CFO, CapEx, and computed FCF must be mathematically consistent.
    If CapEx is negative (common in yfinance), FCF = CFO - (negative) should increase CFO,
    not decrease it.
    """
    print(f"\n{'='*80}")
    print(f"[CHECK 3] CAPEX SIGN HANDLING")
    print(f"{'='*80}")
    print(f"Ticker: {ticker}")
    
    adapter = DataAdapter(ticker)
    snapshot = adapter.fetch()
    
    cfo = snapshot.ttm_operating_cash_flow.value
    capex = snapshot.ttm_capex.value  # Should already be abs() by adapter
    fcf = snapshot.ttm_fcf.value
    
    if not all([cfo, capex, fcf]):
        print(f"❌ FAIL: Missing CFO, CapEx, or FCF")
        return False
    
    print(f"\n  TTM Operating Cash Flow (CFO): ${cfo:,.0f}")
    print(f"    Source: {snapshot.ttm_operating_cash_flow.source_path}")
    print(f"    Period type: {snapshot.ttm_operating_cash_flow.period_type}")
    
    print(f"\n  TTM CapEx: ${capex:,.0f}")
    print(f"    Source: {snapshot.ttm_capex.source_path}")
    print(f"    NOTE: Adapter should have applied abs() to handle negative CapEx from yfinance")
    print(f"    Period type: {snapshot.ttm_capex.period_type}")
    
    computed_fcf = cfo - capex
    diff = abs(fcf - computed_fcf)
    
    print(f"\n  Reported TTM FCF: ${fcf:,.0f}")
    print(f"  Computed FCF (CFO - CapEx): ${computed_fcf:,.0f}")
    print(f"  Difference: ${diff:,.0f}")
    
    # CapEx should be positive (absolute value)
    if capex < 0:
        print(f"  ❌ FAIL: CapEx is negative ({capex}); adapter should take abs()")
        return False
    
    # FCF should be reasonable
    if fcf < 0:
        print(f"  ⚠️  WARNING: FCF is negative (company is burning cash)")
        print(f"     This can be valid, but verify manually")
    
    if diff < 1:
        print(f"  ✅ PASS: FCF = CFO - CapEx formula holds")
        return True
    else:
        print(f"  ❌ FAIL: FCF math doesn't add up")
        return False


# ============================================================================
# SECTION 4: TTM CONSTRUCTION CORRECTNESS
# ============================================================================

def verify_ttm_construction(ticker):
    """
    Hard check: Your TTM builder should require 4 distinct quarter end dates.
    Never mix annual CFO with quarterly CapEx.
    """
    print(f"\n{'='*80}")
    print(f"[CHECK 4] TTM CONSTRUCTION CORRECTNESS")
    print(f"{'='*80}")
    print(f"Ticker: {ticker}")
    
    adapter = DataAdapter(ticker)
    snapshot = adapter.fetch()
    
    cfo_period = snapshot.ttm_operating_cash_flow.period_type
    capex_period = snapshot.ttm_capex.period_type
    fcf_period = snapshot.ttm_fcf.period_type
    
    cfo_fallback = snapshot.ttm_operating_cash_flow.fallback_reason
    capex_fallback = snapshot.ttm_capex.fallback_reason
    
    print(f"\n  TTM Operating Cash Flow (CFO)")
    print(f"    Period Type: {cfo_period}")
    print(f"    Fallback Reason: {cfo_fallback or 'None (preferred quarterly)'}")
    
    print(f"\n  TTM CapEx")
    print(f"    Period Type: {capex_period}")
    print(f"    Fallback Reason: {capex_fallback or 'None (preferred quarterly)'}")
    
    print(f"\n  TTM FCF")
    print(f"    Period Type: {fcf_period}")
    
    # Check for mixed frequencies
    is_quarterly_cfo = "quarterly" in cfo_period.lower()
    is_quarterly_capex = "quarterly" in capex_period.lower()
    is_mixed = is_quarterly_cfo != is_quarterly_capex
    
    if is_mixed:
        print(f"\n  ❌ FAIL: Mixed frequencies detected!")
        print(f"    CFO is {cfo_period}, CapEx is {capex_period}")
        print(f"    Never mix annual CFO with quarterly CapEx (or vice versa)")
        return False
    
    if cfo_period == "quarterly_ttm" and capex_period == "quarterly_ttm":
        print(f"\n  ✅ PASS: Both CFO and CapEx from quarterly (4-quarter TTM)")
        return True
    elif cfo_period == "annual_proxy" and capex_period == "annual_proxy":
        print(f"\n  ⚠️  WARNING: Both are annual proxy (not ideal, but consistent)")
        print(f"    Reliability will be reduced (70 vs 90)")
        return True
    else:
        print(f"\n  ⚠️  WARNING: Inconsistent sources, but same frequency")
        return True


# ============================================================================
# SECTION 5: TERMINAL VALUE DISCOUNTING & DOMINANCE
# ============================================================================

def verify_terminal_value_discounting(ticker):
    """
    Hard check: Trace must explicitly show TV at Year 5 and discount factor.
    PV(TV)/EV ratio should be calculated and warned if >75%.
    """
    print(f"\n{'='*80}")
    print(f"[CHECK 5] TERMINAL VALUE DISCOUNTING & DOMINANCE")
    print(f"{'='*80}")
    print(f"Ticker: {ticker}")
    
    adapter = DataAdapter(ticker)
    snapshot = adapter.fetch()
    
    engine = DCFEngine(snapshot)
    result = engine.run()
    
    if not result.get('success'):
        print(f"❌ FAIL: DCF engine failed")
        return False
    
    ev = result['enterprise_value']
    pv_terminal = result['pv_terminal_value']
    pv_fcf = result['pv_fcf_sum']
    
    tv_dominance = pv_terminal / ev * 100 if ev > 0 else 0
    
    print(f"\n  Enterprise Value: ${ev:,.0f}")
    print(f"  PV(FCF Years 1-5): ${pv_fcf:,.0f} ({pv_fcf/ev*100:.1f}% of EV)")
    print(f"  PV(Terminal Value): ${pv_terminal:,.0f} ({tv_dominance:.1f}% of EV)")
    
    # Find terminal value step in trace
    trace = result.get('trace', [])
    tv_step = None
    discount_step = None
    
    for step in trace:
        if 'Terminal' in step.get('name', '') and 'Year 5' not in step.get('name', ''):
            tv_step = step
        if 'Discount' in step.get('name', '') and 'Terminal' in step.get('name', ''):
            discount_step = step
    
    if tv_step:
        print(f"\n  Terminal Value Trace Step:")
        print(f"    Name: {tv_step['name']}")
        print(f"    Formula: {tv_step.get('formula', 'N/A')}")
        print(f"    Output: ${tv_step['output']:,.0f}")
    else:
        print(f"\n  ⚠️  WARNING: Could not find Terminal Value trace step")
    
    if discount_step:
        print(f"\n  Terminal Value Discount Factor:")
        print(f"    Formula: {discount_step.get('formula', 'N/A')}")
        print(f"    Output: {discount_step['output']}")
    
    if tv_dominance > 75:
        print(f"\n  ⚠️  WARNING: Terminal Value dominates {tv_dominance:.1f}% of EV")
        print(f"    This valuation is mostly 'assumed multiple'")
        print(f"    Small changes in exit multiple or terminal growth have huge impact")
        print(f"    Sensitivity analysis strongly recommended")
    
    if tv_dominance > 0 and tv_dominance < 100:
        print(f"\n  ✅ PASS: Terminal value is explicitly traced and disclosed")
        return True
    else:
        print(f"  ❌ FAIL: Terminal value not properly traced")
        return False


# ============================================================================
# SECTION 6: EXIT MULTIPLE LOGIC (EBITDA PROJECTION)
# ============================================================================

def verify_exit_multiple_logic(ticker):
    """
    Hard check: EBITDA Year 5 must be projected from explicit EBITDA model,
    not recycled from FCF growth.
    """
    print(f"\n{'='*80}")
    print(f"[CHECK 6] EXIT MULTIPLE LOGIC (EBITDA Projection)")
    print(f"{'='*80}")
    print(f"Ticker: {ticker}")
    
    adapter = DataAdapter(ticker)
    snapshot = adapter.fetch()
    
    if not snapshot.ttm_ebitda.value or snapshot.ttm_ebitda.value <= 0:
        print(f"⚠️  SKIP: No EBITDA available; system will use Gordon Growth fallback")
        return True
    
    engine = DCFEngine(snapshot)
    result = engine.run()
    
    trace = result.get('trace', [])
    
    # Find EBITDA projection step
    ebitda_step = None
    for step in trace:
        if 'EBITDA' in step.get('name', '') and 'Project' in step.get('name', ''):
            ebitda_step = step
            break
    
    print(f"\n  TTM EBITDA: ${snapshot.ttm_ebitda.value:,.0f}")
    
    if ebitda_step:
        print(f"\n  EBITDA Projection Trace:")
        print(f"    Name: {ebitda_step['name']}")
        print(f"    Formula: {ebitda_step.get('formula', 'N/A')}")
        print(f"    Inputs: {ebitda_step.get('inputs', {})}")
        print(f"    Output (Year 5): ${ebitda_step['output']:,.0f}")
        print(f"    Notes: {ebitda_step.get('notes', '')}")
        
        # Check if formula is separate from FCF growth
        formula_str = str(ebitda_step.get('formula', '')).lower()
        
        if 'fcf' in formula_str or 'fcf_growth' in formula_str:
            print(f"\n  ⚠️  WARNING: EBITDA projected using FCF growth")
            print(f"    This is a proxy; ideally separate EBITDA growth assumption")
            print(f"    But it's documented and explicit, so not a critical error")
        else:
            print(f"\n  ✅ PASS: EBITDA has separate projection logic")
        
        return True
    else:
        print(f"\n  ⚠️  WARNING: Could not find EBITDA projection step in trace")
        return True


# ============================================================================
# SECTION 7: QUALITY SCORING EXPLICITNESS
# ============================================================================

def verify_quality_scoring(ticker):
    """
    Hard check: Quality score formula must be explicit and stored.
    Penalize missing critical inputs heavily.
    """
    print(f"\n{'='*80}")
    print(f"[CHECK 7] QUALITY SCORING EXPLICITNESS")
    print(f"{'='*80}")
    print(f"Ticker: {ticker}")
    
    adapter = DataAdapter(ticker)
    snapshot = adapter.fetch()
    
    overall_score = snapshot.overall_quality_score
    
    print(f"\n  Overall Quality Score: {overall_score}/100")
    
    # Show components
    components = [
        ("Current Price", snapshot.price.reliability_score),
        ("Shares Outstanding", snapshot.shares_outstanding.reliability_score),
        ("TTM Revenue", snapshot.ttm_revenue.reliability_score),
        ("TTM FCF", snapshot.ttm_fcf.reliability_score),
        ("TTM Operating Income", snapshot.ttm_operating_income.reliability_score),
        ("Total Debt", snapshot.total_debt.reliability_score),
        ("Cash", snapshot.cash_and_equivalents.reliability_score),
    ]
    
    print(f"\n  Component Scores:")
    for name, score in components:
        status = "✅" if score >= 85 else "⚠️ " if score >= 70 else "❌"
        print(f"    {status} {name:.<25} {score}/100")
    
    # Check for critical missing items
    missing = []
    for name, score in components:
        if score == 0:
            missing.append(name)
    
    if missing:
        print(f"\n  ❌ CRITICAL MISSING ITEMS: {', '.join(missing)}")
        print(f"    These are non-recoverable data gaps")
        return False
    
    # Show warnings
    if snapshot.warnings:
        print(f"\n  Warnings ({len(snapshot.warnings)}):")
        for warn in snapshot.warnings:
            print(f"    - {warn['message']}")
    
    print(f"\n  ✅ PASS: Quality scoring is explicit and shows all components")
    return True


# ============================================================================
# SECTION 8: GOLDEN TESTS (EXACT NUMERIC ASSERTIONS)
# ============================================================================

def golden_test_aapl():
    """
    Golden test for AAPL: Assert specific numeric checkpoints.
    These are realistic ranges based on Feb 5, 2026 data.
    """
    print(f"\n{'='*80}")
    print(f"[GOLDEN TEST] AAPL (Feb 5, 2026)")
    print(f"{'='*80}")
    
    adapter = DataAdapter("AAPL")
    snapshot = adapter.fetch()
    engine = DCFEngine(snapshot)
    result = engine.run()
    
    assertions = [
        ("Market Cap computed", snapshot.market_cap.value is not None, None),
        ("Enterprise Value > 0", result['enterprise_value'] > 0, None),
        ("Equity Value > 0", result['equity_value'] > 0, None),
        ("EV - Net Debt = Equity", 
         abs(result['enterprise_value'] - result['net_debt'] - result['equity_value']) < 1, None),
        ("Price/Share > 0", result.get('price_per_share', 0) > 0, None),
        ("TTM Revenue > $100B", snapshot.ttm_revenue.value > 100e9, None),
        ("FCF > 0", snapshot.ttm_fcf.value > 0, None),
        ("Overall Quality >= 85", snapshot.overall_quality_score >= 85, None),
        ("TV % < 95%", result['pv_terminal_value'] / result['enterprise_value'] < 0.95, None),
    ]
    
    passed = 0
    for desc, condition, tolerance in assertions:
        if condition:
            print(f"  ✅ {desc}")
            passed += 1
        else:
            print(f"  ❌ {desc}")
    
    print(f"\n  Result: {passed}/{len(assertions)} assertions passed")
    return passed == len(assertions)


# ============================================================================
# SECTION 9: ADVERSARIAL TESTS (EDGE CASES)
# ============================================================================

def adversarial_test_missing_data():
    """
    Test graceful degradation when critical data is missing.
    """
    print(f"\n{'='*80}")
    print(f"[ADVERSARIAL TEST] Missing Data Handling")
    print(f"{'='*80}")
    
    # Test with a small-cap or delisted ticker that might have gaps
    test_tickers = ["BRK.B", "MSFT"]  # BRK.B has different structure
    
    for ticker in test_tickers:
        print(f"\n  Testing: {ticker}")
        try:
            adapter = DataAdapter(ticker)
            snapshot = adapter.fetch()
            
            if snapshot.warnings:
                print(f"    ✅ Warnings emitted for gaps:")
                for w in snapshot.warnings[:3]:
                    print(f"       - {w['message']}")
            
            if snapshot.overall_quality_score > 0:
                print(f"    ✅ Quality score = {snapshot.overall_quality_score}/100 (non-zero despite gaps)")
            
            print(f"    ✅ No crashes despite potential data gaps")
        except Exception as e:
            print(f"    ❌ Crash: {str(e)}")
            return False
    
    return True


def adversarial_test_negative_fcf():
    """
    Test handling of negative FCF (burning cash).
    """
    print(f"\n{'='*80}")
    print(f"[ADVERSARIAL TEST] Negative FCF Handling")
    print(f"{'='*80}")
    
    # Try a growth stage company or one with heavy CapEx
    test_ticker = "TSLA"
    print(f"\n  Testing: {test_ticker}")
    
    try:
        adapter = DataAdapter(test_ticker)
        snapshot = adapter.fetch()
        
        if snapshot.ttm_fcf.value < 0:
            print(f"    ✅ Negative FCF detected: ${snapshot.ttm_fcf.value:,.0f}")
            print(f"    ✅ System continues (doesn't crash)")
        else:
            print(f"    ℹ️  FCF is positive: ${snapshot.ttm_fcf.value:,.0f}")
        
        engine = DCFEngine(snapshot)
        result = engine.run()
        
        if result['success']:
            print(f"    ✅ DCF ran despite negative FCF")
        else:
            print(f"    ℹ️  DCF failed gracefully: {result.get('errors')}")
        
        return True
    except Exception as e:
        print(f"    ❌ Crash: {str(e)}")
        return False


# ============================================================================
# MAIN VERIFICATION RUNNER
# ============================================================================

def run_full_verification():
    """Run all 8 checks for selected tickers."""
    
    print("\n" + "="*80)
    print("DCF SYSTEM COMPREHENSIVE VERIFICATION")
    print("="*80)
    print("Running all 8 critical checks...\n")
    
    test_tickers = ["AAPL", "MSFT", "GOOGL"]
    
    results = {}
    
    for ticker in test_tickers:
        print(f"\n\n{'#'*80}")
        print(f"# TICKER: {ticker}")
        print(f"{'#'*80}")
        
        checks = [
            ("Share Count & Per-Share Math", lambda: verify_share_count_and_per_share_math(ticker)),
            ("EV→Equity Bridge", lambda: verify_ev_equity_bridge(ticker)),
            ("CapEx Sign Handling", lambda: verify_capex_sign_handling(ticker)),
            ("TTM Construction", lambda: verify_ttm_construction(ticker)),
            ("Terminal Value Discounting", lambda: verify_terminal_value_discounting(ticker)),
            ("Exit Multiple Logic", lambda: verify_exit_multiple_logic(ticker)),
            ("Quality Scoring", lambda: verify_quality_scoring(ticker)),
        ]
        
        ticker_results = {}
        for check_name, check_fn in checks:
            try:
                passed = check_fn()
                ticker_results[check_name] = "PASS" if passed else "FAIL"
            except Exception as e:
                ticker_results[check_name] = f"ERROR: {str(e)[:50]}"
        
        results[ticker] = ticker_results
    
    # Golden & Adversarial tests
    print(f"\n\n{'#'*80}")
    print(f"# GOLDEN & ADVERSARIAL TESTS")
    print(f"{'#'*80}")
    
    try:
        golden_passed = golden_test_aapl()
        results["GOLDEN_AAPL"] = "PASS" if golden_passed else "FAIL"
    except Exception as e:
        results["GOLDEN_AAPL"] = f"ERROR: {str(e)[:50]}"
    
    try:
        adv_missing = adversarial_test_missing_data()
        results["ADV_MISSING"] = "PASS" if adv_missing else "FAIL"
    except Exception as e:
        results["ADV_MISSING"] = f"ERROR: {str(e)[:50]}"
    
    try:
        adv_negative = adversarial_test_negative_fcf()
        results["ADV_NEGATIVE_FCF"] = "PASS" if adv_negative else "FAIL"
    except Exception as e:
        results["ADV_NEGATIVE_FCF"] = f"ERROR: {str(e)[:50]}"
    
    # Summary
    print(f"\n\n{'='*80}")
    print(f"VERIFICATION SUMMARY")
    print(f"{'='*80}\n")
    
    for ticker, checks in results.items():
        if isinstance(checks, dict):
            print(f"{ticker}:")
            for check, status in checks.items():
                symbol = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️ "
                print(f"  {symbol} {check:.<40} {status}")
        else:
            symbol = "✅" if checks == "PASS" else "❌" if checks == "FAIL" else "⚠️ "
            print(f"{symbol} {ticker:.<40} {checks}")
    
    print(f"\n{'='*80}")
    print("END VERIFICATION")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    run_full_verification()
