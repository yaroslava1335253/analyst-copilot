"""
Quick test: Validate new DCF engine with real yfinance data
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_adapter import DataAdapter
from dcf_engine import DCFEngine, DCFAssumptions
from dcf_integration import calculate_dcf_with_traceability, format_dcf_for_ui
import json


def test_with_ticker(ticker: str):
    """Test DCF valuation with a real ticker."""
    print(f"\n{'='*70}")
    print(f"Testing DCF Engine with {ticker}")
    print(f"{'='*70}\n")
    
    try:
        # Step 1: Fetch data
        print(f"[1/4] Fetching data from yfinance...")
        adapter = DataAdapter(ticker)
        snapshot = adapter.fetch()
        
        # Show data quality
        print(f"\n✓ Data fetched")
        print(f"  Overall Quality Score: {snapshot.overall_quality_score:.0f}/100")
        print(f"  Quarters Available: {snapshot.num_quarters_available}")
        
        if snapshot.warnings:
            print(f"\n  ⚠️  Warnings during fetch:")
            for w in snapshot.warnings[:3]:  # Show first 3 warnings
                print(f"    - {w['code']}: {w['message']}")
        
        # Step 2: Show key inputs
        print(f"\n[2/4] Key Financial Data:")
        if snapshot.ttm_revenue.value:
            print(f"  TTM Revenue: ${snapshot.ttm_revenue.value/1e9:.1f}B (reliability: {snapshot.ttm_revenue.reliability_score}/100)")
        if snapshot.ttm_fcf.value:
            print(f"  TTM FCF: ${snapshot.ttm_fcf.value/1e9:.1f}B (reliability: {snapshot.ttm_fcf.reliability_score}/100)")
        if snapshot.ttm_ebitda.value:
            print(f"  TTM EBITDA: ${snapshot.ttm_ebitda.value/1e9:.1f}B (reliability: {snapshot.ttm_ebitda.reliability_score}/100)")
        if snapshot.market_cap.value:
            print(f"  Market Cap: ${snapshot.market_cap.value/1e9:.1f}B")
        if snapshot.shares_outstanding.value:
            print(f"  Shares Outstanding: {snapshot.shares_outstanding.value/1e6:.0f}M")
        
        # Step 3: Run DCF
        print(f"\n[3/4] Running DCF Valuation...")
        assumptions = DCFAssumptions(
            wacc=None,  # Auto-assign based on size
            fcf_growth_rate=None,  # Auto-assign from historical
            terminal_value_method="exit_multiple"  # Prefer exit multiple
        )
        
        engine = DCFEngine(snapshot, assumptions)
        result = engine.run()
        
        if not result["success"]:
            print(f"  ❌ DCF failed:")
            for error in result.get("errors", []):
                print(f"    - {error}")
            return
        
        print(f"  ✓ Valuation complete")
        
        # Step 4: Display results
        print(f"\n[4/4] Valuation Results:")
        print(f"\n  Enterprise Value: ${result['enterprise_value']/1e9:.2f}B")
        print(f"  Less: Net Debt: ${result['net_debt']/1e9:.2f}B")
        print(f"  = Equity Value: ${result['equity_value']/1e9:.2f}B")
        
        if result['price_per_share']:
            print(f"  Price per Share: ${result['price_per_share']:.2f}")
            if snapshot.price.value:
                current_price = snapshot.price.value
                upside = ((result['price_per_share'] - current_price) / current_price) * 100
                print(f"    Current Price: ${current_price:.2f}")
                print(f"    Implied Upside/(Downside): {upside:+.1f}%")
        
        # Assumptions used
        assumptions_used = result.get("assumptions", {})
        print(f"\n  Assumptions:")
        print(f"    WACC: {assumptions_used.get('wacc', 0)*100:.1f}%")
        print(f"    FCF Growth: {assumptions_used.get('fcf_growth_rate', 0)*100:.1f}%")
        print(f"    Terminal Method: {assumptions_used.get('terminal_value_method', 'N/A')}")
        print(f"    Exit Multiple: {assumptions_used.get('exit_multiple', 'N/A')}x")
        
        # Sanity checks
        sanity = result.get("sanity_checks", {})
        if "ev_vs_market_cap" in sanity:
            mc = sanity["ev_vs_market_cap"]
            print(f"\n  Reality Check:")
            print(f"    DCF EV: ${mc.get('dcf_ev_b', 0):.2f}B")
            print(f"    Market Cap: ${mc.get('market_cap_b', 0):.2f}B")
            diff = mc.get('diff_pct', 0)
            print(f"    Difference: {diff:+.1f}%")
        
        if "ev_ebitda_multiple" in sanity:
            print(f"    EV/EBITDA Multiple: {sanity['ev_ebitda_multiple']:.1f}x")
        
        if "ev_revenue_multiple" in sanity:
            print(f"    EV/Revenue Multiple: {sanity['ev_revenue_multiple']:.1f}x")
        
        # Terminal value dominance
        if "terminal_value_dominance" in sanity:
            tv_ratio = sanity["terminal_value_dominance"]
            print(f"    Terminal Value as % of EV: {tv_ratio*100:.1f}%")
        
        # Warnings
        if result.get("warnings"):
            print(f"\n  ⚠️  Warnings:")
            for w in result["warnings"]:
                print(f"    - {w}")
        
        # Show sample trace steps
        if result.get("trace"):
            print(f"\n  Calculation Trace (first 5 steps):")
            for step in result["trace"][:5]:
                print(f"    • {step['name']}")
                if step.get('formula'):
                    print(f"      Formula: {step['formula']}")
                print(f"      Output: {step.get('output', 'N/A')} {step.get('output_units', '')}")
        
        print(f"\n✅ Test completed successfully\n")
        
        return result
    
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Test with a few tickers
    tickers = ["AAPL", "MSFT", "GOOGL", "TSLA"]
    
    results = {}
    for ticker in tickers:
        try:
            result = test_with_ticker(ticker)
            if result:
                results[ticker] = result
        except Exception as e:
            print(f"Failed to test {ticker}: {e}\n")
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}\n")
    
    for ticker, result in results.items():
        if result and result.get("success"):
            ev_b = result.get("enterprise_value", 0) / 1e9
            price = result.get("price_per_share", 0)
            print(f"{ticker:6s}: EV=${ev_b:7.2f}B | Price=${price:7.2f}")
    
    print()
