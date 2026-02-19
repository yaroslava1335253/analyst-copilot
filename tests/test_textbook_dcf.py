"""Test the textbook DCF implementation with horizon rule and ROIC-based reinvestment."""

from data_adapter import DataAdapter
from dcf_engine import DCFEngine, DCFAssumptions

def test_textbook_dcf(ticker="MSFT"):
    print(f"Testing Textbook DCF for {ticker}")
    print("=" * 60)
    
    # Fetch data
    adapter = DataAdapter(ticker)
    snapshot = adapter.fetch()
    print(f"Ticker: {snapshot.ticker}")
    print(f"TTM Revenue: ${snapshot.ttm_revenue.value/1e9:.1f}B" if snapshot.ttm_revenue.value else "TTM Revenue: N/A")
    print(f"Market Cap: ${snapshot.market_cap.value/1e9:.0f}B" if snapshot.market_cap.value else "Market Cap: N/A")
    
    # Run DCF with driver model
    engine = DCFEngine(snapshot)
    result = engine.run()
    
    if result['success']:
        print("\n=== TEXTBOOK DCF RESULTS ===")
        assumptions = result['assumptions']
        
        # Horizon rule
        print(f"\n--- HORIZON RULE ---")
        print(f"Forecast Years: {assumptions.get('forecast_years', 5)}")
        print(f"Display Years: {assumptions.get('display_years', 5)}")
        print(f"Is Large Cap: {assumptions.get('is_large_cap', False)}")
        print(f"Horizon Reason: {assumptions.get('horizon_reason', 'N/A')}")
        
        # Growth fade
        print(f"\n--- GROWTH FADE ---")
        print(f"Near-term Growth: {assumptions.get('near_term_growth_rate', 0)*100:.1f}% (Years 1-3)")
        print(f"Terminal Growth: {assumptions.get('stable_growth_rate', 0)*100:.1f}% (g_perp)")
        growth_rates = assumptions.get('revenue_growth_rates', [])
        if growth_rates:
            print(f"Growth Schedule: {[f'{g*100:.1f}%' for g in growth_rates]}")
        
        # ROIC-based reinvestment
        print(f"\n--- ROIC & REINVESTMENT ---")
        print(f"Current ROIC: {assumptions.get('base_roic', 0)*100:.1f}%")
        print(f"Industry ROIC: {assumptions.get('industry_roic', 0)*100:.1f}%")
        print(f"Terminal ROIC: {assumptions.get('terminal_roic', 0)*100:.1f}%")
        print(f"Terminal Reinv Rate: {assumptions.get('terminal_reinvestment_rate', 0)*100:.1f}%")
        
        # Projections summary
        projs = assumptions.get('yearly_projections', [])
        if projs:
            print(f"\n--- YEARLY PROJECTIONS ({len(projs)} years) ---")
            for p in projs[:5]:  # Show first 5
                print(f"Y{p['year']}: Rev=${p['revenue']/1e9:.1f}B (g={p['revenue_growth']*100:.1f}%), NOPAT=${p['nopat']/1e9:.1f}B, Reinv={p['reinvestment_rate']*100:.0f}%, FCFF=${p['fcff']/1e9:.1f}B")
            if len(projs) > 5:
                print(f"  ... ({len(projs)-5} more years to terminal)")
                p = projs[-1]
                print(f"Y{p['year']}: Rev=${p['revenue']/1e9:.1f}B (g={p['revenue_growth']*100:.1f}%), NOPAT=${p['nopat']/1e9:.1f}B, Reinv={p['reinvestment_rate']*100:.0f}%, FCFF=${p['fcff']/1e9:.1f}B")
        
        # Terminal value
        print(f"\n--- TERMINAL VALUE ---")
        print(f"TV Method: {assumptions.get('terminal_value_method', '')}")
        print(f"Gordon TV: ${assumptions.get('tv_gordon_growth', 0)/1e9:.0f}B" if assumptions.get('tv_gordon_growth') else "Gordon TV: N/A")
        print(f"PV(Gordon TV): ${assumptions.get('pv_tv_gordon_growth', 0)/1e9:.0f}B" if assumptions.get('pv_tv_gordon_growth') else "PV(Gordon TV): N/A")
        
        # Final valuation
        print(f"\n--- VALUATION ---")
        print(f"Enterprise Value: ${result['enterprise_value']/1e9:.0f}B")
        print(f"Equity Value: ${result['equity_value']/1e9:.0f}B")
        print(f"Price per Share: ${result['price_per_share']:.2f}" if result.get('price_per_share') else "Price per Share: N/A")
        
        # Consistency checks
        sanity = result.get('sanity_checks', {})
        print(f"\n--- CONSISTENCY CHECKS ---")
        print(f"TV % of EV: {sanity.get('tv_pct_of_ev', 0):.1f}%")
        print(f"Terminal Growth Valid: {sanity.get('terminal_growth_valid', False)}")
        
        # Warnings
        if result['warnings']:
            print(f"\n--- WARNINGS ({len(result['warnings'])}) ---")
            for w in result['warnings'][:3]:
                print(f"  ⚠️ {w[:80]}..." if len(w) > 80 else f"  ⚠️ {w}")
    else:
        print("DCF FAILED:")
        for e in result['errors']:
            print(f"  ERROR: {e}")

if __name__ == "__main__":
    test_textbook_dcf("MSFT")
