#!/usr/bin/env python3
"""Test the updated FCFF methodology."""
from data_adapter import DataAdapter
from dcf_engine import DCFEngine, DCFAssumptions

print('Testing updated DCF methodology...')
adapter = DataAdapter('MSFT')
snapshot = adapter.fetch()

print('\n=== Working Capital / TTM ΔNWC ===')
if hasattr(snapshot, 'ttm_delta_nwc') and snapshot.ttm_delta_nwc.value:
    print(f"TTM ΔNWC (from CF): ${snapshot.ttm_delta_nwc.value/1e9:.2f}B")
    print(f"Source: {snapshot.ttm_delta_nwc.source_path}")
else:
    print("TTM ΔNWC not available")

print('\n=== Interest Expense ===')
if hasattr(snapshot, 'ttm_interest_expense') and snapshot.ttm_interest_expense.value:
    print(f"TTM Interest Expense: ${snapshot.ttm_interest_expense.value/1e9:.2f}B")
else:
    print("TTM Interest Expense not available")

# Run DCF
assumptions = DCFAssumptions()
assumptions.wacc = 0.10
assumptions.fcf_growth_rate = 0.10

engine = DCFEngine(snapshot, assumptions)
result = engine.run()

if result.get('success'):
    a = result.get('assumptions', {})
    print('\n=== DCF Results ===')
    print(f"FCFF Method: {a.get('fcff_method')}")
    fcff = a.get('ttm_fcff')
    print(f"TTM FCFF: ${fcff/1e9:.2f}B" if fcff else "N/A")
    ratio = a.get('fcff_ebitda_ratio')
    print(f"FCFF/EBITDA Ratio: {ratio*100:.1f}%" if ratio else "N/A")
    gordon = a.get('implied_gordon_ev_ebitda')
    print(f"Implied Gordon EV/EBITDA: {gordon:.1f}x" if gordon else "N/A")
    req = a.get('required_fcff_ebitda_for_exit')
    print(f"Required FCFF/EBITDA for Exit: {req*100:.1f}%" if req else "N/A")
    price_exit = a.get('price_exit_multiple')
    print(f"Price (Exit Multiple): ${price_exit:.2f}" if price_exit else "N/A")
    price_gordon = a.get('price_gordon_growth')
    print(f"Price (Gordon): ${price_gordon:.2f}" if price_gordon else "N/A")
    
    print('\n=== Warnings ===')
    for w in result.get('warnings', []):
        print(f"  - {w[:100]}...")
else:
    print('FAILED')
    print(result.get('errors'))
