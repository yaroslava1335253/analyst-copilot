"""
DCF Architecture Reference & Implementation Guide
==================================================

This file documents the new DCF architecture and how to use it.

ARCHITECTURE OVERVIEW
=====================

Old approach (legacy/engine.py):
  - calculate_comprehensive_analysis() did everything: data fetch, assumptions, calculations
  - Mixed yfinance raw data with financial logic
  - Silent defaults for missing data
  - Terminal value used FCF growth as EBITDA growth proxy
  - EV vs Market Cap comparison but unclear equity bridge

New approach (modular, traceable, robust):
  1. DataAdapter: Fetch + normalize yfinance data, score quality, explicit fallbacks
  2. DCFEngine: Pure calculation engine, pluggable strategies, full trace
  3. Integration Layer: Adapts new engine to UI, maintains backward compatibility

KEY BENEFITS
============

1. Deterministic Finance
   - Every calculation is explicit in code
   - No hidden defaults or assumptions
   - Full formula chain recorded in trace
   
2. Full Traceability
   - Every intermediate value recorded with:
     - name (what was calculated)
     - formula (how it was calculated)
     - inputs (what was used)
     - output (the result)
     - units, notes, timestamps
   - Trace available as structured JSON for inspection

3. Internal Consistency
   - EV = PV(explicit FCF) + PV(Terminal Value)
   - Equity Value = EV - Net Debt (explicit definition)
   - Price/share = Equity Value / Shares Outstanding
   - Terminal value strategies pluggable but not intermingled

4. Robust to yfinance Gaps
   - Quality scores on every field
   - Explicit fallback hierarchy (quarterly → annual)
   - Warnings when fallbacks used
   - Safe failures when data insufficient

USAGE EXAMPLES
==============

Example 1: Simple DCF with auto-assumptions
-------------------------------------------

from data_adapter import DataAdapter
from dcf_engine import DCFEngine, DCFAssumptions

# Fetch and normalize data
adapter = DataAdapter("AAPL")
snapshot = adapter.fetch()

# Create engine with default assumptions
# (will auto-assign WACC, growth, exit multiple based on company size)
engine = DCFEngine(snapshot)
result = engine.run()

if result["success"]:
    print(f"Enterprise Value: ${result['enterprise_value']/1e9:.1f}B")
    print(f"Equity Value: ${result['equity_value']/1e9:.1f}B")
    print(f"Price per share: ${result['price_per_share']:.2f}")
    print(f"Data Quality: {result.get('data_quality', {}).get('overall_score', 0):.0f}%")

# Inspect trace if needed
import json
trace = result.get("trace", [])
for step in trace:
    print(f"{step['name']}: {step['output']} {step['output_units']}")

---

Example 2: DCF with custom assumptions
---------------------------------------

assumptions = DCFAssumptions(
    forecast_years=5,
    wacc=0.095,  # 9.5% (override auto)
    fcf_growth_rate=0.06,  # 6% growth (override auto)
    exit_multiple=14,  # 14x EV/EBITDA for exit
    terminal_value_method="exit_multiple",
    tax_rate=0.28,
    terminal_growth_rate=0.025
)

engine = DCFEngine(snapshot, assumptions)
result = engine.run()

---

Example 3: Fallback to Gordon Growth if EBITDA unavailable
-----------------------------------------------------------

# DataAdapter will warn if EBITDA unreliable
# DCFEngine will auto-switch terminal strategy

# If you force it:
assumptions = DCFAssumptions(terminal_value_method="gordon_growth")
engine = DCFEngine(snapshot, assumptions)
result = engine.run()
# Warnings will show if EBITDA was unavailable

---

Example 4: Check data quality and get warnings
-----------------------------------------------

from dcf_integration import calculate_dcf_with_traceability

full_result = calculate_dcf_with_traceability("TSLA")

# Data quality report
data_quality = full_result["data_quality"]
print(f"Overall Quality Score: {data_quality['overall_score']}/100")
print("Key Inputs:")
for field_name, field_data in data_quality["key_inputs"].items():
    reliability = field_data.get("reliability_score", 0)
    is_estimated = field_data.get("is_estimated", False)
    if reliability < 80:
        print(f"  ⚠️ {field_name}: {reliability}/100 {'[ESTIMATED]' if is_estimated else ''}")

print("\nWarnings:")
for warning in data_quality["warnings"]:
    print(f"  {warning['code']}: {warning['message']}")

---

API REFERENCE
=============

DataAdapter
-----------

adapter = DataAdapter(ticker: str)
snapshot = adapter.fetch() -> NormalizedFinancialSnapshot

Fetches from yfinance and normalizes. Returns snapshot with quality metadata.

Methods:
  - _fetch_price_and_shares(): Market cap, price, shares
  - _fetch_balance_sheet(): Debt, cash, equity items
  - _fetch_cash_flow(): CFO, CapEx, TTM FCF
  - _fetch_income_statement(): Revenue, EBITDA, taxes
  - _fetch_quarterly_history(): Last 8 quarters for trends

NormalizedFinancialSnapshot
---------------------------

Contains all financial data with metadata:

  Prices:
    - price: DataQualityMetadata (current price)
    - market_cap: (current market cap)
    - shares_outstanding: (diluted or basic)
    
  Balance Sheet:
    - total_debt: (sum of all debt lines)
    - cash_and_equivalents: (cash balances)
    - minority_interest, preferred_stock: (if available)
    
  Cash Flow (TTM):
    - ttm_operating_cash_flow: (CFO)
    - ttm_capex: (Capital expenditure, absolute value)
    - ttm_fcf: (CFO - CapEx proxy)
    
  Income Statement (TTM):
    - ttm_revenue
    - ttm_operating_income
    - ttm_net_income
    - ttm_ebitda
    - effective_tax_rate
    
  Metadata:
    - overall_quality_score: (0-100, average of key field scores)
    - warnings: (list of {code, message, severity})
    - errors: (critical missing data)
    - quarterly_history: (last 8 Q's for trend analysis)

DCFAssumptions
--------------

assumptions = DCFAssumptions(
    forecast_years: int = 5,
    wacc: float = None,  # Auto-set if None
    terminal_growth_rate: float = 0.03,  # Gordon Growth perpetual growth
    exit_multiple: int = None,  # Auto-set if None
    tax_rate: float = None,  # Auto-set if None
    fcf_growth_rate: float = None,  # Auto-set if None
    discount_convention: str = "end_of_year",  # or "mid_year"
    terminal_value_method: str = "exit_multiple"  # or "gordon_growth"
)

DCFEngine
---------

engine = DCFEngine(snapshot: NormalizedFinancialSnapshot, 
                   assumptions: DCFAssumptions = None)

Methods:
  - validate_inputs() -> bool: Check minimum data available
  - set_assumptions_from_defaults(): Auto-fill missing assumptions
  - run() -> dict: Execute full DCF, return results + trace

Returns:
  {
    "success": bool,
    "enterprise_value": float,  # PV(FCF) + PV(TV)
    "equity_value": float,  # EV - net debt
    "net_debt": float,  # Total debt - cash
    "price_per_share": float,  # Equity value / shares
    "shares_outstanding": float,
    "pv_fcf_sum": float,  # PV of explicit 5-year FCF
    "pv_terminal_value": float,  # PV of terminal value
    "terminal_value_yearN": float,  # Non-discounted terminal value
    "fcf_projections": [  # 5-year projections
      {
        "year": int,
        "fcf": float,
        "discount_factor": float,
        "pv": float
      }
    ],
    "sanity_checks": {  # Validation results
      "ev_vs_market_cap": {...},
      "ev_revenue_multiple": float,
      "ev_ebitda_multiple": float,
      "terminal_value_dominance": float,  # PV(TV) / EV
      ...
    },
    "assumptions": {...},  # Used assumptions
    "errors": [...],  # Errors encountered
    "warnings": [...],  # Non-critical warnings
    "trace": [  # Full calculation trace
      {
        "name": "Step name",
        "formula": "Mathematical formula",
        "inputs": {...},
        "output": number,
        "output_units": "USD|rate|multiple|...",
        "notes": "Additional context"
      }
    ]
  }

Terminal Value Strategies
--------------------------

Both return (terminal_value, pv_terminal_value) and add to trace.

GordonGrowthTerminalValue:
  - Formula: FCF_{n+1} / (WACC - g)
  - Parameters: terminal_growth_rate
  - When: EBITDA unavailable, stable mature companies
  
ExitMultipleTerminalValue:
  - Formula: EBITDA_n * exit_multiple
  - Parameters: exit_multiple (12-20x typical)
  - When: EBITDA available, exit scenario relevant

NetDebtCalculator
------------------

net_debt, details = NetDebtCalculator.calculate(snapshot, trace)

Computes: Net Debt = Total Debt - Cash
Used for EV → Equity bridge.

DATA QUALITY SCORES
===================

Each field has reliability_score (0-100):
  100: Clean, recent quarterly or annual data, all sources confirm
  90-99: Good, quarterly available, minor inconsistencies
  70-89: Fair, fallback to annual, estimated components
  50-69: Low, missing lines, approximations, stale data
  <50: Very low, significant gaps, may not be reliable
  0: Critical missing data, field cannot be used

Sources of score reduction:
  - None/NaN values (-30 points)
  - Inconsistent across sources (-15 points)
  - Missing quarterly data; using annual (-20 points)
  - Estimated/imputed value (-25 points)
  - Stale data (>90 days) (-10 points)

FALLBACK HIERARCHY (Explicit + Traced)
======================================

TTM Operating Cash Flow:
  1. Sum of last 4 quarters (if all available)
  2. Last annual (with "annual_proxy" note)
  3. None / error

TTM Capital Expenditure:
  Same hierarchy as OCF

TTM Revenue:
  1. Sum of last 4 quarters
  2. Last annual
  3. None / error

TTM EBITDA:
  1. Direct EBITDA line if available
  2. Operating Income + D&A (if both available)
  3. None / fallback to Gordon Growth

Tax Rate:
  1. Effective rate from (1 - NI/OI) if both available
  2. Default 25%

WACC Auto-Assignment (by revenue size):
  >$50B: 8% (mega-cap tech, stable)
  $10B-$50B: 9.5% (mid-cap)
  <$10B: 11% (smaller, higher risk)

Exit Multiple Auto-Assignment (by revenue size):
  >$50B: 18x EV/EBITDA
  $10B-$50B: 15x
  <$10B: 12x

FCF Growth Rate Auto-Assignment:
  1. Historical revenue CAGR (if 8+ quarters available)
  2. Capped to [3%, 25%]
  3. Default 8%

SANITY CHECKS (Automatic)
==========================

Every run performs:

1. Terminal Growth Valid
   - g < WACC (must be true for Gordon Growth)
   - Hard error if violated

2. PV(FCF) <= Nominal
   - Discount factors should reduce nominal values
   - Warning if violated

3. Terminal Value Dominance
   - If PV(TV) > 75% of EV, warn
   - Indicates high sensitivity to terminal assumptions

4. EV vs Market Cap
   - If available, show difference
   - Warning if >50% different (may indicate mispricing or bad assumptions)

5. Output Multiples
   - EV/Revenue: warn if <0.5x or >30x
   - EV/EBITDA: warn if <5x or >50x

6. Equity Value Availability
   - If shares unavailable, mark as "unavailable"
   - Do not impute shares

INTEGRATION WITH UI (app.py)
=============================

Use dcf_integration.py for backward compatibility:

from dcf_integration import calculate_dcf_with_traceability, format_dcf_for_ui

# Full result with all diagnostics
full_result = calculate_dcf_with_traceability("AAPL")

# UI-friendly format (for Streamlit display)
ui_result = format_dcf_for_ui(full_result)

# Extract specific components
data_quality = full_result["data_quality"]
trace_json = full_result["dcf_result"]["trace"]

UI should display:

1. Data Quality Panel (collapsible)
   - Overall quality score
   - Key inputs with reliability scores and period
   - Warnings (yellow icon)
   - Errors (red icon)

2. Main Metrics
   - Enterprise Value
   - Equity Value
   - Price per Share
   - Method (Exit Multiple / Gordon Growth)

3. Assumption Summary
   - WACC, FCF Growth, Tax Rate
   - Terminal method & parameters

4. Projection Table
   - 5-year FCF projection with PVs
   - Terminal value breakdown

5. Scenarios (optional, if computed)
   - Bear/Base/Bull cases

6. Reality Check
   - If EV: compare EV to current EV (Market Cap + Net Debt)
   - If Equity: compare to Market Cap
   - Show difference % and implied growth

7. Calculation Trace (expert mode, collapsible)
   - Full JSON trace with all formulas
   - Download as CSV/JSON option

TESTING
=======

Run unit tests:
  python -m pytest test_dcf_engine.py -v

Coverage includes:
  - Data quality metadata
  - Terminal value strategies
  - Net debt calculation
  - Full DCF workflow
  - Edge cases (zero FCF, negative net debt, missing EBITDA)
  - Integration end-to-end

MIGRATION FROM OLD ENGINE
==========================

Legacy code can use:

from dcf_integration import legacy_calculate_comprehensive_analysis

result = legacy_calculate_comprehensive_analysis(
    income_stmt, balance_sheet, quarterly_data,
    ticker_symbol="AAPL",
    wacc_override=None,
    fcf_growth_override=None
)

This returns a dict compatible with old app.py format, but:
- Internally uses new engine
- Has access to trace and quality scores
- Provides fallbacks explicitly

Gradually migrate by:
  1. Replace calls to old calculate_comprehensive_analysis with new DCFEngine
  2. Update UI to show data_quality panel
  3. Add "View Trace" collapsible for expert mode
  4. Update "Reality Check" to use correct EV/Equity comparison

FUTURE ENHANCEMENTS
====================

1. Scenario builder (user-defined Bear/Base/Bull)
2. Sensitivity analysis (tornado chart)
3. FCFF mode (explicit EBIT*(1-T) + D&A - CapEx - ΔNW)
4. Multi-stage growth model (explicit ramp to perpetual growth)
5. Monte Carlo simulation for probability distributions
6. API to external data sources (Bloomberg, S&P, etc.)
7. Historical trace comparison (month-over-month changes)
"""
