"""
DCF Integration Layer: Compatibility bridge between new DCFEngine and legacy code
====================================================================================
Provides adapter functions that maintain backward compatibility with app.py while
using the new architecture internally.
"""

from data_adapter import DataAdapter, NormalizedFinancialSnapshot
from dcf_engine import DCFEngine, DCFAssumptions


def calculate_dcf_with_traceability(ticker: str, wacc_override: float = None, 
                                    fcf_growth_override: float = None) -> dict:
    """
    Execute complete DCF valuation with new architecture.
    
    Returns dict with:
    - enterprise_value, equity_value, price_per_share
    - detailed trace of all calculations
    - data quality scores
    - sanity checks and warnings
    """
    
    # Step 1: Fetch and normalize data
    adapter = DataAdapter(ticker)
    snapshot = adapter.fetch()
    
    # Step 2: Build assumptions
    assumptions = DCFAssumptions(
        forecast_years=5,
        wacc=wacc_override / 100 if wacc_override else None,
        fcf_growth_rate=fcf_growth_override / 100 if fcf_growth_override else None,
        terminal_value_method="exit_multiple",  # Prefer exit multiple if EBITDA available
        discount_convention="end_of_year"
    )
    
    # Step 3: Run DCF engine
    engine = DCFEngine(snapshot, assumptions)
    result = engine.run()
    
    return {
        "snapshot": snapshot.to_dict(),
        "dcf_result": result,
        "data_quality": {
            "overall_score": snapshot.overall_quality_score,
            "key_inputs": {
                "price": snapshot.price.to_dict(),
                "market_cap": snapshot.market_cap.to_dict(),
                "ttm_revenue": snapshot.ttm_revenue.to_dict(),
                "ttm_fcf": snapshot.ttm_fcf.to_dict(),
                "ttm_ebitda": snapshot.ttm_ebitda.to_dict(),
                "total_debt": snapshot.total_debt.to_dict(),
                "cash_and_equivalents": snapshot.cash_and_equivalents.to_dict(),
                "shares_outstanding": snapshot.shares_outstanding.to_dict(),
            },
            "warnings": snapshot.warnings,
            "errors": snapshot.errors
        }
    }


def format_dcf_for_ui(full_result: dict) -> dict:
    """
    Convert detailed DCF result to UI-friendly format.
    Extracts key metrics and scenarios while preserving trace availability.
    """
    dcf = full_result.get("dcf_result", {})
    snapshot = full_result.get("snapshot", {})
    
    if not dcf.get("success"):
        return {
            "error": True,
            "errors": dcf.get("errors", []),
            "warnings": dcf.get("warnings", [])
        }
    
    # Extract base metrics
    result_dict = {
        "enterprise_value": dcf.get("enterprise_value", 0),
        "enterprise_value_b": dcf.get("enterprise_value", 0) / 1e9,
        "equity_value": dcf.get("equity_value", 0),
        "equity_value_b": dcf.get("equity_value", 0) / 1e9,
        "price_per_share": dcf.get("price_per_share"),
        "net_debt": dcf.get("net_debt"),
        "net_debt_details": dcf.get("net_debt_details", {}),
        
        # Input metrics
        "ttm_revenue_b": snapshot.get("ttm_revenue", {}).get("value", 0) / 1e9,
        "ttm_fcf_b": snapshot.get("ttm_fcf", {}).get("value", 0) / 1e9,
        "fcf_margin": (snapshot.get("ttm_fcf", {}).get("value", 0) / 
                      snapshot.get("ttm_revenue", {}).get("value", 1)) * 100 if snapshot.get("ttm_revenue", {}).get("value") else 0,
        
        # Assumptions
        "assumed_fcf_growth": dcf.get("assumptions", {}).get("fcf_growth_rate", 0) * 100,
        "wacc": dcf.get("assumptions", {}).get("wacc", 0) * 100,
        "terminal_value_method": dcf.get("assumptions", {}).get("terminal_value_method"),
        
        # Projections & scenarios
        "projected_fcf": dcf.get("fcf_projections", []),
        "scenarios": _build_scenarios(full_result),
        
        # Sanity checks
        "sanity_check": _build_sanity_check(full_result),
        
        # Trace & diagnostics (hidden by default, shown on request)
        "trace_available": True,
        "full_trace": dcf.get("trace", []),
        "data_quality_score": full_result.get("data_quality", {}).get("overall_score", 0),
        
        # Warnings
        "warnings": dcf.get("warnings", []),
        "errors": dcf.get("errors", [])
    }
    
    # Historical revenue CAGR (if available)
    snapshot_obj = full_result.get("snapshot", {})
    # Note: quarterly_history in snapshot would need to be extracted if available
    
    return result_dict


def _build_scenarios(full_result: dict) -> dict:
    """Extract and format scenario analysis."""
    dcf = full_result.get("dcf_result", {})
    # For now, keep scenarios from DCF if computed
    # In the future, scenarios could be computed in DCFEngine
    return {}


def _build_sanity_check(full_result: dict) -> dict:
    """Extract sanity checks in UI-friendly format."""
    dcf = full_result.get("dcf_result", {})
    snapshot = full_result.get("snapshot", {})
    
    sanity = dcf.get("sanity_checks", {})
    
    result = {}
    
    # Market cap comparison
    market_cap = snapshot.get("market_cap", {}).get("value")
    if market_cap:
        result["current_market_cap_b"] = market_cap / 1e9
        
        ev = dcf.get("enterprise_value")
        if ev:
            result["dcf_vs_market_diff_pct"] = ((ev - market_cap) / market_cap) * 100
    
    # Multiples
    if "ev_revenue_multiple" in sanity:
        result["ev_revenue_multiple"] = sanity["ev_revenue_multiple"]
    
    if "ev_ebitda_multiple" in sanity:
        result["ev_ebitda_multiple"] = sanity["ev_ebitda_multiple"]
    
    return result


# Backward-compatible function signature for old code
def legacy_calculate_comprehensive_analysis(income_stmt, balance_sheet, quarterly_data, 
                                           ticker_symbol=None, cash_flow=None, 
                                           quarterly_cash_flow=None, wacc_override=None, 
                                           fcf_growth_override=None) -> dict:
    """
    Legacy wrapper that calls new DCF engine but returns format compatible with old app.py.
    
    This allows gradual migration from old to new architecture.
    """
    if ticker_symbol is None:
        return {"error": "Ticker symbol required"}
    
    full_result = calculate_dcf_with_traceability(
        ticker_symbol, 
        wacc_override=wacc_override,
        fcf_growth_override=fcf_growth_override
    )
    
    ui_result = format_dcf_for_ui(full_result)
    
    # Transform to legacy format
    legacy_result = {
        "dcf": {
            "enterprise_value": ui_result.get("enterprise_value"),
            "enterprise_value_b": ui_result.get("enterprise_value_b"),
            "ttm_revenue_b": ui_result.get("ttm_revenue_b"),
            "ttm_fcf_b": ui_result.get("ttm_fcf_b"),
            "fcf_margin": ui_result.get("fcf_margin"),
            "assumed_fcf_growth": ui_result.get("assumed_fcf_growth"),
            "wacc": ui_result.get("wacc"),
            "terminal_value_method": ui_result.get("terminal_value_method"),
            "projected_fcf": ui_result.get("projected_fcf"),
            "scenarios": ui_result.get("scenarios", {}),
            "sanity_check": ui_result.get("sanity_check", {}),
        },
        "dupont": {},  # Keep placeholder for legacy code
        "quality_metrics": {},
        "trend_analysis": {},
    }
    
    return legacy_result


def get_trace_json(full_result: dict) -> str:
    """Export calculation trace as JSON for inspection."""
    import json
    trace = full_result.get("dcf_result", {}).get("trace", [])
    return json.dumps(trace, indent=2, default=str)


def get_data_quality_report(full_result: dict) -> dict:
    """Extract data quality information for UI display."""
    quality = full_result.get("data_quality", {})
    
    return {
        "overall_score": quality.get("overall_score", 0),
        "inputs": quality.get("key_inputs", {}),
        "warnings": quality.get("warnings", []),
        "errors": quality.get("errors", [])
    }
