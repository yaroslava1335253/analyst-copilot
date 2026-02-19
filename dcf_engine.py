"""
DCFEngine: Deterministic DCF valuation with full traceability
==============================================================
Implements:
- Pluggable terminal value strategies (Gordon Growth, Exit Multiple)
- Industry-based exit multiples from Damodaran data
- Explicit discount factor tracking
- Full calculation trace for every intermediate value
- Sanity checks and warnings
- Clear EV→Equity bridge with net debt definition
"""

import json
import copy
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Optional, List, Tuple, Union
from dataclasses import dataclass, asdict, field
from data_adapter import NormalizedFinancialSnapshot, DataQualityMetadata
from industry_multiples import get_industry_multiple, DAMODARAN_SOURCE_URL, DAMODARAN_DATA_DATE


@dataclass
class DCFAssumptions:
    """User-provided or default assumptions for DCF model."""
    forecast_years: int = 5  # Will be auto-set to 10 for large caps
    display_years: int = 5  # Always show 5 years in UI, compute up to 10
    wacc: float = None  # Will be auto-calculated if None
    terminal_growth_rate: float = 0.03  # Gordon Growth terminal growth (3.0% default for developed markets)
    exit_multiple: float = None  # For Exit Multiple strategy (e.g., 15 for 15x EV/EBITDA)
    tax_rate: float = None  # Will use effective rate from snapshot if None
    fcf_growth_rate: float = None  # Will be auto-calculated from historical if None
    discount_convention: str = "end_of_year"  # or "mid_year"
    terminal_value_method: str = "gordon_growth"  # Textbook primary; "exit_multiple" as cross-check
    
    # ========== HORIZON RULE ==========
    # Large-cap threshold: market cap > $200B OR revenue > $50B → 10-year forecast
    is_large_cap: bool = False  # Auto-detected from market cap/revenue
    horizon_reason: str = None  # "large_cap_market_cap", "large_cap_revenue", "standard"
    
    # Industry multiple metadata
    industry_multiple_source: str = None  # "current" or "fallback_gordon"
    damodaran_industry: str = None  # Matched Damodaran industry name
    yf_industry: str = None  # Original yfinance industry
    yf_sector: str = None  # Original yfinance sector
    is_exact_industry_match: bool = False  # True if direct industry mapping found
    # Terminal multiple - current-anchored (no blending)
    current_ev_ebitda: float = None  # Current company EV/EBITDA (THE DEFAULT)
    industry_ev_ebitda: float = None  # Damodaran industry EV/EBITDA (reference only)
    terminal_multiple_source: str = None  # "current" or "industry_fallback" or "gordon_growth"
    # Scenario analysis (for DCF Details view only - NOT used in main calculation)
    scenario_market_anchored: float = None  # = current EV/EBITDA (default)
    scenario_mean_reversion: float = None  # = move partway to industry
    scenario_conservative: float = None  # = lower of current * 0.8 or industry P25
    # Dual Terminal Value cross-check (always compute BOTH methods)
    tv_exit_multiple: float = None  # Terminal Value via Exit Multiple method
    tv_gordon_growth: float = None  # Terminal Value via Gordon Growth method
    pv_tv_exit_multiple: float = None  # PV of Exit Multiple TV
    pv_tv_gordon_growth: float = None  # PV of Gordon Growth TV
    price_exit_multiple: float = None  # Price per share using Exit Multiple TV
    price_gordon_growth: float = None  # Price per share using Gordon Growth TV
    # Implied terminal FCF yield (for sanity checking exit multiple)
    implied_ev_fcf: float = None  # Implied EV/FCF at terminal
    implied_fcf_yield: float = None  # 1 / (EV/FCF) as percentage
    # FCFF metrics (proper enterprise DCF)
    ttm_fcff: float = None  # TTM Free Cash Flow to Firm
    fcff_method: str = None  # "proper_fcff", "approx_unlevered", "levered_proxy"
    fcff_reliability: int = None  # 0-100 score for FCFF calculation quality
    fcff_ebitda_ratio: float = None  # FCFF / EBITDA (cash conversion)
    implied_gordon_ev_ebitda: float = None  # What Gordon implies for EV/EBITDA
    required_fcff_ebitda_for_exit: float = None  # Required cash conversion for exit multiple
    projected_terminal_ebitda: float = None  # Projected Year N EBITDA used by exit-multiple cross-check
    projected_terminal_ebitda_method: str = None  # How Year N EBITDA was derived
    exit_multiple_unavailable_reasons: List[str] = field(default_factory=list)  # Exact missing inputs for exit cross-check
    # Terminal Multiple Scenario (explicit selection)
    terminal_multiple_scenario: str = "current"  # "current", "industry", "blended", "custom"
    terminal_multiple_rerating_pct: float = None  # Implied rerating vs current (e.g., +15%)
    observed_fcff_ebitda_ttm: float = None  # Observed TTM cash conversion
    observed_fcff_ebitda_year5: float = None  # Projected Year 5 cash conversion
    # Confidence flags
    tv_dominance_pct: float = None  # PV(TV) / EV - flag if >80%
    growth_proxy_warning: bool = False  # True if using FCF growth as EBITDA proxy
    data_quality_score: float = None  # Overall data quality
    wacc_is_estimated: bool = True  # True if WACC was auto-calculated
    
    # ========== TEXTBOOK DCF: DRIVER-BASED PROJECTION ==========
    # Revenue drivers
    base_revenue: float = None  # TTM revenue (Year 0 base)
    revenue_growth_rates: List[float] = field(default_factory=list)  # Per-year growth rates (fade schedule)
    near_term_growth_rate: float = None  # Years 1-3 (from analyst estimates or recent trend)
    near_term_growth_source: str = None  # "analyst_cagr", "historical_revenue", "yahoo_trailing"
    stable_growth_rate: float = 0.03  # Terminal perpetual growth (3.0% default)
    
    # Margin drivers
    base_ebit_margin: float = None  # TTM EBIT/Revenue
    ebit_margins: List[float] = field(default_factory=list)  # Per-year EBIT margin (can fade)
    stable_ebit_margin: float = None  # Terminal EBIT margin assumption
    da_to_revenue_ratio: float = None  # D&A as % of revenue (for EBITDA projection)
    
    # ========== TERMINAL YEAR METRICS (for consistent cross-check) ==========
    # These are the Year N outputs that BOTH Gordon AND Exit Multiple must use
    terminal_year_fcff: float = None  # Year N FCFF from driver projection
    terminal_year_ebitda: float = None  # Year N EBITDA from driver projection  
    terminal_year_fcff_ebitda: float = None  # Year N FCFF/EBITDA (terminal cash conversion)
    consistent_exit_multiple: float = None  # = (FCFF/EBITDA)_terminal / (WACC - g)
    
    # ========== EXIT MULTIPLE SCENARIO CROSS-CHECK ==========
    # Three-multiple cross-check with gap diagnostic (not averaged into DCF)
    exit_multiple_scenarios: List[Dict] = field(default_factory=list)  # List of scenario dicts
    # Each scenario: {name, multiple, required_fcff_ebitda, gap, status, price, interpretation}
    # Status: "PASS" (plausible), "FAIL" (implausible/impossible), "WARN" (elevated)
    
    # ========== ROIC-BASED TERMINAL REINVESTMENT ==========
    # Reinvestment Rate_terminal = g_perp / ROIC_terminal
    # FCFF_terminal = NOPAT_terminal × (1 - g_perp / ROIC_terminal)
    sales_to_capital_ratio: float = None  # Revenue / Invested Capital (efficiency metric)
    reinvestment_rates: List[float] = field(default_factory=list)  # Per-year reinvestment rate
    
    # Current ROIC
    base_invested_capital: float = None  # Starting invested capital
    base_roic: float = None  # TTM ROIC = NOPAT / Invested Capital
    
    # Terminal ROIC (fade from current toward industry median)
    industry_roic: float = None  # Industry median ROIC (for fade target)
    terminal_roic: float = None  # Blended ROIC at terminal (current faded toward industry)
    terminal_reinvestment_rate: float = None  # = g_perp / terminal_ROIC
    
    # Derived per-year projections (populated during calculation)
    yearly_projections: List[Dict] = field(default_factory=list)  # Full driver table

    # Analyst FCF anchor metadata
    analyst_fcf_anchors_used: bool = False  # True if analyst revenue estimates anchored Years 1-3
    fcf_sources: List[str] = field(default_factory=list)  # Per-year source: "analyst_revenue_estimate" or "driver_model"

    # Driver-based mode flag
    use_driver_model: bool = True  # True = textbook, False = legacy simple growth
    
    def to_dict(self):
        return asdict(self)


class CalculationTraceStep:
    """Single step in the DCF calculation trace."""
    def __init__(self, name: str, formula: str = None, inputs=None, output=None, 
                 output_units: str = None, notes: str = None):
        self.name = name
        self.formula = formula
        self.inputs = inputs or {}
        self.output = output
        self.output_units = output_units
        self.notes = notes or ""
        self.timestamp = datetime.utcnow().isoformat()
    
    def to_dict(self):
        return {
            "name": self.name,
            "formula": self.formula,
            "inputs": self.inputs,
            "output": self.output,
            "output_units": self.output_units,
            "notes": self.notes,
            "timestamp": self.timestamp
        }


class TerminalValueStrategy(ABC):
    """Abstract base for terminal value calculation strategies."""
    
    @abstractmethod
    def calculate(self, final_year_fcf: float, ttm_ebitda: float, assumptions: DCFAssumptions,
                  trace: List[CalculationTraceStep]) -> Tuple[float, float]:
        """
        Calculate terminal value and return (terminal_value_yearN, pv_terminal_value).
        
        Adds calculation steps to trace automatically.
        """
        pass


class GordonGrowthTerminalValue(TerminalValueStrategy):
    """Terminal Value = FCF_{N+1} / (WACC - g)."""
    
    def calculate(self, final_year_fcf: float, ttm_ebitda: float, assumptions: DCFAssumptions,
                  trace: List[CalculationTraceStep]) -> Tuple[float, float]:
        """
        Gordon Growth Method:
        - Terminal FCF = Year N FCF × (1 + terminal_growth_rate)
        - Terminal Value = Terminal FCF / (WACC - g)
        """
        g = assumptions.terminal_growth_rate
        wacc = assumptions.wacc
        
        # Validation
        if wacc <= g:
            raise ValueError(f"WACC ({wacc:.1%}) must be > terminal growth ({g:.1%})")
        
        # Calculate terminal FCF
        terminal_fcf = final_year_fcf * (1 + g)
        trace.append(CalculationTraceStep(
            name="Terminal FCF Calculation (Gordon Growth)",
            formula="Final Year FCF × (1 + g)",
            inputs={
                "final_year_fcf": final_year_fcf,
                "terminal_growth_rate": g
            },
            output=terminal_fcf,
            output_units="USD"
        ))
        
        # Calculate terminal value
        terminal_value = terminal_fcf / (wacc - g)
        trace.append(CalculationTraceStep(
            name="Terminal Value (Gordon Growth)",
            formula="Terminal FCF / (WACC - g)",
            inputs={
                "terminal_fcf": terminal_fcf,
                "wacc": wacc,
                "growth_rate": g
            },
            output=terminal_value,
            output_units="USD",
            notes=f"Assumes perpetual {g:.1%} growth"
        ))
        
        # Discount back to present
        discount_factor = (1 + wacc) ** assumptions.forecast_years
        pv_terminal_value = terminal_value / discount_factor
        trace.append(CalculationTraceStep(
            name="PV of Terminal Value",
            formula=f"Terminal Value / (1 + WACC)^{assumptions.forecast_years}",
            inputs={
                "terminal_value": terminal_value,
                "wacc": wacc,
                "forecast_years": assumptions.forecast_years,
                "discount_factor": discount_factor
            },
            output=pv_terminal_value,
            output_units="USD"
        ))
        
        return terminal_value, pv_terminal_value


class ExitMultipleTerminalValue(TerminalValueStrategy):
    """Terminal Value = EBITDA_{N} × exit_multiple."""
    
    def calculate(self, final_year_fcf: float, ttm_ebitda: float, assumptions: DCFAssumptions,
                  trace: List[CalculationTraceStep], snapshot=None) -> Tuple[float, float]:
        """
        Exit Multiple Method:
        - Project EBITDA to Year N using revenue growth × stable EBITDA margin
        - Terminal Value = Year N EBITDA × exit_multiple
        - Discount back to present
        """
        if ttm_ebitda is None or ttm_ebitda <= 0:
            raise ValueError("Exit Multiple method requires positive TTM EBITDA; use Gordon Growth fallback")
        
        if assumptions.exit_multiple is None or assumptions.exit_multiple <= 0:
            raise ValueError("Exit multiple must be set (e.g., 15 for 15x EV/EBITDA)")
        
        # Prefer the model's projected terminal EBITDA (Year N) when available.
        # This keeps exit-multiple cross-check tied to explicit forecast outputs.
        if assumptions.projected_terminal_ebitda and assumptions.projected_terminal_ebitda > 0:
            year_n_ebitda = assumptions.projected_terminal_ebitda
            trace.append(CalculationTraceStep(
                name="Projected Terminal EBITDA (for Exit Cross-Check)",
                formula=f"From Year {assumptions.forecast_years} projection",
                inputs={
                    "forecast_years": assumptions.forecast_years,
                    "projection_method": assumptions.projected_terminal_ebitda_method or "model_projection"
                },
                output=year_n_ebitda,
                output_units="USD",
                notes="Using projected terminal EBITDA from explicit forecast horizon"
            ))
        else:
            # Backward-compatible fallback if projection metadata is unavailable
            ttm_revenue = snapshot.ttm_revenue.value if snapshot and hasattr(snapshot, 'ttm_revenue') and snapshot.ttm_revenue else None

            if ttm_revenue and ttm_revenue > 0 and ttm_ebitda > 0:
                ebitda_margin = ttm_ebitda / ttm_revenue
                revenue_growth = assumptions.fcf_growth_rate or 0.05
                year_n_revenue = ttm_revenue * ((1 + revenue_growth) ** assumptions.forecast_years)
                year_n_ebitda = year_n_revenue * ebitda_margin

                trace.append(CalculationTraceStep(
                    name="Project Year N EBITDA",
                    formula=f"Year {assumptions.forecast_years} Revenue × EBITDA Margin",
                    inputs={
                        "ttm_revenue": ttm_revenue,
                        "ttm_ebitda": ttm_ebitda,
                        "ebitda_margin": f"{ebitda_margin:.1%}",
                        "revenue_growth": revenue_growth,
                        "year_n_revenue": year_n_revenue,
                        "forecast_years": assumptions.forecast_years
                    },
                    output=year_n_ebitda,
                    output_units="USD",
                    notes=f"Fallback projection: revenue grows at {revenue_growth:.1%}/yr; EBITDA margin held at {ebitda_margin:.1%}"
                ))
            else:
                ebitda_growth = assumptions.fcf_growth_rate or 0.05
                year_n_ebitda = ttm_ebitda * ((1 + ebitda_growth) ** assumptions.forecast_years)

                trace.append(CalculationTraceStep(
                    name="Project Year N EBITDA",
                    formula=f"TTM EBITDA × (1 + growth_rate)^{assumptions.forecast_years}",
                    inputs={
                        "ttm_ebitda": ttm_ebitda,
                        "growth_rate": ebitda_growth,
                        "forecast_years": assumptions.forecast_years
                    },
                    output=year_n_ebitda,
                    output_units="USD",
                    notes=f"Fallback projection: EBITDA grows at {ebitda_growth:.1%} (revenue data unavailable)"
                ))
        
        # Terminal value = Year N EBITDA × exit multiple
        terminal_value = year_n_ebitda * assumptions.exit_multiple
        trace.append(CalculationTraceStep(
            name="Terminal Value (Exit Multiple)",
            formula=f"Year {assumptions.forecast_years} EBITDA × exit_multiple",
            inputs={
                "year_n_ebitda": year_n_ebitda,
                "exit_multiple": assumptions.exit_multiple
            },
            output=terminal_value,
            output_units="USD",
            notes=f"Exit multiple = {assumptions.exit_multiple}x EV/EBITDA"
        ))
        
        # Discount back to present
        wacc = assumptions.wacc
        discount_factor = (1 + wacc) ** assumptions.forecast_years
        pv_terminal_value = terminal_value / discount_factor
        trace.append(CalculationTraceStep(
            name="PV of Terminal Value",
            formula=f"Terminal Value / (1 + WACC)^{assumptions.forecast_years}",
            inputs={
                "terminal_value": terminal_value,
                "wacc": wacc,
                "forecast_years": assumptions.forecast_years,
                "discount_factor": discount_factor
            },
            output=pv_terminal_value,
            output_units="USD"
        ))
        
        return terminal_value, pv_terminal_value


class NetDebtCalculator:
    """Computes net debt for EV→Equity bridge."""
    
    @staticmethod
    def calculate(snapshot: NormalizedFinancialSnapshot, trace: List[CalculationTraceStep]) -> Tuple[float, Dict]:
        """
        Net Debt = Total Debt - Cash & Equivalents + Minority Interest - Preferred Stock
        
        Returns: (net_debt_value, details_dict)
        """
        total_debt = snapshot.total_debt.value or 0
        cash = snapshot.cash_and_equivalents.value or 0
        
        net_debt = total_debt - cash
        
        details = {
            "total_debt": total_debt,
            "cash_and_equivalents": cash,
            "net_debt": net_debt,
            "total_debt_source": snapshot.total_debt.source_path,
            "cash_source": snapshot.cash_and_equivalents.source_path,
            "total_debt_reliability": snapshot.total_debt.reliability_score,
            "cash_reliability": snapshot.cash_and_equivalents.reliability_score
        }
        
        trace.append(CalculationTraceStep(
            name="Net Debt Calculation",
            formula="Total Debt - Cash & Equivalents",
            inputs={
                "total_debt": total_debt,
                "cash_and_equivalents": cash
            },
            output=net_debt,
            output_units="USD",
            notes="Used for EV→Equity bridge"
        ))
        
        return net_debt, details


class DCFEngine:
    """Main DCF valuation engine."""
    
    def __init__(self, snapshot: NormalizedFinancialSnapshot, assumptions: DCFAssumptions = None):
        self.snapshot = snapshot
        self.assumptions = assumptions or DCFAssumptions()
        self.trace = []
        self.warnings = []
        self.errors = []
    
    def validate_inputs(self) -> bool:
        """Check that we have minimum data to run DCF."""
        checks = []
        
        if self.snapshot.ttm_revenue.value is None or self.snapshot.ttm_revenue.value <= 0:
            self.errors.append("TTM Revenue is required")
            checks.append(False)
        
        if self.snapshot.ttm_fcf.value is None or self.snapshot.ttm_fcf.value <= 0:
            self.errors.append("TTM FCF is required")
            checks.append(False)
        
        if self.snapshot.shares_outstanding.value is None:
            self.errors.append("Shares outstanding required for per-share value")
            checks.append(False)
        
        # Warnings for low-quality data
        if self.snapshot.ttm_fcf.reliability_score < 70:
            self.warnings.append(f"TTM FCF reliability low ({self.snapshot.ttm_fcf.reliability_score}/100): {self.snapshot.ttm_fcf.fallback_reason}")
        
        if self.snapshot.overall_quality_score < 60:
            self.warnings.append(f"Overall data quality low ({self.snapshot.overall_quality_score:.0f}/100)")
        
        return all(checks) if checks else True
    
    def set_assumptions_from_defaults(self):
        """Auto-fill missing assumptions from snapshot."""
        
        # ===== HORIZON RULE: 10-year for large caps =====
        # Large-cap threshold: market cap > $200B OR revenue > $50B → 10-year forecast
        market_cap = self.snapshot.market_cap.value or 0
        ttm_revenue = self.snapshot.ttm_revenue.value or 0
        
        if market_cap > 200e9:
            self.assumptions.forecast_years = 10
            self.assumptions.display_years = 5
            self.assumptions.is_large_cap = True
            self.assumptions.horizon_reason = "large_cap_market_cap"
            self.trace.append(CalculationTraceStep(
                name="Horizon Rule: 10-Year Forecast",
                formula="Market Cap > $200B → 10-year explicit forecast",
                inputs={
                    "market_cap": f"${market_cap/1e9:.0f}B",
                    "threshold": "$200B"
                },
                output=10,
                output_units="years",
                notes="Large-cap company: using 10-year forecast (showing first 5, computing all 10)"
            ))
        elif ttm_revenue > 50e9:
            self.assumptions.forecast_years = 10
            self.assumptions.display_years = 5
            self.assumptions.is_large_cap = True
            self.assumptions.horizon_reason = "large_cap_revenue"
            self.trace.append(CalculationTraceStep(
                name="Horizon Rule: 10-Year Forecast",
                formula="Revenue > $50B → 10-year explicit forecast",
                inputs={
                    "ttm_revenue": f"${ttm_revenue/1e9:.0f}B",
                    "threshold": "$50B"
                },
                output=10,
                output_units="years",
                notes="Large-cap company: using 10-year forecast (showing first 5, computing all 10)"
            ))
        else:
            self.assumptions.forecast_years = 5
            self.assumptions.display_years = 5
            self.assumptions.is_large_cap = False
            self.assumptions.horizon_reason = "standard"
        
        # WACC: use size-based defaults if not provided
        if self.assumptions.wacc is None:
            ttm_rev = self.snapshot.ttm_revenue.value
            if ttm_rev is None:
                self.assumptions.wacc = 0.09
            elif ttm_rev > 50e9:
                self.assumptions.wacc = 0.08
            elif ttm_rev > 10e9:
                self.assumptions.wacc = 0.095
            else:
                self.assumptions.wacc = 0.11
            self.trace.append(CalculationTraceStep(
                name="WACC Auto-Assignment",
                formula="Size-based default",
                inputs={"ttm_revenue_b": ttm_rev / 1e9 if ttm_rev else None},
                output=self.assumptions.wacc,
                output_units="rate",
                notes="Auto-set based on company size (no override provided)"
            ))
        
        # Tax Rate
        if self.assumptions.tax_rate is None:
            self.assumptions.tax_rate = self.snapshot.effective_tax_rate.value or 0.25
            self.trace.append(CalculationTraceStep(
                name="Tax Rate Assignment",
                formula="From snapshot.effective_tax_rate or default 25%",
                inputs={},
                output=self.assumptions.tax_rate,
                output_units="rate"
            ))
        
        # FCF Growth Rate
        if self.assumptions.fcf_growth_rate is None:
            # Estimate from historical revenue growth (if available)
            if self.snapshot.quarterly_history and len(self.snapshot.quarterly_history) >= 8:
                revenues = [q["revenue"] for q in self.snapshot.quarterly_history[:8] if q["revenue"]]
                if len(revenues) >= 2:
                    growth = (revenues[0] / revenues[-1]) ** (1.0 / (len(revenues) - 1)) - 1
                    self.assumptions.fcf_growth_rate = max(0.03, min(growth, 0.25))
                else:
                    self.assumptions.fcf_growth_rate = 0.08
            else:
                self.assumptions.fcf_growth_rate = 0.08
            self.trace.append(CalculationTraceStep(
                name="FCF Growth Rate Assignment",
                formula="Estimated from historical revenue growth or default 8%",
                inputs={},
                output=self.assumptions.fcf_growth_rate,
                output_units="rate"
            ))
        
        # ===== TEXTBOOK DCF: GORDON GROWTH IS PRIMARY =====
        # Per Damodaran: Terminal Value = FCFF_{N+1} / (WACC - g_perpetual)
        # Exit Multiple is a cross-check, not primary method.
        # Gordon Growth requires: g_perpetual < WACC (strictly)
        
        ttm_fcf = self.snapshot.ttm_fcf.value
        ttm_ebitda = self.snapshot.ttm_ebitda.value
        wacc = self.assumptions.wacc
        terminal_g = self.assumptions.terminal_growth_rate
        
        # Validate Gordon Growth is mathematically valid
        gordon_is_valid = wacc and terminal_g and wacc > terminal_g
        
        # Safety constraint: g_perpetual should be ≤ nominal GDP growth (~2-3%)
        if terminal_g and terminal_g > 0.04:  # >4% is aggressive
            self.warnings.append(
                f"⚠️ Terminal growth ({terminal_g:.1%}) exceeds typical long-run nominal GDP (3-4%). "
                f"Consider using 2-3% for mature firms in developed markets."
            )
        
        # CRITICAL CHECK: g < WACC (hard constraint for Gordon Growth)
        if terminal_g and wacc and terminal_g >= wacc:
            self.errors.append(
                f"🔴 INVALID: Terminal growth ({terminal_g:.1%}) ≥ WACC ({wacc:.1%}). "
                f"Gordon Growth requires g < WACC. Please lower terminal growth or raise WACC."
            )
            # Force fail - this is non-negotiable
            return
        
        # TEXTBOOK DEFAULT: Use Gordon Growth as primary
        # Only use exit multiple if explicitly requested AND EBITDA available
        if self.assumptions.terminal_value_method == "exit_multiple":
            # User explicitly requested exit multiple
            if not (ttm_ebitda and ttm_ebitda > 0):
                self.assumptions.terminal_value_method = "gordon_growth"
                self.warnings.append("Exit Multiple requested but EBITDA unavailable; using Gordon Growth (textbook primary).")
            else:
                self.warnings.append(
                    "Using Exit Multiple per user request. Note: Textbook DCF uses Gordon Growth as primary; "
                    "Exit Multiple is a cross-check method."
                )
        else:
            # DEFAULT: Gordon Growth (textbook standard)
            if gordon_is_valid:
                self.assumptions.terminal_value_method = "gordon_growth"
                self.trace.append(CalculationTraceStep(
                    name="TV Method: Gordon Growth (Textbook Primary)",
                    formula="TV_N = FCFF_{N+1} / (WACC - g_perpetual)",
                    inputs={
                        "wacc": f"{wacc:.1%}",
                        "terminal_growth": f"{terminal_g:.1%}",
                        "wacc_minus_g": f"{(wacc - terminal_g):.1%}"
                    },
                    output="gordon_growth",
                    output_units="",
                    notes="Textbook DCF: Gordon Growth is primary. Exit Multiple used as cross-check."
                ))
            elif ttm_ebitda and ttm_ebitda > 0:
                self.assumptions.terminal_value_method = "exit_multiple"
                reason = "gordon_invalid_fallback_to_exit_multiple"
                self.trace.append(CalculationTraceStep(
                    name="TV Method Auto-Selection",
                    formula=f"Exit Multiple selected ({reason})",
                    inputs={
                        "ttm_fcf": ttm_fcf,
                        "wacc": f"{wacc:.1%}" if wacc is not None else None,
                        "terminal_growth": f"{terminal_g:.1%}" if terminal_g is not None else None,
                        "gordon_is_valid": gordon_is_valid
                    },
                    output="exit_multiple",
                    output_units="",
                    notes=f"FCF unstable or distorted; using Exit Multiple with Gordon cross-check."
                ))
            else:
                # Neither method works well
                self.assumptions.terminal_value_method = "gordon_growth"
                self.warnings.append("Neither TV method ideal (no EBITDA, FCF may be volatile). Using Gordon Growth.")
        
        # ===== ALWAYS COMPUTE CURRENT EV/EBITDA FOR CROSS-CHECK (REGARDLESS OF PRIMARY METHOD) =====
        # Store yfinance classification for reference
        self.assumptions.yf_industry = self.snapshot.industry
        self.assumptions.yf_sector = self.snapshot.sector
        
        # Compute Current Company EV/EBITDA
        current_ev_ebitda = None
        market_cap = self.snapshot.market_cap.value
        total_debt = self.snapshot.total_debt.value or 0
        cash = self.snapshot.cash_and_equivalents.value or 0
        ttm_ebitda = self.snapshot.ttm_ebitda.value
        
        if market_cap and ttm_ebitda and ttm_ebitda > 0:
            # EV = Market Cap + Total Debt - Cash
            current_ev = market_cap + total_debt - cash
            current_ev_ebitda = current_ev / ttm_ebitda
            self.assumptions.current_ev_ebitda = round(current_ev_ebitda, 2)
        
        # Set fallback exit_multiple from current EV/EBITDA (for cross-check) if not already set
        if self.assumptions.exit_multiple is None and current_ev_ebitda is not None:
            self.assumptions.exit_multiple = round(current_ev_ebitda, 2)
            self.assumptions.terminal_multiple_source = "current_for_crosscheck"
        
        # ===== ALWAYS GET INDUSTRY MULTIPLE FOR CROSS-CHECK =====
        # This is needed regardless of terminal value method because exit multiples
        # are cross-checks, not inputs. We need industry multiple for the 3-scenario table.
        industry_multiple, damodaran_industry, is_exact_match = get_industry_multiple(
            self.snapshot.industry, 
            self.snapshot.sector
        )
        
        self.assumptions.damodaran_industry = damodaran_industry
        self.assumptions.is_exact_industry_match = is_exact_match
        
        if industry_multiple is not None:
            self.assumptions.industry_ev_ebitda = industry_multiple
            match_type = "exact" if is_exact_match else "approximate"
            
            self.trace.append(CalculationTraceStep(
                name="Industry EV/EBITDA (Damodaran)",
                formula=f"From Damodaran data ({DAMODARAN_DATA_DATE})",
                inputs={
                    "yf_industry": self.snapshot.industry,
                    "yf_sector": self.snapshot.sector,
                    "damodaran_industry": damodaran_industry,
                    "match_type": match_type
                },
                output=industry_multiple,
                output_units="multiple",
                notes=f"{match_type.title()} match to '{damodaran_industry}'. Source: {DAMODARAN_SOURCE_URL}"
            ))
        
        # Exit Multiple scenario selection (only when Exit Multiple is primary method)
        if self.assumptions.terminal_value_method == "exit_multiple":
            if market_cap and ttm_ebitda and ttm_ebitda > 0:
                # EV = Market Cap + Total Debt - Cash (already computed above)
                self.trace.append(CalculationTraceStep(
                    name="Current Company EV/EBITDA",
                    formula="(Market Cap + Debt - Cash) / TTM EBITDA",
                    inputs={
                        "market_cap": market_cap,
                        "total_debt": total_debt,
                        "cash": cash,
                        "current_ev": market_cap + total_debt - cash,
                        "ttm_ebitda": ttm_ebitda
                    },
                    output=round(current_ev_ebitda, 2) if current_ev_ebitda else None,
                    output_units="multiple",
                    notes=f"Current trading multiple: {current_ev_ebitda:.2f}x EV/EBITDA" if current_ev_ebitda else "N/A"
                ))
            else:
                self.warnings.append(
                    f"Cannot compute current EV/EBITDA: Market Cap={market_cap is not None}, "
                    f"TTM EBITDA={(ttm_ebitda or 0) > 0}. Will use industry multiple only."
                )
            
            # Industry multiple already fetched above - check if available
            if industry_multiple is None:
                # Industry is NA (e.g., banks) - fall back to Gordon Growth
                self.assumptions.terminal_value_method = "gordon_growth"
                self.assumptions.industry_multiple_source = "fallback_gordon"
                
                self.warnings.append(
                    f"EV/EBITDA multiple not applicable for industry '{self.snapshot.industry}' "
                    f"(Damodaran: {damodaran_industry or 'not found'}). Using Gordon Growth terminal value."
                )
                self.trace.append(CalculationTraceStep(
                    name="Exit Multiple Not Available",
                    formula="Fallback to Gordon Growth Model",
                    inputs={
                        "yf_industry": self.snapshot.industry,
                        "yf_sector": self.snapshot.sector,
                        "damodaran_industry": damodaran_industry,
                        "reason": "EV/EBITDA not applicable for this industry (e.g., banks use P/B)"
                    },
                    output=None,
                    output_units="multiple",
                    notes=f"Industry multiple N/A. Using Gordon Growth with {self.assumptions.terminal_growth_rate:.1%} perpetual growth."
                ))
                return  # Exit early - Gordon Growth will be used
            
            # Step 3: Apply terminal multiple based on scenario selection
            # Scenarios: "current", "industry", "blended", "custom"
            scenario = self.assumptions.terminal_multiple_scenario or "current"
            
            # Calculate blended multiple (70% current, 30% industry)
            blended_multiple = None
            if current_ev_ebitda is not None and industry_multiple is not None:
                blended_multiple = round(current_ev_ebitda * 0.7 + industry_multiple * 0.3, 2)
            
            # Store all scenario options for UI display
            self.assumptions.scenario_market_anchored = round(current_ev_ebitda, 2) if current_ev_ebitda else None
            self.assumptions.scenario_mean_reversion = blended_multiple
            if industry_multiple and current_ev_ebitda:
                if industry_multiple < current_ev_ebitda:
                    self.assumptions.scenario_conservative = round((current_ev_ebitda + industry_multiple) / 2, 2)
                else:
                    self.assumptions.scenario_conservative = round(current_ev_ebitda * 0.8, 2)
            
            # Determine terminal multiple based on scenario
            if scenario == "current":
                if current_ev_ebitda is not None:
                    terminal_multiple = round(current_ev_ebitda, 2)
                    self.assumptions.terminal_multiple_source = "current"
                elif industry_multiple is not None:
                    terminal_multiple = industry_multiple
                    self.assumptions.terminal_multiple_source = "industry_fallback"
                    self.warnings.append(
                        f"Current EV/EBITDA unavailable; using industry multiple ({industry_multiple}x) as fallback."
                    )
                else:
                    self.assumptions.terminal_value_method = "gordon_growth"
                    return
                    
            elif scenario == "industry":
                if industry_multiple is not None:
                    terminal_multiple = industry_multiple
                    self.assumptions.terminal_multiple_source = "industry"
                elif current_ev_ebitda is not None:
                    terminal_multiple = round(current_ev_ebitda, 2)
                    self.assumptions.terminal_multiple_source = "current_fallback"
                    self.warnings.append(
                        f"Industry multiple unavailable; using current EV/EBITDA ({terminal_multiple}x) as fallback."
                    )
                else:
                    self.assumptions.terminal_value_method = "gordon_growth"
                    return
                    
            elif scenario == "blended":
                if blended_multiple is not None:
                    terminal_multiple = blended_multiple
                    self.assumptions.terminal_multiple_source = "blended"
                elif current_ev_ebitda is not None:
                    terminal_multiple = round(current_ev_ebitda, 2)
                    self.assumptions.terminal_multiple_source = "current_fallback"
                    self.warnings.append("Industry multiple unavailable for blending; using current EV/EBITDA.")
                elif industry_multiple is not None:
                    terminal_multiple = industry_multiple
                    self.assumptions.terminal_multiple_source = "industry_fallback"
                    self.warnings.append("Current EV/EBITDA unavailable for blending; using industry multiple.")
                else:
                    self.assumptions.terminal_value_method = "gordon_growth"
                    return
                    
            elif scenario == "custom":
                # Custom multiple should already be set in assumptions.exit_multiple
                if self.assumptions.exit_multiple and self.assumptions.exit_multiple > 0:
                    terminal_multiple = self.assumptions.exit_multiple
                    self.assumptions.terminal_multiple_source = "custom"
                elif current_ev_ebitda is not None:
                    terminal_multiple = round(current_ev_ebitda, 2)
                    self.assumptions.terminal_multiple_source = "current_fallback"
                    self.warnings.append("Custom multiple not set; using current EV/EBITDA.")
                else:
                    self.assumptions.terminal_value_method = "gordon_growth"
                    return
            else:
                # Unknown scenario - default to current
                terminal_multiple = round(current_ev_ebitda, 2) if current_ev_ebitda else industry_multiple
                self.assumptions.terminal_multiple_source = "current"
            
            self.assumptions.exit_multiple = terminal_multiple
            
            # Calculate implied rerating vs current
            if current_ev_ebitda and current_ev_ebitda > 0:
                rerating_pct = ((terminal_multiple / current_ev_ebitda) - 1) * 100
                self.assumptions.terminal_multiple_rerating_pct = round(rerating_pct, 1)
            else:
                self.assumptions.terminal_multiple_rerating_pct = None
            
            # Calculate industry deviation for reference
            industry_deviation_pct = None
            if industry_multiple and current_ev_ebitda and current_ev_ebitda > 0:
                industry_deviation_pct = ((industry_multiple / current_ev_ebitda) - 1) * 100
            
            # Add trace for terminal multiple scenario selection
            self.trace.append(CalculationTraceStep(
                name="Terminal Multiple Scenario",
                formula=f"Scenario: {scenario.upper()}",
                inputs={
                    "scenario_selected": scenario,
                    "current_ev_ebitda": f"{current_ev_ebitda:.2f}x" if current_ev_ebitda else "N/A",
                    "industry_ev_ebitda": f"{industry_multiple:.2f}x" if industry_multiple else "N/A",
                    "blended_multiple": f"{blended_multiple:.2f}x" if blended_multiple else "N/A",
                    "custom_multiple": f"{self.assumptions.exit_multiple:.2f}x" if scenario == "custom" else "N/A",
                    "industry_vs_current_pct": f"{industry_deviation_pct:+.1f}%" if industry_deviation_pct else "N/A"
                },
                output=terminal_multiple,
                output_units="multiple",
                notes=f"Using {terminal_multiple}x ({scenario}). " + 
                      (f"Rerating vs current: {self.assumptions.terminal_multiple_rerating_pct:+.1f}%." if self.assumptions.terminal_multiple_rerating_pct else "")
            ))
    
    def run(self) -> Dict:
        """Execute DCF valuation and return results + trace."""
        
        # Validate
        if not self.validate_inputs():
            return {
                "success": False,
                "errors": self.errors,
                "warnings": self.warnings,
                "trace": [s.to_dict() for s in self.trace]
            }
        
        # Auto-fill assumptions
        self.set_assumptions_from_defaults()
        
        try:
            # Project 5-year FCF
            fcf_projections = self._project_fcf()
            pv_fcf_sum = sum([p["pv"] for p in fcf_projections])
            
            # Calculate BOTH terminal value methods for cross-check
            terminal_value, pv_terminal_value, dual_tv = self._calculate_dual_terminal_values(
                fcf_projections[-1]["fcf"]
            )
            
            # Enterprise Value (using primary TV method)
            enterprise_value = pv_fcf_sum + pv_terminal_value
            
            # Compute TV dominance
            tv_dominance = (pv_terminal_value / enterprise_value * 100) if enterprise_value > 0 else 0
            self.assumptions.tv_dominance_pct = round(tv_dominance, 1)
            
            self.trace.append(CalculationTraceStep(
                name="Enterprise Value",
                formula="Sum(PV of Explicit FCF) + PV(Terminal Value)",
                inputs={
                    "pv_fcf_sum": pv_fcf_sum,
                    "pv_terminal_value": pv_terminal_value,
                    "tv_dominance_pct": f"{tv_dominance:.1f}%"
                },
                output=enterprise_value,
                output_units="USD",
                notes=f"TV dominance: {tv_dominance:.1f}% of EV" + (" ⚠️ HIGH" if tv_dominance > 80 else "")
            ))
            
            # Equity Value = EV - Net Debt
            net_debt, debt_details = NetDebtCalculator.calculate(self.snapshot, self.trace)
            equity_value = enterprise_value - net_debt
            self.trace.append(CalculationTraceStep(
                name="Equity Value",
                formula="Enterprise Value - Net Debt",
                inputs={
                    "enterprise_value": enterprise_value,
                    "net_debt": net_debt
                },
                output=equity_value,
                output_units="USD"
            ))
            
            # Per-share value (primary method)
            shares = self.snapshot.shares_outstanding.value or self.snapshot.latest_annual_diluted_shares.value
            price_per_share = None
            if shares and shares > 0:
                price_per_share = equity_value / shares
                self.trace.append(CalculationTraceStep(
                    name="Price Per Share",
                    formula="Equity Value / Shares Outstanding",
                    inputs={
                        "equity_value": equity_value,
                        "shares_outstanding": shares
                    },
                    output=price_per_share,
                    output_units="USD",
                    notes=f"Using {shares/1e6:.0f}M shares"
                ))
                
                # Compute cross-check prices from both TV methods
                if dual_tv.get('pv_tv_exit_multiple') is not None:
                    ev_exit = pv_fcf_sum + dual_tv['pv_tv_exit_multiple']
                    equity_exit = ev_exit - net_debt
                    self.assumptions.price_exit_multiple = round(equity_exit / shares, 2)
                
                if dual_tv.get('pv_tv_gordon_growth') is not None:
                    ev_gordon = pv_fcf_sum + dual_tv['pv_tv_gordon_growth']
                    equity_gordon = ev_gordon - net_debt
                    self.assumptions.price_gordon_growth = round(equity_gordon / shares, 2)
                
                # ===== FINALIZE SCENARIO PRICES =====
                # Now we have pv_fcf_sum, compute price per share for each exit multiple scenario
                self._finalize_scenario_prices(pv_fcf_sum, net_debt, shares)
            
            # Store data quality score
            self.assumptions.data_quality_score = self.snapshot.overall_quality_score
            
            # Run sanity checks
            sanity_checks = self._run_sanity_checks(
                enterprise_value, equity_value, pv_fcf_sum, pv_terminal_value
            )
            
            return {
                "success": True,
                "enterprise_value": enterprise_value,
                "equity_value": equity_value,
                "net_debt": net_debt,
                "net_debt_details": debt_details,
                "price_per_share": price_per_share,
                "shares_outstanding": shares,
                "pv_fcf_sum": pv_fcf_sum,
                "pv_terminal_value": pv_terminal_value,
                "terminal_value_yearN": terminal_value,
                "fcf_projections": fcf_projections,
                "sanity_checks": sanity_checks,
                "assumptions": self.assumptions.to_dict(),
                "data_quality_score": self.snapshot.overall_quality_score,
                "errors": self.errors,
                "warnings": self.warnings,
                "trace": [s.to_dict() for s in self.trace]
            }
        
        except Exception as e:
            self.errors.append(f"DCF calculation failed: {str(e)}")
            return {
                "success": False,
                "errors": self.errors,
                "warnings": self.warnings,
                "trace": [s.to_dict() for s in self.trace]
            }
    
    def _project_fcf(self) -> List[Dict]:
        """
        Project 5-year FCFF using proper enterprise DCF methodology.
        
        PRIMARY: FCFF = EBIT × (1 - tax) + D&A - CapEx - ΔNWC
        
        FALLBACK (if EBIT/D&A unavailable):
        FCFF_proxy = CFO + AfterTaxInterest - CapEx
        (CFO already includes ΔNWC, so we don't subtract again)
        
        If interest expense unavailable, mark as "levered proxy" with warning.
        """
        # Get base financial metrics
        ttm_ebit = self.snapshot.ttm_operating_income.value  # Operating Income = EBIT
        ttm_ebitda = self.snapshot.ttm_ebitda.value
        ttm_revenue = self.snapshot.ttm_revenue.value
        ttm_capex = self.snapshot.ttm_capex.value
        ttm_da = self.snapshot.ttm_depreciation_amortization.value
        ttm_cfo = self.snapshot.ttm_operating_cash_flow.value
        tax_rate = self.assumptions.tax_rate or 0.25
        
        # Get TTM interest expense for FCFF proxy unlevering
        ttm_interest_expense = None
        if hasattr(self.snapshot, 'ttm_interest_expense') and self.snapshot.ttm_interest_expense:
            ttm_interest_expense = self.snapshot.ttm_interest_expense.value
        
        # Get TTM ΔNWC from cash flow statement (CRITICAL: use TTM, not quarterly delta)
        # Note: For primary FCFF we need this. For fallback, CFO already includes ΔNWC.
        ttm_delta_nwc = None
        if hasattr(self.snapshot, 'ttm_delta_nwc') and self.snapshot.ttm_delta_nwc and self.snapshot.ttm_delta_nwc.value is not None:
            ttm_delta_nwc = self.snapshot.ttm_delta_nwc.value
        
        # If D&A not available, estimate from EBITDA - EBIT
        if ttm_da is None and ttm_ebitda and ttm_ebit:
            ttm_da = ttm_ebitda - ttm_ebit
        
        # Track cash flow methodology for labeling
        fcff_method = None  # "proper_fcff", "approx_unlevered", "levered_proxy"
        ttm_fcff = None
        fcff_reliability = 100  # Will be reduced for fallback methods
        
        # PRIMARY METHOD: FCFF = EBIT × (1 - tax) + D&A - CapEx - ΔNWC
        if ttm_ebit and ttm_da is not None and ttm_capex:
            nopat = ttm_ebit * (1 - tax_rate)
            # Use TTM ΔNWC from cash flow; default to 0 if unavailable (with warning)
            delta_nwc_use = ttm_delta_nwc if ttm_delta_nwc is not None else 0
            delta_nwc_missing = ttm_delta_nwc is None
            
            ttm_fcff = nopat + ttm_da - ttm_capex - delta_nwc_use
            fcff_method = "proper_fcff"
            fcff_reliability = 95 if not delta_nwc_missing else 80
            
            # ΔNWC sign convention note
            # Positive ΔNWC = increase in working capital = cash USE = reduces FCFF
            # The CF statement reports it as a reduction from net income, so a negative value means cash release
            delta_nwc_note = ""
            if delta_nwc_use > 0:
                delta_nwc_note = f"ΔNWC=${delta_nwc_use/1e9:.2f}B (cash use, reduces FCFF)"
            elif delta_nwc_use < 0:
                delta_nwc_note = f"ΔNWC=${delta_nwc_use/1e9:.2f}B (cash release, adds to FCFF)"
            else:
                delta_nwc_note = "ΔNWC=$0B"
                if delta_nwc_missing:
                    delta_nwc_note += " ⚠️ MISSING - defaulted to 0"
            
            self.trace.append(CalculationTraceStep(
                name="TTM FCFF Calculation (Primary)",
                formula="FCFF = EBIT × (1 - tax) + D&A - CapEx - ΔNWC",
                inputs={
                    "ttm_ebit": f"${ttm_ebit/1e9:.2f}B",
                    "tax_rate": f"{tax_rate:.1%}",
                    "nopat": f"${nopat/1e9:.2f}B",
                    "ttm_da": f"${ttm_da/1e9:.2f}B",
                    "ttm_capex": f"${ttm_capex/1e9:.2f}B",
                    "ttm_delta_nwc": delta_nwc_note
                },
                output=ttm_fcff,
                output_units="USD",
                notes=f"Proper FCFF for enterprise valuation (WACC discounting). Sign: +ΔNWC = cash use = reduces FCFF."
            ))
            
            if delta_nwc_missing:
                self.warnings.append(
                    "⚠️ ΔNWC data unavailable; defaulted to $0. This may overstate FCFF for "
                    "working-capital-intensive businesses or understate for those releasing WC."
                )
            
        # FALLBACK 1: Approximate Unlevering = CFO + AfterTaxInterestExpense - CapEx
        # NOTE: This uses interest EXPENSE (income statement), not interest PAID (cash flow)
        # Interest paid is not available in yfinance, so this is an approximation
        elif ttm_cfo and ttm_capex and ttm_interest_expense:
            after_tax_interest = ttm_interest_expense * (1 - tax_rate)
            ttm_fcff = ttm_cfo + after_tax_interest - ttm_capex
            fcff_method = "approx_unlevered"  # NOT "unlevered_proxy" - we don't have interest paid
            fcff_reliability = 70  # Lower reliability due to interest timing mismatch
            
            self.trace.append(CalculationTraceStep(
                name="TTM FCFF Calculation (Approximate Unlevering ⚠️)",
                formula="FCFF_approx = CFO + AfterTax(InterestExpense) - CapEx",
                inputs={
                    "ttm_cfo": f"${ttm_cfo/1e9:.2f}B",
                    "interest_expense": f"${ttm_interest_expense/1e9:.2f}B (income stmt, NOT cash paid)",
                    "after_tax_interest": f"${after_tax_interest/1e9:.2f}B",
                    "tax_rate": f"{tax_rate:.1%}",
                    "ttm_capex": f"${ttm_capex/1e9:.2f}B"
                },
                output=ttm_fcff,
                output_units="USD",
                notes="⚠️ APPROXIMATE: Uses interest EXPENSE not interest PAID. May double-count or misstate adjustment."
            ))
            self.warnings.append(
                "📊 Using APPROXIMATE unlevered FCFF (CFO + AfterTaxInterestExpense - CapEx). "
                "Interest PAID not available; using interest EXPENSE from income statement. "
                "This may cause ~5% error due to timing/accrual differences."
            )
            
        # FALLBACK 2: Levered Proxy = CFO - CapEx (WARNING: should use cost of equity, not WACC)
        elif ttm_cfo and ttm_capex:
            ttm_fcff = ttm_cfo - ttm_capex
            fcff_method = "levered_proxy"
            fcff_reliability = 50  # Low reliability - framework mixing
            
            self.trace.append(CalculationTraceStep(
                name="TTM FCF Calculation (LEVERED PROXY ⚠️)",
                formula="FCF_levered = CFO - CapEx",
                inputs={
                    "ttm_cfo": f"${ttm_cfo/1e9:.2f}B",
                    "ttm_capex": f"${ttm_capex/1e9:.2f}B"
                },
                output=ttm_fcff,
                output_units="USD",
                notes="⚠️ LEVERED proxy: CFO includes interest paid. Theoretically requires cost of equity, not WACC."
            ))
            self.warnings.append(
                "🔴 Using LEVERED FCF proxy (CFO - CapEx). Neither EBIT/D&A nor interest expense available. "
                "This is cash flow to EQUITY discounted with WACC = framework mixing. "
                "Results may be 10-20% off. Consider this a rough estimate only."
            )
        else:
            # No valid FCF calculation possible
            self.errors.append("Cannot calculate FCFF: insufficient data (need EBIT+D&A or CFO)")
            return []
        
        # Store FCFF metrics for terminal value calculations
        self.assumptions.ttm_fcff = ttm_fcff
        self.assumptions.fcff_reliability = fcff_reliability
        self.assumptions.fcff_method = fcff_method
        self.assumptions.fcff_ebitda_ratio = ttm_fcff / ttm_ebitda if ttm_ebitda and ttm_fcff else None
        
        # Calculate key ratios for terminal state
        if ttm_ebitda and ttm_fcff:
            fcff_ebitda_ratio = ttm_fcff / ttm_ebitda
            self.trace.append(CalculationTraceStep(
                name="Cash Conversion Analysis",
                formula="FCFF / EBITDA",
                inputs={
                    "ttm_fcff": ttm_fcff,
                    "ttm_ebitda": ttm_ebitda
                },
                output=fcff_ebitda_ratio,
                output_units="ratio",
                notes=f"Current cash conversion: {fcff_ebitda_ratio:.1%}"
            ))
        
        # ===== DRIVER-BASED PROJECTION (TEXTBOOK DCF) =====
        # Instead of: FCFF_t = FCFF_{t-1} × (1 + g)
        # Use: Revenue → EBIT → NOPAT → Reinvestment → FCFF
        
        if self.assumptions.use_driver_model:
            return self._project_fcf_with_drivers(ttm_fcff, ttm_ebit, ttm_da, ttm_capex, tax_rate)
        
        # ===== LEGACY: Simple growth projection =====
        # Project FCFF forward
        growth = self.assumptions.fcf_growth_rate
        wacc = self.assumptions.wacc
        
        projections = []
        current_fcff = ttm_fcff
        
        self.trace.append(CalculationTraceStep(
            name="5-Year FCFF Projection",
            formula="Year 1..5: FCFF_t = FCFF_{t-1} × (1 + growth); PV = FCFF_t / (1+WACC)^t",
            inputs={
                "ttm_fcff": ttm_fcff,
                "growth_rate": growth,
                "wacc": wacc
            },
            output="",
            output_units="USD",
            notes="Enterprise DCF: FCFF discounted at WACC"
        ))
        
        for year in range(1, self.assumptions.forecast_years + 1):
            current_fcff = current_fcff * (1 + growth)
            discount_factor = (1 + wacc) ** year
            pv = current_fcff / discount_factor
            
            projections.append({
                "year": year,
                "fcf": current_fcff,  # Keep key as 'fcf' for backward compatibility
                "fcff": current_fcff,
                "discount_factor": discount_factor,
                "pv": pv
            })
        
        return projections
    
    def _project_fcf_with_drivers(self, ttm_fcff: float, ttm_ebit: float, ttm_da: float, 
                                   ttm_capex: float, tax_rate: float) -> List[Dict]:
        """
        TEXTBOOK DCF: Driver-based FCFF projection with proper horizon and growth fade.
        
        FCFF_t = NOPAT_t - Reinvestment_t
        
        Where:
          - Revenue_t = Revenue_{t-1} × (1 + g_t)   [g fades from near-term to stable]
          - EBIT_t = Revenue_t × EBIT_margin_t      [margin can also fade]
          - NOPAT_t = EBIT_t × (1 - tax_rate)
          - Reinvestment_t = ΔRevenue_t / Sales_to_Capital_ratio
          - FCFF_t = NOPAT_t - Reinvestment_t
          
        Growth Fade Schedule (per textbook):
          - Years 1-3: Near-term growth (analyst CAGR if available, else historical proxy)
          - Years 4-N: Linear fade toward g_perp (terminal growth)
          - g_perp: 2.5-3.0% for developed markets, must be < WACC
          
        Terminal Reinvestment (ROIC-based):
          - Reinvestment Rate_terminal = g_perp / ROIC_terminal
          - FCFF_terminal = NOPAT_terminal × (1 - g_perp / ROIC_terminal)
          - ROIC_terminal: Current ROIC faded toward industry median
        """
        wacc = self.assumptions.wacc
        n_years = self.assumptions.forecast_years  # 5 or 10 based on horizon rule
        
        # ===== STEP 1: Establish Base Year (Year 0) Drivers =====
        ttm_revenue = self.snapshot.ttm_revenue.value
        
        if not ttm_revenue or ttm_revenue <= 0:
            self.warnings.append("Cannot run driver-based projection: TTM Revenue unavailable")
            return self._project_fcf_legacy(ttm_fcff)
        
        # Base EBIT margin
        base_ebit_margin = ttm_ebit / ttm_revenue if ttm_ebit else 0.15  # Default 15%
        self.assumptions.base_ebit_margin = round(base_ebit_margin, 4)
        self.assumptions.base_revenue = ttm_revenue
        
        # ===== STEP 2: Calculate Current ROIC and Invested Capital =====
        total_debt = self.snapshot.total_debt.value or 0
        cash = self.snapshot.cash_and_equivalents.value or 0
        nwc = self.snapshot.net_working_capital.value or 0
        
        # Estimate invested capital (PPE + NWC + Debt - Cash or approximation)
        invested_capital = None
        if ttm_capex and ttm_da:
            # Net PPE estimate: CapEx × 5 (assuming ~20% depreciation rate)
            net_ppe_estimate = ttm_capex * 5
            invested_capital = net_ppe_estimate + nwc
        
        if invested_capital and invested_capital > 0:
            sales_to_capital = ttm_revenue / invested_capital
            # Calculate current ROIC = NOPAT / Invested Capital
            nopat_base = ttm_ebit * (1 - tax_rate) if ttm_ebit else 0
            current_roic = nopat_base / invested_capital if invested_capital > 0 else 0.15
        else:
            sales_to_capital = 2.0
            current_roic = 0.15  # Default 15%
            self.warnings.append("Invested capital unavailable; using default Sales-to-Capital (2.0) and ROIC (15%)")
        
        # Clamp to reasonable ranges
        sales_to_capital = max(0.5, min(sales_to_capital, 5.0))
        current_roic = max(0.05, min(current_roic, 0.50))  # 5% - 50%
        
        self.assumptions.sales_to_capital_ratio = round(sales_to_capital, 2)
        self.assumptions.base_roic = round(current_roic, 4)
        self.assumptions.base_invested_capital = invested_capital
        
        # ===== STEP 3: Determine Terminal ROIC (fade current toward industry) =====
        # Industry median ROIC: use sector-based estimate or default
        # Tech/Software: ~20-30%, Industrial: ~10-15%, Retail: ~12-18%
        sector = self.snapshot.sector or ""
        if "Technology" in sector or "Software" in sector:
            industry_roic = 0.20
        elif "Financial" in sector:
            industry_roic = 0.12
        elif "Consumer" in sector or "Retail" in sector:
            industry_roic = 0.15
        elif "Healthcare" in sector:
            industry_roic = 0.18
        else:
            industry_roic = 0.12  # Default to cost of capital level
        
        self.assumptions.industry_roic = round(industry_roic, 4)
        
        # Terminal ROIC: blend current toward industry (70% fade for 10-year, 50% for 5-year)
        if n_years >= 10:
            fade_factor = 0.7  # 70% toward industry for 10-year
        else:
            fade_factor = 0.5  # 50% toward industry for 5-year
        
        terminal_roic = current_roic * (1 - fade_factor) + industry_roic * fade_factor
        terminal_roic = max(wacc, terminal_roic)  # ROIC should not be below WACC at terminal
        self.assumptions.terminal_roic = round(terminal_roic, 4)
        
        # ===== STEP 4: Build Growth Fade Schedule =====
        near_term_g = self.assumptions.fcf_growth_rate or 0.10  # From analyst/historical
        stable_g = self.assumptions.terminal_growth_rate  # 2.5-3.0% default
        
        # Validate g_perp < WACC
        if stable_g >= wacc:
            old_g = stable_g
            stable_g = wacc * 0.3  # Default to 30% of WACC if invalid
            self.assumptions.terminal_growth_rate = stable_g
            self.warnings.append(
                f"Terminal growth ({old_g:.1%}) >= WACC ({wacc:.1%}); adjusted to {stable_g:.1%}"
            )
        
        self.assumptions.near_term_growth_rate = near_term_g
        self.assumptions.stable_growth_rate = stable_g
        self.assumptions.near_term_growth_source = "analyst_cagr" if hasattr(self.snapshot, 'analyst_growth') else "yahoo_trailing"
        
        # Growth fade: smooth linear fade from near-term to g_perp across all years
        # (Avoids cliff at Year 4 when analyst anchors are unavailable)
        growth_rates = []
        for year in range(1, n_years + 1):
            fade_progress = (year - 1) / max(n_years - 1, 1)
            g = near_term_g + fade_progress * (stable_g - near_term_g)
            growth_rates.append(round(g, 4))
        
        self.assumptions.revenue_growth_rates = growth_rates
        
        # ===== STEP 5: Build Reinvestment Rate Schedule (ROIC-based) =====
        # Terminal reinvestment rate = g_perp / ROIC_terminal
        terminal_reinv_rate = stable_g / terminal_roic if terminal_roic > 0 else 0.25
        self.assumptions.terminal_reinvestment_rate = round(terminal_reinv_rate, 4)
        
        # Fade reinvestment rate from current to terminal
        # Current reinvestment rate implied from sales-to-capital
        # reinv_rate = g / (ROIC) where ROIC ≈ margin × sales_to_capital
        current_reinv_rate = near_term_g / current_roic if current_roic > 0 else 0.5
        current_reinv_rate = max(0, min(current_reinv_rate, 0.95))  # Cap at 95%
        
        reinv_rates = []
        for year in range(1, n_years + 1):
            fade_progress = (year - 1) / max(n_years - 1, 1)
            rr = current_reinv_rate + fade_progress * (terminal_reinv_rate - current_reinv_rate)
            reinv_rates.append(round(rr, 4))
        
        self.assumptions.reinvestment_rates = reinv_rates
        
        # ===== STEP 6: Margin fade (stable or slight improvement for mature) =====
        stable_margin = self.assumptions.stable_ebit_margin or base_ebit_margin
        self.assumptions.stable_ebit_margin = stable_margin
        
        ebit_margins = [base_ebit_margin] * n_years  # No margin fade by default
        self.assumptions.ebit_margins = ebit_margins
        
        # ===== STEP 6b: D&A Ratio (for EBITDA projection) =====
        # D&A as % of revenue - used to project EBITDA = EBIT + D&A
        # ttm_da is passed from _project_fcf(), which may have derived it from EBITDA - EBIT
        if ttm_da is not None and ttm_revenue > 0:
            da_to_revenue_ratio = ttm_da / ttm_revenue
        else:
            # Fallback: try to get from snapshot directly
            ttm_ebitda_snap = self.snapshot.ttm_ebitda.value if self.snapshot.ttm_ebitda else None
            if ttm_ebitda_snap and ttm_ebit and ttm_revenue > 0:
                da_to_revenue_ratio = (ttm_ebitda_snap - ttm_ebit) / ttm_revenue
            else:
                da_to_revenue_ratio = 0.05  # Default 5% if unavailable
            
        self.assumptions.da_to_revenue_ratio = round(da_to_revenue_ratio, 4)
        
        # ===== STEP 7: Project Year-by-Year with Full Driver Detail =====
        projections = []
        yearly_details = []
        prev_revenue = ttm_revenue
        
        horizon_note = f"{n_years}-year forecast" if n_years == 10 else "5-year forecast"
        self.trace.append(CalculationTraceStep(
            name=f"Driver-Based FCFF Projection ({horizon_note})",
            formula="FCFF_t = NOPAT_t × (1 - Reinvestment_Rate_t)",
            inputs={
                "base_revenue": f"${ttm_revenue/1e9:.2f}B",
                "base_ebit_margin": f"{base_ebit_margin:.1%}",
                "current_roic": f"{current_roic:.1%}",
                "terminal_roic": f"{terminal_roic:.1%} (faded {fade_factor:.0%} toward industry {industry_roic:.1%})",
                "near_term_growth": f"{near_term_g:.1%} (Years 1-3)",
                "terminal_growth": f"{stable_g:.1%} (g_perp)",
                "terminal_reinv_rate": f"{terminal_reinv_rate:.1%} (= g_perp / ROIC_terminal)"
            },
            output="",
            output_units="",
            notes=f"Textbook DCF: {horizon_note}, growth fades Y4-{n_years}, ROIC-based terminal reinvestment"
        ))
        
        # --- Build analyst FCF anchors (Years 1-3) ---
        # Anchor near-term FCF to analyst revenue consensus × TTM FCF margin to avoid
        # the reinvestment-rate cliff that occurs when the driver model holds reinv_rate
        # constant for Years 1-3 then abruptly fades it from Year 4 onward.
        analyst_fcf_anchors = {}   # {year_index (1-based): fcf_value}
        analyst_estimates = getattr(self.snapshot, 'analyst_revenue_estimates', [])
        ttm_fcff_margin = ttm_fcff / ttm_revenue if ttm_revenue > 0 else None

        if analyst_estimates and ttm_fcff_margin is not None and ttm_fcff_margin > 0:
            raw_revs = [e['revenue'] for e in analyst_estimates if e.get('revenue')]
            for i, rev in enumerate(raw_revs[:2]):   # Year 1, Year 2
                analyst_fcf_anchors[i + 1] = rev * ttm_fcff_margin
            # Year 3: extrapolate from Year 1→2 trend (or compound Year 2 at near-term g)
            if len(raw_revs) >= 2 and raw_revs[0] > 0:
                yr2_growth = raw_revs[1] / raw_revs[0] - 1
                analyst_fcf_anchors[3] = raw_revs[1] * (1 + yr2_growth) * ttm_fcff_margin
            elif len(raw_revs) == 1:
                analyst_fcf_anchors[2] = raw_revs[0] * (1 + near_term_g) * ttm_fcff_margin
                analyst_fcf_anchors[3] = analyst_fcf_anchors[2] * (1 + near_term_g)

        self.assumptions.analyst_fcf_anchors_used = bool(analyst_fcf_anchors)
        self.assumptions.fcf_sources = []  # Populated per-year below

        for year in range(1, n_years + 1):
            g = growth_rates[year - 1]
            margin = ebit_margins[year - 1]
            reinv_rate = reinv_rates[year - 1]

            if year in analyst_fcf_anchors:
                # ── Analyst anchor path: FCF = analyst_revenue × TTM FCF margin ──
                fcff = analyst_fcf_anchors[year]
                fcf_source = "analyst_revenue_estimate"
                # Back-derive implied revenue for downstream fields (EBITDA, etc.)
                revenue = fcff / ttm_fcff_margin if ttm_fcff_margin else prev_revenue * (1 + g)
                delta_revenue = revenue - prev_revenue
                ebit = revenue * margin
                da = revenue * da_to_revenue_ratio
                ebitda = ebit + da
                nopat = ebit * (1 - tax_rate)
                # Implied reinvestment = NOPAT - FCFF (may differ from reinv_rate formula)
                reinvestment = max(0, nopat - fcff)
                implied_roic = g / reinv_rate if reinv_rate > 0 else None
            else:
                # ── Driver model path: FCFF = NOPAT × (1 - reinv_rate) ──
                fcf_source = "driver_model"
                revenue = prev_revenue * (1 + g)
                delta_revenue = revenue - prev_revenue
                ebit = revenue * margin
                da = revenue * da_to_revenue_ratio
                ebitda = ebit + da
                nopat = ebit * (1 - tax_rate)
                reinvestment = nopat * reinv_rate
                fcff = nopat - reinvestment
                implied_roic = g / reinv_rate if reinv_rate > 0 else None

            self.assumptions.fcf_sources.append(fcf_source)

            # FCFF/EBITDA ratio for this year (cash conversion)
            fcff_ebitda_year = fcff / ebitda if ebitda > 0 else None

            # Present value
            discount_factor = (1 + wacc) ** year
            pv = fcff / discount_factor

            # Store detailed projection
            year_detail = {
                "year": year,
                "revenue": revenue,
                "revenue_growth": g,
                "ebit_margin": margin,
                "ebit": ebit,
                "da": da,
                "ebitda": ebitda,
                "nopat": nopat,
                "delta_revenue": delta_revenue,
                "reinvestment": reinvestment,
                "reinvestment_rate": reinv_rate,
                "fcff": fcff,
                "fcff_ebitda": fcff_ebitda_year,
                "discount_factor": discount_factor,
                "pv_fcff": pv,
                "implied_roic": implied_roic,
                "fcf_source": fcf_source,
            }
            yearly_details.append(year_detail)

            # Standard projection format for compatibility
            projections.append({
                "year": year,
                "fcf": fcff,  # Backward compatibility
                "fcff": fcff,
                "discount_factor": discount_factor,
                "pv": pv,
                # Extended driver info
                "revenue": revenue,
                "revenue_growth": g,
                "ebit_margin": margin,
                "ebit": ebit,
                "da": da,
                "ebitda": ebitda,
                "nopat": nopat,
                "reinvestment": reinvestment,
                "reinvestment_rate": reinv_rate,
                "fcff_ebitda": fcff_ebitda_year,
                "fcf_source": fcf_source,
            })

            prev_revenue = revenue
        
        # Store detailed yearly projections in assumptions for UI
        self.assumptions.yearly_projections = yearly_details
        
        # ===== TERMINAL YEAR METRICS (critical for exit multiple consistency) =====
        year_n = yearly_details[-1]
        terminal_fcff = year_n['fcff']
        terminal_ebitda = year_n['ebitda']
        terminal_fcff_ebitda = terminal_fcff / terminal_ebitda if terminal_ebitda > 0 else None
        
        # Store terminal metrics in assumptions for cross-check
        self.assumptions.terminal_year_fcff = terminal_fcff
        self.assumptions.terminal_year_ebitda = terminal_ebitda
        self.assumptions.terminal_year_fcff_ebitda = round(terminal_fcff_ebitda, 4) if terminal_fcff_ebitda else None
        
        # Set projected_terminal_ebitda from SAME forecast (not separate calculation!)
        self.assumptions.projected_terminal_ebitda = terminal_ebitda
        self.assumptions.projected_terminal_ebitda_method = "driver_based_projection"
        
        # ===== DERIVE CONSISTENT EXIT MULTIPLE =====
        # Consistent Multiple = (FCFF/EBITDA)_terminal / (WACC - g)
        wacc_minus_g = wacc - stable_g
        if terminal_fcff_ebitda and wacc_minus_g > 0:
            consistent_exit_multiple = terminal_fcff_ebitda / wacc_minus_g
            self.assumptions.consistent_exit_multiple = round(consistent_exit_multiple, 2)
            
            # If no exit_multiple set, use the consistent one derived from economics
            if self.assumptions.exit_multiple is None:
                self.assumptions.exit_multiple = round(consistent_exit_multiple, 1)
                self.assumptions.terminal_multiple_source = "economics_derived"
        else:
            self.assumptions.consistent_exit_multiple = None
        
        # Add trace for Year N terminal state
        year_label = str(n_years)
        self.trace.append(CalculationTraceStep(
            name=f"Year {n_years} Terminal State",
            formula="Basis for Terminal Value calculation (SAME outputs for Gordon AND Exit Multiple)",
            inputs={
                f"revenue_y{year_label}": f"${year_n['revenue']/1e9:.2f}B",
                f"revenue_growth_y{year_label}": f"{year_n['revenue_growth']:.1%}",
                f"ebit_y{year_label}": f"${year_n['ebit']/1e9:.2f}B",
                f"ebit_margin_y{year_label}": f"{year_n['ebit_margin']:.1%}",
                f"da_y{year_label}": f"${year_n['da']/1e9:.2f}B",
                f"ebitda_y{year_label}": f"${year_n['ebitda']/1e9:.2f}B",
                f"nopat_y{year_label}": f"${year_n['nopat']/1e9:.2f}B",
                f"reinvestment_y{year_label}": f"${year_n['reinvestment']/1e9:.2f}B",
                f"reinvestment_rate_y{year_label}": f"{year_n['reinvestment_rate']:.1%}",
                f"fcff_y{year_label}": f"${year_n['fcff']/1e9:.2f}B",
                f"fcff_ebitda_y{year_label}": f"{terminal_fcff_ebitda:.1%}" if terminal_fcff_ebitda else "N/A"
            },
            output=year_n['fcff'],
            output_units="USD",
            notes=f"Year {n_years}: FCFF=${year_n['fcff']/1e9:.2f}B, EBITDA=${terminal_ebitda/1e9:.2f}B, " +
                  f"Cash Conversion={terminal_fcff_ebitda:.1%}. " +
                  f"Consistent EV/EBITDA = {terminal_fcff_ebitda:.1%} / {wacc_minus_g:.1%} = {self.assumptions.consistent_exit_multiple:.1f}x" 
                  if terminal_fcff_ebitda and self.assumptions.consistent_exit_multiple else ""
        ))
        
        return projections
    
    def _project_fcf_legacy(self, ttm_fcff: float) -> List[Dict]:
        """Legacy simple growth method (fallback)."""
        growth = self.assumptions.fcf_growth_rate
        wacc = self.assumptions.wacc
        
        projections = []
        current_fcff = ttm_fcff
        
        for year in range(1, self.assumptions.forecast_years + 1):
            current_fcff = current_fcff * (1 + growth)
            discount_factor = (1 + wacc) ** year
            pv = current_fcff / discount_factor
            
            projections.append({
                "year": year,
                "fcf": current_fcff,
                "fcff": current_fcff,
                "discount_factor": discount_factor,
                "pv": pv
            })
        
        return projections
    
    def _calculate_dual_terminal_values(self, final_year_fcf: float) -> Tuple[float, float, Dict]:
        """
        Calculate BOTH terminal value methods and return primary + cross-check.
        
        CRITICAL: Both methods MUST use the SAME Year N outputs from the explicit forecast.
        No separate "TTM × (1+g)^N" calculations allowed - that would create two different companies.
        
        Returns: (primary_tv, primary_pv_tv, dual_tv_dict)
        
        The dual_tv_dict contains both methods for display as cross-check.
        """
        ttm_ebitda = self.snapshot.ttm_ebitda.value
        ttm_revenue = self.snapshot.ttm_revenue.value if self.snapshot.ttm_revenue else None
        dual_tv = {}
        
        # Pre-calculate WACC-g
        wacc_minus_g = self.assumptions.wacc - self.assumptions.terminal_growth_rate
        
        # TTM cash conversion (for reference only)
        fcff_ebitda_ratio_ttm = self.assumptions.fcff_ebitda_ratio
        self.assumptions.observed_fcff_ebitda_ttm = fcff_ebitda_ratio_ttm
        
        # ===== GET TERMINAL YEAR VALUES FROM DRIVER PROJECTION =====
        # These were computed in _project_fcf_with_drivers and MUST be used by both TV methods
        terminal_year_fcff = self.assumptions.terminal_year_fcff
        terminal_year_ebitda = self.assumptions.terminal_year_ebitda
        terminal_fcff_ebitda = self.assumptions.terminal_year_fcff_ebitda
        year_n = self.assumptions.forecast_years
        
        # Use terminal EBITDA from projection (NOT TTM × growth!)
        projected_terminal_ebitda = self.assumptions.projected_terminal_ebitda
        projected_method = self.assumptions.projected_terminal_ebitda_method
        
        # Fallback only if driver projection didn't run (legacy mode)
        if projected_terminal_ebitda is None or projected_terminal_ebitda <= 0:
            if (terminal_year_fcff and terminal_year_fcff > 0 and 
                fcff_ebitda_ratio_ttm and fcff_ebitda_ratio_ttm > 0):
                projected_terminal_ebitda = terminal_year_fcff / fcff_ebitda_ratio_ttm
                projected_method = "legacy_fallback"
            elif ttm_ebitda and ttm_ebitda > 0:
                # Last resort: simple growth (NOT preferred)
                ebitda_growth = self.assumptions.fcf_growth_rate or 0.05
                projected_terminal_ebitda = ttm_ebitda * ((1 + ebitda_growth) ** year_n)
                projected_method = "ttm_ebitda_growth_proxy_WARNING"
                self.warnings.append(
                    "⚠️ Using TTM EBITDA × growth^N fallback for exit multiple. "
                    "This creates inconsistency with driver-based FCFF projection."
                )
            
            if projected_terminal_ebitda:
                self.assumptions.projected_terminal_ebitda = projected_terminal_ebitda
                self.assumptions.projected_terminal_ebitda_method = projected_method
            self.assumptions.projected_terminal_ebitda_method = None
        
        # Calculate required cash conversion for exit multiple (diagnostic only - no clamping)
        required_conversion = None
        if wacc_minus_g > 0 and self.assumptions.exit_multiple:
            required_conversion = self.assumptions.exit_multiple * wacc_minus_g
            self.assumptions.required_fcff_ebitda_for_exit = round(required_conversion, 4)
            
            # WARNING: Required cash conversion exceeds TERMINAL forecast (not TTM!)
            # Use terminal FCFF/EBITDA from projection for proper comparison
            terminal_conversion = terminal_fcff_ebitda if terminal_fcff_ebitda else fcff_ebitda_ratio_ttm
            conversion_source = "Terminal" if terminal_fcff_ebitda else "TTM"
            
            if terminal_conversion and required_conversion > 0:
                gap_pp = (required_conversion - terminal_conversion) * 100  # percentage points
                if required_conversion > 0.85:
                    # >85% is economically implausible
                    self.warnings.append(
                        f"🔴 EXIT MULTIPLE REQUIRES IMPLAUSIBLE CASH CONVERSION: "
                        f"{self.assumptions.exit_multiple:.1f}x × (WACC-g = {wacc_minus_g:.1%}) = {required_conversion:.0%} FCFF/EBITDA required. "
                        f"{conversion_source} forecast: {terminal_conversion:.0%}. Gap: {gap_pp:+.0f}pp. "
                        f"Consider lowering terminal multiple or raising perpetual growth."
                    )
                elif gap_pp > 15:
                    # >15pp gap is concerning
                    self.warnings.append(
                        f"⚠️ EXIT MULTIPLE REQUIRES ELEVATED CASH CONVERSION: "
                        f"Required {required_conversion:.0%} vs {conversion_source} {terminal_conversion:.0%} ({gap_pp:+.0f}pp). "
                        f"At terminal, company must convert more EBITDA to free cash than forecast."
                    )
        
        # HIGH WACC-g WARNING: If WACC-g > 9-10%, Gordon implies very low multiples
        if wacc_minus_g > 0.095:
            self.warnings.append(
                f"⚠️ High WACC-g spread ({wacc_minus_g:.1%} > 9.5%): Gordon mechanically implies "
                f"very low multiples unless cash conversion is extremely high. "
                f"Your WACC ({self.assumptions.wacc:.1%}) may be too high for mega-cap quality, "
                f"or perpetual growth ({self.assumptions.terminal_growth_rate:.1%}) too low for nominal GDP."
            )
        
        # Calculate Gordon Growth TV (always possible if we have FCF)
        try:
            gordon_strategy = GordonGrowthTerminalValue()
            gordon_trace = []  # Separate trace to avoid cluttering main trace
            tv_gordon, pv_tv_gordon = gordon_strategy.calculate(
                final_year_fcf, ttm_ebitda, self.assumptions, gordon_trace
            )
            dual_tv['tv_gordon_growth'] = tv_gordon
            dual_tv['pv_tv_gordon_growth'] = pv_tv_gordon
            self.assumptions.tv_gordon_growth = round(tv_gordon, 0)
            self.assumptions.pv_tv_gordon_growth = round(pv_tv_gordon, 0)
        except Exception as e:
            self.warnings.append(f"Gordon Growth TV calculation failed: {e}")
            dual_tv['tv_gordon_growth'] = None
            dual_tv['pv_tv_gordon_growth'] = None
        
        # Calculate Exit Multiple TV cross-check when required inputs are available
        has_exit_multiple = self.assumptions.exit_multiple is not None and self.assumptions.exit_multiple > 0
        has_terminal_ebitda = (
            projected_terminal_ebitda is not None and projected_terminal_ebitda > 0
        ) or (terminal_year_ebitda is not None and terminal_year_ebitda > 0)
        exit_unavailable_reasons = []
        if not has_exit_multiple:
            exit_unavailable_reasons.append("missing terminal exit multiple")
        if not has_terminal_ebitda:
            exit_unavailable_reasons.append(f"missing projected terminal EBITDA (Year {year_n})")

        self.assumptions.exit_multiple_unavailable_reasons = exit_unavailable_reasons
        dual_tv['exit_unavailable_reasons'] = exit_unavailable_reasons

        if has_exit_multiple and has_terminal_ebitda:
            try:
                exit_strategy = ExitMultipleTerminalValue()
                exit_trace = []  # Separate trace
                tv_exit, pv_tv_exit = exit_strategy.calculate(
                    final_year_fcf, ttm_ebitda, self.assumptions, exit_trace, self.snapshot
                )
                dual_tv['tv_exit_multiple'] = tv_exit
                dual_tv['pv_tv_exit_multiple'] = pv_tv_exit
                self.assumptions.tv_exit_multiple = round(tv_exit, 0)
                self.assumptions.pv_tv_exit_multiple = round(pv_tv_exit, 0)
                
                # Calculate implied terminal FCF yield
                # implied EV/FCF = (EV/EBITDA) / (FCF/EBITDA) = (EV/EBITDA) * (EBITDA/FCF)
                ttm_fcf = self.snapshot.ttm_fcf.value if self.snapshot.ttm_fcf else None
                if ttm_fcf and ttm_fcf > 0 and ttm_ebitda > 0:
                    fcf_to_ebitda = ttm_fcf / ttm_ebitda
                    implied_ev_fcf = self.assumptions.exit_multiple / fcf_to_ebitda
                    implied_fcf_yield = 1 / implied_ev_fcf * 100  # As percentage
                    dual_tv['implied_ev_fcf'] = round(implied_ev_fcf, 1)
                    dual_tv['implied_fcf_yield'] = round(implied_fcf_yield, 2)
                    
                    # Store in assumptions for UI
                    self.assumptions.implied_ev_fcf = round(implied_ev_fcf, 1)
                    self.assumptions.implied_fcf_yield = round(implied_fcf_yield, 2)
                    
                    # Flag if yield is absurdly low vs WACC
                    if implied_fcf_yield < (self.assumptions.wacc * 100) * 0.4:  # Yield < 40% of WACC
                        self.warnings.append(
                            f"⚠️ Implied terminal FCF yield ({implied_fcf_yield:.1f}%) is very low vs WACC ({self.assumptions.wacc*100:.1f}%). "
                            f"This implies the market expects significant reinvestment or FCF compression at terminal."
                        )
                
            except Exception as e:
                self.warnings.append(f"Exit Multiple TV calculation failed: {e}")
                dual_tv['tv_exit_multiple'] = None
                dual_tv['pv_tv_exit_multiple'] = None
        else:
            dual_tv['tv_exit_multiple'] = None
            dual_tv['pv_tv_exit_multiple'] = None
        
        # Determine primary method and calculate with proper trace
        if self.assumptions.terminal_value_method == "exit_multiple":
            if ttm_ebitda and ttm_ebitda > 0 and self.assumptions.exit_multiple:
                strategy = ExitMultipleTerminalValue()
                primary_tv, primary_pv_tv = strategy.calculate(
                    final_year_fcf, ttm_ebitda, self.assumptions, self.trace, self.snapshot
                )
            else:
                # Fallback to Gordon
                self.warnings.append("EBITDA unavailable for Exit Multiple; falling back to Gordon Growth")
                self.assumptions.terminal_value_method = "gordon_growth"
                strategy = GordonGrowthTerminalValue()
                primary_tv, primary_pv_tv = strategy.calculate(
                    final_year_fcf, ttm_ebitda, self.assumptions, self.trace
                )
        else:
            strategy = GordonGrowthTerminalValue()
            primary_tv, primary_pv_tv = strategy.calculate(
                final_year_fcf, ttm_ebitda, self.assumptions, self.trace
            )
        
        # Add cross-check trace step
        if dual_tv.get('tv_exit_multiple') and dual_tv.get('tv_gordon_growth'):
            diff_pct = ((dual_tv['pv_tv_exit_multiple'] / dual_tv['pv_tv_gordon_growth']) - 1) * 100
            
            # Calculate implied EV/EBITDA from Gordon DIRECTLY: TV_gordon / EBITDA_N
            # This is the cleanest, unambiguous definition
            wacc_minus_g = self.assumptions.wacc - self.assumptions.terminal_growth_rate
            g = self.assumptions.terminal_growth_rate
            one_plus_g = 1 + g
            
            # Use terminal FCFF/EBITDA from projection (consistent with Gordon)
            terminal_fcff_ebitda_for_calc = terminal_fcff_ebitda if terminal_fcff_ebitda else fcff_ebitda_ratio_ttm
            conversion_label = "Terminal" if terminal_fcff_ebitda else "TTM"
            
            implied_gordon_ev_ebitda = None
            required_fcff_ebitda = None
            
            # DIRECT CALCULATION: implied_gordon_ev_ebitda = TV_gordon / EBITDA_N
            if terminal_year_ebitda and terminal_year_ebitda > 0 and dual_tv.get('tv_gordon_growth'):
                implied_gordon_ev_ebitda = dual_tv['tv_gordon_growth'] / terminal_year_ebitda
                self.assumptions.implied_gordon_ev_ebitda = round(implied_gordon_ev_ebitda, 2)
                dual_tv['implied_gordon_ev_ebitda'] = round(implied_gordon_ev_ebitda, 2)
            elif terminal_fcff_ebitda_for_calc and wacc_minus_g > 0:
                # Fallback to formula if TV not available: (FCFF/EBITDA) × (1+g) / (WACC-g)
                implied_gordon_ev_ebitda = terminal_fcff_ebitda_for_calc * one_plus_g / wacc_minus_g
                self.assumptions.implied_gordon_ev_ebitda = round(implied_gordon_ev_ebitda, 2)
                dual_tv['implied_gordon_ev_ebitda'] = round(implied_gordon_ev_ebitda, 2)
            
            # TIMING-CORRECT FORMULA: Required (FCFF_N/EBITDA_N) = M × (WACC − g) / (1 + g)
            # This accounts for Gordon using FCFF_{N+1} while exit multiple uses EBITDA_N
            if wacc_minus_g > 0 and self.assumptions.exit_multiple:
                required_fcff_ebitda = self.assumptions.exit_multiple * wacc_minus_g / one_plus_g
                self.assumptions.required_fcff_ebitda_for_exit = round(required_fcff_ebitda, 4)
                dual_tv['required_fcff_ebitda_for_exit'] = round(required_fcff_ebitda, 4)
            
            # Build scenario status for trace
            scenario_status = f"Scenario: {self.assumptions.terminal_multiple_scenario.upper()}"
            rerating_str = ""
            if self.assumptions.terminal_multiple_rerating_pct is not None:
                rerating_str = f" (Rerating: {self.assumptions.terminal_multiple_rerating_pct:+.1f}% vs current)"
            
            # Consistent exit multiple from terminal economics
            consistent_multiple = self.assumptions.consistent_exit_multiple
            
            self.trace.append(CalculationTraceStep(
                name="Terminal Value Cross-Check",
                formula="Compare Exit Multiple vs Gordon Growth (USING SAME Year N OUTPUTS)",
                inputs={
                    "pv_tv_exit_multiple": f"${dual_tv['pv_tv_exit_multiple']/1e9:.1f}B",
                    "pv_tv_gordon_growth": f"${dual_tv['pv_tv_gordon_growth']/1e9:.1f}B",
                    "difference_pct": f"{diff_pct:+.1f}%",
                    "exit_multiple_used": f"{self.assumptions.exit_multiple:.1f}x",
                    "consistent_exit_multiple": f"{consistent_multiple:.1f}x (= {conversion_label} FCFF/EBITDA / (WACC-g))" if consistent_multiple else "N/A",
                    "terminal_multiple_scenario": scenario_status + rerating_str,
                    "implied_gordon_ev_ebitda": f"{implied_gordon_ev_ebitda:.1f}x" if implied_gordon_ev_ebitda else "N/A",
                    f"fcff_ebitda_{conversion_label.lower()}": f"{terminal_fcff_ebitda_for_calc:.1%}" if terminal_fcff_ebitda_for_calc else "N/A",
                    "fcff_ebitda_ttm": f"{fcff_ebitda_ratio_ttm:.1%}" if fcff_ebitda_ratio_ttm else "N/A",
                    "required_fcff_ebitda_for_exit": f"{required_fcff_ebitda:.1%}" if required_fcff_ebitda else "N/A",
                    "wacc_minus_g": f"{wacc_minus_g:.1%}",
                    "primary_method": self.assumptions.terminal_value_method
                },
                output=None,
                output_units="",
                notes=f"Consistent Multiple from economics: {consistent_multiple:.1f}x. " if consistent_multiple else "" +
                      f"Exit Multiple ({self.assumptions.exit_multiple:.1f}x) vs Gordon implied ({implied_gordon_ev_ebitda:.1f}x). " if implied_gordon_ev_ebitda else "" +
                      f"Divergence of {diff_pct:+.1f}% indicates inconsistent terminal assumptions."
            ))
            
            # ECONOMICS-BASED WARNINGS (not just % difference)
            
            # Warning 1: Gordon is extremely sensitive when WACC-g < 4-5%
            if wacc_minus_g < 0.045:
                self.warnings.append(
                    f"🔴 Gordon Growth is extremely sensitive: WACC-g = {wacc_minus_g:.1%} (<4.5%). "
                    f"Small changes in growth or WACC will cause large swings in terminal value. "
                    f"Consider using exit multiple as primary method."
                )
            
            # Warning 2: Large divergence between methods - use ACTUAL terminal conversion
            if abs(diff_pct) > 40 and terminal_fcff_ebitda_for_calc:
                self.warnings.append(
                    f"⚠️ Terminal methods disagree by {abs(diff_pct):.0f}%: Exit Multiple implies {self.assumptions.exit_multiple:.1f}x EV/EBITDA, "
                    f"but Gordon/economics imply {consistent_multiple:.1f}x given WACC-g spread ({wacc_minus_g:.1%}) and {conversion_label} cash conversion ({terminal_fcff_ebitda_for_calc:.1%})."
                )
        
        # ===== EXIT MULTIPLE SCENARIO ANALYSIS =====
        # Compute all scenarios with plausibility gating (85% cash conversion threshold)
        self._compute_exit_multiple_scenarios(
            wacc_minus_g=wacc_minus_g,
            terminal_fcff_ebitda=terminal_fcff_ebitda if terminal_fcff_ebitda else fcff_ebitda_ratio_ttm,
            terminal_year_ebitda=terminal_year_ebitda,
            shares_outstanding=self.snapshot.shares_outstanding.value if self.snapshot.shares_outstanding else 1,
            net_debt=(self.snapshot.total_debt.value if self.snapshot.total_debt else 0) - 
                     (self.snapshot.cash_and_equivalents.value if self.snapshot.cash_and_equivalents else 0),
            pv_fcf_sum=None  # Will be set later in calculate()
        )
        
        return primary_tv, primary_pv_tv, dual_tv
    
    def _calculate_terminal_value(self, final_year_fcf: float) -> Tuple[float, float]:
        """Calculate terminal value using selected strategy (legacy method)."""
        ttm_ebitda = self.snapshot.ttm_ebitda.value
        
        # Choose strategy
        if self.assumptions.terminal_value_method == "gordon_growth":
            strategy = GordonGrowthTerminalValue()
            return strategy.calculate(final_year_fcf, ttm_ebitda, self.assumptions, self.trace)
        elif self.assumptions.terminal_value_method == "exit_multiple":
            # If EBITDA unavailable, fall back to Gordon Growth
            if ttm_ebitda is None or ttm_ebitda <= 0:
                self.warnings.append("EBITDA unavailable for Exit Multiple; falling back to Gordon Growth")
                self.assumptions.terminal_value_method = "gordon_growth"
                strategy = GordonGrowthTerminalValue()
                return strategy.calculate(final_year_fcf, ttm_ebitda, self.assumptions, self.trace)
            else:
                strategy = ExitMultipleTerminalValue()
                return strategy.calculate(final_year_fcf, ttm_ebitda, self.assumptions, self.trace, self.snapshot)
        else:
            strategy = GordonGrowthTerminalValue()
            return strategy.calculate(final_year_fcf, ttm_ebitda, self.assumptions, self.trace)
    
    def _run_sanity_checks(self, ev: float, equity_val: float, pv_fcf: float, pv_tv: float) -> Dict:
        """
        Run validation checks and return results + warnings.
        
        TEXTBOOK CONSISTENCY CHECKS (per Damodaran):
        1. g_perpetual < WACC (HARD FAIL if violated)
        2. Terminal ROIC vs WACC (flag if ROIC >> WACC forever implies abnormal returns)
        3. TV % of EV (flag if >85% - heavily dependent on terminal assumptions)
        4. Implied terminal multiples sanity
        """
        checks = {}
        g = self.assumptions.terminal_growth_rate
        wacc = self.assumptions.wacc
        
        # ===== 1. HARD CONSTRAINT: g < WACC =====
        if g >= wacc:
            self.errors.append(
                f"🔴 FATAL: Terminal growth ({g:.1%}) ≥ WACC ({wacc:.1%}). "
                f"This violates the Gordon Growth formula (TV = FCFF / (WACC - g)). "
                f"Reduce terminal growth or increase WACC."
            )
            checks["terminal_growth_valid"] = False
        else:
            checks["terminal_growth_valid"] = True
            wacc_g_spread = wacc - g
            if wacc_g_spread < 0.03:  # <3% spread is risky
                self.warnings.append(
                    f"⚠️ Low WACC-g spread ({wacc_g_spread:.1%}). Terminal value is extremely sensitive to small "
                    f"changes in either WACC or terminal growth. Consider widening the spread."
                )
        
        # ===== 2. TERMINAL ROIC vs WACC =====
        # If driver model is used, check if terminal ROIC >> WACC (implies supernormal returns forever)
        if self.assumptions.use_driver_model and self.assumptions.yearly_projections:
            y5 = self.assumptions.yearly_projections[-1]
            reinv_rate = y5.get('reinvestment_rate', 0)
            if reinv_rate > 0:
                implied_terminal_roic = g / reinv_rate
                checks["implied_terminal_roic"] = implied_terminal_roic
                
                if implied_terminal_roic > wacc * 2:  # ROIC > 2x WACC
                    self.warnings.append(
                        f"⚠️ Implied terminal ROIC ({implied_terminal_roic:.1%}) is very high vs WACC ({wacc:.1%}). "
                        f"This assumes the company earns supernormal returns indefinitely. "
                        f"For mature firms, ROIC should converge toward WACC over time."
                    )
                elif implied_terminal_roic > wacc * 1.5:
                    checks["terminal_roic_note"] = f"Terminal ROIC ({implied_terminal_roic:.1%}) exceeds WACC ({wacc:.1%}) - monitoring"
        
        # ===== 3. TERMINAL VALUE DOMINANCE (FLAG IF >85%) =====
        tv_ratio = pv_tv / ev if ev > 0 else 0
        checks["terminal_value_dominance"] = tv_ratio
        checks["tv_pct_of_ev"] = round(tv_ratio * 100, 1)
        
        if tv_ratio > 0.85:
            self.warnings.append(
                f"🔴 Terminal value dominates ({tv_ratio:.0%} of EV > 85%). "
                f"Valuation is extremely sensitive to terminal assumptions (growth, WACC). "
                f"Consider extending the explicit forecast period or validating terminal assumptions."
            )
        elif tv_ratio > 0.75:
            self.warnings.append(
                f"⚠️ Terminal value is high ({tv_ratio:.0%} of EV). "
                f"Results are sensitive to terminal growth and WACC assumptions."
            )
        
        # ===== 4. REINVESTMENT RATE SANITY =====
        if self.assumptions.use_driver_model and self.assumptions.yearly_projections:
            y5 = self.assumptions.yearly_projections[-1]
            reinv_rate = y5.get('reinvestment_rate', 0)
            
            # Stable reinvestment = g / ROIC. If ROIC ≈ WACC, then reinv ≈ g / WACC
            # For g=3%, WACC=8%, reinv ≈ 37.5%
            theoretical_stable_reinv = g / wacc if wacc > 0 else 0
            
            checks["year5_reinvestment_rate"] = reinv_rate
            checks["theoretical_stable_reinv"] = theoretical_stable_reinv
            
            if reinv_rate < 0:
                self.warnings.append(
                    f"⚠️ Negative Year 5 reinvestment rate ({reinv_rate:.1%}). "
                    f"This implies the company is shrinking its capital base. Verify growth assumptions."
                )
            elif reinv_rate > 0.8:
                self.warnings.append(
                    f"⚠️ High Year 5 reinvestment rate ({reinv_rate:.1%} > 80%). "
                    f"Most of NOPAT is being reinvested, leaving little FCFF. "
                    f"This is typical for high-growth firms but unusual at terminal."
                )
        
        # ===== 5. PV(FCF) REASONABILITY =====
        checks["pv_fcf_reasonable"] = pv_fcf <= ev
        if pv_fcf > ev:
            self.warnings.append("PV(FCF) exceeds EV - check for calculation errors or negative terminal value")
        
        # ===== 6. EV vs MARKET CAP =====
        if self.snapshot.market_cap.value and self.snapshot.market_cap.value > 0:
            market_cap = self.snapshot.market_cap.value
            total_debt = self.snapshot.total_debt.value or 0
            cash = self.snapshot.cash_and_equivalents.value or 0
            market_ev = market_cap + total_debt - cash
            
            ev_market_diff = (ev - market_ev) / market_ev if market_ev > 0 else 0
            checks["ev_vs_market_cap"] = {
                "dcf_ev_b": round(ev / 1e9, 1),
                "market_ev_b": round(market_ev / 1e9, 1),
                "diff_pct": round(ev_market_diff * 100, 1)
            }
            
            # Interpret the difference
            if ev_market_diff > 0.3:
                checks["valuation_signal"] = "Potentially UNDERVALUED by market"
            elif ev_market_diff < -0.3:
                checks["valuation_signal"] = "Potentially OVERVALUED by market"
            else:
                checks["valuation_signal"] = "Fairly valued within margin of error"
        
        # ===== 7. MULTIPLES SANITY CHECK =====
        ttm_revenue = self.snapshot.ttm_revenue.value
        ttm_ebitda = self.snapshot.ttm_ebitda.value
        
        if ttm_revenue and ttm_revenue > 0:
            ev_revenue = ev / ttm_revenue
            checks["ev_revenue_multiple"] = round(ev_revenue, 2)
            if ev_revenue < 0.5:
                self.warnings.append(f"EV/Revenue = {ev_revenue:.1f}x is very low; verify FCFF projections")
            elif ev_revenue > 20:
                self.warnings.append(f"EV/Revenue = {ev_revenue:.1f}x is very high; typical for hypergrowth only")
        
        if ttm_ebitda and ttm_ebitda > 0:
            ev_ebitda = ev / ttm_ebitda
            checks["ev_ebitda_multiple"] = round(ev_ebitda, 2)
            if ev_ebitda < 5:
                self.warnings.append(f"EV/EBITDA = {ev_ebitda:.1f}x is low; verify assumptions or check for distressed scenario")
            elif ev_ebitda > 40:
                self.warnings.append(f"EV/EBITDA = {ev_ebitda:.1f}x is very high; typical only for exceptional growth")
        
        # ===== 8. SHARES AVAILABILITY =====
        if self.snapshot.shares_outstanding.value:
            checks["equity_value_available"] = True
        else:
            self.warnings.append("Shares outstanding unavailable; cannot compute price per share")
            checks["equity_value_available"] = False
        
        return checks
    
    def _compute_exit_multiple_scenarios(
        self, 
        wacc_minus_g: float, 
        terminal_fcff_ebitda: float,
        terminal_year_ebitda: float,
        shares_outstanding: float,
        net_debt: float,
        pv_fcf_sum: Optional[float] = None
    ) -> None:
        """
        Exit Multiple Cross-Check (3 scenarios): DCF-implied, Current, Industry.
        
        For each multiple M:
        1. TV_exit = EBITDA_N × M
        2. PV(TV_exit) = TV_exit / (1 + WACC)^N
        3. EV_exit = PV(FCFF 1..N) + PV(TV_exit)
        4. Price_exit = (EV_exit - NetDebt) / Shares
        
        Diagnostic (timing-correct formula):
        - Gordon: TV = FCFF_{N+1} / (WACC - g) = FCFF_N × (1+g) / (WACC - g)
        - Exit Multiple: TV = EBITDA_N × M
        - For consistency: FCFF_N × (1+g) / (WACC - g) = EBITDA_N × M
        - Required (FCFF_N / EBITDA_N) = M × (WACC - g) / (1 + g)
        
        Flag: >85% or <0% = FAIL (economically inconsistent)
        """
        FAIL_THRESHOLD_HIGH = 0.85  # >85% is economically implausible
        WARN_THRESHOLD = 0.70  # 70-85% is elevated but possible
        
        scenarios = []
        
        # Get terminal growth rate for timing adjustment
        g = self.assumptions.terminal_growth_rate or 0.03
        one_plus_g = 1 + g
        
        # Store for later use in _finalize_scenario_prices() when we have pv_fcf_sum
        self._scenario_params = {
            'wacc_minus_g': wacc_minus_g,
            'terminal_fcff_ebitda': terminal_fcff_ebitda,
            'terminal_year_ebitda': terminal_year_ebitda,
            'shares_outstanding': shares_outstanding,
            'net_debt': net_debt
        }
        
        if wacc_minus_g <= 0 or terminal_fcff_ebitda is None:
            self.assumptions.exit_multiple_scenarios = scenarios
            return
        
        forecasted_conversion = terminal_fcff_ebitda  # What our model produces at Year N (FCFF_N / EBITDA_N)
        
        def build_scenario(name: str, multiple: float, is_industry: bool = False) -> Optional[Dict]:
            """Build a scenario dict with diagnostic."""
            if not multiple or multiple <= 0:
                return None
            
            # TIMING-CORRECT FORMULA:
            # Required FCFF_N / EBITDA_N = M × (WACC − g) / (1 + g)
            # This accounts for Gordon using FCFF_{N+1} while exit multiple uses EBITDA_N
            required_conversion = multiple * wacc_minus_g / one_plus_g
            gap = required_conversion - forecasted_conversion
            
            # Status based on required conversion
            if required_conversion > FAIL_THRESHOLD_HIGH:
                status = 'FAIL'
                note = 'Economically inconsistent—requires impossible cash conversion.'
            elif required_conversion < 0:
                status = 'FAIL'
                note = 'Economically inconsistent—negative conversion nonsensical.'
            elif required_conversion > WARN_THRESHOLD:
                status = 'WARN'
                note = 'Elevated—possible only if reinvestment declines sharply at maturity.'
            else:
                status = 'PASS'
                note = 'Plausible steady-state scenario.'
            
            # Interpretation with gap
            if abs(gap) < 0.01:
                interpretation = f'{note} Gap: ~0pp (matches forecast).'
            elif gap > 0:
                interpretation = f'{note} Gap: +{gap*100:.0f}pp above forecast—market pricing higher conversion or lower WACC-g.'
            else:
                interpretation = f'{note} Gap: {gap*100:.0f}pp below forecast—implies less efficient conversion.'
            
            # Special note for industry multiple on large-caps
            if is_industry and status == 'FAIL':
                interpretation += ' Note: Industry average is not a defensible terminal multiple for large-cap leaders with differentiated economics.'
            
            return {
                'name': name,
                'multiple': round(multiple, 1),
                'required_fcff_ebitda': round(required_conversion, 3),
                'forecasted_fcff_ebitda': round(forecasted_conversion, 3),
                'gap': round(gap, 3),
                'status': status,
                'is_industry': is_industry,
                'source': 'Damodaran industry avg' if is_industry else 'current market EV/EBITDA',
                'interpretation': interpretation
            }
        
        # 1. Model-Implied Multiple: compute DIRECTLY as TV_gordon / EBITDA_N
        # This avoids rounding mismatches from the ratio shortcut
        tv_gordon = self.assumptions.tv_gordon_growth
        if tv_gordon and terminal_year_ebitda and terminal_year_ebitda > 0:
            consistent_multiple = tv_gordon / terminal_year_ebitda
        else:
            # Fallback to ratio method if TV not yet computed
            consistent_multiple = forecasted_conversion * one_plus_g / wacc_minus_g if wacc_minus_g > 0 else None
        
        self.assumptions.consistent_exit_multiple = round(consistent_multiple, 2) if consistent_multiple else None
        
        if consistent_multiple and consistent_multiple > 0:
            scenarios.append({
                'name': 'Model-implied',
                'multiple': round(consistent_multiple, 1),
                'required_fcff_ebitda': round(forecasted_conversion, 3),
                'forecasted_fcff_ebitda': round(forecasted_conversion, 3),
                'gap': 0.0,
                'status': 'PASS',
                'source': 'Gordon forecast',
                'interpretation': f'Baseline: TV_gordon / EBITDA₁₀ = {consistent_multiple:.1f}x. Price ≈ Gordon Growth.'
            })
        
        # 2. Current Trading Multiple
        current_scenario = build_scenario('Current', self.assumptions.current_ev_ebitda, is_industry=False)
        if current_scenario:
            scenarios.append(current_scenario)
        
        # 3. Industry Multiple (reference only for large-caps)
        industry_scenario = build_scenario('Industry (ref)', self.assumptions.industry_ev_ebitda, is_industry=True)
        if industry_scenario:
            scenarios.append(industry_scenario)
        
        self.assumptions.exit_multiple_scenarios = scenarios
    
    def _finalize_scenario_prices(
        self,
        pv_fcf_sum: float,
        net_debt: float,
        shares: float
    ) -> None:
        """
        Compute price per share for each exit multiple scenario.
        Called after calculate() has pv_fcf_sum.
        """
        if not hasattr(self, '_scenario_params') or not self._scenario_params:
            return
        
        params = self._scenario_params
        terminal_year_ebitda = params.get('terminal_year_ebitda', 0)
        
        if terminal_year_ebitda <= 0 or shares <= 0:
            return
        
        # Update each scenario with price
        for scenario in self.assumptions.exit_multiple_scenarios:
            multiple = scenario.get('multiple', 0)
            if multiple > 0:
                # TV = Year N EBITDA × Multiple
                tv = terminal_year_ebitda * multiple
                pv_tv = tv / ((1 + self.assumptions.wacc) ** self.assumptions.forecast_years)
                ev = pv_fcf_sum + pv_tv
                equity = ev - net_debt
                price = equity / shares
                scenario['tv'] = round(tv, 0)
                scenario['pv_tv'] = round(pv_tv, 0)
                scenario['ev'] = round(ev, 0)
                scenario['price'] = round(price, 2)
