"""
DCF UI Adapter: Transform engine output into UI-safe, traceable format
========================================================================
This adapter:
1. Maps engine output keys to UI keys correctly
2. Transforms trace objects into renderable tables
3. Applies smart formatting (no silent zeros, consistent units)
4. Performs consistency checks (equity/shares → per-share validation)
5. Returns a single UI-ready object with all data, trace, and diagnostics
"""

from typing import Dict, Optional, List, Any
from dataclasses import dataclass, asdict


@dataclass
class FinancialMetric:
    """A single financial metric with full context."""
    name: str
    value: Optional[float]
    units: str = "USD"
    period_type: str = "TTM"  # TTM, annual, quarterly
    period_end: Optional[str] = None
    source_path: Optional[str] = None
    reliability_score: Optional[int] = None
    notes: Optional[str] = None
    is_missing: bool = False
    
    def formatted(self, precision: int = 1) -> str:
        """Format value for display (handles None, zero, negative)."""
        if self.value is None or self.is_missing:
            return "—"
        if self.value == 0:
            return "—"  # Don't show zero; use dash for missing/unmeasurable
        if self.units == "USD":
            if abs(self.value) >= 1e9:
                return f"${self.value/1e9:.{precision}f}B"
            elif abs(self.value) >= 1e6:
                return f"${self.value/1e6:.{precision}f}M"
            else:
                return f"${self.value:.{precision}f}"
        elif self.units == "%":
            return f"{self.value:.{precision}f}%"
        elif self.units == "shares":
            if abs(self.value) >= 1e9:
                return f"{self.value/1e9:.{precision}f}B"
            elif abs(self.value) >= 1e6:
                return f"{self.value/1e6:.{precision}f}M"
            else:
                return f"{self.value:.{precision}f}"
        elif self.units == "x":
            return f"{self.value:.{precision}f}x"
        else:
            return f"{self.value:.{precision}f}"
    
    def source_short(self) -> str:
        """Return a short, readable source label."""
        if not self.source_path:
            return "—"
        path = self.source_path
        # Shorten common patterns
        if "yf.Ticker.info" in path:
            return "yfinance.info"
        if "yf.Ticker.fast_info" in path:
            return "yfinance.fast_info"
        if "yahooquery" in path.lower():
            return "yahooquery"
        if "quarterly_income_stmt" in path:
            return "quarterly_income"
        if "quarterly_balance_sheet" in path or "balance_sheet[" in path:
            return "quarterly_balance"
        if "quarterly_cashflow" in path:
            return "quarterly_cf"
        if "annual_income_stmt" in path:
            return "annual_income"
        if "annual_cashflow" in path:
            return "annual_cf"
        if "TTM sum" in path:
            return "TTM (4Q sum)"
        if "info['" in path:
            # Extract the key name
            start = path.find("info['") + 6
            end = path.find("']", start)
            if end > start:
                return f"info.{path[start:end]}"
        return path[:20] + "..." if len(path) > 20 else path
    
    def formatted_with_source(self, precision: int = 1) -> str:
        """Format value with source in parentheses."""
        val = self.formatted(precision)
        src = self.source_short()
        if src and src != "—":
            return f"{val} ({src})"
        return val
    
    def to_dict(self):
        return {
            "name": self.name,
            "value": self.value,
            "units": self.units,
            "period_type": self.period_type,
            "period_end": self.period_end,
            "source_path": self.source_path,
            "reliability_score": self.reliability_score,
            "notes": self.notes,
            "is_missing": self.is_missing,
            "formatted": self.formatted()
        }


class DCFUIAdapter:
    """Transform engine output into UI-safe format."""
    
    def __init__(self, engine_result: Dict[str, Any], snapshot: Any):
        """
        Args:
            engine_result: dict from DCFEngine.run()
            snapshot: NormalizedFinancialSnapshot from DataAdapter.fetch()
        """
        self.engine_result = engine_result
        self.snapshot = snapshot
        self.ui_data = {}
        self.diagnostics = []
        self._transform()
    
    def _transform(self):
        """Perform all transformations and validations."""
        if not self.engine_result.get("success"):
            self.ui_data = {
                "success": False,
                "error": self.engine_result.get("errors", ["Unknown error"]),
                "warnings": self.engine_result.get("warnings", [])
            }
            return
        
        # Extract from engine result
        ev = self.engine_result.get("enterprise_value", 0)
        equity = self.engine_result.get("equity_value", 0)
        shares = self.engine_result.get("shares_outstanding", 0)
        price_per_share = self.engine_result.get("price_per_share", 0)
        
        # Consistency check: if equity and shares both nonzero, per-share MUST be nonzero
        if equity and equity > 0 and shares and shares > 0:
            computed_per_share = equity / shares
            if price_per_share is None or price_per_share == 0:
                price_per_share = computed_per_share
                self.diagnostics.append(
                    f"⚠️ Engine returned price_per_share=0; computed {computed_per_share:.2f} from equity/shares"
                )
            elif abs(price_per_share - computed_per_share) / computed_per_share > 0.01:
                self.diagnostics.append(
                    f"⚠️ price_per_share mismatch: engine={price_per_share:.2f}, computed={computed_per_share:.2f}"
                )
        
        # Extract snapshot data (with metadata)
        inputs = {
            "current_price": FinancialMetric(
                name="Current Price",
                value=self.snapshot.price.value,
                units="USD",
                reliability_score=self.snapshot.price.reliability_score,
                source_path=self.snapshot.price.source_path,
                is_missing=self.snapshot.price.value is None or self.snapshot.price.value == 0
            ),
            "shares_outstanding": FinancialMetric(
                name="Shares Outstanding",
                value=shares,
                units="shares",
                reliability_score=self.snapshot.shares_outstanding.reliability_score,
                source_path=self.snapshot.shares_outstanding.source_path,
                is_missing=shares is None or shares == 0
            ),
            "market_cap": FinancialMetric(
                name="Market Cap",
                value=self.snapshot.market_cap.value,
                units="USD",
                reliability_score=self.snapshot.market_cap.reliability_score,
                source_path=self.snapshot.market_cap.source_path,
                is_missing=self.snapshot.market_cap.value is None or self.snapshot.market_cap.value == 0
            ),
            "ttm_revenue": FinancialMetric(
                name="TTM Revenue",
                value=self.snapshot.ttm_revenue.value,
                units="USD",
                period_type="TTM",
                reliability_score=self.snapshot.ttm_revenue.reliability_score,
                source_path=self.snapshot.ttm_revenue.source_path,
                is_missing=self.snapshot.ttm_revenue.value is None or self.snapshot.ttm_revenue.value == 0
            ),
            "ttm_ebitda": FinancialMetric(
                name="TTM EBITDA",
                value=self.snapshot.ttm_ebitda.value,
                units="USD",
                period_type="TTM",
                reliability_score=self.snapshot.ttm_ebitda.reliability_score,
                source_path=self.snapshot.ttm_ebitda.source_path,
                is_missing=self.snapshot.ttm_ebitda.value is None or self.snapshot.ttm_ebitda.value == 0
            ),
            "ttm_operating_income": FinancialMetric(
                name="TTM Operating Income",
                value=self.snapshot.ttm_operating_income.value,
                units="USD",
                period_type="TTM",
                reliability_score=self.snapshot.ttm_operating_income.reliability_score,
                source_path=self.snapshot.ttm_operating_income.source_path,
                is_missing=self.snapshot.ttm_operating_income.value is None or self.snapshot.ttm_operating_income.value == 0
            ),
            "ttm_operating_cash_flow": FinancialMetric(
                name="TTM Operating Cash Flow",
                value=self.snapshot.ttm_operating_cash_flow.value,
                units="USD",
                period_type=self.snapshot.ttm_operating_cash_flow.period_type or "TTM",
                reliability_score=self.snapshot.ttm_operating_cash_flow.reliability_score,
                source_path=self.snapshot.ttm_operating_cash_flow.source_path,
                notes=self.snapshot.ttm_operating_cash_flow.fallback_reason,
                is_missing=self.snapshot.ttm_operating_cash_flow.value is None or self.snapshot.ttm_operating_cash_flow.value == 0
            ),
            "ttm_capex": FinancialMetric(
                name="TTM CapEx",
                value=self.snapshot.ttm_capex.value,
                units="USD",
                period_type=self.snapshot.ttm_capex.period_type or "TTM",
                reliability_score=self.snapshot.ttm_capex.reliability_score,
                source_path=self.snapshot.ttm_capex.source_path,
                notes=self.snapshot.ttm_capex.fallback_reason or "Converted from yfinance negative",
                is_missing=self.snapshot.ttm_capex.value is None or self.snapshot.ttm_capex.value == 0
            ),
            "ttm_fcf": FinancialMetric(
                name="TTM Free Cash Flow",
                value=self.snapshot.ttm_fcf.value,
                units="USD",
                period_type=self.snapshot.ttm_fcf.period_type or "TTM",
                reliability_score=self.snapshot.ttm_fcf.reliability_score,
                source_path=self.snapshot.ttm_fcf.source_path,
                is_missing=self.snapshot.ttm_fcf.value is None or self.snapshot.ttm_fcf.value == 0
            ),
            "total_debt": FinancialMetric(
                name="Total Debt",
                value=self.snapshot.total_debt.value,
                units="USD",
                reliability_score=self.snapshot.total_debt.reliability_score,
                source_path=self.snapshot.total_debt.source_path,
                is_missing=self.snapshot.total_debt.value is None or self.snapshot.total_debt.value == 0
            ),
            "cash": FinancialMetric(
                name="Cash & Equivalents",
                value=self.snapshot.cash_and_equivalents.value,
                units="USD",
                reliability_score=self.snapshot.cash_and_equivalents.reliability_score,
                source_path=self.snapshot.cash_and_equivalents.source_path,
                is_missing=self.snapshot.cash_and_equivalents.value is None or self.snapshot.cash_and_equivalents.value == 0
            ),
        }
        
        # Extract assumptions
        assumptions = self.engine_result.get("assumptions", {})
        
        # Build assumptions object (handle None/default values)
        assumptions_obj = {
            "wacc": assumptions.get("wacc") or 0.08,  # Default 8%
            "fcf_growth_rate": assumptions.get("fcf_growth_rate") or 0.05,  # Default 5%
            "terminal_growth_rate": assumptions.get("terminal_growth_rate") or 0.03,  # Default 3%
            "exit_multiple": assumptions.get("exit_multiple") or 15,  # Default 15x
            "forecast_years": assumptions.get("forecast_years") or 5,
            "discount_convention": assumptions.get("discount_convention") or "end_of_year",
            "terminal_value_method": assumptions.get("terminal_value_method") or "gordon_growth",
            "tax_rate": assumptions.get("tax_rate") or 0.15,  # Default 15%
            # Industry multiple metadata
            "industry_multiple_source": assumptions.get("industry_multiple_source"),
            "damodaran_industry": assumptions.get("damodaran_industry"),
            "yf_industry": assumptions.get("yf_industry"),
            "yf_sector": assumptions.get("yf_sector"),
            "is_exact_industry_match": assumptions.get("is_exact_industry_match", False),
            # Terminal multiple (current-anchored)
            "current_ev_ebitda": assumptions.get("current_ev_ebitda"),
            "industry_ev_ebitda": assumptions.get("industry_ev_ebitda"),
            "terminal_multiple_source": assumptions.get("terminal_multiple_source"),
            # Scenario analysis
            "scenario_market_anchored": assumptions.get("scenario_market_anchored"),
            "scenario_mean_reversion": assumptions.get("scenario_mean_reversion"),
            "scenario_conservative": assumptions.get("scenario_conservative"),
            # Dual Terminal Value cross-check
            "tv_exit_multiple": assumptions.get("tv_exit_multiple"),
            "tv_gordon_growth": assumptions.get("tv_gordon_growth"),
            "pv_tv_exit_multiple": assumptions.get("pv_tv_exit_multiple"),
            "pv_tv_gordon_growth": assumptions.get("pv_tv_gordon_growth"),
            "price_exit_multiple": assumptions.get("price_exit_multiple"),
            "price_gordon_growth": assumptions.get("price_gordon_growth"),
            # Implied terminal FCF yield
            "implied_ev_fcf": assumptions.get("implied_ev_fcf"),
            "implied_fcf_yield": assumptions.get("implied_fcf_yield"),
            # FCFF metrics (proper enterprise DCF)
            "ttm_fcff": assumptions.get("ttm_fcff"),
            "fcff_method": assumptions.get("fcff_method"),  # "proper_fcff", "approx_unlevered", "levered_proxy"
            "fcff_reliability": assumptions.get("fcff_reliability"),  # 95, 80, 70, or 50 (higher=better)
            "fcff_ebitda_ratio": assumptions.get("fcff_ebitda_ratio"),
            "implied_gordon_ev_ebitda": assumptions.get("implied_gordon_ev_ebitda"),
            "required_fcff_ebitda_for_exit": assumptions.get("required_fcff_ebitda_for_exit"),
            # Terminal Multiple Scenario Analysis
            "terminal_multiple_scenario": assumptions.get("terminal_multiple_scenario", "current"),
            "terminal_multiple_rerating_pct": assumptions.get("terminal_multiple_rerating_pct"),
            "observed_fcff_ebitda_ttm": assumptions.get("observed_fcff_ebitda_ttm"),
            "observed_fcff_ebitda_year5": assumptions.get("observed_fcff_ebitda_year5"),
            # Exit Multiple Scenarios (3-scenario cross-check)
            "exit_multiple_scenarios": assumptions.get("exit_multiple_scenarios", []),
            "terminal_year_fcff_ebitda": assumptions.get("terminal_year_fcff_ebitda"),
            "consistent_exit_multiple": assumptions.get("consistent_exit_multiple"),
            "projected_terminal_ebitda": assumptions.get("projected_terminal_ebitda"),
            # Terminal year absolutes (CRITICAL for market-implied reconciliation)
            "terminal_year_fcff": assumptions.get("terminal_year_fcff"),
            "terminal_year_ebitda": assumptions.get("terminal_year_ebitda"),
            # Confidence flags
            "tv_dominance_pct": assumptions.get("tv_dominance_pct"),
            "growth_proxy_warning": assumptions.get("growth_proxy_warning", False),
            "data_quality_score": assumptions.get("data_quality_score"),
            "wacc_is_estimated": assumptions.get("wacc_is_estimated", True),
            # Cash-flow regime gating metadata
            "cashflow_regime": assumptions.get("cashflow_regime"),
            "cashflow_confidence": assumptions.get("cashflow_confidence"),
            "discount_rate_input": assumptions.get("discount_rate_input"),
            "discount_rate_used": assumptions.get("discount_rate_used"),
            "discount_rate_label": assumptions.get("discount_rate_label"),
            "discount_rate_source": assumptions.get("discount_rate_source"),
            "proxy_adjustment_applied": assumptions.get("proxy_adjustment_applied", False),
            # ── Textbook driver model fields ──
            "use_driver_model": assumptions.get("use_driver_model", True),
            "yearly_projections": assumptions.get("yearly_projections", []),
            "display_years": assumptions.get("display_years", 5),
            "is_large_cap": assumptions.get("is_large_cap", False),
            "horizon_reason": assumptions.get("horizon_reason"),
            "near_term_growth_rate": assumptions.get("near_term_growth_rate"),
            "effective_near_term_growth_rate": assumptions.get("effective_near_term_growth_rate"),
            "analyst_long_term_growth_rate": assumptions.get("analyst_long_term_growth_rate"),
            "growth_schedule_method": assumptions.get("growth_schedule_method"),
            "high_growth_company": assumptions.get("high_growth_company", False),
            "high_growth_reasons": assumptions.get("high_growth_reasons", []),
            "stable_growth_rate": assumptions.get("stable_growth_rate"),
            "base_roic": assumptions.get("base_roic"),
            "terminal_roic": assumptions.get("terminal_roic"),
            "industry_roic": assumptions.get("industry_roic"),
            "terminal_reinvestment_rate": assumptions.get("terminal_reinvestment_rate"),
            "base_revenue": assumptions.get("base_revenue"),
            # Analyst FCF anchor metadata
            "analyst_fcf_anchors_used": assumptions.get("analyst_fcf_anchors_used", False),
            "consensus_revenue_used_years": assumptions.get("consensus_revenue_used_years", []),
            "fcf_sources": assumptions.get("fcf_sources", []),
            "revenue_sources": assumptions.get("revenue_sources", []),
        }
        
        # Extract outputs with validation
        net_debt = self.engine_result.get("net_debt", 0)
        pv_fcf = self.engine_result.get("pv_fcf_sum", 0)
        pv_tv = self.engine_result.get("pv_terminal_value", 0)
        
        # Data quality score
        quality_score = self.engine_result.get("data_quality_score", 
                                               self.snapshot.overall_quality_score if self.snapshot else 0)
        
        # Build UI data
        self.ui_data = {
            "success": True,
            "enterprise_value": ev,
            "equity_value": equity,
            "net_debt": net_debt,
            "net_debt_details": self.engine_result.get("net_debt_details", {}),
            "price_per_share": price_per_share,
            "shares_outstanding": shares,
            "current_price": self.snapshot.price.value if self.snapshot else 0,
            "pv_fcf_sum": pv_fcf,
            "pv_terminal_value": pv_tv,
            "terminal_value_yearN": self.engine_result.get("terminal_value_yearN", 0),
            "data_quality_score": quality_score,
            
            # Inputs with full metadata
            "inputs": inputs,
            
            # Assumptions
            "assumptions": assumptions_obj,
            
            # FCF Projections
            "fcf_projections": self.engine_result.get("fcf_projections", []),
            
            # Trace (for details page)
            "trace": self.engine_result.get("trace", []),
            
            # Warnings and errors
            "warnings": self.engine_result.get("warnings", []) + self.snapshot.warnings if self.snapshot else [],
            "errors": self.engine_result.get("errors", []) + self.snapshot.errors if self.snapshot else [],
            
            # Diagnostics from transformation
            "diagnostics": self.diagnostics,
            
            # Sanity checks
            "sanity_checks": self.engine_result.get("sanity_checks", {}),
        }
        
        # Data sufficiency gate
        self.ui_data["data_sufficient"] = self._check_data_sufficiency()
    
    def _check_data_sufficiency(self) -> bool:
        """Gate: is data sufficient to display confident valuations?"""
        inputs = self.ui_data.get("inputs", {})
        quality = self.ui_data.get("data_quality_score", 0)
        
        # Required inputs
        required = ["current_price", "shares_outstanding", "ttm_revenue", "ttm_fcf"]
        for req in required:
            metric = inputs.get(req)
            if metric and metric.is_missing:
                return False
        
        # Quality gate
        if quality < 60:
            return False
        
        return True
    
    def get_ui_data(self) -> Dict[str, Any]:
        """Return UI-ready data object."""
        return self.ui_data
    
    def format_input_table(self) -> List[Dict]:
        """Transform inputs into table rows."""
        inputs = self.ui_data.get("inputs", {})
        rows = []
        for key, metric in inputs.items():
            if metric.is_missing:
                rows.append({
                    "Item": metric.name,
                    "Value": "—",
                    "Units": metric.units,
                    "Period": metric.period_type,
                    "Source": metric.source_path or "—",
                    "Reliability": f"{metric.reliability_score}/100" if metric.reliability_score else "—",
                    "Notes": metric.notes or "—"
                })
            else:
                rows.append({
                    "Item": metric.name,
                    "Value": metric.formatted(),
                    "Units": metric.units,
                    "Period": metric.period_type,
                    "Source": metric.source_path or "—",
                    "Reliability": f"{metric.reliability_score}/100" if metric.reliability_score else "—",
                    "Notes": metric.notes or "—"
                })
        return rows
    
    def format_assumptions_table(self) -> List[Dict]:
        """Transform assumptions into table rows."""
        assumptions = self.ui_data.get("assumptions", {})
        
        # Determine exit multiple source description
        tv_method = assumptions.get('terminal_value_method', 'gordon_growth')
        if tv_method == 'exit_multiple':
            damodaran_industry = assumptions.get('damodaran_industry')
            is_exact = assumptions.get('is_exact_industry_match', False)
            if damodaran_industry:
                match_type = "exact" if is_exact else "approx"
                exit_notes = f"Damodaran '{damodaran_industry}' ({match_type})"
            else:
                exit_notes = "Size-based default"
            exit_value = f"{assumptions.get('exit_multiple', 15):.1f}x EV/EBITDA"
        else:
            exit_notes = "Not used (Gordon Growth)"
            exit_value = "N/A"
        
        return [
            {"Assumption": "WACC (Discount Rate)", "Value": f"{assumptions['wacc']:.2%}", "Notes": "Cost of equity + weighted cost of debt"},
            {"Assumption": "FCF Growth Rate (Years 1-5)", "Value": f"{assumptions['fcf_growth_rate']:.2%}", "Notes": "Applied to each year"},
            {"Assumption": "Terminal Growth Rate", "Value": f"{assumptions['terminal_growth_rate']:.2%}", "Notes": "Perpetual growth (Gordon Growth only)"},
            {"Assumption": "Terminal Value Method", "Value": tv_method.replace("_", " ").title(), "Notes": "Exit Multiple (industry-based) or Gordon Growth"},
            {"Assumption": "Exit Multiple (if used)", "Value": exit_value, "Notes": exit_notes},
            {"Assumption": "Tax Rate", "Value": f"{assumptions['tax_rate']:.2%}", "Notes": "For FCFF calculation"},
            {"Assumption": "Forecast Period", "Value": f"{assumptions['forecast_years']} years", "Notes": "Years 1-5"},
        ]
    
    def format_fcf_projection_table(self) -> List[Dict]:
        """Transform FCF projections into table rows."""
        projections = self.ui_data.get("fcf_projections", [])
        rows = []
        for proj in projections:
            rows.append({
                "Year": f"Year {proj.get('year', 0)}",
                "FCF": f"${proj.get('fcf', 0)/1e9:.1f}B",
                "Discount Factor": f"{proj.get('discount_factor', 0):.4f}",
                "PV(FCF)": f"${proj.get('pv', 0)/1e9:.1f}B"
            })
        return rows
    
    def format_bridge_table(self) -> List[Dict]:
        """EV → Equity Value → Per-Share bridge."""
        ev = self.ui_data.get("enterprise_value", 0)
        nd = self.ui_data.get("net_debt", 0)
        nd_details = self.ui_data.get("net_debt_details", {})
        total_debt = nd_details.get("total_debt", 0)
        cash = nd_details.get("cash_and_equivalents", 0)
        total_debt_source = nd_details.get("total_debt_source", "balance_sheet")
        cash_source = nd_details.get("cash_source", "balance_sheet")
        equity = self.ui_data.get("equity_value", 0)
        shares = self.ui_data.get("shares_outstanding", 0)
        per_share = self.ui_data.get("price_per_share", 0)
        
        return [
            {
                "Component": "PV(FCF Years 1–5)",
                "Value": f"${self.ui_data.get('pv_fcf_sum', 0)/1e9:.1f}B",
                "Formula/Notes": "Σ(FCF_t / (1+WACC)^t) for t=1..5"
            },
            {
                "Component": "PV(Terminal Value)",
                "Value": f"${self.ui_data.get('pv_terminal_value', 0)/1e9:.1f}B",
                "Formula/Notes": "Terminal Value / (1+WACC)^5"
            },
            {
                "Component": "= Enterprise Value",
                "Value": f"${ev/1e9:.1f}B",
                "Formula/Notes": "PV(FCF) + PV(TV)"
            },
            {
                "Component": "  Total Debt",
                "Value": f"${total_debt/1e9:.2f}B",
                "Formula/Notes": f"Source: {total_debt_source}"
            },
            {
                "Component": "  Cash & Equivalents",
                "Value": f"${cash/1e9:.2f}B",
                "Formula/Notes": f"Source: {cash_source}"
            },
            {
                "Component": "− Net Debt",
                "Value": f"${nd/1e9:.2f}B",
                "Formula/Notes": f"Total Debt (${total_debt/1e9:.2f}B) − Cash (${cash/1e9:.2f}B)"
            },
            {
                "Component": "= Equity Value",
                "Value": f"${equity/1e9:.1f}B",
                "Formula/Notes": "EV − Net Debt"
            },
            {
                "Component": "÷ Shares Outstanding",
                "Value": f"{shares/1e9:.2f}B shares",
                "Formula/Notes": "Diluted share count"
            },
            {
                "Component": "= Intrinsic Value/Share",
                "Value": f"${per_share:.2f}",
                "Formula/Notes": "Equity Value ÷ Shares"
            }
        ]
