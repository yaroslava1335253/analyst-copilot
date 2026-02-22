"""
Unit tests for DCF engine and data adapter
"""

import pytest
import pandas as pd
from datetime import datetime
from data_adapter import DataAdapter, NormalizedFinancialSnapshot, DataQualityMetadata
from dcf_engine import (
    DCFEngine, DCFAssumptions, CalculationTraceStep, 
    GordonGrowthTerminalValue, ExitMultipleTerminalValue, NetDebtCalculator
)


class TestDataQualityMetadata:
    """Test metadata tracking."""
    
    def test_metadata_creation(self):
        meta = DataQualityMetadata(
            value=1000000,
            units="USD",
            period_end="2025-12-31",
            reliability_score=95
        )
        assert meta.value == 1000000
        assert meta.reliability_score == 95
    
    def test_metadata_to_dict(self):
        meta = DataQualityMetadata(value=100, reliability_score=80)
        d = meta.to_dict()
        assert "value" in d
        assert "reliability_score" in d
        assert d["reliability_score"] == 80


class TestNormalizedFinancialSnapshot:
    """Test financial data snapshot."""
    
    def test_snapshot_initialization(self):
        snap = NormalizedFinancialSnapshot("AAPL")
        assert snap.ticker == "AAPL"
        assert snap.overall_quality_score == 100
    
    def test_add_warning(self):
        snap = NormalizedFinancialSnapshot("MSFT")
        snap.add_warning("TEST_CODE", "Test warning")
        assert len(snap.warnings) == 1
        assert snap.warnings[0]["code"] == "TEST_CODE"
    
    def test_recalculate_overall_quality(self):
        snap = NormalizedFinancialSnapshot("GOOG")
        snap.price.reliability_score = 100
        snap.shares_outstanding.reliability_score = 50
        snap.ttm_revenue.reliability_score = 80
        snap.ttm_fcf.reliability_score = 60
        snap.ttm_operating_income.reliability_score = 70
        snap.recalculate_overall_quality()
        # Average of [100, 50, 80, 60, 70] = 72
        assert snap.overall_quality_score == 72


class TestCalculationTraceStep:
    """Test trace step recording."""
    
    def test_trace_step_creation(self):
        step = CalculationTraceStep(
            name="Test Step",
            formula="x + y",
            inputs={"x": 10, "y": 20},
            output=30
        )
        assert step.name == "Test Step"
        assert step.output == 30
    
    def test_trace_step_to_dict(self):
        step = CalculationTraceStep("Test", "x+y", {"x": 1}, 2)
        d = step.to_dict()
        assert d["name"] == "Test"
        assert d["formula"] == "x+y"


class TestGordonGrowthTerminalValue:
    """Test Gordon Growth Method."""
    
    def test_gordon_growth_calculation(self):
        assumptions = DCFAssumptions(
            forecast_years=5,
            wacc=0.10,
            terminal_growth_rate=0.03
        )
        
        strategy = GordonGrowthTerminalValue()
        trace = []
        tv, pv_tv = strategy.calculate(
            final_year_fcf=100,
            ttm_ebitda=None,
            assumptions=assumptions,
            trace=trace
        )
        
        # Terminal FCF = 100 * 1.03 = 103
        # TV = 103 / (0.10 - 0.03) = 103 / 0.07 ≈ 1471.43
        assert tv > 1400
        # Discount back 5 years: TV / (1.10)^5
        assert pv_tv < tv
        assert len(trace) > 0
    
    def test_gordon_growth_invalid_wacc(self):
        assumptions = DCFAssumptions(
            wacc=0.05,  # Less than terminal growth
            terminal_growth_rate=0.06
        )
        
        strategy = GordonGrowthTerminalValue()
        trace = []
        
        with pytest.raises(ValueError):
            strategy.calculate(100, None, assumptions, trace)


class TestExitMultipleTerminalValue:
    """Test Exit Multiple method."""
    
    def test_exit_multiple_calculation(self):
        assumptions = DCFAssumptions(
            forecast_years=5,
            wacc=0.10,
            fcf_growth_rate=0.05,
            exit_multiple=15
        )
        
        strategy = ExitMultipleTerminalValue()
        trace = []
        tv, pv_tv = strategy.calculate(
            final_year_fcf=100,
            ttm_ebitda=500,
            assumptions=assumptions,
            trace=trace
        )
        
        # Year 5 EBITDA = 500 * (1.05)^5 ≈ 638.14
        # TV = 638.14 * 15 ≈ 9571.98
        assert tv > 9000
        # Discount back
        assert pv_tv < tv
        assert len(trace) > 0
    
    def test_exit_multiple_no_ebitda(self):
        assumptions = DCFAssumptions(exit_multiple=15)
        strategy = ExitMultipleTerminalValue()
        trace = []
        
        with pytest.raises(ValueError):
            strategy.calculate(100, None, assumptions, trace)


class TestNetDebtCalculator:
    """Test net debt calculation."""
    
    def test_net_debt_calculation(self):
        snap = NormalizedFinancialSnapshot("TEST")
        snap.total_debt.value = 10000
        snap.cash_and_equivalents.value = 3000
        
        trace = []
        net_debt, details = NetDebtCalculator.calculate(snap, trace)
        
        assert net_debt == 7000
        assert details["total_debt"] == 10000
        assert details["cash_and_equivalents"] == 3000
        assert len(trace) == 1
    
    def test_net_debt_zero_debt(self):
        snap = NormalizedFinancialSnapshot("TEST")
        snap.total_debt.value = 0
        snap.cash_and_equivalents.value = 5000
        
        trace = []
        net_debt, details = NetDebtCalculator.calculate(snap, trace)
        
        # Negative net debt (net cash position)
        assert net_debt == -5000


class TestDCFEngine:
    """Test DCF valuation engine."""
    
    def setup_method(self):
        """Create a basic snapshot for testing."""
        self.snap = NormalizedFinancialSnapshot("TEST")
        self.snap.price.value = 100
        self.snap.market_cap.value = 100e9
        self.snap.shares_outstanding.value = 1e9
        self.snap.ttm_revenue.value = 500e9
        self.snap.ttm_fcf.value = 50e9
        self.snap.ttm_ebitda.value = 100e9
        self.snap.total_debt.value = 20e9
        self.snap.cash_and_equivalents.value = 5e9
        self.snap.effective_tax_rate.value = 0.25
        
        # Set reliabilities
        for field in [self.snap.price, self.snap.shares_outstanding, self.snap.ttm_revenue,
                      self.snap.ttm_fcf, self.snap.ttm_ebitda]:
            field.reliability_score = 95
    
    def test_dcf_engine_initialization(self):
        assumptions = DCFAssumptions(wacc=0.10, fcf_growth_rate=0.05)
        engine = DCFEngine(self.snap, assumptions)
        assert engine.snapshot == self.snap
        assert engine.assumptions.wacc == 0.10
    
    def test_validate_inputs_success(self):
        engine = DCFEngine(self.snap)
        assert engine.validate_inputs() == True
    
    def test_validate_inputs_missing_revenue(self):
        snap = NormalizedFinancialSnapshot("TEST")
        snap.ttm_fcf.value = 50e9
        snap.shares_outstanding.value = 1e9
        
        engine = DCFEngine(snap)
        assert engine.validate_inputs() == False
        assert len(engine.errors) > 0
    
    def test_set_assumptions_from_defaults(self):
        engine = DCFEngine(self.snap)
        engine.set_assumptions_from_defaults()
        
        # Should auto-assign WACC based on size
        assert engine.assumptions.wacc is not None
        # Should auto-assign growth
        assert engine.assumptions.fcf_growth_rate is not None
        # Should auto-assign exit multiple
        assert engine.assumptions.exit_multiple is not None

    def test_high_growth_watchlist_defaults_to_gordon_first(self):
        snap = NormalizedFinancialSnapshot("TSLA")
        snap.ttm_revenue.value = 100e9
        snap.ttm_fcf.value = 8e9
        snap.ttm_ebitda.value = 20e9
        snap.market_cap.value = 700e9
        snap.shares_outstanding.value = 3e9
        snap.total_debt.value = 10e9
        snap.cash_and_equivalents.value = 20e9
        snap.effective_tax_rate.value = 0.20

        assumptions = DCFAssumptions(
            wacc=0.11,
            terminal_growth_rate=0.03,
            terminal_value_method="gordon_growth"
        )
        engine = DCFEngine(snap, assumptions)
        engine.set_assumptions_from_defaults()

        assert engine.assumptions.high_growth_company is True
        assert engine.assumptions.terminal_value_method == "gordon_growth"

    def test_extreme_gordon_discount_triggers_exit_fallback(self):
        snap = NormalizedFinancialSnapshot("TSLA")
        snap.price.value = 500
        snap.market_cap.value = 500e9
        snap.shares_outstanding.value = 1e9
        snap.ttm_revenue.value = 100e9
        snap.ttm_fcf.value = 5e9
        snap.ttm_operating_income.value = 8e9
        snap.ttm_ebitda.value = 20e9
        snap.ttm_depreciation_amortization.value = 12e9
        snap.ttm_capex.value = 10e9
        snap.ttm_delta_nwc.value = 1e9
        snap.total_debt.value = 10e9
        snap.cash_and_equivalents.value = 5e9
        snap.effective_tax_rate.value = 0.20

        assumptions = DCFAssumptions(
            wacc=0.13,
            terminal_growth_rate=0.03,
            fcf_growth_rate=0.02,
            terminal_value_method="gordon_growth"
        )
        engine = DCFEngine(snap, assumptions)
        result = engine.run()

        assert result["success"] is True
        assert engine.assumptions.price_gordon_growth is not None
        assert engine.assumptions.price_exit_multiple is not None
        assert engine.assumptions.price_gordon_growth < (
            snap.price.value * DCFEngine.EXTREME_GORDON_PRICE_TO_MARKET_THRESHOLD
        )
        assert engine.assumptions.terminal_value_method == "exit_multiple"

    def test_analyst_consensus_anchors_all_available_years(self):
        snap = NormalizedFinancialSnapshot("ANCHOR")
        snap.ttm_revenue.value = 100e9
        snap.ttm_fcf.value = 10e9
        snap.ttm_operating_income.value = 12e9
        snap.ttm_ebitda.value = 18e9
        snap.ttm_depreciation_amortization.value = 6e9
        snap.ttm_capex.value = 4e9
        snap.ttm_delta_nwc.value = 0.5e9
        snap.market_cap.value = 300e9  # Forces 10-year horizon
        snap.shares_outstanding.value = 1e9
        snap.total_debt.value = 5e9
        snap.cash_and_equivalents.value = 10e9
        snap.effective_tax_rate.value = 0.21
        snap.suggested_fcf_growth.value = 0.10
        snap.analyst_revenue_estimates = [
            {"year_label": "0y", "revenue": 110e9},
            {"year_label": "+1y", "revenue": 125e9},
            {"year_label": "+2y", "revenue": 140e9},
            {"year_label": "+3y", "revenue": 160e9},
            {"year_label": "+4y", "revenue": 180e9},
        ]

        engine = DCFEngine(snap, DCFAssumptions(wacc=0.10, terminal_growth_rate=0.03))
        result = engine.run()

        assert result["success"] is True
        assert engine.assumptions.consensus_revenue_used_years[:5] == [1, 2, 3, 4, 5]
        assert len(engine.assumptions.yearly_projections) >= 5

        expected_revenues = [110e9, 125e9, 140e9, 160e9, 180e9]
        for idx, expected in enumerate(expected_revenues):
            assert engine.assumptions.yearly_projections[idx]["revenue"] == pytest.approx(expected)
    
    def test_dcf_run_success(self):
        assumptions = DCFAssumptions(
            wacc=0.10,
            fcf_growth_rate=0.05,
            exit_multiple=15,
            terminal_value_method="exit_multiple"
        )
        engine = DCFEngine(self.snap, assumptions)
        result = engine.run()
        
        assert result["success"] == True
        assert result["enterprise_value"] > 0
        assert result["equity_value"] > 0
        assert result["price_per_share"] > 0
        assert len(result["trace"]) > 0
    
    def test_dcf_missing_inputs(self):
        snap = NormalizedFinancialSnapshot("TEST")
        engine = DCFEngine(snap)
        result = engine.run()
        
        assert result["success"] == False
        assert len(result["errors"]) > 0
    
    def test_fcf_projection(self):
        """Test 5-year FCF projection."""
        assumptions = DCFAssumptions(wacc=0.10, fcf_growth_rate=0.05)
        engine = DCFEngine(self.snap, assumptions)
        
        projections = engine._project_fcf()
        
        assert len(projections) == 5
        # FCF should grow each year
        for i, proj in enumerate(projections):
            assert proj["year"] == i + 1
            assert proj["pv"] > 0
            assert "discount_factor" in proj
    
    def test_terminal_value_gordon_growth(self):
        assumptions = DCFAssumptions(
            wacc=0.10,
            fcf_growth_rate=0.05,
            terminal_value_method="gordon_growth",
            terminal_growth_rate=0.03
        )
        engine = DCFEngine(self.snap, assumptions)
        
        final_fcf = 50e9 * ((1.05) ** 5)
        tv, pv_tv = engine._calculate_terminal_value(final_fcf)
        
        assert tv > 0
        assert pv_tv > 0
        assert pv_tv < tv  # Should be discounted
    
    def test_terminal_value_exit_multiple(self):
        assumptions = DCFAssumptions(
            wacc=0.10,
            fcf_growth_rate=0.05,
            exit_multiple=15,
            terminal_value_method="exit_multiple"
        )
        engine = DCFEngine(self.snap, assumptions)
        
        final_fcf = 50e9 * ((1.05) ** 5)
        tv, pv_tv = engine._calculate_terminal_value(final_fcf)
        
        assert tv > 0
        assert pv_tv > 0
    
    def test_sanity_checks(self):
        """Test sanity check logic."""
        assumptions = DCFAssumptions(
            wacc=0.10,
            fcf_growth_rate=0.05,
            exit_multiple=15
        )
        engine = DCFEngine(self.snap, assumptions)
        
        # Run full DCF to generate sanity checks
        result = engine.run()
        
        checks = result.get("sanity_checks", {})
        assert "terminal_value_dominance" in checks
        # For this size company, EV/EBITDA should be reasonable
        if "ev_ebitda_multiple" in checks:
            assert 5 <= checks["ev_ebitda_multiple"] <= 50


class TestDCFIntegration:
    """Integration tests for full DCF workflow."""
    
    def test_full_dcf_with_exit_multiple(self):
        """Test complete DCF with Exit Multiple terminal value."""
        snap = NormalizedFinancialSnapshot("INTEGRATION_TEST")
        snap.price.value = 150
        snap.market_cap.value = 150e9
        snap.shares_outstanding.value = 1e9
        snap.ttm_revenue.value = 500e9
        snap.ttm_fcf.value = 50e9
        snap.ttm_ebitda.value = 100e9
        snap.total_debt.value = 20e9
        snap.cash_and_equivalents.value = 5e9
        snap.effective_tax_rate.value = 0.25
        
        for field in [snap.price, snap.shares_outstanding, snap.ttm_revenue,
                      snap.ttm_fcf, snap.ttm_ebitda]:
            field.reliability_score = 95
        
        assumptions = DCFAssumptions(
            wacc=0.08,
            fcf_growth_rate=0.07,
            exit_multiple=16,
            terminal_value_method="exit_multiple"
        )
        
        engine = DCFEngine(snap, assumptions)
        result = engine.run()
        
        assert result["success"] == True
        assert result["enterprise_value"] > 0
        assert result["equity_value"] > 0
        assert result["net_debt"] == 15e9  # 20B - 5B
        
        # Equity should be EV - net debt
        assert abs(result["equity_value"] - (result["enterprise_value"] - 15e9)) < 1e6
    
    def test_full_dcf_with_gordon_growth(self):
        """Test complete DCF with Gordon Growth terminal value."""
        snap = NormalizedFinancialSnapshot("GORDON_TEST")
        snap.price.value = 100
        snap.market_cap.value = 100e9
        snap.shares_outstanding.value = 1e9
        snap.ttm_revenue.value = 400e9
        snap.ttm_fcf.value = 40e9
        snap.ttm_ebitda.value = None  # No EBITDA; use Gordon Growth
        snap.total_debt.value = 15e9
        snap.cash_and_equivalents.value = 3e9
        snap.effective_tax_rate.value = 0.25
        
        for field in [snap.price, snap.shares_outstanding, snap.ttm_revenue,
                      snap.ttm_fcf]:
            field.reliability_score = 90
        
        assumptions = DCFAssumptions(
            wacc=0.09,
            fcf_growth_rate=0.04,
            terminal_value_method="gordon_growth",
            terminal_growth_rate=0.025
        )
        
        engine = DCFEngine(snap, assumptions)
        result = engine.run()
        
        assert result["success"] == True
        assert result["enterprise_value"] > 0
        # Should fall back to Gordon Growth and warn about EBITDA
        assert any("EBITDA" in w for w in result.get("warnings", []))


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_zero_fcf(self):
        snap = NormalizedFinancialSnapshot("ZERO_FCF")
        snap.shares_outstanding.value = 1e9
        snap.ttm_revenue.value = 100e9
        snap.ttm_fcf.value = 0  # Zero FCF
        
        engine = DCFEngine(snap)
        result = engine.run()
        
        assert result["success"] == False
        assert len(result["errors"]) > 0
    
    def test_negative_net_debt(self):
        """Test company with net cash position."""
        snap = NormalizedFinancialSnapshot("NET_CASH")
        snap.price.value = 200
        snap.shares_outstanding.value = 1e9
        snap.ttm_revenue.value = 300e9
        snap.ttm_fcf.value = 30e9
        snap.ttm_ebitda.value = 60e9
        snap.total_debt.value = 5e9
        snap.cash_and_equivalents.value = 20e9  # More cash than debt
        snap.effective_tax_rate.value = 0.25
        
        for field in [snap.price, snap.shares_outstanding, snap.ttm_revenue,
                      snap.ttm_fcf, snap.ttm_ebitda]:
            field.reliability_score = 95
        
        assumptions = DCFAssumptions(
            wacc=0.08,
            fcf_growth_rate=0.06,
            exit_multiple=14
        )
        
        engine = DCFEngine(snap, assumptions)
        result = engine.run()
        
        assert result["success"] == True
        # Net debt should be negative
        assert result["net_debt"] < 0
        # Equity value > EV (because adding cash)
        assert result["equity_value"] > result["enterprise_value"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
