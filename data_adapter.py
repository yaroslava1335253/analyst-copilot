"""
DataAdapter: Normalize and quality-score yfinance financial data
==================================================================
Handles yfinance's inconsistencies and gaps with full traceability.

Key responsibilities:
1. Fetch from multiple yfinance sources (fast_info, info, quarterly, annual)
2. Score data reliability (None/NaN/missing statements drop reliability)
3. Standardize output with metadata (period, source_path, reliability_score)
4. Build TTM intelligently (prefer quarterly sum, fallback to annual)
5. Track which line items are estimated vs actual
"""

import yfinance as yf
import pandas as pd
import json
from datetime import datetime
from typing import Dict, Optional, Tuple, List
from yahooquery import Ticker as YQTicker


class DataQualityMetadata:
    """Metadata for a single financial data point."""
    def __init__(self, value=None, units="USD", period_end=None, period_type=None,
                 source_path=None, retrieved_at=None, reliability_score=100, notes=None,
                 is_estimated=False, fallback_reason=None):
        self.value = value
        self.units = units
        self.period_end = period_end
        self.period_type = period_type  # "annual", "quarterly", "ttm"
        self.source_path = source_path  # e.g., "yf.Ticker.info['marketCap']"
        self.retrieved_at = retrieved_at
        self.reliability_score = reliability_score  # 0-100
        self.notes = notes or ""
        self.is_estimated = is_estimated  # True if calculated/imputed
        self.fallback_reason = fallback_reason  # e.g., "No quarterly CFO; used annual"
    
    def to_dict(self):
        return {
            "value": self.value,
            "units": self.units,
            "period_end": self.period_end,
            "period_type": self.period_type,
            "source_path": self.source_path,
            "retrieved_at": self.retrieved_at,
            "reliability_score": self.reliability_score,
            "notes": self.notes,
            "is_estimated": self.is_estimated,
            "fallback_reason": self.fallback_reason
        }


class NormalizedFinancialSnapshot:
    """Standardized financial data for a ticker with quality metadata."""
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.retrieved_at = datetime.utcnow().isoformat()
        
        # Core price/shares data
        self.price = DataQualityMetadata()
        self.shares_outstanding = DataQualityMetadata()
        self.market_cap = DataQualityMetadata()
        self.currency = "USD"
        
        # Company classification (for industry multiples)
        self.sector = None  # e.g., "Technology"
        self.industry = None  # e.g., "Software—Application"
        self.company_name = None  # e.g., "Apple Inc."
        
        # Debt/Cash (for EV->Equity bridge)
        self.total_debt = DataQualityMetadata()
        self.cash_and_equivalents = DataQualityMetadata()
        self.minority_interest = DataQualityMetadata()
        self.preferred_stock = DataQualityMetadata()
        
        # Working Capital (for FCFF calculation)
        self.current_assets = DataQualityMetadata()  # Excl. cash
        self.current_liabilities = DataQualityMetadata()  # Excl. debt
        self.net_working_capital = DataQualityMetadata()  # CA - CL (operating)
        self.prior_net_working_capital = DataQualityMetadata()  # Prior period NWC (quarterly)
        self.delta_nwc = DataQualityMetadata()  # Quarterly change in NWC (LEGACY - DO NOT USE FOR TTM)
        self.ttm_delta_nwc = DataQualityMetadata()  # TTM change in NWC from cash flow statement
        self.nwc_pct_revenue = DataQualityMetadata()  # NWC as % of revenue
        
        # TTM/LTM Cash Flow
        self.ttm_operating_cash_flow = DataQualityMetadata()
        self.ttm_capex = DataQualityMetadata()
        self.ttm_fcf = DataQualityMetadata()  # CFO - CapEx proxy (LEVERED - do NOT use with WACC)
        self.ttm_fcff = DataQualityMetadata()  # Proper FCFF = EBIT(1-t) + D&A - CapEx - ΔNWC
        self.ttm_interest_paid = DataQualityMetadata()  # For FCFF proxy unlevering
        
        # TTM/LTM Income statement
        self.ttm_revenue = DataQualityMetadata()
        self.ttm_operating_income = DataQualityMetadata()  # EBIT
        self.ttm_net_income = DataQualityMetadata()
        self.ttm_ebitda = DataQualityMetadata()
        self.ttm_depreciation_amortization = DataQualityMetadata()
        self.ttm_interest_expense = DataQualityMetadata()  # From income statement (for after-tax add-back)
        
        # Tax rate (for FCFF)
        self.effective_tax_rate = DataQualityMetadata()
        self.tax_rate_override = None
        
        # Beta and WACC components
        self.beta = DataQualityMetadata()
        self.suggested_wacc = DataQualityMetadata()
        self.suggested_fcf_growth = DataQualityMetadata()
        self.analyst_revenue_estimates = []  # List of {year_label, revenue, source, reliability_score}
        self.analyst_long_term_growth = DataQualityMetadata()  # Analyst long-term growth (e.g., +5Y consensus)

        # Latest annual
        self.latest_annual_date = None
        self.latest_annual_diluted_shares = DataQualityMetadata()
        
        # Quarterly availability
        self.quarterly_history = []  # List of quarterly snapshots for trend analysis
        self.num_quarters_available = 0
        
        # Warnings/errors collected during fetch
        self.warnings = []
        self.errors = []
        
        # Overall quality score (0-100)
        self.overall_quality_score = 100
    
    def add_warning(self, code: str, message: str, severity="warn"):
        """Log a warning during data fetch."""
        self.warnings.append({
            "code": code,
            "message": message,
            "severity": severity  # "warn" or "error"
        })
    
    def recalculate_overall_quality(self):
        """Average reliability scores of key inputs."""
        key_fields = [
            self.price.reliability_score,
            self.shares_outstanding.reliability_score,
            self.ttm_revenue.reliability_score,
            self.ttm_fcf.reliability_score,
            self.ttm_operating_income.reliability_score
        ]
        valid_scores = [s for s in key_fields if s is not None]
        if valid_scores:
            self.overall_quality_score = sum(valid_scores) / len(valid_scores)
        else:
            self.overall_quality_score = 0
    
    def to_dict(self):
        """Serialize to dict for JSON logging/trace."""
        return {
            "ticker": self.ticker,
            "retrieved_at": self.retrieved_at,
            "currency": self.currency,
            "company_name": self.company_name,
            "sector": self.sector,
            "industry": self.industry,
            "price": self.price.to_dict(),
            "shares_outstanding": self.shares_outstanding.to_dict(),
            "market_cap": self.market_cap.to_dict(),
            "total_debt": self.total_debt.to_dict(),
            "cash_and_equivalents": self.cash_and_equivalents.to_dict(),
            "ttm_revenue": self.ttm_revenue.to_dict(),
            "ttm_operating_cash_flow": self.ttm_operating_cash_flow.to_dict(),
            "ttm_capex": self.ttm_capex.to_dict(),
            "ttm_fcf": self.ttm_fcf.to_dict(),
            "ttm_ebitda": self.ttm_ebitda.to_dict(),
            "ttm_operating_income": self.ttm_operating_income.to_dict(),
            "ttm_net_income": self.ttm_net_income.to_dict(),
            "effective_tax_rate": self.effective_tax_rate.to_dict(),
            "beta": self.beta.to_dict(),
            "suggested_wacc": self.suggested_wacc.to_dict(),
            "suggested_fcf_growth": self.suggested_fcf_growth.to_dict(),
            "analyst_revenue_estimates": self.analyst_revenue_estimates,
            "analyst_long_term_growth": self.analyst_long_term_growth.to_dict(),
            "num_quarters_available": self.num_quarters_available,
            "overall_quality_score": self.overall_quality_score,
            "warnings": self.warnings,
            "errors": self.errors
        }


class DataAdapter:
    """Fetches and normalizes yfinance data with quality tracking."""
    
    def __init__(self, ticker: str):
        self.ticker = ticker.upper()
        self.snapshot = NormalizedFinancialSnapshot(self.ticker)
    
    def fetch(self) -> NormalizedFinancialSnapshot:
        """Fetch all available data from yfinance."""
        try:
            stock = yf.Ticker(self.ticker)
            
            # Step 1: Price and shares data
            self._fetch_price_and_shares(stock)
            
            # Step 2: Balance sheet data (debt, cash)
            self._fetch_balance_sheet(stock)
            
            # Step 3: Cash flow data (TTM FCF)
            self._fetch_cash_flow(stock)
            
            # Step 4: Income statement data (Revenue, EBITDA, etc)
            self._fetch_income_statement(stock)
            
            # Step 5: Quarterly history for trends
            self._fetch_quarterly_history(stock)
            
            # Step 6: Fetch analyst revenue and long-term growth consensus
            self._fetch_analyst_revenue_estimates(stock)

            # Step 7: Calculate suggested WACC and FCF growth
            # (can use analyst forecasts fetched above)
            self._calculate_suggested_assumptions()
            
        except Exception as e:
            self.snapshot.add_warning(
                "FETCH_ERROR",
                f"Failed to fetch data for {self.ticker}: {str(e)}",
                severity="error"
            )
        
        self.snapshot.recalculate_overall_quality()
        return self.snapshot
    
    def _fetch_price_and_shares(self, stock):
        """Fetch current price, market cap, shares outstanding, and company classification."""
        try:
            info = stock.info
            fast_info = stock.fast_info
            
            # Store info for later use (e.g., analyst estimates)
            self._ticker_info = info
            
            # Company classification (sector, industry, name)
            self.snapshot.company_name = info.get('longName') or info.get('shortName')
            self.snapshot.sector = info.get('sector')
            self.snapshot.industry = info.get('industry')
            
            # Price
            price = info.get('currentPrice') or fast_info.get('lastPrice')
            if price:
                self.snapshot.price = DataQualityMetadata(
                    value=price,
                    units="USD",
                    period_end=datetime.utcnow().isoformat(),
                    period_type="current",
                    source_path="yf.Ticker.info['currentPrice'] or fast_info['lastPrice']",
                    retrieved_at=datetime.utcnow().isoformat(),
                    reliability_score=95
                )
            else:
                self.snapshot.price.reliability_score = 0
                self.snapshot.add_warning("NO_CURRENT_PRICE", f"No current price available for {self.ticker}")
            
            # Market Cap
            market_cap = info.get('marketCap') or fast_info.get('marketCap')
            if market_cap:
                self.snapshot.market_cap = DataQualityMetadata(
                    value=market_cap,
                    units="USD",
                    period_end=datetime.utcnow().isoformat(),
                    period_type="current",
                    source_path="yf.Ticker.info['marketCap']",
                    retrieved_at=datetime.utcnow().isoformat(),
                    reliability_score=95
                )
            else:
                self.snapshot.market_cap.reliability_score = 50
                self.snapshot.add_warning("NO_MARKET_CAP", f"Market cap missing; will compute from price × shares")
            
            # Shares Outstanding
            shares = info.get('sharesOutstanding') or fast_info.get('shares')
            if shares:
                self.snapshot.shares_outstanding = DataQualityMetadata(
                    value=shares,
                    units="shares",
                    period_end=datetime.utcnow().isoformat(),
                    period_type="current",
                    source_path="yf.Ticker.info['sharesOutstanding']",
                    retrieved_at=datetime.utcnow().isoformat(),
                    reliability_score=90
                )
            else:
                self.snapshot.shares_outstanding.reliability_score = 0
                self.snapshot.add_warning("NO_SHARES_OUTSTANDING", "Shares outstanding unavailable—cannot compute per-share values")
            
            # Diluted shares (for equity value calculations)
            diluted_shares = info.get('sharesFullyDiluted')
            if diluted_shares:
                self.snapshot.latest_annual_diluted_shares = DataQualityMetadata(
                    value=diluted_shares,
                    units="shares",
                    period_type="current",
                    source_path="yf.Ticker.info['sharesFullyDiluted']",
                    reliability_score=85
                )
            
            # Beta (for WACC calculation)
            beta = info.get('beta')
            if beta:
                self.snapshot.beta = DataQualityMetadata(
                    value=beta,
                    units="ratio",
                    period_type="5-year monthly",
                    source_path="yf.Ticker.info['beta']",
                    retrieved_at=datetime.utcnow().isoformat(),
                    reliability_score=85,
                    notes="5-year monthly beta vs S&P 500 (Yahoo Finance)"
                )
        
        except Exception as e:
            self.snapshot.add_warning("PRICE_FETCH_ERROR", f"Error fetching price/shares: {str(e)}")
    
    def _fetch_balance_sheet(self, stock):
        """Fetch debt, cash, and other balance sheet items needed for EV."""
        try:
            # Try quarterly first, fallback to annual
            balance_sheet = stock.quarterly_balance_sheet if not stock.quarterly_balance_sheet.empty else stock.balance_sheet
            
            if balance_sheet.empty:
                self.snapshot.add_warning("NO_BALANCE_SHEET", "No balance sheet data available")
                return
            
            most_recent_date = balance_sheet.columns[0]
            
            # Total Debt - PRIMARY: Use yahooquery for accurate Total Debt
            try:
                yq_ticker = YQTicker(self.ticker)
                yq_bs = yq_ticker.balance_sheet(frequency='q')
                if isinstance(yq_bs, pd.DataFrame) and 'TotalDebt' in yq_bs.columns:
                    # Get most recent value
                    yq_bs_sorted = yq_bs.sort_values('asOfDate', ascending=False)
                    yq_total_debt = yq_bs_sorted['TotalDebt'].iloc[0]
                    yq_as_of = yq_bs_sorted['asOfDate'].iloc[0]
                    if pd.notna(yq_total_debt) and yq_total_debt > 0:
                        self.snapshot.total_debt = DataQualityMetadata(
                            value=yq_total_debt,
                            units="USD",
                            period_end=str(yq_as_of)[:10],
                            period_type="quarterly",
                            source_path="yahooquery_balance_sheet['TotalDebt']",
                            retrieved_at=datetime.utcnow().isoformat(),
                            reliability_score=98
                        )
                    else:
                        raise ValueError("TotalDebt is NaN or zero")
                else:
                    raise ValueError("yahooquery TotalDebt unavailable")
            except Exception as yq_err:
                # FALLBACK: Sum debt line items from yfinance
                debt_line_items = ['Long Term Debt', 'Current Portion Of Long Term Debt', 'Short Term Borrowings', 'Total Debt']
                total_debt = 0
                debt_sources = []
                
                # First try 'Total Debt' directly
                if 'Total Debt' in balance_sheet.index:
                    val = balance_sheet.loc['Total Debt', most_recent_date]
                    if pd.notna(val) and val > 0:
                        total_debt = val
                        debt_sources = ['Total Debt']
                
                # If not found, sum components
                if total_debt == 0:
                    for item in ['Long Term Debt', 'Current Portion Of Long Term Debt', 'Short Term Borrowings']:
                        if item in balance_sheet.index:
                            val = balance_sheet.loc[item, most_recent_date]
                            if pd.notna(val) and val > 0:
                                total_debt += val
                                debt_sources.append(item)
                
                if total_debt > 0:
                    self.snapshot.total_debt = DataQualityMetadata(
                        value=total_debt,
                        units="USD",
                        period_end=str(most_recent_date)[:10],
                        period_type="quarterly" if "quarterly" in str(type(stock.quarterly_balance_sheet)) else "annual",
                        source_path=f"balance_sheet[{', '.join(debt_sources)}]",
                        retrieved_at=datetime.utcnow().isoformat(),
                        reliability_score=80,
                        notes=f"Fallback from yahooquery: {str(yq_err)}"
                    )
                else:
                    self.snapshot.total_debt.reliability_score = 30
                    self.snapshot.add_warning("LOW_DEBT_QUALITY", f"Total debt = 0 or missing")
            
            # Cash and Equivalents
            cash_items = ['Cash And Cash Equivalents', 'Cash', 'CashCash Equivalents And Marketable Securities']
            cash = 0
            cash_sources = []
            for item in cash_items:
                if item in balance_sheet.index:
                    val = balance_sheet.loc[item, most_recent_date]
                    if pd.notna(val):
                        cash = val
                        cash_sources.append(item)
                        break
            
            if cash > 0:
                self.snapshot.cash_and_equivalents = DataQualityMetadata(
                    value=cash,
                    units="USD",
                    period_end=str(most_recent_date)[:10],
                    period_type="quarterly",
                    source_path=f"balance_sheet[{cash_sources[0] if cash_sources else 'Cash'}]",
                    retrieved_at=datetime.utcnow().isoformat(),
                    reliability_score=90
                )
            else:
                self.snapshot.cash_and_equivalents.reliability_score = 50
                self.snapshot.add_warning("NO_CASH_DATA", "Cash and equivalents missing or zero")
            
            # ===== WORKING CAPITAL CALCULATION =====
            # Current Assets (excluding cash) and Current Liabilities (excluding short-term debt)
            # NWC = (Current Assets - Cash) - (Current Liabilities - Short-term Debt)
            try:
                current_assets = 0
                current_liabilities = 0
                short_term_debt = 0
                
                # Get current assets
                for item in ['Current Assets', 'Total Current Assets']:
                    if item in balance_sheet.index:
                        val = balance_sheet.loc[item, most_recent_date]
                        if pd.notna(val):
                            current_assets = val
                            break
                
                # Get current liabilities
                for item in ['Current Liabilities', 'Total Current Liabilities']:
                    if item in balance_sheet.index:
                        val = balance_sheet.loc[item, most_recent_date]
                        if pd.notna(val):
                            current_liabilities = val
                            break
                
                # Get short-term debt (to exclude from operating liabilities)
                for item in ['Current Portion Of Long Term Debt', 'Short Term Borrowings', 'Current Debt']:
                    if item in balance_sheet.index:
                        val = balance_sheet.loc[item, most_recent_date]
                        if pd.notna(val):
                            short_term_debt += val
                
                # Operating NWC = (CA - Cash) - (CL - Short-term debt)
                if current_assets > 0 and current_liabilities > 0:
                    operating_ca = current_assets - cash
                    operating_cl = current_liabilities - short_term_debt
                    nwc = operating_ca - operating_cl
                    
                    self.snapshot.current_assets = DataQualityMetadata(
                        value=operating_ca, units="USD", period_type="quarterly",
                        source_path="balance_sheet[Current Assets] - Cash",
                        reliability_score=85
                    )
                    self.snapshot.current_liabilities = DataQualityMetadata(
                        value=operating_cl, units="USD", period_type="quarterly",
                        source_path="balance_sheet[Current Liabilities] - ST Debt",
                        reliability_score=85
                    )
                    self.snapshot.net_working_capital = DataQualityMetadata(
                        value=nwc, units="USD", period_type="quarterly",
                        source_path="(CA - Cash) - (CL - ST Debt)",
                        reliability_score=85
                    )
                    
                    # Try to get prior period NWC for delta calculation
                    if len(balance_sheet.columns) >= 2:
                        prior_date = balance_sheet.columns[1]
                        prior_ca = prior_cl = prior_std = prior_cash = 0
                        
                        for item in ['Current Assets', 'Total Current Assets']:
                            if item in balance_sheet.index:
                                val = balance_sheet.loc[item, prior_date]
                                if pd.notna(val):
                                    prior_ca = val
                                    break
                        for item in ['Current Liabilities', 'Total Current Liabilities']:
                            if item in balance_sheet.index:
                                val = balance_sheet.loc[item, prior_date]
                                if pd.notna(val):
                                    prior_cl = val
                                    break
                        for item in ['Current Portion Of Long Term Debt', 'Short Term Borrowings', 'Current Debt']:
                            if item in balance_sheet.index:
                                val = balance_sheet.loc[item, prior_date]
                                if pd.notna(val):
                                    prior_std += val
                        for item in cash_items:
                            if item in balance_sheet.index:
                                val = balance_sheet.loc[item, prior_date]
                                if pd.notna(val):
                                    prior_cash = val
                                    break
                        
                        if prior_ca > 0 and prior_cl > 0:
                            prior_nwc = (prior_ca - prior_cash) - (prior_cl - prior_std)
                            delta_nwc = nwc - prior_nwc
                            
                            self.snapshot.prior_net_working_capital = DataQualityMetadata(
                                value=prior_nwc, units="USD", period_type="quarterly",
                                source_path="Prior quarter NWC",
                                reliability_score=80
                            )
                            self.snapshot.delta_nwc = DataQualityMetadata(
                                value=delta_nwc, units="USD", period_type="quarterly",
                                source_path="Current NWC - Prior NWC",
                                reliability_score=80,
                                notes=f"Quarterly change: ${delta_nwc/1e9:.2f}B"
                            )
            except Exception as nwc_err:
                self.snapshot.add_warning("NWC_CALC_ERROR", f"Could not compute NWC: {str(nwc_err)}")
        
        except Exception as e:
            self.snapshot.add_warning("BALANCE_SHEET_ERROR", f"Error parsing balance sheet: {str(e)}")
    
    def _fetch_cash_flow(self, stock):
        """Fetch operating cash flow and capex for TTM FCF."""
        try:
            # Prefer quarterly for TTM calculation
            quarterly_cf = stock.quarterly_cashflow
            annual_cf = stock.cashflow
            
            # Build TTM CFO
            ttm_cfo = None
            quarterly_cfo_list = []
            
            if not quarterly_cf.empty and 'Operating Cash Flow' in quarterly_cf.index:
                # Sum last 4 quarters
                cfo_series = quarterly_cf.loc['Operating Cash Flow']
                if len(cfo_series) >= 4:
                    ttm_cfo = cfo_series.iloc[:4].sum()
                    quarterly_cfo_list = cfo_series.iloc[:4].tolist()
                    period_type = "quarterly_ttm"
                    fallback_reason = None
            
            # Fallback to annual
            if ttm_cfo is None and not annual_cf.empty and 'Operating Cash Flow' in annual_cf.index:
                ttm_cfo = annual_cf.loc['Operating Cash Flow', annual_cf.columns[0]]
                period_type = "annual_proxy"
                fallback_reason = "No quarterly CFO; used last annual CFO as proxy"
            
            if ttm_cfo is not None and pd.notna(ttm_cfo):
                self.snapshot.ttm_operating_cash_flow = DataQualityMetadata(
                    value=ttm_cfo,
                    units="USD",
                    period_end="TTM",
                    period_type=period_type,
                    source_path="cash_flow['Operating Cash Flow'] (sum of last 4 Q or last annual)",
                    retrieved_at=datetime.utcnow().isoformat(),
                    reliability_score=90 if period_type == "quarterly_ttm" else 70,
                    fallback_reason=fallback_reason
                )
            else:
                self.snapshot.ttm_operating_cash_flow.reliability_score = 0
                self.snapshot.add_warning("NO_OCF", "Operating cash flow data unavailable")
            
            # Build TTM CapEx
            ttm_capex = None
            quarterly_capex_list = []
            
            if not quarterly_cf.empty and 'Capital Expenditure' in quarterly_cf.index:
                capex_series = quarterly_cf.loc['Capital Expenditure']
                if len(capex_series) >= 4:
                    # CapEx is negative; take absolute value
                    ttm_capex = abs(capex_series.iloc[:4].sum())
                    quarterly_capex_list = [abs(x) for x in capex_series.iloc[:4].tolist()]
                    period_type = "quarterly_ttm"
                    fallback_reason = None
            
            # Fallback to annual
            if ttm_capex is None and not annual_cf.empty and 'Capital Expenditure' in annual_cf.index:
                ttm_capex = abs(annual_cf.loc['Capital Expenditure', annual_cf.columns[0]])
                period_type = "annual_proxy"
                fallback_reason = "No quarterly CapEx; used last annual CapEx as proxy"
            
            if ttm_capex is not None and pd.notna(ttm_capex):
                self.snapshot.ttm_capex = DataQualityMetadata(
                    value=ttm_capex,
                    units="USD",
                    period_end="TTM",
                    period_type=period_type,
                    source_path="cash_flow['Capital Expenditure'] (sum of last 4 Q, absolute value, or last annual)",
                    retrieved_at=datetime.utcnow().isoformat(),
                    reliability_score=90 if period_type == "quarterly_ttm" else 70,
                    fallback_reason=fallback_reason
                )
            else:
                self.snapshot.ttm_capex.reliability_score = 0
                self.snapshot.add_warning("NO_CAPEX", "Capital expenditure data unavailable; cannot compute FCF proxy")
            
            # Compute TTM FCF = CFO - CapEx (LEVERED - includes interest, embedded ΔNWC)
            if (self.snapshot.ttm_operating_cash_flow.value is not None and 
                self.snapshot.ttm_capex.value is not None):
                ttm_fcf = self.snapshot.ttm_operating_cash_flow.value - self.snapshot.ttm_capex.value
                self.snapshot.ttm_fcf = DataQualityMetadata(
                    value=ttm_fcf,
                    units="USD",
                    period_end="TTM",
                    period_type="ttm_proxy",
                    source_path="TTM_OCF - TTM_CapEx",
                    retrieved_at=datetime.utcnow().isoformat(),
                    reliability_score=min(
                        self.snapshot.ttm_operating_cash_flow.reliability_score,
                        self.snapshot.ttm_capex.reliability_score
                    ),
                    is_estimated=False,
                    notes="LEVERED PROXY: CFO - CapEx. CFO already includes interest paid and ΔNWC."
                )
            else:
                self.snapshot.ttm_fcf.reliability_score = 0
                self.snapshot.add_warning("NO_FCF", "Cannot compute TTM FCF: OCF or CapEx missing")
            
            # Fetch TTM Change in Working Capital from cash flow statement
            # CRITICAL: This is the proper TTM ΔNWC, NOT quarter-over-quarter balance sheet delta
            if not quarterly_cf.empty and 'Change In Working Capital' in quarterly_cf.index:
                delta_nwc_series = quarterly_cf.loc['Change In Working Capital']
                if len(delta_nwc_series) >= 4:
                    # Sum last 4 quarters for TTM ΔNWC
                    # Note: This is a USE of cash when positive (inventory build, receivables grow)
                    # and a SOURCE when negative
                    ttm_delta_nwc = delta_nwc_series.iloc[:4].sum()
                    if pd.notna(ttm_delta_nwc):
                        self.snapshot.ttm_delta_nwc = DataQualityMetadata(
                            value=ttm_delta_nwc,
                            units="USD",
                            period_end="TTM",
                            period_type="quarterly_ttm",
                            source_path="cash_flow['Change In Working Capital'] (sum of last 4 Q)",
                            retrieved_at=datetime.utcnow().isoformat(),
                            reliability_score=90,
                            notes=f"TTM ΔNWC from CF statement: ${ttm_delta_nwc/1e9:.2f}B (already in CFO)"
                        )
        
        except Exception as e:
            self.snapshot.add_warning("CASH_FLOW_ERROR", f"Error parsing cash flow: {str(e)}")
    
    def _fetch_income_statement(self, stock):
        """Fetch revenue, EBITDA, operating income, net income, tax rate."""
        try:
            quarterly_is = stock.quarterly_income_stmt
            annual_is = stock.income_stmt
            
            # Helper to fetch from quarterly first, then annual
            def fetch_line_item(line_item_name, quarterly=True):
                val = None
                source = None
                period_type = None
                if quarterly and not quarterly_is.empty and line_item_name in quarterly_is.index:
                    series = quarterly_is.loc[line_item_name]
                    if len(series) >= 4:
                        val = series.iloc[:4].sum()
                        source = f"quarterly_{line_item_name}"
                        period_type = "quarterly_ttm"
                if val is None and not annual_is.empty and line_item_name in annual_is.index:
                    val = annual_is.loc[line_item_name, annual_is.columns[0]]
                    source = f"annual_{line_item_name}"
                    period_type = "annual_proxy"
                return val, source, period_type
            
            # TTM Revenue
            ttm_rev, rev_source, rev_period = fetch_line_item('Total Revenue')
            if ttm_rev is not None and pd.notna(ttm_rev):
                self.snapshot.ttm_revenue = DataQualityMetadata(
                    value=ttm_rev,
                    units="USD",
                    period_end="TTM",
                    period_type=rev_period,
                    source_path=rev_source,
                    retrieved_at=datetime.utcnow().isoformat(),
                    reliability_score=95 if rev_period == "quarterly_ttm" else 80,
                )
            else:
                self.snapshot.ttm_revenue.reliability_score = 0
                self.snapshot.add_warning("NO_REVENUE", "Total revenue data unavailable")
            
            # TTM Operating Income
            ttm_oi, oi_source, oi_period = fetch_line_item('Operating Income')
            if ttm_oi is not None and pd.notna(ttm_oi):
                self.snapshot.ttm_operating_income = DataQualityMetadata(
                    value=ttm_oi,
                    units="USD",
                    period_end="TTM",
                    period_type=oi_period,
                    source_path=oi_source,
                    retrieved_at=datetime.utcnow().isoformat(),
                    reliability_score=90 if oi_period == "quarterly_ttm" else 75
                )
            
            # TTM Net Income
            ttm_ni, ni_source, ni_period = fetch_line_item('Net Income')
            if ttm_ni is not None and pd.notna(ttm_ni):
                self.snapshot.ttm_net_income = DataQualityMetadata(
                    value=ttm_ni,
                    units="USD",
                    period_end="TTM",
                    period_type=ni_period,
                    source_path=ni_source,
                    retrieved_at=datetime.utcnow().isoformat(),
                    reliability_score=90 if ni_period == "quarterly_ttm" else 75
                )
            
            # TTM EBITDA - PRIMARY: Use yahooquery to get real TTM from Yahoo Finance
            try:
                yq_ticker = YQTicker(self.ticker)
                inc_stmt = yq_ticker.income_statement(frequency='q', trailing=True)
                if isinstance(inc_stmt, pd.DataFrame) and 'EBITDA' in inc_stmt.columns and 'periodType' in inc_stmt.columns:
                    ttm_rows = inc_stmt[inc_stmt['periodType'] == 'TTM']
                    if not ttm_rows.empty:
                        # Get most recent TTM row
                        ttm_rows_sorted = ttm_rows.sort_values('asOfDate', ascending=False)
                        ttm_ebitda_yq = ttm_rows_sorted['EBITDA'].iloc[0]
                        as_of_date = ttm_rows_sorted['asOfDate'].iloc[0]
                        if pd.notna(ttm_ebitda_yq) and ttm_ebitda_yq > 0:
                            self.snapshot.ttm_ebitda = DataQualityMetadata(
                                value=ttm_ebitda_yq,
                                units="USD",
                                period_end=str(as_of_date),
                                period_type="ttm",
                                source_path="yahooquery_income_statement_TTM",
                                retrieved_at=datetime.utcnow().isoformat(),
                                reliability_score=98  # Highest reliability - actual TTM from Yahoo
                            )
                        else:
                            raise ValueError("TTM EBITDA is NaN or zero")
                    else:
                        raise ValueError("No TTM rows found")
                else:
                    raise ValueError("yahooquery response invalid")
            except Exception as yq_err:
                # FALLBACK 1: Use yfinance info['ebitda'] (Statistics page value)
                info_ebitda = stock.info.get('ebitda')
                if info_ebitda is not None and info_ebitda > 0:
                    self.snapshot.ttm_ebitda = DataQualityMetadata(
                        value=info_ebitda,
                        units="USD",
                        period_end="TTM",
                        period_type="ttm",
                        source_path="yfinance_info['ebitda']",
                        retrieved_at=datetime.utcnow().isoformat(),
                        reliability_score=85,
                        notes=f"Fallback from yahooquery: {str(yq_err)}"
                    )
                else:
                    # FALLBACK 2: Try income statement EBITDA line item
                    ttm_ebitda, ebitda_source, ebitda_period = fetch_line_item('EBITDA')
                    if ttm_ebitda is not None and pd.notna(ttm_ebitda):
                        self.snapshot.ttm_ebitda = DataQualityMetadata(
                            value=ttm_ebitda,
                            units="USD",
                            period_end="TTM",
                            period_type=ebitda_period,
                            source_path=ebitda_source,
                            retrieved_at=datetime.utcnow().isoformat(),
                            reliability_score=75 if ebitda_period == "quarterly_ttm" else 65
                        )
                    else:
                        # FALLBACK 3: Approximate EBITDA from Operating Income + D&A
                        ttm_da, da_source, da_period = fetch_line_item('Depreciation And Amortization')
                        if ttm_oi is not None and ttm_da is not None:
                            approx_ebitda = ttm_oi + ttm_da
                            self.snapshot.ttm_ebitda = DataQualityMetadata(
                                value=approx_ebitda,
                                units="USD",
                                period_end="TTM",
                                period_type=da_period,
                                source_path="OI + D&A",
                                retrieved_at=datetime.utcnow().isoformat(),
                                reliability_score=55,
                                is_estimated=True,
                                notes="Approximated from Operating Income + D&A; direct EBITDA unavailable"
                            )
                        else:
                            self.snapshot.ttm_ebitda.reliability_score = 0
                            self.snapshot.add_warning("NO_EBITDA", "EBITDA unavailable; cannot estimate from OI+D&A either")
            
            # Effective tax rate - use Tax Provision / Pretax Income (most accurate)
            tax_rate_set = False
            if not annual_is.empty:
                if 'Tax Provision' in annual_is.index and 'Pretax Income' in annual_is.index:
                    latest_col = annual_is.columns[0]
                    tax_provision = annual_is.loc['Tax Provision', latest_col]
                    pretax_income = annual_is.loc['Pretax Income', latest_col]
                    if pretax_income is not None and pretax_income > 0 and tax_provision is not None:
                        effective_rate = tax_provision / pretax_income
                        effective_rate = max(0, min(0.50, effective_rate))  # Clamp to 0-50%
                        # Get the period date
                        period_date = latest_col.strftime("%Y-%m-%d") if hasattr(latest_col, 'strftime') else str(latest_col)
                        self.snapshot.effective_tax_rate = DataQualityMetadata(
                            value=effective_rate,
                            units="rate",
                            period_end=period_date,
                            period_type="annual",
                            source_path="Tax Provision / Pretax Income (annual income statement)",
                            retrieved_at=datetime.utcnow().isoformat(),
                            reliability_score=90,
                            is_estimated=False,
                            notes=f"From annual filing: Tax Provision ${tax_provision/1e9:.2f}B / Pretax Income ${pretax_income/1e9:.2f}B"
                        )
                        tax_rate_set = True
            
            # Fallback: estimate from Net Income / Operating Income
            if not tax_rate_set:
                if ttm_oi is not None and ttm_ni is not None and ttm_oi > 0:
                    implied_tax_rate = max(0, min(0.50, 1.0 - (ttm_ni / ttm_oi)))
                    self.snapshot.effective_tax_rate = DataQualityMetadata(
                        value=implied_tax_rate,
                        units="rate",
                        period_end="TTM",
                        period_type="estimated",
                        source_path="1 - (Net Income / Operating Income)",
                        retrieved_at=datetime.utcnow().isoformat(),
                        reliability_score=60,
                        is_estimated=True,
                        notes="Fallback: Inferred from NI/OI; Tax Provision data unavailable"
                    )
                else:
                    self.snapshot.effective_tax_rate.value = 0.25  # Default assumption
                    self.snapshot.effective_tax_rate.reliability_score = 20
            
            # TTM Interest Expense (for FCFF proxy unlevering)
            # Use "Interest Expense" or "Interest Expense Non Operating"
            ttm_int_exp = None
            for int_item in ['Interest Expense', 'Interest Expense Non Operating']:
                int_val, int_source, int_period = fetch_line_item(int_item)
                if int_val is not None and pd.notna(int_val) and int_val > 0:
                    ttm_int_exp = int_val
                    self.snapshot.ttm_interest_expense = DataQualityMetadata(
                        value=ttm_int_exp,
                        units="USD",
                        period_end="TTM",
                        period_type=int_period,
                        source_path=int_source,
                        retrieved_at=datetime.utcnow().isoformat(),
                        reliability_score=85 if int_period == "quarterly_ttm" else 70,
                        notes=f"TTM Interest Expense: ${ttm_int_exp/1e9:.2f}B (for FCFF proxy)"
                    )
                    break
        
        except Exception as e:
            self.snapshot.add_warning("INCOME_STMT_ERROR", f"Error parsing income statement: {str(e)}")
    
    def _fetch_quarterly_history(self, stock):
        """Fetch quarterly history for trend analysis."""
        try:
            quarterly_is = stock.quarterly_income_stmt
            quarterly_cf = stock.quarterly_cashflow
            
            if quarterly_is.empty:
                self.snapshot.num_quarters_available = 0
                return
            
            self.snapshot.num_quarters_available = len(quarterly_is.columns)
            
            # Store a few recent quarters for trend analysis
            for i, date in enumerate(quarterly_is.columns[:8]):  # Last 8 quarters
                q_data = {
                    "date": str(date)[:10],
                    "revenue": quarterly_is.loc['Total Revenue', date] if 'Total Revenue' in quarterly_is.index else None,
                    "operating_income": quarterly_is.loc['Operating Income', date] if 'Operating Income' in quarterly_is.index else None,
                    "net_income": quarterly_is.loc['Net Income', date] if 'Net Income' in quarterly_is.index else None
                }
                self.snapshot.quarterly_history.append(q_data)
        
        except Exception as e:
            self.snapshot.add_warning("QUARTERLY_HISTORY_ERROR", f"Error fetching quarterly history: {str(e)}")
    
    def _calculate_suggested_assumptions(self):
        """Calculate suggested WACC and FCF growth rate based on company data."""
        from datetime import datetime
        
        # ═══════════════════════════════════════════════════════════════
        # SUGGESTED WACC using CAPM: WACC = Rf + Beta × (Rm - Rf)
        # ═══════════════════════════════════════════════════════════════
        # Dynamic risk-free rate from 10-year Treasury (^TNX)
        # Market risk premium from Damodaran (updated annually)
        
        # Get live 10-year Treasury yield from Yahoo Finance
        try:
            tnx = yf.Ticker("^TNX")
            tnx_hist = tnx.history(period="5d")
            if not tnx_hist.empty:
                RISK_FREE_RATE = tnx_hist['Close'].iloc[-1] / 100  # Convert from % to decimal
                RF_SOURCE = f"^TNX live ({tnx_hist.index[-1].strftime('%Y-%m-%d')})"
            else:
                RISK_FREE_RATE = 0.045  # Fallback
                RF_SOURCE = "Fallback (^TNX unavailable)"
        except Exception as e:
            RISK_FREE_RATE = 0.045  # Fallback if ^TNX fetch fails
            RF_SOURCE = f"Fallback (^TNX error: {str(e)[:30]})"
        
        MARKET_RISK_PREMIUM = 0.05  # Damodaran implied ERP (5.0%)
        DAMODARAN_DATE = "January 2026"  # Update date
        
        # Store for UI display
        self.snapshot.risk_free_rate = RISK_FREE_RATE
        self.snapshot.rf_source = RF_SOURCE
        
        beta = self.snapshot.beta.value
        if beta is not None and beta > 0:
            # CAPM: Cost of Equity = Rf + Beta × Market Risk Premium
            cost_of_equity = RISK_FREE_RATE + beta * MARKET_RISK_PREMIUM
            
            # For simplicity, use cost of equity as WACC proxy
            # (Most large-cap tech has minimal debt impact on WACC)
            suggested_wacc = cost_of_equity
            
            # Adjust for company size/risk (add small premium for smaller caps)
            ttm_rev = self.snapshot.ttm_revenue.value
            if ttm_rev is not None:
                if ttm_rev < 10e9:  # Small cap
                    suggested_wacc += 0.02
                elif ttm_rev < 50e9:  # Mid cap
                    suggested_wacc += 0.01
            
            # Clamp to reasonable range
            suggested_wacc = max(0.06, min(0.15, suggested_wacc))
            
            self.snapshot.suggested_wacc = DataQualityMetadata(
                value=suggested_wacc,
                units="rate",
                period_type="calculated",
                source_path="CAPM: Rf + Beta × MRP",
                retrieved_at=datetime.utcnow().isoformat(),
                reliability_score=80,
                is_estimated=True,
                notes=f"CAPM: {RISK_FREE_RATE*100:.2f}% + {beta:.2f} × {MARKET_RISK_PREMIUM*100:.1f}% = {cost_of_equity*100:.2f}%. Rf from {RF_SOURCE}, MRP from Damodaran ({DAMODARAN_DATE})"
            )
        else:
            # Fallback: size-based default
            ttm_rev = self.snapshot.ttm_revenue.value
            if ttm_rev is not None and ttm_rev > 50e9:
                suggested_wacc = 0.09
            elif ttm_rev is not None and ttm_rev > 10e9:
                suggested_wacc = 0.10
            else:
                suggested_wacc = 0.11
            
            self.snapshot.suggested_wacc = DataQualityMetadata(
                value=suggested_wacc,
                units="rate",
                period_type="default",
                source_path="Size-based default (beta unavailable)",
                retrieved_at=datetime.utcnow().isoformat(),
                reliability_score=50,
                is_estimated=True,
                notes="Fallback: Beta unavailable, using size-based industry average"
            )
        
        # ═══════════════════════════════════════════════════════════════
        # SUGGESTED FCF GROWTH RATE PRIORITY
        # 1) Analyst forward consensus (revenue estimate chain; no LT blending)
        # 2) Yahoo trailing growth
        # 3) Historical YoY fallback
        # 4) Default
        # ═══════════════════════════════════════════════════════════════

        def _period_rank(label: str):
            if label == "0y":
                return 0
            if isinstance(label, str) and label.startswith("+") and label.endswith("y"):
                num = label[1:-1]
                if num.isdigit():
                    return int(num)
            return 999

        analyst_estimates = sorted(
            self.snapshot.analyst_revenue_estimates or [],
            key=lambda x: _period_rank(x.get("year_label"))
        )
        analyst_lt_growth = getattr(self.snapshot.analyst_long_term_growth, "value", None)
        ttm_revenue = self.snapshot.ttm_revenue.value

        if analyst_estimates:
            # Build an annualized forward growth chain from consensus estimates only:
            # 0y->+1y, +1y->+2y, etc. This avoids mixing in potentially misaligned TTM->0y jumps.
            ranked_estimates = []
            for est in analyst_estimates:
                label = est.get("year_label")
                rank = _period_rank(label)
                rev = est.get("revenue")
                try:
                    rev = float(rev) if rev is not None else None
                except Exception:
                    rev = None
                if rank != 999 and rev is not None and rev > 0:
                    ranked_estimates.append((rank, label, rev))
            ranked_estimates.sort(key=lambda x: x[0])

            forward_growth_candidates = []
            forward_chain_labels = []
            for i in range(1, len(ranked_estimates)):
                prev_rank, prev_label, prev_rev = ranked_estimates[i - 1]
                curr_rank, curr_label, curr_rev = ranked_estimates[i]
                if prev_rev > 0 and curr_rev > 0 and curr_rank > prev_rank:
                    year_span = max(1, curr_rank - prev_rank)
                    annualized_growth = (curr_rev / prev_rev) ** (1.0 / year_span) - 1
                    forward_growth_candidates.append(annualized_growth)
                    forward_chain_labels.append(f"{prev_label}->{curr_label}")

            suggested_fcf_growth = None
            reliability = 0
            source_note = ""

            if forward_growth_candidates:
                suggested_fcf_growth = sum(forward_growth_candidates) / len(forward_growth_candidates)
                reliability = 88 if len(forward_growth_candidates) >= 2 else 82
                source_note = ", ".join(forward_chain_labels)
            else:
                # Single-point fallback if only one forward estimate is available.
                if ranked_estimates and ttm_revenue and ttm_revenue > 0:
                    _, label0, rev0 = ranked_estimates[0]
                    suggested_fcf_growth = (rev0 / ttm_revenue) - 1
                    reliability = 72
                    source_note = f"TTM->{label0}"

            if suggested_fcf_growth is not None:
                suggested_fcf_growth = max(0.00, min(0.35, suggested_fcf_growth))
                lt_text = f" (LT anchor available: {analyst_lt_growth*100:.1f}%, used in fade, not in near-term suggestion)" if analyst_lt_growth is not None else ""

                self.snapshot.suggested_fcf_growth = DataQualityMetadata(
                    value=suggested_fcf_growth,
                    units="rate",
                    period_type="forward_analyst_consensus",
                    source_path="yf.Ticker.revenue_estimate (forward consensus chain)",
                    retrieved_at=datetime.utcnow().isoformat(),
                    reliability_score=reliability,
                    is_estimated=True,
                    notes=(
                        f"Analyst forward revenue chain: {source_note}. "
                        f"Suggested near-term growth = {suggested_fcf_growth*100:.1f}%."
                        f"{lt_text}"
                    )
                )
                return

        if analyst_lt_growth is not None and analyst_lt_growth > 0:
            suggested_fcf_growth = max(0.00, min(0.35, analyst_lt_growth))
            self.snapshot.suggested_fcf_growth = DataQualityMetadata(
                value=suggested_fcf_growth,
                units="rate",
                period_type="forward_analyst_long_term",
                source_path=self.snapshot.analyst_long_term_growth.source_path,
                retrieved_at=datetime.utcnow().isoformat(),
                reliability_score=max(70, self.snapshot.analyst_long_term_growth.reliability_score or 70),
                is_estimated=True,
                notes=f"Analyst long-term growth anchor used as fallback: {suggested_fcf_growth*100:.1f}%."
            )
            return

        # Priority 2: Yahoo trailing revenue growth
        yf_revenue_growth = getattr(self, '_ticker_info', {}).get('revenueGrowth')
        if yf_revenue_growth is not None and yf_revenue_growth > 0:
            suggested_fcf_growth = max(0.03, min(0.30, yf_revenue_growth))

            self.snapshot.suggested_fcf_growth = DataQualityMetadata(
                value=suggested_fcf_growth,
                units="rate",
                period_type="trailing_historical",
                source_path="yf.Ticker.info['revenueGrowth']",
                retrieved_at=datetime.utcnow().isoformat(),
                reliability_score=80,
                is_estimated=True,
                notes=f"Yahoo Finance trailing revenue growth: {suggested_fcf_growth*100:.1f}% (historical fallback)"
            )
            return

        # Priority 3: Historical YoY fallback (less reliable - only 5 quarters)
        quarterly_history = self.snapshot.quarterly_history
        if len(quarterly_history) >= 5:
            revenues = [q['revenue'] for q in quarterly_history[:5] if q.get('revenue')]
            if len(revenues) >= 5:
                latest_rev = revenues[0]
                yoy_rev = revenues[4]  # 4 quarters ago
                if yoy_rev and yoy_rev > 0:
                    yoy_growth = (latest_rev / yoy_rev) - 1
                    
                    # Apply 0.7x dampening since this is unreliable single-point YoY
                    suggested_fcf_growth = yoy_growth * 0.7
                    suggested_fcf_growth = max(0.03, min(0.25, suggested_fcf_growth))  # Clamp 3-25%
                    
                    self.snapshot.suggested_fcf_growth = DataQualityMetadata(
                        value=suggested_fcf_growth,
                        units="rate",
                        period_type="calculated_fallback",
                        source_path="Calculated YoY Revenue × 0.7 (Yahoo revenueGrowth unavailable)",
                        retrieved_at=datetime.utcnow().isoformat(),
                        reliability_score=60,  # Lower reliability - only 5 quarters
                        is_estimated=True,
                        notes=f"⚠️ Fallback: Based on single YoY calc ({yoy_growth*100:.1f}% × 0.7 = {suggested_fcf_growth*100:.1f}%). Limited to 5 quarters - less reliable."
                    )
                    return
        
        # Priority 4: Default fallback
        self.snapshot.suggested_fcf_growth = DataQualityMetadata(
            value=0.08,
            units="rate",
            period_type="default",
            source_path="Industry average default",
            retrieved_at=datetime.utcnow().isoformat(),
            reliability_score=40,
            is_estimated=True,
            notes="Fallback: No analyst estimates or historical data, using 8% default"
        )

    def _fetch_analyst_revenue_estimates(self, stock):
        """Fetch analyst consensus revenue and long-term growth estimates."""

        def _safe_rate(raw):
            if raw is None:
                return None
            try:
                if isinstance(raw, str):
                    value = raw.strip()
                    is_pct = value.endswith("%")
                    if is_pct:
                        value = value[:-1]
                    parsed = float(value)
                    raw = parsed / 100.0 if is_pct else parsed
                else:
                    raw = float(raw)
                if pd.isna(raw):
                    return None
                if abs(raw) > 1.0:
                    raw = raw / 100.0
                return raw
            except Exception:
                return None

        def _period_rank(label: str):
            if label == "0y":
                return 0
            if isinstance(label, str) and label.startswith("+") and label.endswith("y"):
                num = label[1:-1]
                if num.isdigit():
                    return int(num)
            return None

        try:
            estimates = []
            reliability_map = {
                "0y": 85,
                "+1y": 80,
                "+2y": 75,
                "+3y": 70,
                "+4y": 65,
                "+5y": 60,
            }

            rev_est = stock.revenue_estimate
            if rev_est is not None and not rev_est.empty:
                labels = [idx for idx in rev_est.index if _period_rank(idx) is not None]
                labels = sorted(labels, key=lambda x: _period_rank(x))
                for label in labels:
                    row = rev_est.loc[label]
                    avg_rev = row.get("avg") if hasattr(row, "get") else row["avg"] if "avg" in row else None
                    try:
                        avg_rev = float(avg_rev)
                    except Exception:
                        avg_rev = None
                    if avg_rev is not None and not pd.isna(avg_rev) and avg_rev > 0:
                        estimates.append(
                            {
                                "year_label": label,
                                "revenue": avg_rev,
                                "source": "Yahoo Finance analyst consensus",
                                "reliability_score": reliability_map.get(label, 60),
                            }
                        )

            self.snapshot.analyst_revenue_estimates = estimates

            # Long-term analyst growth (used as a 10Y curve anchor when available)
            lt_growth = None
            lt_source = None
            lt_reliability = 0

            try:
                growth_est = stock.growth_estimates
                if growth_est is not None and isinstance(growth_est, pd.DataFrame) and not growth_est.empty and "+5y" in growth_est.columns:
                    row_key = None
                    for candidate in ["stock", self.ticker]:
                        if candidate in growth_est.index:
                            row_key = candidate
                            break
                    if row_key is None:
                        row_key = growth_est.index[0]
                    candidate_rate = _safe_rate(growth_est.loc[row_key, "+5y"])
                    if candidate_rate is not None:
                        lt_growth = candidate_rate
                        lt_source = f"yf.Ticker.growth_estimates[{row_key}]['+5y']"
                        lt_reliability = 85
            except Exception:
                pass

            if lt_growth is None:
                try:
                    earnings_trend = stock.earnings_trend
                    if earnings_trend is not None and isinstance(earnings_trend, pd.DataFrame) and not earnings_trend.empty:
                        candidate_rate = None
                        if "+5y" in earnings_trend.index:
                            row = earnings_trend.loc["+5y"]
                            if hasattr(row, "get"):
                                candidate_rate = row.get("growth")
                        elif "+5y" in earnings_trend.columns:
                            candidate_rate = earnings_trend["+5y"].iloc[0]
                        candidate_rate = _safe_rate(candidate_rate)
                        if candidate_rate is not None:
                            lt_growth = candidate_rate
                            lt_source = "yf.Ticker.earnings_trend['+5y']"
                            lt_reliability = 78
                except Exception:
                    pass

            if lt_growth is None:
                info = getattr(self, "_ticker_info", {}) or {}
                info_rate = _safe_rate(
                    info.get("longTermPotentialGrowthRate")
                    or info.get("earningsGrowth")
                )
                if info_rate is not None:
                    lt_growth = info_rate
                    lt_source = "yf.Ticker.info['longTermPotentialGrowthRate' or 'earningsGrowth']"
                    lt_reliability = 60

            if lt_growth is not None:
                lt_growth = max(-0.10, min(0.40, lt_growth))
                self.snapshot.analyst_long_term_growth = DataQualityMetadata(
                    value=lt_growth,
                    units="rate",
                    period_type="forward_long_term",
                    source_path=lt_source,
                    retrieved_at=datetime.utcnow().isoformat(),
                    reliability_score=lt_reliability,
                    notes="Long-term analyst growth estimate (used as 10Y mid-curve anchor when available).",
                )

        except Exception as e:
            self.snapshot.add_warning(
                "ANALYST_REVENUE_ESTIMATES_ERROR",
                f"Could not fetch analyst revenue/growth estimates: {str(e)[:80]}"
            )
