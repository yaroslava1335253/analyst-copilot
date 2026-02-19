# engine.py
"""
Analyst Co-Pilot Engine (Lightweight)
=====================================
Uses Google Gemini (FREE) for AI analysis.
Includes financial math logic to pre-process data for the LLM.
"""

import os
import requests
import yfinance as yf
import pandas as pd
import google.generativeai as genai
from industry_multiples import get_industry_multiple, DAMODARAN_SOURCE_URL, DAMODARAN_DATA_DATE

def config_genai():
    """Configures the Gemini API."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
    return api_key

def get_financials(ticker_symbol: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fetches the income statement, balance sheet, and cash flow statement for a given stock ticker.
    Returns: (income_stmt, balance_sheet, cash_flow, quarterly_cash_flow)
    """
    try:
        stock = yf.Ticker(ticker_symbol)
        income_statement = stock.income_stmt
        balance_sheet = stock.balance_sheet
        cash_flow = stock.cashflow
        quarterly_cash_flow = stock.quarterly_cashflow
        return income_statement, balance_sheet, cash_flow, quarterly_cash_flow
    except Exception as e:
        print(f"Error fetching data for {ticker_symbol}: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def calculate_metrics(income_stmt: pd.DataFrame, balance_sheet: pd.DataFrame) -> dict:
    """
    Calculates key financial ratios and trends.
    Returns a dictionary of results to displaying and prompting.
    """
    metrics = {}
    
    # Clean up column names (timestamps) to be strings for the LLM
    # income_stmt.columns = income_stmt.columns.astype(str)
    
    try:
        # Helper to get series safely
        def get_series(df, key):
            if key in df.index:
                return df.loc[key]
            return pd.Series(dtype='float64')

        rev = get_series(income_stmt, 'Total Revenue')
        net_income = get_series(income_stmt, 'Net Income')
        op_income = get_series(income_stmt, 'Operating Income')
        
        # Balance sheet keys might vary slightly, checking exact list from debug output
        equity_key = 'Stockholders Equity' if 'Stockholders Equity' in balance_sheet.index else 'Total Equity Gross Minority Interest'
        equity = get_series(balance_sheet, equity_key)
        
        assets = get_series(balance_sheet, 'Total Assets')
        
        # --- Growth Rates (CAGR / YoY) ---
        # Assuming strictly chronological reverse order (Newest -> Oldest) which yf provides
        if len(rev) > 1:
            recent_rev = rev.iloc[0]
            oldest_rev = rev.iloc[-1]
            years = len(rev) - 1
            if oldest_rev > 0 and years > 0:
                cagr_rev = (recent_rev / oldest_rev) ** (1/years) - 1
                metrics['Revenue CAGR (Last ~4y)'] = f"{cagr_rev:.2%}"
            else:
                 metrics['Revenue CAGR'] = "N/A"
        
        # --- Margins (Latest Year) ---
        if not rev.empty and not net_income.empty:
            latest_rev = rev.iloc[0]
            latest_ni = net_income.iloc[0]
            metrics['Latest Revenue'] = f"${latest_rev:,.0f}"
            
            if latest_rev != 0:
                metrics['Net Profit Margin'] = f"{(latest_ni / latest_rev):.2%}"
                if not op_income.empty:
                    metrics['Operating Margin'] = f"{(op_income.iloc[0] / latest_rev):.2%}"
        
        # --- DuPont Analysis (Latest Year) ---
        # ROE = (Net Income / Revenue) * (Revenue / Assets) * (Assets / Equity)
        if not assets.empty and not equity.empty and not rev.empty and not net_income.empty:
            ni = net_income.iloc[0]
            rv = rev.iloc[0]
            ast = assets.iloc[0]
            eq = equity.iloc[0]
            
            if rv != 0 and ast != 0 and eq != 0:
                npm = ni / rv  # Net Profit Margin
                fat = rv / ast  # Asset Turnover
                em = ast / eq   # Equity Multiplier
                roe = npm * fat * em
                
                metrics['DuPont ROE'] = f"{roe:.2%}"
                metrics['  - Net Profit Margin'] = f"{npm:.2%}"
                metrics['  - Asset Turnover'] = f"{fat:.2f}"
                metrics['  - Equity Multiplier'] = f"{em:.2f}"

    except Exception as e:
        metrics['Error'] = str(e)
        
    return metrics


def calculate_comprehensive_analysis(
    income_stmt: pd.DataFrame,
    balance_sheet: pd.DataFrame,
    quarterly_data: list,
    ticker_symbol: str = None,
    cash_flow: pd.DataFrame = None,
    quarterly_cash_flow: pd.DataFrame = None,
    wacc_override: float = None,
    fcf_growth_override: float = None
) -> dict:
    """
    Investment Banking-Grade Financial Analysis (Goldman/Morgan Stanley approach)
    ============================================================================
    Phase 1: Data Preparation - Annualize quarterly data, calculate actual FCF margins
    Phase 2: Valuation - 5-Year DCF with Exit Multiple terminal value
    Phase 3: Stress Test - Sanity check vs current price, reverse DCF, scenario analysis
    """
    analysis = {
        "dupont": {},
        "dcf": {},
        "quality_metrics": {},
        "trend_analysis": {},
        "scenarios": {}
    }
    
    try:
        def get_series(df, key):
            if key in df.index:
                return df.loc[key]
            return pd.Series(dtype='float64')
        
        # Get financial data
        rev = get_series(income_stmt, 'Total Revenue')
        net_income = get_series(income_stmt, 'Net Income')
        op_income = get_series(income_stmt, 'Operating Income')
        fcf = get_series(income_stmt, 'Free Cash Flow')
        ebitda = get_series(income_stmt, 'ebitda')
        
        equity_key = 'Stockholders Equity' if 'Stockholders Equity' in balance_sheet.index else 'Total Equity Gross Minority Interest'
        equity = get_series(balance_sheet, equity_key)
        assets = get_series(balance_sheet, 'Total Assets')
        
        # Get current stock price for sanity check
        current_price = None
        shares_outstanding = None
        current_market_cap = None
        
        if ticker_symbol:
            try:
                stock = yf.Ticker(ticker_symbol)
                info = stock.info
                current_price = info.get('currentPrice') or info.get('regularMarketPrice')
                shares_outstanding = info.get('sharesOutstanding')
                current_market_cap = info.get('marketCap')
            except:
                pass
        
        # ========== DUPONT ANALYSIS (5-STEP) ==========
        if not any([x.empty for x in [rev, net_income, assets, equity]]):
            ni = net_income.iloc[0]
            rv = rev.iloc[0]
            ast = assets.iloc[0]
            eq = equity.iloc[0]
            
            if rv > 0 and ast > 0 and eq > 0:
                npm = (ni / rv) * 100  # Net Profit Margin %
                ato = rv / ast  # Asset Turnover
                em = ast / eq  # Equity Multiplier
                roe = (ni / eq) * 100  # Return on Equity %
                
                analysis["dupont"] = {
                    "roe": round(roe, 2),
                    "net_profit_margin": round(npm, 2),
                    "asset_turnover": round(ato, 2),
                    "equity_multiplier": round(em, 2),
                    "roa": round((ni / ast) * 100, 2),
                    "interpretation": f"ROE of {roe:.1f}% driven by {npm:.1f}% margin, {ato:.2f}x asset efficiency, {em:.2f}x leverage"
                }
        
        # ========== INVESTMENT BANKING DCF VALUATION (UFCF METHOD) ==========
        # Phase 1: Data Preparation - Calculate Unlevered Free Cash Flow
        if quarterly_data and len(quarterly_data) >= 4:
            # Step 1: Annualize by summing last 4 quarters (TTM)
            recent_4q_rev = [q.get("revenue") for q in quarterly_data[:4] if q.get("revenue")]
            
            if recent_4q_rev and len(recent_4q_rev) == 4:
                ttm_revenue = sum(recent_4q_rev)
                
                # Step 2: Get Free Cash Flow directly from Cash Flow Statement
                ttm_fcf = None
                fcf_margin = 15.0  # Default fallback
                ufcf_breakdown = {}
                
                # Try to get FCF from quarterly cash flow statement first (most accurate for TTM)
                if quarterly_cash_flow is not None and not quarterly_cash_flow.empty:
                    fcf_series = get_series(quarterly_cash_flow, 'Free Cash Flow')
                    if not fcf_series.empty and len(fcf_series) >= 4:
                        # Sum last 4 quarters for TTM
                        ttm_fcf = fcf_series.iloc[:4].sum()
                        if ttm_revenue > 0:
                            fcf_margin = (ttm_fcf / ttm_revenue) * 100
                        
                        # Get components for breakdown display
                        operating_cf_series = get_series(quarterly_cash_flow, 'Operating Cash Flow')
                        capex_series = get_series(quarterly_cash_flow, 'Capital Expenditure')
                        
                        if not operating_cf_series.empty and len(operating_cf_series) >= 4:
                            ttm_operating_cf = operating_cf_series.iloc[:4].sum()
                            ufcf_breakdown["operating_cash_flow"] = ttm_operating_cf
                        
                        if not capex_series.empty and len(capex_series) >= 4:
                            ttm_capex = abs(capex_series.iloc[:4].sum())  # CapEx is negative
                            ufcf_breakdown["capex"] = ttm_capex
                        
                        ufcf_breakdown["free_cash_flow"] = ttm_fcf
                
                # Fallback to annual cash flow if quarterly not available
                if ttm_fcf is None and cash_flow is not None and not cash_flow.empty:
                    fcf_series = get_series(cash_flow, 'Free Cash Flow')
                    if not fcf_series.empty and fcf_series.iloc[0]:
                        ttm_fcf = fcf_series.iloc[0]
                        if ttm_revenue > 0:
                            fcf_margin = (ttm_fcf / ttm_revenue) * 100
                        
                        # Get components
                        operating_cf_series = get_series(cash_flow, 'Operating Cash Flow')
                        capex_series = get_series(cash_flow, 'Capital Expenditure')
                        
                        if not operating_cf_series.empty:
                            ufcf_breakdown["operating_cash_flow"] = operating_cf_series.iloc[0]
                        if not capex_series.empty:
                            ufcf_breakdown["capex"] = abs(capex_series.iloc[0])
                        
                        ufcf_breakdown["free_cash_flow"] = ttm_fcf
                
                # Last resort: estimate if no FCF data available
                if ttm_fcf is None:
                    fcf_margin = 15.0
                    ttm_fcf = ttm_revenue * (fcf_margin / 100)
                    ufcf_breakdown["free_cash_flow"] = ttm_fcf
                    ufcf_breakdown["estimated"] = True
                
                # Step 3: Context-aware WACC (not generic 10%)
                # Large-cap stable tech: 7.5-9%, Mid-cap: 9-11%, Small-cap/distressed: 11%+
                if ttm_revenue > 50e9:  # >$50B revenue = large-cap stable
                    wacc = 0.08  # 8% for mega-cap tech
                elif ttm_revenue > 10e9:  # $10B-$50B = mid-cap
                    wacc = 0.095  # 9.5%
                else:
                    wacc = 0.11  # 11% for smaller/riskier
                
                if wacc_override is not None:
                    try:
                        wacc = max(0.01, float(wacc_override) / 100)
                    except Exception:
                        pass
                
                # Phase 2: The Valuation
                # Calculate historical revenue CAGR
                if len(quarterly_data) >= 12:
                    # Compare recent 4Q vs 4Q three years ago
                    old_4q_rev = [q.get("revenue") for q in quarterly_data[8:12] if q.get("revenue")]
                    if len(old_4q_rev) == 4 and sum(old_4q_rev) > 0:
                        years_span = 2.0  # 8 quarters = 2 years
                        historical_cagr = ((sum(recent_4q_rev) / sum(old_4q_rev)) ** (1/years_span)) - 1
                    else:
                        historical_cagr = 0.10  # 10% default
                else:
                    historical_cagr = 0.10
                
                # Base case FCF growth: use historical CAGR, capped at reasonable levels
                base_fcf_growth = min(historical_cagr, 0.25)  # Cap at 25%
                base_fcf_growth = max(base_fcf_growth, 0.03)  # Floor at 3%
                
                if fcf_growth_override is not None:
                    try:
                        base_fcf_growth = max(0.0, float(fcf_growth_override) / 100)
                    except Exception:
                        pass
                
                # 5-Year DCF Projection
                forecast_years = 5
                projected_fcf = []
                current_fcf = ttm_fcf
                
                for year in range(1, forecast_years + 1):
                    current_fcf = current_fcf * (1 + base_fcf_growth)
                    pv = current_fcf / ((1 + wacc) ** year)
                    projected_fcf.append({"year": year, "fcf": current_fcf, "pv": pv})
                
                # Terminal Value: Exit Multiple Method using Damodaran Industry Multiples
                # Get company industry from yfinance for multiple lookup
                terminal_value_method = "exit_multiple"
                exit_multiple = None
                damodaran_industry = None
                yf_industry = None
                yf_sector = None
                
                if ticker_symbol:
                    try:
                        stock_info = yf.Ticker(ticker_symbol).info
                        yf_industry = stock_info.get('industry')
                        yf_sector = stock_info.get('sector')
                        
                        # Look up Damodaran industry multiple
                        industry_multiple, damodaran_industry, is_exact_match = get_industry_multiple(
                            yf_industry, yf_sector
                        )
                        
                        if industry_multiple is not None:
                            exit_multiple = industry_multiple
                        else:
                            # Industry multiple N/A (e.g., banks) - use Gordon Growth
                            terminal_value_method = "gordon_growth"
                    except Exception:
                        pass
                
                # Fallback to size-based if no industry multiple found
                if exit_multiple is None and terminal_value_method == "exit_multiple":
                    if ttm_revenue > 50e9:
                        exit_multiple = 18
                    elif ttm_revenue > 10e9:
                        exit_multiple = 15
                    else:
                        exit_multiple = 12
                    damodaran_industry = "Size-based fallback"
                
                if terminal_value_method == "exit_multiple" and (not ebitda.empty and ebitda.iloc[0] and ebitda.iloc[0] > 0):
                    # Calculate TTM EBITDA from quarterly data
                    recent_4q_ebitda = []
                    for q in quarterly_data[:4]:
                        q_rev = q.get("revenue")
                        q_ni = q.get("net_income")
                        if q_rev and q_ni:
                            # Rough EBITDA estimate if not available
                            recent_4q_ebitda.append(q_ni * 1.3)  # Rough approximation
                    
                    if recent_4q_ebitda and len(recent_4q_ebitda) >= 3:
                        ttm_ebitda = sum(recent_4q_ebitda)
                    else:
                        # Fallback: estimate from FCF
                        ttm_ebitda = ttm_fcf * 1.5
                    
                    # Project Year 5 EBITDA
                    year5_ebitda = ttm_ebitda * ((1 + base_fcf_growth) ** forecast_years)
                    terminal_value = year5_ebitda * exit_multiple
                else:
                    # Fallback to Gordon Growth if EBITDA unavailable or industry not applicable
                    terminal_value_method = "gordon_growth"
                    terminal_growth = 0.03
                    terminal_fcf = projected_fcf[-1]["fcf"] * (1 + terminal_growth)
                    terminal_value = terminal_fcf / (wacc - terminal_growth)
                
                terminal_value_pv = terminal_value / ((1 + wacc) ** forecast_years)
                
                # Enterprise Value
                pv_fcf_sum = sum([p["pv"] for p in projected_fcf])
                base_enterprise_value = pv_fcf_sum + terminal_value_pv
                
                # Phase 3: Stress Test & Scenario Analysis
                # Sanity check: Compare to current market cap
                sanity_check = {}
                if current_market_cap:
                    ev_vs_mcap_diff_pct = ((base_enterprise_value - current_market_cap) / current_market_cap) * 100
                    sanity_check["current_market_cap_b"] = round(current_market_cap / 1e9, 2)
                    sanity_check["dcf_vs_market_diff_pct"] = round(ev_vs_mcap_diff_pct, 1)
                    
                    # Warning if >50% different
                    if abs(ev_vs_mcap_diff_pct) > 50:
                        sanity_check["warning"] = f"DCF differs by {ev_vs_mcap_diff_pct:.0f}% from market - verify assumptions"
                    
                    # Reverse DCF: What growth rate does current price imply?
                    # Solve for growth rate that makes EV = Market Cap
                    # Simplified: assume linear relationship
                    implied_growth = base_fcf_growth * (current_market_cap / base_enterprise_value)
                    sanity_check["market_implied_growth_pct"] = round(implied_growth * 100, 1)
                
                # Scenario Analysis: Bear/Base/Bull
                scenarios = {}
                
                # Helper function for scenario calculations
                def calc_scenario_ev(initial_fcf, growth, discount_rate, years, terminal_pv):
                    pv_sum = 0
                    cf = initial_fcf
                    for yr in range(1, years + 1):
                        cf = cf * (1 + growth)
                        pv_sum += cf / ((1 + discount_rate) ** yr)
                    return pv_sum + terminal_pv
                
                # Bear Case: Lower growth, margin compression
                bear_fcf_growth = max(base_fcf_growth * 0.5, 0.02)  # Half the growth, min 2%
                bear_fcf_margin = fcf_margin * 0.9  # 10% margin compression
                bear_fcf = ttm_revenue * (bear_fcf_margin / 100)
                bear_ev = calc_scenario_ev(bear_fcf, bear_fcf_growth, wacc, forecast_years, terminal_value_pv * 0.8)
                scenarios["bear"] = {
                    "ev_b": round(bear_ev / 1e9, 2),
                    "fcf_growth": round(bear_fcf_growth * 100, 1),
                    "fcf_margin": round(bear_fcf_margin, 1)
                }
                
                # Base Case: Most likely (already calculated)
                scenarios["base"] = {
                    "ev_b": round(base_enterprise_value / 1e9, 2),
                    "fcf_growth": round(base_fcf_growth * 100, 1),
                    "fcf_margin": round(fcf_margin, 1)
                }
                
                # Bull Case: Optimistic execution
                bull_fcf_growth = min(base_fcf_growth * 1.5, 0.30)  # 50% higher, max 30%
                bull_fcf_margin = min(fcf_margin * 1.1, 50)  # 10% expansion, cap at 50%
                bull_fcf = ttm_revenue * (bull_fcf_margin / 100)
                bull_ev = calc_scenario_ev(bull_fcf, bull_fcf_growth, wacc, forecast_years, terminal_value_pv * 1.2)
                scenarios["bull"] = {
                    "ev_b": round(bull_ev / 1e9, 2),
                    "fcf_growth": round(bull_fcf_growth * 100, 1),
                    "fcf_margin": round(bull_fcf_margin, 1)
                }
                
                analysis["dcf"] = {
                    "enterprise_value": round(base_enterprise_value, 0),
                    "enterprise_value_b": round(base_enterprise_value / 1e9, 2),
                    "ttm_revenue_b": round(ttm_revenue / 1e9, 2),
                    "ttm_fcf_b": round(ttm_fcf / 1e9, 2),
                    "fcf_margin": round(fcf_margin, 1),
                    "historical_fcf_margin": round(fcf_margin, 1),
                    "assumed_fcf_growth": round(base_fcf_growth * 100, 1),
                    "historical_revenue_cagr": round(historical_cagr * 100, 1),
                    "wacc": round(wacc * 100, 1),
                    "terminal_value_method": f"Exit Multiple ({exit_multiple}x EV/EBITDA)" if terminal_value_method == "exit_multiple" else "Gordon Growth (3% perpetual)",
                    "exit_multiple": exit_multiple if terminal_value_method == "exit_multiple" else None,
                    "damodaran_industry": damodaran_industry,
                    "yf_industry": yf_industry,
                    "yf_sector": yf_sector,
                    "damodaran_source_url": DAMODARAN_SOURCE_URL if damodaran_industry and damodaran_industry != "Size-based fallback" else None,
                    "interpretation": f"Base Case EV: ${base_enterprise_value/1e9:.1f}B | Bear: ${bear_ev/1e9:.1f}B | Bull: ${bull_ev/1e9:.1f}B",
                    "sanity_check": sanity_check,
                    "scenarios": scenarios,
                    "ufcf_breakdown": ufcf_breakdown if ufcf_breakdown else None,
                    "projected_fcf": projected_fcf  # Store 5-year projection
                }
        
        # ========== QUALITY METRICS ==========
        if not rev.empty and len(rev) >= 2:
            analysis["quality_metrics"] = {
                "revenue_stability": "High" if all(rev.iloc[i] > rev.iloc[i+1] * 0.95 for i in range(min(3, len(rev)-1))) else "Moderate",
                "margin_trend": "Expanding" if not net_income.empty and len(net_income) >= 2 and (net_income.iloc[0]/rev.iloc[0]) > (net_income.iloc[1]/rev.iloc[1]) else "Stable"
            }
    
    except Exception as e:
        analysis["error"] = str(e)
    
    return analysis


def get_gemini_model():
    """Returns a configured Gemini GenerativeModel."""
    try:
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        # Prefer flash for speed/cost
        flash_models = [m for m in available_models if 'flash' in m]
        model_name = flash_models[0] if flash_models else 'gemini-1.5-flash'
        return genai.GenerativeModel(model_name)
    except Exception as e:
        # Fallback if list_models fails or token issues
        return genai.GenerativeModel('gemini-1.5-flash')

def run_structured_prompt(system_role: str, user_prompt: str, context_data: str) -> str:
    """
    Sends a structured prompt to Google Gemini.
    """
    if not config_genai():
        return "Error: No API key found. Please enter your Gemini API key."
    
    full_prompt = f"""
    SYSTEM ROLE: {system_role}
    
    DATA CONTEXT:
    {context_data}
    
    YOUR TASK:
    {user_prompt}
    """
    
    try:
        model = get_gemini_model()
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"AI Error: {e}"

def run_chat(chat_history: list, new_message: str, context_data: str) -> str:
    """
    Runs a chat session with the data context.
    chat_history: list of {"role": "user"/"assistant", "content": ...}
    """
    if not config_genai():
        return "Error: No API key found."

    try:
        model = get_gemini_model()
        
        # Construct history for Gemini (user/model)
        gemini_history = []
        
        # Seed context in the first message
        gemini_history.append({
            "role": "user", 
            "parts": [f"You are a financial analyst. Use this data for all answers:\n\n{context_data}"]
        })
        gemini_history.append({
            "role": "model", 
            "parts": ["Understood. I have analyzed the data you provided."]
        })
            
        for msg in chat_history:
            role = "user" if msg["role"] == "user" else "model"
            gemini_history.append({"role": role, "parts": [msg["content"]]})
            
        chat = model.start_chat(history=gemini_history)
        response = chat.send_message(new_message)
        return response.text
    except Exception as e:
        return f"Chat Error: {e}"


def get_latest_date_info(ticker_symbol: str) -> dict:
    """
    Lightweight function to fetch just the most recent report date for a ticker.
    Does NOT run full analysis - just checks the column headers.
    
    Returns:
        dict with 'date' (formatted as "Dec 31, 2025"), 'year', 'month', 'quarter', 'raw_date'
    """
    try:
        stock = yf.Ticker(ticker_symbol)
        quarterly_income = stock.quarterly_income_stmt
        
        if quarterly_income.empty:
            return {"date": None, "error": "No data available"}
        
        most_recent = quarterly_income.columns[0]
        
        if hasattr(most_recent, 'year') and hasattr(most_recent, 'month'):
            year = most_recent.year
            month = most_recent.month
            day = most_recent.day if hasattr(most_recent, 'day') else 1
            calendar_quarter = (month - 1) // 3 + 1
            
            # Format as "Dec 31, 2025"
            month_names = {
                1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
            }
            formatted_date = f"{month_names[month]} {day}, {year}"
            
            return {
                "date": formatted_date,
                "year": year,
                "month": month,
                "quarter": calendar_quarter,
                "raw_date": str(most_recent)[:10]
            }
        else:
            return {"date": str(most_recent)[:10], "raw_date": str(most_recent)[:10]}
            
    except Exception as e:
        return {"date": None, "error": str(e)}


def get_available_report_dates(ticker_symbol: str) -> list:
    """
    Fetches all available quarterly report dates for a ticker.
    
    Returns:
        List of dicts with 'display' (human readable) and 'value' (ISO date string)
        e.g., [{'display': 'Dec 31, 2025', 'value': '2025-12-31'}, ...]
    """
    try:
        stock = yf.Ticker(ticker_symbol)
        quarterly_income = stock.quarterly_income_stmt
        
        if quarterly_income.empty:
            return []
        
        month_names = {
            1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
            7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
        }
        
        dates = []
        for col in quarterly_income.columns:
            if hasattr(col, 'year') and hasattr(col, 'month'):
                year = col.year
                month = col.month
                day = col.day if hasattr(col, 'day') else 1
                
                display = f"{month_names[month]} {day}, {year}"
                value = f"{year}-{month:02d}-{day:02d}"
                
                dates.append({"display": display, "value": value})
            else:
                # Fallback for non-datetime columns
                dates.append({"display": str(col)[:10], "value": str(col)[:10]})
        
        return dates
        
    except Exception as e:
        return []


def get_financial_data(ticker: str, fmp_api_key: str = None) -> tuple:
    """
    Primary data-fetching function. Fetches quarterly financial statements.
    Uses FMP API if key provided (up to 20 quarters), otherwise falls back to yfinance (5-8 quarters).
    
    Args:
        ticker: Stock ticker symbol
        fmp_api_key: Optional Financial Modeling Prep API key for extended data
    
    Returns:
        Tuple of (income_stmt, balance_sheet, cash_flow, quarterly_cash_flow, data_source, warning)
    """
    warning_message = None
    
    # Try FMP first if API key provided
    if fmp_api_key:
        try:
            # Fetch all three statements from FMP
            income_url = f"https://financialmodelingprep.com/api/v3/income-statement/{ticker}?period=quarter&limit=20&apikey={fmp_api_key}"
            balance_url = f"https://financialmodelingprep.com/api/v3/balance-sheet-statement/{ticker}?period=quarter&limit=20&apikey={fmp_api_key}"
            cashflow_url = f"https://financialmodelingprep.com/api/v3/cash-flow-statement/{ticker}?period=quarter&limit=20&apikey={fmp_api_key}"
            
            # Make API calls with error checking
            income_resp = requests.get(income_url, timeout=10)
            balance_resp = requests.get(balance_url, timeout=10)
            cashflow_resp = requests.get(cashflow_url, timeout=10)
            
            # Check HTTP status
            income_resp.raise_for_status()
            balance_resp.raise_for_status()
            cashflow_resp.raise_for_status()
            
            income_response = income_resp.json()
            balance_response = balance_resp.json()
            cashflow_response = cashflow_resp.json()
            
            # Check for FMP-specific errors
            def check_fmp_error(response):
                if isinstance(response, dict):
                    if 'Error Message' in response:
                        return response['Error Message']
                    if 'error' in response:
                        return response['error']
                    if 'message' in response and not isinstance(response.get('message'), (list, dict)):
                        # Some APIs return {"message": "error text"}
                        return response['message']
                return None
            
            error_msg = (check_fmp_error(income_response) or 
                        check_fmp_error(balance_response) or 
                        check_fmp_error(cashflow_response))
            
            if error_msg:
                raise Exception(f"FMP API: {error_msg}")
            
            # Check if data is empty
            if not income_response or not isinstance(income_response, list) or len(income_response) == 0:
                raise Exception("FMP API returned no data for this ticker")
            
            # Convert to DataFrames with yfinance-like structure
            def convert_fmp_to_df(data, statement_type):
                if not data:
                    return pd.DataFrame()
                
                df = pd.DataFrame(data)
                if df.empty:
                    return pd.DataFrame()
                
                # Convert date column to datetime and set as columns
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date').T
                
                # Map FMP field names to yfinance equivalents
                if statement_type == 'income':
                    field_mapping = {
                        'revenue': 'Total Revenue',
                        'operatingIncome': 'Operating Income',
                        'netIncome': 'Net Income',
                        'eps': 'Basic EPS',
                        'epsdiluted': 'Diluted EPS',
                        'ebitda': 'EBITDA'
                    }
                elif statement_type == 'balance':
                    field_mapping = {
                        'totalAssets': 'Total Assets',
                        'totalStockholdersEquity': 'Stockholders Equity',
                        'totalEquity': 'Total Equity Gross Minority Interest'
                    }
                elif statement_type == 'cashflow':
                    field_mapping = {
                        'freeCashFlow': 'Free Cash Flow',
                        'operatingCashFlow': 'Operating Cash Flow',
                        'capitalExpenditure': 'Capital Expenditure'
                    }
                else:
                    field_mapping = {}
                
                # Rename fields
                df.index = df.index.map(lambda x: field_mapping.get(x, x))
                return df
            
            income_stmt = convert_fmp_to_df(income_response, 'income')
            balance_sheet = convert_fmp_to_df(balance_response, 'balance')
            cash_flow = convert_fmp_to_df(cashflow_response, 'cashflow')
            quarterly_cash_flow = cash_flow.copy()  # FMP only has quarterly data
            
            if not income_stmt.empty:
                return income_stmt, balance_sheet, cash_flow, quarterly_cash_flow, "Financial Modeling Prep (Extended)", None
        
        except Exception as e:
            error_detail = str(e)
            print(f"FMP API failed for {ticker}: {error_detail}. Falling back to yfinance.")
            warning_message = f"⚠️ FMP API error: {error_detail}. Using limited yfinance data (5-8 quarters). Check your FMP API key or try again."
    
    # Fallback to yfinance
    if not fmp_api_key:
        warning_message = "⚠️ Using limited yfinance data (5-8 quarters). For accurate YoY growth calculations based on 12+ quarters, provide an FMP API key in the sidebar."
    
    try:
        stock = yf.Ticker(ticker)
        income_stmt = stock.quarterly_income_stmt
        balance_sheet = stock.quarterly_balance_sheet
        cash_flow = stock.quarterly_cashflow
        quarterly_cash_flow = cash_flow.copy()
        return income_stmt, balance_sheet, cash_flow, quarterly_cash_flow, "Yahoo Finance", warning_message
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), "Error", f"❌ Failed to fetch data: {e}"


def get_extended_financials_fmp(ticker: str, fmp_api_key: str, statement_type: str = 'income', limit: int = 20) -> pd.DataFrame:
    """
    Fetches extended quarterly financial data from Financial Modeling Prep API.
    
    Args:
        ticker: Stock ticker symbol
        fmp_api_key: FMP API key
        statement_type: 'income' or 'balance'
        limit: Number of quarters to fetch (max 20)
    
    Returns:
        DataFrame with quarterly data, columns as dates
    """
    if not fmp_api_key:
        return pd.DataFrame()
    
    try:
        if statement_type == 'income':
            url = f"https://financialmodelingprep.com/api/v3/income-statement/{ticker}?period=quarter&limit={limit}&apikey={fmp_api_key}"
        elif statement_type == 'balance':
            url = f"https://financialmodelingprep.com/api/v3/balance-sheet-statement/{ticker}?period=quarter&limit={limit}&apikey={fmp_api_key}"
        else:
            return pd.DataFrame()
        
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if not data or isinstance(data, dict) and data.get('Error Message'):
            return pd.DataFrame()
        
        # Convert to DataFrame with structure similar to yfinance
        df = pd.DataFrame(data)
        
        if df.empty:
            return pd.DataFrame()
        
        # Convert date column to datetime and set as columns
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').T
        
        # Map FMP field names to yfinance equivalents
        if statement_type == 'income':
            field_mapping = {
                'revenue': 'Total Revenue',
                'operatingIncome': 'Operating Income',
                'netIncome': 'Net Income',
                'eps': 'Basic EPS',
                'epsdiluted': 'Diluted EPS',
                'ebitda': 'EBITDA'
            }
        elif statement_type == 'balance':
            field_mapping = {
                'totalAssets': 'Total Assets',
                'totalStockholdersEquity': 'Stockholders Equity',
                'totalEquity': 'Total Equity Gross Minority Interest'
            }
        
        # Rename fields
        df.index = df.index.map(lambda x: field_mapping.get(x, x))
        
        return df
        
    except Exception as e:
        print(f"FMP API error for {ticker}: {e}")
        return pd.DataFrame()


def analyze_quarterly_trends(ticker_symbol: str, num_quarters: int = 8, end_date: str = None) -> dict:
    """
    Analyzes historical quarterly trends and fetches consensus estimates using yfinance.
    
    Args:
        ticker_symbol: Stock ticker
        num_quarters: Number of quarters to analyze (default 8)
        end_date: Optional ending date (ISO format like '2025-12-31') - the most recent quarter to include
    
    Returns a structured dictionary containing:
    - Historical quarterly data (Revenue, Operating Income, EPS)
    - YoY and QoQ growth rates
    - Average YoY growth projections
    - Consensus analyst estimates from web search
    - Next forecast quarter information
    """
    import json
    from datetime import datetime
    
    result = {
        "ticker": ticker_symbol,
        "analysis_date": datetime.now().strftime("%Y-%m-%d"),
        "historical_trends": {},
        "growth_rates": {},
        "projections": {},
        "consensus_estimates": {},
        "next_forecast_quarter": {},
        "errors": []
    }
    
    try:
        # --- Market data (for consensus total value) ---
        current_price = None
        shares_outstanding = None
        current_market_cap = None
        try:
            stock_info = yf.Ticker(ticker_symbol).info
            current_price = stock_info.get('currentPrice') or stock_info.get('regularMarketPrice')
            shares_outstanding = stock_info.get('sharesOutstanding')
            current_market_cap = stock_info.get('marketCap')
        except Exception:
            pass
        
        if shares_outstanding is None and current_market_cap and current_price:
            try:
                shares_outstanding = current_market_cap / current_price
            except Exception:
                pass
        
        result["market_data"] = {
            "current_price": current_price,
            "shares_outstanding": shares_outstanding,
            "market_cap": current_market_cap
        }
        
        # --- PART 1: Historical Quarterly Data ---
        stock = yf.Ticker(ticker_symbol)
        quarterly_income = stock.quarterly_income_stmt
        
        result["data_source"] = "Yahoo Finance"
        
        if quarterly_income.empty:
            result["errors"].append("No quarterly income statement data available")
            return result
        
        # Get available quarters and filter by user selection
        all_quarters = quarterly_income.columns
        
        # Find the most recent quarter
        most_recent = all_quarters[0]
        most_recent_year = most_recent.year if hasattr(most_recent, 'year') else None
        most_recent_q = (most_recent.month - 1) // 3 + 1 if hasattr(most_recent, 'month') else None
        
        result["historical_trends"]["most_recent_quarter"] = {
            "year": most_recent_year,
            "quarter": most_recent_q,
            "label": f"FY{most_recent_year} Q{most_recent_q}" if most_recent_year and most_recent_q else str(most_recent)[:10],
            "date": str(most_recent)[:10]
        }
        
        # Filter quarters based on end_date selection or use default (most recent)
        if end_date:
            # Find the index of the selected end_date
            start_idx = 0
            for i, q in enumerate(all_quarters):
                q_date = str(q)[:10]
                if q_date == end_date:
                    start_idx = i
                    break
            quarters = all_quarters[start_idx:start_idx + num_quarters]
            
            # Update most_recent_quarter to reflect the selected end_date
            if len(quarters) > 0:
                selected_q = quarters[0]
                selected_year = selected_q.year if hasattr(selected_q, 'year') else None
                selected_q_num = (selected_q.month - 1) // 3 + 1 if hasattr(selected_q, 'month') else None
                result["historical_trends"]["most_recent_quarter"] = {
                    "year": selected_year,
                    "quarter": selected_q_num,
                    "label": f"FY{selected_year} Q{selected_q_num}" if selected_year and selected_q_num else str(selected_q)[:10],
                    "date": str(selected_q)[:10]
                }
                most_recent_year = selected_year
                most_recent_q = selected_q_num
        else:
            quarters = all_quarters[:num_quarters]
        
        # Calculate next forecast quarter (based on the selected/most recent quarter)
        if most_recent_year and most_recent_q:
            if most_recent_q == 4:
                next_q_year = most_recent_year + 1
                next_q_num = 1
            else:
                next_q_year = most_recent_year
                next_q_num = most_recent_q + 1
            
            result["next_forecast_quarter"] = {
                "year": next_q_year,
                "quarter": next_q_num,
                "label": f"FY{next_q_year} Q{next_q_num}"
            }
        
        # Extract key metrics
        def safe_get(df, key, col):
            try:
                if key in df.index:
                    val = df.loc[key, col]
                    return float(val) if pd.notna(val) else None
            except:
                pass
            return None
        
        quarterly_data = []
        for q in quarters:
            q_str = q.strftime("%Y-Q%q") if hasattr(q, 'strftime') else str(q)[:10]
            quarter_num = (q.month - 1) // 3 + 1 if hasattr(q, 'month') else None
            q_label = f"{q.year}-Q{quarter_num}" if hasattr(q, 'year') else q_str
            
            revenue = safe_get(quarterly_income, 'Total Revenue', q)
            op_income = safe_get(quarterly_income, 'Operating Income', q)
            
            # EPS - try different keys
            eps = safe_get(quarterly_income, 'Basic EPS', q)
            if eps is None:
                eps = safe_get(quarterly_income, 'Diluted EPS', q)
            
            quarterly_data.append({
                "quarter": q_label,
                "date": str(q)[:10],
                "revenue": revenue,
                "operating_income": op_income,
                "eps": eps
            })
        
        result["historical_trends"]["quarterly_data"] = quarterly_data
        result["historical_trends"]["quarters_available"] = len(quarterly_data)
        
        # --- PART 2: Calculate Growth Rates ---
        def calc_growth(current, previous):
            if current is None or previous is None or previous == 0:
                return None
            return ((current - previous) / abs(previous)) * 100
        
        growth_data = []
        for i, q in enumerate(quarterly_data):
            growth_entry = {"quarter": q["quarter"]}
            
            # QoQ Growth (compare to previous quarter)
            if i < len(quarterly_data) - 1:
                prev_q = quarterly_data[i + 1]
                growth_entry["revenue_qoq"] = calc_growth(q["revenue"], prev_q["revenue"])
                growth_entry["op_income_qoq"] = calc_growth(q["operating_income"], prev_q["operating_income"])
                growth_entry["eps_qoq"] = calc_growth(q["eps"], prev_q["eps"])
            
            # YoY Growth (compare to same quarter last year - 4 quarters back)
            if i < len(quarterly_data) - 4:
                yoy_q = quarterly_data[i + 4]
                growth_entry["revenue_yoy"] = calc_growth(q["revenue"], yoy_q["revenue"])
                growth_entry["op_income_yoy"] = calc_growth(q["operating_income"], yoy_q["operating_income"])
                growth_entry["eps_yoy"] = calc_growth(q["eps"], yoy_q["eps"])
            
            growth_data.append(growth_entry)
        
        result["growth_rates"]["detailed"] = growth_data
        
        # Calculate average YoY growth for projections (all available YoY comparisons)
        yoy_revenues = [g.get("revenue_yoy") for g in growth_data if g.get("revenue_yoy") is not None]
        yoy_eps = [g.get("eps_yoy") for g in growth_data if g.get("eps_yoy") is not None]
        
        result["growth_rates"]["summary"] = {
            "avg_revenue_yoy": round(sum(yoy_revenues) / len(yoy_revenues), 2) if yoy_revenues else None,
            "avg_eps_yoy": round(sum(yoy_eps) / len(yoy_eps), 2) if yoy_eps else None,
            "samples_used": len(quarterly_data)  # Total quarters analyzed, not just YoY comparisons
        }
        
        # --- PART 3: Project Next Quarter Based on Historical Average ---
        if quarterly_data and len(quarterly_data) >= 5:
            # Use data from same quarter last year + average growth
            last_year_same_q = quarterly_data[4] if len(quarterly_data) > 4 else None
            avg_rev_growth = result["growth_rates"]["summary"].get("avg_revenue_yoy")
            avg_eps_growth = result["growth_rates"]["summary"].get("avg_eps_yoy")
            
            if last_year_same_q:
                projected = {
                    "basis": "Historical YoY average applied to same quarter last year",
                    "base_quarter": last_year_same_q["quarter"]
                }
                
                if last_year_same_q["revenue"] and avg_rev_growth:
                    projected["projected_revenue"] = round(
                        last_year_same_q["revenue"] * (1 + avg_rev_growth / 100), 0
                    )
                    projected["revenue_growth_rate_used"] = avg_rev_growth
                
                if last_year_same_q["eps"] and avg_eps_growth:
                    projected["projected_eps"] = round(
                        last_year_same_q["eps"] * (1 + avg_eps_growth / 100), 2
                    )
                    projected["eps_growth_rate_used"] = avg_eps_growth
                
                result["projections"]["next_quarter_estimate"] = projected
        
        # --- PART 4: Fetch Consensus Estimates via AI Web Search ---
        next_q_label = result.get("next_forecast_quarter", {}).get("label", "next quarter")
        result["consensus_estimates"] = fetch_consensus_estimates(ticker_symbol, next_q_label)
        
    except Exception as e:
        result["errors"].append(f"Analysis error: {str(e)}")
    
    return result


def fetch_consensus_estimates(ticker_symbol: str, next_quarter_label: str = "next quarter") -> dict:
    """
    Fetches consensus analyst estimates directly from Yahoo Finance via yfinance.
    Falls back to AI search only for qualitative summary.
    
    Args:
        ticker_symbol: Stock ticker
        next_quarter_label: Label for the upcoming quarter (e.g., "FY2026 Q3")
    """
    try:
        stock = yf.Ticker(ticker_symbol)
        info = stock.info
        
        # Get price targets from yfinance
        target_mean = info.get('targetMeanPrice')
        target_high = info.get('targetHighPrice')
        target_low = info.get('targetLowPrice')
        num_analysts = info.get('numberOfAnalystOpinions')
        recommendation = info.get('recommendationKey', '')
        
        # Get earnings and revenue estimates
        earnings_est = None
        revenue_est = None
        try:
            earnings_est = stock.earnings_estimate
            revenue_est = stock.revenue_estimate
        except:
            pass
        
        # Get recommendations summary (buy/hold/sell)
        rec_summary = None
        try:
            rec_summary = stock.recommendations_summary
        except:
            pass
        
        # Parse next quarter estimates (0q = current quarter)
        next_q_revenue = None
        next_q_eps = None
        next_q_analysts = None
        if revenue_est is not None and not revenue_est.empty:
            try:
                # 0q is current quarter estimate
                if '0q' in revenue_est.index:
                    next_q_revenue = revenue_est.loc['0q', 'avg']
                    next_q_analysts = int(revenue_est.loc['0q', 'numberOfAnalysts']) if 'numberOfAnalysts' in revenue_est.columns else None
                elif len(revenue_est) > 0:
                    next_q_revenue = revenue_est.iloc[0]['avg']
                    next_q_analysts = int(revenue_est.iloc[0]['numberOfAnalysts']) if 'numberOfAnalysts' in revenue_est.columns else None
            except:
                pass
        
        if earnings_est is not None and not earnings_est.empty:
            try:
                if '0q' in earnings_est.index:
                    next_q_eps = earnings_est.loc['0q', 'avg']
                elif len(earnings_est) > 0:
                    next_q_eps = earnings_est.iloc[0]['avg']
            except:
                pass
        
        # Parse full year estimates (0y = current year)
        full_year_revenue = None
        full_year_eps = None
        if revenue_est is not None and not revenue_est.empty:
            try:
                if '0y' in revenue_est.index:
                    full_year_revenue = revenue_est.loc['0y', 'avg']
                elif len(revenue_est) > 2:
                    full_year_revenue = revenue_est.iloc[2]['avg']
            except:
                pass
        
        if earnings_est is not None and not earnings_est.empty:
            try:
                if '0y' in earnings_est.index:
                    full_year_eps = earnings_est.loc['0y', 'avg']
                elif len(earnings_est) > 2:
                    full_year_eps = earnings_est.iloc[2]['avg']
            except:
                pass
        
        # Parse buy/hold/sell from recommendations summary
        buy_ratings = 0
        hold_ratings = 0
        sell_ratings = 0
        total_ratings = 0
        if rec_summary is not None and not rec_summary.empty:
            try:
                current = rec_summary.iloc[0]  # Most recent month
                buy_ratings = int(current.get('strongBuy', 0) or 0) + int(current.get('buy', 0) or 0)
                hold_ratings = int(current.get('hold', 0) or 0)
                sell_ratings = int(current.get('sell', 0) or 0) + int(current.get('strongSell', 0) or 0)
                total_ratings = buy_ratings + hold_ratings + sell_ratings
            except:
                pass
        
        # Format values
        def format_currency(val, is_billions=True):
            if val is None:
                return None
            if is_billions:
                return f"${val/1e9:.2f}B"
            return f"${val:.2f}"
        
        def format_price(val):
            if val is None:
                return None
            return f"${val:.2f}"
        
        # Build result - note: yfinance 0q = upcoming quarter to report, +1q = quarter after that
        # These are FORWARD estimates, not historical data
        result = {
            "next_quarter": {
                "revenue_estimate": format_currency(next_q_revenue) if next_q_revenue else "N/A",
                "eps_estimate": format_price(next_q_eps) if next_q_eps else "N/A",
                "quarter_label": f"{next_quarter_label} (Est.)",
                "source": "Yahoo Finance",
                "source_url": f"https://finance.yahoo.com/quote/{ticker_symbol}/analysis"
            },
            "full_year": {
                "revenue_estimate": format_currency(full_year_revenue) if full_year_revenue else "N/A",
                "eps_estimate": format_price(full_year_eps) if full_year_eps else "N/A",
                "fiscal_year": "Current FY",
                "source": "Yahoo Finance",
                "source_url": f"https://finance.yahoo.com/quote/{ticker_symbol}/analysis"
            },
            "analyst_coverage": {
                "num_analysts": total_ratings if total_ratings > 0 else num_analysts,  # Use ratings total for consistency with buy/hold/sell
                "buy_ratings": buy_ratings,
                "hold_ratings": hold_ratings,
                "sell_ratings": sell_ratings,
                "price_target_analysts": num_analysts,  # Separate field for price target analyst count
                "source": "Yahoo Finance",
                "source_url": f"https://finance.yahoo.com/quote/{ticker_symbol}/analysis"
            },
            "price_targets": {
                "average": format_price(target_mean),
                "high": format_price(target_high),
                "low": format_price(target_low),
                "source": "Yahoo Finance",
                "source_url": f"https://finance.yahoo.com/quote/{ticker_symbol}/analysis"
            },
            "citations": [
                {
                    "source_name": "Yahoo Finance",
                    "url": f"https://finance.yahoo.com/quote/{ticker_symbol}/analysis",
                    "data_type": "EPS & Revenue Estimates, Analyst Ratings",
                    "access_date": "current"
                }
            ],
            "source": "Yahoo Finance (yfinance)",
            "last_updated": "current"
        }
        
        # Generate qualitative summary using AI with source citations
        try:
            if config_genai():
                model = get_gemini_model()
                company_name = info.get('longName', ticker_symbol)
                industry = info.get('industry', 'technology')
                
                qual_prompt = f"""
                Search for recent analyst commentary on {company_name} ({ticker_symbol}) stock.
                
                Current data:
                - {buy_ratings} buy ratings, {hold_ratings} hold, {sell_ratings} sell
                - Average price target: ${target_mean:.0f} ({((target_mean - info.get('currentPrice', target_mean)) / info.get('currentPrice', 1) * 100):+.0f}% from current)
                - Industry: {industry}
                
                Return a JSON object with:
                1. "summary": One sentence explaining the key bull/bear thesis (under 40 words)
                2. "sources": Array of 2-3 recent sources backing this up, each with:
                   - "headline": The actual headline or quote
                   - "source": Publication name (e.g., "Barron's", "Seeking Alpha", "Bloomberg")
                   - "date": Approximate date if known (e.g., "Feb 2026", "Jan 2026")
                
                Example format:
                {{
                    "summary": "Analysts remain bullish citing Azure revenue acceleration and Copilot adoption, though some flag elevated valuation multiples.",
                    "sources": [
                        {{"headline": "Microsoft's AI Push Drives Cloud Growth", "source": "Bloomberg", "date": "Feb 2026"}},
                        {{"headline": "MSFT Upgraded on Strong Enterprise Demand", "source": "Barron's", "date": "Jan 2026"}}
                    ]
                }}
                
                Return ONLY valid JSON, no markdown.
                """
                
                response = model.generate_content(qual_prompt)
                response_text = response.text.strip()
                
                # Clean up response
                if response_text.startswith("```"):
                    lines = response_text.split("\n")
                    response_text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
                
                import json
                try:
                    qual_data = json.loads(response_text)
                    result["qualitative_summary"] = qual_data.get("summary", "")
                    result["qualitative_sources"] = qual_data.get("sources", [])
                except json.JSONDecodeError:
                    # If JSON parsing fails, use the raw text as summary
                    result["qualitative_summary"] = response_text[:200] if len(response_text) < 200 else response_text[:200] + "..."
                    result["qualitative_sources"] = []
        except Exception as e:
            pass  # No qualitative summary if AI fails
        
        return result
        
    except Exception as e:
        return {"error": f"Failed to fetch estimates from Yahoo Finance: {str(e)}"}


def generate_independent_forecast(quarterly_analysis: dict, company_name: str = None, dcf_data: dict = None) -> dict:
    """
    Takes the structured data from analyze_quarterly_trends and generates
    an independent equity research forecast using Gemini.
    
    IMPORTANT: This function distinguishes between:
    1. Operating/Fundamental Outlook (revenue, EPS, margins)
    2. Stock Price Outlook (valuation, discount rates, duration risk)
    
    A stock can decline even with strong fundamentals if:
    - Discount rates rise
    - Multiples compress
    - Cash flows shift out in time (duration lengthens)
    
    Args:
        quarterly_analysis: Output from analyze_quarterly_trends()
        company_name: Optional company name (will use ticker if not provided)
        dcf_data: Optional DCF valuation results from DCF engine
    
    Returns:
        Dictionary containing the AI analyst's independent forecast and justification
    """
    import json
    from datetime import datetime
    
    if not config_genai():
        return {"error": "No API key configured"}
    
    ticker = quarterly_analysis.get("ticker", "Unknown")
    company = company_name or ticker
    
    # Get forecast quarter info
    next_forecast = quarterly_analysis.get("next_forecast_quarter", {})
    forecast_quarter_label = next_forecast.get("label", "next quarter")
    most_recent = quarterly_analysis.get("historical_trends", {}).get("most_recent_quarter", {})
    most_recent_label = most_recent.get("label", "recent quarter")
    
    # Prepare the data packet summary for the prompt
    historical_data = quarterly_analysis.get("historical_trends", {})
    projections = quarterly_analysis.get("projections", {})
    consensus = quarterly_analysis.get("consensus_estimates", {})
    
    # ===== GET CASH FLOW TRAJECTORY DATA =====
    # This is critical for understanding CapEx ramps vs FCF compression
    cash_flow_context = ""
    try:
        stock = yf.Ticker(ticker)
        qcf = stock.quarterly_cashflow
        
        if qcf is not None and not qcf.empty:
            def get_series(df, key):
                return df.loc[key] if key in df.index else pd.Series(dtype='float64')
            
            fcf_series = get_series(qcf, 'Free Cash Flow')
            capex_series = get_series(qcf, 'Capital Expenditure')
            opcf_series = get_series(qcf, 'Operating Cash Flow')
            
            # Calculate TTM vs Prior TTM
            if not fcf_series.empty and len(fcf_series) >= 4:
                ttm_fcf = fcf_series.iloc[:4].sum()
                if len(fcf_series) >= 8:
                    # Check for NaN in prior TTM
                    prior_fcf_values = fcf_series.iloc[4:8].dropna()
                    if len(prior_fcf_values) >= 2:
                        prior_ttm_fcf = prior_fcf_values.sum()
                        fcf_yoy_change = ((ttm_fcf - prior_ttm_fcf) / abs(prior_ttm_fcf) * 100) if prior_ttm_fcf != 0 else 0
                    else:
                        prior_ttm_fcf = None
                        fcf_yoy_change = None
                else:
                    prior_ttm_fcf = None
                    fcf_yoy_change = None
            else:
                ttm_fcf = None
                prior_ttm_fcf = None
                fcf_yoy_change = None
            
            # CapEx trajectory
            if not capex_series.empty and len(capex_series) >= 4:
                ttm_capex = abs(capex_series.iloc[:4].sum())
                if len(capex_series) >= 8:
                    prior_capex_values = capex_series.iloc[4:8].dropna()
                    if len(prior_capex_values) >= 2:
                        prior_ttm_capex = abs(prior_capex_values.sum())
                        capex_yoy_change = ((ttm_capex - prior_ttm_capex) / prior_ttm_capex * 100) if prior_ttm_capex != 0 else 0
                    else:
                        prior_ttm_capex = None
                        capex_yoy_change = None
                else:
                    prior_ttm_capex = None
                    capex_yoy_change = None
            else:
                ttm_capex = None
                prior_ttm_capex = None
                capex_yoy_change = None
            
            # Build cash flow context string
            if ttm_fcf is not None:
                cash_flow_context = f"""
    CASH FLOW TRAJECTORY (CRITICAL FOR VALUATION):
    - TTM Free Cash Flow: ${ttm_fcf/1e9:.2f}B"""
                if prior_ttm_fcf is not None and fcf_yoy_change is not None:
                    cash_flow_context += f" (vs ${prior_ttm_fcf/1e9:.2f}B prior, {fcf_yoy_change:+.0f}% YoY)"
                
                if ttm_capex is not None:
                    cash_flow_context += f"""
    - TTM CapEx: ${ttm_capex/1e9:.2f}B"""
                    if prior_ttm_capex is not None and capex_yoy_change is not None:
                        cash_flow_context += f" (vs ${prior_ttm_capex/1e9:.2f}B prior, {capex_yoy_change:+.0f}% YoY)"
                
                # Add TTM Operating Cash Flow for funding context
                if not opcf_series.empty and len(opcf_series) >= 4:
                    ttm_opcf = opcf_series.iloc[:4].sum()
                    cash_flow_context += f"""
    - TTM Operating Cash Flow: ${ttm_opcf/1e9:.2f}B"""
                    # Check if internally funded (OpCF > CapEx)
                    if ttm_capex and ttm_opcf > ttm_capex:
                        cash_flow_context += f" (CapEx is internally funded)"
                    elif ttm_capex and ttm_opcf < ttm_capex:
                        funding_gap = ttm_capex - ttm_opcf
                        cash_flow_context += f" (Funding gap: ${funding_gap/1e9:.1f}B may require debt/equity)"
                
                # Calculate FCF margin
                quarterly_data = quarterly_analysis.get("historical_trends", {}).get("quarterly_data", [])
                if quarterly_data and len(quarterly_data) >= 4:
                    recent_4q_rev = [q.get("revenue") for q in quarterly_data[:4] if q.get("revenue")]
                    if len(recent_4q_rev) == 4:
                        ttm_revenue = sum(recent_4q_rev)
                        fcf_margin = (ttm_fcf / ttm_revenue * 100) if ttm_revenue > 0 else 0
                        capex_intensity = (ttm_capex / ttm_revenue * 100) if ttm_revenue > 0 and ttm_capex else 0
                        cash_flow_context += f"""
    - FCF Margin: {fcf_margin:.1f}% of revenue
    - CapEx Intensity: {capex_intensity:.1f}% of revenue"""
                
                # Flag major CapEx ramps - be factual, not speculative
                if capex_yoy_change is not None and capex_yoy_change > 30:
                    cash_flow_context += f"""
    
    ⚠️ CAPEX RAMP: CapEx increased {capex_yoy_change:.0f}% YoY
    NOTE: This is a fact from financial statements. For outlook, check management guidance."""
                    
                if fcf_yoy_change is not None and fcf_yoy_change < -30:
                    cash_flow_context += f"""
    
    ⚠️ FCF DECLINE: FCF fell {abs(fcf_yoy_change):.0f}% YoY
    NOTE: FCF = Operating CF - CapEx. If CapEx is elevated, this may be intentional reinvestment."""
    except Exception as e:
        cash_flow_context = f"\n    Cash flow trajectory data unavailable: {str(e)}"
    
    # Format historical data for the prompt (last 5 quarters - what we have access to)
    quarterly_data = historical_data.get("quarterly_data", [])
    historical_summary = "\n".join([
        f"  {q['quarter']}: Revenue=${q['revenue']/1e9:.2f}B, Op.Income=${q['operating_income']/1e9:.2f}B, EPS=${q['eps']:.2f}" 
        if q['revenue'] and q['operating_income'] and q['eps'] 
        else f"  {q['quarter']}: Revenue={q['revenue']}, Op.Income={q['operating_income']}, EPS={q['eps']}"
        for q in quarterly_data[:5]  # Last 5 quarters (what Yahoo Finance provides)
    ])
    
    # Format consensus estimates
    next_q = consensus.get("next_quarter", {})
    full_year = consensus.get("full_year", {})
    coverage = consensus.get("analyst_coverage", {})
    price_targets = consensus.get("price_targets", {})
    
    consensus_text = f"""
    NEXT QUARTER CONSENSUS ({next_q.get('quarter_label', 'Upcoming')}):
    - Revenue Estimate: {next_q.get('revenue_estimate', 'N/A')}
    - EPS Estimate: {next_q.get('eps_estimate', 'N/A')}
    
    FULL YEAR CONSENSUS ({full_year.get('fiscal_year', 'Current FY')}):
    - Revenue Estimate: {full_year.get('revenue_estimate', 'N/A')}
    - EPS Estimate: {full_year.get('eps_estimate', 'N/A')}
    
    ANALYST COVERAGE:
    - Total Analysts: {coverage.get('num_analysts', 'N/A')}
    - Buy: {coverage.get('buy_ratings', 'N/A')} | Hold: {coverage.get('hold_ratings', 'N/A')} | Sell: {coverage.get('sell_ratings', 'N/A')}
    
    PRICE TARGETS:
    - Average: {price_targets.get('average', 'N/A')}
    - Low: {price_targets.get('low', 'N/A')} | High: {price_targets.get('high', 'N/A')}
    """
    
    # Historical projection (our calculated estimate)
    hist_proj = projections.get("next_quarter_estimate", {})
    hist_proj_text = f"""
    HISTORICAL TREND PROJECTION:
    - Projected Revenue: ${hist_proj.get('projected_revenue', 0)/1e9:.2f}B
    - Projected EPS: ${hist_proj.get('projected_eps', 'N/A')}
    """ if hist_proj else "Historical projection not available (insufficient data)"
    
    # Format DCF valuation data if available
    dcf_text = ""
    dcf_fcf_assumption_warning = ""
    market_implied_reconciliation = ""
    if dcf_data:
        dcf_price = dcf_data.get('price_per_share', 0)
        current_price = dcf_data.get('current_price', 0)
        assumptions = dcf_data.get('assumptions', {})
        wacc = assumptions.get('wacc', 0.09)
        fcf_growth = assumptions.get('fcf_growth_rate', 0.08)
        terminal_multiple = assumptions.get('terminal_multiple', 15)
        data_quality = dcf_data.get('data_quality_score', 0)
        
        # Get detailed DCF metrics from trace
        pv_fcf_sum = dcf_data.get('pv_fcf_sum', 0)
        pv_terminal_value = dcf_data.get('pv_terminal_value', 0)
        enterprise_value = dcf_data.get('enterprise_value', 0)
        net_debt = dcf_data.get('net_debt', 0)
        shares = dcf_data.get('shares_outstanding', 1)
        
        # FCFF metrics (proper enterprise cash flow)
        ttm_fcff = assumptions.get('ttm_fcff', 0)
        fcff_method = assumptions.get('fcff_method', 'unknown')
        fcff_ebitda_ratio = assumptions.get('fcff_ebitda_ratio', 0)
        terminal_year_fcff = assumptions.get('terminal_year_fcff', 0)
        terminal_year_ebitda = assumptions.get('terminal_year_ebitda', 0)
        terminal_year_fcff_ebitda = assumptions.get('terminal_year_fcff_ebitda', 0)
        tv_dominance_pct = assumptions.get('tv_dominance_pct', 0)
        consistent_exit_multiple = assumptions.get('consistent_exit_multiple', 0)
        
        upside_pct = ((dcf_price - current_price) / current_price * 100) if current_price else 0
        
        dcf_text = f"""
    DCF VALUATION ANALYSIS (FCFF-BASED, WACC DISCOUNTING):
    - DCF Intrinsic Value: ${dcf_price:.2f} per share
    - Current Market Price: ${current_price:.2f}
    - Implied Upside/Downside: {upside_pct:+.1f}%
    
    DCF ASSUMPTIONS:
    - WACC: {wacc*100:.1f}%
    - Terminal Growth (g_perp): {assumptions.get('terminal_growth_rate', 0.03)*100:.1f}%
    - WACC - g: {(wacc - assumptions.get('terminal_growth_rate', 0.03))*100:.1f}%
    - Forecast Horizon: {assumptions.get('forecast_years', 10)} years
    - FCF Growth Rate (near-term): {fcf_growth*100:.1f}%
    - Data Quality Score: {data_quality}/100
    
    FCFF METRICS (CRITICAL - USE THESE, NOT OCF-CapEx):
    - TTM FCFF: ${ttm_fcff/1e9:.2f}B (Method: {fcff_method})
    - FCFF/EBITDA (TTM): {fcff_ebitda_ratio*100:.1f}%
    - Terminal Year FCFF (Year {assumptions.get('forecast_years', 10)}): ${terminal_year_fcff/1e9:.2f}B
    - Terminal Year EBITDA: ${terminal_year_ebitda/1e9:.2f}B
    - Terminal FCFF/EBITDA: {terminal_year_fcff_ebitda*100:.1f}%
    
    DCF COMPONENTS:
    - PV(Explicit FCF Years 1-{assumptions.get('forecast_years', 10)}): ${pv_fcf_sum/1e9:.2f}B
    - PV(Terminal Value): ${pv_terminal_value/1e9:.2f}B
    - Enterprise Value: ${enterprise_value/1e9:.2f}B
    - TV Dominance: {tv_dominance_pct:.1f}% of EV
    - Net Debt: ${net_debt/1e9:.2f}B
    - Consistent Exit Multiple (from Gordon): {consistent_exit_multiple:.1f}x
    """
        
        # ===== MARKET-IMPLIED RECONCILIATION =====
        # Solve for: What terminal FCFF does the market require to justify current price?
        # Market EV = Market Cap + Net Debt
        market_cap = current_price * shares if current_price and shares else 0
        market_ev = market_cap + net_debt if market_cap else 0
        
        if market_ev > 0 and pv_fcf_sum > 0:
            # Market-implied PV(TV) = Market EV - PV(Explicit FCF)
            market_implied_pv_tv = market_ev - pv_fcf_sum
            
            # Discount factor for terminal year
            discount_factor = (1 + wacc) ** assumptions.get('forecast_years', 10)
            
            # Market-implied TV (undiscounted)
            market_implied_tv = market_implied_pv_tv * discount_factor
            
            # Market-implied terminal FCFF = TV × (WACC - g) / (1 + g)
            terminal_g = assumptions.get('terminal_growth_rate', 0.03)
            wacc_minus_g = wacc - terminal_g
            if wacc_minus_g > 0:
                market_implied_terminal_fcff = market_implied_tv * wacc_minus_g / (1 + terminal_g)
                
                # Compare to model terminal FCFF
                model_terminal_fcff = terminal_year_fcff if terminal_year_fcff else 1
                fcff_multiple_required = market_implied_terminal_fcff / model_terminal_fcff if model_terminal_fcff > 0 else 0
                
                # Market-implied FCFF/EBITDA
                market_implied_fcff_ebitda = market_implied_terminal_fcff / terminal_year_ebitda if terminal_year_ebitda > 0 else 0
                
                market_implied_reconciliation = f"""
    MARKET-IMPLIED RECONCILIATION:
    - Market EV (current price): ${market_ev/1e9:.2f}B
    - Market-Implied PV(TV): ${market_implied_pv_tv/1e9:.2f}B (= Market EV - PV Explicit FCF)
    - Market-Implied TV (Year {assumptions.get('forecast_years', 10)}): ${market_implied_tv/1e9:.2f}B
    - Market-Implied Terminal FCFF: ${market_implied_terminal_fcff/1e9:.2f}B
    - Model Terminal FCFF: ${model_terminal_fcff/1e9:.2f}B
    - MULTIPLE OF CHANGE REQUIRED: {fcff_multiple_required:.2f}x (market requires {fcff_multiple_required:.1f}x the model's terminal FCFF)
    - Market-Implied FCFF/EBITDA: {market_implied_fcff_ebitda*100:.1f}% (vs model {terminal_year_fcff_ebitda*100:.1f}%)
    
    INTERPRETATION: To justify current price, terminal FCFF must be {fcff_multiple_required:.1f}x higher than model forecast.
    {"This implies the market expects higher margins, lower reinvestment, or faster growth convergence." if fcff_multiple_required > 1.2 else "Model and market are roughly aligned on terminal economics." if 0.8 <= fcff_multiple_required <= 1.2 else "Market is pricing in lower terminal FCFF than model forecasts."}
    """
        
        # Add warning if FCF is compressed but DCF shows upside
        if upside_pct > 10 and cash_flow_context and "FCF COMPRESSION" in cash_flow_context:
            dcf_fcf_assumption_warning = f"""
    ⚠️ DCF CONFLICT WARNING:
    The DCF model shows {upside_pct:+.1f}% upside, but TTM FCF has declined significantly.
    This may indicate the DCF growth assumptions don't reflect the current CapEx ramp.
    Consider: Is the assumed {fcf_growth*100:.1f}% FCF growth realistic given current cash flow trends?
    """
    
    try:
        model = get_gemini_model()
        
        # Get DCF-specific data for the prompt
        dcf_intrinsic = dcf_data.get('price_per_share', 0) if dcf_data else 0
        dcf_current = dcf_data.get('current_price', 0) if dcf_data else 0
        dcf_gap_pct = ((dcf_intrinsic - dcf_current) / dcf_current * 100) if dcf_current else 0
        dcf_wacc = assumptions.get('wacc', 0.09) if dcf_data else 0.09
        dcf_growth = assumptions.get('fcf_growth_rate', 0.08) if dcf_data else 0.08
        dcf_terminal_mult = assumptions.get('terminal_multiple', 15) if dcf_data else 15
        terminal_fcff_ebitda = assumptions.get('terminal_year_fcff_ebitda', 0.5) if dcf_data else 0.5
        forecast_years = assumptions.get('forecast_years', 10) if dcf_data else 10
        tv_dominance_pct = assumptions.get('tv_dominance_pct', 70) if dcf_data else 70
        consistent_exit_multiple = assumptions.get('consistent_exit_multiple', 10) if dcf_data else 10
        
        forecast_prompt = f"""
        Act as a senior equity research analyst. Generate an evidence-based outlook in THREE horizons.
        
        ═══════════════════════════════════════════════════════════════
        HARD CONSTRAINTS (MUST FOLLOW):
        ═══════════════════════════════════════════════════════════════
        1. NEVER SAY "NOT PROVIDED" until you have searched ALL provided data sections above. The DCF trace contains TTM FCFF, FCFF/EBITDA, terminal metrics, etc. USE THEM.
        
        2. USE FCFF CONSISTENTLY: This is an FCFF-based DCF (enterprise value, WACC discounting).
           - FCFF = EBIT×(1-tax) + D&A - CapEx - ΔNWC (provided in trace)
           - Do NOT confuse with OCF - CapEx (that's levered FCF for equity)
           - If you mention OCF-CapEx, label it "levered FCF proxy" and note it's a DIFFERENT metric
           
        3. FCF RECONCILIATION: If both "proper FCFF" and "levered FCF proxy" are shown with a large gap:
           - ACKNOWLEDGE the gap explicitly (e.g., "TTM FCFF is 38B vs levered proxy of 7.7B")
           - DO NOT issue categorical "OVERVALUED/UNDERVALUED" verdicts without explaining this gap
           - The gap sources: interest expense, working capital timing, lease treatment, stock comp, CapEx classification
           - If you cannot reconcile, DOWNGRADE CONFIDENCE and note the uncertainty
           
        4. REVENUE ≠ FCFF GROWTH: Do NOT claim that revenue growth "aligns with" or "supports" FCFF growth.
           - FCFF growth depends on: margins, reinvestment rate, working capital changes, tax rate
           - Revenue growth alone says NOTHING about FCFF growth
           - A high-growth company can have NEGATIVE FCFF growth due to heavy reinvestment
           
        5. CITATION REQUIREMENT: Any claim based on outside information (news, macro, analyst reports, segment outlook, rates) MUST include:
           - Specific date
           - Clickable URL
           - NO CITATION = DO NOT STATE IT AS FACT
           - Example: "Fed held rates at 5.25% (Jan 2026, https://federalreserve.gov/...)"
        
        6. NO SEGMENT CLAIMS: Do NOT mention segment drivers (Azure, AWS, Office, etc.) unless:
           - They appear in the provided inputs, OR
           - You provide an external citation with URL
        
        7. DO NOT produce a next-quarter forecast unless explicit quarter-level guidance is provided.
        
        ═══════════════════════════════════════════════════════════════
        DATA PACKET FOR {ticker}
        Analysis Date: {quarterly_analysis.get('analysis_date', datetime.now().strftime('%Y-%m-%d'))}
        ═══════════════════════════════════════════════════════════════
        
        SECTION 1: HISTORICAL QUARTERLY PERFORMANCE (Last 5 Quarters)
        {historical_summary}
        
        SECTION 2: WALL STREET CONSENSUS
        {consensus_text}
        
        SECTION 3: CASH FLOW TRAJECTORY (FROM FINANCIALS)
        {cash_flow_context if cash_flow_context else "Cash flow data not available."}
        
        SECTION 4: DCF VALUATION MODEL (STEP 1 RESULTS)
        {dcf_text if dcf_text else "DCF analysis not yet run."}
        {market_implied_reconciliation if market_implied_reconciliation else ""}
        {dcf_fcf_assumption_warning if dcf_fcf_assumption_warning else ""}
        
        ═══════════════════════════════════════════════════════════════
        FORMAT YOUR RESPONSE EXACTLY AS FOLLOWS:
        ═══════════════════════════════════════════════════════════════
        
        ### A) Short-Term Outlook (0–12 months)
        
        **Directional Stance:** [Bullish / Neutral / Bearish] — CONDITIONAL, not categorical
        
        **Key Drivers (3–5 bullets, tied to provided data or cited sources):**
        - [Driver 1 with data reference or citation URL]
        - [Driver 2 with data reference or citation URL]
        - [etc.]
        
        **Key Risks (3 bullets):**
        - [Risk 1]
        - [Risk 2]
        - [Risk 3]
        
        **What Would Change My View (2–3 measurable triggers):**
        - [Trigger 1: specific, quantifiable]
        - [Trigger 2: specific, quantifiable]
        
        ---
        
        ### B) Mid-Term Outlook (1–3 years)
        
        **Growth and Profitability Trajectory:**
        - Historical trend from Step 1: [summarize revenue/margin trajectory from the data]
        - Analyst multi-year outlook: [compare to consensus estimates above]
        - Gap assessment: [where they agree/disagree, quantify]
        
        **Assumption Reasonableness Check:**
        - DCF WACC ({dcf_wacc*100:.1f}%): [reasonable / aggressive / conservative] — why?
        - DCF Growth ({dcf_growth*100:.1f}%): [reasonable / aggressive / conservative] — why?
        - Terminal Cash Conversion ({terminal_fcff_ebitda*100:.0f}%): [achievable / stretched] — why?
        
        **Risks and Catalysts (tied to data/citations):**
        - [Risk or catalyst with supporting evidence]
        
        ---
        
        ### C) Long-Term Outlook (3–{forecast_years}+ years, Terminal)
        
        **DCF Consistency Audit:**
        - Terminal method used: Gordon Growth (primary) with exit multiple cross-check
        - Terminal value dominance: {tv_dominance_pct:.1f}% of EV comes from terminal — is this reasonable for this business?
        - Terminal FCFF/EBITDA: {terminal_fcff_ebitda*100:.0f}% — does this cash conversion make sense at steady state?
        - Consistent Exit Multiple (derived from Gordon): {consistent_exit_multiple:.1f}x — compare to current trading multiple
        
        **Exit Multiple as Stress Test (NOT Base Case):**
        IMPORTANT: Today's market EV/EBITDA multiple embeds GROWTH EXPECTATIONS and option value. 
        A mature terminal state should NOT trade at today's multiple.
        - If the cross-check shows "impossible" cash conversion, this is EXPECTED — it means:
          "If you FORCED the terminal state to trade at today's multiple, it would require X% FCFF/EBITDA"
        - This is a STRESS TEST, not a base case. Do not say "current multiple is impossible" as a bug.
        - The informative question: How much multiple compression is the Gordon model implying at terminal?
        
        **Market-Implied Reconciliation (REQUIRED):**
        Use the Market-Implied Reconciliation data above to answer:
        - What is the "multiple of change required" for terminal FCFF?
        - What does this imply about market expectations vs model?
        - If multiple > 1.5x: market expects significantly higher terminal cash flows — explain what could justify this (faster growth, higher margins, lower WACC)
        - If multiple < 0.7x: model may be too optimistic — explain the risk
        
        **Analyst vs DCF Comparison:**
        - Where they AGREE: [specific points with data]
        - Where they DISAGREE: [specific points and what this implies for valuation]
        
        **Market-Implied Expectations:**
        - Current price: {dcf_current:.2f} USD
        - DCF intrinsic value: {dcf_intrinsic:.2f} USD
        - Gap: {dcf_gap_pct:+.1f}%
        - What must be TRUE for the current market price to be justified? (quantify: growth rate, margins, duration)
        
        ---
        
        ### D) Final Assessment
        
        **Summary (5–7 lines integrating all horizons):**
        [One paragraph that ties short, mid, and long-term views together. Be specific about what drives each horizon's outlook.]
        
        **Evidence Gaps:**
        What missing inputs would materially improve confidence in this assessment?
        - [Gap 1]
        - [Gap 2]
        - [Gap 3]
        
        **Investment Stance:**
        - **Fundamental Outlook:** [Strong / Stable / Weakening] — Conviction: [High / Medium / Low]
        - **Stock Outlook:** [Bullish / Neutral / Bearish] over [time horizon]
        - **Key Conditional:** "Bullish IF [specific measurable condition], cautious UNTIL [specific measurable milestone]"
        
        ═══════════════════════════════════════════════════════════════
        CITATION FORMAT (MANDATORY FOR ALL EXTERNAL CLAIMS):
        ═══════════════════════════════════════════════════════════════
        Format: "[claim] ([source], [date], [URL])"
        
        Required for:
        - Analyst ratings/price targets
        - Consensus estimates
        - Macro data (rates, GDP, inflation)
        - Segment-specific claims
        - Industry comparisons
        - News or events
        
        Preferred sources: SEC EDGAR, Company IR, Fed/Treasury, Damodaran, Reuters, Bloomberg, Yahoo Finance
        
        Example: "Analysts have a mean price target of 220 USD (Yahoo Finance, Feb 2026, https://finance.yahoo.com/quote/AMZN)"
        
        NO CITATION = DO NOT STATE AS FACT. Say "Hypothesis" or omit the claim entirely.
        
        Do NOT use dollar signs ($) - use 'USD' instead.
        """
        
        response = model.generate_content(forecast_prompt)
        response_text = response.text.strip()
        
        # Also try to extract structured data from the response
        result = {
            "ticker": ticker,
            "company_name": company,
            "forecast_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "full_analysis": response_text,
            "input_data_summary": {
                "quarters_analyzed": len(quarterly_data),
                "consensus_revenue": next_q.get('revenue_estimate'),
                "consensus_eps": next_q.get('eps_estimate'),
                "dcf_price": dcf_data.get('price_per_share') if dcf_data else None,
                "current_price": dcf_data.get('current_price') if dcf_data else None
            },
            "disclaimer": "This is an AI-generated forecast for educational purposes only. It is not financial advice. Always conduct your own research and consult with qualified financial advisors before making investment decisions."
        }
        
        # Try to extract the key numbers from the response
        try:
            extract_prompt = f"""
            Extract ONLY the following from this analyst forecast. Return as JSON:
            {{
                "short_term_stance": "Bullish/Neutral/Bearish",
                "short_term_drivers": ["driver1", "driver2", "driver3"],
                "short_term_risks": ["risk1", "risk2", "risk3"],
                "mid_term_growth_trajectory": "Brief summary of growth/profitability trend",
                "dcf_wacc_assessment": "reasonable/aggressive/conservative",
                "dcf_growth_assessment": "reasonable/aggressive/conservative",
                "terminal_conversion_assessment": "achievable/stretched",
                "long_term_tv_dominance": "Brief assessment of terminal value weight",
                "market_implied_gap": "X% - brief explanation",
                "fundamental_outlook": "Strong/Stable/Weakening",
                "fundamental_conviction": "High/Medium/Low",
                "stock_outlook": "Bullish/Neutral/Bearish",
                "stock_outlook_horizon": "Short-term/Mid-term/Long-term",
                "stock_conviction": "High/Medium/Low",
                "key_conditional": "The exact 'Bullish IF X, cautious UNTIL Y' statement",
                "evidence_gaps": ["gap1", "gap2", "gap3"],
                "summary": "5-7 line summary integrating all horizons"
            }}
            
            EXTRACTION RULES:
            - fundamental_conviction: MUST be Medium or Low if key unknowns remain
            - key_conditional: exact text with "IF" and "UNTIL" format
            - evidence_gaps: list of missing data that would improve confidence
            - If a value cannot be determined, use null.
            
            Text to extract from:
            {response_text[:4000]}
            
            Return ONLY the JSON, no other text.
            """
            extract_response = model.generate_content(extract_prompt)
            extract_text = extract_response.text.strip()
            if extract_text.startswith("```"):
                lines = extract_text.split("\n")
                extract_text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
            
            extracted = json.loads(extract_text)
            result["extracted_forecast"] = extracted
        except:
            result["extracted_forecast"] = None
        
        return result
        
    except Exception as e:
        return {
            "error": f"Failed to generate forecast: {str(e)}",
            "ticker": ticker
        }
