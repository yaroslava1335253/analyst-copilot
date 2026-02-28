# engine.py
"""
Analyst Co-Pilot Engine (Lightweight)
=====================================
Uses Google Gemini (FREE) for AI analysis.
Includes financial math logic to pre-process data for the LLM.
"""

import os
import re
import time
import requests
import yfinance as yf
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
from industry_multiples import get_industry_multiple, DAMODARAN_SOURCE_URL, DAMODARAN_DATA_DATE

_genai_client: "genai.Client | None" = None
_sec_ticker_cik_map_cache = None
SEC_CIK_FALLBACK_MAP = {
    "AAPL": 320193,
    "AMZN": 1018724,
    "GOOG": 1652044,
    "GOOGL": 1652044,
    "META": 1326801,
    "MSFT": 789019,
    "NVDA": 1045810,
    "TSLA": 1318605,
    "NFLX": 1065280,
}
MIN_YOY_PAIRS_FOR_AVG_GROWTH = 2
MIN_REVENUE_POINTS_FOR_SEASONALITY = 8
MIN_TRANSITIONS_PER_QUARTER_FOR_SEASONALITY = 2

# Load local .env regardless of launch directory so FMP/Gemini keys are consistently available.
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")


def _sanitize_valuation_language(value):
    """
    Prevent overconfident valuation phrasing in AI text.
    Intrinsic value should always be framed as model-implied under assumptions.
    """
    if isinstance(value, str):
        text = value
        replacements = [
            (r"(?i)\bfundamental floor\b", "model-implied value under current assumptions"),
            (r"(?i)\bvaluation floor\b", "model-implied value under current assumptions"),
            (r"(?i)\bintrinsic floor\b", "model-implied value under current assumptions"),
            (r"(?i)\bhard floor\b", "assumption-sensitive downside case"),
        ]
        for pattern, replacement in replacements:
            text = re.sub(pattern, replacement, text)
        return text
    if isinstance(value, list):
        return [_sanitize_valuation_language(v) for v in value]
    if isinstance(value, dict):
        return {k: _sanitize_valuation_language(v) for k, v in value.items()}
    return value


def _redact_api_secrets(text: str, known_secret: str = "") -> str:
    if not isinstance(text, str):
        return text
    redacted = re.sub(r"(apikey=)[^&\s]+", r"\1[REDACTED]", text, flags=re.IGNORECASE)
    secret = (known_secret or "").strip()
    if secret:
        redacted = redacted.replace(secret, "[REDACTED]")
    return redacted


def config_genai():
    """Configures the Gemini API client."""
    global _genai_client
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if api_key:
        _genai_client = genai.Client(api_key=api_key)
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


_GEMINI_MODEL = "gemini-2.5-flash"

def get_gemini_model() -> tuple:
    """Returns (client, model_name) for use with the google.genai SDK."""
    return _genai_client, _GEMINI_MODEL

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
        client, model_name = get_gemini_model()
        response = client.models.generate_content(model=model_name, contents=full_prompt)
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
        client, model_name = get_gemini_model()
        
        # Construct history for Gemini (user/model)
        gemini_history = []
        
        # Seed context in the first message
        gemini_history.append(types.Content(
            role="user",
            parts=[types.Part(text=f"You are a financial analyst. Use this data for all answers:\n\n{context_data}")]
        ))
        gemini_history.append(types.Content(
            role="model",
            parts=[types.Part(text="Understood. I have analyzed the data you provided.")]
        ))
            
        for msg in chat_history:
            role = "user" if msg["role"] == "user" else "model"
            gemini_history.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))
            
        chat = client.chats.create(model=model_name, history=gemini_history)
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
        quarterly_income, data_source, _ = _get_quarterly_income_history(ticker_symbol, max_quarters=20)

        if quarterly_income.empty:
            return {"date": None, "error": "No data available", "data_source": "Unavailable"}

        ordered_quarters = _ordered_quarter_columns(quarterly_income)
        if not ordered_quarters:
            return {"date": None, "error": "No quarter columns available", "data_source": data_source}

        most_recent = ordered_quarters[0]
        
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
                "raw_date": str(most_recent)[:10],
                "data_source": data_source,
            }
        else:
            return {"date": str(most_recent)[:10], "raw_date": str(most_recent)[:10], "data_source": data_source}
            
    except Exception as e:
        return {"date": None, "error": str(e), "data_source": "Unavailable"}


def get_available_report_dates(ticker_symbol: str) -> list:
    """
    Fetches all available quarterly report dates for a ticker.
    
    Returns:
        List of dicts with 'display' (human readable) and 'value' (ISO date string)
        e.g., [{'display': 'Dec 31, 2025', 'value': '2025-12-31'}, ...]
    """
    try:
        quarterly_income, _, _ = _get_quarterly_income_history(ticker_symbol, max_quarters=20)
        if quarterly_income.empty:
            return []

        month_names = {
            1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
            7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
        }

        dates = []
        for col in _ordered_quarter_columns(quarterly_income):
            if hasattr(col, 'year') and hasattr(col, 'month'):
                year = col.year
                month = col.month
                day = col.day if hasattr(col, 'day') else 1

                display = f"{month_names[month]} {day}, {year}"
                value = f"{year}-{month:02d}-{day:02d}"
                dates.append({"display": display, "value": value})
            else:
                value = str(col)[:10]
                dates.append({"display": value, "value": value})

        return dates
    except Exception:
        return []


def _ordered_quarter_columns(df: pd.DataFrame) -> list:
    if df is None or df.empty:
        return []

    columns = list(df.columns)
    if not columns:
        return []

    parsed_dates = pd.to_datetime(columns, errors="coerce")
    if isinstance(parsed_dates, pd.DatetimeIndex):
        sortable = []
        unsortable = []
        for original, parsed in zip(columns, parsed_dates):
            if pd.notna(parsed):
                sortable.append((parsed, original))
            else:
                unsortable.append(original)
        if sortable:
            sortable.sort(key=lambda x: x[0], reverse=True)
            return [original for _, original in sortable] + unsortable

    return columns


def _sec_headers() -> dict:
    user_agent = (
        os.getenv("SEC_API_USER_AGENT")
        or os.getenv("SEC_USER_AGENT")
        or "AnalystCoPilot/1.0 (research tool; contact: analystcopilot.app@gmail.com)"
    )
    return {
        "User-Agent": user_agent,
        "Accept-Encoding": "gzip, deflate",
    }


def _sec_get_json(url: str, timeout: int = 12) -> dict | list:
    last_error = None
    for attempt in range(3):
        try:
            response = requests.get(url, headers=_sec_headers(), timeout=timeout)
            if response.status_code in {403, 429} and attempt < 2:
                time.sleep(1.0 + attempt)
                continue
            response.raise_for_status()
            return response.json()
        except Exception as e:
            last_error = e
            if attempt < 2:
                time.sleep(1.0 + attempt)
                continue
            break
    if last_error:
        raise last_error
    raise RuntimeError("SEC request failed without an explicit error.")


def _load_sec_ticker_cik_map() -> dict:
    global _sec_ticker_cik_map_cache
    if isinstance(_sec_ticker_cik_map_cache, dict) and _sec_ticker_cik_map_cache:
        return _sec_ticker_cik_map_cache

    mapping = dict(SEC_CIK_FALLBACK_MAP)
    url = "https://www.sec.gov/files/company_tickers.json"
    try:
        payload = _sec_get_json(url, timeout=12)

        if isinstance(payload, dict):
            rows = payload.values()
        elif isinstance(payload, list):
            rows = payload
        else:
            rows = []

        for row in rows:
            if not isinstance(row, dict):
                continue
            ticker = str(row.get("ticker", "")).strip().upper()
            cik_str = row.get("cik_str")
            if not ticker or cik_str in (None, ""):
                continue
            try:
                mapping[ticker] = int(cik_str)
            except Exception:
                continue
    except Exception:
        # Keep fallback mapping so major tickers still resolve in constrained envs.
        pass

    _sec_ticker_cik_map_cache = mapping
    return mapping


def _sec_is_quarterly_entry(entry: dict) -> bool:
    fp = str(entry.get("fp", "")).upper()
    frame = str(entry.get("frame", "")).upper()
    if fp in {"Q1", "Q2", "Q3", "Q4"}:
        return True
    if "Q" in frame and any(ch.isdigit() for ch in frame):
        return True
    return False


def _sec_duration_days(entry: dict):
    start_raw = entry.get("start")
    end_raw = entry.get("end")
    if not start_raw or not end_raw:
        return None
    try:
        start_dt = pd.to_datetime(start_raw)
        end_dt = pd.to_datetime(end_raw)
        days = int((end_dt - start_dt).days) + 1
        if days <= 0:
            return None
        return days
    except Exception:
        return None


def _sec_is_probably_quarter_duration(entry: dict) -> bool:
    days = _sec_duration_days(entry)
    if days is None:
        return False
    return 70 <= days <= 120


def _sec_is_annual_entry(entry: dict) -> bool:
    fp = str(entry.get("fp", "")).upper()
    frame = str(entry.get("frame", "")).upper()
    if fp == "FY":
        return True
    if frame and "Q" not in frame and re.search(r"(CY|FY)\d{4}", frame):
        return True
    return False


def _sec_is_probably_annual_duration(entry: dict) -> bool:
    days = _sec_duration_days(entry)
    if days is None:
        return False
    # Allow both 52-week and 53-week fiscal years.
    return 300 <= days <= 390


def _sec_pick_unit(units: dict, preferred: str = "") -> str:
    if not isinstance(units, dict) or not units:
        return ""
    if preferred and preferred in units:
        return preferred
    keys = list(units.keys())
    if preferred:
        for key in keys:
            if preferred.upper() in str(key).upper():
                return key
    return keys[0]


def _sec_pick_eps_unit(units: dict) -> str:
    if not isinstance(units, dict) or not units:
        return ""
    keys = list(units.keys())
    for key in keys:
        normalized = str(key).upper().replace(" ", "")
        if "USD" in normalized and "SHARE" in normalized:
            return key
    for key in keys:
        if "USD" in str(key).upper():
            return key
    return keys[0]


def _sec_extract_series(
    company_facts: dict,
    concepts: list[str],
    unit_kind: str = "USD",
    strict_duration: bool = True,
    period: str = "quarterly",
) -> dict:
    facts = company_facts.get("facts", {}) if isinstance(company_facts, dict) else {}
    selected_entries = {}

    for taxonomy in ("us-gaap", "ifrs-full"):
        taxonomy_facts = facts.get(taxonomy, {})
        if not isinstance(taxonomy_facts, dict):
            continue
        for concept_rank, concept in enumerate(concepts):
            concept_payload = taxonomy_facts.get(concept, {})
            if not isinstance(concept_payload, dict):
                continue
            units = concept_payload.get("units", {})
            if not isinstance(units, dict) or not units:
                continue

            if unit_kind == "EPS":
                unit_key = _sec_pick_eps_unit(units)
            else:
                unit_key = _sec_pick_unit(units, preferred=unit_kind)
            if not unit_key or unit_key not in units:
                continue

            entries = units.get(unit_key, [])
            if not isinstance(entries, list):
                continue

            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                if period == "annual":
                    if not _sec_is_annual_entry(entry):
                        continue
                    if strict_duration and not _sec_is_probably_annual_duration(entry):
                        continue
                else:
                    if not _sec_is_quarterly_entry(entry):
                        continue
                    if strict_duration and not _sec_is_probably_quarter_duration(entry):
                        continue
                end = entry.get("end")
                val = entry.get("val")
                if not end or val is None:
                    continue
                try:
                    _ = pd.to_datetime(end)
                    numeric_val = float(val)
                except Exception:
                    continue

                filed = str(entry.get("filed", ""))
                prior = selected_entries.get(end)
                if prior is None:
                    selected_entries[end] = {"val": numeric_val, "filed": filed, "rank": concept_rank}
                else:
                    if concept_rank < prior.get("rank", concept_rank):
                        selected_entries[end] = {"val": numeric_val, "filed": filed, "rank": concept_rank}
                    elif concept_rank == prior.get("rank", concept_rank) and filed > str(prior.get("filed", "")):
                        selected_entries[end] = {"val": numeric_val, "filed": filed, "rank": concept_rank}

    return {end: payload["val"] for end, payload in selected_entries.items()}


def _sec_backfill_q4_from_annual(quarterly_series: dict, annual_series: dict) -> tuple[dict, list]:
    """
    For calendar-year reporters, derive missing Q4 values from:
    FY total - (Q1 + Q2 + Q3).
    Applies only to additive metrics (e.g., revenue, operating income).
    """
    q_series = dict(quarterly_series or {})
    annual = dict(annual_series or {})
    if not q_series or not annual:
        return q_series, []

    quarter_bucket_map = {}
    for date_key, value in q_series.items():
        bucket = _quarter_bucket_from_date_key(date_key)
        if bucket is None or bucket in quarter_bucket_map:
            continue
        quarter_bucket_map[bucket] = (date_key, float(value))

    derived_rows = []
    for annual_date, annual_value in annual.items():
        target_bucket = _quarter_bucket_from_date_key(annual_date)
        if target_bucket is None:
            continue
        # Keep this aligned with calendar-quarter labels used in the UI.
        if target_bucket[1] != 4:
            continue
        if target_bucket in quarter_bucket_map:
            continue

        q3_bucket = _previous_quarter_bucket(target_bucket)
        q2_bucket = _previous_quarter_bucket(q3_bucket)
        q1_bucket = _previous_quarter_bucket(q2_bucket)
        if q1_bucket is None or q2_bucket is None or q3_bucket is None:
            continue

        if q1_bucket not in quarter_bucket_map or q2_bucket not in quarter_bucket_map or q3_bucket not in quarter_bucket_map:
            continue

        q1_val = quarter_bucket_map[q1_bucket][1]
        q2_val = quarter_bucket_map[q2_bucket][1]
        q3_val = quarter_bucket_map[q3_bucket][1]
        try:
            derived_val = float(annual_value) - (float(q1_val) + float(q2_val) + float(q3_val))
        except Exception:
            continue
        if pd.isna(derived_val):
            continue

        q_series[annual_date] = float(derived_val)
        quarter_bucket_map[target_bucket] = (annual_date, float(derived_val))
        derived_rows.append({
            "quarter": f"{target_bucket[0]}-Q{target_bucket[1]}",
            "date": annual_date,
            "value": float(derived_val),
        })

    return q_series, derived_rows


def _get_quarterly_income_history_sec(ticker_symbol: str, max_quarters: int = 20) -> pd.DataFrame:
    ticker = str(ticker_symbol or "").strip().upper()
    if not ticker:
        empty = pd.DataFrame()
        empty.attrs["sec_error"] = "Missing ticker symbol."
        return empty

    try:
        cik_map = _load_sec_ticker_cik_map()
        cik = cik_map.get(ticker)
        if cik is None:
            empty = pd.DataFrame()
            empty.attrs["sec_error"] = f"Ticker-to-CIK mapping unavailable for {ticker}."
            return empty

        cik_padded = f"{int(cik):010d}"
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_padded}.json"
        company_facts = _sec_get_json(url, timeout=15)

        def _extract_with_relaxed_fallback(concepts: list[str], unit_kind: str = "USD", min_points: int = 4) -> dict:
            strict_series = _sec_extract_series(
                company_facts,
                concepts=concepts,
                unit_kind=unit_kind,
                strict_duration=True,
                period="quarterly",
            )
            if len(strict_series) >= min_points:
                return strict_series
            relaxed_series = _sec_extract_series(
                company_facts,
                concepts=concepts,
                unit_kind=unit_kind,
                strict_duration=False,
                period="quarterly",
            )
            return relaxed_series if len(relaxed_series) > len(strict_series) else strict_series

        def _extract_annual_with_relaxed_fallback(concepts: list[str], unit_kind: str = "USD") -> dict:
            strict_series = _sec_extract_series(
                company_facts,
                concepts=concepts,
                unit_kind=unit_kind,
                strict_duration=True,
                period="annual",
            )
            relaxed_series = _sec_extract_series(
                company_facts,
                concepts=concepts,
                unit_kind=unit_kind,
                strict_duration=False,
                period="annual",
            )
            return relaxed_series if len(relaxed_series) > len(strict_series) else strict_series

        revenue_quarterly_series = _extract_with_relaxed_fallback(
            concepts=[
                "Revenues",
                "RevenueFromContractWithCustomerExcludingAssessedTax",
                "RevenueFromContractWithCustomerIncludingAssessedTax",
                "SalesRevenueNet",
            ],
            unit_kind="USD",
        )
        annual_revenue_series = _extract_annual_with_relaxed_fallback(
            concepts=[
                "Revenues",
                "RevenueFromContractWithCustomerExcludingAssessedTax",
                "RevenueFromContractWithCustomerIncludingAssessedTax",
                "SalesRevenueNet",
            ],
            unit_kind="USD",
        )
        revenue_series, revenue_q4_derived = _sec_backfill_q4_from_annual(
            revenue_quarterly_series,
            annual_revenue_series,
        )

        operating_income_quarterly_series = _extract_with_relaxed_fallback(
            concepts=["OperatingIncomeLoss"],
            unit_kind="USD",
        )
        annual_operating_income_series = _extract_annual_with_relaxed_fallback(
            concepts=["OperatingIncomeLoss"],
            unit_kind="USD",
        )
        operating_income_series, operating_income_q4_derived = _sec_backfill_q4_from_annual(
            operating_income_quarterly_series,
            annual_operating_income_series,
        )

        basic_eps_series = _extract_with_relaxed_fallback(
            concepts=["EarningsPerShareBasic", "BasicEarningsPerShare"],
            unit_kind="EPS",
        )
        diluted_eps_series = _extract_with_relaxed_fallback(
            concepts=["EarningsPerShareDiluted", "DilutedEarningsPerShare"],
            unit_kind="EPS",
        )

        all_dates = set(revenue_series.keys()) | set(operating_income_series.keys()) | set(basic_eps_series.keys()) | set(diluted_eps_series.keys())
        if not all_dates:
            empty = pd.DataFrame()
            empty.attrs["sec_error"] = f"No structured quarterly facts returned by SEC for {ticker}."
            return empty

        ordered_dates = sorted(all_dates, key=lambda d: pd.to_datetime(d), reverse=True)
        if max_quarters and len(ordered_dates) > max_quarters:
            ordered_dates = ordered_dates[:max_quarters]

        columns = [pd.to_datetime(d) for d in ordered_dates]
        data = {
            "Total Revenue": [revenue_series.get(d) for d in ordered_dates],
            "Operating Income": [operating_income_series.get(d) for d in ordered_dates],
            "Basic EPS": [basic_eps_series.get(d) for d in ordered_dates],
            "Diluted EPS": [diluted_eps_series.get(d) for d in ordered_dates],
        }

        df = pd.DataFrame(data, index=columns).T
        if df.empty:
            empty = pd.DataFrame()
            empty.attrs["sec_error"] = f"SEC facts response produced an empty frame for {ticker}."
            return empty

        backfill_quarters = sorted(
            set([r.get("quarter") for r in revenue_q4_derived + operating_income_q4_derived if r.get("quarter")]),
            reverse=True,
        )
        df.attrs["sec_q4_backfilled"] = {
            "revenue_q4_derived": len(revenue_q4_derived),
            "operating_income_q4_derived": len(operating_income_q4_derived),
            "total_q4_derived": len(backfill_quarters),
            "quarters": backfill_quarters,
        }
        return df
    except Exception as e:
        empty = pd.DataFrame()
        empty.attrs["sec_error"] = _redact_api_secrets(str(e))
        return empty


def _date_key_from_column(col) -> str:
    try:
        dt = pd.to_datetime(col, errors="coerce")
        if pd.isna(dt):
            return ""
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return ""


def _extract_date_keys(df: pd.DataFrame) -> list[str]:
    if df is None or df.empty:
        return []
    keys = []
    seen = set()
    for col in _ordered_quarter_columns(df):
        key = _date_key_from_column(col)
        if not key or key in seen:
            continue
        seen.add(key)
        keys.append(key)
    return keys


def _quarter_bucket_from_date_key(date_key: str):
    try:
        dt = pd.to_datetime(date_key, errors="coerce")
        if pd.isna(dt):
            return None
        return int(dt.year), int((int(dt.month) - 1) // 3 + 1)
    except Exception:
        return None


def _extract_quarter_bucket_dates(df: pd.DataFrame) -> dict:
    """
    Map each quarter bucket (year, quarter) to a canonical date key from that source.
    Uses the first seen date in descending order (most recent in bucket).
    """
    if df is None or df.empty:
        return {}
    bucket_dates = {}
    for col in _ordered_quarter_columns(df):
        date_key = _date_key_from_column(col)
        if not date_key:
            continue
        bucket = _quarter_bucket_from_date_key(date_key)
        if bucket is None or bucket in bucket_dates:
            continue
        bucket_dates[bucket] = date_key
    return bucket_dates


def _previous_quarter_bucket(bucket):
    if not isinstance(bucket, tuple) or len(bucket) != 2:
        return None
    year, quarter = bucket
    try:
        y = int(year)
        q = int(quarter)
    except Exception:
        return None
    if q <= 1:
        return y - 1, 4
    return y, q - 1


def _quarter_end_date_from_bucket(bucket) -> str:
    if not isinstance(bucket, tuple) or len(bucket) != 2:
        return ""
    year, quarter = bucket
    try:
        period = pd.Period(f"{int(year)}Q{int(quarter)}", freq="Q")
        return str(period.end_time.date())
    except Exception:
        return ""


def _metric_row(df: pd.DataFrame, keys: list[str]) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype="float64")
    for key in keys:
        if key in df.index:
            return pd.to_numeric(df.loc[key], errors="coerce")
    return pd.Series(dtype="float64")


def _metric_map(df: pd.DataFrame, keys: list[str]) -> dict:
    row = _metric_row(df, keys)
    if row.empty:
        return {}
    out = {}
    for col in _ordered_quarter_columns(df):
        date_key = _date_key_from_column(col)
        if not date_key:
            continue
        bucket = _quarter_bucket_from_date_key(date_key)
        if bucket is None:
            continue
        if bucket in out:
            continue
        val = row.get(col)
        if pd.notna(val):
            out[bucket] = float(val)
    return out


def _values_conflict(v1: float, v2: float, is_eps: bool = False) -> bool:
    if v1 is None or v2 is None:
        return False
    if is_eps:
        return abs(v1 - v2) > max(0.03, abs(v1) * 0.08)
    denom = max(1.0, abs(v1), abs(v2))
    return abs(v1 - v2) / denom > 0.08


def _merge_yahoo_sec_quarterly_income(
    yahoo_df: pd.DataFrame,
    sec_df: pd.DataFrame,
    max_quarters: int = 20,
) -> tuple[pd.DataFrame, dict]:
    yahoo_raw_dates = _extract_date_keys(yahoo_df)
    sec_raw_dates = _extract_date_keys(sec_df)
    yahoo_bucket_dates = _extract_quarter_bucket_dates(yahoo_df)
    sec_bucket_dates = _extract_quarter_bucket_dates(sec_df)
    merged_buckets = sorted(set(yahoo_bucket_dates) | set(sec_bucket_dates), reverse=True)
    if max_quarters and len(merged_buckets) > max_quarters:
        merged_buckets = merged_buckets[:max_quarters]

    metric_specs = [
        ("Total Revenue", ["Total Revenue", "Operating Revenue", "Revenue", "Revenues"], False),
        ("Operating Income", ["Operating Income", "EBIT", "OperatingIncome"], False),
        ("Basic EPS", ["Basic EPS", "Diluted EPS", "Normalized EPS", "EPS"], True),
        ("Diluted EPS", ["Diluted EPS", "Basic EPS", "Normalized EPS", "EPS"], True),
    ]

    merged_data = {}
    mismatches = []
    for metric_name, keys, is_eps in metric_specs:
        y_map = _metric_map(yahoo_df, keys)
        s_map = _metric_map(sec_df, keys)
        merged_vals = []
        for bucket in merged_buckets:
            y_val = y_map.get(bucket)
            s_val = s_map.get(bucket)
            chosen = None
            if y_val is not None and s_val is not None:
                if _values_conflict(y_val, s_val, is_eps=is_eps):
                    y_date = yahoo_bucket_dates.get(bucket)
                    s_date = sec_bucket_dates.get(bucket)
                    sampled_date = y_date or s_date or ""
                    mismatches.append({
                        "date": sampled_date,
                        "quarter": f"{bucket[0]}-Q{bucket[1]}",
                        "metric": metric_name,
                        "yahoo": y_val,
                        "sec": s_val,
                    })
                # User requirement: prioritize Yahoo when sources differ.
                chosen = y_val
            elif y_val is not None:
                chosen = y_val
            elif s_val is not None:
                chosen = s_val
            merged_vals.append(chosen)
        merged_data[metric_name] = merged_vals

    if not merged_buckets:
        return pd.DataFrame(), {
            "yahoo_quarters": len(yahoo_bucket_dates),
            "sec_quarters": len(sec_bucket_dates),
            "merged_quarters": 0,
            "yahoo_raw_dates": len(yahoo_raw_dates),
            "sec_raw_dates": len(sec_raw_dates),
            "yahoo_date_collisions_collapsed": max(0, len(yahoo_raw_dates) - len(yahoo_bucket_dates)),
            "sec_date_collisions_collapsed": max(0, len(sec_raw_dates) - len(sec_bucket_dates)),
            "mismatch_points": 0,
            "mismatch_samples": [],
        }

    merged_date_keys = []
    for bucket in merged_buckets:
        y_date = yahoo_bucket_dates.get(bucket)
        s_date = sec_bucket_dates.get(bucket)
        candidates = [d for d in [y_date, s_date] if d]
        if not candidates:
            continue
        try:
            canonical_date = max(candidates, key=lambda d: pd.to_datetime(d, errors="coerce"))
        except Exception:
            canonical_date = candidates[0]
        merged_date_keys.append(canonical_date)

    merged_columns = [pd.to_datetime(d) for d in merged_date_keys]
    merged_df = pd.DataFrame(merged_data, index=merged_columns).T
    if merged_df.empty:
        return pd.DataFrame(), {
            "yahoo_quarters": len(yahoo_bucket_dates),
            "sec_quarters": len(sec_bucket_dates),
            "merged_quarters": 0,
            "yahoo_raw_dates": len(yahoo_raw_dates),
            "sec_raw_dates": len(sec_raw_dates),
            "yahoo_date_collisions_collapsed": max(0, len(yahoo_raw_dates) - len(yahoo_bucket_dates)),
            "sec_date_collisions_collapsed": max(0, len(sec_raw_dates) - len(sec_bucket_dates)),
            "mismatch_points": len(mismatches),
            "mismatch_samples": mismatches[:5],
        }

    return merged_df, {
        "yahoo_quarters": len(yahoo_bucket_dates),
        "sec_quarters": len(sec_bucket_dates),
        "merged_quarters": len(merged_buckets),
        "yahoo_raw_dates": len(yahoo_raw_dates),
        "sec_raw_dates": len(sec_raw_dates),
        "yahoo_date_collisions_collapsed": max(0, len(yahoo_raw_dates) - len(yahoo_bucket_dates)),
        "sec_date_collisions_collapsed": max(0, len(sec_raw_dates) - len(sec_bucket_dates)),
        "mismatch_points": len(mismatches),
        "mismatch_samples": mismatches[:5],
    }


def _get_quarterly_income_history(ticker_symbol: str, max_quarters: int = 20) -> tuple[pd.DataFrame, str, dict]:
    sec_quarterly_income = _get_quarterly_income_history_sec(ticker_symbol, max_quarters=max_quarters)
    sec_error = sec_quarterly_income.attrs.get("sec_error") if hasattr(sec_quarterly_income, "attrs") else None

    yahoo_quarterly_income = pd.DataFrame()
    try:
        stock = yf.Ticker(ticker_symbol)
        yahoo_quarterly_income = stock.quarterly_income_stmt
    except Exception:
        yahoo_quarterly_income = pd.DataFrame()

    merged_df, diagnostics = _merge_yahoo_sec_quarterly_income(
        yahoo_df=yahoo_quarterly_income,
        sec_df=sec_quarterly_income,
        max_quarters=max_quarters,
    )
    if isinstance(diagnostics, dict):
        sec_q4_backfilled = sec_quarterly_income.attrs.get("sec_q4_backfilled") if hasattr(sec_quarterly_income, "attrs") else None
        if isinstance(sec_q4_backfilled, dict) and sec_q4_backfilled.get("total_q4_derived", 0):
            diagnostics["sec_q4_backfilled"] = sec_q4_backfilled
        if sec_error:
            diagnostics["sec_error"] = str(sec_error)
    if not merged_df.empty:
        if not yahoo_quarterly_income.empty and not sec_quarterly_income.empty:
            source = "Yahoo + SEC (Yahoo-priority)"
        elif not yahoo_quarterly_income.empty:
            source = "Yahoo Finance"
        elif not sec_quarterly_income.empty:
            source = "SEC Company Facts"
        else:
            source = "Unavailable"
        return merged_df, source, diagnostics

    if not yahoo_quarterly_income.empty:
        return yahoo_quarterly_income, "Yahoo Finance", diagnostics
    if not sec_quarterly_income.empty:
        return sec_quarterly_income, "SEC Company Facts", diagnostics
    return pd.DataFrame(), "Unavailable", diagnostics


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
            error_detail = _redact_api_secrets(str(e), fmp_api_key)
            print(f"FMP API failed for {ticker}. Falling back to yfinance.")
            warning_message = (
                f"⚠️ FMP API error: {error_detail}. "
                "Using limited yfinance data (5-8 quarters). Check your FMP API key or try again."
            )
    
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
    
    def _fmp_statement_requests():
        if statement_type == 'income':
            return [
                (
                    f"https://financialmodelingprep.com/api/v3/income-statement/{ticker}",
                    {"period": "quarter", "limit": limit},
                ),
                (
                    "https://financialmodelingprep.com/stable/income-statement",
                    {"symbol": ticker, "period": "quarter", "limit": limit},
                ),
            ]
        if statement_type == 'balance':
            return [
                (
                    f"https://financialmodelingprep.com/api/v3/balance-sheet-statement/{ticker}",
                    {"period": "quarter", "limit": limit},
                ),
                (
                    "https://financialmodelingprep.com/stable/balance-sheet-statement",
                    {"symbol": ticker, "period": "quarter", "limit": limit},
                ),
            ]
        return []

    def _extract_records(payload):
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ("data", "results", "financials"):
                candidate = payload.get(key)
                if isinstance(candidate, list):
                    return candidate
        return []

    request_specs = _fmp_statement_requests()
    if not request_specs:
        return pd.DataFrame()

    last_error_detail = None

    for url, base_params in request_specs:
        # Try both documented auth modes: header first, query-param second.
        for auth_mode in ("header", "query"):
            params = dict(base_params or {})
            headers = {}
            if auth_mode == "header":
                headers["apikey"] = fmp_api_key
            else:
                params["apikey"] = fmp_api_key

            try:
                response = requests.get(url, params=params, headers=headers, timeout=10)
                payload = response.json()

                records = _extract_records(payload)
                if not records:
                    if isinstance(payload, dict):
                        message = (
                            payload.get("Error Message")
                            or payload.get("error")
                            or payload.get("message")
                            or payload.get("Note")
                        )
                        if message:
                            last_error_detail = str(message)
                    continue

                # Convert to DataFrame with structure similar to yfinance
                df = pd.DataFrame(records)
                if df.empty or 'date' not in df.columns:
                    continue

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
                else:
                    field_mapping = {
                        'totalAssets': 'Total Assets',
                        'totalStockholdersEquity': 'Stockholders Equity',
                        'totalEquity': 'Total Equity Gross Minority Interest'
                    }

                # Rename fields
                df.index = df.index.map(lambda x: field_mapping.get(x, x))
                return df
            except Exception as e:
                last_error_detail = str(e)

    if last_error_detail:
        safe_error_detail = _redact_api_secrets(last_error_detail, fmp_api_key)
        print(f"FMP API error for {ticker}: {safe_error_detail}")
    else:
        print(f"FMP API error for {ticker}.")
    return pd.DataFrame()


def analyze_quarterly_trends(ticker_symbol: str, num_quarters: int = 8, end_date: str = None) -> dict:
    """
    Analyzes historical quarterly trends and fetches consensus estimates.
    Uses FMP extended quarterly history when FMP_API_KEY is available, otherwise Yahoo Finance.
    
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
        pe_ratio = None
        try:
            stock_info = yf.Ticker(ticker_symbol).info
            current_price = stock_info.get('currentPrice') or stock_info.get('regularMarketPrice')
            shares_outstanding = stock_info.get('sharesOutstanding')
            current_market_cap = stock_info.get('marketCap')
            pe_ratio = (
                stock_info.get('trailingPE')
                or stock_info.get('forwardPE')
                or stock_info.get('priceToEarnings')
            )
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
            "market_cap": current_market_cap,
            "pe_ratio": pe_ratio
        }
        
        # --- PART 1: Historical Quarterly Data ---
        requested_quarters = max(4, int(num_quarters))
        quarterly_income, data_source, source_diagnostics = _get_quarterly_income_history(
            ticker_symbol,
            max_quarters=max(20, requested_quarters),
        )
        result["data_source"] = data_source
        if isinstance(source_diagnostics, dict):
            result["historical_trends"]["source_diagnostics"] = source_diagnostics
        
        if quarterly_income.empty:
            result["errors"].append("No quarterly income statement data available")
            return result
        
        # Get available quarters and filter by user selection
        all_quarters = _ordered_quarter_columns(quarterly_income)
        if not all_quarters:
            result["errors"].append("Quarterly history is present but no valid quarter columns were found")
            return result
        if data_source.startswith("Yahoo + SEC"):
            mismatch_points = source_diagnostics.get("mismatch_points", 0) if isinstance(source_diagnostics, dict) else 0
            if mismatch_points:
                result["warning"] = (
                    f"Cross-source checks found {mismatch_points} metric mismatches between Yahoo and SEC. "
                    "Yahoo values were prioritized where overlaps differed."
                )
            sec_q4_backfilled = source_diagnostics.get("sec_q4_backfilled", {}) if isinstance(source_diagnostics, dict) else {}
            if isinstance(sec_q4_backfilled, dict) and sec_q4_backfilled.get("total_q4_derived", 0):
                derived_quarters = sec_q4_backfilled.get("quarters", []) if isinstance(sec_q4_backfilled.get("quarters", []), list) else []
                preview = ", ".join(derived_quarters[:3]) if derived_quarters else ""
                if len(derived_quarters) > 3:
                    preview += ", ..."
                backfill_note = (
                    f"SEC annual-minus-Q1..Q3 backfill filled missing Q4 for {sec_q4_backfilled.get('total_q4_derived')} quarter(s)"
                    + (f" ({preview})." if preview else ".")
                )
                result["warning"] = ((result.get("warning") + " ") if result.get("warning") else "") + backfill_note
            if len(all_quarters) < requested_quarters:
                result["warning"] = (
                    (result.get("warning") + " " if result.get("warning") else "")
                    + f"Merged Yahoo+SEC history currently has {len(all_quarters)} quarters for {ticker_symbol}."
                )
        elif data_source.startswith("SEC Company Facts") and len(all_quarters) < requested_quarters:
            result["warning"] = (
                f"SEC Company Facts returned {len(all_quarters)} quarterly reports for {ticker_symbol}. "
                "This is all currently available in structured XBRL for this ticker."
            )
        elif data_source.startswith("Yahoo Finance") and len(all_quarters) < requested_quarters:
            sec_error = source_diagnostics.get("sec_error") if isinstance(source_diagnostics, dict) else None
            sec_error_note = ""
            if sec_error:
                sec_error_text = str(sec_error).strip()
                if len(sec_error_text) > 180:
                    sec_error_text = sec_error_text[:177] + "..."
                sec_error_note = f" SEC fetch issue: {sec_error_text}"
            result["warning"] = (
                f"Yahoo Finance returned {len(all_quarters)} quarterly reports for {ticker_symbol}. "
                f"SEC data was unavailable for additional merge coverage.{sec_error_note}"
            )
        
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
        
        # Build a bucket map so quarterly windows stay contiguous even when raw report dates are uneven.
        bucket_to_col = {}
        for q_col in all_quarters:
            bucket = _quarter_bucket_from_date_key(str(q_col)[:10])
            if bucket is None or bucket in bucket_to_col:
                continue
            bucket_to_col[bucket] = q_col

        anchor_col = all_quarters[0]
        if end_date:
            exact_match = None
            for q_col in all_quarters:
                if str(q_col)[:10] == end_date:
                    exact_match = q_col
                    break
            if exact_match is not None:
                anchor_col = exact_match
            else:
                end_bucket = _quarter_bucket_from_date_key(end_date)
                if end_bucket in bucket_to_col:
                    anchor_col = bucket_to_col[end_bucket]

        selected_year = anchor_col.year if hasattr(anchor_col, 'year') else None
        selected_q_num = (anchor_col.month - 1) // 3 + 1 if hasattr(anchor_col, 'month') else None
        if selected_year and selected_q_num:
            result["historical_trends"]["most_recent_quarter"] = {
                "year": selected_year,
                "quarter": selected_q_num,
                "label": f"FY{selected_year} Q{selected_q_num}",
                "date": str(anchor_col)[:10]
            }
            most_recent_year = selected_year
            most_recent_q = selected_q_num

        anchor_bucket = _quarter_bucket_from_date_key(str(anchor_col)[:10])
        expected_buckets = []
        cursor_bucket = anchor_bucket
        while cursor_bucket is not None and len(expected_buckets) < requested_quarters:
            expected_buckets.append(cursor_bucket)
            cursor_bucket = _previous_quarter_bucket(cursor_bucket)
        if not expected_buckets:
            expected_buckets = [
                _quarter_bucket_from_date_key(str(q_col)[:10])
                for q_col in all_quarters[:requested_quarters]
                if _quarter_bucket_from_date_key(str(q_col)[:10]) is not None
            ]
        
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
        def safe_get(df, keys, col):
            key_list = keys if isinstance(keys, (list, tuple)) else [keys]
            try:
                for key in key_list:
                    if key in df.index:
                        val = df.loc[key, col]
                        if pd.notna(val):
                            return float(val)
            except:
                pass
            return None
        
        quarterly_data = []
        missing_report_quarters = []
        for bucket in expected_buckets:
            q_col = bucket_to_col.get(bucket)
            q_label = f"{bucket[0]}-Q{bucket[1]}"
            q_date = str(q_col)[:10] if q_col is not None else _quarter_end_date_from_bucket(bucket)

            revenue = safe_get(quarterly_income, ['Total Revenue', 'Operating Revenue', 'Revenue'], q_col) if q_col is not None else None
            op_income = safe_get(quarterly_income, ['Operating Income', 'EBIT'], q_col) if q_col is not None else None
            eps = safe_get(quarterly_income, ['Basic EPS', 'Diluted EPS', 'Normalized EPS', 'EPS'], q_col) if q_col is not None else None

            if q_col is None:
                missing_report_quarters.append(q_label)

            quarterly_data.append({
                "quarter": q_label,
                "date": q_date,
                "revenue": revenue,
                "operating_income": op_income,
                "eps": eps
            })
        
        result["historical_trends"]["quarterly_data"] = quarterly_data
        result["historical_trends"]["quarters_available"] = len(quarterly_data)
        quarters_with_revenue = sum(1 for q in quarterly_data if q.get("revenue") is not None)
        quarters_with_operating_income = sum(1 for q in quarterly_data if q.get("operating_income") is not None)
        quarters_with_eps = sum(1 for q in quarterly_data if q.get("eps") is not None)
        quarters_with_complete_metrics = sum(
            1
            for q in quarterly_data
            if q.get("revenue") is not None and q.get("operating_income") is not None and q.get("eps") is not None
        )
        result["historical_trends"]["data_coverage"] = {
            "quarters_requested": requested_quarters,
            "quarters_returned": len(quarterly_data),
            "quarters_found_in_source_window": len(quarterly_data) - len(missing_report_quarters),
            "quarters_with_revenue": quarters_with_revenue,
            "quarters_with_operating_income": quarters_with_operating_income,
            "quarters_with_eps": quarters_with_eps,
            "quarters_with_complete_metrics": quarters_with_complete_metrics,
            "missing_report_quarters": missing_report_quarters,
            "missing_revenue_quarters": [q.get("quarter") for q in quarterly_data if q.get("revenue") is None],
            "missing_eps_quarters": [q.get("quarter") for q in quarterly_data if q.get("eps") is None],
        }
        if missing_report_quarters:
            preview = ", ".join(missing_report_quarters[:3])
            if len(missing_report_quarters) > 3:
                preview += ", ..."
            result["warning"] = (
                (result.get("warning") + " " if result.get("warning") else "")
                + f"Source history has {len(missing_report_quarters)} missing report quarter(s) in this window ({preview})."
            )
        
        # --- PART 2: Calculate Growth Rates ---
        def calc_growth(current, previous):
            if current is None or previous is None or previous == 0:
                return None
            return ((current - previous) / abs(previous)) * 100

        def quarter_num_from_entry(entry: dict):
            date_raw = entry.get("date")
            try:
                dt = pd.to_datetime(date_raw)
                if pd.notna(dt):
                    return (int(dt.month) - 1) // 3 + 1
            except Exception:
                pass
            label = str(entry.get("quarter", ""))
            match = re.search(r"Q([1-4])", label)
            if match:
                return int(match.group(1))
            return None

        def compute_seasonality_signal(rows: list):
            revenue_points = [r for r in rows if r.get("revenue") is not None]
            if len(revenue_points) < MIN_REVENUE_POINTS_FOR_SEASONALITY:
                return {
                    "pattern": "N/A",
                    "confidence": "low",
                    "reason": f"Need at least {MIN_REVENUE_POINTS_FOR_SEASONALITY} revenue quarters for seasonality analysis.",
                    "quarter_effects": {},
                }

            # Build chronological transition series so each quarter captures its typical sequential uplift/drop.
            chrono = list(reversed(rows))
            qoq_by_quarter = {1: [], 2: [], 3: [], 4: []}
            all_qoq = []
            for i in range(1, len(chrono)):
                prev_q = chrono[i - 1]
                curr_q = chrono[i]
                quarter_num = quarter_num_from_entry(curr_q)
                qoq = calc_growth(curr_q.get("revenue"), prev_q.get("revenue"))
                if quarter_num is None or qoq is None:
                    continue
                qoq_by_quarter[quarter_num].append(float(qoq))
                all_qoq.append(float(qoq))

            quarter_effects = {}
            for q in (1, 2, 3, 4):
                vals = qoq_by_quarter[q]
                if len(vals) >= MIN_TRANSITIONS_PER_QUARTER_FOR_SEASONALITY:
                    avg_qoq = sum(vals) / len(vals)
                    pos_ratio = sum(1 for v in vals if v > 0) / len(vals)
                    quarter_effects[q] = {
                        "count": len(vals),
                        "avg_qoq": round(avg_qoq, 2),
                        "positive_ratio": round(pos_ratio, 2),
                    }

            if len(quarter_effects) < 2:
                return {
                    "pattern": "N/A",
                    "confidence": "low",
                    "reason": "Insufficient repeated quarter transitions (need at least two years of comparable quarter-over-quarter transitions).",
                    "quarter_effects": quarter_effects,
                }

            dominant_up_q = max(quarter_effects.keys(), key=lambda q: quarter_effects[q]["avg_qoq"])
            dominant_down_q = min(quarter_effects.keys(), key=lambda q: quarter_effects[q]["avg_qoq"])
            dominant_up = quarter_effects[dominant_up_q]
            dominant_down = quarter_effects[dominant_down_q]
            spread = dominant_up["avg_qoq"] - dominant_down["avg_qoq"]

            quarter_label = {1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4"}
            pattern = "Mixed Seasonality"
            confidence = "low"
            reason = "Quarter effects are present but no single quarter effect is consistently dominant."

            if spread < 4:
                pattern = "Low Seasonality"
                confidence = "medium"
                reason = "Quarter-over-quarter revenue changes are relatively even across quarters."
            elif (
                dominant_up["avg_qoq"] >= 6
                and dominant_up["positive_ratio"] >= 0.67
                and spread >= 8
            ):
                pattern = "Strong Q4" if dominant_up_q == 4 else f"{quarter_label[dominant_up_q]} Seasonal Lift"
                confidence = "high" if dominant_up["count"] >= 3 else "medium"
                reason = (
                    f"{quarter_label[dominant_up_q]} shows the strongest and most consistent sequential uplift "
                    f"(avg QoQ {dominant_up['avg_qoq']:+.1f}%, n={dominant_up['count']})."
                )
            elif (
                dominant_down["avg_qoq"] <= -6
                and dominant_down["positive_ratio"] <= 0.33
                and spread >= 8
            ):
                if dominant_down_q == 4:
                    pattern = "Weak Q4"
                elif dominant_down_q == 1:
                    pattern = "Post-Holiday Dip (Q1)"
                else:
                    pattern = f"{quarter_label[dominant_down_q]} Seasonal Dip"
                confidence = "high" if dominant_down["count"] >= 3 else "medium"
                reason = (
                    f"{quarter_label[dominant_down_q]} shows the weakest and most consistent sequential movement "
                    f"(avg QoQ {dominant_down['avg_qoq']:+.1f}%, n={dominant_down['count']})."
                )

            return {
                "pattern": pattern,
                "confidence": confidence,
                "reason": reason,
                "spread_qoq_pct": round(spread, 2),
                "quarter_effects": quarter_effects,
            }
        
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
            "avg_revenue_yoy": (
                round(sum(yoy_revenues) / len(yoy_revenues), 2)
                if len(yoy_revenues) >= MIN_YOY_PAIRS_FOR_AVG_GROWTH
                else None
            ),
            "avg_eps_yoy": (
                round(sum(yoy_eps) / len(yoy_eps), 2)
                if len(yoy_eps) >= MIN_YOY_PAIRS_FOR_AVG_GROWTH
                else None
            ),
            "samples_used": len(quarterly_data),
            "quarters_with_revenue": quarters_with_revenue,
            "quarters_with_eps": quarters_with_eps,
            "quarters_with_complete_metrics": quarters_with_complete_metrics,
            "revenue_yoy_pairs": len(yoy_revenues),
            "eps_yoy_pairs": len(yoy_eps),
        }
        result["growth_rates"]["seasonality"] = compute_seasonality_signal(quarterly_data)
        
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
                
                if last_year_same_q["revenue"] is not None and avg_rev_growth is not None:
                    projected["projected_revenue"] = round(
                        last_year_same_q["revenue"] * (1 + avg_rev_growth / 100), 0
                    )
                    projected["revenue_growth_rate_used"] = avg_rev_growth
                
                if last_year_same_q["eps"] is not None and avg_eps_growth is not None:
                    projected["projected_eps"] = round(
                        last_year_same_q["eps"] * (1 + avg_eps_growth / 100), 2
                    )
                    projected["eps_growth_rate_used"] = avg_eps_growth
                
                result["projections"]["next_quarter_estimate"] = projected
        
        # --- PART 4: Fetch consensus estimates (fast path, no AI on initial load) ---
        next_q_label = result.get("next_forecast_quarter", {}).get("label", "next quarter")
        result["consensus_estimates"] = fetch_consensus_estimates(
            ticker_symbol,
            next_q_label,
            include_qualitative=False,
        )
        
    except Exception as e:
        result["errors"].append(f"Analysis error: {str(e)}")
    
    return result


def fetch_consensus_estimates(
    ticker_symbol: str,
    next_quarter_label: str = "next quarter",
    include_qualitative: bool = False,
) -> dict:
    """
    Fetches consensus analyst estimates directly from Yahoo Finance via yfinance.
    Optionally enriches with AI qualitative commentary.
    
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
        
        # Optional qualitative summary using AI (disabled for initial-load performance).
        try:
            if include_qualitative and config_genai():
                client, model_name = get_gemini_model()
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
                
                response = client.models.generate_content(model=model_name, contents=qual_prompt)
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
        except Exception:
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
    
    # Format historical data for the prompt (use broader history when available, cap for token discipline)
    quarterly_data = historical_data.get("quarterly_data", [])
    prompt_quarters = min(len(quarterly_data), 12)
    historical_summary = "\n".join([
        f"  {q['quarter']}: Revenue=${q['revenue']/1e9:.2f}B, Op.Income=${q['operating_income']/1e9:.2f}B, EPS=${q['eps']:.2f}" 
        if q['revenue'] and q['operating_income'] and q['eps'] 
        else f"  {q['quarter']}: Revenue={q['revenue']}, Op.Income={q['operating_income']}, EPS={q['eps']}"
        for q in quarterly_data[:prompt_quarters]
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
        client, model_name = get_gemini_model()
        
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

        8. DO NOT describe DCF intrinsic value as a "floor", "hard floor", or "guaranteed downside level".
           Treat it as one model output under one assumption set.
           Preferred wording: "model-implied intrinsic value under current assumptions."
        
        ═══════════════════════════════════════════════════════════════
        DATA PACKET FOR {ticker}
        Analysis Date: {quarterly_analysis.get('analysis_date', datetime.now().strftime('%Y-%m-%d'))}
        ═══════════════════════════════════════════════════════════════
        
        SECTION 1: HISTORICAL QUARTERLY PERFORMANCE (Most Recent {prompt_quarters} Quarters)
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
        
        response = client.models.generate_content(model=model_name, contents=forecast_prompt)
        response_text = _sanitize_valuation_language(response.text.strip())
        
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
            extract_response = client.models.generate_content(model=model_name, contents=extract_prompt)
            extract_text = extract_response.text.strip()
            if extract_text.startswith("```"):
                lines = extract_text.split("\n")
                extract_text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
            
            extracted = json.loads(extract_text)
            extracted = _sanitize_valuation_language(extracted)
            result["extracted_forecast"] = extracted
        except:
            result["extracted_forecast"] = None
        
        return result
        
    except Exception as e:
        return {
            "error": f"Failed to generate forecast: {str(e)}",
            "ticker": ticker
        }
