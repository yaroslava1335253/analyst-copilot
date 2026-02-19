# app.py
"""
Analyst Co-Pilot - v2.0
=======================
Features:
- Adversarial AI Debate (Bull vs Bear)
- DuPont Analysis & Trend Calculation
- Visual Dashboard
- Interactive specific Q&A
"""

import os
import streamlit as st
import pandas as pd
import json
# Re-importing new engine functions
from engine import get_financials, run_structured_prompt, calculate_metrics, run_chat, analyze_quarterly_trends, generate_independent_forecast

# --- App Configuration ---
st.set_page_config(
    page_title="Analyst Co-Pilot v2",
    page_icon="📈",
    layout="wide"
)

# --- Custom Professional Dark Theme CSS ---
st.markdown("""
<style>
    /* Import premium fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Playfair+Display:wght@400;500;600;700&display=swap');
    
    /* Root variables for dark theme */
    :root {
        --bg-primary: #0f0f0f;
        --bg-secondary: #1a1a1a;
        --bg-tertiary: #242424;
        --bg-card: #1e1e1e;
        --text-primary: #f5f5f5;
        --text-secondary: #a3a3a3;
        --text-muted: #737373;
        --accent-gold: #d4af37;
        --accent-emerald: #10b981;
        --accent-ruby: #ef4444;
        --accent-sapphire: #3b82f6;
        --border-color: #2a2a2a;
        --border-accent: #3a3a3a;
    }
    
    /* Main app background */
    .stApp {
        background: linear-gradient(180deg, var(--bg-primary) 0%, #0a0a0a 100%);
    }
    
    /* Main content area */
    .main .block-container {
        padding-top: 2rem;
        max-width: 1400px;
    }
    
    /* Headers with fancy font */
    h1, h2, h3 {
        font-family: 'Playfair Display', Georgia, serif !important;
        color: var(--text-primary) !important;
        letter-spacing: -0.02em;
    }
    
    h1 {
        font-size: 2.5rem !important;
        font-weight: 600 !important;
        background: linear-gradient(135deg, #f5f5f5 0%, var(--accent-gold) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    h2 {
        font-size: 1.75rem !important;
        font-weight: 500 !important;
        border-bottom: 1px solid var(--border-accent);
        padding-bottom: 0.5rem;
    }
    
    h3 {
        font-size: 1.25rem !important;
        color: var(--accent-gold) !important;
    }
    
    /* Body text */
    p, span, label, .stMarkdown {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: var(--text-secondary);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #141414 0%, #0a0a0a 100%);
        border-right: 1px solid var(--border-color);
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: var(--text-primary) !important;
        -webkit-text-fill-color: var(--text-primary) !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: var(--text-secondary);
    }
    
    /* Metric cards */
    [data-testid="stMetric"] {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    [data-testid="stMetricLabel"] {
        font-family: 'Inter', sans-serif !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        color: var(--text-muted) !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
    }
    
    [data-testid="stMetricDelta"] {
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--bg-secondary);
        border-radius: 12px;
        padding: 4px;
        gap: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-family: 'Inter', sans-serif !important;
        font-weight: 500;
        color: var(--text-muted);
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text-primary);
        background: var(--bg-tertiary);
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--bg-tertiary) !important;
        color: var(--accent-gold) !important;
        border-bottom: none !important;
    }
    
    /* Buttons */
    .stButton > button {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600;
        background: linear-gradient(135deg, var(--accent-gold) 0%, #b8962e 100%);
        color: #0a0a0a;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(212, 175, 55, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(212, 175, 55, 0.4);
    }
    
    .stButton > button[kind="secondary"] {
        background: var(--bg-tertiary);
        color: var(--text-primary);
        border: 1px solid var(--border-accent);
    }
    
    /* Input fields */
    .stTextInput > div > div > input {
        font-family: 'Inter', sans-serif !important;
        background: var(--bg-tertiary);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        color: var(--text-primary);
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--accent-gold);
        box-shadow: 0 0 0 2px rgba(212, 175, 55, 0.2);
    }
    
    /* DataFrames */
    .stDataFrame {
        font-family: 'JetBrains Mono', monospace !important;
        border-radius: 12px;
        overflow: hidden;
    }
    
    [data-testid="stDataFrame"] > div {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        font-family: 'Inter', sans-serif !important;
        font-weight: 500;
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        color: var(--text-primary);
    }
    
    /* Alerts/Info boxes */
    .stAlert {
        font-family: 'Inter', sans-serif !important;
        border-radius: 8px;
        border: none;
    }
    
    [data-testid="stAlert"] {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
    }
    
    /* Success message */
    .stSuccess {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(16, 185, 129, 0.05) 100%);
        border-left: 3px solid var(--accent-emerald);
    }
    
    /* Warning message */
    .stWarning {
        background: linear-gradient(135deg, rgba(212, 175, 55, 0.1) 0%, rgba(212, 175, 55, 0.05) 100%);
        border-left: 3px solid var(--accent-gold);
    }
    
    /* Error message */
    .stError {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(239, 68, 68, 0.05) 100%);
        border-left: 3px solid var(--accent-ruby);
    }
    
    /* Info message */
    .stInfo {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(59, 130, 246, 0.05) 100%);
        border-left: 3px solid var(--accent-sapphire);
    }
    
    /* Dividers */
    hr {
        border-color: var(--border-color);
        opacity: 0.5;
    }
    
    /* Charts */
    .stPlotlyChart, .stLineChart, .stBarChart {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1rem;
    }
    
    /* Captions */
    .stCaption {
        font-family: 'Inter', sans-serif !important;
        color: var(--text-muted) !important;
        font-style: italic;
    }
    
    /* Links */
    a {
        color: var(--accent-gold) !important;
        text-decoration: none;
    }
    
    a:hover {
        color: #e6c04a !important;
        text-decoration: underline;
    }
    
    /* Chat messages */
    [data-testid="stChatMessage"] {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-color: var(--accent-gold) transparent transparent transparent;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-primary);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--border-accent);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--text-muted);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- Initialize Session State ---
if 'financials' not in st.session_state:
    st.session_state.financials = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'quarterly_analysis' not in st.session_state:
    st.session_state.quarterly_analysis = None
if 'independent_forecast' not in st.session_state:
    st.session_state.independent_forecast = None
if 'api_key_set' not in st.session_state:
    st.session_state.api_key_set = False

# --- Helper Functions ---
def reset_analysis():
    st.session_state.skeptic_analysis = None
    st.session_state.believer_analysis = None
    st.session_state.chat_history = []
    st.session_state.quarterly_analysis = None
    st.session_state.independent_forecast = None

def display_stock_call(call: str):
    """
    Displays the directional stock call with corresponding color and icon.
    - Outperform: Green with upward arrow
    - In-line: Gray with horizontal arrow
    - Underperform: Red with downward arrow
    """
    call_lower = call.lower() if call else ""
    
    if "outperform" in call_lower or "buy" in call_lower:
        st.markdown(
            """
            <div style="
                background: linear-gradient(135deg, #0a2e1f 0%, #052e16 100%);
                border: 1px solid #10b981;
                color: #10b981;
                padding: 24px;
                border-radius: 16px;
                text-align: center;
                box-shadow: 0 8px 32px rgba(16, 185, 129, 0.15);
                font-family: 'Inter', sans-serif;
            ">
                <div style="font-size: 40px; margin-bottom: 12px;">▲</div>
                <div style="font-size: 20px; font-weight: 700; letter-spacing: 0.1em; font-family: 'Playfair Display', serif;">OUTPERFORM</div>
                <div style="font-size: 12px; color: #6ee7b7; margin-top: 8px; text-transform: uppercase; letter-spacing: 0.15em;">Bullish Outlook</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    elif "underperform" in call_lower or "sell" in call_lower:
        st.markdown(
            """
            <div style="
                background: linear-gradient(135deg, #2e0a0a 0%, #1f0505 100%);
                border: 1px solid #ef4444;
                color: #ef4444;
                padding: 24px;
                border-radius: 16px;
                text-align: center;
                box-shadow: 0 8px 32px rgba(239, 68, 68, 0.15);
                font-family: 'Inter', sans-serif;
            ">
                <div style="font-size: 40px; margin-bottom: 12px;">▼</div>
                <div style="font-size: 20px; font-weight: 700; letter-spacing: 0.1em; font-family: 'Playfair Display', serif;">UNDERPERFORM</div>
                <div style="font-size: 12px; color: #fca5a5; margin-top: 8px; text-transform: uppercase; letter-spacing: 0.15em;">Bearish Outlook</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:  # In-line or neutral
        st.markdown(
            """
            <div style="
                background: linear-gradient(135deg, #1a1a1a 0%, #0f0f0f 100%);
                border: 1px solid #d4af37;
                color: #d4af37;
                padding: 24px;
                border-radius: 16px;
                text-align: center;
                box-shadow: 0 8px 32px rgba(212, 175, 55, 0.1);
                font-family: 'Inter', sans-serif;
            ">
                <div style="font-size: 40px; margin-bottom: 12px;">◆</div>
                <div style="font-size: 20px; font-weight: 700; letter-spacing: 0.1em; font-family: 'Playfair Display', serif;">IN-LINE</div>
                <div style="font-size: 12px; color: #fcd34d; margin-top: 8px; text-transform: uppercase; letter-spacing: 0.15em;">Neutral Outlook</div>
            </div>
            """,
            unsafe_allow_html=True
        )

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key
    api_key = st.text_input("Gemini API Key (Free)", type="password")
    if api_key:
        os.environ["GEMINI_API_KEY"] = api_key
        st.session_state.api_key_set = True
        st.success("✅ Key Set")
    
    st.divider()
    
    # Data Loading
    st.header("🏢 Data Source")
    ticker = st.text_input("Stock Ticker", value="MSFT").upper()
    
    if st.button("Fetch Data", type="primary", use_container_width=True):
        if not st.session_state.api_key_set:
            st.warning("Please set API Key first to enable AI features.")
        
        with st.spinner(f"Fetching {ticker}..."):
            inc, bal = get_financials(ticker)
            if not inc.empty:
                st.session_state.financials = {"income": inc, "balance": bal}
                st.session_state.ticker = ticker
                # Pre-calculate metrics
                st.session_state.metrics = calculate_metrics(inc, bal)
                reset_analysis()
                st.success(f"Loaded {ticker}")
            else:
                st.error("Failed to fetch data.")

# --- Main Interface ---
st.title("📊 Analyst Co-Pilot")

if st.session_state.financials:
    inc = st.session_state.financials["income"]
    bal = st.session_state.financials["balance"]
    metrics = st.session_state.metrics
    
    # Create Layout Tabs
    tab_dash, tab_deep, tab_quarterly, tab_debate, tab_chat = st.tabs([
        "📈 Dashboard", "🧮 Deep Dive", "📊 Quarterly Analysis", "⚔️ Bull vs Bear", "💬 Chat with Data"
    ])
    
    # --- TAB 1: DASHBOARD ---
    with tab_dash:
        st.subheader(f"Financial Overview: {st.session_state.ticker}")
        
        # Top line metrics
        if metrics:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Revenue CAGR", metrics.get('Revenue CAGR (Last ~4y)', 'N/A'))
            m2.metric("Latest Revenue", metrics.get('Latest Revenue', 'N/A'))
            m3.metric("Net Margin", metrics.get('Net Profit Margin', 'N/A'))
            m4.metric("ROE (DuPont)", metrics.get('DuPont ROE', 'N/A'))
        
        st.divider()
        
        # Charts - Need to transpose for Streamlit (Rows = Dates)
        try:
            # Prepare data: Transpose and sort index (dates) ascending
            chart_data = inc.T.sort_index()
            
            c1, c2 = st.columns(2)
            with c1:
                st.caption("Revenue & Net Income Trend")
                if 'Total Revenue' in chart_data.columns and 'Net Income' in chart_data.columns:
                    st.line_chart(chart_data[['Total Revenue', 'Net Income']])
            
            with c2:
                st.caption("Operating Expenses Trend")
                if 'Operating Expense' in chart_data.columns:
                   st.bar_chart(chart_data[['Operating Expense']])
        except Exception as e:
            st.error(f"Visualization Error: {e}")

    # --- TAB 2: DEEP DIVE (DUPONT) ---
    with tab_deep:
        st.subheader("Managed Metrics & Ratios")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### DuPont Identity")
            st.info("ROE = Net Margin × Asset Turnover × Equity Multiplier")
            
            if metrics:
                st.write(f"**Net Profit Margin:** {metrics.get('  - Net Profit Margin', 'N/A')}")
                st.write(f"**Asset Turnover:** {metrics.get('  - Asset Turnover', 'N/A')}")
                st.write(f"**Equity Multiplier:** {metrics.get('  - Equity Multiplier', 'N/A')}")
                st.markdown("---")
                st.write(f"**Calculated ROE:** {metrics.get('DuPont ROE', 'N/A')}")

        with col2:
            st.markdown("### Raw Financial Statements")
            with st.expander("View Income Statement"):
                st.dataframe(inc)
            with st.expander("View Balance Sheet"):
                st.dataframe(bal)

    # --- TAB 3: QUARTERLY ANALYSIS ---
    with tab_quarterly:
        st.header("📊 Quarterly Trends & Consensus Estimates")
        st.caption("Analyze historical quarterly performance and compare with Wall Street expectations.")
        
        if st.button("🔍 Run Quarterly Analysis", type="primary"):
            if not st.session_state.api_key_set:
                st.error("API Key required for consensus estimates.")
            else:
                with st.spinner(f"Analyzing {st.session_state.ticker} quarterly data..."):
                    analysis = analyze_quarterly_trends(st.session_state.ticker)
                    st.session_state.quarterly_analysis = analysis
        
        if st.session_state.quarterly_analysis:
            analysis = st.session_state.quarterly_analysis
            
            # Display any errors
            if analysis.get("errors"):
                for err in analysis["errors"]:
                    st.warning(err)
            
            # --- Historical Trends Section ---
            st.subheader("📈 Historical Quarterly Performance")
            
            hist_data = analysis.get("historical_trends", {}).get("quarterly_data", [])
            if hist_data:
                # Convert to DataFrame for display
                df_hist = pd.DataFrame(hist_data)
                df_hist = df_hist.set_index("quarter")
                
                # Format large numbers
                if "revenue" in df_hist.columns:
                    df_hist["revenue_display"] = df_hist["revenue"].apply(
                        lambda x: f"${x/1e9:.2f}B" if x and x > 1e9 else (f"${x/1e6:.1f}M" if x else "N/A")
                    )
                if "operating_income" in df_hist.columns:
                    df_hist["op_income_display"] = df_hist["operating_income"].apply(
                        lambda x: f"${x/1e9:.2f}B" if x and x > 1e9 else (f"${x/1e6:.1f}M" if x else "N/A")
                    )
                if "eps" in df_hist.columns:
                    df_hist["eps_display"] = df_hist["eps"].apply(
                        lambda x: f"${x:.2f}" if x else "N/A"
                    )
                
                # Display table
                display_cols = ["revenue_display", "op_income_display", "eps_display"]
                display_cols = [c for c in display_cols if c in df_hist.columns]
                if display_cols:
                    st.dataframe(
                        df_hist[display_cols].rename(columns={
                            "revenue_display": "Revenue",
                            "op_income_display": "Operating Income",
                            "eps_display": "EPS"
                        }),
                        use_container_width=True
                    )
                
                # Charts
                col1, col2 = st.columns(2)
                with col1:
                    st.caption("Revenue Trend (Quarterly)")
                    chart_df = df_hist[["revenue"]].dropna().iloc[::-1]  # Reverse for chronological
                    if not chart_df.empty:
                        st.line_chart(chart_df)
                
                with col2:
                    st.caption("EPS Trend (Quarterly)")
                    eps_df = df_hist[["eps"]].dropna().iloc[::-1]
                    if not eps_df.empty:
                        st.line_chart(eps_df)
            
            st.divider()
            
            # --- Growth Rates Section ---
            st.subheader("📊 Growth Rates (YoY & QoQ)")
            
            growth_summary = analysis.get("growth_rates", {}).get("summary", {})
            growth_detail = analysis.get("growth_rates", {}).get("detailed", [])
            
            if growth_summary:
                m1, m2, m3 = st.columns(3)
                m1.metric(
                    "Avg Revenue YoY Growth",
                    f"{growth_summary.get('avg_revenue_yoy', 'N/A')}%" if growth_summary.get('avg_revenue_yoy') else "N/A"
                )
                m2.metric(
                    "Avg EPS YoY Growth",
                    f"{growth_summary.get('avg_eps_yoy', 'N/A')}%" if growth_summary.get('avg_eps_yoy') else "N/A"
                )
                m3.metric(
                    "Quarters Analyzed",
                    growth_summary.get('samples_used', 'N/A')
                )
            
            if growth_detail:
                with st.expander("View Detailed Growth Rates"):
                    df_growth = pd.DataFrame(growth_detail)
                    df_growth = df_growth.set_index("quarter")
                    # Format percentages
                    for col in df_growth.columns:
                        df_growth[col] = df_growth[col].apply(
                            lambda x: f"{x:.1f}%" if x is not None else "N/A"
                        )
                    st.dataframe(df_growth, use_container_width=True)
            
            st.divider()
            
            # --- Projections Section ---
            st.subheader("🔮 Historical-Based Projection")
            
            projection = analysis.get("projections", {}).get("next_quarter_estimate", {})
            if projection:
                st.info(f"**Basis:** {projection.get('basis', 'N/A')}")
                st.caption(f"Using data from: {projection.get('base_quarter', 'N/A')}")
                
                p1, p2 = st.columns(2)
                with p1:
                    proj_rev = projection.get('projected_revenue')
                    if proj_rev:
                        st.metric(
                            "Projected Revenue",
                            f"${proj_rev/1e9:.2f}B" if proj_rev > 1e9 else f"${proj_rev/1e6:.1f}M",
                            f"{projection.get('revenue_growth_rate_used', 0):.1f}% YoY"
                        )
                with p2:
                    proj_eps = projection.get('projected_eps')
                    if proj_eps:
                        st.metric(
                            "Projected EPS",
                            f"${proj_eps:.2f}",
                            f"{projection.get('eps_growth_rate_used', 0):.1f}% YoY"
                        )
            else:
                st.warning("Insufficient data for projection (need 5+ quarters)")
            
            st.divider()
            
            # --- Consensus Estimates Section ---
            st.subheader("🎯 Wall Street Consensus Estimates")
            
            consensus = analysis.get("consensus_estimates", {})
            if consensus.get("error"):
                st.error(consensus["error"])
            elif consensus:
                # Next Quarter Estimates
                next_q = consensus.get("next_quarter", {})
                if next_q:
                    st.markdown(f"**Next Quarter: {next_q.get('quarter_label', 'Upcoming')}**")
                    c1, c2 = st.columns(2)
                    c1.metric("Revenue Estimate", next_q.get("revenue_estimate", "N/A"))
                    c2.metric("EPS Estimate", next_q.get("eps_estimate", "N/A"))
                    if next_q.get("source_url"):
                        st.caption(f"📎 Source: [{next_q.get('source', 'Link')}]({next_q.get('source_url')})")
                
                # Full Year Estimates
                full_year = consensus.get("full_year", {})
                if full_year:
                    st.markdown(f"**Full Year: {full_year.get('fiscal_year', 'Current FY')}**")
                    c1, c2 = st.columns(2)
                    c1.metric("Revenue Estimate", full_year.get("revenue_estimate", "N/A"))
                    c2.metric("EPS Estimate", full_year.get("eps_estimate", "N/A"))
                    if full_year.get("source_url"):
                        st.caption(f"📎 Source: [{full_year.get('source', 'Link')}]({full_year.get('source_url')})")
                
                # Analyst Coverage
                coverage = consensus.get("analyst_coverage", {})
                if coverage:
                    st.markdown("**Analyst Ratings**")
                    r1, r2, r3, r4 = st.columns(4)
                    r1.metric("Total Analysts", coverage.get("num_analysts", "N/A"))
                    r2.metric("Buy", coverage.get("buy_ratings", "N/A"))
                    r3.metric("Hold", coverage.get("hold_ratings", "N/A"))
                    r4.metric("Sell", coverage.get("sell_ratings", "N/A"))
                    if coverage.get("source_url"):
                        st.caption(f"📎 Source: [{coverage.get('source', 'Link')}]({coverage.get('source_url')})")
                
                # Price Targets
                targets = consensus.get("price_targets", {})
                if targets:
                    st.markdown("**Price Targets**")
                    t1, t2, t3 = st.columns(3)
                    t1.metric("Low", targets.get("low", "N/A"))
                    t2.metric("Average", targets.get("average", "N/A"))
                    t3.metric("High", targets.get("high", "N/A"))
                    if targets.get("source_url"):
                        st.caption(f"📎 Source: [{targets.get('source', 'Link')}]({targets.get('source_url')})")
                
                st.divider()
                
                # --- Citations Section (Collapsible) ---
                with st.expander("📚 Sources & Citations", expanded=False):
                    st.markdown("Verify the estimates by visiting these sources:")
                    
                    citations = consensus.get("citations", [])
                    if citations:
                        for i, cite in enumerate(citations, 1):
                            source_name = cite.get("source_name", "Unknown Source")
                            url = cite.get("url", "")
                            data_type = cite.get("data_type", "Financial Data")
                            
                            if url:
                                st.markdown(f"{i}. **{source_name}** - {data_type}")
                                st.markdown(f"   🔗 [{url}]({url})")
                            else:
                                st.markdown(f"{i}. **{source_name}** - {data_type}")
                    else:
                        # Fallback: Generate standard links for the ticker
                        ticker = st.session_state.ticker
                        st.markdown("**Recommended Sources to Verify:**")
                        st.markdown(f"""                    
1. **Yahoo Finance** - Analyst Estimates  
   🔗 [https://finance.yahoo.com/quote/{ticker}/analysis](https://finance.yahoo.com/quote/{ticker}/analysis)

2. **MarketWatch** - Analyst Estimates  
   🔗 [https://www.marketwatch.com/investing/stock/{ticker.lower()}/analystestimates](https://www.marketwatch.com/investing/stock/{ticker.lower()}/analystestimates)

3. **TipRanks** - Stock Forecast & Price Target  
   🔗 [https://www.tipranks.com/stocks/{ticker.lower()}/forecast](https://www.tipranks.com/stocks/{ticker.lower()}/forecast)

4. **Nasdaq** - Analyst Research  
   🔗 [https://www.nasdaq.com/market-activity/stocks/{ticker.lower()}/analyst-research](https://www.nasdaq.com/market-activity/stocks/{ticker.lower()}/analyst-research)
                        """)
                    
                    # Show raw JSON
                    st.markdown("---")
                    st.markdown("**Raw Data:**")
                    st.json(consensus)
                
                # Disclaimer
                st.warning(consensus.get('disclaimer', '⚠️ AI-generated estimates based on available data. Always verify with the original sources before making investment decisions.'))
            else:
                st.info("Click 'Run Quarterly Analysis' to fetch consensus estimates.")
            
            st.divider()
            
            # --- Independent AI Forecast Section ---
            st.subheader("🤖 AI Analyst Independent Forecast")
            st.caption("Generate an independent equity research forecast based on the data above.")
            
            if st.button("📝 Generate Independent Forecast", type="secondary"):
                with st.spinner("AI Analyst is reviewing the data and formulating a forecast..."):
                    forecast = generate_independent_forecast(
                        st.session_state.quarterly_analysis,
                        company_name=st.session_state.ticker
                    )
                    st.session_state.independent_forecast = forecast
            
            if st.session_state.independent_forecast:
                forecast = st.session_state.independent_forecast
                
                if forecast.get("error"):
                    st.error(forecast["error"])
                else:
                    # Two-column layout: Left (1/3) for numbers, Right (2/3) for justification
                    col_left, col_right = st.columns([1, 2])
                    
                    extracted = forecast.get("extracted_forecast") or {}
                    
                    # --- LEFT COLUMN: Forecast Numbers & Stock Call ---
                    with col_left:
                        st.markdown("### 📊 Forecast Summary")
                        
                        # Revenue Forecast with YoY delta
                        revenue_forecast = extracted.get("revenue_forecast", "N/A")
                        revenue_yoy = extracted.get("revenue_yoy_growth", None)
                        st.metric(
                            label="💰 Revenue Forecast",
                            value=revenue_forecast,
                            delta=f"{revenue_yoy} YoY" if revenue_yoy else None,
                            delta_color="normal" if revenue_yoy and not revenue_yoy.startswith("-") else "inverse"
                        )
                        
                        # EPS Forecast with YoY delta
                        eps_forecast = extracted.get("eps_forecast", "N/A")
                        eps_yoy = extracted.get("eps_yoy_growth", None)
                        st.metric(
                            label="📈 EPS Forecast",
                            value=eps_forecast,
                            delta=f"{eps_yoy} YoY" if eps_yoy else None,
                            delta_color="normal" if eps_yoy and not eps_yoy.startswith("-") else "inverse"
                        )
                        
                        st.markdown("---")
                        
                        # vs Consensus comparison
                        vs_consensus = extracted.get("vs_consensus", "unknown")
                        vs_consensus_pct = extracted.get("vs_consensus_pct", "")
                        
                        st.markdown("**vs Wall Street Consensus:**")
                        if vs_consensus.lower() == "above":
                            st.success(f"📈 **ABOVE** consensus {vs_consensus_pct}")
                        elif vs_consensus.lower() == "below":
                            st.error(f"📉 **BELOW** consensus {vs_consensus_pct}")
                        else:
                            st.info(f"➡️ **IN-LINE** with consensus")
                        
                        st.markdown("---")
                        
                        # Directional Stock Call
                        st.markdown("**Stock Call:**")
                        stock_call = extracted.get("stock_call", vs_consensus)
                        display_stock_call(stock_call)
                        
                        st.markdown("---")
                        
                        # Confidence Level
                        confidence = extracted.get("confidence", "Medium")
                        confidence_emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(confidence.lower(), "🟡")
                        st.markdown(f"**Confidence:** {confidence_emoji} {confidence}")
                        
                        # Forecast Details Table
                        st.markdown("---")
                        st.markdown("**Forecast Details:**")
                        forecast_table = pd.DataFrame({
                            "Metric": ["Revenue", "EPS"],
                            "Forecast": [revenue_forecast, eps_forecast],
                            "YoY Growth": [revenue_yoy or "N/A", eps_yoy or "N/A"],
                            "vs Consensus": [vs_consensus_pct or vs_consensus, vs_consensus_pct or vs_consensus]
                        })
                        st.dataframe(forecast_table, use_container_width=True, hide_index=True)
                    
                    # --- RIGHT COLUMN: Detailed Justification ---
                    with col_right:
                        st.markdown("### 📝 Analysis & Methodology")
                        
                        # Put justification in an expander
                        with st.expander("📖 Read the Full Justification & Methodology", expanded=False):
                            st.markdown(forecast.get("full_analysis", "No analysis available."))
                        
                        # Quick summary outside the expander
                        st.markdown("#### 🔑 Key Points")
                        full_analysis = forecast.get("full_analysis", "")
                        
                        # Try to extract key sections from the analysis
                        if "### Key Risks" in full_analysis or "Key Risks" in full_analysis:
                            # Find and display key risks section
                            risk_start = full_analysis.find("Key Risks")
                            if risk_start != -1:
                                risk_section = full_analysis[risk_start:risk_start+500]
                                # Find the next section header
                                next_section = risk_section.find("###", 10)
                                if next_section != -1:
                                    risk_section = risk_section[:next_section]
                                st.markdown("**⚠️ Key Risks:**")
                                st.markdown(risk_section.replace("### Key Risks", "").replace("Key Risks", "").strip()[:400] + "...")
                        
                        # Input data summary in a small expander
                        with st.expander("📊 View Input Data Summary"):
                            input_summary = forecast.get("input_data_summary", {})
                            sum1, sum2 = st.columns(2)
                            with sum1:
                                st.metric("Quarters Analyzed", input_summary.get("quarters_analyzed", "N/A"))
                                st.metric("Consensus Revenue", input_summary.get("consensus_revenue", "N/A"))
                            with sum2:
                                st.metric("Avg YoY Rev Growth", f"{input_summary.get('avg_yoy_revenue_growth', 'N/A')}%")
                                st.metric("Consensus EPS", input_summary.get("consensus_eps", "N/A"))
                    
                    # Disclaimer at the bottom (full width)
                    st.divider()
                    st.warning(forecast.get("disclaimer", "This is AI-generated content for educational purposes only."))
                    st.caption(f"🕐 Forecast generated: {forecast.get('forecast_date', 'Unknown')}")

    # --- TAB 4: BULL VS BEAR DEBATE ---
    with tab_debate:
        st.header("Adversarial Analysis")
        st.caption("Two AI agents analyze the data + calculated metrics with opposing biases.")
        
        if st.button("🚀 Start Debate", type="primary"):
            if not st.session_state.api_key_set:
                st.error("API Key required.")
            else:
                # Prepare rich context
                context = f"""
                Ticker: {st.session_state.ticker}
                Calculated Metrics: {metrics}
                Income Statement: {inc.to_string()}
                """
                
                col_bear, col_bull = st.columns(2)
                
                with col_bear:
                    with st.spinner("🐻 Bear is skeptical..."):
                        response = run_structured_prompt(
                            system_role="You are a SKEPTICAL SHORT SELLER. Find flaws, risks, and accounting irregularities.",
                            user_prompt="Identify the top 3 existential risks for this company based on the data. Be harsh.",
                            context_data=context
                        )
                        st.session_state.skeptic_analysis = response
                
                with col_bull:
                    with st.spinner("🐂 Bull is excited..."):
                        response = run_structured_prompt(
                            system_role="You are a VISIONARY GROWTH INVESTOR. Find hidden value and exponential potential.",
                            user_prompt="Identify the top 3 growth catalysts and competitive moats. Be optimistic.",
                            context_data=context
                        )
                        st.session_state.believer_analysis = response
        
        # Display Results
        c_bear, c_bull = st.columns(2)
        with c_bear:
            st.subheader("🔴 The Bear Case")
            if st.session_state.skeptic_analysis:
                st.markdown(st.session_state.skeptic_analysis)
        with c_bull:
            st.subheader("🟢 The Bull Case")
            if st.session_state.believer_analysis:
                st.markdown(st.session_state.believer_analysis)

    # --- TAB 4: CHAT WITH DATA ---
    with tab_chat:
        st.subheader("💬 Ask the Analyst")
        
        # Display history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
        
        if prompt := st.chat_input("Ask about margins, trends, or specific line items..."):
            if not st.session_state.api_key_set:
                st.error("API Key required.")
            else:
                # User message
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.write(prompt)
                
                # Context preparation
                context = f"""
                Ticker: {st.session_state.ticker}
                Calculated Metrics: {metrics}
                Income Statement Data: {inc.to_string()}
                """
                
                # Model response
                with st.spinner("Thinking..."):
                    response_text = run_chat(st.session_state.chat_history[:-1], prompt, context)
                
                st.session_state.chat_history.append({"role": "assistant", "content": response_text})
                
                # Force rerun to show assistant message immediately? 
                # Actually streamlit usually handles this inside the loop or via rerun
                # We'll just write it out manually to be snappy
                with st.chat_message("assistant"):
                    st.write(response_text)

    # --- TAB 5: CHAT WITH DATA --- (comment update)

else:
    st.info("👈 Please enter a Ticker and API Key to begin.")
