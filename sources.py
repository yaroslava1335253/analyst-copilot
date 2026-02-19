"""
sources.py — Data Source Catalog for Analyst Co-Pilot
======================================================
Wikipedia-style citation registry.  Each entry maps a short key to metadata
that the UI uses to render numbered footnotes and a collapsible source expander.

Usage:
    from sources import SOURCE_CATALOG
    src = SOURCE_CATALOG["wacc_capm"]
    st.markdown(f"[{src['id']}] {src['label']} — {src['description']}")
"""

SOURCE_CATALOG = {
    "fcff_analyst": {
        "id": 1,
        "label": "Analyst Revenue Estimates",
        "description": (
            "Consensus analyst forward revenue estimates used to derive FCF for Years 1–3. "
            "FCF = analyst_revenue × TTM FCF margin."
        ),
        "url": "https://finance.yahoo.com/quote/{ticker}/analysis",
        "method": "Yahoo Finance · yfinance revenue_estimate",
    },
    "fcff_driver": {
        "id": 2,
        "label": "FCFF Driver Model",
        "description": (
            "FCFF = NOPAT × (1 − Reinvestment Rate); "
            "NOPAT = Revenue × EBIT Margin × (1 − Tax Rate). "
            "Used for Years 4–10 (or all years when analyst estimates are unavailable)."
        ),
        "url": "https://en.wikipedia.org/wiki/Free_cash_flow",
        "method": "Damodaran DCF framework",
    },
    "ttm_revenue": {
        "id": 3,
        "label": "TTM Revenue",
        "description": "Trailing twelve months revenue from the latest 4 quarterly filings.",
        "url": "https://finance.yahoo.com/",
        "method": "Yahoo Finance · yfinance quarterly financials",
    },
    "ttm_fcff": {
        "id": 4,
        "label": "TTM FCFF",
        "description": "Operating Cash Flow − CapEx (levered proxy for FCFF).",
        "url": "https://en.wikipedia.org/wiki/Free_cash_flow_to_firm",
        "method": "Yahoo Finance · yfinance cash flow statement",
    },
    "ebit_margin": {
        "id": 5,
        "label": "EBIT Margin",
        "description": "Operating income / Revenue from most recent TTM period.",
        "url": "https://finance.yahoo.com/",
        "method": "Yahoo Finance · income statement",
    },
    "reinv_rate": {
        "id": 6,
        "label": "Reinvestment Rate",
        "description": (
            "g / ROIC; terminal rate = g_perp / ROIC_terminal (Damodaran). "
            "Fades smoothly from Year 1 to Year N to avoid discontinuities."
        ),
        "url": "https://pages.stern.nyu.edu/~adamodar/",
        "method": "Damodaran · Investment Valuation",
    },
    "roic": {
        "id": 7,
        "label": "ROIC",
        "description": (
            "NOPAT / Invested Capital; fades toward sector median over the forecast horizon."
        ),
        "url": "https://en.wikipedia.org/wiki/Return_on_invested_capital",
        "method": "Calculated from income + balance sheet",
    },
    "terminal_growth": {
        "id": 8,
        "label": "Terminal Growth Rate",
        "description": "Perpetual nominal GDP growth assumption (2.5–3.0%).",
        "url": "https://pages.stern.nyu.edu/~adamodar/",
        "method": "Damodaran convention",
    },
    "wacc_capm": {
        "id": 9,
        "label": "WACC / Cost of Equity (CAPM)",
        "description": (
            "Rf + β × ERP; uses Cost of Equity as WACC proxy for low-debt companies."
        ),
        "url": "https://en.wikipedia.org/wiki/Capital_asset_pricing_model",
        "method": "CAPM",
    },
    "beta": {
        "id": 10,
        "label": "Beta",
        "description": "5-year monthly returns vs S&P 500.",
        "url": "https://finance.yahoo.com/",
        "method": "Yahoo Finance",
    },
    "risk_free_rate": {
        "id": 11,
        "label": "Risk-Free Rate",
        "description": "10-year U.S. Treasury yield (live from ^TNX; Feb 2026 ~4.5%).",
        "url": "https://fred.stlouisfed.org/series/DGS10",
        "method": "FRED DGS10 via Yahoo Finance ^TNX",
    },
    "erp": {
        "id": 12,
        "label": "Equity Risk Premium",
        "description": "Implied ERP (forward-looking), January 2026.",
        "url": "https://pages.stern.nyu.edu/~adamodar/",
        "method": "Damodaran NYU Stern",
    },
    "exit_multiple": {
        "id": 13,
        "label": "EV/EBITDA Exit Multiple",
        "description": (
            "Derived from terminal FCF/EBITDA ratio divided by (WACC − g); "
            "cross-checked against industry comps."
        ),
        "url": "https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/vebitda.html",
        "method": "Damodaran industry multiples",
    },
    "shares": {
        "id": 14,
        "label": "Shares Outstanding",
        "description": "Diluted shares outstanding from most recent filing.",
        "url": "https://finance.yahoo.com/",
        "method": "Yahoo Finance · yfinance info",
    },
    "net_debt": {
        "id": 15,
        "label": "Net Debt",
        "description": "Total debt − cash and equivalents from balance sheet.",
        "url": "https://finance.yahoo.com/",
        "method": "Yahoo Finance · balance sheet",
    },
}
