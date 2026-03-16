from pdf_export import _font, build_summary_pdf


def test_build_summary_pdf_returns_valid_pdf_bytes():
    pdf_bytes = build_summary_pdf(
        title="NVDA Summary Report",
        subtitle="NVIDIA | As of Dec 31, 2025",
        sections=[
            ("Bottom Line", ["Verdict: Modestly Undervalued | Price $120.00 | Intrinsic $135.00"]),
            ("What Matters Most", ["- Data center demand remains strong", "- Margin expansion is holding"]),
            ("Why This Verdict", ["Demand remains strong, execution is holding, and downside is concentrated in a few identifiable risks."]),
        ],
        footer="For research use only.",
        hero_metrics=[
            {"label": "Market Price", "value": "$120.00", "tone": "neutral"},
            {"label": "Intrinsic Value", "value": "$135.00", "tone": "positive"},
            {"label": "Upside / Downside", "value": "+12.5%", "tone": "positive"},
            {"label": "Verdict", "value": "Modestly Undervalued", "tone": "positive"},
        ],
        price_comparison=[
            {"label": "Market Price", "display": "$120.00", "value": 120.0, "tone": "neutral"},
            {"label": "Intrinsic Value", "display": "$135.00", "value": 135.0, "tone": "positive"},
            {"label": "Street Avg PT", "display": "$142.00", "value": 142.0, "tone": "positive"},
        ],
        revenue_series=[
            {"label": "FY2024 Q3", "value": 26_000_000_000},
            {"label": "FY2024 Q4", "value": 30_000_000_000},
            {"label": "FY2025 Q1", "value": 31_000_000_000},
            {"label": "FY2025 Q2", "value": 33_000_000_000},
        ],
        revenue_subtitle="Avg revenue YoY +18.0%",
        outlook={
            "short_stance": "Bullish",
            "fund_outlook": "Strong",
            "stock_outlook": "Bullish",
            "stock_horizon": "Mid-term",
            "stock_conviction": "High",
            "summary": "Demand remains strong and execution is supporting both growth and margin resilience.",
            "key_conditional": "Watch whether the next quarter sustains large-deal momentum.",
        },
    )

    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF")
    assert b"/Type /Catalog" in pdf_bytes
    assert pdf_bytes.rstrip().endswith(b"%%EOF")


def test_build_summary_pdf_accepts_unicode_text():
    pdf_bytes = build_summary_pdf(
        title="AAPL Summary - Q4",
        sections=[("Drivers", ["- Revenue up 10% -> mix improving", "- Margin >= 40%"])],
    )

    assert pdf_bytes.startswith(b"%PDF")
    assert pdf_bytes.rstrip().endswith(b"%%EOF")


def test_pdf_export_uses_bundled_scalable_fonts():
    regular = _font(24, bold=False)
    bold = _font(24, bold=True)

    assert str(getattr(regular, "path", "")).endswith("assets/fonts/DejaVuSans.ttf")
    assert str(getattr(bold, "path", "")).endswith("assets/fonts/DejaVuSans-Bold.ttf")
