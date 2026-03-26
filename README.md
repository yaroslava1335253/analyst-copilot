# Analyst Co-Pilot

Analyst Co-Pilot is a Streamlit application for single-company equity research. It combines historical financial analysis, discounted cash flow valuation, market comparison, AI-assisted written output, and PDF export in one interface.

The project is designed to support a compact research workflow:

1. Load a public company by ticker.
2. Review operating performance and financial trends.
3. Run a DCF valuation with adjustable assumptions.
4. Compare the model output with market pricing and analyst targets.
5. Generate a written summary and export a report.

## Main Features

- Historical financial analysis and trend review
- DCF valuation with editable assumptions
- Comparison of intrinsic value, market price, and street consensus
- AI-generated written summary based on loaded company data
- PDF report export

## Technology Stack

- Python
- Streamlit
- pandas
- yfinance
- yahooquery
- google-genai

## Key Files

- `app.py`: main Streamlit application and user interface
- `engine.py`: analysis logic, consensus lookup, and AI/report helpers
- `dcf_engine.py`: core discounted cash flow model
- `dcf_ui_adapter.py`: formatting layer for DCF outputs shown in the UI
- `data_adapter.py`: financial data normalization and metadata handling
- `pdf_export.py`: PDF report generation
- `tests/`: automated tests
- `docs/`: supporting documentation
- `legacy/`: archived reference code not required for normal use

## Setup

1. Create and activate a virtual environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Create a local `.env` file.

```dotenv
GEMINI_API_KEY=your_key_here
FMP_API_KEY=your_optional_fmp_key
CONTACT_EMAIL_TO=your_optional_email@example.com
```

4. Start the application.

```bash
streamlit run app.py
```

## Environment Variables

- `GEMINI_API_KEY`: required for AI-generated written output
- `FMP_API_KEY`: optional; improves analyst-consensus coverage and fallback data access
- `CONTACT_EMAIL_TO`: optional; used by the in-app contact feature

## How to Use

1. Launch the app and enter a ticker symbol.
2. Use `Dashboard` to review the company snapshot and historical context.
3. Use `Deep Dive` to adjust assumptions and run the DCF model.
4. Use `Compare` to view valuation output against market and analyst benchmarks.
5. Use `Reports` to generate written output and export a PDF summary.

## Testing

Run the full test suite with:

```bash
pytest
```

## Data Sources

The application primarily uses Yahoo Finance and YahooQuery, with optional Financial Modeling Prep support when an `FMP_API_KEY` is provided.
