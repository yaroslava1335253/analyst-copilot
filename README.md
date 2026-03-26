# Analyst Co-Pilot

Analyst Co-Pilot is a Streamlit app for equity research workflows. It combines historical business analysis, DCF-based valuation, street-consensus comparison, AI-generated written takeaways, and exportable PDF summaries in one interface.

The app is designed for fast single-name analysis:

- `Dashboard`: operating snapshot, business momentum, and quality-of-business context
- `Deep Dive`: valuation verdict, DCF assumptions, and supporting model outputs
- `Compare`: market price vs. DCF value vs. street targets
- `Reports`: generated write-up and PDF export

## What It Does

- Pulls company financials and market data from Yahoo Finance, YahooQuery, and optional FMP fallbacks
- Calculates operating metrics, growth trends, DuPont-style diagnostics, and valuation drivers
- Runs a structured DCF engine with traceable assumptions and explicit fallback logic
- Generates an AI-assisted equity outlook using the loaded company data
- Exports a clean analyst-style PDF summary for sharing or review

## Stack

- Python
- Streamlit
- Pandas
- yfinance
- yahooquery
- google-genai

## Quick Start

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
CONTACT_EMAIL_TO=your_email@example.com
```

Notes:

- `GEMINI_API_KEY` is required for AI outlook generation.
- `FMP_API_KEY` is optional but improves analyst-consensus coverage and fallbacks.
- `CONTACT_EMAIL_TO` is optional and powers the in-app contact panel.

4. Start the app.

```bash
streamlit run app.py
```

## Testing

Run the test suite with:

```bash
pytest
```

Useful focused runs:

```bash
pytest tests/test_dcf_engine.py
pytest tests/test_consensus_estimates.py
pytest tests/test_pdf_export.py
```

## Project Structure

```text
.
├── app.py                  # Streamlit UI entrypoint
├── engine.py               # Analysis, consensus, AI, and data orchestration helpers
├── dcf_engine.py           # Core DCF calculation engine
├── dcf_integration.py      # Integration layer between valuation engine and app flows
├── dcf_ui_adapter.py       # UI formatting helpers for valuation outputs
├── data_adapter.py         # Financial data normalization and quality metadata
├── pdf_export.py           # PDF summary renderer
├── sources.py              # Source metadata and citations
├── yf_cache.py             # Yahoo Finance caching helpers
├── docs/                   # Architecture notes, migration docs, and verification writeups
├── legacy/                 # Older reference implementations kept for comparison
├── scripts/                # Local utilities and diagnostics
├── tests/                  # Automated tests
└── data/                   # Local runtime cache and artifacts
```

## Security Notes

- Keep secrets in local `.env` files or deployment secret stores only.
- `.env`, `.env.save`, and local cache artifacts are gitignored in [.gitignore](./.gitignore).
- `data/user_ui_cache.json` is intended to remain local and should not be reintroduced to version control.

## Documentation

Additional project notes live in [`docs/`](./docs):

- [`docs/DCF_ARCHITECTURE.md`](./docs/DCF_ARCHITECTURE.md)
- [`docs/DCF_UI_USER_GUIDE.md`](./docs/DCF_UI_USER_GUIDE.md)
- [`docs/REPO_ORGANIZATION.md`](./docs/REPO_ORGANIZATION.md)

## Current Status

The repository is actively evolving. The app is functional and tested, but the codebase is still mid-refactor in a few places:

- `app.py` remains larger than ideal and is a strong candidate for UI-module extraction
- `engine.py` still mixes multiple responsibilities
- `legacy/` is retained for reference and can eventually be archived out of the main repo

## Suggested Next Cleanup

- Split `app.py` into per-view UI modules
- Split `engine.py` into data-fetch, consensus, and AI/reporting modules
- Add deployment metadata for the hosting platform you want to standardize on
- Remove stale historical cache artifacts from Git history if the repo will remain public
