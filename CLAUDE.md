# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the app
.venv/bin/streamlit run app.py

# Run all tests
.venv/bin/pytest tests/

# Run a single test file
.venv/bin/pytest tests/test_dcf_engine.py -v

# Run a specific test
.venv/bin/pytest tests/test_dcf_engine.py::test_basic_dcf -v

# Real-world validation (requires network)
.venv/bin/python tests/quick_test.py

# Full system verification
.venv/bin/python tests/verify_dcf_system.py
```

Always use `.venv/bin/python` and `.venv/bin/pytest` — not system Python.

## Architecture

The app is a 3-step Streamlit financial analysis tool:
**Historical Analysis → Consensus Outlook → AI Forecast**

**Data flow:**
```
DataAdapter.fetch(ticker)
  → NormalizedFinancialSnapshot
    → DCFEngine(snapshot, DCFAssumptions)
        → CalculationTraceSteps (full audit trail)
        → engine result dict
          → DCFUIAdapter
              → FinancialMetric objects
                → Streamlit UI (app.py)
```

**Core modules:**

| File | Role |
|---|---|
| `app.py` | Streamlit UI; DCF details page at ~line 97; Step 2 consensus at ~line 1740 |
| `dcf_engine.py` | `DCFEngine` + `DCFAssumptions` dataclass; `_project_fcf_with_drivers()` is main projection |
| `data_adapter.py` | `DataAdapter.fetch()` → `NormalizedFinancialSnapshot` with per-field quality metadata |
| `dcf_ui_adapter.py` | Transforms engine result dict to `FinancialMetric` objects for display |
| `engine.py` | Google Gemini integration + historical financial math (quarterly trends, AI forecast) |
| `industry_multiples.py` | Damodaran EV/EBITDA multiples for exit-multiple terminal value strategy |
| `sources.py` | `SOURCE_CATALOG` — 15 citation entries (IDs 1–15) for data provenance |

## Critical Dataclass Rules

`DCFAssumptions` is a `@dataclass`. Its `to_dict()` calls `dataclasses.asdict(self)` — **only declared fields are serialized**. Any dynamic attribute (`self.assumptions.foo = ...`) will be silently dropped. Add new fields to the class definition.

`assumptions_obj` in `dcf_ui_adapter.py` is a **manually maintained subset** of DCFAssumptions fields that the UI reads. When `app.py` reads a new field from assumptions, it must be explicitly added to `assumptions_obj` in `dcf_ui_adapter.py`.

## Terminal Value Strategies

`DCFEngine` supports two pluggable strategies (ABC pattern):
- **`GordonGrowthTerminalValue`** — `Terminal FCF / (WACC − g)`
- **`ExitMultipleTerminalValue`** — `Year-N EBITDA × exit_multiple` (default; uses Damodaran industry data)

Hard constraint: `WACC > terminal_growth_rate` — engine raises an error otherwise.

## Session State Keys

`app.py` uses these Streamlit session state keys:
- `ticker` — current ticker string
- `dcf_snapshot` — `NormalizedFinancialSnapshot`
- `dcf_engine_result` — raw dict from `DCFEngine.run()`
- `dcf_ui_adapter` — `DCFUIAdapter` instance
- `quarterly_analysis` — result from `engine.py` quarterly analysis

## Environment

Requires a `.env` file with:
```
GEMINI_API_KEY=...
FMP_API_KEY=...   # optional
```

Key dependencies: `streamlit==1.53.1`, `yfinance==1.1.0`, `yahooquery==2.4.1`, `google-generativeai==0.8.6`, `chromadb==1.4.1`, `pytest==9.0.2`.
