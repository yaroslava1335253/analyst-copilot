# Analyst Co-Pilot: Internal Technical Report

As-of: February 22, 2026  
Audience: internal (not website copy)  
Codebase: `/Users/user/Desktop/analyst_copilot`

## 1) What This Tool Is

Analyst Co-Pilot is a Streamlit-based equity research workflow that combines:

- Market and financial data ingestion from Yahoo Finance interfaces.
- A traceable DCF engine (enterprise-value first, then equity bridge).
- Structured, assumption-driven valuation controls.
- AI-assisted synthesis of model outputs and context.
- Source attribution and data quality metadata.

The core design goal is not "one-click price target", but a transparent decision workflow where assumptions, formulas, and intermediate calculations are inspectable.

## 2) What It Does

For a selected ticker and report date context, the tool:

1. Loads historical quarterly financial context and market metadata.
2. Suggests WACC and near-term growth defaults based on CAPM + consensus/historical data quality.
3. Runs a DCF model with explicit projection logic and terminal-value handling.
4. Displays valuation outputs in a 6-step report flow:
   - Step 01: Investment Verdict
   - Step 02: Valuation Drivers
   - Step 03: Business Momentum
   - Step 04: Street Context
   - Step 05: AI Synthesis
   - Step 06: Sources & Methodology
5. Provides a detailed DCF trace page with inputs, assumptions, formulas, bridge checks, warnings, and downloadable JSON trace.

## 3) How It Works (Architecture)

Main modules and responsibilities:

- `app.py`
  - Streamlit UI, navigation, controls, caching lifecycle, section rendering.
  - User-adjustable DCF assumptions.
  - "View DCF Details" sequential audit view.
  - User guide page and disclaimers.
- `data_adapter.py`
  - Normalizes raw data into `NormalizedFinancialSnapshot`.
  - Attaches `DataQualityMetadata` to each metric (source path, reliability, notes).
  - Fetches analyst consensus revenue + long-term growth when available.
  - Computes suggested assumptions (`suggested_wacc`, `suggested_fcf_growth`).
- `dcf_engine.py`
  - Core valuation logic (`DCFEngine` + `DCFAssumptions`).
  - FCFF construction hierarchy, driver-based projection, terminal value logic.
  - Enterprise -> equity -> per-share bridge.
  - Cross-method diagnostics and sanity checks.
  - Full formula trace (`CalculationTraceStep`).
- `dcf_ui_adapter.py`
  - Converts engine output into UI-safe, formatted, traceable tables.
  - Adds consistency diagnostics for display.
- `engine.py`
  - Historical trend analysis and AI forecast generation.
  - Prompt guardrails for evidence, consistency, and valuation language.
  - Text sanitization to avoid overconfident "floor" language.
- `sources.py`
  - Citation catalog used in Sources & Methodology rendering.

## 4) End-to-End Data Flow

1. User selects ticker + ending report date + quarter count.
2. UI loads/caches quarterly analysis context (`analyze_quarterly_trends`).
3. DCF run path:
   - `DataAdapter.fetch()` builds normalized snapshot.
   - `DCFAssumptions` is seeded from user slider overrides.
   - `DCFEngine.run()` performs valuation with trace.
   - `DCFUIAdapter` maps result for display.
4. Result is cached per context key:
   - `ticker|end_date|num_quarters`
   - Both DCF and AI outputs can be restored.

## 5) Underlying DCF Logic

### 5.1 Input Validation

Current hard requirements before run:

- TTM revenue > 0
- TTM FCF > 0
- shares outstanding available for per-share output

Quality warnings are added when reliability is weak.

### 5.2 Assumption Defaults

If user does not override:

- Forecast horizon:
  - 10Y if market cap > 200B USD or revenue > 50B USD
  - Else 5Y
- WACC:
  - CAPM-style suggested value from adapter if available, otherwise size fallback.
- Near-term growth:
  - Priority: adapter suggestion -> annualized quarterly trend -> 8% fallback.
- Terminal growth (`g`):
  - Default 3.0% (must remain below WACC for Gordon validity).

### 5.3 FCFF Construction Hierarchy

Primary method:

- `FCFF = EBIT * (1 - tax) + D&A - CapEx - delta NWC`

Fallback 1:

- Approximate unlevering proxy:
  - `FCFF ~= CFO + after-tax interest expense - CapEx`

Fallback 2:

- Levered proxy:
  - `FCF ~= CFO - CapEx`
  - Explicitly warned as framework-mixed if discounted with WACC.

### 5.4 Projection Engine (Driver Model)

Projection follows:

- Revenue -> EBIT -> NOPAT -> Reinvestment -> FCFF

Key mechanics:

- Revenue growth schedule fades from near-term toward terminal growth.
- Analyst revenue consensus years are used as top-line anchors when available.
- Long-term analyst growth can shape a mid-curve anchor in 10Y models.
- Reinvestment is ROIC-driven; terminal reinvestment is tied to `g / terminal_ROIC`.
- Terminal-year FCFF and EBITDA are computed from the same projection base for consistency.

### 5.5 Terminal Value Policy

Current policy in `DCFEngine`:

- Gordon Growth is the default primary method.
- Exit Multiple is always computed when possible as a secondary diagnostic path.
- Primary can be overridden to Exit Multiple only under guardrail conditions:
  - Gordon-implied price is extremely low vs market
  - Exit-implied price is materially higher

Current hardcoded thresholds:

- `EXTREME_GORDON_PRICE_TO_MARKET_THRESHOLD = 0.35`
- `EXTREME_GORDON_EXIT_UPLIFT_THRESHOLD = 0.25`

The engine stores both methods:

- `tv_gordon_growth`, `pv_tv_gordon_growth`, `price_gordon_growth`
- `tv_exit_multiple`, `pv_tv_exit_multiple`, `price_exit_multiple`

### 5.6 Bridge to Equity and Per-Share

The valuation bridge is explicit:

- `EV = PV(explicit FCFF) + PV(terminal value)`
- `Net Debt = Total Debt - Cash`
- `Equity Value = EV - Net Debt`
- `Intrinsic Value/Share = Equity Value / Shares`

### 5.7 Sanity and Risk Diagnostics

Checks include:

- `g < WACC` validity.
- TV dominance ratio (flags when terminal value dominates EV).
- Implied terminal ROIC reasonableness.
- EV/revenue and EV/EBITDA plausibility.
- EV vs market EV contextual gap.
- Exit-multiple required cash conversion stress diagnostics.

## 6) UI and Interaction Logic

### 6.1 Main Experience

- Collapsible sidebar with hamburger open button (`☰ Menu`) when hidden.
- Two navigation layers in report view:
  - Primary section nav bar.
  - Sticky quick nav panel with hamburger toggle.

### 6.2 Valuation Controls (Step 02)

User inputs:

- WACC slider: 5.0% to 15.0%, step 0.1.
- FCF growth slider: 0.0% to 25.0%, step 0.1.
- Terminal growth `g` slider:
  - Dynamic max capped by `wacc - 0.5` and upper bounded at 6.0.
  - Step 0.1.
  - Default/fallback 3.0% if not set.

Actions:

- `Run DCF Analysis`
- `View DCF Details ->` (available after DCF result exists)

### 6.3 DCF Details Page

The details view is organized as a sequential audit:

1. Input Data
2. Assumptions
3. Forecast and Present Value
4. Terminal Value
5. Enterprise to Equity Bridge
6. Warnings, Diagnostics, and Trace

It also supports trace JSON download (`*_dcf_trace.json`).

### 6.4 Header Tiles and P/E Display

Header tiles show price, market cap, as-of, and ticker context.  
P/E is now conditional: tile renders only when P/E exists and is positive.

## 7) AI Synthesis Logic

AI synthesis is generated from a strict prompt with hard constraints, including:

- Do not present intrinsic value as a guaranteed floor.
- Keep FCFF framework-consistent language.
- Require citations for external claims.
- Avoid unsupported segment-level claims.
- Force conditional, evidence-based phrasing.

Post-processing sanitization further rewrites overconfident terms (e.g., "fundamental floor" -> "model-implied value under current assumptions").

## 8) Traceability and Transparency Features

The system is built for auditability through:

- Per-metric source paths and reliability scores.
- Explicit formulas in trace steps.
- Side-by-side terminal-method outputs.
- Bridge reconciliation checks.
- Contextual warnings instead of silent fallbacks.
- Source catalog for methodology references.

## 9) Dependencies and External Interfaces

Primary external dependencies:

- `yfinance`
- `yahooquery`
- `streamlit`
- `google.generativeai`

Data quality and field availability are constrained by upstream Yahoo endpoints and ticker-specific coverage.

## 10) Current Limitations and Model Risks

1. Assumption sensitivity remains high, especially terminal assumptions.
2. Upstream data coverage gaps can trigger lower-reliability fallback methods.
3. High-growth handling still depends on threshold logic and a limited watchlist seed (`TSLA`, `NFLX`) plus growth signals.
4. Exit vs Gordon method selection is rule-based, not probabilistic.
5. AI narrative quality depends on prompt compliance and available cited context.
6. For some names, market-implied option value may exceed steady-state DCF framing by design.

## 11) Verification Snapshot (Local Session)

Attempted command:

- `pytest -q tests/test_dcf_engine.py`

Current result in this environment:

- Test collection failed due missing dependency:
  - `ModuleNotFoundError: No module named 'yahooquery'`

So no fresh pass/fail status could be produced in this run without installing that package.

## 12) Practical Internal Usage Guidance

For internal decision discipline:

1. Start with suggested assumptions, but immediately run sensitivity ranges (WACC, near-term growth, terminal `g`).
2. Treat intrinsic value as model-implied under assumptions, never as a hard floor.
3. Monitor `TV Dominance` and method divergence before trusting point estimates.
4. Use DCF Details and trace export to verify each critical number lineage.
5. If data quality score is low, downgrade confidence before making valuation calls.

