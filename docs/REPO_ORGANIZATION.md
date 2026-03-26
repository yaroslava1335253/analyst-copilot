# Repo Organization Guide

This document describes the current structure, what already works well, and the most important cleanup steps if you want the repository to read as more professional on GitHub.

## What Already Works

- Runtime code, docs, tests, scripts, and legacy code are at least separated into distinct folders.
- Secrets and local cache files are ignored in [.gitignore](../.gitignore).
- The repo already has a clear main entrypoint in `app.py`.
- Test coverage exists for valuation, consensus fallbacks, PDF generation, and cache behavior.

## Recommended Target Layout

```text
.
├── .github/
│   └── workflows/
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── app/
│   ├── views/
│   ├── components/
│   └── state/
├── analysis/
│   ├── engine/
│   ├── valuation/
│   ├── consensus/
│   └── reporting/
├── data/
│   └── .gitkeep
├── docs/
├── legacy/
├── scripts/
└── tests/
```

You do not need to do this all at once. The main value is directional clarity:

- UI code should move toward `app/views` and `app/components`
- business logic should move out of `engine.py` into smaller modules
- runtime-local files should stay out of git entirely

## Current Practical Layout

For the repo in its current form, keep root limited to:

- runtime entrypoints and core modules: `app.py`, `engine.py`, `dcf_*.py`, `data_adapter.py`, `pdf_export.py`, `sources.py`, `yf_cache.py`
- project metadata: `README.md`, `requirements.txt`, `.gitignore`, `pytest.ini`
- top-level folders: `docs/`, `scripts/`, `tests/`, `legacy/`, `data/`, `.github/`

Avoid adding ad-hoc reports, backups, or cache files at the root.

## Folder Rules

- `docs/`: architecture notes, how-to guides, verification writeups, internal technical documentation
- `scripts/`: one-off utilities, diagnostics, and migration helpers
- `tests/`: deterministic test suites and verification scripts
- `legacy/`: superseded code kept only for temporary reference
- `data/`: local cache and runtime artifacts, generally ignored from git
- `.github/`: CI and repository automation

## Local-Only Files

These should remain local and never be committed:

- `.env`, `.env.save`, and other secret variants
- `data/user_ui_cache.json`
- IDE and runtime artifacts such as `.vscode/`, `.claude/`, `.pytest_cache/`, `__pycache__/`

## Highest-Value Refactors

If you want the repo to look more professional without a giant rewrite, do these next:

1. Split `app.py` by view.
2. Split `engine.py` by responsibility.
3. Keep only active docs in `docs/` and archive outdated writeups elsewhere.
4. Remove or archive `legacy/` once the new flow is stable.
5. Add CI so pushes visibly run tests.

## Practical Workflow

When adding a new file:

1. Choose the folder by purpose, not convenience.
2. Keep root limited to entrypoints and core metadata.
3. If the file is machine-specific, cached, or generated, add it to `.gitignore`.
4. If it exists only to preserve old work, place it in `legacy/` or archive it outside the main repo.
5. If it improves public understanding, prefer `README.md` or a focused doc in `docs/`.

## Git History Cleanup Candidate

One historical cleanup target has already surfaced:

- `data/user_ui_cache.json` was committed in older revisions before being ignored

If the repository remains public, consider rewriting history to remove that file so past UI cache payloads are not publicly preserved.
