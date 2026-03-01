# Repo Organization Guide

This project keeps runtime code stable while preventing local artifacts from cluttering `main`.

## Root (keep minimal)

Only keep these categories at root:

- Entrypoint and core modules: `app.py`, `engine.py`, `dcf_*.py`, `data_adapter.py`, `sources.py`
- Project metadata: `README.md`, `requirements.txt`, `.gitignore`, `CLAUDE.md`
- Top-level folders: `docs/`, `scripts/`, `tests/`, `legacy/`, `data/`

Do not add new ad-hoc files at root unless they are core runtime modules.

## Folder placement

- `docs/`: guides, architecture notes, implementation writeups, reports
- `scripts/`: one-off utilities, diagnostics, local helper scripts
- `tests/`: test suites and verification scripts
- `legacy/`: superseded code kept only for reference
- `data/`: local runtime artifacts/cache (generally ignored from git)

## Local-only files (not committed)

These should remain local:

- `.env`, `.env.save`, and other secret variants
- `data/user_ui_cache.json`
- local IDE/system/runtime artifacts (`.vscode/`, `.claude/`, `.pytest_cache/`, `__pycache__/`)

## Practical workflow

When adding a new file:

1. Choose folder by purpose (docs/scripts/tests/legacy/data).
2. Avoid placing local output files in root.
3. If the file is runtime cache or machine-specific, add it to `.gitignore`.
