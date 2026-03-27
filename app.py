# app.py
"""
Analyst Co-Pilot - v3.0
=======================
Clean, minimalistic financial analysis tool with 3 steps:
1. Historical Analysis & Growth Rates
2. Wall Street Consensus
3. AI Outlook
"""

import os
import re
import html
import math
import ast
import hashlib
import smtplib
import ssl
from collections.abc import Mapping
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from urllib.parse import quote, urlparse
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

# --- App Configuration ---
st.set_page_config(
    page_title="Analyst Co-Pilot",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load API keys from .env file (if exists)
load_dotenv()
try:
    for secret_name in ("GEMINI_API_KEY", "FMP_API_KEY"):
        if not os.environ.get(secret_name):
            secret_key = st.secrets.get(secret_name)
            if secret_key:
                os.environ[secret_name] = str(secret_key)
except Exception:
    # st.secrets is not always available locally.
    pass
import pandas as pd
import altair as alt
import json
from engine import get_financials, run_structured_prompt, calculate_metrics, run_chat, analyze_quarterly_trends, generate_independent_forecast, get_latest_date_info, get_available_report_dates, calculate_comprehensive_analysis
from data_adapter import DataAdapter, DataQualityMetadata, NormalizedFinancialSnapshot
from dcf_engine import DCFEngine, DCFAssumptions
from dcf_ui_adapter import DCFUIAdapter
from pdf_export import build_summary_pdf
from sources import SOURCE_CATALOG
from yf_cache import get_yf_info, get_yf_ticker

_BOOT_DEBUG_ENABLED = os.getenv("ACP_BOOT_DEBUG") == "1"


def _write_boot_log(message: str) -> None:
    if not _BOOT_DEBUG_ENABLED:
        return
    log_line = f"[BOOT] {datetime.now().isoformat()} {message}\n"
    try:
        with Path("/tmp/acp_boot.log").open("a", encoding="utf-8") as log_file:
            log_file.write(log_line)
    except Exception:
        pass
    print(log_line.rstrip(), flush=True)


_write_boot_log("imports_complete")

UI_CACHE_VERSION = 2
REPORT_DATES_CACHE_VERSION = "v5"
UI_CACHE_PATH = Path(__file__).resolve().parent / "data" / "user_ui_cache.json"
MAX_TICKER_LIBRARY_SIZE = 100
MAX_REPORT_DATE_CACHE_TICKERS = 300
TICKER_PATTERN = re.compile(r"^[A-Z0-9.\-]{1,10}$")
MAG7_TICKERS = ["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA"]
DEFAULT_TICKER = MAG7_TICKERS[0]
HISTORY_SOURCE_VERSION = "v13"
AVAILABLE_DATES_TIMEOUT_SECONDS = 8
FINANCIALS_TIMEOUT_SECONDS = 20
QUARTERLY_ANALYSIS_TIMEOUT_SECONDS = 25
SNAPSHOT_TIMEOUT_SECONDS = 12
COMPANY_NAME_TIMEOUT_SECONDS = 3
WACC_SLIDER_MIN_PCT = 0.5
WACC_SLIDER_MAX_PCT = 20.0
SNAPSHOT_SUGGESTION_VERSION = "v4_wacc_source_sync"
DEFAULT_INITIAL_QUARTERS = 7
AI_OUTLOOK_CACHE_VERSION = "v2"
EMAIL_REGEX = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
URL_REGEX = re.compile(r"(?:https?://|www\.)[^\s<>()\[\]\"']+")
EMOJI_SYMBOL_REGEX = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0E\uFE0F\u200D]+")
SNAPSHOT_METADATA_FIELDS = [
    "price",
    "shares_outstanding",
    "market_cap",
    "total_debt",
    "average_total_debt",
    "cash_and_equivalents",
    "ttm_revenue",
    "ttm_operating_cash_flow",
    "ttm_capex",
    "ttm_fcf",
    "ttm_ebitda",
    "ttm_operating_income",
    "ttm_net_income",
    "effective_tax_rate",
    "beta",
    "suggested_wacc",
    "suggested_cost_of_equity",
    "suggested_cost_of_debt",
    "suggested_fcf_growth",
    "analyst_long_term_growth",
]


def _initial_contact_email() -> str:
    env_value = str(
        os.environ.get("CONTACT_EMAIL_TO", "") or os.environ.get("FORMSUBMIT_TO", "")
    ).strip()
    if env_value:
        return env_value
    try:
        top_level = str(
            st.secrets.get("CONTACT_EMAIL_TO", "") or st.secrets.get("FORMSUBMIT_TO", "")
        ).strip()
        if top_level:
            return top_level
        section = st.secrets.get("formsubmit", {})
        if isinstance(section, Mapping):
            section_value = str(
                section.get("to", "") or section.get("email", "") or section.get("recipient", "")
            ).strip()
            if section_value:
                return section_value
    except Exception:
        pass
    return ""


CONTACT_EMAIL_TO = _initial_contact_email()
_write_boot_log("contact_email_ready")


def _default_ui_cache() -> dict:
    return {
        "version": UI_CACHE_VERSION,
        "available_report_dates_version": REPORT_DATES_CACHE_VERSION,
        "ticker_library": MAG7_TICKERS.copy(),
        "last_selected_ticker": DEFAULT_TICKER,
        "available_report_dates": {},
        "results": {},
    }


def _normalize_ticker(ticker: str) -> str:
    if ticker is None:
        return ""
    return str(ticker).strip().upper()


def _is_valid_ticker_format(ticker: str) -> bool:
    return bool(TICKER_PATTERN.match(_normalize_ticker(ticker)))


def _normalize_ticker_library(raw_tickers) -> list:
    ordered = []
    for ticker in MAG7_TICKERS + (raw_tickers or []):
        t = _normalize_ticker(ticker)
        if not t or not _is_valid_ticker_format(t):
            continue
        if t not in ordered:
            ordered.append(t)
        if len(ordered) >= MAX_TICKER_LIBRARY_SIZE:
            break
    return ordered if ordered else MAG7_TICKERS.copy()


def _normalize_report_dates(raw_dates) -> list:
    normalized = []
    seen_values = set()
    if not isinstance(raw_dates, list):
        return normalized
    for entry in raw_dates:
        if not isinstance(entry, dict):
            continue
        value = str(entry.get("value", "")).strip()
        display = str(entry.get("display", "")).strip()
        if not value:
            continue
        if value in seen_values:
            continue
        seen_values.add(value)
        normalized.append({"display": display or value, "value": value})
    return normalized


def _normalize_available_report_dates_cache(raw_cache) -> dict:
    normalized = {}
    if not isinstance(raw_cache, dict):
        return normalized
    for raw_ticker, raw_dates in raw_cache.items():
        ticker = _normalize_ticker(raw_ticker)
        if not _is_valid_ticker_format(ticker):
            continue
        dates = _normalize_report_dates(raw_dates)
        if not dates:
            continue
        normalized[ticker] = dates
        if len(normalized) >= MAX_REPORT_DATE_CACHE_TICKERS:
            break
    return normalized


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    return str(value)


def _build_ai_outlook_cache_key(quarterly_analysis: dict | None, dcf_data: dict | None) -> str:
    payload = {
        "version": AI_OUTLOOK_CACHE_VERSION,
        "quarterly_analysis": _json_safe(quarterly_analysis or {}),
        "dcf_data": _json_safe(dcf_data or {}),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _sanitize_ai_valuation_language(text: str) -> str:
    """
    Keep valuation wording assumption-aware in rendered AI text.
    Also normalizes older cached outputs produced before prompt updates.
    """
    if not isinstance(text, str):
        return text
    sanitized = text
    sanitized = _remove_inline_urls_from_ai_text(sanitized)
    replacements = [
        (r"(?i)\bfundamental floor\b", "model-implied value under current assumptions"),
        (r"(?i)\bvaluation floor\b", "model-implied value under current assumptions"),
        (r"(?i)\bintrinsic floor\b", "model-implied value under current assumptions"),
        (r"(?i)\bhard floor\b", "assumption-sensitive downside case"),
    ]
    for pattern, replacement in replacements:
        sanitized = re.sub(pattern, replacement, sanitized)
    # Remove emoji/symbol pictographs for a cleaner, professional report tone.
    return _strip_visual_markers(sanitized)


def _remove_inline_urls_from_ai_text(text: str) -> str:
    """
    Keep narrative citations readable as source/date text while removing clickable URLs.
    URLs are preserved in Step 06 Citations dropdown only.
    """
    if not isinstance(text, str) or not text:
        return text

    cleaned = text

    # Convert markdown links to link text only.
    cleaned = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r"\1", cleaned)

    # Remove valid URL tokens from narrative text.
    for raw in URL_REGEX.findall(cleaned):
        normalized = _normalize_url_candidate(raw)
        if not normalized:
            continue
        cleaned = cleaned.replace(raw, "")

    # Cleanup punctuation artifacts after URL removal.
    cleaned = re.sub(r"\(\s*,", "(", cleaned)
    cleaned = re.sub(r",\s*,", ", ", cleaned)
    cleaned = re.sub(r"\s+,", ",", cleaned)
    cleaned = re.sub(r",\s*\)", ")", cleaned)
    cleaned = re.sub(r"\(\s*\)", "", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _strip_visual_markers(text: str) -> str:
    if not isinstance(text, str):
        return str(text)
    cleaned = EMOJI_SYMBOL_REGEX.sub("", text)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = cleaned.strip()
    return cleaned


def _pick_keyword(raw_text: str, options: list[str], default: str) -> str:
    cleaned = _strip_visual_markers(raw_text)
    lowered = cleaned.lower()
    for option in options:
        if option.lower() in lowered:
            return option
    return default


def _extract_bullets_from_heading(text: str, heading_tokens: list[str], max_items: int = 6) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    lines = text.splitlines()
    in_section = False
    collected: list[str] = []
    token_set = [t.lower() for t in heading_tokens]

    for raw_line in lines:
        line = raw_line.strip()
        lower = line.lower()
        if any(token in lower for token in token_set):
            in_section = True
            continue
        if not in_section:
            continue
        if not line:
            if collected:
                break
            continue
        if line.startswith("### ") or line.startswith("## ") or line.startswith("---"):
            break
        if line.startswith("**") and line.endswith("**"):
            break
        bullet_match = re.match(r"^[-*•]\s+(.+)$", line)
        if not bullet_match:
            if collected:
                break
            continue
        value = _sanitize_ai_valuation_language(bullet_match.group(1))
        if value and "null" not in value.lower():
            collected.append(value)
        if len(collected) >= max_items:
            break
    return collected


def _build_outlook_view_model(extracted: dict | None, full_analysis: str) -> dict:
    payload = extracted if isinstance(extracted, dict) else {}
    analysis_text = full_analysis if isinstance(full_analysis, str) else ""

    short_text = str(payload.get("short_term_stance", "") or "")
    if not short_text and analysis_text:
        m = re.search(r"Directional Stance:\s*([^\n]+)", analysis_text, flags=re.IGNORECASE)
        short_text = m.group(1) if m else ""
    short_stance = _pick_keyword(short_text, ["Bullish", "Neutral", "Bearish"], "Neutral")

    fund_text = str(payload.get("fundamental_outlook", "") or "")
    if not fund_text and analysis_text:
        m = re.search(r"Fundamental Outlook:\s*([^\n]+)", analysis_text, flags=re.IGNORECASE)
        fund_text = m.group(1) if m else ""
    fund_outlook = _pick_keyword(fund_text, ["Strong", "Stable", "Weakening"], "Stable")

    stock_text = str(payload.get("stock_outlook", "") or "")
    if not stock_text and analysis_text:
        m = re.search(r"Stock Outlook:\s*([^\n]+)", analysis_text, flags=re.IGNORECASE)
        stock_text = m.group(1) if m else ""
    stock_outlook = _pick_keyword(stock_text, ["Bullish", "Neutral", "Bearish"], "Neutral")

    stock_horizon = str(payload.get("stock_outlook_horizon", "") or "")
    if not stock_horizon and analysis_text:
        m = re.search(r"Stock Outlook:.*?\bover\s+([^\n]+)", analysis_text, flags=re.IGNORECASE)
        stock_horizon = m.group(1) if m else ""
    stock_horizon = _pick_keyword(stock_horizon, ["Short-term", "Mid-term", "Long-term"], "Mid-term")

    conviction_text = str(payload.get("stock_conviction", payload.get("fundamental_conviction", "")) or "")
    if not conviction_text and analysis_text:
        m = re.search(r"Conviction:\s*([^\n]+)", analysis_text, flags=re.IGNORECASE)
        conviction_text = m.group(1) if m else ""
    stock_conviction = _pick_keyword(conviction_text, ["High", "Medium", "Low"], "Medium")

    summary = _sanitize_ai_valuation_language(str(payload.get("summary", "") or ""))
    if (not summary or "null" in summary.lower()) and analysis_text:
        paragraphs = [
            _sanitize_ai_valuation_language(p.strip())
            for p in re.split(r"\n\s*\n", analysis_text)
            if p.strip() and not p.strip().startswith("#")
        ]
        paragraphs = [p for p in paragraphs if len(p) >= 40]
        summary = paragraphs[0] if paragraphs else ""

    drivers = payload.get("short_term_drivers") if isinstance(payload.get("short_term_drivers"), list) else []
    drivers = [_sanitize_ai_valuation_language(str(d)) for d in drivers if d]
    if not drivers:
        drivers = _extract_bullets_from_heading(analysis_text, ["key drivers"], max_items=5)

    risks = payload.get("short_term_risks") if isinstance(payload.get("short_term_risks"), list) else []
    risks = [_sanitize_ai_valuation_language(str(r)) for r in risks if r]
    if not risks:
        risks = _extract_bullets_from_heading(analysis_text, ["key risks"], max_items=5)

    gaps = payload.get("evidence_gaps") if isinstance(payload.get("evidence_gaps"), list) else []
    gaps = [_sanitize_ai_valuation_language(str(g)) for g in gaps if g]
    if not gaps:
        gaps = _extract_bullets_from_heading(analysis_text, ["evidence gaps"], max_items=6)

    key_conditional = _sanitize_ai_valuation_language(str(payload.get("key_conditional", "") or ""))
    if (not key_conditional or "null" in key_conditional.lower()) and analysis_text:
        m = re.search(r"Key Conditional:\s*\"?([^\n\"]+)\"?", analysis_text, flags=re.IGNORECASE)
        key_conditional = _sanitize_ai_valuation_language(m.group(1)) if m else ""

    return {
        "short_stance": short_stance,
        "fund_outlook": fund_outlook,
        "stock_outlook": stock_outlook,
        "stock_horizon": stock_horizon,
        "stock_conviction": stock_conviction,
        "summary": summary,
        "drivers": [d for d in drivers if d and "null" not in d.lower()][:5],
        "risks": [r for r in risks if r and "null" not in r.lower()][:5],
        "evidence_gaps": [g for g in gaps if g and "null" not in g.lower()][:6],
        "key_conditional": key_conditional if key_conditional and "null" not in key_conditional.lower() else "",
    }


def load_ui_cache() -> dict:
    default = _default_ui_cache()
    if not UI_CACHE_PATH.exists():
        return default
    try:
        with UI_CACHE_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return default
        stored_version = data.get("version", 0)
        if stored_version != UI_CACHE_VERSION:
            return {
                "version": UI_CACHE_VERSION,
                "available_report_dates_version": REPORT_DATES_CACHE_VERSION,
                "ticker_library": _normalize_ticker_library(data.get("ticker_library", [])),
                "last_selected_ticker": _normalize_ticker(data.get("last_selected_ticker", DEFAULT_TICKER)) or DEFAULT_TICKER,
                "available_report_dates": {},
                "results": {},
            }
        report_dates_version = str(data.get("available_report_dates_version", "")).strip()
        available_dates_cache = (
            _normalize_available_report_dates_cache(data.get("available_report_dates", {}))
            if report_dates_version == REPORT_DATES_CACHE_VERSION
            else {}
        )
        cache = {
            "version": data.get("version", UI_CACHE_VERSION),
            "available_report_dates_version": REPORT_DATES_CACHE_VERSION,
            "ticker_library": _normalize_ticker_library(data.get("ticker_library", [])),
            "last_selected_ticker": _normalize_ticker(data.get("last_selected_ticker", DEFAULT_TICKER)) or DEFAULT_TICKER,
            "available_report_dates": available_dates_cache,
            # Keep DCF/AI outputs session-local; persisting them to disk leaks state across users.
            "results": {},
        }
        return cache
    except Exception:
        return default


def save_ui_cache(cache_obj: dict) -> None:
    try:
        UI_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": UI_CACHE_VERSION,
            "available_report_dates_version": REPORT_DATES_CACHE_VERSION,
            "ticker_library": _normalize_ticker_library(cache_obj.get("ticker_library", [])),
            "last_selected_ticker": _normalize_ticker(cache_obj.get("last_selected_ticker", DEFAULT_TICKER)) or DEFAULT_TICKER,
            "available_report_dates": _normalize_available_report_dates_cache(cache_obj.get("available_report_dates", {})),
        }
        tmp_path = UI_CACHE_PATH.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        tmp_path.replace(UI_CACHE_PATH)
    except Exception:
        # Persistence failure should never break the UI flow.
        pass


def build_context_key(ticker: str, end_date: str, num_quarters: int) -> str:
    return f"{_normalize_ticker(ticker)}|{end_date}|{int(num_quarters)}"


ANALYSIS_VIEWS = {
    "dashboard": {
        "label": "Dashboard",
        "summary": "Primary call, key growth signals, and a fast read on the setup.",
    },
    "deep_dive": {
        "label": "Deep Dive",
        "summary": "Model levers, assumptions, and operating momentum in one place.",
    },
    "compare": {
        "label": "Compare",
        "summary": "Cross-check your DCF against Street expectations and positioning.",
    },
    "reports": {
        "label": "Reports",
        "summary": "AI write-up, export actions, and supporting methodology.",
    },
}
DEFAULT_ANALYSIS_VIEW = "dashboard"


def _query_param_scalar(name: str) -> str | None:
    try:
        value = st.query_params.get(name)
    except Exception:
        return None

    if isinstance(value, list):
        value = value[0] if value else None

    if value is None:
        return None

    text = str(value).strip()
    return text or None


def _normalize_analysis_view(value: str | None) -> str:
    candidate = re.sub(r"[^a-z0-9_-]+", "-", str(value or "").strip().lower())
    return candidate if candidate in ANALYSIS_VIEWS else DEFAULT_ANALYSIS_VIEW


def _render_analysis_topbar(timestamp_text: str) -> None:
    safe_timestamp_text = html.escape(str(timestamp_text))

    st.markdown(
        f"""
<div id="analysis-topbar-anchor"></div>
<div class="app-topbar">
  <div class="app-topbar-brand">
    <span class="app-wordmark">Analyst Co-Pilot</span>
    <span class="app-version">Beta</span>
  </div>
  <div class="app-topbar-right">
    <div class="app-topbar-time">{safe_timestamp_text}</div>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )


def _clear_view_query_param() -> None:
    try:
        del st.query_params["view"]
    except Exception:
        pass


def _widget_value_kwargs(key: str, fallback_value):
    """Only pass an explicit widget default when Session State does not already own it."""
    if key in st.session_state:
        return {}
    return {"value": fallback_value}


def _queue_scroll_to_section(section_id: str) -> None:
    target = str(section_id or "").strip()
    st.session_state.pending_scroll_section = target


def _render_pending_scroll_to_section(section_id: str) -> None:
    pending = str(st.session_state.get("pending_scroll_section") or "").strip()
    if pending != section_id:
        return

    safe_target = re.sub(r"[^A-Za-z0-9_-]", "", section_id)
    if not safe_target:
        st.session_state.pending_scroll_section = None
        return

    components.html(
        f"""
<script>
(function() {{
  const p = window.parent;
  const targetId = "{safe_target}";
  let attempts = 0;
  const jump = () => {{
    const el = p.document.getElementById(targetId);
    if (el) {{
      el.scrollIntoView({{ behavior: "auto", block: "start" }});
      return;
    }}
    attempts += 1;
    if (attempts < 40) p.setTimeout(jump, 50);
  }};
  p.requestAnimationFrame(() => p.setTimeout(jump, 0));
}})();
</script>
        """,
        height=0,
        scrolling=False,
    )
    st.session_state.pending_scroll_section = None


def _extract_urls_from_text(text: str) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    raw_urls = URL_REGEX.findall(text)
    cleaned = []
    seen = set()
    markdown_urls = re.findall(r"\[[^\]]+\]\(([^)]+)\)", text)
    for raw in raw_urls + markdown_urls:
        candidate = _normalize_url_candidate(raw)
        if not candidate:
            continue
        if candidate in seen:
            continue
        seen.add(candidate)
        cleaned.append(candidate)
    return cleaned


def _normalize_url_candidate(raw_url: str) -> str:
    candidate = str(raw_url or "").strip().strip("<>")
    if not candidate:
        return ""
    candidate = candidate.rstrip(".,;:!?)")
    if "..." in candidate:
        return ""
    if candidate.lower().startswith("www."):
        candidate = f"https://{candidate}"
    if re.match(r"^https?://", candidate, flags=re.IGNORECASE):
        return candidate
    if re.match(r"^[A-Za-z0-9.-]+\.[A-Za-z]{2,}(/.*)?$", candidate):
        return f"https://{candidate}"
    return ""


def _compact_search_query(text: str, max_words: int = 10, max_chars: int = 80) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if not cleaned:
        return ""
    words = cleaned.split(" ")
    short = " ".join(words[:max_words]).strip()
    if len(short) > max_chars:
        short = short[:max_chars].rsplit(" ", 1)[0].strip()
    return short


def _source_search_fallback_url(source_name: str, headline_or_claim: str) -> str:
    source = str(source_name or "").strip().lower()
    query = _compact_search_query(headline_or_claim)
    if not source or not query:
        return ""
    encoded = quote(query)
    if "reuters" in source:
        return f"https://www.reuters.com/site-search/?query={encoded}"
    if "wall street journal" in source or source == "wsj" or " wsj" in source:
        return f"https://www.wsj.com/search?query={encoded}"
    if "bloomberg" in source:
        return f"https://www.bloomberg.com/search?query={encoded}"
    if "financial times" in source or source == "ft":
        return f"https://www.ft.com/search?q={encoded}"
    return ""


def _qualitative_source_url(item: dict) -> str:
    if not isinstance(item, dict):
        return ""
    explicit = _normalize_url_candidate(item.get("url", ""))
    if explicit:
        return explicit
    source_name = _clean_citation_field(item.get("source", ""))
    headline = _clean_citation_field(item.get("headline", ""))
    return _source_search_fallback_url(source_name, headline)


def _source_name_from_url(url: str) -> str:
    try:
        host = (urlparse(url).netloc or "").lower().strip()
        if host.startswith("www."):
            host = host[4:]
        return host or "external source"
    except Exception:
        return "external source"


def _clean_citation_field(value) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if text.lower() in {"none", "n/a", "na", "null", "unknown", "nan"}:
        return ""
    return text


def _short_claim_text(claim: str, max_chars: int = 180) -> str:
    text = _clean_citation_field(claim)
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars - 1].rstrip()}…"


def _merge_citations_for_step6(consensus_citations: list, forecast: dict, qualitative_sources: list | None = None) -> list:
    merged = []
    seen_urls = set()
    seen_text_refs = set()

    for cite in consensus_citations or []:
        if not isinstance(cite, dict):
            continue
        source_text = _clean_citation_field(cite.get("source_name", "")) or "Source"
        data_type_text = _clean_citation_field(cite.get("data_type", ""))
        date_text = _clean_citation_field(cite.get("access_date", ""))
        url = _normalize_url_candidate(cite.get("url", ""))
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        merged.append(
            {
                "source_name": source_text,
                "source": source_text,
                "date": date_text,
                "claim": "",
                "url": url,
                "data_type": data_type_text,
            }
        )

    if isinstance(forecast, dict) and not forecast.get("error"):
        ai_text = str(forecast.get("full_analysis", "") or "")
        extracted = forecast.get("extracted_forecast")
        extracted_text = json.dumps(extracted, ensure_ascii=False) if extracted is not None else ""
        for url in _extract_urls_from_text(f"{ai_text}\n{extracted_text}"):
            if url in seen_urls:
                continue
            seen_urls.add(url)
            merged.append(
                {
                    "source_name": f"AI synthesis ({_source_name_from_url(url)})",
                    "source": _source_name_from_url(url),
                    "date": "",
                    "claim": "",
                    "url": url,
                    "data_type": "External reference cited in Step 05",
                }
            )
        for cite in forecast.get("external_citations", []) or []:
            if not isinstance(cite, dict):
                continue
            claim_text = _clean_citation_field(cite.get("claim", ""))
            source_text = _clean_citation_field(cite.get("source", ""))
            date_text = _clean_citation_field(cite.get("date", ""))
            url = _normalize_url_candidate(cite.get("url", ""))
            if not url:
                fallback_urls = _extract_urls_from_text(
                    f"{cite.get('claim', '')}\n{cite.get('source', '')}\n{cite.get('date', '')}"
                )
                if fallback_urls:
                    url = fallback_urls[0]
            if not url:
                url = _source_search_fallback_url(source_text, claim_text)
            claim_label = _short_claim_text(claim_text)
            source_label = source_text or (f"AI synthesis ({_source_name_from_url(url)})" if url else "AI synthesis citation")
            context_parts = []
            if date_text:
                context_parts.append(f"Date: {date_text}")
            if claim_label:
                context_parts.append(f"Claim: {claim_label}")
            data_type_text = "External reference cited in Step 05"
            if context_parts:
                data_type_text = f"{data_type_text} | {' | '.join(context_parts)}"

            if url:
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                merged.append(
                    {
                        "source_name": source_label,
                        "source": source_text,
                        "date": date_text,
                        "claim": claim_text,
                        "url": url,
                        "data_type": data_type_text,
                    }
                )
                continue

            # Keep citation visible as plain text when no valid URL is present.
            text_ref_key = f"{claim_text}|{source_text}|{date_text}"
            if text_ref_key in seen_text_refs:
                continue
            seen_text_refs.add(text_ref_key)
            merged.append(
                {
                    "source_name": source_label,
                    "source": source_text,
                    "date": date_text,
                    "claim": claim_text,
                    "url": "",
                    "data_type": f"{data_type_text} (no URL provided)",
                }
            )

    qualitative_pool = []
    if isinstance(qualitative_sources, list):
        qualitative_pool.extend(qualitative_sources)
    if isinstance(forecast, dict):
        forecast_qualitative = forecast.get("qualitative_sources_used", [])
        if isinstance(forecast_qualitative, list):
            qualitative_pool.extend(forecast_qualitative)

    for item in qualitative_pool:
        if not isinstance(item, dict):
            continue
        url = _qualitative_source_url(item)
        if url and url in seen_urls:
            continue
        headline = _clean_citation_field(item.get("headline", "")) or "Analyst commentary"
        source_name = _clean_citation_field(item.get("source", "")) or "Publication"
        published = _clean_citation_field(item.get("date", ""))
        context = f"{headline}{f' ({published})' if published else ''}"
        if not url:
            text_ref_key = f"qual|{source_name}|{context}"
            if text_ref_key in seen_text_refs:
                continue
            seen_text_refs.add(text_ref_key)
            merged.append(
                {
                    "source_name": source_name,
                    "source": source_name,
                    "date": published,
                    "claim": headline,
                    "url": "",
                    "data_type": f"Qualitative market/company context | {context} (no URL provided)",
                }
            )
            continue
        seen_urls.add(url)
        merged.append(
            {
                "source_name": source_name,
                "source": source_name,
                "date": published,
                "claim": headline,
                "url": url,
                "data_type": f"Qualitative market/company context | {context}",
            }
        )

    return merged


def _citation_date_value(cite: dict) -> str:
    if not isinstance(cite, dict):
        return ""
    date_text = _clean_citation_field(cite.get("date", ""))
    if date_text:
        return date_text
    data_type = _clean_citation_field(cite.get("data_type", ""))
    if not data_type:
        return ""
    match = re.search(r"Date:\s*([^|]+)", data_type, flags=re.IGNORECASE)
    if not match:
        return ""
    return _clean_citation_field(match.group(1))


def _number_citations(citations: list) -> list:
    numbered = []
    for idx, cite in enumerate(citations or [], start=1):
        if not isinstance(cite, dict):
            continue
        entry = dict(cite)
        entry["number"] = idx
        entry["citation_tag"] = f"C{idx}"
        numbered.append(entry)
    return numbered


def _citation_token_pattern(text: str) -> str:
    token = _clean_citation_field(text)
    if not token:
        return ""
    escaped = re.escape(token)
    if token and token[0].isalnum():
        escaped = rf"\b{escaped}"
    if token and token[-1].isalnum():
        escaped = rf"{escaped}\b"
    return escaped


def _apply_inline_numeric_citations(text: str, numbered_citations: list) -> str:
    if not isinstance(text, str) or not text.strip() or not numbered_citations:
        return text
    rendered = re.sub(r"\[(\d{1,2})\]", r"[C\1]", text)
    for cite in numbered_citations:
        citation_tag = str(cite.get("citation_tag") or f"C{cite.get('number')}")
        marker = f"[{citation_tag}]"
        if marker in rendered:
            continue
        source_name = _clean_citation_field(cite.get("source_name", ""))
        date_value = _citation_date_value(cite)
        source_pattern = _citation_token_pattern(source_name)
        date_pattern = _citation_token_pattern(date_value)
        replaced = False

        if source_pattern and date_pattern:
            pattern = rf"(\([^)]*{source_pattern}[^)]*{date_pattern}[^)]*\))(?!\s*\[C\d+\])"
            rendered, count = re.subn(
                pattern,
                lambda m: f"{m.group(1)} {marker}",
                rendered,
                count=1,
                flags=re.IGNORECASE,
            )
            replaced = count > 0

        if not replaced and source_pattern:
            pattern = rf"(\([^)]*{source_pattern}[^)]*\))(?!\s*\[C\d+\])"
            rendered, count = re.subn(
                pattern,
                lambda m: f"{m.group(1)} {marker}",
                rendered,
                count=1,
                flags=re.IGNORECASE,
            )
            replaced = count > 0

        if not replaced and source_pattern:
            rendered, count = re.subn(
                rf"({source_pattern})(?!\s*\[C\d+\])",
                lambda m: f"{m.group(1)} {marker}",
                rendered,
                count=1,
                flags=re.IGNORECASE,
            )
            replaced = count > 0

        if not replaced and date_pattern:
            rendered, _ = re.subn(
                rf"({date_pattern})(?!\s*\[C\d+\])",
                lambda m: f"{m.group(1)} {marker}",
                rendered,
                count=1,
                flags=re.IGNORECASE,
            )

    rendered = re.sub(r"(\[C\d+\])(?:\s*\1)+", r"\1", rendered)
    rendered = re.sub(r"[ \t]{2,}", " ", rendered)
    return rendered.strip()


def _format_numbered_citations_markdown(numbered_citations: list) -> str:
    lines = []
    for cite in numbered_citations or []:
        citation_tag = str(cite.get("citation_tag") or f"C{cite.get('number')}")
        source_name = _clean_citation_field(cite.get("source_name", "")) or "Source"
        date_value = _citation_date_value(cite)
        claim_text = _short_claim_text(cite.get("claim", ""), max_chars=140)
        data_type = _clean_citation_field(cite.get("data_type", ""))
        data_type = re.sub(r"\s*\(no URL provided\)\s*$", "", data_type, flags=re.IGNORECASE)
        data_type = re.sub(r"^External reference cited in Step 05\s*\|\s*", "", data_type, flags=re.IGNORECASE)
        data_type = re.sub(r"\bDate:\s*[^|]+", "", data_type, flags=re.IGNORECASE)
        data_type = re.sub(r"\bClaim:\s*[^|]+", "", data_type, flags=re.IGNORECASE)
        data_type = re.sub(r"\s*\|\s*", " ", data_type).strip(" -|")

        detail_parts = []
        if date_value:
            detail_parts.append(date_value)
        if claim_text:
            detail_parts.append(claim_text)
        elif data_type:
            detail_parts.append(data_type)

        url = _normalize_url_candidate(cite.get("url", ""))
        source_md = f"[{source_name}]({url})" if url else source_name
        prefix = f"- **[{citation_tag}]** {source_md}"
        if detail_parts:
            lines.append(f"{prefix} — {' | '.join(detail_parts)}")
        else:
            lines.append(prefix)
    return "\n".join(lines)


def _secret_section(section_name: str) -> dict:
    try:
        section = st.secrets.get(section_name, {})
        if isinstance(section, Mapping):
            return {str(k): section[k] for k in section.keys()}
        return {}
    except Exception:
        return {}


def _env_or_secret(name: str, default: str = "", aliases: tuple[str, ...] = (), section_name: str = "") -> str:
    env_keys = (name, name.lower(), *aliases)
    for env_key in env_keys:
        env_value = str(os.environ.get(env_key, "")).strip()
        if env_value:
            return env_value

    candidate_secret_keys = (
        name,
        name.lower(),
        name.replace("SMTP_", "").replace("FORMSUBMIT_", "").lower(),
        *aliases,
    )
    for key in candidate_secret_keys:
        try:
            value = st.secrets.get(key, "")
        except Exception:
            value = ""
        text = str(value or "").strip()
        if text:
            return text

    if section_name:
        section = _secret_section(section_name)
        for key in candidate_secret_keys:
            value = section.get(key, "")
            text = str(value or "").strip()
            if text:
                return text

    return str(default or "").strip()


def _smtp_config() -> tuple[dict, list]:
    smtp_host = _env_or_secret("SMTP_HOST", aliases=("host",), section_name="smtp")
    smtp_user = _env_or_secret("SMTP_USER", aliases=("user", "username"), section_name="smtp")
    smtp_password = _env_or_secret("SMTP_PASSWORD", aliases=("password", "pass"), section_name="smtp")
    smtp_from = _env_or_secret("SMTP_FROM", smtp_user, aliases=("from", "from_email", "sender"), section_name="smtp") or smtp_user
    smtp_port_raw = _env_or_secret("SMTP_PORT", "587", aliases=("port",), section_name="smtp")
    smtp_starttls = _env_or_secret("SMTP_STARTTLS", "1", aliases=("starttls", "tls", "use_starttls"), section_name="smtp").lower() in {"1", "true", "yes"}

    # Compatibility path for common Streamlit secret shape:
    # [email]
    # gmail = "your@gmail.com"
    # password = "app_password"
    email_section = _secret_section("email")
    email_user = str(
        email_section.get("gmail", "")
        or email_section.get("user", "")
        or email_section.get("username", "")
    ).strip()
    email_password = str(
        email_section.get("password", "")
        or email_section.get("app_password", "")
    ).strip()
    if not smtp_user and email_user:
        smtp_user = email_user
    if not smtp_password and email_password:
        smtp_password = email_password
    if not smtp_from and smtp_user:
        smtp_from = smtp_user
    if not smtp_host and smtp_user and "@gmail.com" in smtp_user.lower():
        smtp_host = "smtp.gmail.com"
    if (not smtp_port_raw or not str(smtp_port_raw).strip()) and smtp_host == "smtp.gmail.com":
        smtp_port_raw = "587"
    if smtp_host == "smtp.gmail.com":
        smtp_starttls = True

    missing = []
    if not smtp_host:
        missing.append("SMTP_HOST")
    if not smtp_user:
        missing.append("SMTP_USER")
    if not smtp_password:
        missing.append("SMTP_PASSWORD")
    if not smtp_from:
        missing.append("SMTP_FROM")
    return (
        {
            "host": smtp_host,
            "user": smtp_user,
            "password": smtp_password,
            "from": smtp_from,
            "port_raw": smtp_port_raw,
            "starttls": smtp_starttls,
        },
        missing,
    )


def _formsubmit_target_email() -> str:
    candidate = _env_or_secret(
        "FORMSUBMIT_TO",
        CONTACT_EMAIL_TO,
        aliases=("to", "email", "recipient"),
        section_name="formsubmit",
    )
    return candidate if EMAIL_REGEX.match(candidate) else ""


def _render_formsubmit_fallback_form(target_email: str) -> None:
    action = f"https://formsubmit.co/{quote(target_email)}"
    html_form = f"""
    <style>
      .fs-wrap {{
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 16px;
        background: #ffffff;
        font-family: Inter, sans-serif;
      }}
      .fs-grid {{
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 12px;
      }}
      .fs-group {{
        display: flex;
        flex-direction: column;
        gap: 6px;
      }}
      .fs-group label {{
        font-size: 0.95rem;
        font-weight: 600;
        color: #111827;
      }}
      .fs-group input, .fs-group textarea {{
        border: 1px solid #d1d5db;
        border-radius: 8px;
        padding: 10px 12px;
        font-size: 0.95rem;
        width: 100%;
        box-sizing: border-box;
      }}
      .fs-full {{
        margin-top: 12px;
      }}
      .fs-btn {{
        margin-top: 14px;
        background: #2563eb;
        color: #ffffff;
        border: none;
        border-radius: 8px;
        padding: 10px 18px;
        font-size: 0.95rem;
        font-weight: 600;
        cursor: pointer;
      }}
      .fs-note {{
        margin-top: 10px;
        color: #6b7280;
        font-size: 0.85rem;
      }}
      @media (max-width: 760px) {{
        .fs-grid {{
          grid-template-columns: 1fr;
        }}
      }}
    </style>
    <div class="fs-wrap">
      <form action="{action}" method="POST" target="_blank">
        <input type="hidden" name="_subject" value="Analyst Co-Pilot Contact Form">
        <input type="hidden" name="_captcha" value="false">
        <input type="hidden" name="_template" value="table">
        <div class="fs-grid">
          <div class="fs-group">
            <label for="fs_first">First name*</label>
            <input id="fs_first" name="first_name" type="text" required placeholder="First name">
          </div>
          <div class="fs-group">
            <label for="fs_last">Last name*</label>
            <input id="fs_last" name="last_name" type="text" required placeholder="Last name">
          </div>
          <div class="fs-group">
            <label for="fs_email">Email*</label>
            <input id="fs_email" name="email" type="email" required placeholder="you@example.com">
          </div>
        </div>
        <div class="fs-group fs-full">
          <label for="fs_message">Your idea or proposal*</label>
          <textarea id="fs_message" name="message" rows="7" required placeholder="Describe your idea, proposal, or feedback."></textarea>
        </div>
        <button class="fs-btn" type="submit">Submit</button>
      </form>
      <div class="fs-note">Opens FormSubmit in a new tab. If first use, activate recipient email once.</div>
    </div>
    """
    components.html(html_form, height=520, scrolling=False)


def _send_contact_email(
    *,
    first_name: str,
    last_name: str,
    sender_email: str,
    message: str,
) -> tuple[bool, str]:
    if not EMAIL_REGEX.match(CONTACT_EMAIL_TO):
        return (
            False,
            "Contact recipient email is not configured. Set CONTACT_EMAIL_TO or FORMSUBMIT_TO in deployment secrets/env.",
        )

    smtp, missing = _smtp_config()
    if missing:
        return (
            False,
            "Email sending is not configured. Missing: "
            + ", ".join(missing)
            + ". Add SMTP settings in deployment secrets/env (top-level keys or [smtp] section).",
        )

    try:
        smtp_port = int(smtp.get("port_raw") or "587")
    except Exception:
        smtp_port = 587

    full_name = f"{first_name.strip()} {last_name.strip()}".strip()
    msg = EmailMessage()
    msg["Subject"] = f"Analyst Co-Pilot Contact Form - {full_name or sender_email}"
    msg["From"] = smtp["from"]
    msg["To"] = CONTACT_EMAIL_TO
    msg["Reply-To"] = sender_email
    msg.set_content(
        "New contact form submission\n\n"
        f"Name: {full_name or 'N/A'}\n"
        f"Email: {sender_email}\n\n"
        "Message:\n"
        f"{message.strip()}\n"
    )

    try:
        if smtp_port == 465:
            with smtplib.SMTP_SSL(smtp["host"], smtp_port, timeout=20, context=ssl.create_default_context()) as server:
                server.login(smtp["user"], smtp["password"])
                server.send_message(msg)
        else:
            with smtplib.SMTP(smtp["host"], smtp_port, timeout=20) as server:
                server.ehlo()
                if smtp["starttls"]:
                    server.starttls(context=ssl.create_default_context())
                    server.ehlo()
                server.login(smtp["user"], smtp["password"])
                server.send_message(msg)
        return True, "Message sent successfully."
    except Exception:
        return False, "Failed to send message. Check SMTP configuration and server connectivity."


def _metadata_from_dict(raw: dict) -> DataQualityMetadata:
    if not isinstance(raw, dict):
        return DataQualityMetadata()
    return DataQualityMetadata(
        value=raw.get("value"),
        units=raw.get("units", "USD"),
        period_end=raw.get("period_end"),
        period_type=raw.get("period_type"),
        source_path=raw.get("source_path"),
        retrieved_at=raw.get("retrieved_at"),
        reliability_score=raw.get("reliability_score", 0),
        notes=raw.get("notes"),
        is_estimated=raw.get("is_estimated", False),
        fallback_reason=raw.get("fallback_reason"),
    )


def snapshot_from_dict(raw: dict) -> NormalizedFinancialSnapshot:
    base = raw if isinstance(raw, dict) else {}
    ticker = _normalize_ticker(base.get("ticker")) or "UNKNOWN"
    snapshot = NormalizedFinancialSnapshot(ticker)

    snapshot.retrieved_at = base.get("retrieved_at", snapshot.retrieved_at)
    snapshot.currency = base.get("currency", snapshot.currency)
    snapshot.company_name = base.get("company_name")
    snapshot.sector = base.get("sector")
    snapshot.industry = base.get("industry")
    snapshot.num_quarters_available = base.get("num_quarters_available", 0)
    snapshot.overall_quality_score = base.get("overall_quality_score", snapshot.overall_quality_score)
    snapshot.warnings = base.get("warnings", []) if isinstance(base.get("warnings", []), list) else []
    snapshot.errors = base.get("errors", []) if isinstance(base.get("errors", []), list) else []
    snapshot.wacc_components = base.get("wacc_components", {}) if isinstance(base.get("wacc_components", {}), dict) else {}
    snapshot.risk_free_rate = base.get("risk_free_rate")
    snapshot.rf_source = base.get("rf_source")
    snapshot.analyst_revenue_estimates = (
        base.get("analyst_revenue_estimates", [])
        if isinstance(base.get("analyst_revenue_estimates", []), list)
        else []
    )

    for field in SNAPSHOT_METADATA_FIELDS:
        setattr(snapshot, field, _metadata_from_dict(base.get(field, {})))

    return snapshot


def _upsert_ticker_in_library(ticker: str) -> None:
    t = _normalize_ticker(ticker)
    if not _is_valid_ticker_format(t):
        return

    cache = st.session_state.get("ui_cache", _default_ui_cache())
    library = _normalize_ticker_library(cache.get("ticker_library", []))
    if t not in library:
        library.append(t)
        library = _normalize_ticker_library(library)
        cache["ticker_library"] = library
        st.session_state.ui_cache = cache
        st.session_state.ticker_library = library
        save_ui_cache(cache)


def _get_persisted_report_dates(ticker: str) -> list:
    normalized_ticker = _normalize_ticker(ticker)
    if not _is_valid_ticker_format(normalized_ticker):
        return []

    cache = st.session_state.get("ui_cache", _default_ui_cache())
    dates_cache = _normalize_available_report_dates_cache(cache.get("available_report_dates", {}))
    return dates_cache.get(normalized_ticker, [])


def _persist_report_dates_for_ticker(ticker: str, dates: list) -> None:
    normalized_ticker = _normalize_ticker(ticker)
    if not _is_valid_ticker_format(normalized_ticker):
        return

    normalized_dates = _normalize_report_dates(dates)
    if not normalized_dates:
        return

    cache = st.session_state.get("ui_cache", _default_ui_cache())
    dates_cache = _normalize_available_report_dates_cache(cache.get("available_report_dates", {}))
    if normalized_ticker in dates_cache:
        dates_cache.pop(normalized_ticker, None)
    dates_cache[normalized_ticker] = normalized_dates

    while len(dates_cache) > MAX_REPORT_DATE_CACHE_TICKERS:
        oldest_ticker = next(iter(dates_cache))
        dates_cache.pop(oldest_ticker, None)

    cache["available_report_dates"] = dates_cache
    st.session_state.ui_cache = cache
    save_ui_cache(cache)


def _load_report_dates_for_ticker(ticker: str, force_refresh: bool = False) -> list:
    normalized_ticker = _normalize_ticker(ticker)
    if not _is_valid_ticker_format(normalized_ticker):
        return []

    persisted_dates = _get_persisted_report_dates(normalized_ticker)
    if persisted_dates and not force_refresh:
        return persisted_dates

    if force_refresh:
        fetched_dates = _call_with_timeout(
            get_available_report_dates,
            normalized_ticker,
            timeout_seconds=AVAILABLE_DATES_TIMEOUT_SECONDS,
            fallback=[],
        )
    else:
        fetched_dates = cached_available_dates(normalized_ticker)

    normalized_dates = _normalize_report_dates(fetched_dates)
    if normalized_dates:
        _persist_report_dates_for_ticker(normalized_ticker, normalized_dates)
        return normalized_dates

    return persisted_dates if persisted_dates else []


def _restore_cached_results_for_context(ticker: str, end_date: str, num_quarters: int) -> dict:
    restored = {"dcf": False, "ai": False}
    if not ticker or not end_date or num_quarters is None:
        return restored

    cache = st.session_state.get("ui_cache", _default_ui_cache())
    context_key = build_context_key(ticker, end_date, num_quarters)
    entry = cache.get("results", {}).get(context_key, {})
    if not isinstance(entry, dict):
        return restored

    dcf_entry = entry.get("dcf")
    if isinstance(dcf_entry, dict):
        engine_result = dcf_entry.get("engine_result")
        snapshot_dict = dcf_entry.get("snapshot")
        if isinstance(engine_result, dict) and isinstance(snapshot_dict, dict):
            try:
                snapshot = snapshot_from_dict(snapshot_dict)
                ui_adapter = DCFUIAdapter(engine_result, snapshot)
                st.session_state.dcf_ui_adapter = ui_adapter
                st.session_state.dcf_engine_result = engine_result
                st.session_state.dcf_snapshot = snapshot
                st.session_state.dcf_wacc = dcf_entry.get("dcf_wacc")
                st.session_state.dcf_fcf_growth = dcf_entry.get("dcf_fcf_growth")
                terminal_growth_from_cache = dcf_entry.get("dcf_terminal_growth")
                if terminal_growth_from_cache is None:
                    terminal_growth_from_cache = (engine_result.get("assumptions", {}).get("terminal_growth_rate") or 0.03) * 100
                st.session_state.dcf_terminal_growth = terminal_growth_from_cache
                st.session_state.dcf_terminal_scenario = dcf_entry.get("dcf_terminal_scenario")
                st.session_state.dcf_custom_multiple = dcf_entry.get("dcf_custom_multiple")
                restored["dcf"] = True
            except Exception:
                pass

    ai_entry = entry.get("ai_outlook")
    if isinstance(ai_entry, dict) and isinstance(ai_entry.get("independent_forecast"), dict):
        st.session_state.independent_forecast = ai_entry.get("independent_forecast")
        st.session_state.forecast_just_generated = False
        restored["ai"] = True

    if restored["dcf"] or restored["ai"]:
        st.session_state.last_restore_key = context_key

    return restored


def _persist_dcf_result_for_context(ticker: str, end_date: str, num_quarters: int) -> None:
    if not ticker or not end_date or num_quarters is None:
        return

    engine_result = st.session_state.get("dcf_engine_result")
    snapshot = st.session_state.get("dcf_snapshot")
    if not isinstance(engine_result, dict) or snapshot is None:
        return

    snapshot_dict = snapshot.to_dict() if hasattr(snapshot, "to_dict") else None
    if not isinstance(snapshot_dict, dict):
        return

    cache = st.session_state.get("ui_cache", _default_ui_cache())
    results = cache.setdefault("results", {})
    context_key = build_context_key(ticker, end_date, num_quarters)
    entry = results.get(context_key, {})
    if not isinstance(entry, dict):
        entry = {}

    entry.update({
        "ticker": _normalize_ticker(ticker),
        "end_date": end_date,
        "num_quarters": int(num_quarters),
        "updated_at": datetime.utcnow().isoformat(),
    })
    entry["dcf"] = {
        "dcf_wacc": st.session_state.get("dcf_wacc"),
        "dcf_fcf_growth": st.session_state.get("dcf_fcf_growth"),
        "dcf_terminal_growth": st.session_state.get("dcf_terminal_growth"),
        "dcf_terminal_scenario": st.session_state.get("dcf_terminal_scenario"),
        "dcf_custom_multiple": st.session_state.get("dcf_custom_multiple"),
        "engine_result": _json_safe(engine_result),
        "snapshot": _json_safe(snapshot_dict),
    }
    results[context_key] = entry
    cache["results"] = results
    cache["last_selected_ticker"] = _normalize_ticker(ticker)
    st.session_state.ui_cache = cache
    save_ui_cache(cache)


def _persist_ai_result_for_context(ticker: str, end_date: str, num_quarters: int) -> None:
    if not ticker or not end_date or num_quarters is None:
        return

    forecast = st.session_state.get("independent_forecast")
    if not isinstance(forecast, dict):
        return

    cache = st.session_state.get("ui_cache", _default_ui_cache())
    results = cache.setdefault("results", {})
    context_key = build_context_key(ticker, end_date, num_quarters)
    entry = results.get(context_key, {})
    if not isinstance(entry, dict):
        entry = {}

    entry.update({
        "ticker": _normalize_ticker(ticker),
        "end_date": end_date,
        "num_quarters": int(num_quarters),
        "updated_at": datetime.utcnow().isoformat(),
    })
    entry["ai_outlook"] = {
        "independent_forecast": _json_safe(forecast),
        "forecast_date": forecast.get("forecast_date"),
    }
    results[context_key] = entry
    cache["results"] = results
    cache["last_selected_ticker"] = _normalize_ticker(ticker)
    st.session_state.ui_cache = cache
    save_ui_cache(cache)


def _call_with_timeout(func, *args, timeout_seconds: int, fallback=None):
    """
    Prevent long-running upstream data calls from blocking the first paint forever.
    Returns fallback on timeout/error.
    """
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(func, *args)
    try:
        return future.result(timeout=timeout_seconds)
    except FuturesTimeoutError:
        return fallback
    except Exception:
        return fallback
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


# --- Cached API Functions ---
# These decorators cache results so API calls only happen once per input
# TTL (time-to-live) of 1 hour = 3600 seconds

@st.cache_data(ttl=3600, show_spinner=False)
def cached_quarterly_analysis(
    ticker: str,
    num_quarters: int = DEFAULT_INITIAL_QUARTERS,
    end_date: str = None,
    history_source_version: str = HISTORY_SOURCE_VERSION,
) -> dict:
    """Cached version of analyze_quarterly_trends to avoid API rate limits."""
    _ = history_source_version  # cache-key salt when quarterly history sourcing logic changes
    fmp_api_key = _env_or_secret("FMP_API_KEY", section_name="fmp") or None
    fallback = {
        "ticker": ticker,
        "analysis_date": datetime.now().strftime("%Y-%m-%d"),
        "historical_trends": {},
        "growth_rates": {},
        "projections": {},
        "consensus_estimates": {},
        "next_forecast_quarter": {},
        "errors": [f"Quarterly analysis timed out after {QUARTERLY_ANALYSIS_TIMEOUT_SECONDS}s."],
    }
    result = _call_with_timeout(
        analyze_quarterly_trends,
        ticker,
        num_quarters,
        end_date,
        fmp_api_key,
        timeout_seconds=QUARTERLY_ANALYSIS_TIMEOUT_SECONDS,
        fallback=fallback,
    )
    return result if isinstance(result, dict) else fallback

@st.cache_data(ttl=3600, show_spinner=False)
def cached_available_dates(ticker: str, history_source_version: str = HISTORY_SOURCE_VERSION) -> list:
    """Cached wrapper for get_available_report_dates."""
    _ = history_source_version  # cache-key salt when available-date sourcing logic changes
    result = _call_with_timeout(
        get_available_report_dates,
        ticker,
        timeout_seconds=AVAILABLE_DATES_TIMEOUT_SECONDS,
        fallback=[],
    )
    return result if isinstance(result, list) else []

@st.cache_data(ttl=3600, show_spinner=False)
def cached_independent_forecast(
    ticker: str,
    end_date: str,
    num_quarters: int,
    outlook_cache_key: str,
    company_name: str,
    dcf_data: dict = None,
) -> dict:
    """
    Cached version of generate_independent_forecast.
    outlook_cache_key is used to bust cache if underlying analysis or DCF inputs change.
    """
    _ = outlook_cache_key  # cache-key salt when AI synthesis inputs change
    analysis = cached_quarterly_analysis(ticker, num_quarters, end_date)
    return generate_independent_forecast(analysis, company_name, dcf_data)

@st.cache_data(ttl=3600, show_spinner=False)
def cached_financials(ticker: str) -> tuple:
    """Cached version of the annual-statement loader with provider metadata."""
    fallback = (
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        "Yahoo Finance",
        f"Annual financial statement fetch timed out after {FINANCIALS_TIMEOUT_SECONDS}s.",
    )
    result = _call_with_timeout(
        get_financials,
        ticker,
        timeout_seconds=FINANCIALS_TIMEOUT_SECONDS,
        fallback=fallback[:4],
    )
    if not isinstance(result, tuple):
        return fallback
    if len(result) != 4:
        return fallback
    income_stmt, balance_sheet, cash_flow, quarterly_cash_flow = result
    warning = None
    if income_stmt.empty:
        warning = "Yahoo Finance returned empty annual statements for this ticker."
    return income_stmt, balance_sheet, cash_flow, quarterly_cash_flow, "Yahoo Finance", warning

@st.cache_data(ttl=3600, show_spinner=False)
def cached_latest_date_info(ticker: str) -> dict:
    """Cached wrapper for get_latest_date_info from engine."""
    return get_latest_date_info(ticker)

@st.cache_data(ttl=3600, show_spinner=False)
def cached_company_name(ticker: str) -> str:
    """Lightweight cached lookup of company display name."""
    normalized_ticker = _normalize_ticker(ticker)
    if not _is_valid_ticker_format(normalized_ticker):
        return ""
    try:
        def _fetch_info() -> dict:
            return get_yf_info(get_yf_ticker(normalized_ticker))

        info = _call_with_timeout(
            _fetch_info,
            timeout_seconds=COMPANY_NAME_TIMEOUT_SECONDS,
            fallback={},
        ) or {}
        name = str(info.get("longName") or info.get("shortName") or "").strip()
        return name
    except Exception:
        return ""

@st.cache_data(ttl=3600, show_spinner=False)
def cached_financial_snapshot(ticker: str, suggestion_algo_version: str = "v2"):
    """Cached wrapper to get financial snapshot for suggested assumptions."""
    _ = suggestion_algo_version  # cache-key salt for suggestion logic revisions
    try:
        adapter = DataAdapter(ticker)
        snapshot = _call_with_timeout(
            adapter.fetch,
            timeout_seconds=SNAPSHOT_TIMEOUT_SECONDS,
            fallback=None,
        )
        return snapshot
    except Exception:
        return None

def run_dcf_analysis(ticker: str, wacc: float = None, fcf_growth: float = None,
                     terminal_growth: float = None, terminal_scenario: str = "current",
                     custom_multiple: float = None) -> tuple:
    """Run DCF analysis with user-adjustable assumptions. Returns (ui_adapter, engine_result, snapshot)."""
    try:
        adapter = DataAdapter(ticker)
        snapshot = _call_with_timeout(
            adapter.fetch,
            timeout_seconds=SNAPSHOT_TIMEOUT_SECONDS,
            fallback=None,
        )
        if snapshot is None:
            raise TimeoutError(
                f"Snapshot fetch timed out after {SNAPSHOT_TIMEOUT_SECONDS}s. "
                "Data provider may be temporarily unavailable; please retry."
            )
        
        # Create assumptions with user overrides
        assumptions = DCFAssumptions()
        if wacc is not None:
            assumptions.wacc = wacc / 100.0  # Convert from percentage
        if fcf_growth is not None:
            assumptions.fcf_growth_rate = fcf_growth / 100.0  # Convert from percentage
        if terminal_growth is not None:
            assumptions.terminal_growth_rate = terminal_growth / 100.0  # Convert from percentage

        # Set terminal multiple scenario
        assumptions.terminal_multiple_scenario = terminal_scenario
        if terminal_scenario == "custom" and custom_multiple is not None:
            assumptions.exit_multiple = custom_multiple
        assumptions.cashflow_discount_policy = "adaptive"
        assumptions.proxy_risk_premium_pct = 1.0
        
        engine = DCFEngine(snapshot, assumptions)
        engine_result = engine.run()
        ui_adapter = DCFUIAdapter(engine_result, snapshot)
        return (ui_adapter, engine_result, snapshot)
    except Exception as e:
        return (None, {"success": False, "error": str(e), "errors": [str(e)]}, None)


def _dcf_chat_to_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _dcf_chat_fmt_rate(value, decimals: int = 1) -> str:
    num = _dcf_chat_to_float(value)
    if num is None:
        return "N/A"
    return f"{num * 100:.{decimals}f}%"


def _dcf_chat_fmt_number(value, decimals: int = 2) -> str:
    num = _dcf_chat_to_float(value)
    if num is None:
        return "N/A"
    return f"{num:,.{decimals}f}"


def _parse_reliability_label(value):
    if isinstance(value, str) and value.endswith("/100"):
        try:
            return int(value.split("/")[0])
        except ValueError:
            return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return max(0, min(100, int(round(float(value)))))
    return None


def _effective_data_quality_score(metadata) -> int:
    if metadata is None:
        return 0

    score = getattr(metadata, "reliability_score", None)
    value = getattr(metadata, "value", None)
    has_value = value is not None and not pd.isna(value)

    if not has_value and score is None:
        return 0
    if not has_value and score == 100:
        return 0
    if score is None:
        return 0
    return max(0, min(100, int(round(float(score)))))


def _describe_data_quality_band(score: float) -> tuple[str, str]:
    numeric = _dcf_chat_to_float(score) or 0.0
    if numeric >= 90:
        return "High confidence", "Most core inputs are direct current or TTM values with limited fallback usage."
    if numeric >= 75:
        return "Good confidence", "One or two important inputs rely on annual proxies or light estimation."
    if numeric >= 60:
        return "Moderate confidence", "Several fields use fallback logic or partial data coverage."
    if numeric >= 40:
        return "Low confidence", "Key metrics rely on weak or estimated inputs, so the output should be used cautiously."
    return "Very low confidence", "Major data gaps make this output exploratory rather than decision-grade."


def _summarize_data_quality_reason(label: str, metadata, score: int) -> str:
    if metadata is None:
        return "missing or unscored"

    source = str(getattr(metadata, "source_path", "") or "").lower()
    notes = str(getattr(metadata, "notes", "") or "").lower()
    fallback = str(getattr(metadata, "fallback_reason", "") or "").lower()
    period_type = str(getattr(metadata, "period_type", "") or "").lower()
    combined = " ".join(part for part in [source, notes, fallback, period_type] if part)

    if score == 0:
        return "missing or unusable"

    if label == "Current Price":
        return "secondary market feed" if "fast_info" in combined else "direct market field"
    if label == "Shares Outstanding":
        if "derived(marketcap/price)" in combined:
            return "back-calculated from market cap and price"
        return "secondary reported field" if "fast_info" in combined else "direct reported share count"
    if label == "TTM Revenue":
        return "TTM built from the latest four quarters" if "quarterly_ttm" in combined else "annual revenue proxy"
    if label == "TTM Free Cash Flow":
        return "derived as CFO - CapEx and capped by the weaker input"
    if label == "TTM Operating Income":
        return "TTM built from the latest four quarters" if "quarterly_ttm" in combined else "annual operating-income proxy"
    if label == "Total Debt":
        if "yahooquery" in combined:
            return "direct debt field from yahooquery"
        if "balance" in combined:
            return "balance-sheet debt field"
        return "weak debt coverage or fallback handling" if score <= 30 else "direct debt field"
    if label == "Cash & Equivalents":
        return "balance-sheet cash field" if "balance" in combined else "low-confidence cash coverage"
    if label == "Effective Tax Rate":
        if "tax provision / pretax income" in combined:
            return "computed from Tax Provision / Pretax Income"
        if "net income / operating income" in combined:
            return "inferred from Net Income / Operating Income"
        return "default tax assumption" if score <= 20 else "reported tax input"
    if label == "Suggested WACC":
        if "observed debt cost" in combined:
            return "component-based WACC with observed debt cost"
        if "debt fallback" in combined:
            return "component-based WACC with debt-cost fallback"
        if "coe-only proxy" in combined:
            return "cost-of-equity proxy"
        return "component-based WACC estimate"

    if any(token in combined for token in ["fallback", "default", "synthetic", "proxy"]):
        return "fallback or estimated input"
    if "annual" in combined:
        return "annual proxy"
    if any(token in combined for token in ["quarterly_ttm", "ttm", "current"]):
        return "direct current or TTM value"
    return "retrieval-path based assignment"


def _build_data_quality_help_text(display_score, snapshot) -> str:
    score_value = _dcf_chat_to_float(display_score) or 0.0
    confidence_label, confidence_text = _describe_data_quality_band(score_value)

    if snapshot is None:
        return (
            f"**What this means:** {confidence_label}. {confidence_text}\n\n"
            "**How the score is assigned:** It is the simple average of nine core reliability scores. "
            "Scores come from fixed retrieval-path rules rather than a hidden continuous formula: direct current/TTM values score highest, "
            "annual proxies and inferred values score lower, fallback/default values score lower again, and missing values score 0."
        )

    component_specs = [
        ("Current Price", getattr(snapshot, "price", None)),
        ("Shares Outstanding", getattr(snapshot, "shares_outstanding", None)),
        ("TTM Revenue", getattr(snapshot, "ttm_revenue", None)),
        ("TTM Free Cash Flow", getattr(snapshot, "ttm_fcf", None)),
        ("TTM Operating Income", getattr(snapshot, "ttm_operating_income", None)),
        ("Total Debt", getattr(snapshot, "total_debt", None)),
        ("Cash & Equivalents", getattr(snapshot, "cash_and_equivalents", None)),
        ("Effective Tax Rate", getattr(snapshot, "effective_tax_rate", None)),
        ("Suggested WACC", getattr(snapshot, "suggested_wacc", None)),
    ]

    component_lines = []
    component_scores = []
    for label, metadata in component_specs:
        effective_score = _effective_data_quality_score(metadata)
        component_scores.append(effective_score)
        reason = _summarize_data_quality_reason(label, metadata, effective_score)
        component_lines.append(f"- {label}: {effective_score}/100 ({reason})")

    exact_average = sum(component_scores) / len(component_scores) if component_scores else 0.0
    score_formula = " + ".join(str(score) for score in component_scores) if component_scores else "0"

    return (
        f"**What this means:** {confidence_label}. {confidence_text}\n\n"
        "**How the exact number is assigned:** Overall Data Quality is the simple average of 9 fixed reliability scores. "
        "Missing or unscored fields count as 0. Scores are assigned by retrieval path, not by a hidden continuous formula: "
        "direct current/TTM values score highest, annual proxies and inferred values score lower, fallback/default values score lower again, "
        "and missing values score 0.\n\n"
        f"**This run:** ({score_formula}) / 9 = {exact_average:.1f}, displayed as {score_value:.0f}/100.\n"
        + "\n".join(component_lines)
        + "\n\n**Derived-metric guardrail:** TTM Free Cash Flow cannot score above the weaker of CFO and CapEx, and Suggested WACC is reliability-weighted from its component inputs.\n\n"
        "Open View DCF Details to inspect the underlying sources, periods, and notes."
    )


def _estimate_chat_quota_reset_text(error_text: str) -> str:
    """Estimate next usable time from rate-limit text; fall back to next daily reset estimate."""
    text = str(error_text or "")
    lower = text.lower()

    retry_seconds = None
    retry_patterns = [
        r"retry in\s+(\d+)\s*(?:seconds|secs|sec|s)\b",
        r"try again in\s+(\d+)\s*(?:seconds|secs|sec|s)\b",
        r"retry_delay[^\d]{0,20}(\d+)",
        r"seconds[\"']?\s*:\s*(\d+)",
    ]
    for pattern in retry_patterns:
        match = re.search(pattern, lower)
        if match:
            try:
                retry_seconds = int(match.group(1))
                break
            except Exception:
                pass

    if retry_seconds is not None and retry_seconds > 0:
        retry_dt_utc = datetime.now(timezone.utc) + timedelta(seconds=retry_seconds)
        retry_local = retry_dt_utc.astimezone()
        return retry_local.strftime("%b %d, %Y %I:%M %p %Z")

    try:
        pt = ZoneInfo("America/Los_Angeles")
        now_pt = datetime.now(pt)
        next_reset_pt = (now_pt + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return next_reset_pt.strftime("%b %d, %Y %I:%M %p PT (estimated)")
    except Exception:
        return "next daily quota reset (estimated)"


def _format_dcf_chat_runtime_message(raw_text: str) -> str:
    """Create user-facing assistant status when API/chat call fails."""
    message = str(raw_text or "").strip()
    lower = message.lower()
    if lower.startswith("chat error:"):
        message = message.split(":", 1)[1].strip()
        lower = message.lower()
    elif lower.startswith("error:"):
        message = message.split(":", 1)[1].strip()
        lower = message.lower()

    quota_markers = [
        "resource_exhausted",
        "quota",
        "rate limit",
        "too many requests",
        "429",
    ]
    if any(marker in lower for marker in quota_markers):
        reset_text = _estimate_chat_quota_reset_text(message)
        return (
            "**Assistant temporarily unavailable (API request limit reached).**\n\n"
            f"**Estimated retry/reset time:** {reset_text}\n\n"
            "The DCF tables and trace are still available for manual verification."
        )

    if "api key" in lower:
        return (
            "**Assistant unavailable (API key missing or invalid).**\n\n"
            "Set `GEMINI_API_KEY` and rerun."
        )

    return (
        "**Assistant unavailable right now.**\n\n"
        "Retry in a moment. If the issue persists, verify Gemini configuration and inspect server-side logs.\n\n"
        "You can still inspect the DCF details and source tables manually."
    )


def _format_dcf_trace_chat_response(raw_text: str, ticker_symbol: str = "UNKNOWN") -> str:
    """Normalize model output into a consistent, readable structure."""

    def _normalize_credibility_score(value):
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            raw_score = float(value)
        else:
            match = re.search(r"-?\d+(?:\.\d+)?", str(value))
            if not match:
                return None
            raw_score = float(match.group(0))
        if 0 <= raw_score <= 1:
            raw_score *= 100
        return max(0, min(100, int(round(raw_score))))

    def _yahoo_quote_url(section: str = "") -> str:
        ticker_clean = _normalize_ticker(ticker_symbol) or "UNKNOWN"
        if ticker_clean == "UNKNOWN":
            return ""
        base = f"https://finance.yahoo.com/quote/{quote(ticker_clean)}"
        if section:
            return f"{base}/{section.lstrip('/')}"
        return base

    def _infer_source_name(source_name: str, source_detail: str, formula_text: str) -> str:
        combined = f"{source_name or ''} {source_detail or ''} {formula_text or ''}".lower()
        if not combined.strip():
            return "Not explicitly provided"

        if any(token in combined for token in ["yahoo", "yfinance", "yahooquery", "yf.ticker"]):
            if any(token in combined for token in ["income statement", "balance sheet", "cash flow", "quarterly", "ttm"]):
                return "Yahoo Finance (company financial statements)"
            if any(token in combined for token in ["analyst", "consensus", "estimate"]):
                return "Yahoo Finance (analyst consensus)"
            return "Yahoo Finance"
        if any(token in combined for token in ["sec", "10-k", "10-q", "filing"]):
            return "SEC company filing"
        if any(token in combined for token in ["reuters", "bloomberg", "wall street journal", "wsj"]):
            return "External publication source"
        if any(token in combined for token in ["fred", "dgs10", "^tnx", "treasury", "risk-free"]):
            return "Treasury/rates market feed"
        if any(token in combined for token in ["damodaran", "equity risk premium", "erp"]):
            return "Damodaran dataset"
        if any(token in combined for token in ["derived", "calculated", "formula", "pv(", "discount", "wacc"]):
            return "DCF model-derived value"
        return "Model context packet source"

    def _infer_source_url(source_name: str, source_detail: str, formula_text: str) -> str:
        combined = f"{source_name or ''} {source_detail or ''} {formula_text or ''}".lower()
        if any(token in combined for token in ["tax provision", "pretax income", "income statement", "financials"]):
            return _yahoo_quote_url("financials")
        if any(token in combined for token in ["balance sheet", "total debt", "cash & equivalents", "cash and equivalents", "net debt"]):
            return _yahoo_quote_url("balance-sheet")
        if any(token in combined for token in ["cash flow", "operating cash flow", "capex", "free cash flow"]):
            return _yahoo_quote_url("cash-flow")
        if any(token in combined for token in ["analyst", "consensus", "estimate"]):
            return _yahoo_quote_url("analysis")
        if any(token in combined for token in ["price", "market cap", "shares", "beta", "quote"]):
            return _yahoo_quote_url("")
        if any(token in combined for token in ["^tnx", "risk-free", "treasury", "dgs10"]):
            return "https://finance.yahoo.com/quote/%5ETNX"
        if any(token in combined for token in ["damodaran", "equity risk premium", "erp"]):
            return "https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/histimpl.html"
        return ""

    def _infer_credibility(source_name: str, source_detail: str, formula_text: str):
        combined = f"{source_name or ''} {source_detail or ''} {formula_text or ''}".lower()
        if any(token in combined for token in ["sec", "10-k", "10-q", "filing"]):
            return 95, "Company filing data is primary-source and highly reliable."
        if any(token in combined for token in ["income statement", "balance sheet", "cash flow", "ttm", "quarterly"]):
            return 90, "Company-reported statement data is high reliability; minor transformation risk remains."
        if any(token in combined for token in ["yahoo", "consensus", "analyst", "estimate"]):
            return 80, "Consensus/aggregator data is reliable but can drift with provider updates."
        if any(token in combined for token in ["reuters", "bloomberg", "wall street journal", "wsj"]):
            return 78, "Reputable publication data is generally reliable but can include interpretation."
        if any(token in combined for token in ["derived", "calculated", "formula", "wacc", "pv(", "enterprise value"]):
            return 72, "Model-derived value depends on assumptions and input quality."
        if any(token in combined for token in ["fallback", "default", "missing"]):
            return 55, "Fallback/default inputs reduce confidence versus directly observed data."
        return 65, "Moderate confidence because exact upstream field-level provenance was not explicit."

    if not isinstance(raw_text, str) or not raw_text.strip():
        return (
            "**Answer:** Unable to generate a structured response.\n\n"
            "**Value:** N/A\n\n"
            "**Formula Path:** N/A\n\n"
            "**Source Name:** Not explicitly provided\n\n"
            "**Verification Link:** N/A (no external source link available)\n\n"
            "**Source Detail:** Not explicitly provided\n\n"
            "**Credibility:** 65/100 - Moderate confidence because exact upstream field-level provenance was not explicit."
        )

    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

    parsed = None
    try:
        parsed = json.loads(cleaned)
    except Exception:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(cleaned[start:end + 1])
            except Exception:
                parsed = None

    if isinstance(parsed, dict):
        direct_answer = str(parsed.get("direct_answer") or parsed.get("answer") or "").strip()
        value_text = str(parsed.get("value") or "").strip()
        formula_text = str(parsed.get("formula_path") or parsed.get("formula") or "").strip()
        source_name = str(parsed.get("source_name") or parsed.get("source") or "").strip()
        source_detail = str(
            parsed.get("source_detail")
            or parsed.get("source_explanation")
            or parsed.get("source_origin")
            or ""
        ).strip()
        source_url = str(
            parsed.get("source_url")
            or parsed.get("verification_url")
            or parsed.get("url")
            or ""
        ).strip()
        credibility_score = _normalize_credibility_score(
            parsed.get("credibility_score", parsed.get("reliability_score"))
        )
        credibility_explanation = str(
            parsed.get("credibility_explanation")
            or parsed.get("reliability_explanation")
            or parsed.get("credibility_reason")
            or ""
        ).strip()
    else:
        lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
        direct_answer = lines[0] if lines else cleaned

        value_match = re.search(r"(?i)\bValue:\s*(.+)", cleaned)
        formula_match = re.search(r"(?i)\bFormula(?:\s*path)?:\s*(.+)", cleaned)
        source_name_match = re.search(r"(?i)\bSource\s*name:\s*(.+)", cleaned)
        if not source_name_match:
            source_name_match = re.search(r"(?i)\bSource:\s*(.+)", cleaned)
        source_detail_match = re.search(r"(?i)\bSource\s*(?:detail|explanation|origin|notes?):\s*(.+)", cleaned)
        source_url_match = re.search(r"(?i)\b(?:Source\s*url|Verification\s*link|URL):\s*(.+)", cleaned)
        credibility_score_match = re.search(r"(?i)\b(?:Credibility|Reliability)(?:\s*score)?:\s*(.+)", cleaned)
        credibility_explanation_match = re.search(
            r"(?i)\b(?:Credibility|Reliability)\s*(?:explanation|reason|notes?):\s*(.+)",
            cleaned,
        )

        value_text = value_match.group(1).strip() if value_match else ""
        formula_text = formula_match.group(1).strip() if formula_match else ""
        source_name = source_name_match.group(1).strip() if source_name_match else ""
        source_detail = source_detail_match.group(1).strip() if source_detail_match else ""
        source_url = source_url_match.group(1).strip() if source_url_match else ""
        credibility_score = _normalize_credibility_score(
            credibility_score_match.group(1).strip() if credibility_score_match else None
        )
        credibility_explanation = (
            credibility_explanation_match.group(1).strip() if credibility_explanation_match else ""
        )

    if not direct_answer:
        direct_answer = "Direct answer not provided."
    if not value_text:
        value_text = "Not explicitly provided."
    if not formula_text:
        formula_text = "Not explicitly provided."
    if not source_name:
        source_name = _infer_source_name(source_name, source_detail, formula_text)
    source_url = _normalize_url_candidate(source_url)
    if not source_url:
        source_url = _infer_source_url(source_name, source_detail, formula_text)
    if not source_detail:
        source_detail = "Exact metric field/period was not explicitly provided."

    inferred_score, inferred_expl = _infer_credibility(source_name, source_detail, formula_text)
    if credibility_score is None:
        credibility_score = inferred_score
    if not credibility_explanation:
        credibility_explanation = inferred_expl

    source_link_md = (
        f"[Open original source]({source_url})"
        if source_url else
        "N/A (model-derived or no external publisher link)"
    )

    return (
        f"**Answer:** {direct_answer}\n\n"
        f"**Value:** {value_text}\n\n"
        f"**Formula Path:** {formula_text}\n\n"
        f"**Source Name:** {source_name}\n\n"
        f"**Verification Link:** {source_link_md}\n\n"
        f"**Source Detail:** {source_detail}\n\n"
        f"**Credibility:** {credibility_score}/100 - {credibility_explanation}"
    )


def _humanize_dcf_chat_key(label) -> str:
    text = str(label or "").strip()
    if not text:
        return "Value"
    text = text.replace("_", " ")
    text = re.sub(r"(?i)\byear\s+(\d+)\b", r"Year \1", text)
    text = re.sub(r"(?i)\bttm\b", "TTM", text)
    text = re.sub(r"(?i)\bfcff\b", "FCFF", text)
    text = re.sub(r"(?i)\bfcf\b", "FCF", text)
    text = re.sub(r"(?i)\bebitda\b", "EBITDA", text)
    text = re.sub(r"(?i)\bwacc\b", "WACC", text)
    text = re.sub(r"\s+", " ", text).strip()
    if text.islower():
        text = text.title()
    return text


def _extract_dcf_chat_key_numbers(raw_value) -> list[str]:
    def _from_mapping(mapping) -> list[str]:
        rows = []
        for key, value in mapping.items():
            label = _humanize_dcf_chat_key(key)
            rows.append(f"{label}: {value}")
        return rows

    if raw_value is None:
        return []
    if isinstance(raw_value, list):
        rows = []
        for item in raw_value:
            if isinstance(item, dict):
                rows.extend(_from_mapping(item))
            else:
                cleaned = str(item).strip()
                if cleaned:
                    rows.append(cleaned)
        return rows[:6]
    if isinstance(raw_value, dict):
        return _from_mapping(raw_value)[:6]

    text = str(raw_value or "").strip()
    if not text or text.lower() in {"n/a", "not explicitly provided."}:
        return []

    parsed = None
    if text.startswith("{") or text.startswith("["):
        try:
            parsed = json.loads(text)
        except Exception:
            try:
                parsed = ast.literal_eval(text)
            except Exception:
                parsed = None
    if parsed is not None:
        return _extract_dcf_chat_key_numbers(parsed)

    if ";" in text:
        rows = [part.strip() for part in text.split(";") if part.strip()]
        return rows[:6]

    if "\n" in text:
        rows = [part.strip("- ").strip() for part in text.splitlines() if part.strip()]
        return rows[:6]

    return [text]


def _infer_dcf_chat_response_mode(question_text: str) -> str:
    text = str(question_text or "").strip().lower()
    if not text:
        return "reasoning"

    source_markers = [
        "source",
        "where did",
        "where does",
        "where is",
        "where from",
        "which source",
        "citation",
        "cite",
        "link",
        "url",
        "what filing",
    ]
    reasoning_markers = [
        "why",
        "how",
        "explain",
        "reasoning",
        "rationale",
        "walk me through",
        "talk me through",
        "logic",
        "derive",
        "derived",
        "what drives",
        "how come",
    ]
    value_starters = [
        "what is",
        "what's",
        "how much",
        "give me",
        "show me",
        "what are",
        "tell me",
    ]

    if any(marker in text for marker in source_markers):
        return "source"
    if any(marker in text for marker in reasoning_markers):
        return "reasoning"
    if any(text.startswith(starter) for starter in value_starters):
        return "value"
    return "reasoning"


def _normalize_dcf_trace_chat_payload(raw_text: str, ticker_symbol: str = "UNKNOWN", question_text: str = "") -> dict:
    formatted_fallback = _format_dcf_trace_chat_response(raw_text, ticker_symbol)

    def _normalize_credibility_score(value):
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            raw_score = float(value)
        else:
            match = re.search(r"-?\d+(?:\.\d+)?", str(value))
            if not match:
                return None
            raw_score = float(match.group(0))
        if 0 <= raw_score <= 1:
            raw_score *= 100
        return max(0, min(100, int(round(raw_score))))

    def _yahoo_quote_url(section: str = "") -> str:
        ticker_clean = _normalize_ticker(ticker_symbol) or "UNKNOWN"
        if ticker_clean == "UNKNOWN":
            return ""
        base = f"https://finance.yahoo.com/quote/{quote(ticker_clean)}"
        if section:
            return f"{base}/{section.lstrip('/')}"
        return base

    def _infer_source_name(source_name: str, source_detail: str, calculation_summary: str) -> str:
        combined = f"{source_name or ''} {source_detail or ''} {calculation_summary or ''}".lower()
        if not combined.strip():
            return "Not explicitly provided"
        if any(token in combined for token in ["yahoo", "yfinance", "yahooquery", "yf.ticker"]):
            if any(token in combined for token in ["income statement", "balance sheet", "cash flow", "quarterly", "ttm"]):
                return "Yahoo Finance financial statements"
            if any(token in combined for token in ["analyst", "consensus", "estimate"]):
                return "Yahoo Finance analyst consensus"
            return "Yahoo Finance"
        if any(token in combined for token in ["sec", "10-k", "10-q", "filing"]):
            return "SEC filing"
        if any(token in combined for token in ["damodaran", "equity risk premium", "erp"]):
            return "Damodaran dataset"
        if any(token in combined for token in ["treasury", "^tnx", "risk-free", "dgs10"]):
            return "Treasury market data"
        if any(token in combined for token in ["projection", "pv(", "terminal value", "enterprise value", "wacc"]):
            return "DCF model calculation"
        return "DCF run context"

    def _infer_source_url(source_name: str, source_detail: str, calculation_summary: str) -> str:
        combined = f"{source_name or ''} {source_detail or ''} {calculation_summary or ''}".lower()
        if any(token in combined for token in ["tax provision", "pretax income", "income statement", "financials"]):
            return _yahoo_quote_url("financials")
        if any(token in combined for token in ["balance sheet", "total debt", "cash & equivalents", "cash and equivalents", "net debt"]):
            return _yahoo_quote_url("balance-sheet")
        if any(token in combined for token in ["cash flow", "operating cash flow", "capex", "free cash flow"]):
            return _yahoo_quote_url("cash-flow")
        if any(token in combined for token in ["analyst", "consensus", "estimate"]):
            return _yahoo_quote_url("analysis")
        if any(token in combined for token in ["price", "market cap", "shares", "beta", "quote"]):
            return _yahoo_quote_url("")
        if any(token in combined for token in ["^tnx", "risk-free", "treasury", "dgs10"]):
            return "https://finance.yahoo.com/quote/%5ETNX"
        if any(token in combined for token in ["damodaran", "equity risk premium", "erp"]):
            return "https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/histimpl.html"
        return ""

    def _infer_credibility(source_name: str, source_detail: str, calculation_summary: str):
        combined = f"{source_name or ''} {source_detail or ''} {calculation_summary or ''}".lower()
        if any(token in combined for token in ["sec", "10-k", "10-q", "filing"]):
            return 95, "Primary company filing data is highly reliable."
        if any(token in combined for token in ["income statement", "balance sheet", "cash flow", "quarterly", "ttm"]):
            return 90, "Company-reported statement data is strong, with only light transformation risk."
        if any(token in combined for token in ["analyst", "consensus", "estimate", "damodaran", "treasury"]):
            return 80, "The answer uses reputable market or consensus inputs, but those can still move over time."
        if any(token in combined for token in ["projection", "model", "formula", "pv(", "terminal value", "wacc"]):
            return 72, "This is model-derived, so confidence depends on the input assumptions and trace."
        if any(token in combined for token in ["fallback", "default", "missing"]):
            return 55, "Fallback or missing-data paths lower confidence versus directly observed inputs."
        return 65, "Moderate confidence because the exact upstream provenance was not fully explicit."

    cleaned = str(raw_text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    parsed = None
    try:
        parsed = json.loads(cleaned)
    except Exception:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(cleaned[start:end + 1])
            except Exception:
                parsed = None

    if isinstance(parsed, dict):
        response_mode = str(parsed.get("response_mode") or "").strip().lower()
        answer = str(
            parsed.get("answer")
            or parsed.get("direct_answer")
            or ""
        ).strip()
        key_numbers = _extract_dcf_chat_key_numbers(
            parsed.get("key_numbers", parsed.get("value"))
        )
        calculation_summary = str(
            parsed.get("calculation_summary")
            or parsed.get("formula_path")
            or parsed.get("formula")
            or ""
        ).strip()
        source_summary = str(
            parsed.get("source_summary")
            or parsed.get("source_detail")
            or parsed.get("source_explanation")
            or parsed.get("source_origin")
            or ""
        ).strip()
        source_name = str(parsed.get("source_name") or parsed.get("source") or "").strip()
        source_url = _normalize_url_candidate(
            str(parsed.get("source_url") or parsed.get("verification_url") or parsed.get("url") or "").strip()
        )
        confidence_score = _normalize_credibility_score(
            parsed.get("confidence_score", parsed.get("credibility_score", parsed.get("reliability_score")))
        )
        confidence_explanation = str(
            parsed.get("confidence_explanation")
            or parsed.get("credibility_explanation")
            or parsed.get("reliability_explanation")
            or parsed.get("credibility_reason")
            or ""
        ).strip()
    else:
        response_mode = ""
        answer_match = re.search(r"(?i)\*\*Answer:\*\*\s*(.+?)(?:\n\n|\Z)", formatted_fallback, flags=re.S)
        value_match = re.search(r"(?i)\*\*Value:\*\*\s*(.+?)(?:\n\n|\Z)", formatted_fallback, flags=re.S)
        calc_match = re.search(r"(?i)\*\*(?:Formula Path|Calculation):\*\*\s*(.+?)(?:\n\n|\Z)", formatted_fallback, flags=re.S)
        source_name_match = re.search(r"(?i)\*\*Source Name:\*\*\s*(.+?)(?:\n\n|\Z)", formatted_fallback, flags=re.S)
        source_detail_match = re.search(r"(?i)\*\*Source Detail:\*\*\s*(.+?)(?:\n\n|\Z)", formatted_fallback, flags=re.S)
        confidence_match = re.search(r"(?i)\*\*Credibility:\*\*\s*(\d+)\s*/\s*100\s*-\s*(.+?)(?:\n\n|\Z)", formatted_fallback, flags=re.S)
        link_match = re.search(r"\[Open original source\]\((https?://[^\)]+)\)", formatted_fallback)

        answer = answer_match.group(1).strip() if answer_match else cleaned
        key_numbers = _extract_dcf_chat_key_numbers(value_match.group(1).strip() if value_match else "")
        calculation_summary = calc_match.group(1).strip() if calc_match else ""
        source_name = source_name_match.group(1).strip() if source_name_match else ""
        source_summary = source_detail_match.group(1).strip() if source_detail_match else ""
        source_url = _normalize_url_candidate(link_match.group(1).strip()) if link_match else ""
        confidence_score = int(confidence_match.group(1)) if confidence_match else None
        confidence_explanation = confidence_match.group(2).strip() if confidence_match else ""

    if not answer:
        answer = "I couldn't produce a clean answer from this run."
    answer = re.sub(r"^\*\*Answer:\*\*\s*", "", answer).strip()
    if response_mode not in {"reasoning", "value", "source"}:
        response_mode = _infer_dcf_chat_response_mode(question_text)
    source_name = source_name or _infer_source_name(source_name, source_summary, calculation_summary)
    if not source_url:
        source_url = _infer_source_url(source_name, source_summary, calculation_summary)
    if not source_summary:
        source_summary = "The answer comes from the active DCF run and its underlying source tables."
    inferred_score, inferred_explanation = _infer_credibility(source_name, source_summary, calculation_summary)
    if confidence_score is None:
        confidence_score = inferred_score
    if not confidence_explanation:
        confidence_explanation = inferred_explanation

    history_text = answer
    if key_numbers:
        history_text += " Key numbers: " + "; ".join(key_numbers[:4]) + "."
    if calculation_summary:
        history_text += " Calculation basis: " + calculation_summary

    return {
        "kind": "dcf_trace_answer",
        "response_mode": response_mode,
        "answer": answer,
        "key_numbers": key_numbers[:6],
        "calculation_summary": calculation_summary,
        "source_name": source_name,
        "source_summary": source_summary,
        "source_url": source_url,
        "confidence_score": confidence_score,
        "confidence_explanation": confidence_explanation,
        "history_text": history_text.strip(),
    }


def _dcf_chat_message_to_history_text(message) -> str:
    if isinstance(message, dict):
        if message.get("history_text"):
            return str(message.get("history_text"))
        if message.get("kind") == "dcf_trace_answer":
            parts = [str(message.get("answer") or "").strip()]
            key_numbers = message.get("key_numbers") or []
            if key_numbers:
                parts.append("Key numbers: " + "; ".join(str(item) for item in key_numbers[:4]))
            if message.get("calculation_summary"):
                parts.append("Calculation basis: " + str(message.get("calculation_summary")))
            return " ".join(part for part in parts if part).strip()
        return str(message.get("content") or "").strip()
    return str(message or "").strip()


def _render_dcf_chat_message(role: str, content) -> None:
    with st.chat_message(role):
        if role != "assistant" or not isinstance(content, dict) or content.get("kind") != "dcf_trace_answer":
            st.markdown(str(content or ""))
            return

        response_mode = str(content.get("response_mode") or "reasoning").strip().lower()
        answer = str(content.get("answer") or "").strip()
        if answer:
            st.markdown(answer)

        key_numbers = [str(item).strip() for item in (content.get("key_numbers") or []) if str(item).strip()]
        if key_numbers and response_mode == "reasoning":
            st.markdown("\n".join(f"- {item}" for item in key_numbers))

        calculation_summary = str(content.get("calculation_summary") or "").strip()
        source_name = str(content.get("source_name") or "").strip()
        source_summary = str(content.get("source_summary") or "").strip()
        source_url = _normalize_url_candidate(str(content.get("source_url") or "").strip())
        confidence_score = content.get("confidence_score")
        confidence_explanation = str(content.get("confidence_explanation") or "").strip()

        show_details = response_mode == "reasoning"
        has_details = show_details and any([
            calculation_summary,
            source_name,
            source_summary,
            source_url,
            confidence_score is not None,
            confidence_explanation,
        ])
        if not has_details:
            return

        with st.expander("How this answer was derived", expanded=False):
            if calculation_summary:
                st.markdown(f"**Calculation logic**\n\n{calculation_summary}")

            source_lines = []
            if source_name:
                source_lines.append(f"Primary source: {source_name}")
            if source_summary:
                source_lines.append(source_summary)
            if source_url:
                source_lines.append(f"[Open source]({source_url})")
            if source_lines:
                st.markdown("**Source basis**\n\n" + "\n\n".join(source_lines))

            if confidence_score is not None or confidence_explanation:
                score_text = f"{int(confidence_score)}/100" if confidence_score is not None else "N/A"
                detail = f"{score_text} - {confidence_explanation}".strip(" -")
                st.markdown(f"**Confidence**\n\n{detail}")


def _build_dcf_trace_verifiable_sources(ticker_symbol: str, input_table: list[dict], snapshot, assumptions: dict) -> list[dict]:
    ticker_clean = _normalize_ticker(ticker_symbol) or "UNKNOWN"
    yahoo_quote = f"https://finance.yahoo.com/quote/{quote(ticker_clean)}" if ticker_clean != "UNKNOWN" else ""

    urls = {
        "quote": yahoo_quote,
        "analysis": f"{yahoo_quote}/analysis" if yahoo_quote else "",
        "financials": f"{yahoo_quote}/financials" if yahoo_quote else "",
        "balance_sheet": f"{yahoo_quote}/balance-sheet" if yahoo_quote else "",
        "cash_flow": f"{yahoo_quote}/cash-flow" if yahoo_quote else "",
        "key_statistics": f"{yahoo_quote}/key-statistics" if yahoo_quote else "",
        "tnx": "https://finance.yahoo.com/quote/%5ETNX",
        "treasury": "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/textview?type=daily_treasury_yield_curve",
        "damodaran_erp": "https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/histimpl.html",
    }

    input_source_map = {
        "Current Price": ("Yahoo Finance Quote", urls["quote"]),
        "Market Cap": ("Yahoo Finance Quote", urls["quote"]),
        "Shares Outstanding": ("Yahoo Finance Key Statistics", urls["key_statistics"]),
        "TTM Revenue": ("Yahoo Finance Financials", urls["financials"]),
        "TTM Operating Income": ("Yahoo Finance Financials", urls["financials"]),
        "TTM EBITDA": ("Yahoo Finance Financials", urls["financials"]),
        "TTM Operating Cash Flow": ("Yahoo Finance Cash Flow", urls["cash_flow"]),
        "TTM CapEx": ("Yahoo Finance Cash Flow", urls["cash_flow"]),
        "TTM Free Cash Flow": ("Yahoo Finance Cash Flow", urls["cash_flow"]),
        "Total Debt": ("Yahoo Finance Balance Sheet", urls["balance_sheet"]),
        "Cash & Equivalents": ("Yahoo Finance Balance Sheet", urls["balance_sheet"]),
    }

    registry = []
    seen = set()

    def _register(
        *,
        metric: str,
        value_text: str,
        source_name: str,
        source_url: str,
        source_detail: str,
        reliability_score=None,
        model_field: str = "",
    ) -> None:
        metric_clean = str(metric or "").strip()
        source_name_clean = str(source_name or "").strip() or "Not explicitly provided"
        source_url_clean = _normalize_url_candidate(source_url)
        source_detail_clean = str(source_detail or "").strip() or "Not explicitly provided."
        value_clean = str(value_text or "N/A").strip() or "N/A"
        key = (
            metric_clean.lower(),
            source_name_clean.lower(),
            source_url_clean or "",
            source_detail_clean.lower(),
        )
        if key in seen:
            return
        seen.add(key)

        entry = {
            "metric": metric_clean,
            "value": value_clean,
            "source_name": source_name_clean,
            "source_url": source_url_clean or "",
            "source_detail": source_detail_clean,
        }
        if model_field:
            entry["model_field"] = model_field
        score = _dcf_chat_to_float(reliability_score)
        if score is not None:
            entry["reliability_score"] = max(0, min(100, int(round(score))))
        registry.append(entry)

    for row in input_table:
        item = str(row.get("Item", "")).strip()
        if not item:
            continue
        source_raw = str(row.get("Source", "")).strip()
        period_raw = str(row.get("Period", "")).strip()
        notes_raw = str(row.get("Notes", "")).strip()
        reliability = _parse_reliability_label(row.get("Reliability"))
        source_name, source_url = input_source_map.get(item, ("Yahoo Finance", urls["quote"]))
        combined = f"{source_raw} {notes_raw}".lower()
        if any(token in combined for token in ["default", "fallback", "unavailable", "proxy"]):
            source_name = "Model fallback / derived estimate"
            source_url = ""

        detail_parts = []
        if source_raw and source_raw not in {"N/A", "—"}:
            detail_parts.append(f"Loaded from {source_raw}")
        if period_raw and period_raw not in {"N/A", "—"}:
            detail_parts.append(f"Period: {period_raw}")
        if notes_raw and notes_raw not in {"N/A", "—"}:
            detail_parts.append(notes_raw)
        detail_text = " | ".join(detail_parts) if detail_parts else "Input used in this DCF run."

        _register(
            metric=f"Input: {item}",
            value_text=str(row.get("Value", "N/A")),
            source_name=source_name,
            source_url=source_url,
            source_detail=detail_text,
            reliability_score=reliability,
            model_field=f"inputs_table[{item}]",
        )

    effective_tax_meta = getattr(snapshot, "effective_tax_rate", None) if snapshot else None
    effective_tax_source = str(getattr(effective_tax_meta, "source_path", "") or "").strip()
    effective_tax_notes = str(getattr(effective_tax_meta, "notes", "") or "").strip()
    effective_tax_period = str(getattr(effective_tax_meta, "period_end", "") or "").strip()
    effective_tax_score = getattr(effective_tax_meta, "reliability_score", None) if effective_tax_meta else None
    effective_tax_value = _dcf_chat_to_float(getattr(effective_tax_meta, "value", None)) if effective_tax_meta else None
    assumed_tax_value = _dcf_chat_to_float(assumptions.get("tax_rate"))

    if assumed_tax_value is not None:
        tax_name = "Yahoo Finance Income Statement (Tax Provision / Pretax Income)"
        tax_url = urls["financials"]
        if "net income" in effective_tax_source.lower() and "operating income" in effective_tax_source.lower():
            tax_name = "Yahoo Finance Income Statement (derived NI/OI tax proxy)"
        if not effective_tax_source:
            tax_name = "DCF model assumption (source not captured)"
            tax_url = ""
        if effective_tax_value is not None and abs(assumed_tax_value - effective_tax_value) > 1e-6:
            tax_name = "User-adjusted DCF assumption (Tax Rate)"
            tax_url = ""

        tax_detail_parts = []
        if effective_tax_source:
            tax_detail_parts.append(f"Upstream calculation: {effective_tax_source}")
        if effective_tax_period:
            tax_detail_parts.append(f"Period: {effective_tax_period}")
        if effective_tax_notes:
            tax_detail_parts.append(effective_tax_notes)
        if effective_tax_value is not None and abs(assumed_tax_value - effective_tax_value) > 1e-6:
            tax_detail_parts.append(
                f"DCF run uses user-set {assumed_tax_value*100:.2f}% vs snapshot {effective_tax_value*100:.2f}%."
            )
        tax_detail = " | ".join(tax_detail_parts) if tax_detail_parts else "Tax assumption used directly in NOPAT/FCFF."

        _register(
            metric="Assumption: Tax Rate",
            value_text=_dcf_chat_fmt_rate(assumed_tax_value),
            source_name=tax_name,
            source_url=tax_url,
            source_detail=tax_detail,
            reliability_score=effective_tax_score,
            model_field="assumptions.tax_rate",
        )

    wacc_components_local = getattr(snapshot, "wacc_components", {}) if snapshot else {}
    if isinstance(wacc_components_local, dict) and wacc_components_local:
        rf_source = str(wacc_components_local.get("rf_source", "") or "").strip()
        rf_name = "Yahoo Finance ^TNX (10Y Treasury)"
        rf_url = urls["tnx"]
        if rf_source and "^tnx" not in rf_source.lower():
            rf_name = "U.S. Treasury 10Y Yield"
            rf_url = urls["treasury"]

        _register(
            metric="WACC Component: Risk-Free Rate",
            value_text=_dcf_chat_fmt_rate(wacc_components_local.get("risk_free_rate")),
            source_name=rf_name,
            source_url=rf_url,
            source_detail=rf_source or "Risk-free input used in CAPM.",
            model_field="wacc_components.risk_free_rate",
        )
        _register(
            metric="WACC Component: Equity Risk Premium",
            value_text=_dcf_chat_fmt_rate(wacc_components_local.get("market_risk_premium")),
            source_name="Damodaran Implied ERP Dataset",
            source_url=urls["damodaran_erp"],
            source_detail=f"Damodaran date: {wacc_components_local.get('damodaran_date') or 'n/a'}",
            model_field="wacc_components.market_risk_premium",
        )
        _register(
            metric="WACC Component: Beta",
            value_text=_dcf_chat_fmt_number(wacc_components_local.get("beta"), 2),
            source_name="Yahoo Finance Beta",
            source_url=urls["quote"],
            source_detail=str(wacc_components_local.get("cost_of_equity_source") or "CAPM beta input."),
            model_field="wacc_components.beta",
        )

    return registry


def _build_dcf_chat_context_packet(ui_adapter, ui_data, engine_result, snapshot, ticker_symbol: str) -> dict:
    assumptions = ui_data.get("assumptions", {}) or {}
    trace_steps = engine_result.get("trace", []) if isinstance(engine_result, dict) else []

    input_order = {
        "Current Price": 1,
        "Market Cap": 2,
        "Shares Outstanding": 3,
        "TTM Revenue": 4,
        "TTM Operating Income": 5,
        "TTM EBITDA": 6,
        "TTM Operating Cash Flow": 7,
        "TTM CapEx": 8,
        "TTM Free Cash Flow": 9,
        "Total Debt": 10,
        "Cash & Equivalents": 11,
    }
    input_rows_raw = ui_adapter.format_input_table() if ui_adapter else []
    input_rows = sorted(input_rows_raw, key=lambda row: input_order.get(row.get("Item", ""), 999))
    input_table = [
        {
            "Item": row.get("Item", "N/A"),
            "Value": row.get("Value", "N/A"),
            "Period": row.get("Period", "N/A"),
            "Source": row.get("Source", "N/A"),
            "Reliability": row.get("Reliability", "N/A"),
            "Notes": row.get("Notes", "N/A"),
        }
        for row in input_rows
    ]

    assumption_rows = ui_adapter.format_assumptions_table() if ui_adapter else []
    projection_rows = list((assumptions.get("yearly_projections", []) or ui_data.get("fcf_projections", []) or []))[:40]
    bridge_rows = ui_adapter.format_bridge_table() if ui_adapter else []

    growth_summary_rows = [
        {"Parameter": "Near-Term Growth", "Value": _dcf_chat_fmt_rate(assumptions.get("near_term_growth_rate"))},
        {"Parameter": "Effective Near-Term Growth", "Value": _dcf_chat_fmt_rate(assumptions.get("effective_near_term_growth_rate"))},
        {"Parameter": "Analyst Long-Term Growth Anchor", "Value": _dcf_chat_fmt_rate(assumptions.get("analyst_long_term_growth_rate"))},
        {"Parameter": "Terminal Growth", "Value": _dcf_chat_fmt_rate(assumptions.get("stable_growth_rate"))},
        {"Parameter": "Current ROIC", "Value": _dcf_chat_fmt_rate(assumptions.get("base_roic"))},
        {"Parameter": "Terminal ROIC", "Value": _dcf_chat_fmt_rate(assumptions.get("terminal_roic"))},
        {"Parameter": "Terminal Reinvestment Rate", "Value": _dcf_chat_fmt_rate(assumptions.get("terminal_reinvestment_rate"))},
    ]

    ev = ui_data.get("enterprise_value")
    equity = ui_data.get("equity_value")
    price_per_share = ui_data.get("price_per_share")
    pv_fcf_sum = ui_data.get("pv_fcf_sum")
    pv_tv = ui_data.get("pv_terminal_value")
    terminal_value_yearN = ui_data.get("terminal_value_yearN")
    net_debt = ui_data.get("net_debt")
    shares_outstanding = ui_data.get("shares_outstanding")

    calc_ev = (_dcf_chat_to_float(pv_fcf_sum) or 0.0) + (_dcf_chat_to_float(pv_tv) or 0.0)
    calc_equity = (_dcf_chat_to_float(ev) or 0.0) - (_dcf_chat_to_float(net_debt) or 0.0)
    calc_price = calc_equity / _dcf_chat_to_float(shares_outstanding) if _dcf_chat_to_float(shares_outstanding) else None

    reconciliation_rows = [
        {
            "Check": "EV = PV(FCFF) + PV(TV)",
            "Computed": calc_ev,
            "Reported": ev,
        },
        {
            "Check": "Equity = EV - Net Debt",
            "Computed": calc_equity,
            "Reported": equity,
        },
        {
            "Check": "Price = Equity / Shares",
            "Computed": calc_price,
            "Reported": price_per_share,
        },
    ]

    terminal_rows = [
        {
            "Terminal Method": assumptions.get("terminal_value_method"),
            "Terminal Growth Rate": assumptions.get("terminal_growth_rate"),
            "Exit Multiple": assumptions.get("exit_multiple"),
            "Terminal Value (Year N)": terminal_value_yearN,
            "PV(Terminal Value)": pv_tv,
        }
    ]

    verifiable_sources = _build_dcf_trace_verifiable_sources(
        ticker_symbol,
        input_table,
        snapshot,
        assumptions,
    )
    logic_reference = {
        "valuation_identities": [
            "Enterprise Value = PV of explicit FCFF + PV of terminal value.",
            "Equity Value = Enterprise Value - Net Debt.",
            "Intrinsic Value per Share = Equity Value / Shares Outstanding.",
        ],
        "projection_logic": [
            "Revenue projections grow forward from the current base using the year-by-year growth schedule in this run.",
            "Driver-model FCFF follows Revenue -> EBIT -> NOPAT -> Reinvestment -> FCFF.",
            "Terminal value uses either Gordon Growth or Exit Multiple, depending on the selected method for this run.",
        ],
        "data_quality_logic": [
            "Overall Data Quality is the simple average of 9 core reliability scores: Current Price, Shares Outstanding, TTM Revenue, TTM Free Cash Flow, TTM Operating Income, Total Debt, Cash & Equivalents, Effective Tax Rate, and Suggested WACC.",
            "Missing or unscored fields contribute 0 to the overall average.",
            "TTM Free Cash Flow reliability is capped by the weaker of Operating Cash Flow and CapEx reliability.",
            "Suggested WACC reliability is a weighted component score and drops when the model relies on debt-cost fallbacks or proxy logic.",
        ],
    }

    return {
        "ticker": ticker_symbol,
        "company_name": getattr(snapshot, "company_name", None) if snapshot else None,
        "inputs_table": input_table,
        "assumptions_table": assumption_rows,
        "verifiable_sources": verifiable_sources,
        "growth_summary": growth_summary_rows,
        "results": {
            "enterprise_value": ev,
            "equity_value": equity,
            "price_per_share": price_per_share,
            "pv_fcf_sum": pv_fcf_sum,
            "pv_terminal_value": pv_tv,
            "terminal_value_yearN": terminal_value_yearN,
            "net_debt": net_debt,
            "shares_outstanding": shares_outstanding,
            "implied_fcf_yield": ui_data.get("implied_fcf_yield"),
            "implied_ev_fcf": ui_data.get("implied_ev_fcf"),
            "discount_rate_used": assumptions.get("discount_rate_used"),
            "wacc_input": assumptions.get("wacc"),
            "terminal_growth_rate": assumptions.get("terminal_growth_rate"),
        },
        "wacc_components": getattr(snapshot, "wacc_components", {}) if snapshot else {},
        "projection_rows": projection_rows,
        "terminal_rows": terminal_rows,
        "bridge_rows": bridge_rows,
        "reconciliation_rows": reconciliation_rows,
        "logic_reference": logic_reference,
        "trace_steps": trace_steps[:120],
    }


def _render_dcf_trace_chatbot(ui_adapter, ui_data, engine_result, snapshot, *, location_key: str) -> None:
    if not ui_adapter or not isinstance(ui_data, dict) or not ui_data.get("success"):
        return

    st.markdown(
        '<div class="qa-chatbot-hero"><span class="qa-chatbot-pill">Q&A Chatbot</span></div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Ask where any number came from. Answers are grounded in this run's inputs, assumptions, "
        "WACC components, projections, and trace steps."
    )

    chat_ticker = _normalize_ticker(
        st.session_state.get("ticker") or getattr(snapshot, "ticker", "UNKNOWN")
    ) or "UNKNOWN"
    chat_history_key = f"{location_key}_chat_history_{chat_ticker}"
    chat_input_key = f"{location_key}_chat_input_{chat_ticker}"

    if chat_history_key not in st.session_state or not isinstance(st.session_state[chat_history_key], list):
        st.session_state[chat_history_key] = []

    chat_history = st.session_state[chat_history_key]

    _, col_chat_clear = st.columns([4, 1])
    with col_chat_clear:
        if st.button("Clear Q&A", key=f"clear_{chat_history_key}"):
            st.session_state[chat_history_key] = []
            st.rerun()

    for msg in chat_history[-10:]:
        role = msg.get("role", "assistant")
        content = msg.get("content", "")
        _render_dcf_chat_message(role, content)

    user_question = ""
    submitted = False
    with st.form(key=f"{location_key}_chat_form_{chat_ticker}", clear_on_submit=True):
        user_question = st.text_input(
            "Ask a clarification question",
            placeholder="Example: Why is PV(TV) 47.8% of EV? Where does Rd come from?",
            autocomplete="off",
            key=chat_input_key,
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Ask")

    if submitted and user_question:
        response_mode = _infer_dcf_chat_response_mode(user_question)
        context_packet = _build_dcf_chat_context_packet(
            ui_adapter,
            ui_data,
            engine_result or {},
            snapshot,
            chat_ticker,
        )
        context_data = json.dumps(context_packet, indent=2, default=str)
        trace_prompt = (
            "User question: " + user_question + "\n\n"
            f"Detected response mode: {response_mode}\n\n"
            "Answer rules:\n"
            "1) Use only the provided DCF context and logic_reference. If the answer depends on a calculation, explain it from this run's formulas and numbers.\n"
            "2) Return ONLY valid JSON (no markdown outside JSON) with this exact schema:\n"
            "{\n"
            "  \"response_mode\": \"reasoning|value|source\",\n"
            "  \"answer\": \"natural answer in plain English; length should match response_mode\",\n"
            "  \"key_numbers\": [\"short bullet with a number\", \"short bullet with a number\"],\n"
            "  \"calculation_summary\": \"1-2 sentence explanation of how the answer was calculated or inferred from this run's logic\",\n"
            "  \"source_summary\": \"1 sentence naming the upstream source(s), period(s), or model step(s) used\",\n"
            "  \"source_name\": \"human-readable source label\",\n"
            "  \"source_url\": \"direct https URL to the closest external source, or empty string if the answer is purely model-derived\",\n"
            "  \"confidence_score\": 0,\n"
            "  \"confidence_explanation\": \"1 short sentence explaining the confidence score\"\n"
            "}\n"
            "3) Follow the detected response mode strictly.\n"
            "4) If response_mode is `reasoning`, answer in a fuller explanatory style and use key_numbers when helpful.\n"
            "5) If response_mode is `value`, answer briefly with just the requested value or values. Do not add extra reasoning unless needed for clarity.\n"
            "6) If response_mode is `source`, answer briefly with just the source, period, and linkable origin. Do not add extra reasoning.\n"
            "7) Write naturally. Do not output raw dictionaries, snake_case keys, or internal field/container names.\n"
            "8) Keep key_numbers concise and user-facing. Leave key_numbers empty when the question only asks for the source.\n"
            "9) Prefer source_name/source_url from verifiable_sources when available.\n"
            "10) If the answer is model-derived, say that clearly in calculation_summary and still cite the closest upstream external source_url when available.\n"
            "11) If data is missing, say that plainly.\n"
            "12) Do not provide investment advice.\n"
        )
        prior_history = []
        for msg in chat_history:
            if not isinstance(msg, dict) or msg.get("role") not in {"user", "assistant"}:
                continue
            history_text = _dcf_chat_message_to_history_text(msg.get("content"))
            if history_text:
                prior_history.append({"role": msg.get("role"), "content": history_text})

        _render_dcf_chat_message("user", user_question)

        with st.spinner("Working through the DCF logic..."):
            assistant_raw = run_chat(prior_history, trace_prompt, context_data)
            raw_lower = str(assistant_raw or "").lower()
            if raw_lower.startswith("chat error:") or raw_lower.startswith("error:"):
                assistant_reply = _format_dcf_chat_runtime_message(assistant_raw)
            else:
                assistant_reply = _normalize_dcf_trace_chat_payload(assistant_raw, chat_ticker, user_question)

        _render_dcf_chat_message("assistant", assistant_reply)

        st.session_state[chat_history_key].append({"role": "user", "content": user_question})
        st.session_state[chat_history_key].append({"role": "assistant", "content": assistant_reply})


def _show_dcf_details_page():
    """Render DCF details in a clear sequential flow."""
    st.markdown("---")
    st.subheader("DCF Calculation Details")

    if st.button("<- Back to Summary", key="back_from_details_top"):
        _queue_scroll_to_section("valuation")
        st.session_state.show_dcf_details = False
        st.rerun()

    ui_adapter = st.session_state.dcf_ui_adapter
    ui_data = ui_adapter.get_ui_data()
    engine_result = st.session_state.get("dcf_engine_result") or {}
    snapshot = st.session_state.get("dcf_snapshot")

    assumptions = ui_data.get("assumptions", {}) or {}
    yearly_projections = assumptions.get("yearly_projections", []) or []
    fcf_projections = ui_data.get("fcf_projections", []) or []
    trace_steps = engine_result.get("trace", []) if isinstance(engine_result, dict) else []

    forecast_years = int(assumptions.get("forecast_years") or 5)
    wacc_used = assumptions.get("wacc")
    terminal_growth = assumptions.get("terminal_growth_rate")
    tv_method = assumptions.get("terminal_value_method", "gordon_growth")

    ev = ui_data.get("enterprise_value")
    equity = ui_data.get("equity_value")
    price_per_share = ui_data.get("price_per_share")
    pv_fcf_sum = ui_data.get("pv_fcf_sum")
    pv_tv = ui_data.get("pv_terminal_value")
    terminal_value_yearN = ui_data.get("terminal_value_yearN")
    net_debt = ui_data.get("net_debt")
    shares_outstanding = ui_data.get("shares_outstanding")

    def _to_float(value):
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    def _fmt_money(value, decimals=2):
        num = _to_float(value)
        if num is None:
            return "N/A"
        abs_num = abs(num)
        if abs_num >= 1e12:
            return f"${num / 1e12:.{decimals}f}T"
        if abs_num >= 1e9:
            return f"${num / 1e9:.{decimals}f}B"
        if abs_num >= 1e6:
            return f"${num / 1e6:.{decimals}f}M"
        return f"${num:,.{decimals}f}"

    def _fmt_rate(value, decimals=1):
        num = _to_float(value)
        if num is None:
            return "N/A"
        return f"{num * 100:.{decimals}f}%"

    def _fmt_multiple(value, decimals=1):
        num = _to_float(value)
        if num is None:
            return "N/A"
        return f"{num:.{decimals}f}x"

    def _fmt_number(value, decimals=2):
        num = _to_float(value)
        if num is None:
            return "N/A"
        return f"{num:,.{decimals}f}"

    def _parse_reliability(value):
        if isinstance(value, str) and value.endswith("/100"):
            try:
                return int(value.split("/")[0])
            except ValueError:
                return None
        return None

    def _sanitize_notice_text(text: str) -> str:
        """Remove decorative emoji/icons and normalize spacing for professional warning copy."""
        if not isinstance(text, str):
            return str(text)
        cleaned = text.strip()
        for marker in ["⚠️", "⚠", "🔴", "🟡", "🟠", "🟢", "✅", "❌", "📊", "📈", "📉", "ℹ️", "ℹ"]:
            cleaned = cleaned.replace(marker, "")
        cleaned = re.sub(r"^[\s\-\u2022]+", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _classify_notice(text: str, source_kind: str) -> str:
        """Classify message severity for grouped display."""
        if source_kind == "error":
            return "critical"

        lower = text.lower()
        critical_tokens = [
            "fatal",
            "invalid",
            "implausible",
            "framework mixing",
            "cannot calculate",
            "terminal growth",
            "dominates",
        ]
        review_tokens = [
            "very low",
            "very high",
            "disagree",
            "sensitive",
            "fallback",
            "approximate",
            "proxy",
            "elevated",
            "defaulted",
            "missing",
            "low ",
            "high ",
        ]

        if any(token in lower for token in critical_tokens):
            return "critical"
        if any(token in lower for token in review_tokens):
            return "review"
        return "info"

    st.caption("Sequential view of inputs, assumptions, calculations, and outputs.")

    col_h1, col_h2, col_h3, col_h4 = st.columns(4)
    with col_h1:
        st.metric(
            "Intrinsic Value / Share",
            f"${_to_float(price_per_share):.2f}" if _to_float(price_per_share) is not None else "N/A",
        )
    with col_h2:
        st.metric("Enterprise Value", _fmt_money(ev))
    with col_h3:
        st.metric("Equity Value", _fmt_money(equity))
    with col_h4:
        tv_ratio = ((_to_float(pv_tv) or 0.0) / (_to_float(ev) or 1.0) * 100.0) if _to_float(ev) else 0.0
        st.metric("PV(Terminal) as % EV", f"{tv_ratio:.1f}%")

    st.markdown("### 1. Input Data")
    input_order = {
        "Current Price": 1,
        "Market Cap": 2,
        "Shares Outstanding": 3,
        "TTM Revenue": 4,
        "TTM Operating Income": 5,
        "TTM EBITDA": 6,
        "TTM Operating Cash Flow": 7,
        "TTM CapEx": 8,
        "TTM Free Cash Flow": 9,
        "Total Debt": 10,
        "Cash & Equivalents": 11,
    }

    input_rows_raw = ui_adapter.format_input_table()
    input_rows = sorted(input_rows_raw, key=lambda row: input_order.get(row.get("Item", ""), 999))

    input_table = []
    low_reliability_inputs = []
    for row in input_rows:
        score = _parse_reliability(row.get("Reliability", ""))
        if score is not None and score < 60:
            low_reliability_inputs.append(f"{row.get('Item', 'Unknown')} ({score}/100)")

        input_table.append(
            {
                "Item": row.get("Item", "N/A"),
                "Value": row.get("Value", "N/A"),
                "Period": row.get("Period", "N/A"),
                "Source": row.get("Source", "N/A"),
                "Reliability": row.get("Reliability", "N/A"),
                "Notes": row.get("Notes", "N/A"),
            }
        )

    st.dataframe(pd.DataFrame(input_table), use_container_width=True, hide_index=True)
    if low_reliability_inputs:
        st.warning("Low reliability inputs: " + ", ".join(low_reliability_inputs))

    st.markdown("### 2. Assumptions")
    assumption_rows = [
        {
            "Assumption": "Forecast Horizon",
            "Value": f"{forecast_years} years",
            "Where Used": "Projection length and discount periods",
        },
        {
            "Assumption": "WACC",
            "Value": _fmt_rate(wacc_used),
            "Where Used": "Discounting explicit cash flows and terminal value",
        },
        {
            "Assumption": "Near-Term Growth",
            "Value": _fmt_rate(assumptions.get("fcf_growth_rate")),
            "Where Used": "Starting point for growth schedule",
        },
        {
            "Assumption": "Terminal Growth (g)",
            "Value": _fmt_rate(terminal_growth),
            "Where Used": "Terminal value and spread checks",
        },
        {
            "Assumption": "Tax Rate",
            "Value": _fmt_rate(assumptions.get("tax_rate")),
            "Where Used": "NOPAT and FCFF calculations",
        },
        {
            "Assumption": "Terminal Method",
            "Value": str(tv_method).replace("_", " ").title(),
            "Where Used": "Primary terminal valuation branch",
        },
        {
            "Assumption": "Exit Multiple",
            "Value": _fmt_multiple(assumptions.get("exit_multiple")),
            "Where Used": "Exit-multiple terminal valuation",
        },
        {
            "Assumption": "Growth Schedule",
            "Value": assumptions.get("growth_schedule_method", "N/A"),
            "Where Used": "Shape of year-by-year growth path",
        },
    ]
    if snapshot:
        wacc_components = getattr(snapshot, "wacc_components", {}) or {}
        if wacc_components.get("wacc_mode_label"):
            assumption_rows.insert(2, {
                "Assumption": "WACC Mode",
                "Value": str(wacc_components.get("wacc_mode_label")),
                "Where Used": "Interpretation confidence for discount-rate construction",
            })
    st.dataframe(pd.DataFrame(assumption_rows), use_container_width=True, hide_index=True)

    if snapshot:
        wacc_components = getattr(snapshot, "wacc_components", {}) or {}
        if wacc_components:
            wacc_mode = wacc_components.get("wacc_mode", "")
            wacc_mode_label = wacc_components.get("wacc_mode_label", "Component-based WACC estimate")
            wacc_confidence = str(wacc_components.get("wacc_confidence", "n/a")).title()
            mode_explanations = {
                "full_wacc_observed_debt": "Debt cost is directly inferred from interest expense and debt basis.",
                "estimated_wacc_debt_fallback": "Debt cost uses a synthetic spread fallback where observed debt cost is unreliable.",
                "coe_only_proxy_no_debt": "No debt was detected, so discount rate collapses to cost of equity.",
                "coe_only_proxy_financial_sector": "Financial-sector guardrail is active; CoE proxy is used instead of standard WACC.",
            }
            st.markdown("#### WACC Trace")
            st.info(
                "Forward WACC is inherently estimated from observable inputs (it is not directly observable as a single live metric). "
                f"{wacc_mode_label} (confidence: {wacc_confidence}). "
                f"{mode_explanations.get(wacc_mode, 'Discount rate is built from component inputs with explicit source labels.')}"
            )

            rf = _to_float(wacc_components.get("risk_free_rate"))
            erp = _to_float(wacc_components.get("market_risk_premium"))
            beta = _to_float(wacc_components.get("beta"))
            cost_of_equity_rate = _to_float(wacc_components.get("cost_of_equity"))
            rd = _to_float(wacc_components.get("cost_of_debt_pre_tax"))
            rd_after_tax = _to_float(wacc_components.get("cost_of_debt_after_tax"))
            tax_rate = _to_float(wacc_components.get("tax_rate"))
            we = _to_float(wacc_components.get("equity_weight"))
            wd = _to_float(wacc_components.get("debt_weight"))

            rd_mode = wacc_components.get("rd_mode")
            rd_observed_raw = _to_float(wacc_components.get("rd_observed_raw"))
            debt_basis_value = _to_float(wacc_components.get("debt_basis_value"))
            equity_value_used = _to_float(wacc_components.get("equity_value_used"))
            total_debt = _to_float(wacc_components.get("total_debt"))
            ttm_interest_expense = _to_float(getattr(getattr(snapshot, "ttm_interest_expense", None), "value", None))
            suggested_wacc_value = _to_float(getattr(getattr(snapshot, "suggested_wacc", None), "value", None))

            capm_formula = "Re = Rf + beta * ERP"
            if all(v is not None for v in [rf, erp, beta, cost_of_equity_rate]):
                capm_formula = (
                    f"Re = {rf*100:.2f}% + {beta:.2f} * {erp*100:.2f}% = {cost_of_equity_rate*100:.2f}%"
                )

            if rd_mode == "observed":
                rd_formula = "Rd = Interest Expense / Debt Basis"
                if all(v is not None for v in [ttm_interest_expense, debt_basis_value, rd_observed_raw]):
                    rd_formula = (
                        f"Rd = {_fmt_money(ttm_interest_expense)} / {_fmt_money(debt_basis_value)}"
                        f" = {rd_observed_raw*100:.2f}%"
                    )
            elif rd_mode == "synthetic_fallback":
                rd_formula = "Rd = Rf + synthetic spread fallback"
                if all(v is not None for v in [rf, rd]):
                    spread = rd - rf
                    rd_formula = f"Rd = {rf*100:.2f}% + {spread*100:.2f}% = {rd*100:.2f}%"
            elif rd_mode == "no_debt":
                rd_formula = "No debt detected -> Rd branch is not used"
            else:
                rd_formula = "Rd path unavailable"

            rd_after_tax_formula = "Rd_after_tax = Rd * (1 - T)"
            if all(v is not None for v in [rd, tax_rate, rd_after_tax]):
                rd_after_tax_formula = (
                    f"Rd_after_tax = {rd*100:.2f}% * (1 - {tax_rate*100:.2f}%) = {rd_after_tax*100:.2f}%"
                )

            weight_formula = "We = E / (E + D), Wd = D / (E + D)"
            if all(v is not None for v in [equity_value_used, total_debt]) and (equity_value_used + total_debt) > 0:
                weight_formula = (
                    f"We = {_fmt_money(equity_value_used)} / ({_fmt_money(equity_value_used)} + {_fmt_money(total_debt)})"
                    f" = {((we or 0.0) * 100):.2f}%, "
                    f"Wd = {_fmt_money(total_debt)} / ({_fmt_money(equity_value_used)} + {_fmt_money(total_debt)})"
                    f" = {((wd or 0.0) * 100):.2f}%"
                )

            if wacc_mode in {"coe_only_proxy_no_debt", "coe_only_proxy_financial_sector"}:
                final_formula = "Discount rate = Re (CoE proxy mode)"
                if cost_of_equity_rate is not None and suggested_wacc_value is not None:
                    final_formula = f"Discount rate = Re = {cost_of_equity_rate*100:.2f}%"
            else:
                final_formula = "WACC = We * Re + Wd * Rd_after_tax"
                if all(v is not None for v in [we, cost_of_equity_rate, wd, rd_after_tax, suggested_wacc_value]):
                    final_formula = (
                        f"WACC = {we*100:.2f}% * {cost_of_equity_rate*100:.2f}% + {wd*100:.2f}% * {rd_after_tax*100:.2f}%"
                        f" = {suggested_wacc_value*100:.2f}%"
                    )

            wacc_trace_rows = [
                {
                    "Step": "1. CAPM Inputs",
                    "Formula / Calculation": (
                        f"Rf={_fmt_rate(rf)}, ERP={_fmt_rate(erp)}, beta={_fmt_number(beta, 2)}"
                    ),
                    "Result": "Inputs loaded",
                    "Source": f"{wacc_components.get('rf_source', '10Y Treasury')} + Damodaran + Yahoo",
                },
                {
                    "Step": "2. Cost of Equity",
                    "Formula / Calculation": capm_formula,
                    "Result": _fmt_rate(cost_of_equity_rate),
                    "Source": wacc_components.get("cost_of_equity_source", "CAPM"),
                },
                {
                    "Step": "3. Cost of Debt (pre-tax)",
                    "Formula / Calculation": rd_formula,
                    "Result": _fmt_rate(rd),
                    "Source": wacc_components.get("debt_cost_source", "N/A"),
                },
                {
                    "Step": "4. Cost of Debt (after-tax)",
                    "Formula / Calculation": rd_after_tax_formula,
                    "Result": _fmt_rate(rd_after_tax),
                    "Source": "Tax-adjusted debt cost",
                },
                {
                    "Step": "5. Capital Structure Weights",
                    "Formula / Calculation": weight_formula,
                    "Result": f"We={_fmt_rate(we)}, Wd={_fmt_rate(wd)}",
                    "Source": wacc_components.get("weights_source", "N/A"),
                },
                {
                    "Step": "6. Final Discount Rate",
                    "Formula / Calculation": final_formula,
                    "Result": _fmt_rate(suggested_wacc_value),
                    "Source": wacc_mode_label,
                },
            ]
            st.dataframe(pd.DataFrame(wacc_trace_rows), use_container_width=True, hide_index=True)

            wacc_rows = [
                {"Component": "WACC Mode", "Value": str(wacc_mode_label), "Source": "Method classification"},
                {"Component": "WACC Confidence", "Value": wacc_confidence, "Source": "Data quality and fallback diagnostics"},
                {"Component": "Debt Basis", "Value": _fmt_money(debt_basis_value), "Source": wacc_components.get("debt_basis_source", "N/A")},
                {"Component": "Tax Rate (T)", "Value": _fmt_rate(tax_rate), "Source": "Income statement / fallback"},
            ]
            st.markdown("#### WACC Component Sources")
            st.dataframe(pd.DataFrame(wacc_rows), use_container_width=True, hide_index=True)

            rd_guardrail_note = wacc_components.get("rd_guardrail_note")
            if rd_guardrail_note:
                st.caption(f"Rd guardrail: {rd_guardrail_note}")
        else:
            beta = _to_float(getattr(getattr(snapshot, "beta", None), "value", None))
            risk_free_rate = _to_float(getattr(snapshot, "risk_free_rate", None))
            market_risk_premium = 0.05
            rf_source_raw = getattr(snapshot, "rf_source", None)
            rf_source = str(rf_source_raw).strip() if rf_source_raw is not None else ""
            legacy_rf_fallback = False
            if risk_free_rate is None:
                risk_free_rate = 0.045
                legacy_rf_fallback = True
            if not rf_source or rf_source.lower() in {"none", "n/a", "na"}:
                rf_source = (
                    "Fallback default (legacy cache snapshot)"
                    if legacy_rf_fallback
                    else "10-Year Treasury"
                )

            capm_rows = [
                {"Input": "Risk-Free Rate (Rf)", "Value": _fmt_rate(risk_free_rate), "Source": rf_source},
                {"Input": "Equity Risk Premium (ERP)", "Value": _fmt_rate(market_risk_premium), "Source": "Damodaran implied ERP"},
                {"Input": "Beta (beta)", "Value": _fmt_number(beta, 2), "Source": "Yahoo Finance (5Y monthly)"},
            ]

            st.markdown("#### Discount Rate Inputs")
            st.dataframe(pd.DataFrame(capm_rows), use_container_width=True, hide_index=True)
            if legacy_rf_fallback:
                st.caption("This run came from older cached data missing Rf metadata. Using a 4.5% fallback here; rerun DCF to restore full WACC trace.")

    st.markdown("### 3. Forecast and Present Value")
    fcff_method = assumptions.get("fcff_method")
    fcff_reliability = assumptions.get("fcff_reliability")
    fcff_method_labels = {
        "proper_fcff": "Proper FCFF",
        "approx_unlevered": "Approximate Unlevered FCFF",
        "unlevered_proxy": "Approximate Unlevered FCFF",
        "levered_proxy": "Levered FCF Proxy",
    }
    method_label = fcff_method_labels.get(fcff_method, "Unknown")

    if fcff_reliability is not None:
        st.caption(
            "Driver chain: Revenue -> EBIT -> NOPAT -> Reinvestment -> FCFF. "
            f"FCFF basis: {method_label} ({fcff_reliability}/100 reliability)."
        )
    else:
        st.caption(
            "Driver chain: Revenue -> EBIT -> NOPAT -> Reinvestment -> FCFF. "
            f"FCFF basis: {method_label}."
        )

    growth_summary_rows = [
        {"Parameter": "Near-Term Growth", "Value": _fmt_rate(assumptions.get("near_term_growth_rate"))},
        {"Parameter": "Effective Near-Term Growth", "Value": _fmt_rate(assumptions.get("effective_near_term_growth_rate"))},
        {"Parameter": "Analyst Long-Term Growth Anchor", "Value": _fmt_rate(assumptions.get("analyst_long_term_growth_rate"))},
        {"Parameter": "Terminal Growth", "Value": _fmt_rate(assumptions.get("stable_growth_rate"))},
        {"Parameter": "Current ROIC", "Value": _fmt_rate(assumptions.get("base_roic"))},
        {"Parameter": "Terminal ROIC", "Value": _fmt_rate(assumptions.get("terminal_roic"))},
        {"Parameter": "Terminal Reinvestment Rate", "Value": _fmt_rate(assumptions.get("terminal_reinvestment_rate"))},
    ]
    st.dataframe(pd.DataFrame(growth_summary_rows), use_container_width=True, hide_index=True)

    projection_rows = []
    if yearly_projections:
        for proj in yearly_projections:
            year = proj.get("year")
            projection_rows.append(
                {
                    "Year": f"Y{int(year)}" if isinstance(year, (int, float)) else "N/A",
                    "Revenue": _fmt_money(proj.get("revenue")),
                    "Growth": _fmt_rate(proj.get("revenue_growth")),
                    "EBIT Margin": _fmt_rate(proj.get("ebit_margin")),
                    "EBIT": _fmt_money(proj.get("ebit")),
                    "NOPAT": _fmt_money(proj.get("nopat")),
                    "Reinvestment": _fmt_money(proj.get("reinvestment")),
                    "Reinvestment Rate": _fmt_rate(proj.get("reinvestment_rate")),
                    "FCFF": _fmt_money(proj.get("fcff")),
                    "PV(FCFF)": _fmt_money(proj.get("pv_fcff")),
                    "Revenue Source": proj.get("revenue_source", "driver_growth"),
                    "FCFF Source": proj.get("fcf_source", "driver_model"),
                }
            )
    elif fcf_projections:
        for proj in fcf_projections:
            year = proj.get("year")
            projection_rows.append(
                {
                    "Year": f"Y{int(year)}" if isinstance(year, (int, float)) else "N/A",
                    "FCFF": _fmt_money(proj.get("fcff", proj.get("fcf"))),
                    "Discount Factor": _fmt_number(proj.get("discount_factor"), 4),
                    "PV(FCFF)": _fmt_money(proj.get("pv")),
                }
            )

    if projection_rows:
        st.dataframe(pd.DataFrame(projection_rows), use_container_width=True, hide_index=True)
    else:
        st.warning("No forecast rows are available for this run.")

    st.markdown("### 4. Terminal Value")
    st.caption(f"Primary terminal method: {str(tv_method).replace('_', ' ').title()}")

    discount_factor = None
    if _to_float(wacc_used) is not None and forecast_years > 0:
        discount_factor = 1 / ((1 + float(wacc_used)) ** forecast_years)

    terminal_rows = []
    if tv_method == "exit_multiple":
        terminal_ebitda = assumptions.get("terminal_year_ebitda") or assumptions.get("projected_terminal_ebitda")
        terminal_rows = [
            {
                "Step": "Terminal EBITDA (Year N)",
                "Formula": "Projected EBITDA in final forecast year",
                "Value": _fmt_money(terminal_ebitda),
            },
            {
                "Step": "Exit Multiple",
                "Formula": "Selected EV/EBITDA terminal multiple",
                "Value": _fmt_multiple(assumptions.get("exit_multiple")),
            },
            {
                "Step": "Terminal Value (Year N)",
                "Formula": "Terminal EBITDA * Exit Multiple",
                "Value": _fmt_money(terminal_value_yearN),
            },
            {
                "Step": "Discount Factor",
                "Formula": "1 / (1 + WACC)^N",
                "Value": _fmt_number(discount_factor, 4) if discount_factor is not None else "N/A",
            },
            {
                "Step": "PV(Terminal Value)",
                "Formula": "Terminal Value * Discount Factor",
                "Value": _fmt_money(pv_tv),
            },
        ]
    else:
        terminal_fcff = assumptions.get("terminal_year_fcff")
        if terminal_fcff is None and yearly_projections:
            terminal_fcff = yearly_projections[-1].get("fcff")
        if terminal_fcff is None and fcf_projections:
            terminal_fcff = fcf_projections[-1].get("fcff", fcf_projections[-1].get("fcf"))

        terminal_fcff_n1 = None
        if _to_float(terminal_fcff) is not None and _to_float(terminal_growth) is not None:
            terminal_fcff_n1 = float(terminal_fcff) * (1 + float(terminal_growth))

        terminal_rows = [
            {
                "Step": "FCFF (Year N)",
                "Formula": "Projected FCFF in final forecast year",
                "Value": _fmt_money(terminal_fcff),
            },
            {
                "Step": "Terminal Growth (g)",
                "Formula": "Long-run growth assumption",
                "Value": _fmt_rate(terminal_growth),
            },
            {
                "Step": "FCFF (Year N+1)",
                "Formula": "FCFF_N * (1 + g)",
                "Value": _fmt_money(terminal_fcff_n1),
            },
            {
                "Step": "Terminal Value (Year N)",
                "Formula": "FCFF_(N+1) / (WACC - g)",
                "Value": _fmt_money(terminal_value_yearN),
            },
            {
                "Step": "Discount Factor",
                "Formula": "1 / (1 + WACC)^N",
                "Value": _fmt_number(discount_factor, 4) if discount_factor is not None else "N/A",
            },
            {
                "Step": "PV(Terminal Value)",
                "Formula": "Terminal Value * Discount Factor",
                "Value": _fmt_money(pv_tv),
            },
        ]

    st.dataframe(pd.DataFrame(terminal_rows), use_container_width=True, hide_index=True)

    st.markdown("#### Terminal Method Cross-Check")
    cross_check_rows = [
        {
            "Method": "Gordon Growth",
            "TV (Year N)": _fmt_money(assumptions.get("tv_gordon_growth")),
            "PV(TV)": _fmt_money(assumptions.get("pv_tv_gordon_growth")),
            "Implied Price": (
                f"${_to_float(assumptions.get('price_gordon_growth')):.2f}"
                if _to_float(assumptions.get("price_gordon_growth")) is not None
                else "N/A"
            ),
            "Used in EV": "Yes" if tv_method == "gordon_growth" else "No",
        },
        {
            "Method": "Exit Multiple",
            "TV (Year N)": _fmt_money(assumptions.get("tv_exit_multiple")),
            "PV(TV)": _fmt_money(assumptions.get("pv_tv_exit_multiple")),
            "Implied Price": (
                f"${_to_float(assumptions.get('price_exit_multiple')):.2f}"
                if _to_float(assumptions.get("price_exit_multiple")) is not None
                else "N/A"
            ),
            "Used in EV": "Yes" if tv_method == "exit_multiple" else "No",
        },
    ]
    st.dataframe(pd.DataFrame(cross_check_rows), use_container_width=True, hide_index=True)

    exit_scenarios = assumptions.get("exit_multiple_scenarios", []) or []
    if exit_scenarios:
        st.markdown("#### Exit Multiple Scenarios")
        scenario_rows = []
        for scenario in exit_scenarios:
            scenario_rows.append(
                {
                    "Scenario": scenario.get("name", "N/A"),
                    "Multiple": _fmt_multiple(scenario.get("multiple")),
                    "Implied Price": (
                        f"${_to_float(scenario.get('price')):.0f}"
                        if _to_float(scenario.get("price")) is not None
                        else "N/A"
                    ),
                    "Required FCFF/EBITDA": _fmt_rate(scenario.get("required_fcff_ebitda"), 0),
                    "Gap vs Forecast": scenario.get("gap", "N/A"),
                    "Status": scenario.get("status", "N/A"),
                }
            )
        st.dataframe(pd.DataFrame(scenario_rows), use_container_width=True, hide_index=True)

    st.markdown("### 5. Enterprise to Equity Bridge")
    bridge_rows = [
        {
            "Step": "PV of Explicit FCFF",
            "Formula": "Sum of discounted yearly FCFF",
            "Value": _fmt_money(pv_fcf_sum),
        },
        {
            "Step": "PV of Terminal Value",
            "Formula": "Discounted terminal value",
            "Value": _fmt_money(pv_tv),
        },
        {
            "Step": "Enterprise Value",
            "Formula": "PV(FCFF) + PV(TV)",
            "Value": _fmt_money(ev),
        },
        {
            "Step": "Net Debt",
            "Formula": "Total Debt - Cash and Equivalents",
            "Value": _fmt_money(net_debt),
        },
        {
            "Step": "Equity Value",
            "Formula": "Enterprise Value - Net Debt",
            "Value": _fmt_money(equity),
        },
        {
            "Step": "Intrinsic Value / Share",
            "Formula": "Equity Value / Shares Outstanding",
            "Value": f"${_to_float(price_per_share):.2f}" if _to_float(price_per_share) is not None else "N/A",
        },
    ]
    st.dataframe(pd.DataFrame(bridge_rows), use_container_width=True, hide_index=True)

    calc_ev = (_to_float(pv_fcf_sum) or 0.0) + (_to_float(pv_tv) or 0.0)
    calc_equity = (_to_float(ev) or 0.0) - (_to_float(net_debt) or 0.0)
    calc_price = calc_equity / _to_float(shares_outstanding) if _to_float(shares_outstanding) else None

    ev_delta = (_to_float(ev) - calc_ev) if _to_float(ev) is not None else None
    equity_delta = (_to_float(equity) - calc_equity) if _to_float(equity) is not None else None
    price_delta = (
        (_to_float(price_per_share) - calc_price)
        if (_to_float(price_per_share) is not None and calc_price is not None)
        else None
    )

    def _delta_status(delta, scale):
        if delta is None:
            return "N/A"
        tolerance = max(1.0, abs(scale) * 0.001)
        return "PASS" if abs(delta) <= tolerance else "CHECK"

    reconciliation_rows = [
        {
            "Check": "EV = PV(FCFF) + PV(TV)",
            "Computed": _fmt_money(calc_ev),
            "Reported": _fmt_money(ev),
            "Delta": _fmt_money(ev_delta),
            "Status": _delta_status(ev_delta, _to_float(ev) or calc_ev),
        },
        {
            "Check": "Equity = EV - Net Debt",
            "Computed": _fmt_money(calc_equity),
            "Reported": _fmt_money(equity),
            "Delta": _fmt_money(equity_delta),
            "Status": _delta_status(equity_delta, _to_float(equity) or calc_equity),
        },
        {
            "Check": "Price = Equity / Shares",
            "Computed": f"${calc_price:.4f}" if calc_price is not None else "N/A",
            "Reported": f"${_to_float(price_per_share):.4f}" if _to_float(price_per_share) is not None else "N/A",
            "Delta": f"${price_delta:.4f}" if price_delta is not None else "N/A",
            "Status": _delta_status(price_delta, _to_float(price_per_share) or (calc_price or 0.0)),
        },
    ]
    st.markdown("#### Reconciliation Checks")
    st.dataframe(pd.DataFrame(reconciliation_rows), use_container_width=True, hide_index=True)

    st.markdown("### 6. Warnings, Diagnostics, and Trace")
    errors = [err for err in ui_data.get("errors", []) if err]
    warnings = [warn for warn in ui_data.get("warnings", []) if warn]
    diagnostics = [diag for diag in ui_data.get("diagnostics", []) if diag]
    grouped_notices = {"critical": [], "review": [], "info": []}
    seen_notices = set()

    for message in errors:
        cleaned = _sanitize_notice_text(message)
        key = ("critical", cleaned)
        if cleaned and key not in seen_notices:
            grouped_notices["critical"].append(cleaned)
            seen_notices.add(key)

    for message in warnings:
        cleaned = _sanitize_notice_text(message)
        severity = _classify_notice(cleaned, "warning")
        key = (severity, cleaned)
        if cleaned and key not in seen_notices:
            grouped_notices[severity].append(cleaned)
            seen_notices.add(key)

    for message in diagnostics:
        cleaned = _sanitize_notice_text(message)
        severity = _classify_notice(cleaned, "diagnostic")
        key = (severity, cleaned)
        if cleaned and key not in seen_notices:
            grouped_notices[severity].append(cleaned)
            seen_notices.add(key)

    total_notices = sum(len(items) for items in grouped_notices.values())
    if total_notices > 0:
        c_count = len(grouped_notices["critical"])
        r_count = len(grouped_notices["review"])
        i_count = len(grouped_notices["info"])
        st.caption(f"Organized by severity: Critical {c_count} | Review {r_count} | Info {i_count}")

        if grouped_notices["critical"]:
            st.error("Critical Checks")
            for idx, item in enumerate(grouped_notices["critical"], start=1):
                st.markdown(f"{idx}. {item}")

        if grouped_notices["review"]:
            st.warning("Review Items")
            for idx, item in enumerate(grouped_notices["review"], start=1):
                st.markdown(f"{idx}. {item}")

        if grouped_notices["info"]:
            st.info("Model Notes")
            for idx, item in enumerate(grouped_notices["info"], start=1):
                st.markdown(f"{idx}. {item}")
    else:
        st.caption("No warnings or diagnostics were returned for this run.")

    if trace_steps:
        trace_rows = []
        for idx, step in enumerate(trace_steps, start=1):
            output = step.get("output")
            output_units = step.get("output_units")
            if isinstance(output, (int, float)) and output_units == "USD":
                output_text = _fmt_money(output)
            elif isinstance(output, (int, float)):
                output_text = _fmt_number(output, 4)
            else:
                output_text = str(output) if output is not None else "N/A"

            inputs = step.get("inputs", {})
            input_keys = ", ".join(inputs.keys()) if isinstance(inputs, dict) and inputs else "N/A"
            trace_rows.append(
                {
                    "Step #": idx,
                    "Name": _sanitize_notice_text(step.get("name", "N/A") or "N/A"),
                    "Formula": step.get("formula", "N/A") or "N/A",
                    "Input Keys": input_keys,
                    "Output": output_text,
                    "Notes": _sanitize_notice_text(step.get("notes", "N/A") or "N/A"),
                }
            )
        st.dataframe(pd.DataFrame(trace_rows), use_container_width=True, hide_index=True)
    else:
        st.caption("No trace steps available.")

    trace_payload = {
        "inputs": {k: str(v) for k, v in ui_data.get("inputs", {}).items()},
        "assumptions": assumptions,
        "results": {
            "enterprise_value": ev,
            "equity_value": equity,
            "price_per_share": price_per_share,
            "pv_fcf_sum": pv_fcf_sum,
            "pv_terminal_value": pv_tv,
        },
        "trace_steps": trace_steps,
    }
    trace_json = json.dumps(trace_payload, indent=2, default=str)
    st.download_button(
        label="Download Trace (JSON)",
        data=trace_json,
        file_name=f"{st.session_state.get('ticker', 'valuation')}_dcf_trace.json",
        mime="application/json",
    )

    st.divider()
    if st.button("<- Back to Summary", key="back_from_details_bottom"):
        _queue_scroll_to_section("valuation")
        st.session_state.show_dcf_details = False
        st.rerun()


def _show_user_guide_page():
    """Render standalone user guide so the main analysis page stays uncluttered."""
    st.markdown("---")
    st.markdown(
        '<div class="section-header"><span class="step-badge">Guide</span><span class="section-title">User Guide</span></div>',
        unsafe_allow_html=True,
    )
    st.caption("How to run the workflow, interpret outputs, and avoid common mistakes.")

    if st.button("<- Back to Analysis", key="back_from_user_guide_top"):
        st.session_state.show_user_guide = False
        _clear_view_query_param()
        st.rerun()

    st.markdown(
        """
<div class="guide-card">
  <div class="guide-card-title">What This Tool Is</div>
  <p class="guide-step-desc">Analyst Co-Pilot is a traceable DCF workflow with market context and AI synthesis.</p>
  <p class="guide-step-desc"><strong>Not:</strong> a one-click price target tool.</p>
</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="guide-card guide-card-priority">
  <div class="guide-card-title">How to Use It</div>
  <div class="guide-onboarding-grid">
    <div class="guide-onboarding-step"><span class="guide-onboarding-num">1</span><div><div class="guide-step-name">Select ticker and load data</div><div class="guide-step-desc">Choose symbol, ending report, then click <code>Load Data</code>.</div></div></div>
    <div class="guide-onboarding-step"><span class="guide-onboarding-num">2</span><div><div class="guide-step-name">Read the dashboard first</div><div class="guide-step-desc">Use Dashboard for the general stock picture, momentum, and DuPont context.</div></div></div>
    <div class="guide-onboarding-step"><span class="guide-onboarding-num">3</span><div><div class="guide-step-name">Set assumptions and run DCF</div><div class="guide-step-desc">In Deep Dive, adjust inputs, run <code>Run DCF Analysis</code>, and read the verdict at the top.</div></div></div>
    <div class="guide-onboarding-step"><span class="guide-onboarding-num">4</span><div><div class="guide-step-name">Generate AI synthesis last</div><div class="guide-step-desc">Use Step 05 only after DCF and diagnostics are complete.</div></div></div>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="guide-card">
  <div class="guide-card-title">How to Read the Result</div>
  <ul class="guide-list">
    <li><strong>Intrinsic value:</strong> model-implied under assumptions, not objective truth.</li>
    <li><strong>TV dominance:</strong> a sensitivity warning when terminal value drives most of EV.</li>
    <li><strong>Data quality:</strong> confidence in input reliability and fallback usage.</li>
    <li><strong>Method divergence:</strong> larger Gordon vs Exit differences imply higher terminal uncertainty.</li>
  </ul>
</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="guide-card">
  <div class="guide-card-title">What Each App Step Means</div>
  <ul class="guide-list">
    <li><strong>Step 01:</strong> business momentum and DuPont context on Dashboard.</li>
    <li><strong>Verdict:</strong> live valuation result at the top of Deep Dive after a rerun.</li>
    <li><strong>Step 03:</strong> DCF controls and core valuation outputs.</li>
    <li><strong>Step 04:</strong> analyst expectations.</li>
    <li><strong>Step 05:</strong> AI interpretation.</li>
    <li><strong>Step 06:</strong> sources and method notes.</li>
  </ul>
</div>
        """,
        unsafe_allow_html=True,
    )

    col_workflow, col_warnings = st.columns(2)
    with col_workflow:
        st.markdown(
            """
<div class="guide-card">
  <div class="guide-card-title">Recommended Workflow for Real Use</div>
  <ul class="guide-list">
    <li>Start with suggested assumptions.</li>
    <li>Change one variable at a time.</li>
    <li>Run bull/base/bear scenarios.</li>
    <li>Check whether the conclusion survives sensitivity.</li>
    <li>Audit the trace before trusting the output.</li>
  </ul>
</div>
            """,
            unsafe_allow_html=True,
        )
    with col_warnings:
        st.markdown(
            """
<div class="guide-card">
  <div class="guide-card-title">Common Mistakes and Warnings</div>
  <ul class="guide-list">
    <li>Do not treat intrinsic value as a floor.</li>
    <li>Do not rely on one point estimate alone.</li>
    <li>Do not ignore TV dominance warnings.</li>
    <li>Do not compare runs across dates without reloading context.</li>
    <li>Do not set terminal growth near WACC.</li>
  </ul>
</div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
<div class="guide-card">
  <div class="guide-card-title">Data Quality and Methodology</div>
  <p class="guide-step-desc"><strong>What the score means:</strong> confidence in how complete and direct the valuation inputs are versus estimated or fallback handling.</p>
  <div class="guide-score-table-wrap">
    <table class="guide-score-table">
      <thead>
        <tr>
          <th>Score</th>
          <th>Confidence Label</th>
          <th>Why It Usually Scores Here</th>
          <th>How to Use It</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>90-100</td>
          <td>High confidence</td>
          <td>Most core inputs are direct current/TTM values with little or no fallback usage.</td>
          <td>Use as primary case, then run normal sensitivity checks.</td>
        </tr>
        <tr>
          <td>75-89</td>
          <td>Good confidence</td>
          <td>One or two inputs may rely on annual proxies or light estimation.</td>
          <td>Usable for base case; validate discount rate and terminal assumptions.</td>
        </tr>
        <tr>
          <td>60-74</td>
          <td>Moderate confidence</td>
          <td>Multiple fields use fallback logic or partial data coverage.</td>
          <td>Use directionally; widen bull/base/bear ranges.</td>
        </tr>
        <tr>
          <td>40-59</td>
          <td>Low confidence</td>
          <td>Key metrics are missing or heavily estimated (for example synthetic fallback paths).</td>
          <td>Avoid strong single-point conclusions; prioritize data verification.</td>
        </tr>
        <tr>
          <td>0-39</td>
          <td>Very low confidence</td>
          <td>Major data gaps across core valuation inputs.</td>
          <td>Treat output as exploratory only until data quality improves.</td>
        </tr>
      </tbody>
    </table>
  </div>
  <p class="guide-step-desc"><strong>Single metric vs calculated metric reliability:</strong></p>
  <div class="guide-score-table-wrap">
    <table class="guide-score-table">
      <thead>
        <tr>
          <th>Metric Type</th>
          <th>What Is Being Scored</th>
          <th>Scoring Logic</th>
          <th>Examples</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Single metric</td>
          <td>One directly observed field from a source feed or statement.</td>
          <td>Higher when current/TTM is direct; lower when missing, stale, annual-only, or fallback-derived.</td>
          <td>Price, Shares, TTM Revenue, Total Debt, Cash, Tax Rate input.</td>
        </tr>
        <tr>
          <td>Calculated metric</td>
          <td>A model output built from several upstream inputs and assumptions.</td>
          <td>Inherited from component quality plus penalties for proxy paths and estimation layers.</td>
          <td>Suggested WACC, FCFF projection chain.</td>
        </tr>
      </tbody>
    </table>
  </div>
  <p class="guide-step-desc"><strong>Audit path:</strong> open <span class="guide-inline-label">View DCF Details</span> and review each input's Reliability plus fallback notes.</p>
</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="guide-disclaimer-card">
  <div class="guide-disclaimer-title">Disclaimer</div>
  <div class="guide-disclaimer-text">This tool supports research workflows only. Outputs may contain mistakes and are assumption-sensitive. It is not investment advice.</div>
</div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("<- Back to Analysis", key="back_from_user_guide_bottom"):
        st.session_state.show_user_guide = False
        _clear_view_query_param()
        st.rerun()


def _show_contact_page():
    """Render a simple contact destination without any form submission flow."""
    _contact_title_col, _contact_close_col = st.columns([10, 1])
    with _contact_title_col:
        st.markdown(
            '<div class="section-header"><span class="step-badge">Contact</span><span class="section-title">Share an Idea</span></div>',
            unsafe_allow_html=True,
        )
    with _contact_close_col:
        if st.button("Close", key="close_contact_panel_top", type="tertiary"):
            st.session_state.show_contact_panel = False
            st.rerun()
    st.caption("For suggestions or feedback, email us directly.")
    st.markdown("### Suggestions Email")
    if EMAIL_REGEX.match(CONTACT_EMAIL_TO):
        st.markdown(f"**{CONTACT_EMAIL_TO}**")
        st.markdown(f"[Compose Email](mailto:{CONTACT_EMAIL_TO})")
    else:
        st.info("Contact email is not configured for this deployment.")
# --- App Configuration ---
_write_boot_log("page_config_set")


def _boot_log(message: str) -> None:
    _write_boot_log(message)

# --- Design System CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ═══════════════════════════════════════════
       DESIGN TOKENS
    ═══════════════════════════════════════════ */
    :root {
      --clr-bg:             #f8fafc;
      --clr-surface:        #ffffff;
      --clr-sidebar-bg:     #0d1630;
      --clr-sidebar-text:   #e2e8f0;
      --clr-sidebar-muted:  #94a3b8;
      --clr-accent:         #2563eb;
      --clr-accent-hover:   #1d4ed8;
      --clr-success:        #10b981;
      --clr-success-bg:     #ecfdf5;
      --clr-success-text:   #065f46;
      --clr-danger:         #f43f5e;
      --clr-danger-bg:      #fff1f2;
      --clr-danger-text:    #9f1239;
      --clr-warn:           #f59e0b;
      --clr-warn-bg:        #fffbeb;
      --clr-warn-text:      #92400e;
      --clr-text-primary:   #0f172a;
      --clr-text-secondary: #475569;
      --clr-text-muted:     #94a3b8;
      --clr-border:         #e2e8f0;
      --clr-border-strong:  #cbd5e1;
      --shadow-sm:  0 1px 3px rgba(0,0,0,0.08);
      --shadow-md:  0 4px 12px rgba(0,0,0,0.10);
      --radius-sm: 4px;
      --radius-md: 8px;
      --radius-lg: 12px;
    }

    /* QW-1: Global background */
    .stApp {
        background: var(--clr-bg) !important;
    }
    .main .block-container {
        padding-top: 0.9rem !important;
        padding-bottom: 1rem !important;
    }

    /* QW-2: Dark navy sidebar */
    [data-testid="stSidebar"] {
        background: var(--clr-sidebar-bg) !important;
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: var(--clr-sidebar-text) !important;
    }
    /* Input container wrapper (this is what shows the white bg) */
    [data-testid="stSidebar"] [data-baseweb="input"],
    [data-testid="stSidebar"] [data-baseweb="base-input"],
    [data-testid="stSidebar"] .stTextInput > div > div {
        background: rgba(255,255,255,0.10) !important;
        border: 1px solid rgba(255,255,255,0.20) !important;
        border-radius: var(--radius-sm) !important;
    }
    /* Remove BaseWeb's red/blue focus outline; replace with accent glow */
    [data-testid="stSidebar"] [data-baseweb="input"]:focus-within,
    [data-testid="stSidebar"] .stTextInput > div > div:focus-within {
        border-color: var(--clr-accent) !important;
        box-shadow: 0 0 0 2px rgba(37,99,235,0.35) !important;
        outline: none !important;
    }
    /* Inner input element — transparent bg so container colour shows */
    [data-testid="stSidebar"] [data-baseweb="input"] input,
    [data-testid="stSidebar"] input {
        background: transparent !important;
        color: var(--clr-sidebar-text) !important;
        caret-color: var(--clr-sidebar-text) !important;
        border: none !important;
        box-shadow: none !important;
    }
    [data-testid="stSidebar"] [data-baseweb="input"] input::placeholder,
    [data-testid="stSidebar"] input::placeholder {
        color: var(--clr-sidebar-muted) !important;
        opacity: 1 !important;
    }
    /* "Press Enter to apply" helper text */
    [data-testid="stSidebar"] [data-testid="InputInstructions"],
    [data-testid="stSidebar"] .stTextInput small,
    [data-testid="stSidebar"] .stTextInput [class*="instructions"] {
        color: var(--clr-sidebar-muted) !important;
    }
    /* Selectbox */
    [data-testid="stSidebar"] [data-baseweb="select"],
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background: rgba(255,255,255,0.10) !important;
        border: 1px solid rgba(255,255,255,0.20) !important;
        border-radius: var(--radius-sm) !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] [data-baseweb="selected-option"],
    [data-testid="stSidebar"] [data-baseweb="select"] [data-baseweb="placeholder"],
    [data-testid="stSidebar"] [data-baseweb="select"] svg {
        color: var(--clr-sidebar-text) !important;
        fill: var(--clr-sidebar-text) !important;
    }
    [data-testid="stSidebar"] [data-testid="stSliderThumb"],
    [data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"] {
        background: var(--clr-accent) !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.1) !important;
    }
    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] small {
        color: var(--clr-sidebar-muted) !important;
    }
    [data-testid="stSidebar"] .stButton > button[kind="primary"],
    [data-testid="stSidebar"] .stButton > button[data-testid="baseButton-primary"] {
        background: var(--clr-accent) !important;
        border: none !important;
        color: #fff !important;
    }
    [data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
        background: var(--clr-accent-hover) !important;
    }
    [data-testid="stSidebar"] .stButton > button[kind="secondary"],
    [data-testid="stSidebar"] .stButton > button[data-testid="baseButton-secondary"] {
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        color: var(--clr-sidebar-text) !important;
    }
    [data-testid="stSidebar"] .stButton > button {
        color: var(--clr-sidebar-text) !important;
    }

    /* QW-3: Metric tile cards */
    [data-testid="metric-container"] {
        background: var(--clr-surface) !important;
        border: 1px solid var(--clr-border) !important;
        border-radius: var(--radius-md) !important;
        padding: 14px 16px !important;
        box-shadow: var(--shadow-sm) !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.68rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
        color: var(--clr-text-muted) !important;
    }
    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 1.35rem !important;
        font-weight: 600 !important;
        color: var(--clr-text-primary) !important;
    }
    [data-testid="stMetricDelta"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.78rem !important;
    }

    /* QW-4: Section divider upgrade */
    hr {
        border: none !important;
        border-top: 2px solid var(--clr-border) !important;
        margin: 1.15rem 0 !important;
    }

    /* QW-5: Typography hierarchy */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: var(--clr-text-primary) !important;
    }
    h1 { font-size: 1.5rem !important; font-weight: 700 !important; }
    h2 { font-size: 1.15rem !important; font-weight: 700 !important; text-transform: uppercase; letter-spacing: 0.04em; }
    h3 { font-size: 0.95rem !important; font-weight: 700 !important; }
    .stApp, .stApp p, .stApp div, .stApp label {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    .stApp [class*="material-symbols"],
    .stApp .material-symbols-outlined,
    .stApp .material-symbols-rounded {
        font-family: "Material Symbols Rounded", "Material Symbols Outlined", "Material Icons" !important;
        font-style: normal;
        line-height: 1;
    }
    .stApp [data-testid="stExpanderToggleIcon"],
    .stApp [data-testid="stExpanderToggleIcon"] * {
        font-family: "Material Symbols Rounded", "Material Symbols Outlined", "Material Icons" !important;
        font-style: normal !important;
        font-weight: 400 !important;
        line-height: 1 !important;
    }
    .stCaption { color: var(--clr-text-muted) !important; font-size: 0.78rem !important; }

    /* QW-6: Call-box token migration */
    .call-outperform {
        background: var(--clr-success-bg) !important;
        border: 1px solid var(--clr-success) !important;
        color: var(--clr-success-text) !important;
        padding: 1rem;
        border-radius: var(--radius-md);
        text-align: center;
    }
    .call-underperform {
        background: var(--clr-danger-bg) !important;
        border: 1px solid var(--clr-danger) !important;
        color: var(--clr-danger-text) !important;
        padding: 1rem;
        border-radius: var(--radius-md);
        text-align: center;
    }
    .call-inline {
        background: var(--clr-bg) !important;
        border: 1px solid var(--clr-border-strong) !important;
        color: var(--clr-text-secondary) !important;
        padding: 1rem;
        border-radius: var(--radius-md);
        text-align: center;
    }
    .call-label {
        font-size: 0.875rem;
        font-weight: 700;
        letter-spacing: 0.05em;
    }

    /* QW-7: Expander modernization */
    [data-testid="stExpander"] {
        background: var(--clr-surface) !important;
        border: 1px solid var(--clr-border) !important;
        border-radius: var(--radius-md) !important;
        margin-top: 1rem !important;
        box-shadow: var(--shadow-sm) !important;
    }
    [data-testid="stExpander"] summary {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        list-style: none;
        padding: 0.75rem 1rem !important;
        font-weight: 600;
        color: var(--clr-text-primary) !important;
        cursor: pointer;
        transition: background 0.15s ease;
    }
    [data-testid="stExpander"] summary:hover {
        background: #f8fafc !important;
    }
    [data-testid="stExpander"] summary:focus-visible {
        outline: 2px solid var(--clr-accent) !important;
        outline-offset: 2px;
    }
    [data-testid="stExpander"] [data-testid="stExpanderToggleIcon"] {
        color: var(--clr-accent) !important;
    }

    /* QW-8: DataFrame finance density */
    [data-testid="stDataFrame"] th {
        font-size: 0.68rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        color: var(--clr-text-muted) !important;
        background: var(--clr-bg) !important;
        padding: 6px 10px !important;
    }
    [data-testid="stDataFrame"] td {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.8rem !important;
        padding: 5px 10px !important;
        color: var(--clr-text-primary) !important;
    }
    [data-testid="stDataFrame"] tr:hover td {
        background: #eff6ff !important;
    }

    /* QW-9: Buttons + alert boxes */
    .stButton > button {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        border-radius: var(--radius-sm) !important;
        transition: background 0.15s ease !important;
    }
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="baseButton-primary"] {
        background: var(--clr-accent) !important;
        border: none !important;
        color: #fff !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: var(--clr-accent-hover) !important;
    }
    [data-testid="stAlert"] {
        border-radius: var(--radius-md) !important;
    }

    /* QW-10: Column gap */
    [data-testid="column"] { padding: 0 8px !important; }
    .stMarkdown br { display: block; margin: 0.25rem 0; }

    /* ═══════════════════════════════════════════
       PHASE 2 UTILITY CLASSES
    ═══════════════════════════════════════════ */

    /* Section header with step badge */
    .section-header {
        display: flex; align-items: center; gap: 12px;
        margin-bottom: 0.7rem; padding: 0.2rem 0;
    }
    .step-badge {
        font-size: .65rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: .1em; color: var(--clr-accent); background: #eff6ff;
        border: 1px solid #bfdbfe; padding: 3px 8px; border-radius: 4px;
    }
    .section-title { font-size: 1.05rem; font-weight: 700; color: var(--clr-text-primary); }

    /* Unified badge system */
    .badge {
        display: inline-block; padding: 3px 10px; border-radius: var(--radius-sm);
        font-size: .7rem; font-weight: 700; letter-spacing: .06em; text-transform: uppercase;
    }
    .badge-pass  { background: var(--clr-success-bg); color: var(--clr-success-text); border: 1px solid var(--clr-success); }
    .badge-warn  { background: var(--clr-warn-bg);    color: var(--clr-warn-text);    border: 1px solid var(--clr-warn); }
    .badge-fail  { background: var(--clr-danger-bg);  color: var(--clr-danger-text);  border: 1px solid var(--clr-danger); }
    .badge-neutral-plain { background: var(--clr-bg); color: var(--clr-text-secondary); border: 1px solid var(--clr-border-strong); }

    /* Prominent disclaimer banner */
    .disclaimer-banner {
        background: var(--clr-warn-bg);
        border: 1px solid #fcd34d;
        border-left: 4px solid var(--clr-warn);
        border-radius: var(--radius-md);
        padding: 10px 12px;
        margin-bottom: 12px;
    }
    .disclaimer-title {
        font-size: .7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .08em;
        color: var(--clr-warn-text);
        margin-bottom: 4px;
    }
    .disclaimer-text {
        font-size: .8rem;
        color: #78350f;
        line-height: 1.35;
    }

    /* Stance cards */
    .stance-card {
        background: var(--clr-surface); border: 1px solid var(--clr-border);
        border-radius: var(--radius-md); padding: 14px 16px; border-left-width: 3px;
        border-left-style: solid;
    }
    .stance-card-bull { border-left-color: var(--clr-success) !important; }
    .stance-card-bear { border-left-color: var(--clr-danger) !important; }
    .stance-card-neut { border-left-color: var(--clr-text-muted) !important; }
    .final-verdict-card {
        background: #eef4ff;
        border: 1px solid #bfdbfe;
        border-left: 4px solid var(--clr-primary);
        border-radius: var(--radius-md);
        padding: 14px 16px;
        margin: 8px 0 12px 0;
    }
    .final-verdict-title {
        font-size: .72rem;
        letter-spacing: .08em;
        text-transform: uppercase;
        color: var(--clr-primary);
        font-weight: 700;
        margin-bottom: 4px;
    }
    .final-verdict-main {
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--clr-text-primary);
    }
    .final-verdict-meta {
        margin-top: 4px;
        font-size: .85rem;
        color: var(--clr-text-secondary);
    }

    /* Chips */
    .input-chip {
        background: #eff6ff; border: 1px solid #bfdbfe; color: var(--clr-text-secondary);
        padding: 3px 10px; border-radius: 12px; font-size: .78rem;
        font-family: 'JetBrains Mono', monospace; display: inline-block;
    }
    .param-chip {
        background: var(--clr-bg); border: 1px solid var(--clr-border);
        color: var(--clr-text-secondary); padding: 3px 10px; border-radius: 12px;
        font-size: .78rem; display: inline-block;
    }

    /* Subsection header */
    .subsection-header {
        font-size: .85rem; font-weight: 700; color: var(--clr-text-primary);
        text-transform: uppercase; letter-spacing: .06em;
        padding-bottom: .5rem; border-bottom: 1px solid var(--clr-border);
        margin: 1.5rem 0 .75rem 0;
    }

    /* Sidebar data badge */
    .sidebar-data-badge {
        background: rgba(16,185,129,.12); border: 1px solid rgba(16,185,129,.3);
        border-radius: var(--radius-sm); padding: 6px 12px; text-align: center;
        font-size: .78rem; font-weight: 600; color: #6ee7b7; letter-spacing: .02em;
        margin: 8px 0;
    }

    /* Spacers */
    .spacer-sm { height: 1rem; }
    .spacer-md { height: 1.75rem; }

    /* App top bar */
    .app-topbar {
        display: grid;
        grid-template-columns: auto minmax(0, 1fr) auto;
        align-items: center;
        gap: 18px;
        padding: 12px 16px;
        margin-bottom: 1rem;
        background: rgba(255,255,255,0.88);
        border: 1px solid rgba(203, 213, 225, 0.95);
        border-radius: 18px;
        box-shadow: var(--shadow-sm);
        backdrop-filter: blur(10px);
    }
    .app-topbar-brand {
        display: flex;
        align-items: center;
        gap: 10px;
        min-width: 0;
    }
    .app-topbar-center {
        min-width: 0;
    }
    .app-topbar-nav {
        display: flex;
        align-items: center;
        justify-content: center;
        flex-wrap: wrap;
        gap: 10px;
    }
    .app-topbar-right {
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        gap: 4px;
        min-width: 0;
    }
    .app-wordmark {
        font-size: 1.1rem; font-weight: 700; color: var(--clr-text-primary); letter-spacing: -.01em;
        white-space: nowrap;
    }
    .app-version {
        font-size: .6rem; font-weight: 700; text-transform: uppercase; letter-spacing: .08em;
        color: var(--clr-accent); background: #eff6ff; border: 1px solid #bfdbfe;
        padding: 2px 6px; border-radius: 999px; vertical-align: middle;
    }
    .app-topnav-link {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-height: 40px;
        padding: 0 16px;
        border-radius: 999px;
        border: 1px solid transparent;
        background: transparent;
        color: #334155;
        font-size: 1rem;
        font-weight: 600;
        letter-spacing: -.01em;
        text-decoration: none;
        transition: background .15s ease, border-color .15s ease, color .15s ease, box-shadow .15s ease;
    }
    .app-topnav-link:hover {
        color: var(--clr-accent);
        background: #eff6ff;
        border-color: #bfdbfe;
    }
    .app-topnav-link.is-active {
        color: var(--clr-accent);
        background: #dbeafe;
        border-color: #60a5fa;
        box-shadow: inset 0 0 0 1px #93c5fd;
    }
    .app-topbar-note {
        margin-top: 7px;
        text-align: center;
        font-size: .74rem;
        color: var(--clr-text-muted);
        line-height: 1.35;
    }
    .app-topbar-time {
        font-family: 'JetBrains Mono', monospace;
        font-size: .88rem;
        color: var(--clr-text-primary);
        text-align: right;
        white-space: nowrap;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        margin-bottom: 1rem;
        padding: 0 4px 8px;
    }
    .stTabs [data-baseweb="tab"] {
        min-height: 40px;
        padding: 0 18px;
        border-radius: 999px;
        border: 1px solid transparent;
        background: transparent;
        color: #334155;
        font-size: 1rem;
        font-weight: 600;
        letter-spacing: -.01em;
        transition: background .15s ease, border-color .15s ease, color .15s ease, box-shadow .15s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--clr-accent);
        background: #eff6ff;
        border-color: #bfdbfe;
    }
    .stTabs [aria-selected="true"] {
        color: var(--clr-accent);
        background: #dbeafe;
        border-color: #60a5fa;
        box-shadow: inset 0 0 0 1px #93c5fd;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        display: none;
    }
    .stButton > button[kind="tertiary"] {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        min-height: 0 !important;
        height: auto !important;
        color: #64748b !important;
        font-size: .86rem !important;
        text-decoration: none !important;
        box-shadow: none !important;
    }
    .stButton > button[kind="tertiary"]:hover {
        background: transparent !important;
        color: var(--clr-accent);
        text-decoration: underline !important;
    }

    /* Hero KPI strip */
    .hero-strip {
        background: var(--clr-surface); border: 1px solid var(--clr-border);
        border-radius: var(--radius-lg); padding: 14px 20px;
        display: flex; align-items: center; margin-bottom: 0.9rem;
        box-shadow: var(--shadow-sm);
    }
    .hero-item { flex: 1; padding: 0 20px; border-right: 1px solid var(--clr-border); }
    .hero-item:first-child { padding-left: 0; }
    .hero-label {
        font-size: .65rem; font-weight: 600; text-transform: uppercase;
        letter-spacing: .07em; color: var(--clr-text-muted); margin-bottom: 4px;
    }
    .hero-value {
        font-size: 1.3rem; font-weight: 700; font-family: 'JetBrains Mono', monospace;
        color: var(--clr-text-primary);
    }
    .hero-divider { width: 1px; background: var(--clr-border); height: 48px; margin: 0 24px; flex-shrink: 0; }
    .hero-ticker { text-align: right; flex-shrink: 0; }
    .hero-ticker-symbol {
        display: block; font-size: 1.8rem; font-weight: 800; color: var(--clr-accent);
        font-family: 'JetBrains Mono', monospace; letter-spacing: -.02em;
    }
    .hero-ticker-source {
        font-size: .70rem; color: var(--clr-text-muted); letter-spacing: .02em;
    }

    /* Report navigation + decision strip */
    .report-nav {
        display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 1.25rem;
        padding: 10px 12px; border: 1px solid var(--clr-border);
        border-radius: var(--radius-md); background: var(--clr-surface);
        box-shadow: var(--shadow-sm);
    }
    .report-nav a {
        font-size: .72rem; font-weight: 700; letter-spacing: .05em;
        text-transform: uppercase; color: var(--clr-text-secondary);
        text-decoration: none; padding: 6px 10px; border-radius: var(--radius-sm);
        border: 1px solid var(--clr-border);
    }
    .report-nav a:hover {
        color: var(--clr-accent); border-color: #bfdbfe; background: #eff6ff;
    }
    .report-nav a.is-active {
        color: var(--clr-accent);
        border-color: #93c5fd;
        background: #dbeafe;
        box-shadow: inset 0 0 0 1px #bfdbfe;
    }
    .floating-toc-wrap {
        position: fixed;
        right: 18px;
        top: 86px;
        z-index: 999;
        display: flex;
        flex-direction: row-reverse;
        align-items: flex-start;
        gap: 8px;
        opacity: 0;
        transform: translateX(12px);
        pointer-events: none;
        transition: opacity 0.18s ease, transform 0.18s ease;
    }
    .floating-toc-wrap.is-visible {
        opacity: 1;
        transform: translateX(0);
        pointer-events: auto;
    }
    .floating-toc-toggle {
        width: 36px;
        height: 36px;
        border: 1px solid var(--clr-border);
        border-radius: 999px;
        background: var(--clr-surface);
        color: var(--clr-text-secondary);
        font-size: 1rem;
        font-weight: 700;
        line-height: 1;
        cursor: pointer;
        box-shadow: var(--shadow-sm);
    }
    .floating-toc-toggle:hover {
        color: var(--clr-accent);
        border-color: #bfdbfe;
        background: #eff6ff;
    }
    .floating-toc-wrap.is-expanded .floating-toc-toggle {
        color: var(--clr-accent);
        border-color: #93c5fd;
        background: #dbeafe;
    }
    .floating-toc {
        width: 196px;
        background: var(--clr-surface);
        border: 1px solid var(--clr-border);
        border-radius: var(--radius-md);
        box-shadow: var(--shadow-md);
        padding: 10px;
        display: none;
        flex-direction: column;
        gap: 6px;
    }
    .floating-toc-wrap.is-expanded .floating-toc {
        display: flex;
    }
    .floating-toc-title {
        font-size: .62rem;
        font-weight: 700;
        color: var(--clr-text-muted);
        text-transform: uppercase;
        letter-spacing: .08em;
        margin-bottom: 2px;
    }
    .floating-toc a {
        font-size: .66rem;
        font-weight: 700;
        letter-spacing: .05em;
        text-transform: uppercase;
        color: var(--clr-text-secondary);
        text-decoration: none;
        border: 1px solid var(--clr-border);
        border-radius: var(--radius-sm);
        background: var(--clr-bg);
        padding: 6px 8px;
    }
    .floating-toc a:hover {
        color: var(--clr-accent);
        border-color: #bfdbfe;
        background: #eff6ff;
    }
    .floating-toc a.is-active {
        color: var(--clr-accent);
        border-color: #93c5fd;
        background: #dbeafe;
        box-shadow: inset 0 0 0 1px #bfdbfe;
    }
    .decision-strip {
        background: var(--clr-surface); border: 1px solid var(--clr-border);
        border-radius: var(--radius-md); box-shadow: var(--shadow-sm);
        padding: 12px 14px; margin-bottom: 12px;
    }
    .decision-grid {
        display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 10px;
    }
    .decision-tile-label {
        font-size: .62rem; font-weight: 700; letter-spacing: .08em;
        text-transform: uppercase; color: var(--clr-text-muted);
    }
    .decision-tile-value {
        font-size: 1.05rem; font-weight: 700; color: var(--clr-text-primary);
        font-family: 'JetBrains Mono', monospace;
    }
    .confidence-strip {
        display: flex; flex-wrap: wrap; gap: 6px;
        font-size: .72rem; color: var(--clr-text-secondary);
    }
    .confidence-pill {
        background: var(--clr-bg); border: 1px solid var(--clr-border);
        border-radius: 999px; padding: 4px 10px;
    }
    .qa-chatbot-hero {
        margin: 2px 0 8px 0;
    }
    .qa-chatbot-pill {
        display: inline-block;
        padding: 9px 13px;
        border-radius: 10px;
        background: linear-gradient(180deg, #ffffff 0%, #f7f9fc 100%);
        border: 1px solid #d7deea;
        color: #1f2937;
        font-weight: 600;
        font-size: 1rem;
        letter-spacing: .02em;
        box-shadow: 0 1px 4px rgba(15, 23, 42, 0.08);
    }
    .analysis-view-boundary {
        display: none !important;
    }

    @media (max-width: 1100px) {
        .app-topbar {
            grid-template-columns: 1fr;
            gap: 12px;
        }
        .app-topbar-brand,
        .app-topbar-right {
            justify-content: space-between;
            align-items: flex-start;
        }
        .app-topbar-nav,
        .app-topbar-note {
            justify-content: flex-start;
            text-align: left;
        }
        .app-topbar-time {
            text-align: left;
        }
        .hero-strip {
            flex-wrap: wrap;
            gap: 14px;
        }
        .hero-divider {
            display: none;
        }
        .hero-ticker {
            width: 100%;
            text-align: left;
        }
    }

    @media (max-width: 820px) {
        .decision-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
    }

    /* User guide page */
    .guide-shell {
        display: flex;
        flex-direction: column;
        gap: 12px;
        margin: 10px 0 12px 0;
    }
    .guide-hero {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 16px;
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
        border: 1px solid #1d4ed8;
        border-radius: var(--radius-lg);
        padding: 16px 18px;
        color: #e2e8f0;
        box-shadow: var(--shadow-md);
    }
    .guide-kicker {
        font-size: .62rem;
        font-weight: 700;
        letter-spacing: .10em;
        text-transform: uppercase;
        color: #93c5fd;
        margin-bottom: 5px;
    }
    .guide-hero-title {
        font-size: 1.2rem;
        font-weight: 700;
        letter-spacing: -.01em;
        color: #f8fafc;
        margin-bottom: 5px;
    }
    .guide-hero-sub {
        font-size: .83rem;
        line-height: 1.4;
        color: #cbd5e1;
        max-width: 820px;
    }
    .guide-hero-chip {
        border: 1px solid rgba(147, 197, 253, 0.45);
        background: rgba(148, 163, 184, 0.15);
        color: #dbeafe;
        border-radius: 999px;
        padding: 6px 10px;
        font-size: .65rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .06em;
        white-space: nowrap;
    }
    .guide-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 12px;
    }
    .guide-card {
        background: var(--clr-surface);
        border: 1px solid var(--clr-border);
        border-radius: var(--radius-md);
        padding: 14px 16px;
        box-shadow: var(--shadow-sm);
    }
    .guide-card-priority {
        border-color: #bfdbfe;
        box-shadow: 0 0 0 1px #dbeafe;
        background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
    }
    .guide-card-title {
        font-size: .72rem;
        font-weight: 700;
        color: var(--clr-accent);
        text-transform: uppercase;
        letter-spacing: .08em;
        margin-bottom: 10px;
    }
    .guide-onboarding-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 10px;
    }
    .guide-onboarding-step {
        display: grid;
        grid-template-columns: auto 1fr;
        align-items: start;
        gap: 10px;
        border: 1px solid #dbeafe;
        background: #f8fbff;
        border-radius: var(--radius-sm);
        padding: 10px;
    }
    .guide-onboarding-num {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 20px;
        height: 20px;
        border-radius: 999px;
        background: var(--clr-accent);
        color: #ffffff;
        font-size: .66rem;
        font-weight: 700;
        line-height: 1;
        margin-top: 2px;
    }
    .guide-list {
        margin: 0;
        padding-left: 1.1rem;
        color: var(--clr-text-secondary);
        font-size: .84rem;
        line-height: 1.45;
    }
    .guide-list li {
        margin-bottom: 6px;
    }
    .guide-score-table-wrap {
        margin-top: 8px;
        overflow-x: auto;
    }
    .guide-score-table {
        width: 100%;
        border-collapse: collapse;
        font-size: .79rem;
        color: var(--clr-text-secondary);
    }
    .guide-score-table th {
        text-align: left;
        font-size: .62rem;
        font-weight: 700;
        letter-spacing: .06em;
        text-transform: uppercase;
        color: var(--clr-text-muted);
        background: #f8fafc;
        border: 1px solid var(--clr-border);
        padding: 8px 9px;
    }
    .guide-score-table td {
        border: 1px solid var(--clr-border);
        padding: 8px 9px;
        vertical-align: top;
        line-height: 1.35;
    }
    .guide-score-table tbody tr:nth-child(even) {
        background: #fcfdff;
    }
    .guide-inline-label {
        display: inline-block;
        border: 1px solid #bfdbfe;
        background: #eff6ff;
        color: #1e40af;
        border-radius: 999px;
        padding: 1px 8px;
        font-size: .74rem;
        font-weight: 600;
    }
    .guide-list-numbered li::marker {
        color: var(--clr-accent);
        font-weight: 700;
    }
    .guide-step-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 10px;
    }
    .guide-step-row {
        display: grid;
        grid-template-columns: auto 1fr;
        gap: 10px;
        align-items: start;
        border: 1px solid var(--clr-border);
        background: #f8fafc;
        border-radius: var(--radius-sm);
        padding: 9px 10px;
    }
    .guide-step-pill {
        display: inline-block;
        font-size: .62rem;
        font-weight: 700;
        letter-spacing: .08em;
        text-transform: uppercase;
        color: var(--clr-accent);
        background: #dbeafe;
        border: 1px solid #bfdbfe;
        border-radius: 999px;
        padding: 3px 8px;
        line-height: 1.4;
    }
    .guide-step-name {
        font-size: .82rem;
        font-weight: 700;
        color: var(--clr-text-primary);
        margin-bottom: 2px;
    }
    .guide-step-desc {
        font-size: .78rem;
        color: var(--clr-text-secondary);
        line-height: 1.35;
    }
    .guide-card code,
    .guide-step-desc code,
    .guide-disclaimer-card code,
    .stMarkdown code {
        display: inline-block;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: .74rem;
        color: #334155 !important;
        background: #edf2f7 !important;
        border: 1px solid #d7dfea !important;
        border-radius: 6px;
        padding: 1px 6px;
        line-height: 1.25;
    }
    .stMarkdown pre code,
    [data-testid="stCodeBlock"] code {
        color: inherit !important;
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        display: block;
    }
    .guide-disclaimer-card {
        background: #f8fafc;
        border: 1px solid var(--clr-border);
        border-radius: var(--radius-md);
        padding: 12px 14px;
    }
    .guide-disclaimer-title {
        font-size: .66rem;
        font-weight: 700;
        letter-spacing: .08em;
        text-transform: uppercase;
        color: var(--clr-text-muted);
        margin-bottom: 4px;
    }
    .guide-disclaimer-text {
        font-size: .82rem;
        color: var(--clr-text-secondary);
        line-height: 1.35;
    }

    @media (max-width: 1024px) {
        .decision-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
        .floating-toc-wrap { right: 10px; }
        .floating-toc { width: 176px; }
        .guide-step-grid { grid-template-columns: 1fr; }
    }
    @media (max-width: 640px) {
        .decision-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        .hero-strip { flex-wrap: wrap; padding: 14px; }
        .hero-item {
            min-width: calc(50% - 10px);
            border-right: none;
            border-bottom: 1px solid var(--clr-border);
            padding: 0 8px 8px 0;
            margin-bottom: 8px;
        }
        .hero-divider { display: none; }
        .hero-ticker { width: 100%; text-align: left; }
        .floating-toc-wrap { display: none; }
        .guide-grid { grid-template-columns: 1fr; }
        .guide-hero { flex-direction: column; }
        .guide-hero-chip { align-self: flex-start; }
        .guide-step-row { grid-template-columns: 1fr; gap: 6px; }
        .guide-onboarding-grid { grid-template-columns: 1fr; }
        .guide-onboarding-step { grid-template-columns: auto 1fr; }
        .app-topbar {
            align-items: flex-start;
        }
        .app-topbar-right {
            align-items: flex-end;
            gap: 2px;
        }
        .app-top-links {
            gap: 12px;
        }
    }

    /* Step progress indicator */
    .step-progress {
        display: flex; align-items: center; justify-content: center;
        gap: 0; margin-bottom: 1.5rem; padding: 12px 0;
    }
    .step-pill {
        display: flex; align-items: center; gap: 8px; padding: 6px 16px; border-radius: 20px;
        font-size: .72rem; font-weight: 700; letter-spacing: .05em; text-transform: uppercase;
    }
    .step-pill-active  { background: #eff6ff; border: 1.5px solid var(--clr-accent); color: var(--clr-accent); }
    .step-pill-inactive { background: var(--clr-bg); border: 1.5px solid var(--clr-border); color: var(--clr-text-muted); }
    .step-num {
        width: 20px; height: 20px; border-radius: 50%;
        display: flex; align-items: center; justify-content: center; font-size: .65rem; font-weight: 800;
    }
    .step-num-active   { background: var(--clr-accent); color: #fff; }
    .step-num-inactive { background: var(--clr-border); color: var(--clr-text-muted); }
    .step-connector         { flex: 1; height: 2px; background: var(--clr-border); max-width: 40px; }
    .step-connector-active  { background: var(--clr-accent); }

    /* Sidebar brand */
    .sidebar-brand {
        padding: 16px 0 20px 0;
        border-bottom: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 16px;
    }
    .sidebar-brand-logo {
        width: 32px; height: 32px; background: var(--clr-accent); border-radius: 6px;
        display: inline-flex; align-items: center; justify-content: center;
        font-size: .85rem; font-weight: 800; color: #fff;
    }
    .sidebar-brand-name {
        font-size: .95rem; font-weight: 700; color: var(--clr-sidebar-text);
        margin-left: 10px; vertical-align: middle;
    }
    .sidebar-brand-sub {
        font-size: .65rem; color: var(--clr-sidebar-muted); text-transform: uppercase;
        letter-spacing: .08em; margin-top: 6px;
    }
    .sidebar-section-label {
        font-size: .6rem; font-weight: 700; text-transform: uppercase; letter-spacing: .1em;
        color: var(--clr-sidebar-muted); margin-bottom: 8px;
    }

    /* Fix chart legend overlap */
    [data-testid="stVegaLiteChart"] { margin-bottom: 2rem; }

    /* Hide streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    /* Hide the default sidebar collapse control */
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapseButton"],
    button[kind="header"],
    [data-testid="baseButton-header"] {
        display: none !important;
        visibility: hidden !important;
        width: 0 !important;
        height: 0 !important;
        opacity: 0 !important;
        pointer-events: none !important;
    }

    /* Fixed sidebar width */
    [data-testid="stSidebar"] > div:first-child {
        width: 300px !important;
    }
</style>
""", unsafe_allow_html=True)
_boot_log("css_loaded")

# --- Initialize Session State ---
if 'financials' not in st.session_state:
    st.session_state.financials = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'quarterly_analysis' not in st.session_state:
    st.session_state.quarterly_analysis = None
if 'independent_forecast' not in st.session_state:
    st.session_state.independent_forecast = None
if 'api_key_set' not in st.session_state:
    st.session_state.api_key_set = False
if 'dcf_ui_adapter' not in st.session_state:
    st.session_state.dcf_ui_adapter = None
if 'dcf_engine_result' not in st.session_state:
    st.session_state.dcf_engine_result = None
if 'dcf_snapshot' not in st.session_state:
    st.session_state.dcf_snapshot = None
if 'show_dcf_details' not in st.session_state:
    st.session_state.show_dcf_details = False
if 'show_user_guide' not in st.session_state:
    st.session_state.show_user_guide = False
if 'show_contact_panel' not in st.session_state:
    st.session_state.show_contact_panel = False
if 'forecast_just_generated' not in st.session_state:
    st.session_state.forecast_just_generated = False
if 'ai_outlook_error' not in st.session_state:
    st.session_state.ai_outlook_error = None
if 'ui_cache' not in st.session_state:
    st.session_state.ui_cache = load_ui_cache()
    save_ui_cache(st.session_state.ui_cache)
if st.session_state.ui_cache.get("available_report_dates_version") != REPORT_DATES_CACHE_VERSION:
    st.session_state.ui_cache["available_report_dates_version"] = REPORT_DATES_CACHE_VERSION
    st.session_state.ui_cache["available_report_dates"] = {}
    save_ui_cache(st.session_state.ui_cache)
if 'ticker_library' not in st.session_state:
    st.session_state.ticker_library = _normalize_ticker_library(st.session_state.ui_cache.get("ticker_library", []))
if 'last_restore_key' not in st.session_state:
    st.session_state.last_restore_key = None
if 'cache_restore_notice' not in st.session_state:
    st.session_state.cache_restore_notice = ""
if 'financials_load_notice' not in st.session_state:
    st.session_state.financials_load_notice = ""
if 'ticker_dropdown' not in st.session_state:
    last_selected = _normalize_ticker(st.session_state.ui_cache.get("last_selected_ticker", DEFAULT_TICKER))
    library = st.session_state.ticker_library
    st.session_state.ticker_dropdown = last_selected if last_selected in library else library[0]
if 'pending_ticker_dropdown' not in st.session_state:
    st.session_state.pending_ticker_dropdown = None
if 'assumption_suggestions_loaded' not in st.session_state:
    st.session_state.assumption_suggestions_loaded = False
if 'assumption_suggestions_ticker' not in st.session_state:
    st.session_state.assumption_suggestions_ticker = None
if 'valuation_inputs_seeded_context' not in st.session_state:
    st.session_state.valuation_inputs_seeded_context = None
if 'config_num_quarters' not in st.session_state:
    existing_num_quarters = st.session_state.get("num_quarters", DEFAULT_INITIAL_QUARTERS)
    if not isinstance(existing_num_quarters, int):
        existing_num_quarters = DEFAULT_INITIAL_QUARTERS
    st.session_state.config_num_quarters = min(20, max(4, existing_num_quarters))
if 'pending_config_num_quarters' not in st.session_state:
    st.session_state.pending_config_num_quarters = None
if 'momentum_display_quarters' not in st.session_state:
    st.session_state.momentum_display_quarters = DEFAULT_INITIAL_QUARTERS
if 'pending_momentum_display_quarters' not in st.session_state:
    st.session_state.pending_momentum_display_quarters = None
if 'pending_scroll_section' not in st.session_state:
    st.session_state.pending_scroll_section = None
_boot_log("session_state_ready")

# --- Helper Functions ---
def reset_analysis():
    st.session_state.quarterly_analysis = None
    st.session_state.independent_forecast = None
    st.session_state.forecast_just_generated = False
    st.session_state.ai_outlook_error = None
    st.session_state.cache_restore_notice = ""
    st.session_state.financials_load_notice = ""
    st.session_state.last_restore_key = None
    # Reset DCF assumptions so they get re-calculated for new ticker
    st.session_state.dcf_wacc = None
    st.session_state.dcf_fcf_growth = None
    st.session_state.dcf_terminal_growth = None
    st.session_state.dcf_ui_adapter = None
    st.session_state.dcf_engine_result = None
    st.session_state.dcf_snapshot = None
    st.session_state.assumption_suggestions_loaded = False
    st.session_state.assumption_suggestions_ticker = None
    st.session_state.valuation_inputs_seeded_context = None
    configured_quarters = st.session_state.get("config_num_quarters", DEFAULT_INITIAL_QUARTERS)
    if not isinstance(configured_quarters, int):
        configured_quarters = DEFAULT_INITIAL_QUARTERS
    st.session_state.momentum_display_quarters = min(20, max(4, configured_quarters))
    st.session_state.pending_config_num_quarters = None
    st.session_state.pending_momentum_display_quarters = None

def display_stock_call(call: str):
    """Displays the stock call with clean styling."""
    call_lower = call.lower() if call else ""
    
    if "outperform" in call_lower or "above" in call_lower or "buy" in call_lower:
        st.markdown("""
            <div class="call-outperform">
                <div style="font-size: 1.5rem; margin-bottom: 4px;">▲</div>
                <div class="call-label">OUTPERFORM</div>
            </div>
        """, unsafe_allow_html=True)
    elif "underperform" in call_lower or "below" in call_lower or "sell" in call_lower:
        st.markdown("""
            <div class="call-underperform">
                <div style="font-size: 1.5rem; margin-bottom: 4px;">▼</div>
                <div class="call-label">UNDERPERFORM</div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class="call-inline">
                <div style="font-size: 1.5rem; margin-bottom: 4px;">◆</div>
                <div class="call-label">IN-LINE</div>
            </div>
        """, unsafe_allow_html=True)

def parse_price_value(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).replace(",", "").strip()
    match = re.search(r"[-+]?\d*\.?\d+", text)
    return float(match.group(0)) if match else None

def is_quota_or_rate_limit_error(error_text: str) -> bool:
    text = str(error_text or "").lower()
    markers = ["resource_exhausted", "quota", "rate limit", "too many requests", "429"]
    return any(marker in text for marker in markers)

def estimate_api_retry_time(error_text: str) -> str:
    text = str(error_text or "").lower()
    retry_seconds = None
    retry_patterns = [
        r"retry in\s+(\d+)\s*(?:seconds|secs|sec|s)\b",
        r"try again in\s+(\d+)\s*(?:seconds|secs|sec|s)\b",
        r"retry_delay[^\d]{0,20}(\d+)",
        r"seconds[\"']?\s*:\s*(\d+)",
    ]
    for pattern in retry_patterns:
        match = re.search(pattern, text)
        if match:
            try:
                retry_seconds = int(match.group(1))
                break
            except Exception:
                pass

    if retry_seconds is not None and retry_seconds > 0:
        retry_dt_utc = datetime.now(timezone.utc) + timedelta(seconds=retry_seconds)
        return retry_dt_utc.astimezone().strftime("%b %d, %Y %I:%M %p %Z")

    # Provider quota reset timing is plan-dependent; this is an explicit estimate fallback.
    try:
        pt = ZoneInfo("America/Los_Angeles")
        now_pt = datetime.now(pt)
        next_reset_pt = (now_pt + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return next_reset_pt.strftime("%b %d, %Y %I:%M %p PT (estimated)")
    except Exception:
        return "next daily reset window (estimated)"

def get_valuation_verdict(upside_pct: float):
    """Map upside/downside to verdict label, style class, and short rationale."""
    if upside_pct > 25:
        return "Significantly Undervalued", "badge-pass", "DCF >25% above market"
    if upside_pct > 10:
        return "Modestly Undervalued", "badge-pass", "DCF 10-25% above market"
    if upside_pct < -25:
        return "Significantly Overvalued", "badge-fail", "DCF >25% below market"
    if upside_pct < -10:
        return "Modestly Overvalued", "badge-fail", "DCF 10-25% below market"
    return "Fairly Valued", "badge-neutral-plain", "DCF within +/-10% of market"


def _render_valuation_verdict_section(ticker: str, dcf_ui_data: dict | None, empty_message: str) -> None:
    """Render the top-line DCF call for the active ticker."""
    if dcf_ui_data:
        if not dcf_ui_data.get("success"):
            st.error("DCF analysis failed. Review assumptions and rerun.")
            for err in dcf_ui_data.get("errors", []):
                st.error(f"• {err}")
            return

        current_price = dcf_ui_data.get("current_price", 0)
        intrinsic = dcf_ui_data.get("price_per_share", 0)
        data_quality = dcf_ui_data.get("data_quality_score", 0)
        assumptions = dcf_ui_data.get("assumptions", {})
        tv_dominance = assumptions.get("tv_dominance_pct", 0)
        terminal_method = assumptions.get("terminal_value_method", "gordon_growth")
        high_growth_profile = bool(assumptions.get("high_growth_company", False))
        cashflow_regime = assumptions.get("cashflow_regime")
        cashflow_confidence = assumptions.get("cashflow_confidence")
        discount_rate_used = assumptions.get("discount_rate_used")
        discount_rate_input = assumptions.get("discount_rate_input")
        discount_rate_label = assumptions.get("discount_rate_label")

        upside_downside = None
        verdict_label = "Pending"
        verdict_badge = "badge-neutral-plain"
        verdict_hint = "Run DCF to generate a valuation verdict."
        if current_price and current_price > 0 and intrinsic and intrinsic > 0:
            upside_downside = ((intrinsic - current_price) / current_price * 100)
            verdict_label, verdict_badge, verdict_hint = get_valuation_verdict(upside_downside)

        terminal_method_label = "Exit Multiple" if terminal_method == "exit_multiple" else "Gordon Growth"
        if terminal_method == "exit_multiple" and high_growth_profile:
            terminal_method_context = "High-Growth Adaptive"
        elif terminal_method == "exit_multiple":
            terminal_method_context = "Adaptive Fallback"
        else:
            terminal_method_context = "Default"

        cashflow_regime_label = "FCFF"
        if cashflow_regime == "approx_unlevered":
            cashflow_regime_label = "Approx FCFF Proxy"
        elif cashflow_regime == "levered_proxy":
            cashflow_regime_label = "Levered Proxy"
        confidence_label = (cashflow_confidence or "n/a").capitalize()
        if discount_rate_used is not None:
            if discount_rate_input is not None and abs(discount_rate_used - discount_rate_input) > 1e-9:
                discount_rate_text = (
                    f"{discount_rate_label or 'Discount rate'}: {discount_rate_used*100:.1f}% "
                    f"(input {discount_rate_input*100:.1f}%)"
                )
            else:
                discount_rate_text = f"{discount_rate_label or 'Discount rate'}: {discount_rate_used*100:.1f}%"
        else:
            discount_rate_text = discount_rate_label or "Discount rate: N/A"

        current_price_text = f"${current_price:.2f}" if current_price else "—"
        intrinsic_text = f"${intrinsic:.2f}" if intrinsic else "—"
        upside_text = f"{upside_downside:+.1f}%" if upside_downside is not None else "—"
        safe_ticker = html.escape(str(ticker))
        safe_current_price_text = html.escape(current_price_text)
        safe_intrinsic_text = html.escape(intrinsic_text)
        safe_upside_text = html.escape(upside_text)
        safe_verdict_label = html.escape(str(verdict_label))
        safe_terminal_method_label = html.escape(str(terminal_method_label))
        safe_terminal_method_context = html.escape(str(terminal_method_context))
        safe_cashflow_regime_label = html.escape(str(cashflow_regime_label))
        safe_confidence_label = html.escape(str(confidence_label))
        safe_discount_rate_text = html.escape(str(discount_rate_text))

        st.markdown(
            f"""
<div class="decision-strip">
  <div class="decision-grid">
    <div><div class="decision-tile-label">Ticker</div><div class="decision-tile-value">{safe_ticker}</div></div>
    <div><div class="decision-tile-label">Current Price</div><div class="decision-tile-value">{safe_current_price_text}</div></div>
    <div>
      <div class="decision-tile-label">Intrinsic Value</div>
      <div class="decision-tile-value">{safe_intrinsic_text}</div>
    </div>
    <div><div class="decision-tile-label">Upside/Downside</div><div class="decision-tile-value">{safe_upside_text}</div></div>
    <div><div class="decision-tile-label">Verdict</div><span class="badge {verdict_badge}">{safe_verdict_label}</span></div>
  </div>
</div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
<div class="confidence-strip">
  <span class="confidence-pill">Data Quality: {data_quality:.0f}/100</span>
  <span class="confidence-pill">TV Dominance: {tv_dominance:.0f}%</span>
  <span class="confidence-pill">Terminal Method: {safe_terminal_method_label} ({safe_terminal_method_context})</span>
  <span class="confidence-pill">Cash Flow Regime: {safe_cashflow_regime_label} ({safe_confidence_label})</span>
  <span class="confidence-pill">{safe_discount_rate_text}</span>
</div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(verdict_hint)
        if cashflow_regime in {"approx_unlevered", "levered_proxy"}:
            st.warning(
                "Proxy cash-flow mode is active. Intrinsic value is still produced, but should be treated as lower-confidence "
                "than a clean FCFF run."
            )
        if not dcf_ui_data.get("data_sufficient"):
            st.warning("Insufficient data quality: interpretation confidence is reduced.")
        return

    st.info(empty_message)


def _format_summary_currency(value, decimals: int = 2) -> str:
    try:
        number = float(value)
    except Exception:
        return "N/A"
    if not math.isfinite(number):
        return "N/A"
    return f"${number:,.{decimals}f}"


def _format_summary_pct(value, signed: bool = False, decimals: int = 1) -> str:
    try:
        number = float(value)
    except Exception:
        return "N/A"
    if not math.isfinite(number):
        return "N/A"
    sign = "+" if signed else ""
    return f"{number:{sign}.{decimals}f}%"


def _clean_summary_items(items, max_items: int = 3) -> list[str]:
    cleaned = []
    for item in items or []:
        text = _sanitize_ai_valuation_language(str(item or "")).strip()
        if not text or "null" in text.lower():
            continue
        cleaned.append(text)
        if len(cleaned) >= max_items:
            break
    return cleaned


def _join_summary_phrases(items: list[str]) -> str:
    cleaned = [str(item).strip().rstrip(".") for item in (items or []) if str(item).strip()]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) == 2:
        return f"{cleaned[0]} and {cleaned[1]}"
    return f"{', '.join(cleaned[:-1])}, and {cleaned[-1]}"


def _build_summary_pdf_bytes(
    *,
    ticker: str,
    company_name: str,
    as_of_label: str,
    data_source: str,
    dcf_ui_data: dict | None,
    consensus: dict | None,
    next_forecast_label: str,
    view_model: dict | None,
    wacc_source_summary: str,
    hist_data: list | None,
    growth_summary: dict | None,
    fiscal_note: str = "",
) -> bytes | None:
    if not isinstance(dcf_ui_data, dict) or not dcf_ui_data.get("success"):
        return None

    current_price = dcf_ui_data.get("current_price")
    intrinsic_value = dcf_ui_data.get("price_per_share")
    data_quality = dcf_ui_data.get("data_quality_score")
    assumptions = dcf_ui_data.get("assumptions", {}) or {}

    upside_pct = None
    if current_price and intrinsic_value:
        try:
            if float(current_price) > 0 and float(intrinsic_value) > 0:
                upside_pct = ((float(intrinsic_value) - float(current_price)) / float(current_price)) * 100.0
        except Exception:
            upside_pct = None

    verdict_label = "Pending"
    if upside_pct is not None:
        verdict_label, _, _ = get_valuation_verdict(upside_pct)
    verdict_tone = "neutral"
    if upside_pct is not None:
        if upside_pct > 10:
            verdict_tone = "positive"
        elif upside_pct < -10:
            verdict_tone = "negative"

    current_price_numeric = None
    intrinsic_value_numeric = None
    try:
        current_price_numeric = float(current_price)
    except Exception:
        pass
    try:
        intrinsic_value_numeric = float(intrinsic_value)
    except Exception:
        pass

    summary_model = view_model if isinstance(view_model, dict) else {}
    summary_text = str(summary_model.get("summary", "") or "").strip()
    drivers = _clean_summary_items(summary_model.get("drivers"), max_items=3)
    risks = _clean_summary_items(summary_model.get("risks"), max_items=3)
    key_conditional = _sanitize_ai_valuation_language(str(summary_model.get("key_conditional", "") or "")).strip()
    evidence_gaps = _clean_summary_items(summary_model.get("evidence_gaps"), max_items=1)

    title = f"{ticker} Summary Report"
    subtitle_parts = []
    if company_name and company_name != ticker:
        subtitle_parts.append(company_name)
    if as_of_label:
        subtitle_parts.append(f"As of {as_of_label}")
    subtitle = " | ".join(subtitle_parts)

    bottom_line = []
    if summary_text:
        bottom_line.append(summary_text)
    valuation_line = (
        f"Verdict: {verdict_label} | "
        f"Price {_format_summary_currency(current_price)} | "
        f"Intrinsic {_format_summary_currency(intrinsic_value)}"
    )
    if upside_pct is not None:
        valuation_line += f" | Upside/Downside {_format_summary_pct(upside_pct, signed=True)}"
    bottom_line.append(valuation_line)

    valuation_items = []
    wacc_display = assumptions.get("discount_rate_used")
    if wacc_display is not None:
        try:
            wacc_display = float(wacc_display) * 100.0
        except Exception:
            wacc_display = None
    fcf_growth = assumptions.get("fcf_growth_rate")
    if fcf_growth is not None:
        try:
            fcf_growth = float(fcf_growth) * 100.0
        except Exception:
            fcf_growth = None
    terminal_growth = assumptions.get("terminal_growth_rate")
    if terminal_growth is not None:
        try:
            terminal_growth = float(terminal_growth) * 100.0
        except Exception:
            terminal_growth = None

    valuation_items.append(
        "Assumptions: "
        f"WACC {_format_summary_pct(wacc_display)} | "
        f"FCF growth {_format_summary_pct(fcf_growth)} | "
        f"Terminal growth {_format_summary_pct(terminal_growth, decimals=2)}"
    )

    tv_dominance = assumptions.get("tv_dominance_pct")
    terminal_method = assumptions.get("terminal_value_method", "gordon_growth")
    terminal_method_label = "Exit Multiple" if terminal_method == "exit_multiple" else "Gordon Growth"
    valuation_items.append(
        f"Quality: {data_quality:.0f}/100 | TV dominance {_format_summary_pct(tv_dominance)} | "
        f"Terminal method {terminal_method_label}"
        if isinstance(data_quality, (int, float))
        else f"TV dominance {_format_summary_pct(tv_dominance)} | Terminal method {terminal_method_label}"
    )

    hero_metrics = [
        {"label": "Market Price", "value": _format_summary_currency(current_price), "tone": "neutral"},
        {"label": "Intrinsic Value", "value": _format_summary_currency(intrinsic_value), "tone": verdict_tone},
        {
            "label": "Upside / Downside",
            "value": _format_summary_pct(upside_pct, signed=True) if upside_pct is not None else "N/A",
            "tone": verdict_tone,
        },
        {"label": "Verdict", "value": verdict_label, "tone": verdict_tone},
    ]

    price_comparison = []
    if current_price_numeric is not None:
        price_comparison.append(
            {
                "label": "Market Price",
                "value": current_price_numeric,
                "display": _format_summary_currency(current_price_numeric),
                "tone": "neutral",
            }
        )
    if intrinsic_value_numeric is not None:
        price_comparison.append(
            {
                "label": "Intrinsic Value",
                "value": intrinsic_value_numeric,
                "display": _format_summary_currency(intrinsic_value_numeric),
                "tone": verdict_tone,
            }
        )
    revenue_series = []
    revenue_subtitle = ""
    hist_rows = hist_data if isinstance(hist_data, list) else []
    recent_revenue_rows = []
    for row in hist_rows:
        revenue_value = row.get("revenue") if isinstance(row, dict) else None
        if revenue_value is None:
            continue
        try:
            revenue_numeric = float(revenue_value)
        except Exception:
            continue
        recent_revenue_rows.append(
            {
                "label": str(row.get("quarter", "")) if isinstance(row, dict) else "",
                "value": revenue_numeric,
            }
        )
        if len(recent_revenue_rows) >= 6:
            break
    revenue_series = list(reversed(recent_revenue_rows))
    if isinstance(growth_summary, dict):
        avg_revenue_yoy = growth_summary.get("avg_revenue_yoy")
        if isinstance(avg_revenue_yoy, (int, float)):
            revenue_subtitle = f"Avg revenue YoY {_format_summary_pct(avg_revenue_yoy, signed=True)}"

    key_points = []
    key_points.extend(f"- {driver}" for driver in drivers)
    if key_conditional and "null" not in key_conditional.lower():
        key_points.append(f"- Decision trigger: {key_conditional}")

    risk_items = []
    risk_items.extend(f"- {risk}" for risk in risks)
    if evidence_gaps:
        risk_items.append(f"- Evidence gap: {evidence_gaps[0]}")
    if not risk_items and isinstance(tv_dominance, (int, float)) and tv_dominance >= 75:
        risk_items.append("- Terminal value dominates the DCF, so small assumption changes can move fair value materially.")

    street_items = []
    consensus_payload = consensus if isinstance(consensus, dict) else {}
    if consensus_payload and not consensus_payload.get("error"):
        next_q = consensus_payload.get("next_quarter", {}) or {}
        coverage = consensus_payload.get("analyst_coverage", {}) or {}
        targets = consensus_payload.get("price_targets", {}) or {}
        quarter_label = next_q.get("quarter_label") or next_forecast_label or "Next quarter"
        revenue_estimate = next_q.get("revenue_estimate")
        eps_estimate = next_q.get("eps_estimate")
        analyst_count = coverage.get("num_analysts")
        street_line_parts = [quarter_label]
        if revenue_estimate:
            street_line_parts.append(f"Revenue {revenue_estimate}")
        if eps_estimate:
            street_line_parts.append(f"EPS {eps_estimate}")
        if analyst_count:
            street_line_parts.append(f"Analysts {analyst_count}")
        if len(street_line_parts) > 1:
            street_items.append(" | ".join(street_line_parts))

        avg_pt = parse_price_value(targets.get("average"))
        if avg_pt is not None:
            target_line = f"Average price target {_format_summary_currency(avg_pt)}"
            if current_price:
                try:
                    pt_upside = ((avg_pt - float(current_price)) / float(current_price)) * 100.0
                    target_line += f" ({_format_summary_pct(pt_upside, signed=True)} vs current)"
                except Exception:
                    pass
            street_items.append(target_line)
            price_comparison.append(
                {
                    "label": "Street Avg PT",
                    "value": float(avg_pt),
                    "display": _format_summary_currency(avg_pt),
                    "tone": "positive" if current_price_numeric is None or avg_pt >= current_price_numeric else "negative",
                }
            )

    source_items = []
    if fiscal_note:
        source_items.append(f"Quarter convention: {fiscal_note}")
    if data_source:
        source_items.append(f"Operating data: {data_source}")
    if wacc_source_summary:
        source_items.append(f"WACC inputs: {wacc_source_summary}")
    if isinstance(consensus_payload, dict) and consensus_payload and not consensus_payload.get("error"):
        estimate_sources = []
        next_q = consensus_payload.get("next_quarter", {}) or {}
        coverage = consensus_payload.get("analyst_coverage", {}) or {}
        targets = consensus_payload.get("price_targets", {}) or {}
        for source_value in [
            next_q.get("source"),
            coverage.get("source"),
            targets.get("source"),
        ]:
            text = str(source_value or "").strip()
            if text and text not in estimate_sources:
                estimate_sources.append(text)
        if estimate_sources:
            source_items.append(f"Street data: {', '.join(estimate_sources)}")

    sections = [
        ("Bottom Line", bottom_line),
        ("Valuation", valuation_items),
    ]
    if key_points:
        sections.append(("What Matters Most", key_points))
    if risk_items:
        sections.append(("Main Risks", risk_items))
    if street_items:
        sections.append(("Street Context", street_items))
    if source_items:
        sections.append(("Core Sources", source_items))

    qualitative_items = []
    if summary_text:
        qualitative_items.append(summary_text)
    driver_sentence = _join_summary_phrases(drivers)
    if driver_sentence:
        qualitative_items.append(f"The call is supported by {driver_sentence}.")
    risk_sentence = _join_summary_phrases(risks)
    if risk_sentence:
        qualitative_items.append(f"The main risks to the view are {risk_sentence}.")
    if evidence_gaps:
        qualitative_items.append(f"A remaining evidence gap is {evidence_gaps[0].rstrip('.')}.")
    if key_conditional and "null" not in key_conditional.lower():
        qualitative_items.append(f"The view would need to be revisited if {key_conditional.rstrip('.')}.")
    if qualitative_items:
        sections.append(("Why This Verdict", qualitative_items))

    outlook_payload = {}
    if summary_model:
        outlook_payload = {
            "short_stance": summary_model.get("short_stance", "Neutral"),
            "fund_outlook": summary_model.get("fund_outlook", "Stable"),
            "stock_outlook": summary_model.get("stock_outlook", "Neutral"),
            "stock_horizon": summary_model.get("stock_horizon", ""),
            "stock_conviction": summary_model.get("stock_conviction", ""),
            "summary": summary_text,
            "key_conditional": key_conditional,
        }

    return build_summary_pdf(
        title=title,
        subtitle=subtitle,
        sections=sections,
        footer="For research use only. Assumption-sensitive output, not investment advice.",
        hero_metrics=hero_metrics,
        price_comparison=price_comparison,
        revenue_series=revenue_series,
        revenue_subtitle=revenue_subtitle,
        outlook=outlook_payload,
    )

# --- Sidebar Toggle State ---
if "sidebar_visible" not in st.session_state:
    st.session_state.sidebar_visible = True
_boot_log("sidebar_toggle_ready")

# Floating button to open sidebar when closed
if not st.session_state.sidebar_visible:
    st.markdown("""
        <style>
            [data-testid="stSidebar"] { display: none !important; }
        </style>
    """, unsafe_allow_html=True)
    
    # Show a floating open button in the corner
    if st.button("☰ Menu", key="open_sidebar"):
        st.session_state.sidebar_visible = True
        st.rerun()

# --- Sidebar ---
_boot_log("sidebar_enter")
with st.sidebar:
    # Close button at top of sidebar
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("✕", key="close_sidebar", help="Close sidebar"):
            st.session_state.sidebar_visible = False
            st.rerun()
    
    # Sidebar brand header
    st.markdown("""
        <div class="sidebar-brand">
            <div>
                <span class="sidebar-brand-logo">AC</span>
                <span class="sidebar-brand-name">Analyst Co-Pilot</span>
            </div>
            <div class="sidebar-brand-sub">Equity Research</div>
        </div>
        <div class="sidebar-section-label">Configuration</div>
    """, unsafe_allow_html=True)

    # Auto-load API key from .env (no UI shown)
    env_api_key = os.environ.get("GEMINI_API_KEY", "")
    if env_api_key:
        st.session_state.api_key_set = True

    # Persistent ticker picker: MAG7 defaults + learned custom tickers
    st.session_state.ticker_library = _normalize_ticker_library(
        st.session_state.ui_cache.get("ticker_library", st.session_state.ticker_library)
    )
    ticker_options = st.session_state.ticker_library

    # Apply deferred widget updates before widget instantiation (Streamlit-safe).
    pending_dropdown = _normalize_ticker(st.session_state.get("pending_ticker_dropdown"))
    if pending_dropdown and pending_dropdown in ticker_options:
        st.session_state.ticker_dropdown = pending_dropdown
    st.session_state.pending_ticker_dropdown = None

    current_ticker_value = _normalize_ticker(st.session_state.get("ticker_dropdown", ""))
    if current_ticker_value in ticker_options:
        st.session_state.ticker_dropdown = current_ticker_value
    elif _is_valid_ticker_format(current_ticker_value):
        # Keep user-typed valid symbols even before they are persisted in library.
        st.session_state.ticker_dropdown = current_ticker_value
    else:
        st.session_state.ticker_dropdown = ticker_options[0]

    selected_ticker = st.selectbox(
        "Stock Ticker",
        options=ticker_options,
        key="ticker_dropdown",
        accept_new_options=True,
        help="Type to search saved/default tickers or enter any new symbol (e.g., MS).",
    )

    ticker = _normalize_ticker(selected_ticker)
    _boot_log(f"sidebar_ticker_selected:{ticker or 'NONE'}")
    if ticker and _is_valid_ticker_format(ticker) and ticker not in ticker_options:
        _upsert_ticker_in_library(ticker)
        st.session_state.pending_ticker_dropdown = ticker
    ticker_valid = _is_valid_ticker_format(ticker)

    # Date-fetch state
    if "available_dates" not in st.session_state:
        st.session_state.available_dates = []
    if "available_dates_ticker" not in st.session_state:
        st.session_state.available_dates_ticker = None
    if "selected_end_date" not in st.session_state:
        st.session_state.selected_end_date = None
    if "report_dates_hint" not in st.session_state:
        st.session_state.report_dates_hint = ""

    # Load report-date state whenever ticker changes.
    if ticker_valid and ticker != st.session_state.available_dates_ticker:
        st.session_state.available_dates_ticker = ticker
        persisted_dates = _get_persisted_report_dates(ticker)
        if persisted_dates:
            dates = persisted_dates
            persisted_values = [d.get("value") for d in dates if isinstance(d, dict)]
            if st.session_state.selected_end_date not in persisted_values:
                st.session_state.selected_end_date = dates[0]["value"]
            st.session_state.report_dates_hint = ""
        else:
            dates = []
            st.session_state.selected_end_date = None
            st.session_state.report_dates_hint = (
                "Report dates are not loaded yet. Click Load Data or Refresh Available Dates to fetch them."
            )
        st.session_state.available_dates = dates
    elif not ticker_valid:
        st.session_state.available_dates_ticker = None
        st.session_state.available_dates = []
        st.session_state.selected_end_date = None
        st.session_state.report_dates_hint = ""
    _boot_log("sidebar_dates_ready")

    available_dates = st.session_state.available_dates
    selected_end_date = st.session_state.selected_end_date

    def _derive_default_context_quarters(date_rows, end_date_value) -> int:
        """Default to a focused 7-quarter window; users can expand with Quarter Rail."""
        if not isinstance(date_rows, list) or not date_rows:
            return DEFAULT_INITIAL_QUARTERS
        date_values = [d.get("value") for d in date_rows if isinstance(d, dict) and d.get("value")]
        if not date_values:
            return DEFAULT_INITIAL_QUARTERS
        if end_date_value in date_values:
            selected_idx = date_values.index(end_date_value)
            remaining = len(date_values) - selected_idx
        else:
            remaining = len(date_values)
        preferred = min(int(remaining), DEFAULT_INITIAL_QUARTERS)
        return min(20, max(4, preferred))

    if available_dates:
        latest_date = available_dates[0]["display"]
        safe_latest_date = html.escape(str(latest_date))
        st.markdown(f"""
            <div class="sidebar-data-badge">Latest Data: {safe_latest_date}</div>
        """, unsafe_allow_html=True)
    elif ticker_valid:
        st.caption(st.session_state.report_dates_hint or "No report dates loaded yet.")
    
    # Analysis Period selection
    st.markdown("**Analysis Period**")
    
    # Single selectbox for ending report date - only shows ACTUAL available dates
    if available_dates:
        date_options = [d["display"] for d in available_dates]
        date_values = [d["value"] for d in available_dates]
        default_idx = 0
        if st.session_state.selected_end_date in date_values:
            default_idx = date_values.index(st.session_state.selected_end_date)
        else:
            st.session_state.selected_end_date = date_values[0]

        selected_display = st.selectbox(
            "Select Ending Report",
            options=date_options,
            index=default_idx,
            help=f"Select the most recent quarter to include. {len(available_dates)} reports available."
        )

        # Get the corresponding ISO date value
        selected_idx = date_options.index(selected_display)
        selected_end_date = date_values[selected_idx]
        st.session_state.selected_end_date = selected_end_date
    else:
        st.selectbox(
            "Select Ending Report",
            options=["No report dates available"],
            disabled=True
        )
        selected_end_date = None

    pending_config_quarters = st.session_state.get("pending_config_num_quarters")
    if isinstance(pending_config_quarters, int):
        st.session_state.config_num_quarters = min(20, max(4, int(pending_config_quarters)))
    st.session_state.pending_config_num_quarters = None

    # Sidebar quarter slider removed: default context depth is 7 quarters (expand via Quarter Rail).
    num_quarters = _derive_default_context_quarters(available_dates, selected_end_date)

    active_loaded_ticker = _normalize_ticker(st.session_state.get("ticker", ""))
    loaded_end_date = st.session_state.get("end_date")
    if active_loaded_ticker != ticker or loaded_end_date != selected_end_date:
        st.session_state.config_num_quarters = num_quarters

    if ticker_valid:
        if st.button(
            "Refresh Available Dates",
            key="refresh_available_reports",
            help="Manually re-fetch quarter-end report dates for this ticker."
        ):
            with st.spinner(f"Refreshing available reports for {ticker}..."):
                refreshed_dates = _load_report_dates_for_ticker(ticker, force_refresh=True)
            st.session_state.available_dates_ticker = ticker
            st.session_state.available_dates = refreshed_dates
            if refreshed_dates:
                st.session_state.selected_end_date = refreshed_dates[0]["value"]
                st.session_state.report_dates_hint = ""
            else:
                st.session_state.selected_end_date = None
                st.session_state.report_dates_hint = (
                    f"No report dates returned within {AVAILABLE_DATES_TIMEOUT_SECONDS}s. "
                    "Please try Refresh Available Dates again."
                )
            st.rerun()

    # Best-effort immediate restore for same active ticker/context
    active_ticker = _normalize_ticker(st.session_state.get("ticker", ""))
    if ticker_valid and selected_end_date:
        context_key = build_context_key(ticker, selected_end_date, num_quarters)
        loaded_num_quarters = st.session_state.get("num_quarters")
        same_loaded_context = (
            active_ticker == ticker
            and loaded_end_date == selected_end_date
            and loaded_num_quarters == num_quarters
        )
        can_restore_now = st.session_state.quarterly_analysis is None or same_loaded_context
        if can_restore_now and st.session_state.get("last_restore_key") != context_key:
            restored = _restore_cached_results_for_context(ticker, selected_end_date, num_quarters)
            if restored["dcf"] or restored["ai"]:
                restored_parts = []
                if restored["dcf"]:
                    restored_parts.append("DCF")
                if restored["ai"]:
                    restored_parts.append("AI outlook")
                st.session_state.cache_restore_notice = f"Restored cached {' + '.join(restored_parts)} for {ticker} ({selected_end_date}, {num_quarters}Q)."
            else:
                st.session_state.cache_restore_notice = ""
                st.session_state.last_restore_key = None
        elif st.session_state.get("last_restore_key") == context_key:
            # no-op; already restored for this exact context in current session
            pass
    else:
        st.session_state.cache_restore_notice = ""
        st.session_state.last_restore_key = None
    
    if st.button("Load Data", type="primary", use_container_width=True):
        # Loading a new analysis context should return to the main analysis view.
        st.session_state.show_contact_panel = False
        st.session_state.show_user_guide = False
        _clear_view_query_param()
        if not st.session_state.api_key_set:
            st.error("API key not found. Add `GEMINI_API_KEY=your_key` to `.env` file and restart.")
        elif not ticker_valid:
            st.warning("Please enter/select a valid ticker symbol.")
        else:
            if not selected_end_date:
                with st.spinner(f"Fetching available reports for {ticker}..."):
                    dates = _load_report_dates_for_ticker(ticker)
                st.session_state.available_dates = dates if isinstance(dates, list) else []
                if st.session_state.available_dates:
                    st.session_state.selected_end_date = st.session_state.available_dates[0]["value"]
                    selected_end_date = st.session_state.selected_end_date
                    st.session_state.report_dates_hint = ""
                else:
                    st.session_state.selected_end_date = None
                    selected_end_date = None
                    st.session_state.report_dates_hint = (
                        f"No report dates returned within {AVAILABLE_DATES_TIMEOUT_SECONDS}s. "
                        "Use Refresh Available Dates to try again."
                    )

            if not selected_end_date:
                st.warning("Please select a valid ending report date first.")
            else:
                num_quarters = _derive_default_context_quarters(
                    st.session_state.get("available_dates", []),
                    selected_end_date,
                )
                with st.spinner(f"Loading {ticker}..."):
                    inc, bal, cf, qcf, financials_source, financials_warning = cached_financials(ticker)
                    analysis = cached_quarterly_analysis(ticker, num_quarters, selected_end_date)
                    quarterly_data = analysis.get("historical_trends", {}).get("quarterly_data", [])
                    analysis_errors = analysis.get("errors", []) if isinstance(analysis, dict) else []

                    has_statement_data = not inc.empty
                    has_quarterly_data = bool(quarterly_data)

                    if has_statement_data or has_quarterly_data:
                        st.session_state.financials = {"income": inc, "balance": bal, "cashflow": cf, "quarterly_cashflow": qcf}
                        st.session_state.ticker = ticker
                        st.session_state.metrics = calculate_metrics(inc, bal) if has_statement_data else {}
                        st.session_state.config_num_quarters = num_quarters
                        st.session_state.num_quarters = num_quarters
                        st.session_state.end_date = selected_end_date
                        reset_analysis()

                        # Persist last selected ticker
                        cache = st.session_state.get("ui_cache", _default_ui_cache())
                        cache["last_selected_ticker"] = ticker
                        st.session_state.ui_cache = cache
                        save_ui_cache(cache)

                        st.session_state.quarterly_analysis = analysis
                        st.session_state.momentum_display_quarters = min(
                            max(1, len(quarterly_data)),
                            max(4, num_quarters)
                        ) if quarterly_data else max(4, num_quarters)
                        st.session_state.comprehensive_analysis = calculate_comprehensive_analysis(
                            inc,
                            bal,
                            quarterly_data,
                            ticker,
                            cf,
                            qcf
                        ) if has_statement_data else {}

                        partial_notice = ""
                        if not has_statement_data and has_quarterly_data:
                            partial_notice = (
                                "Loaded quarterly analysis, but annual financial statements were unavailable "
                                f"from {financials_source or 'the upstream provider'}. Some metrics and DuPont-style views may be limited."
                            )
                        elif financials_warning and has_statement_data:
                            partial_notice = str(financials_warning)
                        st.session_state.financials_load_notice = partial_notice

                        # Restore cached per-context DCF / AI outputs (if available)
                        restored = _restore_cached_results_for_context(ticker, selected_end_date, num_quarters)
                        if restored["dcf"] or restored["ai"]:
                            restored_parts = []
                            if restored["dcf"]:
                                restored_parts.append("DCF")
                            if restored["ai"]:
                                restored_parts.append("AI outlook")
                            restore_message = f"Restored cached {' + '.join(restored_parts)} for {ticker} ({selected_end_date}, {num_quarters}Q)."
                            st.session_state.cache_restore_notice = restore_message
                        else:
                            st.session_state.cache_restore_notice = ""
                            st.session_state.last_restore_key = None

                        # Show what was loaded
                        most_recent = analysis.get("historical_trends", {}).get("most_recent_quarter", {})
                        next_q = analysis.get("next_forecast_quarter", {})
                        if most_recent.get("label"):
                            st.success(f"Loaded {ticker} through {most_recent.get('label')}")
                            if next_q.get("label"):
                                st.info(f"Forecasting: {next_q.get('label')}")
                        else:
                            st.success(f"Loaded {ticker}")
                    else:
                        error_parts = []
                        if financials_warning:
                            error_parts.append(str(financials_warning))
                        if analysis_errors:
                            error_parts.append(str(analysis_errors[0]))
                        if financials_source and financials_source != "Error":
                            error_parts.append(f"Statement source attempted: {financials_source}.")
                        error_message = " ".join(part for part in error_parts if part).strip() or "Failed to fetch data."
                        st.error(error_message)

    st.divider()
    st.markdown('<div class="sidebar-section-label">Help</div>', unsafe_allow_html=True)
    help_col1, help_col2 = st.columns(2)
    with help_col1:
        if st.button("User Guide", use_container_width=True, key="open_user_guide_sidebar"):
            st.session_state.show_user_guide = True
            st.session_state.show_contact_panel = False
            st.rerun()
    with help_col2:
        if st.button("Contact", use_container_width=True, key="open_contact_sidebar"):
            st.session_state.show_user_guide = False
            st.session_state.show_contact_panel = True
            st.rerun()
    
    # Clear cache button
    st.divider()
    if st.button("🔄 Clear Cache", help="Force refresh API data"):
        st.cache_data.clear()
        reset_analysis()
        st.success("Cache cleared! Click 'Load Data' to fetch fresh data.")
_boot_log("sidebar_exit")



# --- Main Interface ---
if st.session_state.get("show_contact_panel"):
    st.session_state.show_user_guide = False
if st.session_state.get("show_user_guide"):
    _show_user_guide_page()
    st.stop()
if st.session_state.get("show_contact_panel"):
    _show_contact_page()
    st.stop()

topbar_timestamp = datetime.now().astimezone().strftime("%m/%d/%Y %H:%M %Z")
_render_analysis_topbar(topbar_timestamp)
_boot_log("topbar_ready")

dashboard_tab, deep_dive_tab, compare_tab, reports_tab = st.tabs(
    [view_meta["label"] for view_meta in ANALYSIS_VIEWS.values()]
)

if st.session_state.quarterly_analysis:
    _boot_log("analysis_branch_start")
    analysis = st.session_state.quarterly_analysis
    ticker = st.session_state.ticker
    most_recent = analysis.get("historical_trends", {}).get("most_recent_quarter", {})
    next_forecast = analysis.get("next_forecast_quarter", {})
    data_source = analysis.get("data_source", "Unknown")
    warning = analysis.get("warning")
    hist_data = analysis.get("historical_trends", {}).get("quarterly_data", [])
    data_coverage = analysis.get("historical_trends", {}).get("data_coverage", {})
    source_diagnostics = analysis.get("historical_trends", {}).get("source_diagnostics", {})
    growth_summary = analysis.get("growth_rates", {}).get("summary", {})
    seasonality_info = analysis.get("growth_rates", {}).get("seasonality", {})
    growth_detail = analysis.get("growth_rates", {}).get("detailed", [])
    comp_analysis = st.session_state.get("comprehensive_analysis", {})
    consensus = analysis.get("consensus_estimates", {})
    next_forecast_label = next_forecast.get("label", "Next Quarter")
    dcf_ui = st.session_state.get("dcf_ui_adapter")
    dcf_ui_data = dcf_ui.get_ui_data() if dcf_ui else None
    consensus_next_q = consensus.get("next_quarter", {}) if isinstance(consensus, dict) else {}
    consensus_coverage = consensus.get("analyst_coverage", {}) if isinstance(consensus, dict) else {}
    consensus_targets = consensus.get("price_targets", {}) if isinstance(consensus, dict) else {}
    fiscal_calendar = analysis.get("fiscal_calendar", {}) if isinstance(analysis.get("fiscal_calendar", {}), dict) else {}
    fiscal_note = str(fiscal_calendar.get("note", "") or "").strip()
    fiscal_note_short = str(fiscal_calendar.get("short_note", "") or fiscal_note).strip()
    fiscal_example = str(fiscal_calendar.get("current_period_example", "") or "").strip()

    st.markdown("""
<div class="disclaimer-banner">
  <div class="disclaimer-title">Disclaimer</div>
  <div class="disclaimer-text">AI-generated outputs may contain mistakes. This is not investment advice. Results are highly dependent on assumptions and input data quality.</div>
</div>
    """, unsafe_allow_html=True)

    financials_load_notice = str(st.session_state.get("financials_load_notice", "") or "").strip()
    if financials_load_notice:
        st.warning(financials_load_notice)

    # Top context strip
    _market_data = analysis.get("market_data", {})
    if not isinstance(_market_data, dict):
        _market_data = {}

    def _positive_float(value):
        try:
            if value is None:
                return None
            num = float(value)
            if not math.isfinite(num) or num <= 0:
                return None
            return num
        except Exception:
            return None

    _price = _positive_float(_market_data.get("current_price"))
    _mcap = _positive_float(_market_data.get("market_cap"))
    _pe = _positive_float(_market_data.get("pe_ratio"))
    _as_of = most_recent.get("label", "—")

    _snapshot_for_name = st.session_state.get("dcf_snapshot")
    _company_name_raw = getattr(_snapshot_for_name, "company_name", None) if _snapshot_for_name is not None else None
    _company_name = _company_name_raw.strip() if isinstance(_company_name_raw, str) else ""

    # Fallback hierarchy for missing market strip values:
    # quarterly analysis -> restored DCF snapshot -> fresh snapshot lookup.
    _snapshot_for_strip = _snapshot_for_name
    if (_price is None or _mcap is None or not _company_name) and _snapshot_for_strip is None:
        _snapshot_for_strip = cached_financial_snapshot(
            ticker,
            suggestion_algo_version=SNAPSHOT_SUGGESTION_VERSION,
        )

    if _snapshot_for_strip is not None:
        _snap_price = _positive_float(getattr(getattr(_snapshot_for_strip, "price", None), "value", None))
        _snap_mcap = _positive_float(getattr(getattr(_snapshot_for_strip, "market_cap", None), "value", None))
        _snap_shares = _positive_float(getattr(getattr(_snapshot_for_strip, "shares_outstanding", None), "value", None))

        if _price is None:
            _price = _snap_price
        if _mcap is None:
            _mcap = _snap_mcap
            if _mcap is None and _price is not None and _snap_shares is not None:
                _mcap = _price * _snap_shares
        if not _company_name:
            _snap_company_name = getattr(_snapshot_for_strip, "company_name", None)
            if isinstance(_snap_company_name, str):
                _company_name = _snap_company_name.strip()

    if not _company_name:
        _company_name = cached_company_name(ticker)

    _price_str = f"${_price:,.2f}" if _price is not None else "—"
    _mcap_str = f"${_mcap/1e9:.1f}B" if _mcap is not None else "—"
    _show_pe = _pe is not None and _pe > 0
    _pe_str = f"{_pe:.1f}x" if _show_pe else None
    _company_name_display = _company_name if _company_name else ticker

    _hero_tiles = [
        ("Price", _price_str, None),
        ("Market Cap", _mcap_str, None),
    ]
    if _show_pe:
        _hero_tiles.append(("P/E Ratio", _pe_str, None))
    _hero_tiles.append(("As Of", _as_of, "font-size:1rem"))

    _hero_items_html = []
    for idx, (label, value, value_style) in enumerate(_hero_tiles):
        style = ' style="border-right:none"' if idx == len(_hero_tiles) - 1 else ""
        value_style_html = f' style="{value_style}"' if value_style else ""
        safe_label = html.escape(str(label))
        safe_value = html.escape(str(value))
        _hero_items_html.append(
            f'<div class="hero-item"{style}><div class="hero-label">{safe_label}</div>'
            f'<div class="hero-value"{value_style_html}>{safe_value}</div></div>'
        )
    _hero_items_html = "".join(_hero_items_html)
    safe_ticker = html.escape(str(ticker))
    safe_company_name = html.escape(str(_company_name_display))

    hero_strip_html = f"""
<div class="hero-strip">
  {_hero_items_html}
  <div class="hero-divider"></div>
  <div class="hero-ticker">
    <span class="hero-ticker-symbol">{safe_ticker}</span>
    <span class="hero-ticker-source">{safe_company_name}</span>
  </div>
</div>
    """

    # Show DCF Details page if requested
    if st.session_state.get("show_dcf_details") and dcf_ui:
        _show_dcf_details_page()
        st.stop()

    with dashboard_tab:
        st.markdown(hero_strip_html, unsafe_allow_html=True)
        if fiscal_note:
            st.caption(f"Quarter convention: {fiscal_note}")

    with deep_dive_tab:
        # SECTION B: Investment Verdict
        st.markdown('<div id="verdict"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header"><span class="step-badge">Verdict</span><span class="section-title">Investment Verdict</span></div>', unsafe_allow_html=True)
        st.caption("This verdict updates after you rerun the valuation with the assumptions below.")
        _render_valuation_verdict_section(
            ticker,
            dcf_ui_data,
            empty_message="Run DCF Analysis below to generate the investment verdict.",
        )

        # SECTION C: Valuation Drivers
        st.markdown('<div id="valuation"></div>', unsafe_allow_html=True)
        _render_pending_scroll_to_section("valuation")
        st.markdown("---")
        st.markdown('<div class="section-header"><span class="step-badge">Step 03</span><span class="section-title">Valuation Drivers</span></div>', unsafe_allow_html=True)
        st.caption("Adjust assumptions, rerun the model, and review the valuation outputs below.")
        snapshot_for_suggestions = None
        _active_dcf_snapshot = st.session_state.get("dcf_snapshot")
        _active_dcf_ticker = _normalize_ticker(getattr(_active_dcf_snapshot, "ticker", "")) if _active_dcf_snapshot is not None else ""
        if _active_dcf_snapshot is not None and _active_dcf_ticker == ticker:
            snapshot_for_suggestions = _active_dcf_snapshot
        else:
            snapshot_for_suggestions = cached_financial_snapshot(
                ticker,
                suggestion_algo_version=SNAPSHOT_SUGGESTION_VERSION,
            )
        st.caption("Suggested assumptions are auto-loaded for the active ticker.")

        suggested_wacc = 9.0
        suggested_fcf_growth = 8.0
        suggested_fcf_reliability = None
        suggested_fcf_period_type = None
        raw_suggested_wacc = None
        wacc_was_bounded = False
        wacc_bound_reason = ""
        wacc_source_summary = ""
        if snapshot_for_suggestions:
            if snapshot_for_suggestions.suggested_wacc.value:
                suggested_wacc = round(snapshot_for_suggestions.suggested_wacc.value * 100, 1)
                wacc_components = getattr(snapshot_for_suggestions, "wacc_components", {}) or {}
                raw_val = wacc_components.get("raw_suggested_wacc")
                if raw_val is not None:
                    try:
                        raw_suggested_wacc = float(raw_val) * 100.0
                    except Exception:
                        raw_suggested_wacc = None
                wacc_was_bounded = bool(wacc_components.get("wacc_was_bounded", False))
                wacc_bound_reason = str(wacc_components.get("wacc_bound_reason", "") or "")
                rf_source = str(wacc_components.get("rf_source") or getattr(snapshot_for_suggestions, "rf_source", "") or "").strip()
                damodaran_date = str(wacc_components.get("damodaran_date", "") or "").strip()
                beta_source_path = str(getattr(snapshot_for_suggestions.beta, "source_path", "") or "").lower()
                rf_label = ""
                if rf_source.startswith("^TNX live"):
                    rf_label = rf_source.replace(" live", "")
                elif rf_source:
                    rf_label = rf_source
                beta_label = "beta Yahoo Finance" if "yf.ticker.info['beta']" in beta_source_path else "beta source"
                source_parts = []
                if rf_label:
                    source_parts.append(f"Rf {rf_label}")
                source_parts.append(f"ERP Damodaran ({damodaran_date or 'latest'})")
                if getattr(snapshot_for_suggestions.beta, "value", None):
                    source_parts.append(beta_label)
                wacc_source_summary = " · ".join(source_parts)
            if snapshot_for_suggestions.suggested_fcf_growth.value is not None:
                suggested_fcf_growth = round(snapshot_for_suggestions.suggested_fcf_growth.value * 100, 1)
                suggested_fcf_reliability = snapshot_for_suggestions.suggested_fcf_growth.reliability_score
                suggested_fcf_period_type = snapshot_for_suggestions.suggested_fcf_growth.period_type

        # Seed sliders from suggestions once per loaded valuation context (ticker + selected end date + quarters).
        valuation_end_date = st.session_state.get("end_date") or st.session_state.get("selected_end_date") or "latest"
        valuation_num_quarters = (
            st.session_state.get("num_quarters")
            or st.session_state.get("config_num_quarters")
            or DEFAULT_INITIAL_QUARTERS
        )
        valuation_context_key = build_context_key(ticker, valuation_end_date, int(valuation_num_quarters))
        wacc_slider_key = f"wacc_slider_{ticker}"
        fcf_slider_key = f"fcf_growth_slider_{ticker}"
        terminal_growth_key = f"terminal_growth_slider_{ticker}"

        if st.session_state.get("valuation_inputs_seeded_context") != valuation_context_key:
            seeded_fcf_growth = suggested_fcf_growth
            if suggested_fcf_reliability is not None and suggested_fcf_reliability < 65:
                seeded_fcf_growth = 8.0
            st.session_state[wacc_slider_key] = float(min(WACC_SLIDER_MAX_PCT, max(WACC_SLIDER_MIN_PCT, suggested_wacc)))
            st.session_state[fcf_slider_key] = float(min(25.0, max(0.0, seeded_fcf_growth)))
            st.session_state[terminal_growth_key] = 3.0
            st.session_state.valuation_inputs_seeded_context = valuation_context_key

        default_wacc = float(st.session_state.get(wacc_slider_key, suggested_wacc))
        default_wacc = max(WACC_SLIDER_MIN_PCT, min(WACC_SLIDER_MAX_PCT, default_wacc))
        default_fcf_growth = float(st.session_state.get(fcf_slider_key, suggested_fcf_growth))
        default_fcf_growth = max(0.0, min(25.0, default_fcf_growth))
        default_terminal_growth = float(st.session_state.get(terminal_growth_key, 3.0))

        col_wacc, col_growth, col_terminal = st.columns(3)
        with col_wacc:
            user_wacc = st.slider(
                "WACC / Discount Rate (%)",
                min_value=WACC_SLIDER_MIN_PCT,
                max_value=WACC_SLIDER_MAX_PCT,
                step=0.1,
                format="%.1f",
                key=wacc_slider_key,
                help=(
                    "Forward discount-rate assumption used in DCF. "
                    "View DCF Details to see full component trace (Re, Rd, weights, and sources)."
                ),
                **_widget_value_kwargs(wacc_slider_key, default_wacc),
            )
            if snapshot_for_suggestions and snapshot_for_suggestions.suggested_wacc.value:
                beta_val = snapshot_for_suggestions.beta.value
                rf_source = getattr(snapshot_for_suggestions, "rf_source", "^TNX")
                wacc_components = getattr(snapshot_for_suggestions, "wacc_components", {}) or {}
                we = wacc_components.get("equity_weight")
                wd = wacc_components.get("debt_weight")
                cost_of_equity_rate = wacc_components.get("cost_of_equity")
                rd = wacc_components.get("cost_of_debt_pre_tax")
                tax_rate = wacc_components.get("tax_rate")
                beta_text = f", beta {beta_val:.2f}" if beta_val else ""
                if all(v is not None for v in [we, wd, cost_of_equity_rate, rd, tax_rate]):
                    inputs_line = (
                        f"We {we*100:.0f}% / Wd {wd*100:.0f}% | "
                        f"Re {cost_of_equity_rate*100:.1f}% / Rd {rd*100:.1f}% / T {tax_rate*100:.1f}%"
                    )
                else:
                    inputs_line = f"Rf {rf_source}{beta_text}"
                suggested_text = f"Suggested WACC: {suggested_wacc:.1f}%"
                if raw_suggested_wacc is not None:
                    suggested_text = f"Suggested WACC: {suggested_wacc:.1f}%"
                st.caption(f"{suggested_text} | {inputs_line}")
                if wacc_source_summary:
                    st.caption(f"Sources: {wacc_source_summary}")
                if wacc_was_bounded and wacc_bound_reason:
                    st.caption(f"Guardrail applied: {wacc_bound_reason}")
                if we is not None and wd is not None and we <= 0.01 and wd >= 0.99:
                    st.caption("Capital structure signal: equity weight is near 0% (often means market cap/equity data is missing for this run).")

        with col_growth:
            user_fcf_growth = st.slider(
                "FCF Growth Rate (%)",
                min_value=0.0,
                max_value=25.0,
                step=0.1,
                format="%.1f",
                key=fcf_slider_key,
                help="Annual free-cash-flow growth for projection period.",
                **_widget_value_kwargs(fcf_slider_key, default_fcf_growth),
            )
            if snapshot_for_suggestions and snapshot_for_suggestions.suggested_fcf_growth.value is not None:
                source_label = "fallback"
                if suggested_fcf_period_type == "forward_analyst_consensus":
                    source_label = "analyst consensus"
                elif suggested_fcf_period_type == "forward_analyst_long_term":
                    source_label = "analyst LT only"
                elif suggested_fcf_period_type == "trailing_historical":
                    source_label = "trailing historical"
                elif suggested_fcf_period_type == "calculated_fallback":
                    source_label = "historical fallback"

                quality_text = f", quality {suggested_fcf_reliability}/100" if suggested_fcf_reliability is not None else ""
                st.caption(f"Suggested: {suggested_fcf_growth:.1f}% ({source_label}{quality_text})")
                if suggested_fcf_reliability is not None and suggested_fcf_reliability < 65:
                    st.caption("Default set to 8.0% because suggestion quality is low.")

        with col_terminal:
            terminal_growth_min = 0.0
            terminal_growth_max = max(0.5, min(6.0, round(user_wacc - 0.5, 1)))
            if terminal_growth_key in st.session_state:
                st.session_state[terminal_growth_key] = min(
                    max(st.session_state[terminal_growth_key], terminal_growth_min),
                    terminal_growth_max
                )
            terminal_growth_default = min(max(default_terminal_growth, terminal_growth_min), terminal_growth_max)
            user_terminal_growth = st.slider(
                "Terminal Growth g (%)",
                min_value=terminal_growth_min,
                max_value=terminal_growth_max,
                step=0.1,
                key=terminal_growth_key,
                help="Perpetual growth rate used in Gordon Growth terminal value.",
                **_widget_value_kwargs(terminal_growth_key, terminal_growth_default),
            )
            st.caption("Fallback/default: 3.0% (if not set)")

        col_run, col_details = st.columns([1, 1])
        with col_run:
            if st.button("Run DCF Analysis", type="primary", key="run_dcf"):
                with st.spinner("Running DCF analysis with full verification..."):
                    st.session_state.dcf_wacc = user_wacc
                    st.session_state.dcf_fcf_growth = user_fcf_growth
                    st.session_state.dcf_terminal_growth = user_terminal_growth
                    st.session_state.dcf_terminal_scenario = "current"
                    st.session_state.dcf_custom_multiple = None
                    ui_adapter_result, engine_result, snapshot = run_dcf_analysis(
                        ticker,
                        user_wacc,
                        user_fcf_growth,
                        terminal_growth=user_terminal_growth,
                        terminal_scenario="current",
                        custom_multiple=None,
                    )
                    st.session_state.dcf_ui_adapter = ui_adapter_result
                    st.session_state.dcf_engine_result = engine_result
                    st.session_state.dcf_snapshot = snapshot
                    _persist_dcf_result_for_context(
                        ticker=ticker,
                        end_date=st.session_state.get("end_date") or st.session_state.get("selected_end_date"),
                        num_quarters=st.session_state.get("num_quarters"),
                    )
                    st.rerun()

        with col_details:
            if st.session_state.get("dcf_ui_adapter"):
                if st.button("View DCF Details →", key="view_details"):
                    st.session_state.show_dcf_details = True
                    st.rerun()

        dcf_ui = st.session_state.get("dcf_ui_adapter")
        dcf_ui_data = dcf_ui.get_ui_data() if dcf_ui else None
        if dcf_ui_data:
            if not dcf_ui_data.get("success"):
                st.error("DCF analysis failed.")
                for err in dcf_ui_data.get("errors", []):
                    st.error(f"• {err}")
            else:
                current_price = dcf_ui_data.get("current_price", 0)
                intrinsic = dcf_ui_data.get("price_per_share", 0)
                upside_downside = ((intrinsic - current_price) / current_price * 100) if (current_price and current_price > 0 and intrinsic and intrinsic > 0) else None
                dcf_snapshot = st.session_state.get("dcf_snapshot")
                data_quality_score = dcf_ui_data.get("data_quality_score", 0)
                data_quality_help = _build_data_quality_help_text(data_quality_score, dcf_snapshot)

                col_ev, col_equity, col_intrinsic, col_quality = st.columns(4)
                with col_ev:
                    ev = dcf_ui_data.get("enterprise_value", 0)
                    st.metric("Enterprise Value", f"${ev/1e9:.1f}B" if ev >= 1e9 else (f"${ev/1e6:.1f}M" if ev >= 1e6 else "—"))
                with col_equity:
                    equity = dcf_ui_data.get("equity_value", 0)
                    st.metric("Equity Value", f"${equity/1e9:.1f}B" if equity >= 1e9 else (f"${equity/1e6:.1f}M" if equity >= 1e6 else "—"))
                with col_intrinsic:
                    st.metric("Intrinsic Value/Share", f"${intrinsic:.2f}" if intrinsic else "—", delta=f"{upside_downside:+.1f}%" if upside_downside is not None else None)
                with col_quality:
                    st.metric("Data Quality", f"{data_quality_score:.0f}/100", help=data_quality_help)

                st.caption("For full traceability use 'View DCF Details' for inputs, projections, bridge, and assumptions.")
                _render_dcf_trace_chatbot(
                    dcf_ui,
                    dcf_ui_data,
                    st.session_state.get("dcf_engine_result") or {},
                    dcf_snapshot,
                    location_key="deep_dive",
                )
                st.caption("Assistant responses can make mistakes. Verify against the DCF details and source columns.")
        else:
            st.info("No DCF output yet. Set assumptions and run the model.")

    with dashboard_tab:
        # SECTION A: Business Momentum
        st.markdown('<div id="momentum"></div>', unsafe_allow_html=True)
        st.markdown("---")
        st.markdown('<div class="section-header"><span class="step-badge">Step 01</span><span class="section-title">Business Momentum</span></div>', unsafe_allow_html=True)
        source_caption = str(data_source or "").replace(" (Yahoo-priority)", "").strip()
        st.caption(f"Source: {source_caption}")

        latest_quarter_snapshot = hist_data[0] if hist_data else {}
        latest_revenue_value = _positive_float(latest_quarter_snapshot.get("revenue")) if isinstance(latest_quarter_snapshot, dict) else None
        latest_eps_value = None
        if isinstance(latest_quarter_snapshot, dict):
            try:
                latest_eps_candidate = latest_quarter_snapshot.get("eps")
                if latest_eps_candidate is not None:
                    latest_eps_value = float(latest_eps_candidate)
            except Exception:
                latest_eps_value = None

        displayed_quarters_metric = None
        if hist_data:
            loaded_quarters_for_metric = len(hist_data)
            requested_quarters_for_metric = st.session_state.get(
                "momentum_display_quarters",
                loaded_quarters_for_metric,
            )
            if not isinstance(requested_quarters_for_metric, int):
                requested_quarters_for_metric = loaded_quarters_for_metric
            displayed_quarters_metric = min(
                loaded_quarters_for_metric,
                max(1, int(requested_quarters_for_metric)),
            )
        seasonality_reason = str(seasonality_info.get("reason", "") or "").strip()
        seasonality_confidence = str(seasonality_info.get("confidence", "") or "").strip()
        seasonality_pattern = str(seasonality_info.get("pattern", "N/A") or "N/A").strip()
        if seasonality_pattern == "N/A" and seasonality_reason:
            seasonality_pattern = "Need More Data"
        seasonality_tip = ""
        if (
            isinstance(displayed_quarters_metric, int)
            and displayed_quarters_metric < 12
        ) or seasonality_confidence.lower() == "low":
            seasonality_tip = "Tip: Load more quarters from the Quarter Rail to improve the seasonality read."
        seasonality_help_parts = [
            "Determined from repeated quarter-over-quarter revenue changes across the loaded history.",
            "The model looks for quarter-specific uplift or weakness patterns and needs at least 8 revenue quarters plus repeated comparable transitions.",
        ]
        if seasonality_pattern == "Mixed Seasonality":
            seasonality_help_parts.append(
                "Mixed Seasonality means some quarter effects exist, but no single quarter is consistently dominant."
            )
        elif seasonality_pattern == "Low Seasonality":
            seasonality_help_parts.append(
                "Low Seasonality means quarter-to-quarter revenue changes are relatively even across the year."
            )
        elif seasonality_pattern == "Need More Data":
            seasonality_help_parts.append(
                "Need More Data means there are not yet enough comparable quarters to score a reliable pattern."
            )
        if seasonality_confidence:
            seasonality_help_parts.append(f"Confidence: {seasonality_confidence.title()}.")
        if seasonality_reason:
            seasonality_help_parts.append(f"Current run: {seasonality_reason}")
        if seasonality_tip:
            seasonality_help_parts.append(seasonality_tip)
        seasonality_help = " ".join(part for part in seasonality_help_parts if part)

        col_h1, col_h2, col_h3, col_h4 = st.columns(4)
        with col_h1:
            if latest_revenue_value is not None:
                latest_revenue_text = f"${latest_revenue_value/1e9:.1f}B" if latest_revenue_value >= 1e9 else f"${latest_revenue_value/1e6:.1f}M"
            else:
                latest_revenue_text = "N/A"
            st.metric("Latest Revenue", latest_revenue_text)
        with col_h2:
            st.metric("Latest EPS", f"${latest_eps_value:.2f}" if latest_eps_value is not None else "N/A")
        with col_h3:
            quarters_with_revenue = growth_summary.get("quarters_with_revenue")
            quarters_with_eps = growth_summary.get("quarters_with_eps")
            quarters_plotted = displayed_quarters_metric if displayed_quarters_metric is not None else 0
            if not quarters_plotted:
                if isinstance(quarters_with_revenue, int):
                    quarters_plotted = max(quarters_plotted, quarters_with_revenue)
                if isinstance(quarters_with_eps, int):
                    quarters_plotted = max(quarters_plotted, quarters_with_eps)
                if not quarters_plotted:
                    fallback_samples = growth_summary.get("samples_used")
                    quarters_plotted = fallback_samples if isinstance(fallback_samples, int) else 0
            st.metric("Quarters Plotted", quarters_plotted if quarters_plotted else "N/A")
        with col_h4:
            st.metric("Seasonality Pattern", seasonality_pattern, help=seasonality_help)

        total_quarters = growth_summary.get("samples_used")
        revenue_yoy_pairs = growth_summary.get("revenue_yoy_pairs")
        eps_yoy_pairs = growth_summary.get("eps_yoy_pairs")
        coverage_parts = []
        if isinstance(quarters_with_revenue, int) and isinstance(total_quarters, int):
            coverage_parts.append(f"Revenue points: {quarters_with_revenue}/{total_quarters}")
        if isinstance(quarters_with_eps, int) and isinstance(total_quarters, int):
            coverage_parts.append(f"EPS points: {quarters_with_eps}/{total_quarters}")
        if isinstance(revenue_yoy_pairs, int):
            coverage_parts.append(f"Revenue YoY pairs: {revenue_yoy_pairs}")
        if isinstance(eps_yoy_pairs, int):
            coverage_parts.append(f"EPS YoY pairs: {eps_yoy_pairs}")
        mismatch_points = source_diagnostics.get("mismatch_points")
        if isinstance(mismatch_points, int) and mismatch_points > 0:
            coverage_parts.append(f"Cross-source mismatches: {mismatch_points} (Yahoo-priority)")
        yahoo_collisions = source_diagnostics.get("yahoo_date_collisions_collapsed")
        sec_collisions = source_diagnostics.get("sec_date_collisions_collapsed")
        collision_total = 0
        if isinstance(yahoo_collisions, int):
            collision_total += yahoo_collisions
        if isinstance(sec_collisions, int):
            collision_total += sec_collisions
        if collision_total > 0:
            coverage_parts.append(f"Quarter-date collisions normalized: {collision_total}")
        sec_error = source_diagnostics.get("sec_error")
        if isinstance(sec_error, str) and sec_error.strip():
            sec_error_preview = sec_error.strip()
            if len(sec_error_preview) > 120:
                sec_error_preview = sec_error_preview[:117] + "..."
            coverage_parts.append(f"SEC unavailable: {sec_error_preview}")
        sec_annual_end_backfilled = {}
        if isinstance(source_diagnostics, dict):
            sec_annual_end_backfilled = (
                source_diagnostics.get("sec_annual_end_backfilled")
                or source_diagnostics.get("sec_q4_backfilled", {})
            )
        if isinstance(sec_annual_end_backfilled, dict):
            total_annual_end_derived = sec_annual_end_backfilled.get("total_annual_end_derived")
            if not isinstance(total_annual_end_derived, int):
                total_annual_end_derived = sec_annual_end_backfilled.get("total_q4_derived")
            if isinstance(total_annual_end_derived, int) and total_annual_end_derived > 0:
                derived_quarters = (
                    sec_annual_end_backfilled.get("display_quarters", sec_annual_end_backfilled.get("quarters", []))
                    if isinstance(sec_annual_end_backfilled.get("display_quarters", sec_annual_end_backfilled.get("quarters", [])), list)
                    else []
                )
                preview = ", ".join(derived_quarters[:3]) if derived_quarters else ""
                if len(derived_quarters) > 3:
                    preview += ", ..."
                coverage_parts.append(
                    f"SEC annual-end quarter backfilled: {total_annual_end_derived}"
                    + (f" ({preview})" if preview else "")
                )
        missing_revenue_quarters = data_coverage.get("missing_revenue_quarters", [])
        missing_eps_quarters = data_coverage.get("missing_eps_quarters", [])
        missing_report_quarters = data_coverage.get("missing_report_quarters", [])
        if missing_report_quarters:
            missing_report_display = ", ".join(missing_report_quarters[:3])
            if len(missing_report_quarters) > 3:
                missing_report_display += ", ..."
            coverage_parts.append(f"Missing report quarters: {missing_report_display}")
        if missing_revenue_quarters:
            missing_rev_display = ", ".join(missing_revenue_quarters[:3])
            if len(missing_revenue_quarters) > 3:
                missing_rev_display += ", ..."
            coverage_parts.append(f"Missing revenue: {missing_rev_display}")
        if missing_eps_quarters:
            missing_eps_display = ", ".join(missing_eps_quarters[:3])
            if len(missing_eps_quarters) > 3:
                missing_eps_display += ", ..."
            coverage_parts.append(f"Missing EPS: {missing_eps_display}")
        if coverage_parts:
            st.caption(" | ".join(coverage_parts))
        if seasonality_reason:
            st.caption(f"Seasonality method: {seasonality_reason}")
        if seasonality_tip:
            st.caption(seasonality_tip)

        if hist_data:
            loaded_quarters = len(hist_data)
            configured_quarters = st.session_state.get("num_quarters", loaded_quarters)
            if not isinstance(configured_quarters, int):
                configured_quarters = loaded_quarters
            available_dates = st.session_state.get("available_dates", []) or []
            selected_end_date = st.session_state.get("selected_end_date")
            context_available_quarters = len(available_dates)
            if available_dates and selected_end_date:
                date_values = [d.get("value") for d in available_dates if isinstance(d, dict)]
                if selected_end_date in date_values:
                    selected_idx = date_values.index(selected_end_date)
                    context_available_quarters = max(1, len(date_values) - selected_idx)

            source_total_quarters = 0
            if isinstance(source_diagnostics, dict):
                source_total_quarters = (
                    source_diagnostics.get("merged_quarters")
                    or source_diagnostics.get("yahoo_quarters")
                    or source_diagnostics.get("sec_quarters")
                    or 0
                )
            if not isinstance(source_total_quarters, int):
                source_total_quarters = 0

            if source_total_quarters > 0:
                max_context_quarters = source_total_quarters
            else:
                max_context_quarters = context_available_quarters
            max_available_quarters = min(20, max(loaded_quarters, max_context_quarters or loaded_quarters))
            min_display_quarters = 4 if max_available_quarters >= 4 else 1

            pending_display_quarters = st.session_state.get("pending_momentum_display_quarters")
            if isinstance(pending_display_quarters, int):
                clamped_pending = min(
                    max_available_quarters,
                    max(min_display_quarters, pending_display_quarters),
                )
                st.session_state.momentum_display_quarters = clamped_pending
            st.session_state.pending_momentum_display_quarters = None

            current_display_quarters = st.session_state.get("momentum_display_quarters", configured_quarters)
            if not isinstance(current_display_quarters, int):
                current_display_quarters = configured_quarters
            current_display_quarters = min(max_available_quarters, max(min_display_quarters, current_display_quarters))
            st.session_state.momentum_display_quarters = current_display_quarters

            trend_col, rail_col = st.columns([6, 1], gap="medium")

            requested_quarters = current_display_quarters
            with rail_col:
                st.markdown("**Quarter Rail**")
                st.caption(f"Loaded: {loaded_quarters} | Max: {max_available_quarters}")
                if available_dates and selected_end_date:
                    date_values = [d.get("value") for d in available_dates if isinstance(d, dict)]
                    if selected_end_date in date_values:
                        selected_idx = date_values.index(selected_end_date)
                        if selected_idx > 0:
                            st.caption(
                                f"Yahoo reports from this anchor: {context_available_quarters} quarter(s). "
                                "SEC-adjusted history may extend further."
                            )
                st.slider(
                    "Display quarters",
                    min_value=min_display_quarters,
                    max_value=max_available_quarters,
                    key="momentum_display_quarters",
                    label_visibility="collapsed",
                    help="Move the rail to request more history. If you move above loaded data, additional quarters are fetched automatically.",
                )
                requested_quarters = int(st.session_state.get("momentum_display_quarters", current_display_quarters))
                requested_quarters = min(max_available_quarters, max(min_display_quarters, requested_quarters))
                if requested_quarters > loaded_quarters:
                    st.caption(f"Auto-loading +{requested_quarters - loaded_quarters} quarter(s)")
                else:
                    st.caption("Using currently loaded history.")

            if requested_quarters > loaded_quarters:
                end_date_for_reload = st.session_state.get("end_date") or st.session_state.get("selected_end_date")
                with st.spinner(f"Loading up to {requested_quarters} quarters for {ticker}..."):
                    expanded_analysis = cached_quarterly_analysis(ticker, requested_quarters, end_date_for_reload)
                expanded_hist_data = expanded_analysis.get("historical_trends", {}).get("quarterly_data", [])
                if len(expanded_hist_data) > loaded_quarters:
                    st.session_state.quarterly_analysis = expanded_analysis
                    st.session_state.num_quarters = requested_quarters
                    st.session_state.pending_config_num_quarters = requested_quarters
                    st.session_state.pending_momentum_display_quarters = min(requested_quarters, len(expanded_hist_data))

                    financials = st.session_state.get("financials", {}) if isinstance(st.session_state.get("financials", {}), dict) else {}
                    inc = financials.get("income", pd.DataFrame())
                    bal = financials.get("balance", pd.DataFrame())
                    cf = financials.get("cashflow", pd.DataFrame())
                    qcf = financials.get("quarterly_cashflow", pd.DataFrame())
                    st.session_state.comprehensive_analysis = calculate_comprehensive_analysis(
                        inc,
                        bal,
                        expanded_hist_data,
                        ticker,
                        cf,
                        qcf
                    )
                    st.rerun()

                st.session_state.pending_config_num_quarters = max(4, loaded_quarters)
                st.session_state.pending_momentum_display_quarters = loaded_quarters
                st.info("No additional quarters were returned for this ticker and selected ending report.")
                st.rerun()

            display_quarters = min(requested_quarters, loaded_quarters)
            hist_window = hist_data[:display_quarters]
            df_hist = pd.DataFrame(hist_window)
            quarter_order = list(reversed(df_hist["quarter"].tolist())) if not df_hist.empty and "quarter" in df_hist.columns else []
            if not df_hist.empty:
                df_hist = df_hist.iloc[::-1].reset_index(drop=True)

            def render_trend_chart(chart_source: pd.DataFrame, value_col: str, y_title: str, color: str):
                chart_data = chart_source[["quarter", value_col]].copy()
                if chart_data.empty or not chart_data[value_col].notna().any():
                    st.caption("No data available for this trend.")
                    return

                line = (
                    alt.Chart(chart_data)
                    .mark_line(point=True, strokeWidth=3, color=color)
                    .encode(
                        x=alt.X(
                            "quarter:N",
                            sort=quarter_order if quarter_order else None,
                            axis=alt.Axis(title=None, labelAngle=-45, labelOverlap=False),
                        ),
                        y=alt.Y(f"{value_col}:Q", title=y_title),
                        tooltip=[
                            alt.Tooltip("quarter:N", title="Quarter"),
                            alt.Tooltip(f"{value_col}:Q", title=y_title, format=",.2f"),
                        ],
                    )
                )
                st.altair_chart(line, use_container_width=True)

            with trend_col:
                col_chart1, col_chart2 = st.columns(2)
                with col_chart1:
                    st.caption("Revenue Trend")
                    render_trend_chart(df_hist, "revenue", "Revenue (USD)", "#1f77b4")
                with col_chart2:
                    st.caption("EPS Trend")
                    render_trend_chart(df_hist, "eps", "EPS", "#1f77b4")

        col_growth_avg1, col_growth_avg2 = st.columns(2)
        with col_growth_avg1:
            avg_rev = growth_summary.get("avg_revenue_yoy")
            st.metric("Avg Revenue Growth (YoY)", f"{avg_rev:.1f}%" if avg_rev is not None else "N/A")
        with col_growth_avg2:
            avg_eps = growth_summary.get("avg_eps_yoy")
            st.metric("Avg EPS Growth (YoY)", f"{avg_eps:.1f}%" if avg_eps is not None else "N/A")

        st.markdown("**Fundamental Drivers (DuPont)**")
        dupont = comp_analysis.get("dupont", {}) if comp_analysis else {}
        if dupont:
            col_d1, col_d2, col_d3, col_d4 = st.columns(4)
            with col_d1:
                st.metric("ROE", f"{dupont.get('roe', 0):.1f}%")
            with col_d2:
                st.metric("Profit Margin", f"{dupont.get('net_profit_margin', 0):.1f}%")
            with col_d3:
                st.metric("Asset Turnover", f"{dupont.get('asset_turnover', 0):.2f}x")
            with col_d4:
                st.metric("Leverage (EM)", f"{dupont.get('equity_multiplier', 0):.2f}x")
        else:
            st.info("DuPont data unavailable.")

        with st.expander("Quarterly Raw Detail", expanded=False, icon="📋"):
            if hist_data:
                df_display = pd.DataFrame(hist_data).set_index("quarter")
                if "revenue" in df_display.columns:
                    df_display["Revenue"] = df_display["revenue"].apply(
                        lambda x: f"${x/1e9:.2f}B" if x and x > 1e9 else (f"${x/1e6:.1f}M" if x else "N/A")
                    )
                if "eps" in df_display.columns:
                    df_display["EPS"] = df_display["eps"].apply(lambda x: f"${x:.2f}" if x else "N/A")
                cols_to_show = [c for c in ["Revenue", "EPS"] if c in df_display.columns]
                if cols_to_show:
                    st.dataframe(df_display[cols_to_show], use_container_width=True)

            if growth_detail:
                st.caption("Growth Rates")
                df_growth = pd.DataFrame(growth_detail).set_index("quarter")
                for col in df_growth.columns:
                    df_growth[col] = df_growth[col].apply(lambda x: f"{x:.1f}%" if x is not None else "—")
                st.dataframe(df_growth, use_container_width=True)

    with compare_tab:
        compare_current_price = _positive_float(_market_data.get("current_price"))
        compare_dcf_value = None
        compare_dcf_delta = None
        if dcf_ui_data and dcf_ui_data.get("success"):
            compare_dcf_value = _positive_float(dcf_ui_data.get("price_per_share"))
            if compare_current_price and compare_dcf_value:
                compare_dcf_delta = f"{((compare_dcf_value - compare_current_price) / compare_current_price) * 100:+.1f}%"

        compare_avg_pt = parse_price_value(consensus_targets.get("average")) if consensus_targets else None
        col_compare1, col_compare2 = st.columns(2)
        with col_compare1:
            st.metric("Current Price", f"${compare_current_price:,.2f}" if compare_current_price is not None else "N/A")
        with col_compare2:
            st.metric("DCF Value", f"${compare_dcf_value:,.2f}" if compare_dcf_value is not None else "N/A", delta=compare_dcf_delta)

        compare_chart_rows = []
        if compare_current_price is not None:
            compare_chart_rows.append({"Series": "Current Price", "Price": compare_current_price})
        if compare_dcf_value is not None:
            compare_chart_rows.append({"Series": "DCF Implied", "Price": compare_dcf_value})
        if compare_avg_pt is not None:
            compare_chart_rows.append({"Series": "Street Avg PT", "Price": compare_avg_pt})

        if compare_chart_rows:
            compare_chart_df = pd.DataFrame(compare_chart_rows)
            compare_price_chart = (
                alt.Chart(compare_chart_df)
                .mark_bar(size=72, cornerRadiusTopLeft=12, cornerRadiusTopRight=12)
                .encode(
                    x=alt.X("Series:N", title=None, axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("Price:Q", title="Price (USD)"),
                    color=alt.Color(
                        "Series:N",
                        legend=None,
                        scale=alt.Scale(
                            domain=["Current Price", "DCF Implied", "Street Avg PT"],
                            range=["#111c3d", "#2563eb", "#6aa6ff"],
                        ),
                    ),
                    tooltip=[
                        alt.Tooltip("Series:N", title="Series"),
                        alt.Tooltip("Price:Q", title="Price", format=",.2f"),
                    ],
                )
            )
            compare_price_labels = compare_price_chart.mark_text(
                dy=-10,
                fontSize=12,
                fontWeight=600,
                color="#111c3d",
            ).encode(text=alt.Text("Price:Q", format="$,.2f"))
            st.markdown("**Price Comparison**")
            st.altair_chart(
                (compare_price_chart + compare_price_labels).properties(height=280),
                use_container_width=True,
            )

        # SECTION D: Street Context
        st.markdown('<div id="consensus"></div>', unsafe_allow_html=True)
        st.markdown("---")
        st.markdown('<div class="section-header"><span class="step-badge">Step 04</span><span class="section-title">Street Context</span></div>', unsafe_allow_html=True)
        if fiscal_note:
            st.caption(f"Quarter convention: {fiscal_note}")

        consensus_citations = []
        qual_sources = []
        if consensus.get("error"):
            st.error(consensus["error"])
        elif consensus:
            next_q = consensus.get("next_quarter", {})
            coverage = consensus.get("analyst_coverage", {})
            targets = consensus.get("price_targets", {})
            consensus_citations = consensus.get("citations", [])
            qual_sources = consensus.get("qualitative_sources", [])
            consensus_warning = str(consensus.get("warning", "") or "").strip()

            has_next_estimate = any((next_q.get("revenue_estimate"), next_q.get("eps_estimate"))) and any(
                value not in (None, "", "N/A") for value in (next_q.get("revenue_estimate"), next_q.get("eps_estimate"))
            )
            has_coverage = any((coverage.get("num_analysts"), coverage.get("buy_ratings"), coverage.get("hold_ratings"), coverage.get("sell_ratings")))
            has_targets = any(value not in (None, "", "N/A") for value in (targets.get("low"), targets.get("average"), targets.get("high")))

            if consensus_warning and not (has_next_estimate or has_coverage or has_targets):
                st.info(consensus_warning)

            quarter_label = next_q.get("quarter_label") or next_forecast_label
            st.markdown(f"**{quarter_label} Estimates**")
            col_c1, col_c2, col_c3, col_c4 = st.columns(4)
            with col_c1:
                st.metric("Revenue", next_q.get("revenue_estimate") or "N/A")
            with col_c2:
                st.metric("EPS", next_q.get("eps_estimate") or "N/A")
            with col_c3:
                st.metric("Analysts", coverage.get("num_analysts") or "N/A")
            with col_c4:
                buy = coverage.get("buy_ratings", 0) or 0
                hold = coverage.get("hold_ratings", 0) or 0
                sell = coverage.get("sell_ratings", 0) or 0
                total = buy + hold + sell
                st.metric("Buy/Hold/Sell", f"{buy}/{hold}/{sell}" if total > 0 else "N/A")

            estimate_sources = []
            if next_q.get("source"):
                estimate_sources.append(next_q.get("source"))
            if coverage.get("source") and coverage.get("source") not in estimate_sources:
                estimate_sources.append(coverage.get("source"))
            if estimate_sources:
                st.caption(f"Source: {', '.join(estimate_sources)}")

            if has_targets:
                col_t1, col_t2, col_t3 = st.columns(3)
                with col_t1:
                    st.metric("Price Target (Low)", targets.get("low") or "N/A")
                with col_t2:
                    st.metric("Price Target (Avg)", targets.get("average") or "N/A")
                with col_t3:
                    st.metric("Price Target (High)", targets.get("high") or "N/A")
                if targets.get("source"):
                    st.caption(f"Price target source: {targets.get('source')}")

            market_data = analysis.get("market_data", {})
            shares_outstanding = market_data.get("shares_outstanding")
            current_market_cap = market_data.get("market_cap")
            if has_targets and shares_outstanding:
                avg_pt = parse_price_value(targets.get("average"))
                high_pt = parse_price_value(targets.get("high"))
                low_pt = parse_price_value(targets.get("low"))

                def format_mcap(value):
                    if value is None:
                        return "N/A"
                    return f"${value/1e9:.1f}B"

                def format_delta(value):
                    if value is None or not current_market_cap:
                        return None
                    delta_pct = (value - current_market_cap) / current_market_cap * 100
                    return f"{delta_pct:+.0f}%"

                col_i1, col_i2, col_i3 = st.columns(3)
                with col_i1:
                    implied_low = low_pt * shares_outstanding if low_pt is not None else None
                    st.metric("Implied Value (Low PT)", format_mcap(implied_low), delta=format_delta(implied_low))
                with col_i2:
                    implied_avg = avg_pt * shares_outstanding if avg_pt is not None else None
                    st.metric("Implied Value (Avg PT)", format_mcap(implied_avg), delta=format_delta(implied_avg))
                with col_i3:
                    implied_high = high_pt * shares_outstanding if high_pt is not None else None
                    st.metric("Implied Value (High PT)", format_mcap(implied_high), delta=format_delta(implied_high))
            elif has_targets and not shares_outstanding:
                st.caption("Implied total value unavailable (shares outstanding missing).")

        else:
            st.info("No consensus data available.")

    with reports_tab:
        # SECTION E: AI Synthesis
        st.markdown('<div id="outlook"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header"><span class="step-badge">Step 05</span><span class="section-title">AI Synthesis</span></div>', unsafe_allow_html=True)
        if fiscal_note:
            st.caption(f"Quarter convention: {fiscal_note}")
            if fiscal_example:
                st.caption(f"Example: {fiscal_example}")

        dcf_ui = st.session_state.get("dcf_ui_adapter")
        dcf_data_for_forecast = dcf_ui.get_ui_data() if dcf_ui else None

        if not dcf_ui:
            st.warning("Run DCF Analysis first for a more complete synthesis.")
        st.caption("Combines valuation, consensus, and historical momentum into a multi-horizon view.")

        has_existing_forecast = bool(st.session_state.get("independent_forecast"))
        outlook_button_label = "Regenerate Multi-Horizon Outlook" if has_existing_forecast else "Generate Multi-Horizon Outlook"
        if st.button(
            outlook_button_label,
            type="primary",
            key="generate_outlook",
            help="Reuses the cached outlook when the ticker, report date, quarterly context, and DCF inputs have not changed.",
        ):
            with st.spinner("Analyzing data and generating multi-horizon outlook..."):
                outlook_cache_key = _build_ai_outlook_cache_key(
                    st.session_state.get("quarterly_analysis"),
                    dcf_data_for_forecast,
                )
                forecast = cached_independent_forecast(
                    ticker,
                    st.session_state.get("end_date") or st.session_state.get("selected_end_date"),
                    int(st.session_state.get("num_quarters") or DEFAULT_INITIAL_QUARTERS),
                    outlook_cache_key,
                    company_name=_company_name_display,
                    dcf_data=dcf_data_for_forecast,
                )
                if isinstance(forecast, dict) and forecast.get("error"):
                    st.session_state.ai_outlook_error = str(forecast.get("error"))
                    if not st.session_state.get("independent_forecast"):
                        st.session_state.independent_forecast = forecast
                    st.session_state.forecast_just_generated = False
                else:
                    st.session_state.ai_outlook_error = None
                    st.session_state.independent_forecast = forecast
                    st.session_state.forecast_just_generated = True
                    _persist_ai_result_for_context(
                        ticker=ticker,
                        end_date=st.session_state.get("end_date") or st.session_state.get("selected_end_date"),
                        num_quarters=st.session_state.get("num_quarters"),
                    )
                st.rerun()

        ai_outlook_error = st.session_state.get("ai_outlook_error")
        if ai_outlook_error:
            if is_quota_or_rate_limit_error(ai_outlook_error):
                retry_text = estimate_api_retry_time(ai_outlook_error)
                st.warning(
                    "AI Synthesis is temporarily unavailable because API request quota is exhausted.\n\n"
                    f"Estimated retry/reset time: {retry_text}"
                )
            else:
                st.warning("AI Synthesis is temporarily unavailable. Please try again shortly.")
            with st.expander("AI Synthesis Error Details", expanded=False):
                st.code(str(ai_outlook_error))

        pdf_outlook_view_model = {}
        forecast_for_citations = st.session_state.get("independent_forecast")
        merged_citations = _merge_citations_for_step6(consensus_citations, forecast_for_citations, qual_sources)
        numbered_citations = _number_citations(merged_citations)

        if st.session_state.independent_forecast:
            forecast = st.session_state.independent_forecast
            if forecast.get("error"):
                if not ai_outlook_error:
                    if is_quota_or_rate_limit_error(forecast["error"]):
                        retry_text = estimate_api_retry_time(forecast["error"])
                        st.warning(
                            "AI Synthesis is temporarily unavailable because API request quota is exhausted.\n\n"
                            f"Estimated retry/reset time: {retry_text}"
                        )
                    else:
                        st.warning("AI Synthesis is temporarily unavailable. Please try again shortly.")
                    with st.expander("AI Synthesis Error Details", expanded=False):
                        st.code(str(forecast["error"]))
            else:
                extracted = forecast.get("extracted_forecast") or {}
                full_analysis = _sanitize_ai_valuation_language(forecast.get("full_analysis", "") or "")
                full_analysis = _apply_inline_numeric_citations(full_analysis, numbered_citations)
                full_analysis = full_analysis.replace("$", "\\$")
                expanded_default = bool(st.session_state.get("forecast_just_generated", False))
                view_model = _build_outlook_view_model(extracted, full_analysis)
                summary_text = str(view_model.get("summary", "") or "").strip()
                pdf_outlook_view_model = view_model
                if summary_text:
                    st.markdown(
                        f"""
<div class="final-verdict-card">
  <div class="final-verdict-title">Final Assessment Summary</div>
  <div class="final-verdict-meta">{html.escape(summary_text)}</div>
</div>
                        """,
                        unsafe_allow_html=True,
                    )
                has_extracted = extracted and (extracted.get("short_term_stance") or extracted.get("fundamental_outlook"))

                if has_extracted:
                    short_stance = _pick_keyword(str(extracted.get("short_term_stance", "Neutral") or "Neutral"), ["Bullish", "Neutral", "Bearish"], "Neutral")
                    fund_outlook = _pick_keyword(str(extracted.get("fundamental_outlook", "Stable") or "Stable"), ["Strong", "Stable", "Weakening"], "Stable")
                    stock_outlook = _pick_keyword(str(extracted.get("stock_outlook", "Neutral") or "Neutral"), ["Bullish", "Neutral", "Bearish"], "Neutral")
                    stock_conviction = _pick_keyword(str(extracted.get("stock_conviction", extracted.get("fundamental_conviction", "Medium")) or "Medium"), ["High", "Medium", "Low"], "Medium")

                    col_s1, col_s2, col_s3 = st.columns(3)
                    with col_s1:
                        _sc1 = "stance-card-bull" if short_stance == "Bullish" else "stance-card-bear" if short_stance == "Bearish" else "stance-card-neut"
                        st.markdown(f"""
<div class="stance-card {_sc1}">
  <div style="font-size:11px; color:var(--clr-text-muted);">SHORT-TERM (0-12m)</div>
  <div style="font-size:18px; font-weight:600;">{html.escape(short_stance)}</div>
</div>
                        """, unsafe_allow_html=True)
                    with col_s2:
                        _sc2 = "stance-card-bull" if fund_outlook == "Strong" else "stance-card-neut" if fund_outlook == "Stable" else "stance-card-bear"
                        st.markdown(f"""
<div class="stance-card {_sc2}">
  <div style="font-size:11px; color:var(--clr-text-muted);">FUNDAMENTALS</div>
  <div style="font-size:18px; font-weight:600;">{html.escape(fund_outlook)}</div>
</div>
                        """, unsafe_allow_html=True)
                    with col_s3:
                        _sc3 = "stance-card-bull" if stock_outlook == "Bullish" else "stance-card-bear" if stock_outlook == "Bearish" else "stance-card-neut"
                        st.markdown(f"""
<div class="stance-card {_sc3}">
  <div style="font-size:11px; color:var(--clr-text-muted);">STOCK OUTLOOK (CONVICTION: {html.escape(stock_conviction.upper())})</div>
  <div style="font-size:18px; font-weight:600;">{html.escape(stock_outlook)}</div>
</div>
                        """, unsafe_allow_html=True)

                    key_conditional = _sanitize_ai_valuation_language(extracted.get("key_conditional", ""))
                    if key_conditional and "null" not in str(key_conditional).lower():
                        st.info(f"**Key Conditional:** {key_conditional}")

                    evidence_gaps = extracted.get("evidence_gaps", [])
                    if evidence_gaps:
                        sanitized_gaps = [
                            _sanitize_ai_valuation_language(g) for g in evidence_gaps
                            if g and "null" not in str(g).lower()
                        ]
                        gaps_text = " • ".join(sanitized_gaps)
                        if gaps_text:
                            st.caption(f"Evidence gaps: {gaps_text}")

                    if full_analysis:
                        with st.expander("Full Analysis & Final Assessment", expanded=expanded_default):
                            st.markdown(full_analysis.strip())
                else:
                    if full_analysis:
                        with st.expander("Full Analysis & Final Assessment", expanded=expanded_default):
                            st.markdown(full_analysis.strip())
                    else:
                        st.warning("No analysis generated. Please try again.")

                st.caption(f"Generated: {forecast.get('forecast_date', 'Unknown')}")
                st.session_state.forecast_just_generated = False

        # SECTION F: Sources & Methodology
        st.markdown('<div id="sources"></div>', unsafe_allow_html=True)
        st.markdown("---")
        st.markdown('<div class="section-header"><span class="step-badge">Step 06</span><span class="section-title">Sources & Methodology</span></div>', unsafe_allow_html=True)
        st.caption("Reference material and citations for all report sections.")

        summary_pdf_bytes = _build_summary_pdf_bytes(
            ticker=ticker,
            company_name=_company_name_display,
            as_of_label=most_recent.get("label", ""),
            data_source=data_source,
            dcf_ui_data=dcf_ui_data,
            consensus=consensus,
            next_forecast_label=next_forecast_label,
            view_model=pdf_outlook_view_model,
            wacc_source_summary=wacc_source_summary,
            hist_data=hist_data,
            growth_summary=growth_summary,
            fiscal_note=fiscal_note_short,
        )
        if summary_pdf_bytes:
            st.download_button(
                "Download Summary PDF",
                data=summary_pdf_bytes,
                file_name=f"{ticker.lower()}_summary_report.pdf",
                mime="application/pdf",
                help="Analyst-style PDF with price comparison, revenue trend, outlook, key drivers, risks, and sources.",
            )

        with st.expander("Methodology", expanded=False, icon="📚"):
            st.markdown("Core data sources and method notes used in this report:")
            ticker_for_url = st.session_state.get("ticker", "{ticker}")
            for src in SOURCE_CATALOG.values():
                url = src["url"].replace("{ticker}", ticker_for_url)
                st.markdown(
                    f"**[{src['id']}] {src['label']}** — {src['description']}  \n"
                    f"*Method: {src['method']}*  \n"
                    f"[{url}]({url})"
                )

        with st.expander("Citations", expanded=False, icon="🔗"):
            if numbered_citations:
                st.markdown(_format_numbered_citations_markdown(numbered_citations))
            else:
                st.markdown(f"- **[C1]** [Yahoo Finance](https://finance.yahoo.com/quote/{ticker}/analysis) — EPS & Revenue estimates, analyst ratings")

else:
    _boot_log("empty_state_branch")
    with dashboard_tab:
        st.info("Use the left sidebar to choose a ticker, select a report date, and click Load Data.")
    with deep_dive_tab:
        st.info("Deep Dive unlocks after loading a ticker. You’ll get the valuation verdict, assumption controls, and DCF outputs here.")
    with compare_tab:
        st.info("Compare unlocks after loading a ticker. You’ll see DCF versus Street context here.")
    with reports_tab:
        st.info("Reports unlock after loading a ticker. AI synthesis, exports, and methodology live here.")
_boot_log("render_complete")
