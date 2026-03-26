from __future__ import annotations

from functools import lru_cache
from io import BytesIO
from pathlib import Path
import re

from PIL import Image, ImageDraw, ImageFont


BASE_DIR = Path(__file__).resolve().parent
PAGE_WIDTH = 1275
PAGE_HEIGHT = 1650
PAGE_MARGIN = 84
HEADER_HEIGHT = 214
CARD_GAP = 24
COLUMN_GAP = 24
CARD_RADIUS = 26
FOOTER_SPACE = 72
OUTLOOK_SUMMARY_MAX_LINES = 12
OUTLOOK_KEY_CONDITIONAL_MAX_LINES = 4

COLORS = {
    "page": (248, 250, 252),
    "panel": (255, 255, 255),
    "panel_alt": (239, 246, 255),
    "ink": (15, 23, 42),
    "secondary": (71, 85, 105),
    "muted": (148, 163, 184),
    "navy": (13, 22, 48),
    "navy_soft": (37, 99, 235),
    "line": (226, 232, 240),
    "line_strong": (203, 213, 225),
    "positive": (16, 185, 129),
    "negative": (244, 63, 94),
    "neutral": (37, 99, 235),
    "warning": (245, 158, 11),
    "footer": (100, 116, 139),
    "white": (255, 255, 255),
}

TONE_COLORS = {
    "positive": COLORS["positive"],
    "negative": COLORS["negative"],
    "neutral": COLORS["neutral"],
    "warning": COLORS["warning"],
}

FONT_CANDIDATES = [
    (BASE_DIR / "assets" / "fonts" / "DejaVuSans.ttf", None, False),
    (BASE_DIR / "assets" / "fonts" / "DejaVuSans-Bold.ttf", None, True),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", None, False),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", None, True),
    ("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf", None, False),
    ("/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf", None, True),
    ("/System/Library/Fonts/Avenir.ttc", 0, False),
    ("/System/Library/Fonts/Avenir.ttc", 1, True),
    ("/System/Library/Fonts/Helvetica.ttc", 0, False),
    ("/System/Library/Fonts/Helvetica.ttc", 1, True),
    ("/System/Library/Fonts/Supplemental/Arial.ttf", None, False),
    ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", None, True),
]

CHAR_REPLACEMENTS = {
    "\u2013": "-",
    "\u2014": "-",
    "\u2022": "-",
    "\u2192": "->",
    "\u00d7": "x",
    "\u2264": "<=",
    "\u2265": ">=",
    "\u00a0": " ",
}


def _ascii_text(text: str) -> str:
    value = str(text or "")
    for old, new in CHAR_REPLACEMENTS.items():
        value = value.replace(old, new)
    return value.encode("ascii", "ignore").decode("ascii")


def _mix(color_a: tuple[int, int, int], color_b: tuple[int, int, int], ratio: float) -> tuple[int, int, int]:
    ratio = max(0.0, min(1.0, float(ratio)))
    return tuple(
        int(round((color_a[idx] * ratio) + (color_b[idx] * (1.0 - ratio))))
        for idx in range(3)
    )


@lru_cache(maxsize=None)
def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path, font_index, is_bold in FONT_CANDIDATES:
        if is_bold != bold:
            continue
        try:
            path_str = str(path)
            kwargs = {"size": size}
            if font_index is not None:
                kwargs["index"] = font_index
            return ImageFont.truetype(path_str, **kwargs)
        except Exception:
            continue
    return ImageFont.load_default()


def _text_width(font, text: str) -> float:
    safe_text = _ascii_text(text)
    if hasattr(font, "getlength"):
        return float(font.getlength(safe_text))
    bbox = font.getbbox(safe_text)
    return float(bbox[2] - bbox[0])


def _line_height(font, padding: int = 8) -> int:
    bbox = font.getbbox("Ag")
    return int((bbox[3] - bbox[1]) + padding)


def _truncate_line(text: str, font, max_width: int) -> str:
    candidate = _ascii_text(text).strip()
    if not candidate:
        return ""
    ellipsis = "..."
    while candidate and _text_width(font, candidate + ellipsis) > max_width:
        candidate = candidate[:-1].rstrip()
    return f"{candidate}{ellipsis}" if candidate else ellipsis


def _wrap_text(text: str, font, max_width: int, max_lines: int | None = None) -> list[str]:
    raw = _ascii_text(text).strip()
    if not raw:
        return []

    wrapped: list[str] = []
    for paragraph in [segment.strip() for segment in raw.splitlines() if segment.strip()]:
        words = paragraph.split()
        if not words:
            continue
        line = words[0]
        for word in words[1:]:
            candidate = f"{line} {word}"
            if _text_width(font, candidate) <= max_width:
                line = candidate
                continue
            wrapped.append(line)
            line = word
            if max_lines and len(wrapped) >= max_lines:
                wrapped[-1] = _truncate_line(wrapped[-1], font, max_width)
                return wrapped[:max_lines]
        wrapped.append(line)
        if max_lines and len(wrapped) >= max_lines:
            if len(wrapped) > max_lines:
                wrapped = wrapped[:max_lines]
            wrapped[-1] = _truncate_line(wrapped[-1], font, max_width)
            return wrapped
    if max_lines and len(wrapped) > max_lines:
        wrapped = wrapped[:max_lines]
        wrapped[-1] = _truncate_line(wrapped[-1], font, max_width)
    return wrapped


def _draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    x: int,
    y: int,
    font,
    fill: tuple[int, int, int],
    max_width: int,
    line_padding: int = 6,
    max_lines: int | None = None,
) -> int:
    lines = _wrap_text(text, font, max_width=max_width, max_lines=max_lines)
    line_height = _line_height(font, padding=line_padding)
    current_y = y
    for line in lines:
        draw.text((x, current_y), line, font=font, fill=fill)
        current_y += line_height
    return current_y


def _measure_text_block(
    text: str,
    *,
    font,
    max_width: int,
    line_padding: int = 6,
    max_lines: int | None = None,
) -> int:
    lines = _wrap_text(text, font, max_width=max_width, max_lines=max_lines)
    if not lines:
        return 0
    return len(lines) * _line_height(font, padding=line_padding)


def _draw_metric_card(
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    label: str,
    value: str,
    tone: str = "neutral",
) -> None:
    accent = TONE_COLORS.get(tone, COLORS["neutral"])
    tinted_fill = _mix(accent, COLORS["white"], 0.08 if label.lower() == "verdict" else 0.03)
    draw.rounded_rectangle(
        (x, y, x + width, y + height),
        radius=CARD_RADIUS,
        fill=tinted_fill,
        outline=COLORS["line"],
        width=2,
    )
    draw.rounded_rectangle(
        (x + 20, y + 18, x + width - 20, y + 26),
        radius=4,
        fill=accent,
    )
    label_font = _font(20, bold=False)
    draw.text((x + 22, y + 42), _ascii_text(label).upper(), font=label_font, fill=COLORS["muted"])

    for size in (38, 34, 30, 26):
        value_font = _font(size, bold=True)
        if _text_width(value_font, value) <= width - 44:
            break
    draw.text((x + 22, y + 74), _ascii_text(value), font=value_font, fill=COLORS["ink"])


def _compact_quarter_label(label: str) -> str:
    raw = _ascii_text(label).strip()
    match = re.search(r"(?i)(?:fy)?\s*(\d{4}).*?Q([1-4])", raw)
    if match:
        year = match.group(1)[-2:]
        quarter = match.group(2)
        return f"Q{quarter} '{year}"
    return raw[-8:] if len(raw) > 8 else raw


def _format_compact_money(value: float | int | None) -> str:
    try:
        number = float(value)
    except Exception:
        return "N/A"
    if abs(number) >= 1e12:
        return f"${number/1e12:.1f}T"
    if abs(number) >= 1e9:
        return f"${number/1e9:.1f}B"
    if abs(number) >= 1e6:
        return f"${number/1e6:.0f}M"
    return f"${number:,.0f}"


def _draw_panel_shell(
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    title: str,
    subtitle: str = "",
    alt_fill: bool = False,
) -> int:
    fill = COLORS["panel_alt"] if alt_fill else COLORS["panel"]
    draw.rounded_rectangle(
        (x, y, x + width, y + height),
        radius=CARD_RADIUS,
        fill=fill,
        outline=COLORS["line"],
        width=2,
    )
    draw.rounded_rectangle((x + 22, y + 20, x + 118, y + 28), radius=4, fill=COLORS["navy"])
    draw.text((x + 24, y + 40), _ascii_text(title), font=_font(24, bold=True), fill=COLORS["ink"])
    content_y = y + 86
    if subtitle:
        draw.text((x + 24, y + 72), _ascii_text(subtitle), font=_font(18, bold=False), fill=COLORS["muted"])
        content_y = y + 100
    return content_y


def _draw_price_comparison_panel(
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    series: list[dict],
) -> None:
    content_y = _draw_panel_shell(
        draw,
        x=x,
        y=y,
        width=width,
        height=height,
        title="Price Comparison",
        subtitle="Market vs model vs street",
    )
    usable = [item for item in series if isinstance(item.get("value"), (int, float))]
    if not usable:
        draw.text((x + 28, content_y + 24), "No price comparison available.", font=_font(20, bold=False), fill=COLORS["muted"])
        return

    chart_left = x + 34
    chart_right = x + width - 28
    chart_top = content_y + 18
    chart_bottom = y + height - 88
    label_y = chart_bottom + 18
    value_max = max(float(item["value"]) for item in usable)
    value_max = value_max * 1.12 if value_max > 0 else 1.0

    for idx in range(4):
        line_y = chart_top + int((chart_bottom - chart_top) * idx / 3)
        draw.line((chart_left, line_y, chart_right, line_y), fill=_mix(COLORS["navy"], COLORS["white"], 0.15), width=2)

    draw.text((chart_left, chart_top - 24), _format_compact_money(value_max), font=_font(17, bold=False), fill=COLORS["muted"])
    draw.text((chart_left, chart_bottom + 2), "$0", font=_font(17, bold=False), fill=COLORS["muted"])

    bar_count = len(usable)
    bar_width = min(74, int((chart_right - chart_left) / max(bar_count * 2, 4)))
    gap = int((chart_right - chart_left - (bar_width * bar_count)) / (bar_count + 1))

    for idx, item in enumerate(usable):
        tone = str(item.get("tone", "neutral") or "neutral")
        accent = TONE_COLORS.get(tone, COLORS["neutral"])
        value = float(item["value"])
        bar_x = chart_left + gap + idx * (bar_width + gap)
        bar_height = 0 if value_max <= 0 else int((value / value_max) * (chart_bottom - chart_top - 8))
        bar_top = chart_bottom - bar_height
        draw.rounded_rectangle((bar_x, bar_top, bar_x + bar_width, chart_bottom), radius=14, fill=accent)

        value_label = _ascii_text(item.get("display") or _format_compact_money(value))
        value_width = int(_text_width(_font(18, bold=True), value_label))
        draw.text((bar_x + (bar_width - value_width) / 2, bar_top - 28), value_label, font=_font(18, bold=True), fill=accent)

        label_lines = _wrap_text(str(item.get("label", "")), _font(16, bold=False), bar_width + 18, max_lines=2)
        label_line_height = _line_height(_font(16, bold=False), padding=2)
        for line_idx, line in enumerate(label_lines):
            line_width = int(_text_width(_font(16, bold=False), line))
            draw.text(
                (bar_x + (bar_width - line_width) / 2, label_y + (line_idx * label_line_height)),
                line,
                font=_font(16, bold=False),
                fill=COLORS["muted"],
            )


def _draw_revenue_trend_panel(
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    points: list[dict],
    subtitle: str = "",
) -> None:
    content_y = _draw_panel_shell(
        draw,
        x=x,
        y=y,
        width=width,
        height=height,
        title="Revenue Trend",
        subtitle=subtitle or "Recent reported quarters",
    )
    usable = [
        {"label": _compact_quarter_label(str(item.get("label", ""))), "value": float(item.get("value"))}
        for item in points
        if isinstance(item.get("value"), (int, float))
    ]
    if len(usable) < 2:
        draw.text((x + 28, content_y + 24), "Not enough revenue history for a chart.", font=_font(20, bold=False), fill=COLORS["muted"])
        return

    chart_left = x + 48
    chart_right = x + width - 28
    chart_top = content_y + 14
    chart_bottom = y + height - 82
    labels_y = chart_bottom + 18

    values = [point["value"] for point in usable]
    min_value = min(values)
    max_value = max(values)
    if max_value <= min_value:
        max_value = min_value + 1.0
    range_pad = (max_value - min_value) * 0.18
    min_axis = max(0.0, min_value - range_pad)
    max_axis = max_value + range_pad

    for idx in range(4):
        line_y = chart_top + int((chart_bottom - chart_top) * idx / 3)
        draw.line((chart_left, line_y, chart_right, line_y), fill=_mix(COLORS["navy"], COLORS["white"], 0.15), width=2)

    draw.text((chart_left - 2, chart_top - 24), _format_compact_money(max_axis), font=_font(17, bold=False), fill=COLORS["muted"])
    draw.text((chart_left - 2, chart_bottom + 2), _format_compact_money(min_axis), font=_font(17, bold=False), fill=COLORS["muted"])

    step_x = (chart_right - chart_left) / max(len(usable) - 1, 1)
    coords: list[tuple[float, float]] = []
    for idx, point in enumerate(usable):
        ratio = (point["value"] - min_axis) / (max_axis - min_axis)
        px = chart_left + (idx * step_x)
        py = chart_bottom - (ratio * (chart_bottom - chart_top))
        coords.append((px, py))

    for idx in range(len(coords) - 1):
        draw.line((coords[idx][0], coords[idx][1], coords[idx + 1][0], coords[idx + 1][1]), fill=COLORS["navy"], width=5)

    for idx, ((px, py), point) in enumerate(zip(coords, usable)):
        draw.ellipse((px - 8, py - 8, px + 8, py + 8), fill=COLORS["white"], outline=COLORS["navy"], width=4)
        if idx == len(coords) - 1:
            latest_label = _format_compact_money(point["value"])
            latest_width = int(_text_width(_font(18, bold=True), latest_label))
            draw.rounded_rectangle(
                (px - latest_width - 24, py - 50, px + 12, py - 18),
                radius=10,
                fill=COLORS["panel_alt"],
                outline=COLORS["line"],
                width=2,
            )
            draw.text((px - latest_width - 12, py - 43), latest_label, font=_font(18, bold=True), fill=COLORS["navy"])

        label = point["label"]
        label_width = int(_text_width(_font(16, bold=False), label))
        draw.text((px - label_width / 2, labels_y), label, font=_font(16, bold=False), fill=COLORS["muted"])


def _outlook_tone(value: str) -> str:
    normalized = _ascii_text(value).strip().lower()
    if normalized in {"bullish", "strong", "high"}:
        return "positive"
    if normalized in {"bearish", "weakening", "low"}:
        return "negative"
    return "neutral"


def _measure_outlook_panel(outlook: dict, width: int) -> int:
    summary = _ascii_text(outlook.get("summary", ""))
    key_conditional = _ascii_text(outlook.get("key_conditional", ""))
    text_width = width - 56
    total = 196
    if summary:
        total += _measure_text_block(
            summary,
            font=_font(21, bold=False),
            max_width=text_width,
            line_padding=7,
            max_lines=OUTLOOK_SUMMARY_MAX_LINES,
        )
        total += 16
    if key_conditional:
        total += _measure_text_block(
            key_conditional,
            font=_font(18, bold=False),
            max_width=text_width - 70,
            line_padding=5,
            max_lines=OUTLOOK_KEY_CONDITIONAL_MAX_LINES,
        )
        total += 18
    return max(total, 234)


def _draw_outlook_panel(
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    outlook: dict,
) -> None:
    stock_horizon = _ascii_text(outlook.get("stock_horizon", ""))
    stock_conviction = _ascii_text(outlook.get("stock_conviction", ""))
    subtitle_parts = []
    if stock_horizon:
        subtitle_parts.append(f"Horizon: {stock_horizon}")
    if stock_conviction:
        subtitle_parts.append(f"Conviction: {stock_conviction}")
    content_y = _draw_panel_shell(
        draw,
        x=x,
        y=y,
        width=width,
        height=height,
        title="Outlook",
        subtitle=" | ".join(subtitle_parts),
    )

    chips = [
        ("Short-Term", _ascii_text(outlook.get("short_stance", "Neutral")) or "Neutral"),
        ("Fundamentals", _ascii_text(outlook.get("fund_outlook", "Stable")) or "Stable"),
        ("Stock View", _ascii_text(outlook.get("stock_outlook", "Neutral")) or "Neutral"),
    ]
    chip_width = int((width - 56 - (2 * CARD_GAP)) / 3)
    chip_height = 86
    chip_y = content_y + 8
    for idx, (label, value) in enumerate(chips):
        chip_x = x + 28 + idx * (chip_width + CARD_GAP)
        tone = _outlook_tone(value)
        accent = TONE_COLORS.get(tone, COLORS["neutral"])
        draw.rounded_rectangle(
            (chip_x, chip_y, chip_x + chip_width, chip_y + chip_height),
            radius=18,
            fill=_mix(accent, COLORS["white"], 0.06),
            outline=COLORS["line"],
            width=2,
        )
        draw.rounded_rectangle((chip_x + 18, chip_y + 14, chip_x + chip_width - 18, chip_y + 22), radius=4, fill=accent)
        draw.text((chip_x + 20, chip_y + 34), label.upper(), font=_font(16, bold=False), fill=COLORS["muted"])
        draw.text((chip_x + 20, chip_y + 54), value, font=_font(22, bold=True), fill=COLORS["ink"])

    body_y = chip_y + chip_height + 22
    summary = _ascii_text(outlook.get("summary", ""))
    if summary:
        body_y = _draw_wrapped_text(
            draw,
            summary,
            x=x + 28,
            y=body_y,
            font=_font(21, bold=False),
            fill=COLORS["ink"],
            max_width=width - 56,
            line_padding=7,
            max_lines=OUTLOOK_SUMMARY_MAX_LINES,
        )
        body_y += 16

    key_conditional = _ascii_text(outlook.get("key_conditional", ""))
    if key_conditional:
        watch_label = "Key watch:"
        draw.text((x + 28, body_y), watch_label, font=_font(18, bold=True), fill=COLORS["navy"])
        label_width = int(_text_width(_font(18, bold=True), watch_label))
        _draw_wrapped_text(
            draw,
            key_conditional,
            x=x + 28 + label_width + 10,
            y=body_y,
            font=_font(18, bold=False),
            fill=COLORS["ink"],
            max_width=width - 56 - label_width - 10,
            line_padding=5,
            max_lines=OUTLOOK_KEY_CONDITIONAL_MAX_LINES,
        )


def _measure_section_height(heading: str, items: list[str], width: int) -> int:
    heading_font = _font(24, bold=True)
    is_bottom_line = heading == "Bottom Line"
    is_qualitative = heading == "Why This Verdict"
    body_font = _font(21 if is_bottom_line else 19 if is_qualitative else 20, bold=False)
    body_width = width - 56
    total_height = 30 + _line_height(heading_font, padding=4) + 10

    for item in items:
        raw_item = _ascii_text(item).strip()
        bullet = raw_item.startswith("- ")
        content = raw_item[2:].strip() if bullet else raw_item
        indent = 22 if bullet else 0
        max_lines = 5 if is_bottom_line else 9 if is_qualitative else 4
        total_height += _measure_text_block(
            content,
            font=body_font,
            max_width=body_width - indent,
            line_padding=8 if is_bottom_line else 7 if is_qualitative else 6,
            max_lines=max_lines,
        )
        total_height += 14

    minimum = 176 if is_qualitative else 132
    return max(total_height + 18, minimum)


def _draw_section_card(
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    heading: str,
    items: list[str],
) -> None:
    is_sources = heading == "Core Sources"
    is_qualitative = heading == "Why This Verdict"
    panel_fill = COLORS["panel_alt"] if (is_sources or is_qualitative) else COLORS["panel"]
    accent = COLORS["navy_soft"] if (is_sources or is_qualitative) else COLORS["navy"]
    heading_font = _font(24, bold=True)
    body_font = _font(23 if heading == "Bottom Line" else 19 if is_qualitative else 20, bold=False)
    draw.rounded_rectangle(
        (x, y, x + width, y + height),
        radius=CARD_RADIUS,
        fill=panel_fill,
        outline=COLORS["line"],
        width=2,
    )
    draw.rounded_rectangle((x + 22, y + 20, x + 120, y + 28), radius=4, fill=accent)
    draw.text((x + 24, y + 40), _ascii_text(heading), font=heading_font, fill=COLORS["ink"])

    body_x = x + 26
    body_y = y + 84
    body_width = width - 52
    for item in items:
        raw_item = _ascii_text(item).strip()
        if not raw_item:
            continue
        bullet = raw_item.startswith("- ")
        content = raw_item[2:].strip() if bullet else raw_item
        if bullet:
            bullet_y = body_y + 10
            draw.ellipse((body_x, bullet_y, body_x + 10, bullet_y + 10), fill=accent)
            body_y = _draw_wrapped_text(
                draw,
                content,
                x=body_x + 20,
                y=body_y,
                font=body_font,
                fill=COLORS["ink"],
                max_width=body_width - 20,
                line_padding=7 if is_qualitative else 6,
                max_lines=8 if is_qualitative else 4,
            )
        else:
            body_y = _draw_wrapped_text(
                draw,
                content,
                x=body_x,
                y=body_y,
                font=body_font,
                fill=COLORS["ink"],
                max_width=body_width,
                line_padding=8 if heading == "Bottom Line" else 7 if is_qualitative else 6,
                max_lines=5 if heading == "Bottom Line" else 9 if is_qualitative else 4,
            )
        body_y += 14


def _start_page(title: str, subtitle: str, page_number: int) -> tuple[Image.Image, ImageDraw.ImageDraw, int]:
    image = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), COLORS["page"])
    draw = ImageDraw.Draw(image)

    if page_number == 0:
        draw.rectangle((0, 0, PAGE_WIDTH, HEADER_HEIGHT), fill=COLORS["navy"])
        draw.rectangle((0, 0, PAGE_WIDTH, 10), fill=COLORS["neutral"])
        draw.text((PAGE_MARGIN, 34), "ANALYST SUMMARY", font=_font(20, bold=True), fill=(206, 217, 230))
        draw.text((PAGE_MARGIN, 74), _ascii_text(title), font=_font(52, bold=True), fill=COLORS["white"])
        if subtitle:
            draw.text((PAGE_MARGIN, 150), _ascii_text(subtitle), font=_font(24, bold=False), fill=(209, 217, 227))
        y = HEADER_HEIGHT + 30
    else:
        draw.rectangle((0, 0, PAGE_WIDTH, 78), fill=_mix(COLORS["navy"], COLORS["white"], 0.92))
        draw.text((PAGE_MARGIN, 22), _ascii_text(title), font=_font(26, bold=True), fill=COLORS["ink"])
        if subtitle:
            draw.text((PAGE_MARGIN, 48), _ascii_text(subtitle), font=_font(18, bold=False), fill=COLORS["muted"])
        y = 104

    return image, draw, y


def build_summary_pdf(
    *,
    title: str,
    subtitle: str = "",
    sections: list[tuple[str, list[str]]],
    footer: str = "",
    hero_metrics: list[dict] | None = None,
    price_comparison: list[dict] | None = None,
    revenue_series: list[dict] | None = None,
    revenue_subtitle: str = "",
    outlook: dict | None = None,
) -> bytes:
    sanitized_title = _ascii_text(title)
    sanitized_subtitle = _ascii_text(subtitle)
    sanitized_sections = [
        (_ascii_text(heading), [_ascii_text(item) for item in items if _ascii_text(item).strip()])
        for heading, items in sections
        if _ascii_text(heading).strip()
    ]
    sanitized_sections = [(heading, items) for heading, items in sanitized_sections if items]

    hero_metrics = hero_metrics or []
    price_comparison = price_comparison or []
    revenue_series = revenue_series or []
    outlook = outlook or {}
    sanitized_footer = _ascii_text(footer)

    section_map = {heading: items for heading, items in sanitized_sections}
    ordered_rows: list[tuple[str, object]] = []

    if "Bottom Line" in section_map:
        ordered_rows.append(("full", ("Bottom Line", section_map.pop("Bottom Line"))))
    if "What Matters Most" in section_map or "Main Risks" in section_map:
        ordered_rows.append(
            (
                "pair",
                (
                    ("What Matters Most", section_map.pop("What Matters Most", [])),
                    ("Main Risks", section_map.pop("Main Risks", [])),
                ),
            )
        )
    if "Valuation" in section_map or "Street Context" in section_map:
        ordered_rows.append(
            (
                "pair",
                (
                    ("Valuation", section_map.pop("Valuation", [])),
                    ("Street Context", section_map.pop("Street Context", [])),
                ),
            )
        )
    for heading, items in list(section_map.items()):
        if heading in {"Core Sources", "Why This Verdict"}:
            continue
        ordered_rows.append(("full", (heading, items)))
    if "Core Sources" in section_map:
        ordered_rows.append(("full", ("Core Sources", section_map["Core Sources"])))
    if "Why This Verdict" in section_map:
        ordered_rows.append(("full", ("Why This Verdict", section_map["Why This Verdict"])))

    pages: list[Image.Image] = []
    current_page, draw, current_y = _start_page(sanitized_title, sanitized_subtitle, page_number=0)
    content_width = PAGE_WIDTH - (2 * PAGE_MARGIN)
    half_width = int((content_width - COLUMN_GAP) / 2)

    def ensure_space(required_height: int) -> None:
        nonlocal current_page, draw, current_y
        if current_y + required_height <= PAGE_HEIGHT - PAGE_MARGIN - FOOTER_SPACE:
            return
        pages.append(current_page)
        current_page, draw, current_y = _start_page(sanitized_title, sanitized_subtitle, page_number=len(pages))

    if hero_metrics:
        hero_card_width = int((PAGE_WIDTH - (2 * PAGE_MARGIN) - (3 * CARD_GAP)) / 4)
        hero_card_height = 126
        ensure_space(hero_card_height)
        for idx, metric in enumerate(hero_metrics[:4]):
            card_x = PAGE_MARGIN + (idx * (hero_card_width + CARD_GAP))
            _draw_metric_card(
                draw,
                x=card_x,
                y=current_y,
                width=hero_card_width,
                height=hero_card_height,
                label=str(metric.get("label", "")),
                value=str(metric.get("value", "")),
                tone=str(metric.get("tone", "neutral") or "neutral"),
            )
        current_y += hero_card_height + CARD_GAP

    if price_comparison or revenue_series:
        chart_height = 332
        ensure_space(chart_height)
        if price_comparison and revenue_series:
            chart_width = int((content_width - COLUMN_GAP) / 2)
            _draw_price_comparison_panel(
                draw,
                x=PAGE_MARGIN,
                y=current_y,
                width=chart_width,
                height=chart_height,
                series=price_comparison,
            )
            _draw_revenue_trend_panel(
                draw,
                x=PAGE_MARGIN + chart_width + COLUMN_GAP,
                y=current_y,
                width=chart_width,
                height=chart_height,
                points=revenue_series,
                subtitle=revenue_subtitle,
            )
        elif price_comparison:
            _draw_price_comparison_panel(
                draw,
                x=PAGE_MARGIN,
                y=current_y,
                width=content_width,
                height=chart_height,
                series=price_comparison,
            )
        else:
            _draw_revenue_trend_panel(
                draw,
                x=PAGE_MARGIN,
                y=current_y,
                width=content_width,
                height=chart_height,
                points=revenue_series,
                subtitle=revenue_subtitle,
            )
        current_y += chart_height + CARD_GAP

    if outlook:
        outlook_height = _measure_outlook_panel(outlook, content_width)
        ensure_space(outlook_height)
        _draw_outlook_panel(
            draw,
            x=PAGE_MARGIN,
            y=current_y,
            width=content_width,
            height=outlook_height,
            outlook=outlook,
        )
        current_y += outlook_height + CARD_GAP

    for row_type, payload in ordered_rows:
        if row_type == "pair":
            left_section, right_section = payload
            left_heading, left_items = left_section
            right_heading, right_items = right_section
            left_exists = bool(left_items)
            right_exists = bool(right_items)
            if left_exists and right_exists:
                row_height = max(
                    _measure_section_height(left_heading, left_items, half_width),
                    _measure_section_height(right_heading, right_items, half_width),
                )
                ensure_space(row_height)
                _draw_section_card(
                    draw,
                    x=PAGE_MARGIN,
                    y=current_y,
                    width=half_width,
                    height=row_height,
                    heading=left_heading,
                    items=left_items,
                )
                _draw_section_card(
                    draw,
                    x=PAGE_MARGIN + half_width + COLUMN_GAP,
                    y=current_y,
                    width=half_width,
                    height=row_height,
                    heading=right_heading,
                    items=right_items,
                )
                current_y += row_height + CARD_GAP
                continue
            payload = (left_heading, left_items) if left_exists else (right_heading, right_items)

        heading, items = payload
        row_height = _measure_section_height(heading, items, content_width)
        ensure_space(row_height)
        _draw_section_card(
            draw,
            x=PAGE_MARGIN,
            y=current_y,
            width=content_width,
            height=row_height,
            heading=heading,
            items=items,
        )
        current_y += row_height + CARD_GAP

    pages.append(current_page)

    footer_font = _font(18, bold=False)
    page_font = _font(18, bold=True)
    for idx, image in enumerate(pages):
        footer_draw = ImageDraw.Draw(image)
        footer_y = PAGE_HEIGHT - PAGE_MARGIN - 34
        footer_draw.line(
            (PAGE_MARGIN, footer_y - 18, PAGE_WIDTH - PAGE_MARGIN, footer_y - 18),
            fill=COLORS["line"],
            width=2,
        )
        if idx == len(pages) - 1 and sanitized_footer:
            footer_draw.text((PAGE_MARGIN, footer_y), sanitized_footer, font=footer_font, fill=COLORS["footer"])
        page_label = f"Page {idx + 1}"
        page_label_width = int(_text_width(page_font, page_label))
        footer_draw.text(
            (PAGE_WIDTH - PAGE_MARGIN - page_label_width, footer_y),
            page_label,
            font=page_font,
            fill=COLORS["footer"],
        )

    buffer = BytesIO()
    pages[0].save(buffer, format="PDF", resolution=150.0, save_all=True, append_images=pages[1:])
    return buffer.getvalue()
