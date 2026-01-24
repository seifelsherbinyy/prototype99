"""
Shared numeric parsing utilities for values with currencies, percents, and bps.

Percent values are normalized to decimals (e.g., "45%" -> 0.45).
Basis points are normalized to decimals (e.g., "25 bps" -> 0.0025).
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

_NULL_TOKENS = {"", "null", "n/a", "none", "nan", "-", "--"}
_CURRENCY_CODES = [
    "usd", "aed", "sar", "gbp", "eur", "jpy", "cny", "cad", "aud",
    "chf", "inr", "krw", "sek", "nok", "dkk",
]
_CURRENCY_SYMBOLS = r"[$€£¥]"
_ARROW_NEGATIVE = ("↓", "↘", "▼")
_ARROW_POSITIVE = ("↑", "↗", "▲")


def _strip_currency_tokens(text: str) -> str:
    pattern = r"\b(" + "|".join(_CURRENCY_CODES) + r")\b"
    return re.sub(pattern, "", text, flags=re.IGNORECASE)


def _normalize_number_string(text: str) -> str:
    cleaned = text.replace(" ", "").replace("\u00a0", "")
    cleaned = re.sub(_CURRENCY_SYMBOLS, "", cleaned)
    cleaned = _strip_currency_tokens(cleaned)

    # Handle European decimals: "1.234,56" -> "1234.56"
    if "," in cleaned and "." in cleaned:
        if cleaned.rfind(",") > cleaned.rfind("."):
            cleaned = cleaned.replace(".", "")
            cleaned = cleaned.replace(",", ".")
    elif "," in cleaned and "." not in cleaned:
        # Treat comma as decimal if it looks like cents (one or two digits)
        parts = cleaned.split(",")
        if len(parts[-1]) in (1, 2):
            cleaned = ".".join(parts)
        else:
            cleaned = cleaned.replace(",", "")

    cleaned = cleaned.replace(",", "")
    cleaned = cleaned.replace("%", "")
    cleaned = cleaned.replace("bps", "")
    cleaned = cleaned.replace("bp", "")
    cleaned = cleaned.strip()
    return cleaned


def parse_numeric_value(value: Any) -> float | None:
    if pd.isna(value):
        return None

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)

    text = str(value).strip()
    if text.lower() in _NULL_TOKENS:
        return None

    is_negative = False
    if text.startswith("(") and text.endswith(")"):
        is_negative = True
        text = text[1:-1].strip()

    if any(token in text for token in _ARROW_NEGATIVE) or "down" in text.lower():
        is_negative = True
    if text.startswith("-"):
        is_negative = True

    is_percentage = "%" in text or "percent" in text.lower() or "pct" in text.lower()
    is_bps = "bps" in text.lower() or re.search(r"\bbp\b", text.lower()) is not None

    for token in _ARROW_NEGATIVE + _ARROW_POSITIVE:
        text = text.replace(token, "")
    text = text.replace("+", "")

    cleaned = _normalize_number_string(text)
    if cleaned.lower() in _NULL_TOKENS:
        return None

    try:
        numeric = float(cleaned)
    except ValueError:
        return None

    if is_bps:
        numeric = numeric / 10000.0
    elif is_percentage:
        numeric = numeric / 100.0

    if is_negative:
        numeric = -abs(numeric)

    return numeric


def parse_numeric_series(series: pd.Series) -> pd.Series:
    return series.apply(parse_numeric_value)
