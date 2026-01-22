"""
4-5-4 NRF retail calendar helper.

Maps calendar dates to fiscal_year, fiscal_week, fiscal_month, fiscal_quarter.
Fiscal year: Feb 1 - Jan 31. Weeks start Sunday (approximated via 7-day blocks).
Pattern: 4-5-4 weeks per quarter (13 weeks/quarter, 52-week year; 53 in long years).
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pandas as pd


# 4-5-4 pattern: weeks per fiscal month (1-12). Each quarter = 4 + 5 + 4.
_WEEKS_PER_MONTH = (4, 5, 4, 4, 5, 4, 4, 5, 4, 4, 5, 4)


def _fiscal_year_start(d: date) -> date:
    """Fiscal year starts Feb 1. Jan belongs to previous FY."""
    if d.month >= 2:
        return date(d.year, 2, 1)
    return date(d.year - 1, 2, 1)


def _days_since_fy_start(d: date) -> int:
    start = _fiscal_year_start(d)
    return (d - start).days


def _fiscal_year(d: date) -> int:
    """Fiscal year (Feb-Jan). E.g. 2024-02-01 -> 2024; 2024-01-15 -> 2023."""
    start = _fiscal_year_start(d)
    return start.year


def _fiscal_week(d: date) -> int:
    """1-based fiscal week (1-52 or 1-53). Approximate 7-day blocks from FY start."""
    days = _days_since_fy_start(d)
    if days < 0:
        return 1
    w = (days // 7) + 1
    # Normal years: 52 weeks; long years: 53. Use 53 if we're past week 52.
    if w > 52:
        return min(w, 53)
    return w


def _fiscal_month(d: date) -> int:
    """Fiscal month 1-12 from week-in-year. 4-5-4 pattern."""
    w = _fiscal_week(d)
    if w > 52:
        w = 52  # treat week 53 as part of month 12
    acc = 0
    for i, n in enumerate(_WEEKS_PER_MONTH):
        acc += n
        if w <= acc:
            return i + 1
    return 12


def _fiscal_quarter(d: date) -> int:
    """Fiscal quarter 1-4."""
    m = _fiscal_month(d)
    return (m - 1) // 3 + 1


def date_to_454(d: date) -> dict[str, int]:
    """Map a date to fiscal_year, fiscal_week, fiscal_month, fiscal_quarter."""
    return {
        "fiscal_year": _fiscal_year(d),
        "fiscal_week": _fiscal_week(d),
        "fiscal_month": _fiscal_month(d),
        "fiscal_quarter": _fiscal_quarter(d),
    }


def build_454_lookup(
    start_date: date | str,
    end_date: date | str,
) -> pd.DataFrame:
    """
    Build a lookup DataFrame: calendar_date -> fiscal_year, fiscal_week, fiscal_month, fiscal_quarter.
    """
    if isinstance(start_date, str):
        start_date = datetime.fromisoformat(start_date.replace("Z", "+00:00")).date()
    if isinstance(end_date, str):
        end_date = datetime.fromisoformat(end_date.replace("Z", "+00:00")).date()

    rows = []
    for ts in pd.date_range(start_date, end_date, freq="D"):
        d = ts.date() if hasattr(ts, "date") else ts
        row = {"calendar_date": d, **date_to_454(d)}
        rows.append(row)

    return pd.DataFrame(rows)


def get_454_lookup_path(selection_dir: str | Path = "data/selection") -> Path:
    """Path to calendar_454.csv in selection directory."""
    p = Path(selection_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p / "calendar_454.csv"


def ensure_calendar_454_csv(
    selection_dir: str | Path = "data/selection",
    start_date: date | str = "2020-01-01",
    end_date: date | str = "2031-12-31",
) -> Path:
    """
    Ensure calendar_454.csv exists. Create it if missing.
    Returns path to the CSV.
    """
    path = get_454_lookup_path(selection_dir)
    if path.exists():
        return path

    df = build_454_lookup(start_date, end_date)
    df["calendar_date"] = pd.to_datetime(df["calendar_date"]).dt.strftime("%Y-%m-%d")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_calendar_454(selection_dir: str | Path = "data/selection") -> pd.DataFrame:
    """
    Load 4-5-4 lookup. Ensures CSV exists first.
    """
    ensure_calendar_454_csv(selection_dir)
    path = get_454_lookup_path(selection_dir)
    df = pd.read_csv(path)
    df["calendar_date"] = pd.to_datetime(df["calendar_date"]).dt.normalize()
    return df
