"""
Temporal calculations for period-over-period and rolling metrics.

Provides WoW/MoM/QoQ/YoY deltas (decimal) plus rolling windows (L4W/L12W/MTD/QTD/YTD).
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Iterable

import numpy as np
import pandas as pd


_CHANGE_METRIC_ALIASES = {
    "Ordered_Revenue": "Revenue",
    "Ordered_Units": "Units",
    "Glance_Views": "GlanceViews",
    "Net_PPM": "NetPPM",
    "Average_Selling_Price": "ASP",
}

_RATE_HINTS = ("ppm", "pct", "rate", "margin", "conversion", "soroos")


def is_rate_metric(metric: str) -> bool:
    metric_lower = metric.lower()
    return any(hint in metric_lower for hint in _RATE_HINTS)


def _safe_pct_change(current: pd.Series, previous: pd.Series) -> pd.Series:
    return np.where(
        (previous.notna()) & (previous != 0),
        (current - previous) / previous.abs(),
        np.nan,
    )


def _infer_week_years(weeks: pd.Series, base_year: int) -> pd.Series:
    weeks_numeric = pd.to_numeric(weeks, errors="coerce")
    weeks_int = weeks_numeric.fillna(0).astype("int64")
    if weeks_int.min() <= 2 and weeks_int.max() >= 52:
        return weeks_int.apply(lambda w: base_year + 1 if w <= 2 else base_year)
    return pd.Series([base_year] * len(weeks_int), index=weeks.index)


def add_week_dates(
    df: pd.DataFrame,
    *,
    week_col: str = "Week_Number",
    year_col: str | None = None,
    base_year: int | None = None,
    date_col: str = "Week_Start_Date",
) -> pd.DataFrame:
    if week_col not in df.columns:
        return df

    result = df.copy()
    if year_col and year_col in result.columns:
        years = result[year_col].astype("int64")
    else:
        base_year = base_year or datetime.now().year
        years = _infer_week_years(result[week_col], base_year)
        result["Week_Year"] = years

    def to_date(y: int, w: int) -> date | None:
        if w is None or pd.isna(w) or int(w) <= 0:
            return None
        try:
            return date.fromisocalendar(int(y), int(w), 1)
        except Exception:
            return None

    result[date_col] = [
        to_date(y, w) for y, w in zip(years.tolist(), result[week_col].tolist())
    ]
    result[date_col] = pd.to_datetime(result[date_col], errors="coerce")
    return result


def add_calendar_periods(
    df: pd.DataFrame,
    *,
    date_col: str,
    prefix: str = "Period_",
) -> pd.DataFrame:
    if date_col not in df.columns:
        return df

    result = df.copy()
    dates = pd.to_datetime(result[date_col], errors="coerce")
    result[f"{prefix}Year"] = dates.dt.year
    result[f"{prefix}Month"] = dates.dt.month
    result[f"{prefix}Quarter"] = dates.dt.quarter
    result[f"{prefix}Week"] = dates.dt.isocalendar().week.astype("int64")
    return result


def add_period_over_period(
    df: pd.DataFrame,
    *,
    group_cols: Iterable[str],
    period_cols: Iterable[str],
    metric_map: dict[str, str],
    overwrite: bool = False,
) -> pd.DataFrame:
    if not metric_map:
        return df

    result = df.copy()
    group_cols = [c for c in group_cols if c in result.columns]
    period_cols = list(period_cols)
    sort_cols = [c for c in group_cols + period_cols if c in result.columns]
    if not sort_cols:
        return result

    result = result.sort_values(sort_cols)
    grouped = result.groupby(group_cols, dropna=False) if group_cols else result.groupby(lambda _: 0)

    for metric, output in metric_map.items():
        if metric not in result.columns:
            continue
        if output in result.columns and not overwrite:
            continue
        previous = grouped[metric].shift(1)
        result[output] = _safe_pct_change(result[metric], previous)

    return result


def add_rolling_windows(
    df: pd.DataFrame,
    *,
    group_cols: Iterable[str],
    period_cols: Iterable[str],
    metrics: Iterable[str],
    windows: dict[str, int],
    overwrite: bool = False,
) -> pd.DataFrame:
    result = df.copy()
    group_cols = [c for c in group_cols if c in result.columns]
    period_cols = list(period_cols)
    sort_cols = [c for c in group_cols + period_cols if c in result.columns]
    if not sort_cols:
        return result

    result = result.sort_values(sort_cols)
    grouped = result.groupby(group_cols, dropna=False) if group_cols else result.groupby(lambda _: 0)

    for metric in metrics:
        if metric not in result.columns:
            continue
        for label, window in windows.items():
            output = f"{metric}_{label}"
            if output in result.columns and not overwrite:
                continue
            series = grouped[metric]
            if is_rate_metric(metric):
                rolled = series.rolling(window, min_periods=1).mean()
            else:
                rolled = series.rolling(window, min_periods=1).sum()
            result[output] = rolled.reset_index(level=group_cols, drop=True)

    return result


def add_to_date_metrics(
    df: pd.DataFrame,
    *,
    group_cols: Iterable[str],
    metrics: Iterable[str],
    year_col: str = "Period_Year",
    month_col: str = "Period_Month",
    quarter_col: str = "Period_Quarter",
    overwrite: bool = False,
) -> pd.DataFrame:
    result = df.copy()
    group_cols = [c for c in group_cols if c in result.columns]
    if year_col not in result.columns:
        return result

    for metric in metrics:
        if metric not in result.columns:
            continue

        # MTD
        if month_col in result.columns:
            output = f"{metric}_MTD"
            if output not in result.columns or overwrite:
                key = group_cols + [year_col, month_col]
                if is_rate_metric(metric):
                    result[output] = (
                        result.groupby(key, dropna=False)[metric]
                        .expanding()
                        .mean()
                        .reset_index(level=key, drop=True)
                    )
                else:
                    result[output] = result.groupby(key, dropna=False)[metric].cumsum()

        # QTD
        if quarter_col in result.columns:
            output = f"{metric}_QTD"
            if output not in result.columns or overwrite:
                key = group_cols + [year_col, quarter_col]
                if is_rate_metric(metric):
                    result[output] = (
                        result.groupby(key, dropna=False)[metric]
                        .expanding()
                        .mean()
                        .reset_index(level=key, drop=True)
                    )
                else:
                    result[output] = result.groupby(key, dropna=False)[metric].cumsum()

        # YTD
        output = f"{metric}_YTD"
        if output not in result.columns or overwrite:
            key = group_cols + [year_col]
            if is_rate_metric(metric):
                result[output] = (
                    result.groupby(key, dropna=False)[metric]
                    .expanding()
                    .mean()
                    .reset_index(level=key, drop=True)
                )
            else:
                result[output] = result.groupby(key, dropna=False)[metric].cumsum()

    return result


def add_temporal_metrics(
    df: pd.DataFrame,
    *,
    group_cols: Iterable[str] = ("ASIN",),
    week_col: str = "Week_Number",
    year_col: str | None = None,
    date_col: str = "Snapshot_Date",
    base_year: int | None = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    if df.empty:
        return df

    result = df.copy()
    active_date_col = date_col if date_col in result.columns else None

    if active_date_col is None and week_col in result.columns:
        result = add_week_dates(
            result,
            week_col=week_col,
            year_col=year_col,
            base_year=base_year,
            date_col="Week_Start_Date",
        )
        active_date_col = "Week_Start_Date"

    if active_date_col is None:
        return result

    result = add_calendar_periods(result, date_col=active_date_col)

    metrics_for_change = [m for m in _CHANGE_METRIC_ALIASES if m in result.columns]
    metric_map = {
        metric: f"{_CHANGE_METRIC_ALIASES.get(metric, metric)}_WoW"
        for metric in metrics_for_change
    }

    if week_col in result.columns:
        period_cols = [col for col in ["Week_Year", week_col] if col in result.columns]
        result = add_period_over_period(
            result,
            group_cols=group_cols,
            period_cols=period_cols,
            metric_map=metric_map,
            overwrite=overwrite,
        )

        result = add_rolling_windows(
            result,
            group_cols=group_cols,
            period_cols=period_cols,
            metrics=metrics_for_change,
            windows={"L4W": 4, "L12W": 12},
            overwrite=overwrite,
        )

    # Calendar month/quarter/year changes
    period_cols = ["Period_Year", "Period_Month"]
    if all(col in result.columns for col in period_cols):
        result = add_period_over_period(
            result,
            group_cols=group_cols,
            period_cols=period_cols,
            metric_map={
                metric: f"{_CHANGE_METRIC_ALIASES.get(metric, metric)}_MoM"
                for metric in metrics_for_change
            },
            overwrite=overwrite,
        )

    period_cols = ["Period_Year", "Period_Quarter"]
    if all(col in result.columns for col in period_cols):
        result = add_period_over_period(
            result,
            group_cols=group_cols,
            period_cols=period_cols,
            metric_map={
                metric: f"{_CHANGE_METRIC_ALIASES.get(metric, metric)}_QoQ"
                for metric in metrics_for_change
            },
            overwrite=overwrite,
        )

    if "Period_Year" in result.columns:
        result = add_period_over_period(
            result,
            group_cols=group_cols,
            period_cols=["Period_Year"],
            metric_map={
                metric: f"{_CHANGE_METRIC_ALIASES.get(metric, metric)}_YoY"
                for metric in metrics_for_change
            },
            overwrite=overwrite,
        )

        result = add_to_date_metrics(
            result,
            group_cols=group_cols,
            metrics=metrics_for_change,
            overwrite=overwrite,
        )

    return result
