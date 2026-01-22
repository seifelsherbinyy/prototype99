"""
Analytical logic (DuckDB) and financial metrics.

Consumes normalized `raw_sales` and `ref_selection` from Phase 2.
Provides 4-5-4 enrichment, ASIN-level profitability queries, and pipeline entry points.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from src.calendar_454 import load_calendar_454
from src.logger import debug_watcher

if TYPE_CHECKING:
    import duckdb

_REQUIRED_RAW_SALES_COLUMNS = (
    "ASIN",
    "SKU",
    "Snapshot_Date",
    "Product_Title",
    "Ordered_Revenue",
    "Ordered_Units",
    "CCOGS",
    "Promotional_Rebates",
)

_DEFAULT_SELECTION_DIR: str | Path = "data/selection"


def _registered(con: duckdb.DuckDBPyConnection, name: str) -> bool:
    """Return True if a view/table named `name` is registered."""
    try:
        con.execute(f"SELECT 1 FROM {name} LIMIT 0")
        return True
    except Exception:
        return False


def _validate_raw_sales(con: duckdb.DuckDBPyConnection) -> None:
    if not _registered(con, "raw_sales"):
        raise RuntimeError("`raw_sales` not registered; run ingestion first.")
    # Schema check: required columns exist
    info = con.execute("DESCRIBE raw_sales").fetchall()
    cols = {r[0] for r in info}
    missing = [c for c in _REQUIRED_RAW_SALES_COLUMNS if c not in cols]
    if missing:
        raise RuntimeError(
            f"`raw_sales` missing required columns: {missing}. "
            "Expected: ASIN, SKU, Snapshot_Date, Product_Title, Ordered_Revenue, "
            "Ordered_Units, CCOGS, Promotional_Rebates."
        )


def _validate_ref_selection(con: duckdb.DuckDBPyConnection) -> bool:
    """Return True if ref_selection is registered and has an ASIN column."""
    if not _registered(con, "ref_selection"):
        return False
    info = con.execute("DESCRIBE ref_selection").fetchall()
    cols = {r[0] for r in info}
    return "ASIN" in cols


def _ref_selection_empty(con: duckdb.DuckDBPyConnection) -> bool:
    try:
        n = con.execute("SELECT COUNT(*) FROM ref_selection").fetchone()[0]
        return n == 0
    except Exception:
        return True


def _ensure_calendar_registered(
    con: duckdb.DuckDBPyConnection,
    selection_dir: str | Path = "data/selection",
) -> None:
    """Load 4-5-4 lookup and register as ref_calendar_454 if not already."""
    if _registered(con, "ref_calendar_454"):
        return
    sel = selection_dir if selection_dir is not None else _DEFAULT_SELECTION_DIR
    df = load_calendar_454(sel)
    con.register("ref_calendar_454", df)


def get_fiscal_calendar(
    selection_dir: str | Path = "data/selection",
    years: tuple[int, int] = (2024, 2026),
) -> pd.DataFrame:
    """
    Generate or load a standard Retail 4-5-4 calendar table.

    Maps Calendar_Date to Fiscal_Year, Fiscal_Month (1-12), Fiscal_Week (1-52).
    Uses load_calendar_454; restricted to the given fiscal year range (inclusive).
    """
    sel = selection_dir if selection_dir is not None else _DEFAULT_SELECTION_DIR
    df = load_calendar_454(sel)
    lo, hi = years[0], years[1]
    df = df[(df["fiscal_year"] >= lo) & (df["fiscal_year"] <= hi)].copy()
    df = df.rename(columns={
        "calendar_date": "Calendar_Date",
        "fiscal_year": "Fiscal_Year",
        "fiscal_month": "Fiscal_Month",
        "fiscal_week": "Fiscal_Week",
    })
    return df[["Calendar_Date", "Fiscal_Year", "Fiscal_Month", "Fiscal_Week"]]


@debug_watcher
def calculate_metrics(
    con: duckdb.DuckDBPyConnection,
    *,
    selection_dir: str | Path | None = None,
    filter_by_selection: bool = False,
) -> pd.DataFrame:
    """
    Core profitability query: join raw_sales with ref_selection and fiscal calendar,
    compute Net_Sales, Gross_Profit, Net_PPM; aggregate by ASIN, Product_Title,
    Fiscal_Year, Fiscal_Week; add 4-week moving averages for Net_PPM and Ordered_Units.
    Return a Pandas DataFrame ready for visualization.
    """
    _validate_raw_sales(con)
    sel = selection_dir if selection_dir is not None else _DEFAULT_SELECTION_DIR
    _ensure_calendar_registered(con, sel)

    join_ref_selection = _validate_ref_selection(con)
    selection_filter = ""
    if filter_by_selection:
        if not join_ref_selection:
            warnings.warn(
                "filter_by_selection=True but ref_selection not registered or has no ASIN column; "
                "skipping selection filter.",
                UserWarning,
                stacklevel=2,
            )
        elif _ref_selection_empty(con):
            warnings.warn(
                "filter_by_selection=True but ref_selection is empty; skipping selection filter.",
                UserWarning,
                stacklevel=2,
            )
        else:
            selection_filter = " AND r.ASIN IN (SELECT ASIN FROM ref_selection)"

    ref_selection_join = ""
    if join_ref_selection:
        ref_selection_join = " LEFT JOIN ref_selection s ON r.ASIN = s.ASIN"

    sql = f"""
    WITH base AS (
        SELECT
            r.ASIN,
            r.Product_Title,
            c.fiscal_year   AS Fiscal_Year,
            c.fiscal_month  AS Fiscal_Month,
            c.fiscal_week   AS Fiscal_Week,
            r.Ordered_Revenue,
            r.Ordered_Units,
            COALESCE(r.CCOGS, 0) AS CCOGS,
            COALESCE(r.Promotional_Rebates, 0) AS Promotional_Rebates,
            (r.Ordered_Revenue - COALESCE(r.Promotional_Rebates, 0)) AS Net_Sales,
            (r.Ordered_Revenue - COALESCE(r.Promotional_Rebates, 0) - COALESCE(r.CCOGS, 0)) AS Gross_Profit
        FROM raw_sales r
        {ref_selection_join}
        LEFT JOIN ref_calendar_454 c ON DATE(r.Snapshot_Date) = c.calendar_date
        WHERE c.calendar_date IS NOT NULL
        {selection_filter}
    ),
    agg AS (
        SELECT
            ASIN,
            MAX(Product_Title) AS Product_Title,
            Fiscal_Year,
            Fiscal_Week,
            Fiscal_Month,
            SUM(Ordered_Revenue) AS Ordered_Revenue,
            SUM(Ordered_Units)   AS Ordered_Units,
            SUM(Net_Sales)       AS Net_Sales,
            SUM(CCOGS)           AS CCOGS,
            SUM(Gross_Profit)    AS Gross_Profit,
            100.0 * SUM(Gross_Profit) / NULLIF(SUM(Net_Sales), 0) AS Net_PPM
        FROM base
        GROUP BY ASIN, Fiscal_Year, Fiscal_Week, Fiscal_Month
    ),
    with_ma AS (
        SELECT
            *,
            AVG(Net_PPM) OVER (
                PARTITION BY ASIN
                ORDER BY Fiscal_Year, Fiscal_Week
                ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
            ) AS Net_PPM_4w_MA,
            AVG(Ordered_Units) OVER (
                PARTITION BY ASIN
                ORDER BY Fiscal_Year, Fiscal_Week
                ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
            ) AS Ordered_Units_4w_MA
        FROM agg
    )
    SELECT * FROM with_ma
    """
    return con.execute(sql).df()


def get_454_enriched_sales(
    con: duckdb.DuckDBPyConnection,
    selection_dir: str | Path | None = None,
) -> pd.DataFrame:
    """
    Join raw_sales with 4-5-4 calendar to produce enriched sales (raw_sales_454).
    Returns a DataFrame; does not register.
    """
    _validate_raw_sales(con)
    sel = selection_dir if selection_dir is not None else _DEFAULT_SELECTION_DIR
    _ensure_calendar_registered(con, sel)

    sql = """
    SELECT
        r.ASIN,
        r.SKU,
        r.Snapshot_Date,
        r.Product_Title,
        r.Ordered_Revenue,
        r.Ordered_Units,
        COALESCE(r.CCOGS, 0) AS CCOGS,
        COALESCE(r.Promotional_Rebates, 0) AS Promotional_Rebates,
        r.source_file,
        c.fiscal_year,
        c.fiscal_week,
        c.fiscal_month,
        c.fiscal_quarter
    FROM raw_sales r
    LEFT JOIN ref_calendar_454 c
        ON DATE(r.Snapshot_Date) = c.calendar_date
    """
    return con.execute(sql).df()


def _base_metrics_sql(
    *,
    from_clause: str,
    group_by: str,
    date_cols: str,
    filter_selection: bool,
    con: duckdb.DuckDBPyConnection,
) -> str:
    """Build SQL for profitability metrics. from_clause must alias table as 'r'."""
    selection_filter = ""
    if filter_selection:
        if not _validate_ref_selection(con):
            warnings.warn(
                "filter_by_selection=True but ref_selection not registered or has no ASIN column; "
                "skipping selection filter.",
                UserWarning,
                stacklevel=2,
            )
        elif _ref_selection_empty(con):
            warnings.warn(
                "filter_by_selection=True but ref_selection is empty; skipping selection filter.",
                UserWarning,
                stacklevel=2,
            )
        else:
            selection_filter = " AND r.ASIN IN (SELECT ASIN FROM ref_selection)"

    return f"""
    WITH base AS (
        SELECT
            r.ASIN,
            r.SKU,
            {date_cols}
            r.Ordered_Revenue,
            r.Ordered_Units,
            COALESCE(r.CCOGS, 0) AS CCOGS,
            COALESCE(r.Promotional_Rebates, 0) AS Promotional_Rebates,
            (r.Ordered_Revenue - COALESCE(r.CCOGS, 0) - COALESCE(r.Promotional_Rebates, 0)) AS Gross_Profit
        FROM {from_clause}
        WHERE 1=1 {selection_filter}
    ),
    agg AS (
        SELECT
            {group_by},
            SUM(Ordered_Revenue)     AS Ordered_Revenue,
            SUM(Ordered_Units)       AS Ordered_Units,
            SUM(CCOGS)               AS CCOGS,
            SUM(Promotional_Rebates) AS Promotional_Rebates,
            SUM(Gross_Profit)        AS Gross_Profit
        FROM base
        GROUP BY {group_by}
    )
    SELECT
        *,
        100.0 * Gross_Profit / NULLIF(Ordered_Revenue, 0) AS Gross_Margin_Pct,
        Gross_Profit / NULLIF(Ordered_Units, 0)            AS Unit_Margin,
        Ordered_Revenue / NULLIF(Ordered_Units, 0)         AS Avg_Selling_Price
    FROM agg
    """


def compute_asin_profitability(
    con: duckdb.DuckDBPyConnection,
    *,
    by_454: bool = False,
    filter_by_selection: bool = False,
    selection_dir: str | Path | None = None,
) -> pd.DataFrame:
    """
    ASIN-level profitability: Gross_Revenue, Gross_Profit, Gross_Margin_Pct,
    Unit_Margin, Avg_Selling_Price. Optional 4-5-4 grouping and ref_selection filter.

    Assumes raw_sales (and ref_selection when filter_by_selection) are registered.
    """
    _validate_raw_sales(con)
    sel = selection_dir if selection_dir is not None else _DEFAULT_SELECTION_DIR

    if by_454:
        _ensure_calendar_registered(con, sel)
        from_clause = """
            raw_sales r
            LEFT JOIN ref_calendar_454 c ON DATE(r.Snapshot_Date) = c.calendar_date
        """
        group_by = "ASIN, SKU, fiscal_year, fiscal_week, fiscal_month, fiscal_quarter"
        date_cols = "c.fiscal_year, c.fiscal_week, c.fiscal_month, c.fiscal_quarter,"
        sql = _base_metrics_sql(
            from_clause=from_clause,
            group_by=group_by,
            date_cols=date_cols,
            filter_selection=filter_by_selection,
            con=con,
        )
    else:
        from_clause = "raw_sales r"
        group_by = "ASIN, SKU, Snapshot_Date"
        date_cols = "r.Snapshot_Date,"
        sql = _base_metrics_sql(
            from_clause=from_clause,
            group_by=group_by,
            date_cols=date_cols,
            filter_selection=filter_by_selection,
            con=con,
        )

    return con.execute(sql).df()


def asins_profitability_daily(
    con: duckdb.DuckDBPyConnection,
    filter_by_selection: bool = False,
) -> pd.DataFrame:
    """
    ASIN-level profitability by Snapshot_Date.
    Columns: ASIN, SKU, Snapshot_Date, Ordered_Revenue, Ordered_Units, CCOGS,
    Promotional_Rebates, Gross_Profit, Gross_Margin_Pct, Unit_Margin, Avg_Selling_Price.
    """
    return compute_asin_profitability(
        con, by_454=False, filter_by_selection=filter_by_selection
    )


def asins_profitability_454(
    con: duckdb.DuckDBPyConnection,
    filter_by_selection: bool = False,
) -> pd.DataFrame:
    """
    ASIN-level profitability by 4-5-4 period (fiscal_year, fiscal_week, fiscal_month, fiscal_quarter).
    Same metric columns as daily, aggregated per period.
    """
    return compute_asin_profitability(
        con, by_454=True, filter_by_selection=filter_by_selection
    )
