"""
Optional pipeline orchestration.

1. Run Phase 2 ingestion (scan_dropzone, scan_selection) and register
   raw_sales + ref_selection — when available.
2. Call Phase 3 analytics: get_454_enriched_sales, compute_asin_profitability.

Without Phase 2, uses mock raw_sales/ref_selection for smoke-testing analytics.
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pandas as pd

# Phase 3 analytics (always available)
from src.analytics import (
    asins_profitability_454,
    asins_profitability_daily,
    compute_asin_profitability,
    get_454_enriched_sales,
)


def _mock_raw_sales() -> pd.DataFrame:
    """Minimal canonical raw_sales for demo when Phase 2 is not run."""
    return pd.DataFrame({
        "ASIN": ["A1", "A1", "A2"],
        "SKU": ["S1", "S1", "S2"],
        "Snapshot_Date": pd.to_datetime(["2024-03-01", "2024-03-02", "2024-03-01"]),
        "Product_Title": ["P1", "P1", "P2"],
        "Ordered_Revenue": [100.0, 50.0, 200.0],
        "Ordered_Units": [2, 1, 4],
        "CCOGS": [40.0, 20.0, 80.0],
        "Promotional_Rebates": [5.0, 2.5, 10.0],
        "source_file": ["mock", "mock", "mock"],
    })


def _mock_ref_selection() -> pd.DataFrame:
    """Minimal ref_selection (ASIN watch list) for demo."""
    return pd.DataFrame({"ASIN": ["A1"]})


def run(selection_dir: str | Path = "00_selection") -> None:
    con = duckdb.connect(":memory:")

    # Phase 2: register raw_sales and ref_selection
    try:
        from src import ingestion  # type: ignore[attr-defined]
        
        # Scan dropzone for raw sales data
        raw_df = ingestion.scan_dropzone(path="01_dropzone")
        
        # Scan selection for reference data
        selection_data = ingestion.scan_selection(path="00_selection")
        sel_df = selection_data.get("ref_selection", pd.DataFrame(columns=["ASIN"]))
        
        # Register with DuckDB
        if not raw_df.empty:
            con.register("raw_sales", raw_df)
            print(f"Phase 2: loaded {len(raw_df)} rows from 01_dropzone")
        else:
            raw_df = _mock_raw_sales()
            con.register("raw_sales", raw_df)
            print("Phase 2: dropzone empty, using mock raw_sales")
        
        if not sel_df.empty:
            con.register("ref_selection", sel_df)
            print(f"Phase 2: loaded {len(sel_df)} ASINs from 00_selection")
        else:
            sel_df = _mock_ref_selection()
            con.register("ref_selection", sel_df)
            print("Phase 2: selection empty, using mock ref_selection")
            
    except ImportError as e:
        raw_df = _mock_raw_sales()
        sel_df = _mock_ref_selection()
        con.register("raw_sales", raw_df)
        con.register("ref_selection", sel_df)
        print(f"Phase 2: using mock data (ingestion import failed: {e})")
    except Exception as e:
        raw_df = _mock_raw_sales()
        sel_df = _mock_ref_selection()
        con.register("raw_sales", raw_df)
        con.register("ref_selection", sel_df)
        print(f"Phase 2: using mock data (ingestion error: {e})")

    # Phase 3: analytics
    enriched = get_454_enriched_sales(con, selection_dir=selection_dir)
    print(f"4-5-4 enriched rows: {len(enriched)}")

    daily = asins_profitability_daily(con)
    print(f"Daily profitability rows: {len(daily)}")

    by_454 = asins_profitability_454(con)
    print(f"4-5-4 profitability rows: {len(by_454)}")

    con.close()


if __name__ == "__main__":
    sel = sys.argv[1] if len(sys.argv) > 1 else "00_selection"
    run(selection_dir=Path(sel))
