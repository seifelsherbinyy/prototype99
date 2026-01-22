
import pandas as pd
import duckdb
from pathlib import Path
from src import ingestion, normalization, analytics, calendar_454

def showcase():
    print("--- SHOWCASING SRC MODULES ---")
    
    # 1. Ingestion Showcase
    print("\n1. [src/ingestion.py] Scanning dropzone...")
    raw_df = ingestion.scan_dropzone(path="01_dropzone")
    print(f"Loaded {len(raw_df)} rows from 01_dropzone.")
    if not raw_df.empty:
        print("Sample raw data columns:", raw_df.columns.tolist())
        print("Sample data (first 2 rows):\n", raw_df.head(2))

    # 2. Normalization Showcase (implicitly done in scan_dropzone, but let's show schema)
    print("\n2. [src/normalization.py] Canonical Schema:")
    schema = normalization.get_canonical_schema()
    print(schema)

    # 3. Calendar Showcase
    print("\n3. [src/calendar_454.py] 4-5-4 Calendar Mapping:")
    test_date = pd.to_datetime("2024-03-15").date()
    mapping = calendar_454.date_to_454(test_date)
    print(f"Date {test_date} maps to: {mapping}")

    # 4. Analytics Showcase
    print("\n4. [src/analytics.py] Profitability Metrics:")
    con = duckdb.connect(":memory:")
    if not raw_df.empty:
        con.register("raw_sales", raw_df)
        # Scan selection for reference data
        selection_data = ingestion.scan_selection(path="00_selection")
        sel_df = selection_data.get("ref_selection", pd.DataFrame(columns=["ASIN"]))
        if not sel_df.empty:
            con.register("ref_selection", sel_df)
        
        # Calculate metrics
        metrics_df = analytics.calculate_metrics(con)
        print(f"Calculated metrics for {len(metrics_df)} rows.")
        if not metrics_df.empty:
            print("Top 5 items by Net PPM:")
            print(metrics_df.sort_values("Net_PPM", ascending=False)[["ASIN", "Product_Title", "Net_PPM"]].head())
    else:
        print("No data available for analytics showcase.")
    
    con.close()
    print("\n--- SHOWCASE COMPLETE ---")

if __name__ == "__main__":
    showcase()
