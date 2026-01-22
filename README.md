# Rapid AI-Driven Profitability Engine

A zero-setup Streamlit application for analyzing product profitability using sales data, reference selections, and 4-5-4 retail calendar calculations.

## Overview

This application processes sales data from multiple sources, normalizes it to a canonical schema, enriches it with fiscal calendar information, and calculates profitability metrics including Net PPM (Net Profit Percentage Margin). The dashboard provides interactive visualizations, KPI metrics, and identifies low-profitability items.

## Features

- **Automated Data Ingestion**: Recursively scans dropzone directory for CSV/Excel files
- **Data Normalization**: Transforms heterogeneous data formats into a canonical schema
- **4-5-4 Calendar Integration**: Maps calendar dates to fiscal weeks, months, and quarters
- **Profitability Analytics**: Calculates Net PPM, Gross Profit, and moving averages
- **Interactive Dashboard**: 
  - KPI metrics (Total Revenue, Average Net PPM, Total Units)
  - Trend charts comparing weekly Net PPM vs 4-week moving average
  - Filtered data grid for low-profitability items
- **Auto-Refresh Detection**: Monitors dropzone for new files and prompts for refresh
- **System Health Monitoring**: Displays data quality metrics and processing status

## Directory Structure

```
Prototype/
├── 00_selection/          # Reference data directory
│   ├── calendar_454.csv   # 4-5-4 fiscal calendar lookup (auto-generated)
│   ├── vendor_map.csv     # Vendor code mappings (optional)
│   └── vendors/           # Vendor selection files (optional)
│       └── *.csv          # ASIN watch lists per vendor
├── 01_dropzone/           # Sales data dropzone
│   ├── daily/             # Daily order summaries
│   ├── historical/        # Historical aggregated data (T12M)
│   └── weekly/            # Weekly performance data
├── data/                  # Auto-generated data directory
│   ├── selection/         # Selection data cache
│   └── dropzone/          # Legacy dropzone (not used)
├── src/                   # Source code modules
│   ├── analytics.py       # DuckDB analytics and metrics calculation
│   ├── calendar_454.py    # 4-5-4 calendar helper functions
│   ├── ingestion.py       # Data scanning and loading
│   ├── normalization.py   # Data normalization to canonical schema
│   └── visuals.py         # Dashboard visualization functions
├── app.py                 # Main Streamlit application
├── main.py                # CLI pipeline orchestration
├── setup_dirs.py          # Directory initialization script
└── requirements.txt       # Python dependencies
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Initialize Directories

```bash
python setup_dirs.py
```

This creates the necessary directory structure if it doesn't exist.

### 3. Prepare Data Files

#### Dropzone (`01_dropzone/`)

Place your sales data files in the `01_dropzone/` directory. The application will recursively scan all subdirectories for CSV and Excel files.

**Supported file types:**
- CSV files (`.csv`)
- Excel files (`.xlsx`, `.xls`)

**Expected data columns** (will be auto-mapped):
- ASIN (product identifier)
- SKU (stock keeping unit)
- Product Title/Name
- Ordered Revenue / Product Sales / Net OPS
- Ordered Units / Net Units
- CCOGS / CP (Cost of Goods Sold)
- Promotional Rebates / Deal GMS
- Date/Snapshot Date columns

**File naming patterns:**
- Daily files: `*DailyOrdersSummary*.csv`
- Historical files: `T12M_*.csv` or `*historical*.csv`
- Weekly files: `*W49*.csv` or `*weekly*.csv`

#### Selection Directory (`00_selection/`)

**Required:**
- `calendar_454.csv` - Automatically generated on first run if missing

**Optional:**
- `vendor_map.csv` or `vendor_map.xlsx` - Vendor code mappings
- `vendors/*.csv` - Vendor-specific ASIN selection lists

The application will extract ASIN columns from vendor files to create a reference selection list.

## Usage

### Running the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

### Using the Dashboard

1. **Place Data Files**: Copy your sales data CSV/Excel files into `01_dropzone/` (or subdirectories)

2. **Run Analysis**: 
   - Click the **"Run Analysis"** button in the sidebar
   - The system will:
     - Scan the dropzone for data files
     - Load reference data from `00_selection/`
     - Normalize and register data in DuckDB
     - Calculate profitability metrics
     - Display the dashboard

3. **Configure Threshold**: 
   - Adjust the **"Low Profitability Threshold"** slider (default: 20%)
   - Items with Net PPM below this threshold appear in the Low Profitability section

4. **Monitor System Health**:
   - Expand the **"System Health"** section in the sidebar
   - View data quality metrics, record counts, and date ranges

5. **Auto-Refresh**:
   - When new files are added to the dropzone, a warning appears
   - Click **"Refresh Analysis"** to reprocess with new data

### Dashboard Components

#### KPI Metrics
- **Total Revenue**: Sum of all Ordered Revenue
- **Average Net PPM**: Mean Net Profit Percentage Margin across all records
- **Total Units**: Sum of all Ordered Units

#### Trend Chart
- Line chart showing:
  - **Weekly Net PPM**: Average Net PPM per fiscal week
  - **4-Week Moving Average**: Smoothed trend line

#### Low Profitability Items
- Filtered data grid showing products with Net PPM below the threshold
- Columns: ASIN, Product Title, Fiscal Year/Week, Net PPM, Revenue, Units, Gross Profit
- Sorted by Net PPM (lowest first)

## Data Processing Pipeline

```
01_dropzone/ files
    ↓
[ingestion.scan_dropzone()]
    ↓
[normalization.normalize_data()]
    ↓
Canonical Schema DataFrame
    ↓
[DuckDB: raw_sales table]
    ↓
[analytics.calculate_metrics()]
    ↓
Profitability Metrics DataFrame
    ↓
[visuals.render_dashboard()]
    ↓
Interactive Dashboard
```

## Canonical Schema

All data is normalized to the following schema:

- `ASIN` (string): Product identifier
- `SKU` (string): Stock keeping unit
- `Snapshot_Date` (datetime): Date of the sales record
- `Product_Title` (string): Product name/description
- `Ordered_Revenue` (float): Total revenue
- `Ordered_Units` (int): Number of units sold
- `CCOGS` (float): Cost of goods sold
- `Promotional_Rebates` (float): Promotional discounts/rebates
- `source_file` (string): Original file path

## Analytics Output

The `calculate_metrics()` function returns a DataFrame with:

- `ASIN`, `Product_Title`
- `Fiscal_Year`, `Fiscal_Week`, `Fiscal_Month`
- `Ordered_Revenue`, `Ordered_Units`
- `Net_Sales` (Revenue - Promotional Rebates)
- `CCOGS`, `Gross_Profit`
- `Net_PPM` (Net Profit Percentage Margin)
- `Net_PPM_4w_MA` (4-week moving average of Net PPM)
- `Ordered_Units_4w_MA` (4-week moving average of units)

## Troubleshooting

### "No data found in dropzone"
- Ensure files are placed in `01_dropzone/` directory
- Check that files have `.csv`, `.xlsx`, or `.xls` extensions
- Verify file permissions allow reading

### "Missing required columns" error
- Check that your data files contain columns that can be mapped to the canonical schema
- Review the column mapping logic in `src/normalization.py`
- Ensure date columns are present or dates can be extracted from filenames

### "Analysis calculation produced no results"
- Verify that date columns can be mapped to the 4-5-4 calendar
- Check that `calendar_454.csv` exists in `00_selection/` (auto-generated if missing)
- Ensure data has valid ASIN values

### Dashboard shows "No data available"
- Run the analysis first using the "Run Analysis" button
- Check System Health section for error messages
- Verify data files were successfully processed

## Command Line Interface

For batch processing without the UI:

```bash
python main.py
```

This runs the full pipeline and prints summary statistics to the console.

## Development

### Module Structure

- **`src/ingestion.py`**: File scanning, loading, and concatenation
- **`src/normalization.py`**: Column mapping, data type conversion, date extraction
- **`src/calendar_454.py`**: 4-5-4 fiscal calendar generation and lookup
- **`src/analytics.py`**: DuckDB SQL queries for profitability calculations
- **`src/visuals.py`**: Streamlit dashboard rendering functions

### Adding New Data Sources

1. Update column mappings in `src/normalization.py` (`_COLUMN_MAPPINGS`)
2. Add file type detection logic if needed (`_detect_file_type`)
3. Implement custom handler if required (`_handle_*_file`)

## License

[Add your license information here]

## Support

[Add support contact information here]
