# WBR Pipeline - Weekly Business Review Generator

A command-line pipeline for generating Weekly Business Review (WBR) reports from Amazon sales performance data. The pipeline processes raw CSV files, parses dynamic headers, transforms data into a canonical format, analyzes performance metrics with RAG (Red/Amber/Green) status, and generates formatted Excel workbooks with insights and validation reports.

## Overview

The WBR Pipeline automates the generation of weekly business review reports by:

- **Automated Header Parsing**: Intelligently parses dynamic column headers with week numbers and metric types (100% success rate)
- **Data Transformation**: Converts raw performance data into standardized WBR format with week-over-week calculations
- **Performance Analysis**: Evaluates metrics against thresholds and assigns RAG status (Red/Amber/Green)
- **Insight Generation**: Generates priority-scored insights with actionable commentary (Amendment C)
- **History Archiving**: Maintains historical data in DuckDB for WoW calculations (Amendment B)
- **Excel Output**: Creates formatted workbooks with multiple sheets including Validation Tab (Amendment A) and Executive Summary

## Features

- **100% Header Parse Rate**: Handles complex header formats including week-based, T12M historical, and date range patterns
- **RAG Status Evaluation**: Automatic Red/Amber/Green classification based on configurable thresholds
- **Priority Scoring**: Financial impact-weighted scoring to identify top 3 critical issues
- **Week-over-Week Analysis**: Calculates WoW changes using embedded data or historical archive
- **Validation Tab**: Transparent audit trail of header parsing with confidence scores
- **Executive Summary**: High-level overview with key metrics and top issues
- **Vendor Mapping**: Joins raw data with vendor reference files for attribution
- **DuckDB Archive**: Persistent storage of historical metrics for trend analysis

## Directory Structure

```
Prototype/
├── 00_selection/              # Reference data directory
│   ├── calendar_454.csv      # 4-5-4 fiscal calendar (optional)
│   ├── vendor_map.csv         # Vendor code to name mappings (required)
│   └── vendors/               # Vendor-specific ASIN selection files
│       └── *.csv              # ASIN watch lists per vendor
│
├── 01_dropzone/               # Raw data drop zone
│   ├── daily/                 # Daily order summaries
│   ├── historical/            # Historical aggregated data (T12M format)
│   └── weekly/
│       ├── performance/       # Weekly performance data (primary input)
│       └── ratings/           # Weekly ratings data
│
├── 02_output/                 # Generated WBR workbooks
│   └── WBR_*.xlsx            # Output files with timestamp
│
├── config/                    # Configuration files (JSON)
│   ├── column_mapping.json   # Metric name to WBR column mappings
│   ├── thresholds.json       # RAG thresholds for metrics
│   └── impact_weights.json   # Financial impact weights for priority scoring
│
├── data/                      # Data persistence
│   └── history.duckdb        # Historical metrics archive (DuckDB)
│
├── src/                       # Source code modules
│   ├── config.py             # Configuration management
│   ├── header_parser.py       # Dynamic header parsing
│   ├── data_loader.py         # File loading and history handler
│   ├── data_transformer.py    # Data transformation and WoW calculations
│   ├── analyzer.py            # Performance analysis and RAG evaluation
│   ├── insight_generator.py   # Priority scoring and commentary generation
│   └── excel_formatter.py    # Excel workbook creation
│
├── tests/                     # Test suite
│   ├── test_header_parser.py   # Unit tests for header parser
│   └── test_pipeline.py       # Integration tests
│
├── main.py                    # Pipeline orchestrator (entry point)
└── requirements.txt           # Python dependencies
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `duckdb>=0.9.0` - Historical data archive
- `pandas>=2.0.0` - Data manipulation
- `openpyxl>=3.0.0` - Excel file reading
- `xlsxwriter>=3.0.0` - Excel workbook creation
- `numpy>=1.24.0` - Numerical operations

### 2. Prepare Data Files

#### Dropzone (`01_dropzone/weekly/performance/`)

Place your weekly performance CSV files in this directory. The pipeline will automatically detect the latest file.

**Supported file formats:**
- CSV files (`.csv`) with delimiter sniffing
- Excel files (`.xlsx`, `.xls`, `.xlsm`) with optional sheet selection
- JSON files (`.json`)

**Expected data structure:**
- Must contain `ASIN` column for product identification
- Headers with week numbers: `Metric($)(Week N )` or `Metric(Week N )(%)`
- Multiple weeks per file supported (e.g., `W49-W52`)

**File naming examples:**
- `KY2O0-W49-W52.csv` - Vendor KY2O0, weeks 49-52
- `SWM6A-Nov2-23-2025.csv` - Date range format
- `T072Y-W48-2025.csv` - Single week with year

#### Selection Directory (`00_selection/`)

**Required:**
- `vendor_map.csv` - Vendor code mappings (required for vendor attribution)

  **Format:**
  ```csv
  vendor_code,vendor_name
  KY2O0,McLane
  722XP,AlMaya
  SWM6A,LinkMax
  T072Y,Champions
  WJTP1,WJTowell
  ```
  
  **Columns:**
  - `vendor_code` (required) - Vendor identifier extracted from filenames (e.g., "KY2O0", "722XP")
  - `vendor_name` (required) - Human-readable vendor display name
  
  **Generation:**
  The vendor_map.csv can be auto-generated from filenames in the dropzone:
  ```bash
  python scripts/generate_vendor_map.py
  ```
  
  This script:
  - Scans `01_dropzone/` for vendor codes in filenames (e.g., "KY2O0-W49-W52.csv" → "KY2O0")
  - Preserves existing vendor names if vendor_map.csv already exists
  - Uses known vendor mappings for common codes
  - Outputs to `00_selection/vendor_map.csv`

**Optional:**
- `vendors/*.csv` - Vendor-specific ASIN selection files
  - Must contain `ASIN` column
  - Used to filter/attribute data to specific vendors

## Usage

### Basic Usage

Process the latest file in the dropzone:

```bash
python main.py
```

### Advanced Usage

Process a specific file:

```bash
python main.py --file "01_dropzone/weekly/performance/KY2O0-W49-W52.csv"
```

Filter by vendor code:

```bash
python main.py --vendor KY2O0
```

Override week number:

```bash
python main.py --week 49
```

Specify custom output directory:

```bash
python main.py --output "custom_output/"
```

Skip data archiving:

```bash
python main.py --no-archive
```

### Command-Line Options

```
python main.py [OPTIONS]

Options:
  --file, -f PATH          Path to raw data file (auto-detects latest if not provided)
  --vendor, -v CODE        Vendor code filter
  --week, -w NUMBER        Week number override
  --output, -o PATH        Output directory (default: 02_output/)
  --no-archive             Skip archiving data for WoW calculations
```

## Output Structure

The pipeline generates Excel workbooks with the following sheets:

### 1. Executive Summary
- Key metrics overview
- RAG status distribution
- Top 3 priority issues with commentary
- Week number and generation timestamp

### 2. Weekly Business Review
- Main data sheet with all metrics
- RAG color coding (Red/Amber/Green)
- Conditional formatting based on thresholds
- Frozen header row for easy navigation

### 3. Validation Tab
- Complete header parsing audit trail
- Confidence scores for each parsed header
- Flag status (OK/WARNING/ERROR)
- Mapped WBR column names

### 4. Raw Data Archive
- Unmodified source data
- Preserved for reference and debugging

### 5. Configuration Log
- Settings used during processing
- Thresholds and impact weights applied
- Source file information

## Data Processing Pipeline

```
[Raw CSV File]
    ↓
[Header Parser] → Parse dynamic headers with week numbers
    ↓
[Data Loader] → Load raw data + reference files
    ↓
[Data Transformer] → Map columns, unpivot weeks, calculate WoW
    ↓
[History Handler] → Retrieve previous week data (if needed)
    ↓
[Analyzer] → Evaluate RAG status against thresholds
    ↓
[Insight Generator] → Calculate priority scores, generate commentary
    ↓
[Excel Formatter] → Create formatted workbook
    ↓
[History Archive] → Save current week for future WoW calculations
    ↓
[WBR Workbook Output]
```

## Configuration

### Thresholds (`config/thresholds.json`)

Define RAG thresholds for each metric (percentages use decimal form):

```json
{
  "Net_PPM": {
    "red": 0.05,
    "amber": 0.15,
    "direction": "higher_is_better"
  }
}
```

### Impact Weights (`config/impact_weights.json`)

Set financial impact weights for priority scoring:

```json
{
  "Ordered_Revenue": {
    "weight": 10,
    "rationale": "Direct financial impact - highest priority"
  }
}
```

### Column Mapping (`config/column_mapping.json`)

Map parsed metric names to canonical WBR columns:

```json
{
  "Product GMS": "Ordered_Revenue",
  "Net PPM": "Net_PPM"
}
```

## Testing

Run the test suite to validate the pipeline:

```bash
# Unit tests (header parser)
python tests/test_header_parser.py

# Integration tests (full pipeline)
python tests/test_pipeline.py
```

**Current Test Status:** ✅ 41/41 tests passing (100%)

## Troubleshooting

### "No data files found in drop zone!"

- Ensure files are placed in `01_dropzone/weekly/performance/`
- Check file extensions (`.csv`, `.xlsx`, `.xls`, `.xlsm`, `.json`)
- Verify file permissions allow reading

### "Required column 'ASIN' not found"

- Ensure your CSV file contains an `ASIN` column
- Check for case sensitivity (should be uppercase `ASIN`)

### "Parse success rate below 95%"

- Review the Validation Tab in the output workbook
- Check for new header formats not covered by parser
- Header parser supports 9 distinct patterns (100% success rate on tested data)

### "Configuration errors"

- Verify `vendor_map.csv` exists in `00_selection/`
- Check that `config/` directory contains JSON files
- Run `python src/config.py` to validate configuration

### Output file not created

- Check write permissions on `02_output/` directory
- Verify sufficient disk space
- Review error messages in console output

## Key Metrics

The pipeline processes and analyzes:

- **Revenue Metrics**: Product GMS, Net Receipts, Deal GMS
- **Unit Metrics**: Ordered Units, Net Receipts (Units)
- **Performance Metrics**: Net PPM, PPM, Contribution Margin
- **Traffic Metrics**: Glance Views, Sessions
- **Availability Metrics**: Fill Rate, SoROOS%, Vendor Confirmation Rate
- **Week-over-Week Changes**: Calculated for all metrics

## RAG Status Logic

- **RED**: Metric value below red threshold (critical issue)
- **AMBER**: Metric value below amber threshold (needs attention)
- **GREEN**: Metric value meets or exceeds amber threshold (healthy)

Direction can be "higher_is_better" (e.g., Revenue, PPM) or "lower_is_better" (e.g., SoROOS%).
Percent values are stored as decimals (e.g., 0.12 = 12%) and thresholds should use the same convention.

## History Archive

The pipeline maintains a DuckDB archive (`data/history.duckdb`) of processed metrics. This enables:

- Week-over-week calculations when raw files don't contain embedded WoW data
- Historical trend analysis
- Multi-week comparisons

Data is automatically archived after each successful pipeline run (unless `--no-archive` is used).

## Development

### Module Overview

- **`src/config.py`**: Centralized configuration management
- **`src/header_parser.py`**: Regex-based header parsing with confidence scoring
- **`src/data_loader.py`**: File detection, loading, and HistoryHandler (Amendment B)
- **`src/data_transformer.py`**: Column mapping, unpivot, join operations, WoW calculations
- **`src/analyzer.py`**: RAG evaluation and threshold checking
- **`src/insight_generator.py`**: Priority scoring and dynamic commentary (Amendment C)
- **`src/excel_formatter.py`**: Workbook creation with Validation Tab (Amendment A) and Executive Summary

### Adding New Metrics

1. Update `config/column_mapping.json` to map new metric names
2. Add thresholds in `config/thresholds.json` if RAG evaluation is needed
3. Add impact weights in `config/impact_weights.json` for priority scoring
4. Test with sample data files

### Extending Header Patterns

The header parser uses regex patterns to identify metrics. To add new patterns:

1. Review existing patterns in `src/header_parser.py`
2. Add new regex pattern to `HeaderParser` class
3. Update pattern matching logic
4. Add unit tests in `tests/test_header_parser.py`

## License

[Add your license information here]

## Support

[Add support contact information here]
