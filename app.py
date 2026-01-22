"""Rapid AI-Driven Profitability Engine — Zero-Setup Streamlit app."""

import datetime
import os
from io import BytesIO
from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st

from setup_dirs import ensure_dirs

# Import analytics and visualization modules
from src import analytics, ingestion
from src.logger import get_logger
from src.visuals import render_dashboard

# Get logger
logger = get_logger(__name__)

ensure_dirs()

st.set_page_config(
    page_title="Rapid AI-Driven Profitability Engine",
    layout="wide",
)

# Initialize session state
if "main_df" not in st.session_state:
    st.session_state.main_df = None
if "is_loaded" not in st.session_state:
    st.session_state.is_loaded = False
if "processing_log" not in st.session_state:
    st.session_state.processing_log = []
if "analytics_df" not in st.session_state:
    st.session_state.analytics_df = None
if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False
if "duckdb_con" not in st.session_state:
    st.session_state.duckdb_con = None
if "last_dropzone_mtime" not in st.session_state:
    st.session_state.last_dropzone_mtime = None
if "files_processed_count" not in st.session_state:
    st.session_state.files_processed_count = 0
if "analysis_timestamp" not in st.session_state:
    st.session_state.analysis_timestamp = None

PROJECT_ROOT = Path(__file__).resolve().parent
DROPZONE = PROJECT_ROOT / "01_dropzone"  # Standardized to match ingestion.py
SELECTION_DIR = PROJECT_ROOT / "00_selection"
ALLOWED_SUFFIXES = (".csv", ".xlsx", ".xls")


def get_dropzone_mtime() -> float | None:
    """Get the maximum modification time of files in the dropzone directory."""
    if not DROPZONE.exists() or not DROPZONE.is_dir():
        return None

    max_mtime = 0.0
    has_files = False

    for file_path in DROPZONE.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in ALLOWED_SUFFIXES:
            try:
                mtime = os.path.getmtime(file_path)
                max_mtime = max(max_mtime, mtime)
                has_files = True
            except OSError:
                continue

    return max_mtime if has_files else None


def check_dropzone_changed() -> bool:
    """Check if dropzone has been modified since last analysis."""
    current_mtime = get_dropzone_mtime()
    if current_mtime is None:
        return False

    if st.session_state.last_dropzone_mtime is None:
        return False

    return current_mtime > st.session_state.last_dropzone_mtime


def run_analysis() -> tuple[pd.DataFrame | None, str]:
    """
    Run the full analytics pipeline.

    Returns:
        Tuple of (analytics DataFrame, status message)
    """
    try:
        # Initialize DuckDB connection with error handling for database locks
        if st.session_state.duckdb_con is None:
            try:
                st.session_state.duckdb_con = duckdb.connect(":memory:")
            except duckdb.IOException as e:
                error_msg = "CRITICAL: Database is locked. Please kill any other running Python instances."
                logger.critical(error_msg)
                logger.debug(f"DuckDB IOException details: {e}", exc_info=True)
                return None, error_msg

        con = st.session_state.duckdb_con

        # Step 1: Scan dropzone
        with st.spinner("Scanning dropzone for sales data..."):
            dropzone_df = ingestion.scan_dropzone(path=str(DROPZONE))

        if dropzone_df.empty:
            return None, "No data found in dropzone. Please add CSV/Excel files to `01_dropzone/`."

        # Step 2: Scan selection directory
        with st.spinner("Scanning selection directory for reference data..."):
            selection_data = ingestion.scan_selection(path=str(SELECTION_DIR))

        # Step 3: Register tables in DuckDB
        with st.spinner("Registering data in DuckDB..."):
            con.register("raw_sales", dropzone_df)

            ref_selection_df = selection_data.get("ref_selection", pd.DataFrame(columns=["ASIN"]))
            if not ref_selection_df.empty:
                con.register("ref_selection", ref_selection_df)

        # Step 4: Calculate metrics
        with st.spinner("Calculating profitability metrics..."):
            analytics_df = analytics.calculate_metrics(
                con,
                selection_dir=str(SELECTION_DIR),
                filter_by_selection=False,  # Set to True if you want to filter by ref_selection
            )

        if analytics_df.empty:
            return None, "Analytics calculation produced no results. Check data quality."

        # Update session state
        st.session_state.analytics_df = analytics_df
        st.session_state.analysis_complete = True
        st.session_state.last_dropzone_mtime = get_dropzone_mtime()
        st.session_state.analysis_timestamp = datetime.datetime.now()

        # Count files processed (approximate)
        file_count = len(list(DROPZONE.rglob("*"))) if DROPZONE.exists() else 0
        st.session_state.files_processed_count = file_count

        return analytics_df, f"Analysis complete! Processed {len(analytics_df)} records."

    except Exception as e:
        return None, f"Error during analysis: {str(e)}"


def get_system_health() -> dict:
    """Get system health metrics."""
    health = {
        "duckdb_connected": st.session_state.duckdb_con is not None,
        "analysis_complete": st.session_state.analysis_complete,
        "dropzone_exists": DROPZONE.exists(),
        "selection_exists": SELECTION_DIR.exists(),
        "files_processed": st.session_state.files_processed_count,
        "last_analysis": st.session_state.analysis_timestamp.isoformat() if st.session_state.analysis_timestamp else None,
    }

    if st.session_state.analytics_df is not None and not st.session_state.analytics_df.empty:
        df = st.session_state.analytics_df
        health["data_quality"] = {
            "total_records": len(df),
            "null_net_ppm": int(df["Net_PPM"].isna().sum()) if "Net_PPM" in df.columns else 0,
            "null_revenue": int(df["Ordered_Revenue"].isna().sum()) if "Ordered_Revenue" in df.columns else 0,
            "date_range": {
                "min_fiscal_year": int(df["Fiscal_Year"].min()) if "Fiscal_Year" in df.columns else None,
                "max_fiscal_year": int(df["Fiscal_Year"].max()) if "Fiscal_Year" in df.columns else None,
                "min_fiscal_week": int(df["Fiscal_Week"].min()) if "Fiscal_Week" in df.columns else None,
                "max_fiscal_week": int(df["Fiscal_Week"].max()) if "Fiscal_Week" in df.columns else None,
            },
            "unique_asins": int(df["ASIN"].nunique()) if "ASIN" in df.columns else 0,
        }
    else:
        health["data_quality"] = None

    return health


# --- Sidebar: Analysis Control ---

st.sidebar.header("Analysis Control")

# Low profitability threshold slider
low_profitability_threshold = st.sidebar.slider(
    "Low Profitability Threshold (%)",
    min_value=0.0,
    max_value=100.0,
    value=20.0,
    step=0.5,
    help="Items with Net PPM below this threshold will be shown in the Low Profitability section.",
)

# Run Analysis button
if st.sidebar.button("Run Analysis", type="primary", use_container_width=True):
    result_df, message = run_analysis()
    if result_df is not None:
        st.sidebar.success(message)
    else:
        st.sidebar.error(message)

# Auto-refresh detection
if st.session_state.analysis_complete and check_dropzone_changed():
    st.sidebar.warning("⚠️ New data detected in dropzone!")
    if st.sidebar.button("Refresh Analysis", use_container_width=True):
        result_df, message = run_analysis()
        if result_df is not None:
            st.sidebar.success(message)
            st.rerun()
        else:
            st.sidebar.error(message)

# Debug Mode section
st.sidebar.divider()
enable_debug = st.sidebar.checkbox("Enable Verbose Debugging", value=False)

if enable_debug:
    debug_log_path = Path(__file__).resolve().parent / "system_debug.log"
    with st.sidebar.expander("Debug Log", expanded=True):
        if debug_log_path.exists():
            try:
                with open(debug_log_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    # Show last 50 lines
                    last_lines = lines[-50:] if len(lines) > 50 else lines
                    if last_lines:
                        st.code("".join(last_lines), language="text")
                    else:
                        st.info("Debug log is empty.")
            except Exception as e:
                st.error(f"Error reading debug log: {e}")
        else:
            st.info("Debug log file not found. Run an analysis to generate logs.")

# System Health section
with st.sidebar.expander("System Health", expanded=False):
    health = get_system_health()

    if health["duckdb_connected"]:
        st.success("✓ DuckDB Connected")
    else:
        st.info("○ DuckDB Not Connected")

    if health["analysis_complete"]:
        st.success("✓ Analysis Complete")
        if health["last_analysis"]:
            st.caption(f"Last run: {health['last_analysis']}")
    else:
        st.info("○ Analysis Not Run")

    if health["dropzone_exists"]:
        st.success(f"✓ Dropzone: {DROPZONE.name}")
    else:
        st.warning(f"✗ Dropzone not found: {DROPZONE}")

    if health["selection_exists"]:
        st.success(f"✓ Selection: {SELECTION_DIR.name}")
    else:
        st.info(f"○ Selection: {SELECTION_DIR.name} (optional)")

    if health["data_quality"]:
        dq = health["data_quality"]
        st.divider()
        st.write("**Data Quality:**")
        st.write(f"• Total Records: {dq['total_records']:,}")
        st.write(f"• Unique ASINs: {dq['unique_asins']:,}")
        st.write(f"• Null Net PPM: {dq['null_net_ppm']}")
        st.write(f"• Null Revenue: {dq['null_revenue']}")

        if dq["date_range"]["min_fiscal_year"]:
            st.write(
                f"• Date Range: FY{dq['date_range']['min_fiscal_year']}-W{dq['date_range']['min_fiscal_week']} "
                f"to FY{dq['date_range']['max_fiscal_year']}-W{dq['date_range']['max_fiscal_week']}"
            )

# Legacy data source control (kept for backward compatibility)
st.sidebar.divider()
st.sidebar.header("Legacy Data Source Control")
mode = st.sidebar.radio(
    "Source",
    ["Manual Upload", "Local Directory Scan"],
    label_visibility="collapsed",
)


def _load_csv(buf: BytesIO) -> pd.DataFrame:
    return pd.read_csv(buf)


def _load_excel(buf: BytesIO) -> pd.DataFrame:
    return pd.read_excel(buf, engine="openpyxl")


def load_from_buffer(f) -> pd.DataFrame:
    """Load a single file-like (UploadedFile or BytesIO) into a DataFrame."""
    buf = BytesIO(f.read())
    name = getattr(f, "name", "")
    suffix = Path(name).suffix.lower() if name else ""
    if suffix == ".csv":
        return _load_csv(buf)
    if suffix in (".xlsx", ".xls"):
        return _load_excel(buf)
    raise ValueError(f"Unsupported format: {suffix or 'unknown'}")


def load_from_path(path: Path) -> pd.DataFrame:
    """Load a file from disk into a DataFrame."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path, engine="openpyxl")
    raise ValueError(f"Unsupported format: {suffix}")


def log_ingestion(source: str, files: list[str]) -> None:
    st.session_state.processing_log.append(
        {
            "timestamp": datetime.datetime.now().isoformat(),
            "source": source,
            "files": list(files),
        }
    )


def apply_ingestion(df: pd.DataFrame, source: str, files: list[str]) -> None:
    st.session_state.main_df = df
    st.session_state.is_loaded = True
    log_ingestion(source, files)


if mode == "Manual Upload":
    uploaded = st.sidebar.file_uploader(
        "Upload CSV or Excel",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True,
    )
    if uploaded:
        load_btn = st.sidebar.button("Load into session")
        if load_btn:
            dfs = []
            names = []
            for f in uploaded:
                try:
                    dfs.append(load_from_buffer(f))
                    names.append(f.name)
                except Exception as e:
                    st.sidebar.error(f"Failed to load {f.name}: {e}")
            if dfs:
                combined = pd.concat(dfs, ignore_index=True)
                apply_ingestion(combined, "manual_upload", names)
                st.sidebar.success(f"Loaded {len(names)} file(s).")

else:
    if not DROPZONE.exists():
        st.sidebar.warning("Dropzone directory not found. Run `python setup_dirs.py`.")
    else:
        found = [
            p
            for p in DROPZONE.iterdir()
            if p.is_file() and p.suffix.lower() in ALLOWED_SUFFIXES
        ]
        if not found:
            st.sidebar.info("No CSV/Excel files in `01_dropzone`.")
        else:
            options = [p.name for p in sorted(found)]
            selected = st.sidebar.multiselect("Select files to load", options)
            load_btn = st.sidebar.button("Load into session")
            if load_btn and selected:
                dfs = []
                for name in selected:
                    path = DROPZONE / name
                    try:
                        dfs.append(load_from_path(path))
                    except Exception as e:
                        st.sidebar.error(f"Failed to load {name}: {e}")
                if dfs:
                    combined = pd.concat(dfs, ignore_index=True)
                    apply_ingestion(combined, "local_scan", selected)
                    st.sidebar.success(f"Loaded {len(selected)} file(s).")

# --- Main area ---

st.title("Rapid AI-Driven Profitability Engine")

# Display dashboard if analysis is complete
if st.session_state.analysis_complete and st.session_state.analytics_df is not None:
    render_dashboard(st.session_state.analytics_df, low_profitability_threshold)
elif st.session_state.is_loaded and st.session_state.main_df is not None:
    # Legacy view for manually loaded data
    df = st.session_state.main_df
    st.info("💡 Use **Run Analysis** in the sidebar to process data through the analytics pipeline.")
    st.metric("Rows", len(df))
    st.dataframe(df, use_container_width=True)
else:
    st.info(
        "👋 Welcome! Use **Run Analysis** in the sidebar to process data from `01_dropzone/`.\n\n"
        "The system will:\n"
        "1. Scan the dropzone for sales data files\n"
        "2. Load reference data from `00_selection/`\n"
        "3. Calculate profitability metrics\n"
        "4. Display interactive dashboard with KPIs and trends"
    )

# Processing log
if st.session_state.processing_log:
    with st.expander("Processing log"):
        for entry in reversed(st.session_state.processing_log):
            st.json(entry)
