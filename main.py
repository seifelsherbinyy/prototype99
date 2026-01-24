"""
WBR Pipeline Orchestrator - End-to-End Execution

This is the main entry point for the Weekly Business Review (WBR) pipeline.
It orchestrates the full data processing workflow:

1. Load configuration and validate paths
2. Load reference data (vendor_map, ASIN selections)
3. Detect and load latest raw data file from Drop Zone
4. Parse headers using HeaderParser (with confidence scoring)
5. Transform data using DataTransformer
6. Analyze performance using Analyzer (RAG status)
7. Generate insights using InsightGenerator (priority scoring)
8. Format and export Excel workbook (with Validation Tab, Executive Summary)
9. Archive current data to HistoryHandler for future WoW calculations

Usage:
    python main.py [--file <filepath>] [--vendor <vendor_code>] [--week <week_number>]
    
Examples:
    python main.py                          # Process latest file
    python main.py --file path/to/data.csv  # Process specific file
    python main.py --vendor KY2O0           # Filter by vendor
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# WBR Pipeline modules
from src.config import (
    load_config,
    validate_config,
    ensure_directories,
    DROPZONE_PATH,
    OUTPUT_PATH,
    THRESHOLDS,
    IMPACT_WEIGHTS,
)
from src.header_parser import HeaderParser
from src.data_loader import (
    detect_latest_file,
    load_raw_data,
    load_reference_data,
    HistoryHandler,
    extract_week_from_filename,
    validate_asin_overlap,
)
from src.data_transformer import transform_raw_to_wbr, segment_by_week
from src.analyzer import (
    analyze_performance,
    generate_trend_analysis,
    calculate_statistical_insights,
    generate_comparative_analysis,
)
from src.insight_generator import generate_insights_for_dataframe, get_top_issues_summary
from src.excel_formatter import create_wbr_workbook
from src.vendor_detector import (
    detect_vendor_from_filename,
    load_vendor_selections,
    map_asins_to_vendors,
    identify_file_scope,
)


def log(message: str, level: str = "INFO") -> None:
    """Simple logging function."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")


def run_pipeline(
    filepath: Path | str | None = None,
    vendor_code: str | None = None,
    week_number: int | None = None,
    output_path: Path | str | None = None,
    archive_data: bool = True,
    segment_vendors: bool = True,
    generate_portfolio_view: bool = True,
) -> list[Path] | Path | None:
    """
    Run the complete WBR pipeline.
    
    Args:
        filepath: Path to raw data file. If None, detects latest file.
        vendor_code: Optional vendor code filter.
        week_number: Optional week number override.
        output_path: Output directory. If None, uses default.
        archive_data: Whether to archive data for WoW calculations.
        segment_vendors: Whether to generate separate files per vendor for portfolio files.
        generate_portfolio_view: Whether to generate combined portfolio view.
    
    Returns:
        List of Paths to generated WBR workbooks, or single Path, or None if failed.
    """
    start_time = datetime.now()
    log("=" * 60)
    log("WBR PIPELINE EXECUTION")
    log("=" * 60)
    
    # Step 1: Validate configuration
    log("Step 1: Validating configuration...")
    is_valid, errors = validate_config()
    if not is_valid:
        log(f"Configuration errors: {errors}", "ERROR")
        return None
    ensure_directories()
    log("Configuration validated successfully")
    
    # Step 2: Load reference data
    log("Step 2: Loading reference data...")
    ref_data = load_reference_data()
    vendor_map = ref_data["vendor_map"]
    asin_selection = ref_data["asin_selection"]
    log(f"Loaded {len(vendor_map)} vendors, {len(asin_selection)} ASINs")
    
    # Load vendor selections for ASIN-to-vendor mapping
    vendor_selections = load_vendor_selections()
    log(f"Loaded {len(vendor_selections)} vendor selection files")
    
    # Step 3: Detect and load raw data
    log("Step 3: Loading raw data...")
    if filepath:
        filepath = Path(filepath)
    else:
        filepath = detect_latest_file(DROPZONE_PATH)
        if filepath is None:
            log("No data files found in drop zone!", "ERROR")
            return None
    
    raw_df, metadata = load_raw_data(filepath)
    log(f"Loaded: {metadata['filename']}")
    log(f"  Rows: {metadata['rows']}, Columns: {metadata['columns']}")
    log(f"  Size: {metadata['file_size_kb']:.1f} KB")
    
    # Step 3a: Validate ASIN overlap
    log("Step 3a: Validating ASIN overlap...")
    asin_validation = validate_asin_overlap(raw_df, ref_data)
    log(f"ASIN Validation Results:")
    log(f"  Raw data ASINs: {asin_validation['total_raw_asins']}")
    log(f"  Reference ASINs: {asin_validation['total_reference_asins']}")
    log(f"  Matched: {asin_validation['matched_asins']} ({asin_validation['match_percentage']:.1f}%)")
    log(f"  Orphaned (in raw, not in ref): {asin_validation['orphaned_asins']}")
    log(f"  Missing (in ref, not in raw): {asin_validation['missing_asins']}")
    if asin_validation.get('duplicate_reference_asins', 0) > 0:
        log(f"  Warning: {asin_validation['duplicate_reference_asins']} duplicate ASINs in reference files", "WARN")
    if asin_validation['orphaned_asins'] > 0:
        log(f"  Note: {asin_validation['orphaned_asins']} ASINs in raw data are not in reference selection", "WARN")
    
    # Extract week from filename if not provided (for initial detection)
    year = None
    initial_week_number = week_number
    if week_number is None:
        initial_week_number, year = extract_week_from_filename(metadata['filename'], filepath)
    
    # Use inferred year from filename/file date, or fall back to current year
    if year is None:
        current_year = datetime.now().year
        # If initial week is 50-52, likely from previous year
        if initial_week_number and initial_week_number >= 50:
            current_year = current_year - 1
    else:
        current_year = year
    
    log(f"  Base year: {current_year}")
    
    # Step 4: Parse headers
    log("Step 4: Parsing headers...")
    parser = HeaderParser()
    parse_results = parser.parse_all(raw_df.columns.tolist())
    stats = parser.get_parse_statistics(parse_results)
    log(f"Parsed {stats['total_headers']} headers")
    log(f"  Success rate: {stats['success_rate']:.1f}%")
    log(f"  Confidence: {stats['confidence_distribution']}")
    
    if stats['success_rate'] < 95:
        log(f"Warning: Parse success rate below 95%!", "WARN")
        log(f"  Failed headers: {stats['failed_headers']}", "WARN")
    
    # Step 5: Initialize HistoryHandler
    log("Step 5: Initializing HistoryHandler...")
    history_handler = HistoryHandler()
    # Use initial week number for WoW availability check
    check_week = initial_week_number or datetime.now().isocalendar()[1]
    wow_availability = history_handler.check_wow_availability(raw_df, check_week, current_year)
    log(f"  Embedded WoW: {wow_availability['embedded']}")
    log(f"  Historical data: {wow_availability['historical_available']}")
    
    # Step 6: Transform data
    log("Step 6: Transforming data...")
    wbr_df = transform_raw_to_wbr(
        raw_df,
        parse_results,
        vendor_map=vendor_map,
        asin_selection=asin_selection,
        history_handler=history_handler,
        current_year=current_year,
        unpivot=True,
    )
    log(f"Transformed: {len(wbr_df)} rows, {len(wbr_df.columns)} columns")
    
    # Step 6a: Map ASINs to vendors
    log("Step 6a: Mapping ASINs to vendors...")
    wbr_df = map_asins_to_vendors(wbr_df, vendor_selections, vendor_map)
    vendor_counts = wbr_df[wbr_df["Vendor_Code"] != ""]["Vendor_Code"].value_counts()
    log(f"Vendor distribution: {dict(vendor_counts)}")
    
    # Step 6b: Segment by week
    log("Step 6b: Segmenting data by week...")
    week_segments = segment_by_week(wbr_df)
    log(f"Found {len(week_segments)} week(s): {sorted(week_segments.keys())}")
    
    # Determine output path
    if output_path is None:
        output_path = OUTPUT_PATH
    
    # Process each week segment
    output_files = []
    
    for week_num, week_df in week_segments.items():
        if week_num is None:
            # Skip if no week number
            continue
        
        log(f"\n--- Processing Week {week_num} ---")
        
        # Determine year for this week
        week_year = current_year
        if week_num >= 50 and current_year == datetime.now().year:
            # Likely from previous year
            week_year = current_year - 1
        
        # Step 7: Analyze performance for this week
        log(f"Step 7: Analyzing performance for Week {week_num}...")
        analyzed_df, summary_stats = analyze_performance(week_df)
        log(f"Analysis complete")
        log(f"  ASINs analyzed: {summary_stats['total_asins']}")
        log(f"  RAG distribution: {summary_stats.get('rag_distribution', {})}")
        
        # Step 7a: Add trend analysis
        log(f"Step 7a: Adding trend analysis for Week {week_num}...")
        analyzed_df = generate_trend_analysis(analyzed_df, history_handler, week_num, week_year)
        
        # Step 7b: Add comparative analysis
        log(f"Step 7b: Adding comparative analysis for Week {week_num}...")
        analyzed_df = generate_comparative_analysis(analyzed_df, group_by="Vendor_Code")
        
        # Step 7c: Calculate statistical insights
        log(f"Step 7c: Calculating statistical insights for Week {week_num}...")
        statistical_insights = calculate_statistical_insights(analyzed_df)
        summary_stats["statistical_insights"] = statistical_insights
        
        # Step 8: Generate insights
        log(f"Step 8: Generating insights for Week {week_num}...")
        insights_df = generate_insights_for_dataframe(analyzed_df, top_n=3)
        top_issues = get_top_issues_summary(analyzed_df, top_n=3)
        log(f"Insights generated: {len(top_issues)} top issues identified")
        
        # Step 9: Identify vendor scope
        log(f"Step 9: Identifying vendor scope for Week {week_num}...")
        file_scope = identify_file_scope(insights_df, metadata['filename'], vendor_selections)
        log(f"File scope: {file_scope['type']}, Vendors: {file_scope['vendors']}")
        
        # Prepare configuration for Config Log
        config_log = {
            "Source File": metadata['filename'],
            "Week Number": week_num,
            "Year": week_year,
            "Rows Processed": len(week_df),
            "Parse Success Rate": f"{stats['success_rate']:.1f}%",
            "Thresholds Applied": str(list(THRESHOLDS.keys())),
            "Impact Weights": str(IMPACT_WEIGHTS),
            "Generated At": datetime.now().isoformat(),
        }
        
        # Generate workbooks based on vendor scope
        if file_scope['type'] == 'single':
            # Single vendor - generate one file
            vendor_code = file_scope['vendors'][0] if file_scope['vendors'] else "UNKNOWN"
            if vendor_code_from_filename := detect_vendor_from_filename(metadata['filename']):
                vendor_code = vendor_code_from_filename
            
            log(f"Step 10: Creating WBR workbook for {vendor_code}, Week {week_num}...")
            output_file = create_wbr_workbook(
                wbr_data=insights_df,
                parse_results=parse_results,
                raw_data=raw_df,
                top_issues=top_issues,
                summary_stats=summary_stats,
                config=config_log,
                vendor_code=vendor_code,
                week_number=week_num,
                output_path=output_path,
            )
            output_files.append(output_file)
            log(f"Workbook created: {output_file.name}")
        
        elif file_scope['type'] == 'portfolio' and segment_vendors:
            # Portfolio - generate separate files per vendor
            log(f"Step 10: Creating vendor-specific WBR workbooks for Week {week_num}...")
            
            for vendor_code in file_scope['vendors']:
                vendor_df = insights_df[insights_df["Vendor_Code"] == vendor_code].copy()
                if vendor_df.empty:
                    continue
                
                # Analyze vendor-specific data
                vendor_analyzed, vendor_summary = analyze_performance(vendor_df)
                vendor_insights = generate_insights_for_dataframe(vendor_analyzed, top_n=3)
                vendor_top_issues = get_top_issues_summary(vendor_analyzed, top_n=3)
                
                # Create vendor-specific workbook
                vendor_output = create_wbr_workbook(
                    wbr_data=vendor_insights,
                    parse_results=parse_results,
                    raw_data=raw_df,
                    top_issues=vendor_top_issues,
                    summary_stats=vendor_summary,
                    config=config_log,
                    vendor_code=vendor_code,
                    week_number=week_num,
                    output_path=output_path,
                )
                output_files.append(vendor_output)
                log(f"  Created: {vendor_output.name}")
            
            # Generate combined portfolio view if requested
            if generate_portfolio_view:
                log(f"Step 11: Creating portfolio view for Week {week_num}...")
                portfolio_output = create_wbr_workbook(
                    wbr_data=insights_df,
                    parse_results=parse_results,
                    raw_data=raw_df,
                    top_issues=top_issues,
                    summary_stats=summary_stats,
                    config=config_log,
                    vendor_code="ALL",
                    week_number=week_num,
                    output_path=output_path,
                )
                output_files.append(portfolio_output)
                log(f"Portfolio view created: {portfolio_output.name}")
        
        else:
            # Unknown or fallback - generate single file
            vendor_code = detect_vendor_from_filename(metadata['filename']) or "ALL"
            log(f"Step 10: Creating WBR workbook for {vendor_code}, Week {week_num}...")
            output_file = create_wbr_workbook(
                wbr_data=insights_df,
                parse_results=parse_results,
                raw_data=raw_df,
                top_issues=top_issues,
                summary_stats=summary_stats,
                config=config_log,
                vendor_code=vendor_code,
                week_number=week_num,
                output_path=output_path,
            )
            output_files.append(output_file)
            log(f"Workbook created: {output_file.name}")
        
        # Step 12: Archive current data for this week
        if archive_data:
            log(f"Step 12: Archiving data for Week {week_num}...")
            rows_archived = history_handler.archive_current_data(
                week_df, week_num, week_year, metadata['filename']
            )
            log(f"Archived {rows_archived} metric records")
    
    # Cleanup
    history_handler.close()
    
    # Summary
    elapsed = (datetime.now() - start_time).total_seconds()
    log("=" * 60)
    log("PIPELINE COMPLETE")
    log("=" * 60)
    log(f"Generated {len(output_files)} workbook(s):")
    for f in output_files:
        log(f"  - {f.name}")
    log(f"Execution time: {elapsed:.1f} seconds")
    
    # Return single file if only one, otherwise return list
    if len(output_files) == 1:
        return output_files[0]
    return output_files


def main() -> int:
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="WBR Pipeline - Weekly Business Review Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Process latest file in drop zone
  python main.py --file data/sales.csv    # Process specific file
  python main.py --vendor KY2O0           # Filter by vendor code
  python main.py --week 49                # Override week number
        """
    )
    
    parser.add_argument(
        "--file", "-f",
        type=str,
        default=None,
        help="Path to raw data file (optional, detects latest if not provided)"
    )
    parser.add_argument(
        "--vendor", "-v",
        type=str,
        default=None,
        help="Vendor code filter (optional)"
    )
    parser.add_argument(
        "--week", "-w",
        type=int,
        default=None,
        help="Week number override (optional, extracted from filename if not provided)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory (optional, uses default 02_output/ if not provided)"
    )
    parser.add_argument(
        "--no-archive",
        action="store_true",
        help="Skip archiving data for WoW calculations"
    )
    
    args = parser.parse_args()
    
    try:
        result = run_pipeline(
            filepath=args.file,
            vendor_code=args.vendor,
            week_number=args.week,
            output_path=args.output,
            archive_data=not args.no_archive,
        )
        
        if result is not None:
            # Handle both single file and list of files
            if isinstance(result, list) and len(result) > 0:
                return 0
            elif isinstance(result, Path):
                return 0
            else:
                return 1
        else:
            return 1
            
    except Exception as e:
        log(f"Pipeline failed: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
