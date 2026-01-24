"""
Integration tests for the WBR Pipeline.

Tests the complete end-to-end pipeline execution.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

from src.config import DROPZONE_PATH, OUTPUT_PATH, load_config, validate_config
from src.header_parser import HeaderParser
from src.data_loader import (
    detect_latest_file,
    load_raw_data,
    load_reference_data,
    HistoryHandler,
)
from src.data_transformer import (
    transform_raw_to_wbr,
    clean_numeric_value,
)
from src.analyzer import (
    analyze_performance,
    get_rag_status,
)
from src.insight_generator import (
    generate_insights_for_dataframe,
    get_top_issues_summary,
    calculate_impact_score,
)
from src.excel_formatter import create_wbr_workbook


class TestConfiguration:
    """Tests for configuration module."""
    
    def test_config_validation(self):
        is_valid, errors = validate_config()
        assert is_valid, f"Configuration errors: {errors}"
    
    def test_config_paths_exist(self):
        config = load_config()
        assert config["dropzone_path"].exists(), "Dropzone path does not exist"
        assert config["selection_path"].exists(), "Selection path does not exist"


class TestDataLoader:
    """Tests for data loader module."""
    
    def test_detect_latest_file(self):
        latest = detect_latest_file(DROPZONE_PATH)
        assert latest is not None, "No files found in dropzone"
        assert latest.suffix.lower() in (".csv", ".xlsx", ".xls", ".xlsm", ".json"), "Expected supported file type"
    
    def test_load_raw_data(self):
        latest = detect_latest_file(DROPZONE_PATH)
        if latest:
            df, metadata = load_raw_data(latest)
            assert not df.empty, "DataFrame is empty"
            assert "ASIN" in df.columns, "ASIN column missing"
            assert metadata["rows"] > 0, "No rows loaded"
    
    def test_load_reference_data(self):
        ref_data = load_reference_data()
        assert "vendor_map" in ref_data, "vendor_map missing"
        assert "asin_selection" in ref_data, "asin_selection missing"


class TestDataTransformer:
    """Tests for data transformer module."""
    
    def test_clean_numeric_parentheses(self):
        assert clean_numeric_value("(15.23)") == -15.23
    
    def test_clean_numeric_currency(self):
        assert clean_numeric_value("$1,234.56") == 1234.56
    
    def test_clean_numeric_percentage(self):
        assert clean_numeric_value("45.6%") == 0.456
    
    def test_transform_raw_to_wbr(self):
        latest = detect_latest_file(DROPZONE_PATH)
        if latest:
            raw_df, _ = load_raw_data(latest)
            parser = HeaderParser()
            parse_results = parser.parse_all(raw_df.columns.tolist())
            
            wbr_df = transform_raw_to_wbr(raw_df, parse_results, unpivot=True)
            
            assert not wbr_df.empty, "Transformed DataFrame is empty"
            assert "ASIN" in wbr_df.columns, "ASIN column missing after transform"


class TestAnalyzer:
    """Tests for analyzer module."""
    
    def test_rag_status_red(self):
        status = get_rag_status(0.03, "Net_PPM")
        assert status == "RED", f"Expected RED, got {status}"
    
    def test_rag_status_amber(self):
        status = get_rag_status(0.10, "Net_PPM")
        assert status == "AMBER", f"Expected AMBER, got {status}"
    
    def test_rag_status_green(self):
        status = get_rag_status(0.20, "Net_PPM")
        assert status == "GREEN", f"Expected GREEN, got {status}"
    
    def test_analyze_performance(self):
        latest = detect_latest_file(DROPZONE_PATH)
        if latest:
            raw_df, _ = load_raw_data(latest)
            parser = HeaderParser()
            parse_results = parser.parse_all(raw_df.columns.tolist())
            wbr_df = transform_raw_to_wbr(raw_df, parse_results, unpivot=True)
            
            analyzed_df, summary = analyze_performance(wbr_df)
            
            assert "total_rows" in summary, "total_rows missing from summary"
            assert summary["total_rows"] > 0, "No rows analyzed"
            
            # Check for RAG columns
            rag_cols = [c for c in analyzed_df.columns if c.endswith("_RAG")]
            assert len(rag_cols) > 0, "No RAG columns added"


class TestInsightGenerator:
    """Tests for insight generator module."""
    
    def test_impact_score_calculation(self):
        score = calculate_impact_score("Ordered_Revenue", -0.20)
        assert score == 200.0, f"Expected 200.0, got {score}"
    
    def test_impact_score_improvement(self):
        score = calculate_impact_score("Ordered_Revenue", 0.10)
        # Improvements get half weight
        assert score == 50.0, f"Expected 50.0, got {score}"
    
    def test_generate_insights(self):
        latest = detect_latest_file(DROPZONE_PATH)
        if latest:
            raw_df, _ = load_raw_data(latest)
            parser = HeaderParser()
            parse_results = parser.parse_all(raw_df.columns.tolist())
            wbr_df = transform_raw_to_wbr(raw_df, parse_results, unpivot=True)
            analyzed_df, _ = analyze_performance(wbr_df)
            
            insights_df = generate_insights_for_dataframe(analyzed_df, top_n=3)
            
            assert "Automated_Commentary" in insights_df.columns, "Commentary column missing"
            assert "Next_Actions" in insights_df.columns, "Next_Actions column missing"


class TestExcelFormatter:
    """Tests for Excel formatter module."""
    
    def test_create_workbook(self):
        latest = detect_latest_file(DROPZONE_PATH)
        if latest:
            raw_df, metadata = load_raw_data(latest)
            parser = HeaderParser()
            parse_results = parser.parse_all(raw_df.columns.tolist())
            ref_data = load_reference_data()
            wbr_df = transform_raw_to_wbr(
                raw_df, parse_results,
                vendor_map=ref_data["vendor_map"],
                unpivot=True
            )
            analyzed_df, summary = analyze_performance(wbr_df)
            insights_df = generate_insights_for_dataframe(analyzed_df)
            top_issues = get_top_issues_summary(analyzed_df, top_n=3)
            
            output_file = create_wbr_workbook(
                wbr_data=insights_df,
                parse_results=parse_results,
                raw_data=raw_df,
                top_issues=top_issues,
                summary_stats=summary,
                vendor_code="TEST",
                week_number=99,
                output_filename="test_output.xlsx",
            )
            
            assert output_file.exists(), "Output file not created"
            
            # Cleanup
            output_file.unlink()


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""
    
    def test_date_range_file_processing(self):
        """Test processing files with date range format (Nov2-23-2025)."""
        # Find a date range file
        date_range_files = [f for f in DROPZONE_PATH.glob("*.csv") if "Nov" in f.name]
        
        if not date_range_files:
            print("  [SKIP] No date range files found for testing")
            return
        
        test_file = date_range_files[0]
        raw_df, metadata = load_raw_data(test_file)
        
        # Parse headers
        parser = HeaderParser()
        parse_results = parser.parse_all(raw_df.columns.tolist())
        stats = parser.get_parse_statistics(parse_results)
        
        # Should parse successfully
        assert stats["success_rate"] >= 95, f"Date range file parse rate {stats['success_rate']}% < 95%"
        
        # Transform should work
        wbr_df = transform_raw_to_wbr(raw_df, parse_results, unpivot=True)
        assert not wbr_df.empty, "Date range file transformation failed"
        
        print(f"  [OK] Date range file processed: {test_file.name}")
    
    def test_cross_year_week_boundary(self):
        """Test processing file with cross-year week boundary (W50-W1)."""
        # Find a cross-year file
        cross_year_files = [f for f in DROPZONE_PATH.glob("*.csv") if "W50-W1" in f.name or "W52-W1" in f.name]
        
        if not cross_year_files:
            print("  [SKIP] No cross-year files found for testing")
            return
        
        test_file = cross_year_files[0]
        raw_df, metadata = load_raw_data(test_file)
        
        # Parse headers
        parser = HeaderParser()
        parse_results = parser.parse_all(raw_df.columns.tolist())
        
        # Extract weeks from parsed results
        weeks = [r.week_number for r in parse_results if r.week_number is not None]
        unique_weeks = sorted(set(weeks))
        
        # Should have both W52/W1 or W50/W1
        assert len(unique_weeks) >= 2, "Cross-year file should have multiple weeks"
        
        # Transform should handle both weeks
        wbr_df = transform_raw_to_wbr(raw_df, parse_results, unpivot=True)
        assert not wbr_df.empty, "Cross-year file transformation failed"
        
        # Check that Week_Number column has both weeks
        if "Week_Number" in wbr_df.columns:
            week_numbers = wbr_df["Week_Number"].unique()
            assert len(week_numbers) >= 2, "Should have multiple weeks in output"
        
        print(f"  [OK] Cross-year file processed: {test_file.name}, weeks: {unique_weeks}")
    
    def test_week_extraction_from_filename(self):
        """Test week extraction from various filename formats."""
        from src.data_loader import extract_week_from_filename
        
        # Test standard week format
        week, year = extract_week_from_filename("KY2O0-W49-W52.csv")
        assert week == 52, f"Expected week 52, got {week}"
        
        # Test single week with year
        week, year = extract_week_from_filename("WJTP1-W48-2025.csv")
        assert week == 48, f"Expected week 48, got {week}"
        assert year == 2025, f"Expected year 2025, got {year}"
        
        # Test cross-year format
        week, year = extract_week_from_filename("WJTP1-W50-W1.csv")
        assert week == 1, f"Expected week 1 (last week), got {week}"
        
        print("  [OK] Week extraction from filename working correctly")
    
    def test_large_file_performance(self):
        """Test that processing completes in reasonable time for larger files."""
        import time
        
        # Find the largest file
        files = list(DROPZONE_PATH.glob("*.csv"))
        if not files:
            print("  [SKIP] No files found for performance testing")
            return
        
        # Get file sizes and find largest
        file_sizes = [(f, f.stat().st_size) for f in files]
        largest_file = max(file_sizes, key=lambda x: x[1])[0]
        
        print(f"  Testing performance with: {largest_file.name} ({largest_file.stat().st_size / 1024:.1f} KB)")
        
        start_time = time.time()
        
        # Load and process
        raw_df, metadata = load_raw_data(largest_file)
        parser = HeaderParser()
        parse_results = parser.parse_all(raw_df.columns.tolist())
        wbr_df = transform_raw_to_wbr(raw_df, parse_results, unpivot=True)
        analyzed_df, _ = analyze_performance(wbr_df)
        
        elapsed = time.time() - start_time
        
        # Should complete in reasonable time (adjust threshold as needed)
        # For files up to ~1MB, should process in < 10 seconds
        max_time = 30.0  # 30 seconds max for any file
        assert elapsed < max_time, f"Processing took {elapsed:.1f}s, exceeds {max_time}s threshold"
        
        print(f"  [OK] Processed {len(wbr_df)} rows in {elapsed:.2f}s ({len(wbr_df)/elapsed:.0f} rows/sec)")


class TestFullPipeline:
    """Integration test for the complete pipeline."""
    
    def test_end_to_end_pipeline(self):
        """Test complete pipeline execution."""
        # Step 1: Configuration
        is_valid, errors = validate_config()
        assert is_valid, f"Config errors: {errors}"
        
        # Step 2: Reference data
        ref_data = load_reference_data()
        assert not ref_data["vendor_map"].empty, "Vendor map empty"
        
        # Step 3: Raw data
        latest = detect_latest_file(DROPZONE_PATH)
        assert latest is not None, "No raw data files"
        raw_df, metadata = load_raw_data(latest)
        assert not raw_df.empty, "Raw data empty"
        
        # Step 4: Parse headers
        parser = HeaderParser()
        parse_results = parser.parse_all(raw_df.columns.tolist())
        stats = parser.get_parse_statistics(parse_results)
        assert stats["success_rate"] >= 95, f"Parse rate {stats['success_rate']}% < 95%"
        
        # Step 5: Transform
        wbr_df = transform_raw_to_wbr(
            raw_df, parse_results,
            vendor_map=ref_data["vendor_map"],
            unpivot=True
        )
        assert not wbr_df.empty, "Transformed data empty"
        
        # Step 6: Analyze
        analyzed_df, summary = analyze_performance(wbr_df)
        assert summary["total_rows"] > 0, "No rows analyzed"
        
        # Step 7: Generate insights
        insights_df = generate_insights_for_dataframe(analyzed_df)
        assert "Automated_Commentary" in insights_df.columns
        
        # Step 8: Create output
        top_issues = get_top_issues_summary(analyzed_df, top_n=3)
        output_file = create_wbr_workbook(
            wbr_data=insights_df,
            parse_results=parse_results,
            raw_data=raw_df,
            top_issues=top_issues,
            summary_stats=summary,
            vendor_code="INTEGRATION_TEST",
            week_number=99,
            output_filename="integration_test.xlsx",
        )
        
        assert output_file.exists(), "Output file not created"
        
        # Verify file size is reasonable
        file_size_kb = output_file.stat().st_size / 1024
        assert file_size_kb > 10, f"Output file too small: {file_size_kb:.1f} KB"
        
        # Cleanup
        output_file.unlink()
        print(f"\n[OK] Integration test passed - processed {len(wbr_df)} rows")


def run_all_tests():
    """Run all tests and print results."""
    import traceback
    
    test_classes = [
        TestConfiguration,
        TestDataLoader,
        TestDataTransformer,
        TestAnalyzer,
        TestInsightGenerator,
        TestExcelFormatter,
        TestEdgeCases,
        TestFullPipeline,
    ]
    
    passed = 0
    failed = 0
    
    print("=" * 70)
    print("WBR PIPELINE INTEGRATION TESTS")
    print("=" * 70)
    
    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        instance = test_class()
        methods = [m for m in dir(instance) if m.startswith("test_")]
        
        for method_name in methods:
            method = getattr(instance, method_name)
            try:
                method()
                print(f"  [PASS] {method_name}")
                passed += 1
            except AssertionError as e:
                print(f"  [FAIL] {method_name}: {e}")
                failed += 1
            except Exception as e:
                print(f"  [ERROR] {method_name}: {e}")
                traceback.print_exc()
                failed += 1
    
    print("\n" + "=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
