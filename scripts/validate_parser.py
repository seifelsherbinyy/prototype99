"""
Phase 0 Parser Validation Script

Tests the HeaderParser against all raw data files in the workspace
and generates a validation report.

Success Criteria:
- >= 95% of headers successfully parsed
- All critical metrics (Revenue, Units, Conversion, GV, NPM) correctly identified
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime

import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.header_parser import HeaderParser, ParseResult


# Critical metrics that MUST be identified correctly
CRITICAL_METRICS = [
    "Product GMS",      # Revenue
    "Net Ordered Units", # Units
    "Net PPM",          # Profitability
    "GV",               # Glance Views
    "ASP",              # Average Selling Price
]


def load_file_headers(filepath: Path) -> list[str]:
    """Load headers from a CSV or Excel file."""
    if filepath.suffix.lower() == ".csv":
        # Try multiple encodings
        for encoding in ["utf-8", "latin-1", "cp1252"]:
            try:
                df = pd.read_csv(filepath, nrows=0, encoding=encoding)
                return df.columns.tolist()
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Could not decode {filepath}")
    else:
        df = pd.read_excel(filepath, nrows=0)
        return df.columns.tolist()


def check_critical_metrics(results: list[ParseResult]) -> dict:
    """
    Check if all critical metrics were correctly identified.
    
    Returns:
        Dictionary with critical metric identification results.
    """
    found_metrics = {}
    
    for metric in CRITICAL_METRICS:
        found = False
        for r in results:
            if r.metric_name == metric and r.confidence in ["HIGH", "MEDIUM"]:
                found = True
                break
        found_metrics[metric] = found
    
    all_found = all(found_metrics.values())
    
    return {
        "all_critical_found": all_found,
        "found_metrics": found_metrics,
        "missing_metrics": [m for m, found in found_metrics.items() if not found],
    }


def validate_single_file(filepath: Path, parser: HeaderParser) -> dict:
    """Validate parser against a single file."""
    try:
        headers = load_file_headers(filepath)
        results = parser.parse_all(headers)
        stats = parser.get_parse_statistics(results)
        critical = check_critical_metrics(results)
        
        return {
            "file": filepath.name,
            "path": str(filepath),
            "total_headers": stats["total_headers"],
            "successful_parses": stats["successful_parses"],
            "success_rate": stats["success_rate"],
            "confidence_distribution": stats["confidence_distribution"],
            "failed_headers": stats["failed_headers"],
            "critical_metrics": critical,
            "error": None,
            "results": results,
        }
    except Exception as e:
        return {
            "file": filepath.name,
            "path": str(filepath),
            "total_headers": 0,
            "successful_parses": 0,
            "success_rate": 0.0,
            "confidence_distribution": {},
            "failed_headers": [],
            "critical_metrics": {"all_critical_found": False, "found_metrics": {}, "missing_metrics": CRITICAL_METRICS},
            "error": str(e),
            "results": [],
        }


def run_validation() -> dict:
    """
    Run validation against all raw data files.
    
    Returns:
        Dictionary with overall validation results.
    """
    parser = HeaderParser()
    
    # Find all raw data files
    dropzone = project_root / "01_dropzone"
    weekly_perf = dropzone / "weekly" / "performance"
    historical = dropzone / "historical"
    
    files_to_test = []
    
    # Weekly performance files
    if weekly_perf.exists():
        files_to_test.extend(weekly_perf.glob("*.csv"))
    
    # Historical files
    if historical.exists():
        files_to_test.extend(historical.glob("*.csv"))
    
    print(f"Found {len(files_to_test)} files to validate\n")
    
    # Validate each file
    file_results = []
    all_results = []
    
    for filepath in sorted(files_to_test):
        print(f"Validating: {filepath.name}...")
        result = validate_single_file(filepath, parser)
        file_results.append(result)
        all_results.extend(result["results"])
        
        status = "PASS" if result["success_rate"] >= 95 else "FAIL"
        print(f"  {status} Success rate: {result['success_rate']:.1f}% ({result['successful_parses']}/{result['total_headers']})")
        if result["error"]:
            print(f"  ERROR: {result['error']}")
    
    # Calculate overall statistics
    total_headers = sum(r["total_headers"] for r in file_results)
    total_successful = sum(r["successful_parses"] for r in file_results)
    overall_success_rate = (total_successful / total_headers * 100) if total_headers > 0 else 0
    
    # Check critical metrics across all files
    overall_critical = check_critical_metrics(all_results)
    
    # Collect all unique failed headers
    all_failed = set()
    for r in file_results:
        all_failed.update(r["failed_headers"])
    
    # Aggregate confidence distribution
    overall_confidence = {}
    for r in file_results:
        for conf, count in r["confidence_distribution"].items():
            overall_confidence[conf] = overall_confidence.get(conf, 0) + count
    
    return {
        "timestamp": datetime.now().isoformat(),
        "files_tested": len(files_to_test),
        "total_headers": total_headers,
        "successful_parses": total_successful,
        "overall_success_rate": overall_success_rate,
        "passed_threshold": overall_success_rate >= 95,
        "confidence_distribution": overall_confidence,
        "all_failed_headers": sorted(all_failed),
        "critical_metrics": overall_critical,
        "file_results": file_results,
        "all_results": all_results,
    }


def generate_report(validation: dict, output_path: Path) -> None:
    """Generate CSV report from validation results."""
    # Create detailed results DataFrame
    rows = []
    for file_result in validation["file_results"]:
        for r in file_result["results"]:
            rows.append({
                "Source_File": file_result["file"],
                "Raw_Header_Found": r.raw_header,
                "Regex_Pattern_Matched": r.matched_pattern,
                "Parsed_Week": r.time_period,
                "Parsed_Metric": r.metric_name,
                "Parsed_Currency_Unit": r.currency_unit,
                "Is_Percentage": r.is_percentage,
                "Mapped_To_WBR_Column": r.mapped_wbr_column,
                "Confidence_Score": r.confidence,
                "Flag_Status": r.flag_status,
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\nDetailed report saved to: {output_path}")


def print_summary(validation: dict) -> None:
    """Print validation summary."""
    print("\n" + "=" * 80)
    print("PHASE 0 VALIDATION SUMMARY")
    print("=" * 80)
    
    print(f"\nFiles tested: {validation['files_tested']}")
    print(f"Total headers: {validation['total_headers']}")
    print(f"Successful parses: {validation['successful_parses']}")
    print(f"Overall success rate: {validation['overall_success_rate']:.2f}%")
    
    passed = validation["passed_threshold"]
    print(f"\n{'[PASS]' if passed else '[FAIL]'}: Success rate {'>=' if passed else '<'} 95%")
    
    print("\nConfidence Distribution:")
    for conf, count in sorted(validation["confidence_distribution"].items()):
        pct = count / validation["total_headers"] * 100
        print(f"  {conf}: {count} ({pct:.1f}%)")
    
    print("\nCritical Metrics Identification:")
    critical = validation["critical_metrics"]
    for metric, found in critical["found_metrics"].items():
        status = "[OK]" if found else "[MISSING]"
        print(f"  {status} {metric}")
    
    if critical["missing_metrics"]:
        print(f"\n[WARNING] Missing critical metrics: {critical['missing_metrics']}")
    
    if validation["all_failed_headers"]:
        print(f"\nFailed headers ({len(validation['all_failed_headers'])}):")
        for header in validation["all_failed_headers"][:20]:  # Show first 20
            print(f"  - {header}")
        if len(validation["all_failed_headers"]) > 20:
            print(f"  ... and {len(validation['all_failed_headers']) - 20} more")
    else:
        print("\n[OK] All headers parsed successfully!")


def main():
    """Main entry point."""
    print("=" * 80)
    print("PHASE 0: HEADER PARSER VALIDATION")
    print("=" * 80 + "\n")
    
    validation = run_validation()
    
    # Generate report
    output_dir = project_root / "02_output"
    output_dir.mkdir(exist_ok=True)
    report_path = output_dir / "phase0_validation_report.csv"
    generate_report(validation, report_path)
    
    # Print summary
    print_summary(validation)
    
    # Return exit code based on success
    if validation["passed_threshold"] and validation["critical_metrics"]["all_critical_found"]:
        print("\n" + "=" * 80)
        print("[SUCCESS] PHASE 0 COMPLETE - Ready to proceed to Phase 1")
        print("=" * 80)
        return 0
    else:
        print("\n" + "=" * 80)
        print("[FAILED] PHASE 0 FAILED - Refine patterns before proceeding")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
