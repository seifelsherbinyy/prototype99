"""
Execute ASIN Overlap Validation

Runs ASIN overlap validation on sample files to establish baseline match rates
and identify orphaned ASINs. Generates a validation report.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from src.data_loader import (
    load_raw_data,
    load_reference_data,
    validate_asin_overlap,
    DROPZONE_PATH,
)
from src.config import OUTPUT_PATH


def main():
    """Execute ASIN validation on all available files."""
    print("=" * 80)
    print("ASIN Overlap Validation Report")
    print("=" * 80)
    print()
    
    # Load reference data
    print("Loading reference data...")
    ref_data = load_reference_data()
    vendor_map = ref_data["vendor_map"]
    asin_selection = ref_data["asin_selection"]
    
    print(f"  Vendor map: {len(vendor_map)} vendors")
    print(f"  ASIN selection: {len(asin_selection)} ASINs")
    print()
    
    # Get all weekly performance files
    performance_files = list(DROPZONE_PATH.glob("*.csv"))
    
    if not performance_files:
        print("No performance files found in dropzone!")
        return
    
    print(f"Found {len(performance_files)} performance files")
    print()
    
    # Results storage
    all_results = []
    
    # Process each file
    for filepath in sorted(performance_files):
        print(f"Processing: {filepath.name}")
        try:
            # Load raw data
            raw_df, metadata = load_raw_data(filepath)
            
            # Execute validation
            validation_result = validate_asin_overlap(
                raw_df,
                reference_data=ref_data
            )
            
            # Store results
            result = {
                "file": filepath.name,
                "total_raw_asins": validation_result["total_raw_asins"],
                "total_reference_asins": validation_result["total_reference_asins"],
                "matched_asins": validation_result["matched_asins"],
                "orphaned_asins": validation_result["orphaned_asins"],
                "missing_asins": validation_result["missing_asins"],
                "duplicate_reference_asins": validation_result.get("duplicate_reference_asins", 0),
                "match_percentage": validation_result["match_percentage"],
                "overlap_percentage": validation_result.get("overlap_percentage", 0.0),
            }
            all_results.append(result)
            
            # Print summary
            print(f"  Raw ASINs: {result['total_raw_asins']}")
            print(f"  Reference ASINs: {result['total_reference_asins']}")
            print(f"  Matched: {result['matched_asins']} ({result['match_percentage']:.1f}%)")
            print(f"  Orphaned (in raw, not in ref): {result['orphaned_asins']}")
            print(f"  Missing (in ref, not in raw): {result['missing_asins']}")
            if result['duplicate_reference_asins'] > 0:
                print(f"  ⚠️  Duplicate ASINs in reference: {result['duplicate_reference_asins']}")
            print()
            
        except Exception as e:
            print(f"  ❌ Error processing file: {e}")
            print()
    
    # Generate summary report
    print("=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print()
    
    if all_results:
        df_results = pd.DataFrame(all_results)
        
        print(f"Files processed: {len(df_results)}")
        print(f"Total raw ASINs (across all files): {df_results['total_raw_asins'].sum()}")
        print(f"Total reference ASINs: {df_results['total_reference_asins'].max() if len(df_results) > 0 else 0}")
        print(f"Average match percentage: {df_results['match_percentage'].mean():.1f}%")
        print(f"Total orphaned ASINs: {df_results['orphaned_asins'].sum()}")
        print(f"Total missing ASINs: {df_results['missing_asins'].sum()}")
        print(f"Total duplicate reference ASINs: {df_results['duplicate_reference_asins'].sum()}")
        print()
        
        # Save detailed report
        output_file = OUTPUT_PATH / "asin_validation_report.csv"
        df_results.to_csv(output_file, index=False)
        print(f"Detailed report saved to: {output_file}")
        print()
        
        # Identify common orphaned ASINs (appear in multiple files)
        print("=" * 80)
        print("ORPHANED ASIN ANALYSIS")
        print("=" * 80)
        print()
        
        # Collect orphaned ASINs from each file
        orphaned_by_file = {}
        for filepath in sorted(performance_files):
            try:
                raw_df, _ = load_raw_data(filepath)
                validation_result = validate_asin_overlap(raw_df, reference_data=ref_data)
                orphaned_list = validation_result.get("orphaned_list", [])
                orphaned_by_file[filepath.name] = set(orphaned_list[:100])  # Limit to first 100
            except Exception:
                continue
        
        # Find ASINs that appear as orphaned in multiple files
        all_orphaned = set()
        for orphaned_set in orphaned_by_file.values():
            all_orphaned.update(orphaned_set)
        
        print(f"Total unique orphaned ASINs found: {len(all_orphaned)}")
        
        # Count occurrences across files
        orphaned_counts = {}
        for asin in all_orphaned:
            count = sum(1 for orphaned_set in orphaned_by_file.values() if asin in orphaned_set)
            if count > 1:
                orphaned_counts[asin] = count
        
        if orphaned_counts:
            print(f"ASINs orphaned in multiple files: {len(orphaned_counts)}")
            print("\nTop 10 most frequently orphaned ASINs:")
            sorted_orphaned = sorted(orphaned_counts.items(), key=lambda x: x[1], reverse=True)
            for asin, count in sorted_orphaned[:10]:
                print(f"  {asin}: appears in {count} files")
        else:
            print("No ASINs found orphaned in multiple files")
        
        print()
    
    print("=" * 80)
    print("Validation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
