"""
Unit tests for the HeaderParser module.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.header_parser import HeaderParser, ParseResult, KNOWN_METRICS, STATIC_COLUMNS


class TestHeaderParser:
    """Tests for HeaderParser class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = HeaderParser()
    
    # Test static columns
    def test_parse_asin(self):
        result = self.parser.parse("ASIN")
        assert result.metric_name == "ASIN"
        assert result.confidence == "HIGH"
        assert result.flag_status == "OK"
        assert result.matched_pattern == "static_column"
        assert result.mapped_wbr_column == "ASIN"
    
    def test_parse_brand_name(self):
        result = self.parser.parse("Brand Name")
        assert result.metric_name == "Brand Name"
        assert result.confidence == "HIGH"
        assert result.mapped_wbr_column == "Brand_Name"
    
    # Test currency with week format
    def test_parse_gms_week(self):
        result = self.parser.parse("Product GMS($)(Week 49 )")
        assert result.metric_name == "Product GMS"
        assert result.currency_unit == "$"
        assert result.week_number == 49
        assert result.time_period == "Week 49"
        assert result.confidence == "HIGH"
        assert result.mapped_wbr_column == "Ordered_Revenue"
    
    def test_parse_list_price_week(self):
        result = self.parser.parse("List Price($)(Week 50 )")
        assert result.metric_name == "List Price"
        assert result.currency_unit == "$"
        assert result.week_number == 50
        assert result.confidence == "HIGH"
    
    # Test percentage with week format
    def test_parse_net_ppm_week(self):
        result = self.parser.parse("Net PPM(Week 49 )(%)")
        assert result.metric_name == "Net PPM"
        assert result.currency_unit == "%"
        assert result.is_percentage == True
        assert result.week_number == 49
        assert result.confidence == "HIGH"
        assert result.mapped_wbr_column == "Net_PPM"
    
    def test_parse_cm_week(self):
        result = self.parser.parse("CM(Week 51 )(%)")
        assert result.metric_name == "CM"
        assert result.is_percentage == True
        assert result.week_number == 51
        assert result.confidence == "HIGH"
    
    # Test metric with (%) in name
    def test_parse_soroos_week(self):
        result = self.parser.parse("SoROOS(%)(Week 49 )(%)")
        assert result.metric_name == "SoROOS(%)"
        assert result.is_percentage == True
        assert result.week_number == 49
        assert result.confidence == "HIGH"
    
    # Test metric with % suffix in name
    def test_parse_daily_soroos_week(self):
        result = self.parser.parse("Daily SoRoos%(Week 49 )(%)")
        assert result.metric_name == "Daily SoRoos%"
        assert result.is_percentage == True
        assert result.week_number == 49
    
    # Test no unit indicator
    def test_parse_gv_week(self):
        result = self.parser.parse("GV(Week 49 )")
        assert result.metric_name == "GV"
        assert result.currency_unit == ""
        assert result.week_number == 49
        assert result.confidence == "HIGH"
        assert result.mapped_wbr_column == "Glance_Views"
    
    def test_parse_asp_week(self):
        result = self.parser.parse("ASP(Week 52 )")
        assert result.metric_name == "ASP"
        assert result.week_number == 52
        assert result.confidence == "HIGH"
        assert result.mapped_wbr_column == "Average_Selling_Price"
    
    # Test units format
    def test_parse_net_receipts_units_week(self):
        result = self.parser.parse("Net Receipts (Units)(Week 49 )")
        assert result.metric_name == "Net Receipts"
        assert result.currency_unit == "units"
        assert result.week_number == 49
    
    # Test complex metrics
    def test_parse_vendor_confirmation_rate(self):
        result = self.parser.parse("Vendor Confirmation Rate - Sourceable(Week 49 )(%)")
        assert result.metric_name == "Vendor Confirmation Rate - Sourceable"
        assert result.is_percentage == True
        assert result.week_number == 49
    
    def test_parse_fill_rate(self):
        result = self.parser.parse("Fill Rate - Sourceable(Week 49 )(%)")
        assert result.metric_name == "Fill Rate - Sourceable"
        assert result.is_percentage == True
    
    def test_parse_ccogs_pct(self):
        result = self.parser.parse("CCOGS As A % Of PCOGS(Week 49 )(%)")
        assert result.metric_name == "CCOGS As A % Of PCOGS"
        assert result.is_percentage == True
    
    # Test T12M format
    def test_parse_gms_t12m(self):
        result = self.parser.parse("Product GMS($)(T12M )")
        assert result.metric_name == "Product GMS"
        assert result.currency_unit == "$"
        assert result.time_period == "T12M"
        assert result.week_number is None
        assert result.confidence == "HIGH"
    
    def test_parse_net_ppm_t12m(self):
        result = self.parser.parse("Net PPM(T12M )(%)")
        assert result.metric_name == "Net PPM"
        assert result.is_percentage == True
        assert result.time_period == "T12M"
    
    def test_parse_gv_t12m(self):
        result = self.parser.parse("GV(T12M )")
        assert result.metric_name == "GV"
        assert result.time_period == "T12M"
    
    # Test date range format
    def test_parse_gms_daterange(self):
        result = self.parser.parse("Product GMS($)(11/02/2025-11/09/2025)")
        assert result.metric_name == "Product GMS"
        assert result.currency_unit == "$"
        assert result.time_period == "11/02/2025-11/09/2025"
        assert result.confidence == "HIGH"
    
    def test_parse_net_ppm_daterange(self):
        result = self.parser.parse("Net PPM(11/02/2025-11/09/2025)(%)")
        assert result.metric_name == "Net PPM"
        assert result.is_percentage == True
        assert result.time_period == "11/02/2025-11/09/2025"
    
    def test_parse_gv_daterange(self):
        result = self.parser.parse("GV(11/02/2025-11/09/2025)")
        assert result.metric_name == "GV"
        assert result.time_period == "11/02/2025-11/09/2025"
    
    # Test month format
    def test_parse_gms_month(self):
        result = self.parser.parse("Product GMS($)(Jan 2025)")
        assert result.metric_name == "Product GMS"
        assert result.time_period == "Jan 2025"
        assert result.period_type == "month"
    
    # Test quarter format
    def test_parse_ppm_quarter(self):
        result = self.parser.parse("Net PPM(Q1 2025)(%)")
        assert result.metric_name == "Net PPM"
        assert result.time_period == "Q1 2025"
        assert result.period_type == "quarter"

    # Test batch parsing
    def test_parse_all(self):
        headers = [
            "ASIN",
            "Product GMS($)(Week 49 )",
            "Net PPM(Week 49 )(%)",
            "GV(Week 49 )",
        ]
        results = self.parser.parse_all(headers)
        assert len(results) == 4
        assert all(r.confidence == "HIGH" for r in results)
    
    # Test statistics
    def test_get_parse_statistics(self):
        headers = [
            "ASIN",
            "Product GMS($)(Week 49 )",
            "Net PPM(Week 49 )(%)",
        ]
        results = self.parser.parse_all(headers)
        stats = self.parser.get_parse_statistics(results)
        
        assert stats["total_headers"] == 3
        assert stats["successful_parses"] == 3
        assert stats["success_rate"] == 100.0
        assert "HIGH" in stats["confidence_distribution"]
        assert len(stats["failed_headers"]) == 0
    
    # Test validation report
    def test_generate_validation_report(self):
        headers = ["ASIN", "Product GMS($)(Week 49 )"]
        results = self.parser.parse_all(headers)
        df = self.parser.generate_validation_report(results)
        
        assert len(df) == 2
        assert "Raw_Header_Found" in df.columns
        assert "Confidence_Score" in df.columns
        assert "Flag_Status" in df.columns
        assert "Mapped_To_WBR_Column" in df.columns


def run_tests():
    """Run all tests and print results."""
    import traceback
    
    test_class = TestHeaderParser()
    methods = [m for m in dir(test_class) if m.startswith("test_")]
    
    passed = 0
    failed = 0
    
    print("=" * 60)
    print("Running HeaderParser Unit Tests")
    print("=" * 60)
    
    for method_name in methods:
        test_class.setup_method()
        method = getattr(test_class, method_name)
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
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
