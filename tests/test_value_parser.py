"""
Unit tests for shared numeric value parser.
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.value_parser import parse_numeric_value


def test_parse_currency():
    assert parse_numeric_value("$1,234.56") == 1234.56
    assert parse_numeric_value("AED 1,234.56") == 1234.56


def test_parse_percent_decimal():
    assert parse_numeric_value("45%") == 0.45
    assert parse_numeric_value("12.5 percent") == 0.125


def test_parse_bps():
    assert parse_numeric_value("25 bps") == 0.0025
    assert parse_numeric_value("10bp") == 0.001


def test_parse_arrows_and_parentheses():
    assert parse_numeric_value("↓2.5%") == -0.025
    assert parse_numeric_value("(1,234)") == -1234.0


def test_parse_european_decimal():
    assert parse_numeric_value("1.234,56") == 1234.56
