"""
Unit tests for temporal calculation utilities.
"""

import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.temporal_calcs import add_temporal_metrics


def test_weekly_temporal_metrics():
    df = pd.DataFrame({
        "ASIN": ["A"] * 4,
        "Week_Number": [1, 2, 3, 4],
        "Ordered_Revenue": [10.0, 20.0, 30.0, 40.0],
        "Net_PPM": [0.10, 0.20, 0.30, 0.40],
    })

    result = add_temporal_metrics(df, base_year=2025)

    assert "Revenue_WoW" in result.columns
    assert result.loc[result["Week_Number"] == 2, "Revenue_WoW"].iloc[0] == 1.0
    assert result.loc[result["Week_Number"] == 4, "Ordered_Revenue_L4W"].iloc[0] == 100.0
    assert round(result.loc[result["Week_Number"] == 4, "Net_PPM_L4W"].iloc[0], 4) == 0.25


def test_monthly_temporal_metrics():
    df = pd.DataFrame({
        "ASIN": ["A", "A"],
        "Snapshot_Date": [pd.Timestamp("2025-01-31"), pd.Timestamp("2025-02-28")],
        "Ordered_Revenue": [100.0, 120.0],
    })

    result = add_temporal_metrics(df)
    assert "Revenue_MoM" in result.columns

    feb = result.loc[result["Snapshot_Date"] == pd.Timestamp("2025-02-28")]
    assert feb["Revenue_MoM"].iloc[0] == 0.2
    assert feb["Ordered_Revenue_YTD"].iloc[0] == 220.0
