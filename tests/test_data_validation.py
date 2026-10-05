"""Automated tests for Restok Data Pipeline (Phases 1-4)."""
from pathlib import Path
import pandas as pd
import pytest

from src.config import PROCESSED_DIR


@pytest.fixture(scope="session")
def simulated_data() -> pd.DataFrame:
    path = PROCESSED_DIR / "canonical_simulated.parquet"
    assert path.exists(), "canonical_simulated.parquet missing. Run supply_sim.py first."
    return pd.read_parquet(path)


def test_row_count_exceeds_40k(simulated_data: pd.DataFrame):
    """Verifies that data slicing maintains at least 40,000 rows as requested."""
    assert len(simulated_data) >= 40000, f"Expected >= 40,000 rows, found {len(simulated_data)}"


def test_no_negative_values(simulated_data: pd.DataFrame):
    """Verifies non-negative constraints on demand, sales, inventory, and prices."""
    for col in ["true_demand", "units_sold", "on_hand_inventory", "sell_price", "unit_cost"]:
        assert (simulated_data[col] < 0).sum() == 0, f"Found negative values in column {col}"


def test_no_duplicate_series_dates(simulated_data: pd.DataFrame):
    """Verifies no duplicate (store_id, item_id, date) rows exist."""
    dupes = simulated_data.duplicated(subset=["store_id", "item_id", "date"]).sum()
    assert dupes == 0, f"Found {dupes} duplicate entries"


def test_continuous_dates_per_series(simulated_data: pd.DataFrame):
    """Verifies there are no gaps in daily date sequence per series."""
    for (store, item), group in simulated_data.groupby(["store_id", "item_id"]):
        group_sorted = group.sort_values("date")
        expected_days = (group_sorted["date"].max() - group_sorted["date"].min()).days + 1
        assert len(group_sorted) == expected_days, f"Gap in dates for ({store}, {item})"


def test_inventory_conservation_law(simulated_data: pd.DataFrame):
    """Verifies that units_sold + lost_sales equals true_demand."""
    diff = ((simulated_data["units_sold"] + simulated_data["lost_sales"]) - simulated_data["true_demand"]).abs().sum()
    assert diff == 0, f"Conservation law violated: discrepancy of {diff}"


def test_stockout_rate_in_target_range(simulated_data: pd.DataFrame):
    """Verifies simulated stockout rate is strictly within target bounds [5%, 12%]."""
    stockout_rate = simulated_data["is_stockout"].mean()
    assert 0.05 <= stockout_rate <= 0.12, f"Stockout rate {stockout_rate:.4f} outside [0.05, 0.12]"


def test_required_canonical_columns(simulated_data: pd.DataFrame):
    """Verifies all required canonical columns from the data dictionary are present."""
    required = [
        "id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "date", "d",
        "true_demand", "units_sold", "lost_sales", "is_stockout", "on_hand_inventory",
        "sell_price", "unit_cost", "unit_margin", "is_promotion", "is_holiday",
        "is_perishable", "shelf_life", "lead_time", "case_pack", "critical_ratio", "snap"
    ]
    for col in required:
        assert col in simulated_data.columns, f"Missing required column: {col}"


def test_phase5_features():
    """Verifies Phase 5 feature matrix row count >= 40,000 and required features exist."""
    path = PROCESSED_DIR / "features.parquet"
    assert path.exists(), "features.parquet missing. Run feature_pipeline.py first."
    df = pd.read_parquet(path)
    assert len(df) >= 40000, f"Expected >= 40,000 rows in feature matrix, found {len(df)}"

    expected_features = [
        "lag_14", "lag_28", "rolling_mean_7_lag14", "rolling_mean_28_lag14",
        "price_ratio", "dow_sin", "dow_cos", "month_sin", "month_cos",
        "days_to_holiday", "days_since_holiday", "sku_dept_share", "sku_age_days"
    ]
    for feat in expected_features:
        assert feat in df.columns, f"Missing feature '{feat}'"
        assert df[feat].isnull().sum() == 0, f"Feature '{feat}' has null values!"
