"""Phase 2: Data Cleaner for Restok.

Transforms wide sales into long format, maps calendar dates, joins weekly sell prices,
trims pre-launch zero stretches without price, validates data integrity,
and writes the verified canonical cleaned dataset to data/processed/clean.parquet.
"""
import logging
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR
from src.data.loader import reduce_mem_usage

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def validate_clean_data(df: pd.DataFrame, min_rows: int = 40000) -> None:
    """Rigorous integrity validation suite for Phase 2 output."""
    logger.info("Executing Phase 2 data validation suite...")

    # 1. Row count check (must satisfy user's requirement of at least 40,000 rows)
    row_count = len(df)
    logger.info("Total rows in cleaned dataset: %d", row_count)
    assert row_count >= min_rows, f"Expected at least {min_rows} rows, but got {row_count}"

    # 2. No negative sales
    neg_sales = (df["sales"] < 0).sum()
    assert neg_sales == 0, f"Found {neg_sales} negative sales entries!"

    # 3. No duplicate store-item-date combinations
    dupes = df.duplicated(subset=["store_id", "item_id", "date"]).sum()
    assert dupes == 0, f"Found {dupes} duplicate (store_id, item_id, date) rows!"

    # 4. No nulls in key identifiers and targets
    for col in ["store_id", "item_id", "date", "sales", "sell_price"]:
        null_count = df[col].isnull().sum()
        assert null_count == 0, f"Found {null_count} nulls in required column '{col}'"

    # 5. Check continuity per series (no missing dates)
    for (store, item), group in df.groupby(["store_id", "item_id"]):
        group_sorted = group.sort_values("date")
        expected_days = (group_sorted["date"].max() - group_sorted["date"].min()).days + 1
        actual_days = len(group_sorted)
        assert actual_days == expected_days, (
            f"Series ({store}, {item}) has date gaps: expected {expected_days} days, got {actual_days}"
        )

    logger.info("Validation PASSED: All 5 integrity checks succeeded.")


def clean_and_reshape() -> pd.DataFrame:
    """Melts wide sales data, links calendar and prices, and produces clean.parquet."""
    sales_path = PROCESSED_DIR / "subset_sales_wide.parquet"
    cal_path = PROCESSED_DIR / "subset_calendar.parquet"
    prices_path = PROCESSED_DIR / "subset_prices.parquet"

    assert sales_path.exists(), "Phase 1 sales subset missing! Run loader first."
    assert cal_path.exists(), "Phase 1 calendar subset missing! Run loader first."
    assert prices_path.exists(), "Phase 1 prices subset missing! Run loader first."

    logger.info("Loading Phase 1 parquet files...")
    sales_wide = pd.read_parquet(sales_path)
    cal = pd.read_parquet(cal_path)
    prices = pd.read_parquet(prices_path)

    id_vars = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_cols = [c for c in sales_wide.columns if c.startswith("d_")]

    logger.info("Melting wide sales (%d series x %d days)...", len(sales_wide), len(day_cols))
    sales_long = pd.melt(
        sales_wide,
        id_vars=id_vars,
        value_vars=day_cols,
        var_name="d",
        value_name="sales",
    )

    # 2. Merge with calendar
    logger.info("Merging calendar dates and event flags...")
    cal["date"] = pd.to_datetime(cal["date"])
    cal_cols = [
        "d",
        "date",
        "wm_yr_wk",
        "weekday",
        "wday",
        "month",
        "year",
        "event_name_1",
        "event_type_1",
        "event_name_2",
        "event_type_2",
        "snap_CA",
        "snap_TX",
        "snap_WI",
    ]
    df = pd.merge(sales_long, cal[cal_cols], on="d", how="inner")

    # Unified SNAP column based on state_id
    df["snap"] = np.where(
        df["state_id"] == "CA",
        df["snap_CA"],
        np.where(df["state_id"] == "TX", df["snap_TX"], df["snap_WI"]),
    ).astype(np.int8)

    # 3. Merge with sell prices
    logger.info("Merging weekly sell prices...")
    df = pd.merge(
        df,
        prices[["store_id", "item_id", "wm_yr_wk", "sell_price"]],
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left",
    )

    # 4. Trim pre-launch period:
    # If an item was not yet on sale in a given week, sell_price is NaN.
    # Specification rule: "no price means not on sale, so don't impute"
    initial_rows = len(df)
    df = df.dropna(subset=["sell_price"]).copy()
    trimmed_rows = initial_rows - len(df)
    if trimmed_rows > 0:
        logger.info("Trimmed %d unpriced / pre-launch rows", trimmed_rows)

    # 5. Sort canonically by store, item, date
    df = df.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)

    # Downcast memory
    df = reduce_mem_usage(df)

    # 6. Run validation checks
    validate_clean_data(df, min_rows=40000)

    # 7. Save output
    clean_out = PROCESSED_DIR / "clean.parquet"
    df.to_parquet(clean_out, index=False)
    logger.info("Phase 2 Complete: Clean dataset saved to %s (%d rows)", clean_out, len(df))

    return df


if __name__ == "__main__":
    clean_and_reshape()
