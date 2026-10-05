"""Phase 3: Preprocessor & Derived Fields for Restok.

Enriches the cleaned dataset with supply chain attributes, promotional tags,
calendar/holiday flags, and cost/margin economics based on config rules.
Outputs the canonical table to data/processed/canonical.parquet.
"""
import logging
from pathlib import Path
import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, SUPPLY_CHAIN_CFG

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def derive_features_and_attributes(df: pd.DataFrame) -> pd.DataFrame:
    """Enriches clean dataset with promotion, holiday, and supply chain fields."""
    logger.info("Computing rolling price median and promotional status...")
    # Calculate rolling median price (28-day window) per store & item
    # Note: Shifted or min_periods=1 to prevent nulls
    df = df.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)

    df["rolling_price_median"] = (
        df.groupby(["store_id", "item_id"])["sell_price"]
        .transform(lambda s: s.rolling(window=28, min_periods=1).median())
    )

    # Is_Promotion: price at least 10% below rolling median
    df["is_promotion"] = (
        (df["sell_price"] <= 0.90 * df["rolling_price_median"])
    ).astype(np.int8)

    # Is_Holiday: calendar event indicator
    logger.info("Deriving holiday and event indicators...")
    df["is_holiday"] = (
        df["event_name_1"].notnull() | df["event_name_2"].notnull()
    ).astype(np.int8)

    # Map supply chain parameters by department from config
    logger.info("Injecting supply chain parameters (perishable, lead time, case pack, costs)...")
    dept_configs = SUPPLY_CHAIN_CFG["categories"]
    margin_rate = SUPPLY_CHAIN_CFG.get("margin_rate", 0.35)

    df["is_perishable"] = df["dept_id"].map(
        lambda d: int(dept_configs[d]["is_perishable"])
    ).astype(np.int8)

    df["shelf_life"] = df["dept_id"].map(
        lambda d: int(dept_configs[d]["shelf_life_days"])
    ).astype(np.int16)

    df["lead_time"] = df["dept_id"].map(
        lambda d: int(dept_configs[d]["lead_time_days"])
    ).astype(np.int8)

    df["case_pack"] = df["dept_id"].map(
        lambda d: int(dept_configs[d]["case_pack"])
    ).astype(np.int8)

    df["critical_ratio"] = df["dept_id"].map(
        lambda d: float(dept_configs[d]["critical_ratio"])
    ).astype(np.float32)

    # Unit cost and unit margin
    df["unit_cost"] = (df["sell_price"] * (1.0 - margin_rate)).round(2).astype(np.float32)
    df["unit_margin"] = (df["sell_price"] - df["unit_cost"]).round(2).astype(np.float32)

    # Rename 'sales' to 'true_demand' to prepare for supply simulation censoring in Phase 4
    df = df.rename(columns={"sales": "true_demand"})

    return df


def validate_canonical_table(df: pd.DataFrame) -> None:
    """Validates that all columns from the data dictionary exist and are non-null."""
    logger.info("Validating canonical table schema and completeness...")
    required_cols = [
        "id",
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id",
        "date",
        "d",
        "true_demand",
        "sell_price",
        "unit_cost",
        "unit_margin",
        "is_promotion",
        "is_holiday",
        "is_perishable",
        "shelf_life",
        "lead_time",
        "case_pack",
        "critical_ratio",
        "snap",
    ]

    for col in required_cols:
        assert col in df.columns, f"Missing required canonical column: '{col}'"
        null_count = df[col].isnull().sum()
        assert null_count == 0, f"Found {null_count} nulls in canonical column '{col}'"

    logger.info("Canonical table validation PASSED: %d records with all required fields.", len(df))


def process_canonical() -> pd.DataFrame:
    """Phase 3 execution: loads clean.parquet, derives features, and writes canonical.parquet."""
    clean_path = PROCESSED_DIR / "clean.parquet"
    assert clean_path.exists(), f"Missing {clean_path}. Run Phase 2 cleaner first."

    logger.info("Loading cleaned dataset: %s", clean_path)
    clean_df = pd.read_parquet(clean_path)

    canonical_df = derive_features_and_attributes(clean_df)
    validate_canonical_table(canonical_df)

    canonical_out = PROCESSED_DIR / "canonical.parquet"
    canonical_df.to_parquet(canonical_out, index=False)
    logger.info("Phase 3 Complete: Canonical table saved to %s (%d rows)", canonical_out, len(canonical_df))

    return canonical_df


if __name__ == "__main__":
    process_canonical()
