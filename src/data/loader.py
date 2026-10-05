"""Phase 1: Data Loader for Restok.

Loads M5 raw files, applies dtype downcasting, extracts the balanced
FOODS subset across target stores and saves the optimized subset.
"""
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from src.config import (
    RAW_DIR,
    PROCESSED_DIR,
    STORES,
    CATEGORY,
    START_D,
    END_D,
    SKUS_PER_DEPT,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Downcast numeric columns to minimize memory footprint."""
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            c_min = df[col].min()
            c_max = df[col].max()
            if pd.api.types.is_integer_dtype(df[col]):
                if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            elif pd.api.types.is_float_dtype(df[col]):
                if c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        logger.info(
            "Memory usage reduced from %.2f MB to %.2f MB (%.1f%% reduction)",
            start_mem,
            end_mem,
            100 * (start_mem - end_mem) / start_mem if start_mem > 0 else 0,
        )
    return df


def select_top_skus(
    sales_path: Path,
    stores: List[str],
    skus_per_dept: Dict[str, int],
    start_d: int,
    end_d: int,
) -> Tuple[List[str], pd.DataFrame]:
    """Selects top active SKUs per department within the specified day window."""
    day_cols = [f"d_{i}" for i in range(start_d, end_d + 1)]
    meta_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]

    logger.info("Scanning sales evaluation data to identify top SKUs...")
    df = pd.read_csv(sales_path, usecols=meta_cols + day_cols)
    df = df[(df["cat_id"] == CATEGORY) & (df["store_id"].isin(stores))].copy()

    # Sum sales across the window to identify consistently high-velocity SKUs
    df["window_sales"] = df[day_cols].sum(axis=1)

    selected_item_ids = []
    item_stats = df.groupby(["item_id", "dept_id"])["window_sales"].sum().reset_index()

    for dept, count in skus_per_dept.items():
        dept_items = (
            item_stats[item_stats["dept_id"] == dept]
            .sort_values("window_sales", ascending=False)
            .head(count)
        )
        selected_item_ids.extend(dept_items["item_id"].tolist())
        logger.info("Selected top %d SKUs for department %s", count, dept)

    filtered_sales = df[df["item_id"].isin(selected_item_ids)].drop(columns=["window_sales"])
    logger.info(
        "Filtered sales dataset: %d series across %d stores for %d days (%d SKUs total)",
        len(filtered_sales),
        len(stores),
        len(day_cols),
        len(selected_item_ids),
    )
    return selected_item_ids, filtered_sales


def load_and_subset_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Phase 1 main entrypoint: loads, filters, downcasts, and saves raw subset."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    sales_path = RAW_DIR / "sales_train_evaluation.csv"
    calendar_path = RAW_DIR / "calendar.csv"
    prices_path = RAW_DIR / "sell_prices.csv"

    # 1. Select SKUs and filter sales
    selected_items, sales_df = select_top_skus(
        sales_path=sales_path,
        stores=STORES,
        skus_per_dept=SKUS_PER_DEPT,
        start_d=START_D,
        end_d=END_D,
    )
    sales_df = reduce_mem_usage(sales_df)

    # 2. Filter calendar
    logger.info("Loading and filtering calendar...")
    calendar_df = pd.read_csv(calendar_path)
    cal_day_indices = [int(x.split("_")[1]) for x in calendar_df["d"]]
    calendar_df = calendar_df[
        (calendar_df["d"].isin([f"d_{i}" for i in range(START_D, END_D + 1)]))
    ].copy()
    calendar_df = reduce_mem_usage(calendar_df)

    # 3. Filter sell prices for target stores and items
    logger.info("Loading and filtering sell prices...")
    prices_df = pd.read_csv(prices_path)
    valid_wm_yr_wks = set(calendar_df["wm_yr_wk"].unique())
    prices_df = prices_df[
        (prices_df["store_id"].isin(STORES))
        & (prices_df["item_id"].isin(selected_items))
        & (prices_df["wm_yr_wk"].isin(valid_wm_yr_wks))
    ].copy()
    prices_df = reduce_mem_usage(prices_df)

    # Save Phase 1 subset files
    sales_out = PROCESSED_DIR / "subset_sales_wide.parquet"
    cal_out = PROCESSED_DIR / "subset_calendar.parquet"
    prices_out = PROCESSED_DIR / "subset_prices.parquet"

    sales_df.to_parquet(sales_out, index=False)
    calendar_df.to_parquet(cal_out, index=False)
    prices_df.to_parquet(prices_out, index=False)

    logger.info("Phase 1 Complete: saved wide sales (%s), calendar (%s), and prices (%s)",
                sales_out.name, cal_out.name, prices_out.name)

    return sales_df, calendar_df, prices_df


if __name__ == "__main__":
    load_and_subset_data()
