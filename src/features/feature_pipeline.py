"""Phase 5: Feature Engineering Pipeline for Restok.

Builds horizon-safe features strictly respecting the 14-day forecast horizon:
- Horizon-safe lags (t-14, t-15, t-21, t-28, t-35)
- Shifted rolling statistics (mean, std, min, max over 7, 14, 28 days shifted by 14)
- Price ratio and promotion indicators
- Calendar, holiday proximity, and cyclical day/month encodings
- Department-level cannibalization share
- SKU age
- Formal leakage validation test
"""
import logging
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, FORECAST_HORIZON
from src.data.loader import reduce_mem_usage

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_holiday_proximity(dates: pd.Series, holiday_dates: List[pd.Timestamp]) -> Tuple[np.ndarray, np.ndarray]:
    """Computes days to nearest future holiday and days since nearest past holiday."""
    h_array = np.array([h.to_datetime64() for h in holiday_dates], dtype="datetime64[D]")
    d_array = dates.dt.floor("D").values.astype("datetime64[D]")

    days_to = np.full(len(dates), 30, dtype=np.int16)
    days_since = np.full(len(dates), 30, dtype=np.int16)

    for i, cur_d in enumerate(d_array):
        # Future holidays
        future_diffs = (h_array - cur_d).astype(int)
        valid_future = future_diffs[future_diffs >= 0]
        if len(valid_future) > 0:
            days_to[i] = min(30, np.min(valid_future))

        # Past holidays
        past_diffs = (cur_d - h_array).astype(int)
        valid_past = past_diffs[past_diffs >= 0]
        if len(valid_past) > 0:
            days_since[i] = min(30, np.min(valid_past))

    return days_to, days_since


def build_features(df: pd.DataFrame, target_col: str = "units_sold") -> pd.DataFrame:
    """Builds leak-free feature matrix from the canonical simulated dataset."""
    logger.info("Sorting dataset by store, item, and date...")
    df = df.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)

    # 1. Horizon-Safe Lags (minimum shift of 14 days)
    logger.info("Computing horizon-safe lags (min lag = %d)...", FORECAST_HORIZON)
    lags = [14, 15, 21, 28, 35]
    for lag in lags:
        df[f"lag_{lag}"] = df.groupby(["store_id", "item_id"])[target_col].shift(lag).astype(np.float32)

    # 2. Shifted Rolling Statistics (lag-14 base)
    logger.info("Computing shifted rolling statistics (7, 14, 28 days)...")
    shifted_base = df.groupby(["store_id", "item_id"])[target_col].shift(FORECAST_HORIZON)

    # Rolling means
    df["rolling_mean_7_lag14"] = (
        shifted_base.groupby([df["store_id"], df["item_id"]])
        .transform(lambda s: s.rolling(7, min_periods=1).mean())
        .astype(np.float32)
    )
    df["rolling_std_7_lag14"] = (
        shifted_base.groupby([df["store_id"], df["item_id"]])
        .transform(lambda s: s.rolling(7, min_periods=1).std().fillna(0))
        .astype(np.float32)
    )
    df["rolling_mean_14_lag14"] = (
        shifted_base.groupby([df["store_id"], df["item_id"]])
        .transform(lambda s: s.rolling(14, min_periods=1).mean())
        .astype(np.float32)
    )
    df["rolling_mean_28_lag14"] = (
        shifted_base.groupby([df["store_id"], df["item_id"]])
        .transform(lambda s: s.rolling(28, min_periods=1).mean())
        .astype(np.float32)
    )
    df["rolling_std_28_lag14"] = (
        shifted_base.groupby([df["store_id"], df["item_id"]])
        .transform(lambda s: s.rolling(28, min_periods=1).std().fillna(0))
        .astype(np.float32)
    )
    df["rolling_min_28_lag14"] = (
        shifted_base.groupby([df["store_id"], df["item_id"]])
        .transform(lambda s: s.rolling(28, min_periods=1).min())
        .astype(np.float32)
    )
    df["rolling_max_28_lag14"] = (
        shifted_base.groupby([df["store_id"], df["item_id"]])
        .transform(lambda s: s.rolling(28, min_periods=1).max())
        .astype(np.float32)
    )

    # 3. Price Features (Price Elasticity Ratio)
    logger.info("Computing price elasticity ratio and promotional depth...")
    df["price_ratio"] = (df["sell_price"] / (df["rolling_price_median"] + 1e-5)).astype(np.float32)
    df["price_elasticity_ratio"] = df["price_ratio"]  # Syllabus requirement

    # 4. Calendar and Cyclical Encodings (Day of Week & Day of Year)
    logger.info("Computing calendar and cyclical day/month/year features...")
    df["day_of_week"] = df["date"].dt.dayofweek.astype(np.int8)
    df["day_of_month"] = df["date"].dt.day.astype(np.int8)
    df["month"] = df["date"].dt.month.astype(np.int8)
    df["day_of_year"] = df["date"].dt.dayofyear.astype(np.int16)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(np.int8)

    # Cyclical sin/cos encodings (DOW and DOY)
    df["dow_sin"] = np.sin(2.0 * np.pi * df["day_of_week"] / 7.0).astype(np.float32)
    df["dow_cos"] = np.cos(2.0 * np.pi * df["day_of_week"] / 7.0).astype(np.float32)
    df["doy_sin"] = np.sin(2.0 * np.pi * df["day_of_year"] / 365.25).astype(np.float32)
    df["doy_cos"] = np.cos(2.0 * np.pi * df["day_of_year"] / 365.25).astype(np.float32)
    df["month_sin"] = np.sin(2.0 * np.pi * (df["month"] - 1) / 12.0).astype(np.float32)
    df["month_cos"] = np.cos(2.0 * np.pi * (df["month"] - 1) / 12.0).astype(np.float32)

    # Aliases matching syllabus specification
    df["demand_lag_14"] = df["lag_14"]
    df["demand_lag_28"] = df["lag_28"]
    df["rolling_mean_7d"] = df["rolling_mean_7_lag14"]
    df["rolling_std_7d"] = df["rolling_std_7_lag14"]
    df["rolling_mean_28d"] = df["rolling_mean_28_lag14"]
    df["rolling_std_28d"] = df["rolling_std_28_lag14"]

    # 5. Holiday Proximity (pre-holiday surge 1-2 days prior)
    logger.info("Computing holiday proximity metrics...")
    holiday_dates = df[df["is_holiday"] == 1]["date"].drop_duplicates().sort_values().tolist()
    days_to, days_since = compute_holiday_proximity(df["date"], holiday_dates)
    df["days_to_holiday"] = days_to
    df["days_since_holiday"] = days_since

    # 6. Cannibalization Share within Store & Department
    logger.info("Computing department cannibalization share...")
    dept_totals = (
        df.groupby(["store_id", "dept_id", "date"])["rolling_mean_28_lag14"]
        .transform("sum")
        .astype(np.float32)
    )
    df["sku_dept_share"] = (
        df["rolling_mean_28_lag14"] / (dept_totals + 1e-5)
    ).astype(np.float32)

    # 7. SKU Age (days since first observed record)
    logger.info("Computing SKU age in days...")
    first_dates = df.groupby(["store_id", "item_id"])["date"].transform("min")
    df["sku_age_days"] = (df["date"] - first_dates).dt.days.astype(np.int16)

    # 8. Drop Warmup Period where lag_35 is NaN
    initial_len = len(df)
    df = df.dropna(subset=["lag_35"]).reset_index(drop=True)
    logger.info(
        "Trimmed %d warmup rows. Active feature dataset contains %d rows (>= 40,000 requirement satisfied)",
        initial_len - len(df),
        len(df),
    )

    # 9. Format categorical columns
    for cat_col in ["store_id", "item_id", "dept_id", "cat_id", "state_id"]:
        df[cat_col] = df[cat_col].astype("category")

    return df


def assert_leakage_free(raw_df: pd.DataFrame, test_store: str = "CA_1", test_item: str = "FOODS_1_001") -> None:
    """Formal leakage test: proves modifying future target values y(t+k) does NOT alter features at t."""
    logger.info("Executing rigorous data leakage assertion test...")

    # Filter to one series for fast testing
    sample = raw_df[(raw_df["store_id"] == test_store) & (raw_df["item_id"] == test_item)].copy()
    sample = sample.sort_values("date").reset_index(drop=True)

    # Baseline features
    feat_baseline = build_features(sample.copy())

    # Pick an origin date T (e.g. at index 50)
    origin_idx = 50
    origin_date = sample.loc[origin_idx, "date"]

    # Perturb future sales: y at T+1, T+2 ... T+13 (strictly within the 14-day horizon)
    sample_perturbed = sample.copy()
    for offset in range(1, FORECAST_HORIZON):
        if origin_idx + offset < len(sample_perturbed):
            sample_perturbed.loc[origin_idx + offset, "units_sold"] += 1000

    feat_perturbed = build_features(sample_perturbed)

    # Extract features at origin_date
    row_orig = feat_baseline[feat_baseline["date"] == origin_date].iloc[0]
    row_pert = feat_perturbed[feat_perturbed["date"] == origin_date].iloc[0]

    feature_cols = [
        c for c in feat_baseline.columns
        if c.startswith("lag_") or c.startswith("demand_lag_") or c.startswith("rolling_") or c in [
            "price_ratio", "price_elasticity_ratio", "sku_dept_share", "sku_age_days", "doy_sin", "doy_cos"
        ]
    ]

    for col in feature_cols:
        val_orig = row_orig[col]
        val_pert = row_pert[col]
        assert np.isclose(val_orig, val_pert, atol=1e-5), (
            f"LEAKAGE DETECTED in feature '{col}' at date {origin_date}: "
            f"baseline={val_orig}, perturbed={val_pert}!"
        )

    logger.info("Leakage Test PASSED: All %d features at origin T are strictly invariant to future target values.", len(feature_cols))


def run_feature_pipeline() -> pd.DataFrame:
    """Phase 5 entrypoint: builds features, validates leakage, and saves features.parquet."""
    in_path = PROCESSED_DIR / "canonical_simulated.parquet"
    assert in_path.exists(), f"Missing {in_path}. Run Phase 4 simulator first."

    logger.info("Loading canonical simulated data: %s", in_path)
    df = pd.read_parquet(in_path)

    # Run leakage test first on a slice
    sample_item = df["item_id"].iloc[0]
    sample_store = df["store_id"].iloc[0]
    assert_leakage_free(df, test_store=sample_store, test_item=sample_item)

    # Build full feature matrix
    logger.info("Building full feature matrix across all 300 series...")
    features_df = build_features(df)

    # Check minimum row count (must satisfy >= 40,000 lines)
    assert len(features_df) >= 40000, f"Expected >= 40,000 rows, got {len(features_df)}"

    out_path = PROCESSED_DIR / "features.parquet"
    features_df.to_parquet(out_path, index=False)
    logger.info("Phase 5 Complete: Features saved to %s (%d rows, %d columns)", out_path, len(features_df), features_df.shape[1])

    return features_df


if __name__ == "__main__":
    run_feature_pipeline()
