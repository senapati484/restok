"""Phase 7: Censoring Study for Restok.

Compares 4 distinct strategies for handling stockout censoring in historical sales:
1. Naive (train on raw censored units_sold)
2. Dropping Stockout Days
3. Rolling Mean Imputation (replace stockout days with 28-day rolling average)
4. Iterative Model Imputation (EM-style: predict latent demand on stockout days and re-fit)

All 4 models are evaluated against true_demand on the test set to quantify
the true demand recovery and under-forecasting bias reduction.
Outputs headline table to data/processed/censoring_comparison.csv.
"""
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, FORECAST_HORIZON
from src.models.train import prepare_temporal_splits, compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_true_demand_metrics(y_true_demand: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes evaluation metrics against ground-truth true demand."""
    denom = np.sum(y_true_demand)
    wape = np.sum(np.abs(y_true_demand - y_pred)) / denom if denom > 0 else 0.0
    rmse = np.sqrt(np.mean((y_true_demand - y_pred) ** 2))
    mae = np.mean(np.abs(y_true_demand - y_pred))
    bias = np.sum(y_pred - y_true_demand) / denom if denom > 0 else 0.0
    return {
        "WAPE_vs_True_Demand": float(wape),
        "RMSE_vs_True_Demand": float(rmse),
        "MAE_vs_True_Demand": float(mae),
        "Forecast_Bias": float(bias),
    }


def fit_and_evaluate(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test_true_demand: np.ndarray,
    method_name: str,
) -> Dict:
    """Helper to train a standard LightGBM model and evaluate against true demand."""
    params = {
        "objective": "tweedie",
        "tweedie_variance_power": 1.5,
        "learning_rate": 0.08,
        "num_leaves": 35,
        "min_child_samples": 25,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_estimators": 120,
        "random_state": 42,
        "verbose": -1,
        "n_jobs": -1,
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)

    preds = np.clip(model.predict(X_test), 0, None)
    metrics = compute_true_demand_metrics(y_test_true_demand, preds)
    return {"Method": method_name, **metrics}


def run_censoring_study() -> pd.DataFrame:
    """Executes the Phase 7 Censoring Study."""
    features_path = PROCESSED_DIR / "features.parquet"
    assert features_path.exists(), f"Missing {features_path}. Run Phase 5 first."

    logger.info("Loading feature dataset: %s", features_path)
    df = pd.read_parquet(features_path)

    train_df, val_df, test_df, feature_cols, cat_cols = prepare_temporal_splits(df, horizon_days=FORECAST_HORIZON)
    train_val_df = pd.concat([train_df, val_df]).copy()

    X_test = test_df[feature_cols]
    y_test_true_demand = test_df["true_demand"].values

    results = []

    # Method 1: Naive (train on raw censored units_sold)
    logger.info("Evaluating Method 1: Naive (raw censored sales)...")
    res_naive = fit_and_evaluate(
        train_val_df[feature_cols],
        train_val_df["units_sold"].values,
        X_test,
        y_test_true_demand,
        "1. Naive (Censored Sales)",
    )
    results.append(res_naive)

    # Method 2: Dropping Stockout Days from Training Set
    logger.info("Evaluating Method 2: Dropping Stockout Days...")
    clean_train_val = train_val_df[train_val_df["is_stockout"] == 0].copy()
    res_drop = fit_and_evaluate(
        clean_train_val[feature_cols],
        clean_train_val["units_sold"].values,
        X_test,
        y_test_true_demand,
        "2. Drop Stockout Days",
    )
    results.append(res_drop)

    # Method 3: Rolling Mean Imputation
    logger.info("Evaluating Method 3: Rolling Mean Imputation...")
    imputed_df = train_val_df.copy()
    imputed_target = np.where(
        imputed_df["is_stockout"] == 1,
        np.maximum(imputed_df["units_sold"], imputed_df["rolling_mean_28_lag14"]),
        imputed_df["units_sold"],
    )
    res_rolling = fit_and_evaluate(
        imputed_df[feature_cols],
        imputed_target,
        X_test,
        y_test_true_demand,
        "3. Rolling Mean Imputation",
    )
    results.append(res_rolling)

    # Method 4: Iterative Model Imputation (EM-style)
    logger.info("Evaluating Method 4: Iterative Model Imputation...")
    # Step A: Fit model on non-stockout days
    initial_model = lgb.LGBMRegressor(
        objective="tweedie",
        tweedie_variance_power=1.5,
        learning_rate=0.08,
        num_leaves=35,
        min_child_samples=25,
        n_estimators=100,
        random_state=42,
        verbose=-1,
        n_jobs=-1,
    )
    initial_model.fit(clean_train_val[feature_cols], clean_train_val["units_sold"].values)

    # Step B: Predict latent demand on all training records
    latent_preds = np.clip(initial_model.predict(train_val_df[feature_cols]), 0, None)

    # Step C: Replace stockout days with max(units_sold, latent_pred)
    em_target = np.where(
        train_val_df["is_stockout"] == 1,
        np.maximum(train_val_df["units_sold"].values, latent_preds),
        train_val_df["units_sold"].values,
    )

    res_em = fit_and_evaluate(
        train_val_df[feature_cols],
        em_target,
        X_test,
        y_test_true_demand,
        "4. Iterative Imputation (EM)",
    )
    results.append(res_em)

    # Compile Headline Comparison Table
    summary_df = pd.DataFrame(results)
    out_path = PROCESSED_DIR / "censoring_comparison.csv"
    summary_df.to_csv(out_path, index=False)
    logger.info("Phase 7 Complete: Headline Censoring Comparison saved to %s", out_path)

    return summary_df


if __name__ == "__main__":
    run_censoring_study()
