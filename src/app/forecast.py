"""Forecasting and replenishment engine for the Restok Streamlit app.

Designed to be understandable first:

1. Build one-step-ahead training rows from each store/item history.
2. Fit a single global LightGBM Tweedie model (or a weekly-average fallback
   when there is not enough history).
3. Forecast the next N days by feeding predictions back one day at a time.
4. Convert forecasts into purchase orders with the classic
   reorder-point + safety-stock formula.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:  # LightGBM is a core dependency, but never let the UI hard-crash without it.
    import lightgbm as lgb

    _HAS_LIGHTGBM = True
except Exception:  # pragma: no cover - optional safety net
    lgb = None
    _HAS_LIGHTGBM = False

MIN_HISTORY_DAYS = 28
MIN_TRAIN_ROWS = 60

FEATURE_COLUMNS = [
    "dow",
    "dom",
    "month",
    "is_weekend",
    "dow_sin",
    "dow_cos",
    "doy_sin",
    "doy_cos",
    "month_sin",
    "month_cos",
    "promo",
    "price",
    "lag_1",
    "lag_2",
    "lag_7",
    "lag_14",
    "lag_28",
    "mean_7",
    "mean_14",
    "mean_28",
    "std_7",
    "std_28",
    "min_7",
    "max_7",
    "trend_7_28",
]


@dataclass
class Settings:
    """Everything the user can tune in the sidebar."""

    horizon: int = 14
    lead_time: int = 7
    case_pack: int = 6
    review_period: int = 7
    service_level: float = 0.90
    cover_days: float = 10.0
    sell_price: float = 9.99
    unit_cost: float = 6.5
    shelf_life: int = 30


@dataclass
class AnalysisResult:
    forecasts: pd.DataFrame
    replenishment: pd.DataFrame
    history: pd.DataFrame
    metrics: Dict[str, object] = field(default_factory=dict)


def _safe_float(value: object, fallback: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if np.isfinite(number) else fallback


def _feature_row(
    history: np.ndarray,
    date: pd.Timestamp,
    promo: float,
    price: float,
) -> Dict[str, float]:
    """Builds the features for one day using only prior observations."""
    values = np.asarray(history, dtype=float)
    if values.size == 0:
        values = np.zeros(1)

    def lag(k: int) -> float:
        return float(values[-k]) if values.size >= k else float(values.mean())

    last7 = values[-7:]
    last14 = values[-14:]
    last28 = values[-28:]

    dow = date.dayofweek
    doy = date.dayofyear
    month = date.month

    return {
        "dow": dow,
        "dom": date.day,
        "month": month,
        "is_weekend": int(dow >= 5),
        "dow_sin": float(np.sin(2 * np.pi * dow / 7)),
        "dow_cos": float(np.cos(2 * np.pi * dow / 7)),
        "doy_sin": float(np.sin(2 * np.pi * doy / 365.25)),
        "doy_cos": float(np.cos(2 * np.pi * doy / 365.25)),
        "month_sin": float(np.sin(2 * np.pi * (month - 1) / 12)),
        "month_cos": float(np.cos(2 * np.pi * (month - 1) / 12)),
        "promo": _safe_float(promo, 0.0),
        "price": _safe_float(price, 0.0),
        "lag_1": lag(1),
        "lag_2": lag(2),
        "lag_7": lag(7),
        "lag_14": lag(14),
        "lag_28": lag(28),
        "mean_7": float(np.mean(last7)),
        "mean_14": float(np.mean(last14)),
        "mean_28": float(np.mean(last28)),
        "std_7": float(np.std(last7)),
        "std_28": float(np.std(last28)),
        "min_7": float(np.min(last7)),
        "max_7": float(np.max(last7)),
        "trend_7_28": float(np.mean(last7) - np.mean(last28)),
    }


def build_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Turns every store/item history into supervised one-step-ahead rows."""
    records: List[Dict[str, float]] = []
    for _, group in df.groupby(["store_id", "item_id"], sort=False):
        group = group.sort_values("date")
        if len(group) <= MIN_HISTORY_DAYS:
            continue

        sales = group["units_sold"].to_numpy(dtype=float)
        dates = group["date"].to_numpy()
        promos = group["is_promotion"].to_numpy(dtype=float)
        prices = group["sell_price"].to_numpy(dtype=float)

        for index in range(MIN_HISTORY_DAYS, len(group)):
            row = _feature_row(
                sales[:index],
                pd.Timestamp(dates[index]),
                promos[index],
                prices[index],
            )
            row["target"] = float(sales[index])
            records.append(row)

    return pd.DataFrame(records)


def _new_model():
    return lgb.LGBMRegressor(
        objective="tweedie",
        tweedie_variance_power=1.5,
        n_estimators=350,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=15,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )


def _recursive_forecast(
    group: pd.DataFrame,
    start_date: pd.Timestamp,
    horizon: int,
    model,
    residual_std: float,
    service_level: float,
) -> List[Dict[str, object]]:
    """Forecasts `horizon` days, feeding each prediction back into the history."""
    history = group.sort_values("date")["units_sold"].to_numpy(dtype=float).tolist()
    last_price = _safe_float(group.sort_values("date")["sell_price"].iloc[-1], 0.0)
    z = NormalDist().inv_cdf(min(max(service_level, 0.5), 0.999))
    rows: List[Dict[str, object]] = []

    for step in range(1, horizon + 1):
        target_date = start_date + pd.Timedelta(days=step)
        features = _feature_row(np.asarray(history), target_date, 0.0, last_price)
        x_row = pd.DataFrame([features])[FEATURE_COLUMNS]
        prediction = max(0.0, float(model.predict(x_row)[0]))

        band = z * max(residual_std, 0.0) * np.sqrt(step)
        rows.append(
            {
                "date": target_date,
                "step": step,
                "forecast": prediction,
                "lower": max(0.0, prediction - band),
                "upper": prediction + band,
            }
        )
        history.append(prediction)

    return rows


def _weekly_average_forecast(
    group: pd.DataFrame,
    horizon: int,
    residual_std: float,
    service_level: float,
) -> List[Dict[str, object]]:
    """Simple, explainable fallback: recent level x weekday seasonality."""
    group = group.sort_values("date")
    sales = group["units_sold"].to_numpy(dtype=float)
    dates = pd.to_datetime(group["date"])
    last_date = dates.max()

    recent = sales[-14:] if len(sales) >= 14 else sales
    level = float(np.mean(recent)) if len(recent) else 0.0

    weekly = pd.DataFrame({"dow": dates.dt.dayofweek.to_numpy(), "sales": sales})
    weekday_means = weekly.groupby("dow")["sales"].mean()
    overall = float(weekday_means.mean()) if len(weekday_means) else 0.0
    factors = (weekday_means / overall) if overall > 0 else weekday_means * 0 + 1.0

    if residual_std <= 0 and len(sales) > 1:
        residual_std = float(np.std(sales[-28:]))

    z = NormalDist().inv_cdf(min(max(service_level, 0.5), 0.999))
    rows: List[Dict[str, object]] = []
    for step in range(1, horizon + 1):
        target_date = last_date + pd.Timedelta(days=step)
        factor = float(factors.get(target_date.dayofweek, 1.0))
        prediction = max(0.0, level * factor)
        band = z * max(residual_std, 0.0) * np.sqrt(step)
        rows.append(
            {
                "date": target_date,
                "step": step,
                "forecast": prediction,
                "lower": max(0.0, prediction - band),
                "upper": prediction + band,
            }
        )
    return rows


def _backtest(
    df: pd.DataFrame,
    settings: Settings,
) -> Tuple[Optional[float], float, int]:
    """Holds out the last `horizon` days to estimate accuracy and uncertainty."""
    global_cutoff = df["date"].max() - pd.Timedelta(days=settings.horizon)
    train_df = df[df["date"] <= global_cutoff]

    training = build_training_frame(train_df)
    if training.empty or len(training) < MIN_TRAIN_ROWS:
        return None, 0.0, 0

    model = _new_model()
    model.fit(training[FEATURE_COLUMNS], training["target"])

    actuals: List[float] = []
    predictions: List[float] = []
    for _, group in df.groupby(["store_id", "item_id"], sort=False):
        group = group.sort_values("date")
        valid_history = group[group["date"] <= global_cutoff]
        future = group[group["date"] > global_cutoff]
        if valid_history.empty or future.empty:
            continue

        origin = valid_history["date"].max()
        forecast = _recursive_forecast(
            valid_history,
            origin,
            settings.horizon,
            model,
            residual_std=0.0,
            service_level=0.5,
        )
        pred_by_date = {row["date"]: row["forecast"] for row in forecast}
        for _, actual_row in future.iterrows():
            predicted = pred_by_date.get(actual_row["date"])
            if predicted is not None:
                actuals.append(float(actual_row["units_sold"]))
                predictions.append(float(predicted))

    if not actuals:
        return None, 0.0, 0

    y_true = np.asarray(actuals)
    y_pred = np.asarray(predictions)
    denominator = float(np.sum(y_true))
    wape = float(np.sum(np.abs(y_true - y_pred)) / denominator) if denominator > 0 else None
    residuals = y_true - y_pred
    residual_std = float(np.std(residuals)) if len(residuals) > 1 else 0.0
    return wape, residual_std, len(actuals)


def build_replenishment(
    forecasts: pd.DataFrame,
    history: pd.DataFrame,
    settings: Settings,
    residual_std: float,
) -> pd.DataFrame:
    """Turns daily forecasts into a reorder-point purchase order table."""
    z = NormalDist().inv_cdf(min(max(settings.service_level, 0.5), 0.999))
    rows: List[Dict[str, object]] = []

    for (store_id, item_id), group in history.groupby(["store_id", "item_id"], sort=False):
        group = group.sort_values("date")
        last = group.iloc[-1]
        series_forecast = forecasts[
            (forecasts["store_id"] == store_id) & (forecasts["item_id"] == item_id)
        ].sort_values("date")

        if series_forecast.empty:
            continue

        lead_time = max(1, int(_safe_float(last.get("lead_time"), settings.lead_time)))
        case_pack = max(1, int(_safe_float(last.get("case_pack"), settings.case_pack)))
        unit_cost = _safe_float(last.get("unit_cost"), settings.unit_cost)
        sell_price = _safe_float(last.get("sell_price"), settings.sell_price)

        daily = series_forecast["forecast"].to_numpy(dtype=float)
        lead_slice = daily[:lead_time] if len(daily) >= lead_time else daily
        daily_mean = float(np.mean(lead_slice)) if len(lead_slice) else 0.0

        recent_sales = group["units_sold"].to_numpy(dtype=float)[-28:]
        series_std = float(np.std(recent_sales)) if len(recent_sales) > 1 else 0.0
        daily_std = residual_std if residual_std > 0 else series_std

        safety_stock = z * daily_std * np.sqrt(lead_time)
        reorder_point = daily_mean * lead_time + safety_stock
        target_stock = daily_mean * (lead_time + settings.review_period) + safety_stock

        raw_on_hand = last.get("on_hand")
        on_hand = _safe_float(raw_on_hand, np.nan)
        assumed = False
        if not np.isfinite(on_hand) or on_hand < 0:
            on_hand = daily_mean * settings.cover_days
            assumed = True
        on_hand = max(0.0, float(on_hand))

        needed = target_stock - on_hand
        if on_hand <= reorder_point and needed > 0:
            order_qty = int(np.ceil(needed / case_pack) * case_pack)
            if on_hand <= 0:
                urgency = "Critical"
            elif on_hand < reorder_point:
                urgency = "High"
            else:
                urgency = "Normal"
            status = "Pending"
        else:
            order_qty = 0
            urgency = "Covered"
            status = "Covered"

        rows.append(
            {
                "store_id": store_id,
                "item_id": item_id,
                "on_hand": round(on_hand, 1),
                "on_hand_assumed": assumed,
                "avg_daily_demand": round(daily_mean, 2),
                "lead_time": lead_time,
                "case_pack": case_pack,
                "reorder_point": round(reorder_point, 1),
                "safety_stock": round(safety_stock, 1),
                "target_stock": round(target_stock, 1),
                "recommended_qty": order_qty,
                "days_of_cover": round(on_hand / daily_mean, 1) if daily_mean > 0 else np.inf,
                "unit_cost": round(unit_cost, 2),
                "sell_price": round(sell_price, 2),
                "order_cost": round(order_qty * unit_cost, 2),
                "urgency": urgency,
                "status": status,
            }
        )

    table = pd.DataFrame(rows)
    if table.empty:
        return table
    urgency_rank = {"Critical": 0, "High": 1, "Normal": 2, "Covered": 3}
    table["_rank"] = table["urgency"].map(urgency_rank).fillna(4)
    table = table.sort_values(["_rank", "order_cost"], ascending=[True, False]).drop(columns="_rank")
    return table.reset_index(drop=True)


def run_analysis(df: pd.DataFrame, settings: Settings) -> AnalysisResult:
    """Main entry point: forecast every series and produce purchase orders."""
    if df.empty:
        raise ValueError("The dataset is empty.")

    history = df.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)
    history = history[["store_id", "item_id", "date", "units_sold", "sell_price",
                       "is_promotion", "on_hand", "lead_time", "case_pack",
                       "unit_cost", "shelf_life"]]

    method = "LightGBM Tweedie (global model)"
    wape: Optional[float] = None
    residual_std = 0.0

    training = build_training_frame(history)
    model = None
    if _HAS_LIGHTGBM and len(training) >= MIN_TRAIN_ROWS:
        wape, residual_std, _ = _backtest(history, settings)
        model = _new_model()
        model.fit(training[FEATURE_COLUMNS], training["target"])
    else:
        method = "Weekly average (fallback)"

    forecast_rows: List[Dict[str, object]] = []
    for (store_id, item_id), group in history.groupby(["store_id", "item_id"], sort=False):
        group = group.sort_values("date")
        origin = group["date"].max()

        if model is not None:
            series_rows = _recursive_forecast(
                group,
                origin,
                settings.horizon,
                model,
                residual_std,
                settings.service_level,
            )
        else:
            series_rows = _weekly_average_forecast(
                group,
                settings.horizon,
                residual_std,
                settings.service_level,
            )

        for row in series_rows:
            row.update({"store_id": store_id, "item_id": item_id})
        forecast_rows.extend(series_rows)

    forecasts = pd.DataFrame(forecast_rows)
    if forecasts.empty:
        raise ValueError("No forecasts could be produced from this data.")

    forecasts = forecasts[["store_id", "item_id", "date", "step", "forecast", "lower", "upper"]]
    forecasts[["forecast", "lower", "upper"]] = forecasts[
        ["forecast", "lower", "upper"]
    ].round(2)

    replenishment = build_replenishment(forecasts, history, settings, residual_std)

    metrics = {
        "method": method,
        "wape": wape,
        "residual_std": round(residual_std, 3),
        "series": int(history.groupby(["store_id", "item_id"]).ngroups),
        "rows": int(len(history)),
        "training_rows": int(len(training)),
        "horizon": settings.horizon,
        "service_level": settings.service_level,
        "date_min": history["date"].min(),
        "date_max": history["date"].max(),
    }

    return AnalysisResult(
        forecasts=forecasts,
        replenishment=replenishment,
        history=history,
        metrics=metrics,
    )
