"""Data ingestion, validation, and demo loading for the Restok Streamlit app.

The whole app speaks one simple long-format schema:

    date | store_id | item_id | units_sold | + optional columns

User files are auto-mapped (common aliases like ``sku`` or ``sales`` work),
validated, and normalised here so the forecasting code never has to care
where the data came from.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

REQUIRED_COLUMNS = ["date", "item_id", "units_sold"]

# Order used in the downloadable template and in the docs.
TEMPLATE_COLUMNS = [
    "date",
    "store_id",
    "item_id",
    "units_sold",
    "sell_price",
    "is_promotion",
    "on_hand",
    "lead_time",
    "case_pack",
    "unit_cost",
    "shelf_life",
]

NUMERIC_COLUMNS = [
    "units_sold",
    "sell_price",
    "is_promotion",
    "on_hand",
    "lead_time",
    "case_pack",
    "unit_cost",
    "shelf_life",
]

COLUMN_ALIASES: Dict[str, List[str]] = {
    "date": ["date", "ds", "day", "timestamp", "datetime", "sales_date", "order_date"],
    "store_id": ["store_id", "store", "shop_id", "shop", "location_id", "location", "outlet_id", "branch"],
    "item_id": ["item_id", "sku", "sku_id", "product_id", "product", "item", "article_id"],
    "units_sold": ["units_sold", "sales", "units", "quantity", "qty", "demand", "sold", "unit_sales"],
    "sell_price": ["sell_price", "price", "unit_price", "retail_price"],
    "is_promotion": ["is_promotion", "promo", "promotion", "on_promo", "is_promo"],
    "on_hand": ["on_hand", "on_hand_inventory", "inventory", "stock", "stock_on_hand", "current_stock"],
    "lead_time": ["lead_time", "lead_time_days", "supplier_lead_time"],
    "case_pack": ["case_pack", "case_size", "pack_size", "moq"],
    "unit_cost": ["unit_cost", "cost", "cost_price", "unit_cost_usd"],
    "shelf_life": ["shelf_life", "shelf_life_days"],
}


@dataclass
class DataReport:
    """Human-readable summary of what happened while loading a file."""

    rows_in: int = 0
    rows_out: int = 0
    series: int = 0
    dropped: int = 0
    duplicates: int = 0
    negatives: int = 0
    date_min: Optional[pd.Timestamp] = None
    date_max: Optional[pd.Timestamp] = None
    mapped_columns: Dict[str, str] = field(default_factory=dict)
    missing_optional: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


def _normalise_name(name: object) -> str:
    """Lowercase and snake_case a column name so aliases match reliably."""
    text = str(name).strip().lower()
    for ch in (" ", "-", "/", "."):
        text = text.replace(ch, "_")
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


def _map_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Renames flexible input columns to the canonical schema."""
    normalised = {col: _normalise_name(col) for col in df.columns}
    chosen: Dict[str, object] = {}

    # First pass: exact canonical names win over aliases.
    for target in COLUMN_ALIASES:
        for col, name in normalised.items():
            if name == target and target not in chosen:
                chosen[target] = col

    # Second pass: fill gaps using aliases.
    for target, aliases in COLUMN_ALIASES.items():
        if target in chosen:
            continue
        for col, name in normalised.items():
            if name in aliases and col not in chosen.values():
                chosen[target] = col
                break

    rename = {source: target for target, source in chosen.items()}
    mapped = {target: str(source) for target, source in chosen.items()}
    return df.rename(columns=rename), mapped


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Converts known numeric columns without raising on messy values."""
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def clean_sales_data(raw: pd.DataFrame) -> Tuple[pd.DataFrame, DataReport]:
    """Normalises any user table into the canonical long-format sales schema.

    Raises ``ValueError`` only when nothing usable is left (missing required
    columns). Everything else is reported back in ``DataReport`` so the UI can
    explain exactly what was cleaned.
    """
    report = DataReport(rows_in=len(raw))

    if raw.empty:
        raise ValueError("The uploaded file is empty.")

    df, report.mapped_columns = _map_columns(raw.copy())

    missing_required = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_required:
        raise ValueError(
            "Missing required column(s): "
            + ", ".join(missing_required)
            + ". Download the template to see the expected layout."
        )

    # Keep only canonical columns, but remember which optional ones were absent.
    for col in TEMPLATE_COLUMNS:
        if col not in df.columns:
            report.missing_optional.append(col)
    available = [c for c in TEMPLATE_COLUMNS if c in df.columns]
    df = df[available].copy()

    # Guarantee every canonical column exists so downstream code is simple.
    for col in TEMPLATE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    # Dates.
    original_dates = df["date"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    bad_dates = int(df["date"].isna().sum() - original_dates.isna().sum())
    if bad_dates > 0:
        report.warnings.append(f"{bad_dates} row(s) had an unreadable date and were dropped.")

    # Identifiers and measure.
    df["item_id"] = df["item_id"].astype("string").str.strip()
    store_missing = df["store_id"].isna().all()
    df["store_id"] = df["store_id"].astype("string").str.strip().fillna("STORE_1")
    if store_missing:
        report.warnings.append("No store column found, so all rows were treated as one store ('STORE_1').")

    df = _coerce_numeric(df)
    df["units_sold"] = df["units_sold"].fillna(0.0)

    before = len(df)
    df = df.dropna(subset=["date", "item_id", "units_sold"])
    df = df[df["item_id"].str.len() > 0]
    report.dropped = before - len(df)
    if report.dropped > 0:
        report.warnings.append(f"{report.dropped} row(s) with missing date/item/sales were dropped.")

    if df.empty:
        raise ValueError("No valid rows remained after cleaning. Check the date and sales columns.")

    # Negative demand is almost always a return/adjustment; clamp it to zero.
    negative_mask = df["units_sold"] < 0
    report.negatives = int(negative_mask.sum())
    if report.negatives:
        report.warnings.append(f"{report.negatives} negative sales value(s) were clipped to 0.")
        df.loc[negative_mask, "units_sold"] = 0.0

    # Promotions become a simple 0/1 flag.
    if "is_promotion" in df.columns:
        df["is_promotion"] = (df["is_promotion"].fillna(0) > 0).astype(int)
    else:
        df["is_promotion"] = 0

    # Collapse duplicate store/item/date rows produced by exports or joins.
    before = len(df)
    agg = {"units_sold": "sum"}
    for col in df.columns:
        if col not in ("date", "store_id", "item_id", "units_sold"):
            agg[col] = "last"
    df = (
        df.sort_values("date")
        .groupby(["store_id", "item_id", "date"], as_index=False)
        .agg(agg)
    )
    report.duplicates = before - len(df)
    if report.duplicates:
        report.warnings.append(f"{report.duplicates} duplicate store/item/date row(s) were merged.")

    # Give missing prices/costs sensible defaults so the math never breaks.
    if "sell_price" not in df.columns or df["sell_price"].isna().all():
        df["sell_price"] = np.nan
    if "unit_cost" not in df.columns or df["unit_cost"].isna().all():
        df["unit_cost"] = np.nan

    df = df.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)

    report.rows_out = len(df)
    report.series = int(df.groupby(["store_id", "item_id"]).ngroups)
    report.date_min = df["date"].min()
    report.date_max = df["date"].max()

    history_days = (report.date_max - report.date_min).days + 1
    if history_days < 28:
        report.warnings.append(
            "Less than 28 days of history. Forecasts will use a simple weekly-average "
            "fallback rather than the machine-learning model."
        )

    return df, report


def load_csv_bytes(payload: bytes) -> pd.DataFrame:
    """Reads uploaded CSV bytes with a helpful error message."""
    try:
        return pd.read_csv(io.BytesIO(payload))
    except Exception as exc:  # pragma: no cover - depends on user file
        raise ValueError(f"Could not read the CSV file: {exc}") from exc


def load_demo_data(max_series: int = 120) -> pd.DataFrame:
    """Loads the bundled M5 sample so the app works with zero setup.

    We use real observed sales from the project's feature store. A subset keeps
    the interactive demo quick while still showing many stores and items.
    """
    path = PROCESSED_DIR / "features.parquet"
    if not path.exists():
        raise FileNotFoundError(
            "Demo data not found. Run the data pipeline first (make pipeline) "
            "or upload your own CSV."
        )
    df = pd.read_parquet(
        path,
        columns=[
            "store_id",
            "item_id",
            "date",
            "units_sold",
            "sell_price",
            "is_promotion",
            "on_hand_inventory",
            "lead_time",
            "case_pack",
            "unit_cost",
            "shelf_life",
        ],
    )
    df = df.rename(columns={"on_hand_inventory": "on_hand"})
    df["store_id"] = df["store_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)

    if max_series and df.groupby(["store_id", "item_id"]).ngroups > max_series:
        keep = (
            df[["store_id", "item_id"]]
            .drop_duplicates()
            .head(max_series)
            .apply(tuple, axis=1)
            .tolist()
        )
        df = df[df[["store_id", "item_id"]].apply(tuple, axis=1).isin(keep)]

    df = df.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)
    return df


def template_csv_bytes() -> bytes:
    """Builds a small example CSV users can download and fill in."""
    stores = ["STORE_1"]
    items = ["ITEM_A", "ITEM_B"]
    dates = pd.date_range("2024-01-01", periods=90, freq="D")
    rows = []
    for store in stores:
        for i, item in enumerate(items):
            for j, date in enumerate(dates):
                base = 12 + 4 * i
                weekly = 3 if date.dayofweek >= 5 else 0
                trend = j * 0.05
                promo = int(j % 14 == 0)
                units = max(0, round(base + weekly + trend + promo * 6))
                rows.append(
                    {
                        "date": date.strftime("%Y-%m-%d"),
                        "store_id": store,
                        "item_id": item,
                        "units_sold": units,
                        "sell_price": 4.99 + i,
                        "is_promotion": promo,
                        "on_hand": 40 if j == len(dates) - 1 else "",
                        "lead_time": 3 + i,
                        "case_pack": 6,
                        "unit_cost": 3.0 + i,
                        "shelf_life": 30,
                    }
                )
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


def infer_defaults(df: pd.DataFrame) -> Dict[str, float]:
    """Picks sensible fallback values from whatever the user did provide."""
    def _median(col: str, fallback: float) -> float:
        if col in df.columns and df[col].notna().any():
            value = float(df[col].median())
            return value if np.isfinite(value) else fallback
        return fallback

    sell_price = _median("sell_price", 9.99)
    unit_cost = _median("unit_cost", sell_price * 0.65)
    return {
        "sell_price": sell_price,
        "unit_cost": unit_cost,
        "lead_time": float(_median("lead_time", 7)),
        "case_pack": float(_median("case_pack", 6)),
        "shelf_life": float(_median("shelf_life", 30)),
    }
