"""Tests for the Streamlit Studio's data ingestion and forecasting engine."""
import io

import pandas as pd
import pytest

from src.app.data import clean_sales_data, template_csv_bytes
from src.app.forecast import Settings, run_analysis


@pytest.fixture(scope="module")
def template_df() -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(template_csv_bytes()))


def test_template_roundtrip_and_forecast(template_df: pd.DataFrame):
    clean, report = clean_sales_data(template_df)
    assert report.errors == []
    assert report.rows_out == len(template_df)
    assert report.series == 2

    result = run_analysis(clean, Settings(horizon=14))
    assert len(result.forecasts) == report.series * 14
    assert set(result.forecasts.columns) == {
        "store_id", "item_id", "date", "step", "forecast", "lower", "upper"
    }
    assert (result.forecasts["forecast"] >= 0).all()
    assert (result.forecasts["upper"] >= result.forecasts["lower"]).all()
    assert result.metrics["series"] == report.series

    plan = result.replenishment
    assert {"recommended_qty", "reorder_point", "safety_stock", "urgency", "order_cost"} <= set(plan.columns)
    assert (plan["recommended_qty"] >= 0).all()


def test_alias_columns_and_missing_optional_are_handled():
    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    raw = pd.DataFrame(
        {
            "ds": dates.strftime("%Y-%m-%d"),
            "sku": "SKU-A",
            "sales": [10 + (3 if d.dayofweek >= 5 else 0) for d in dates],
        }
    )
    clean, report = clean_sales_data(raw)
    assert report.mapped_columns["date"] == "ds"
    assert report.mapped_columns["item_id"] == "sku"
    assert report.mapped_columns["units_sold"] == "sales"
    assert set(clean["store_id"].unique()) == {"STORE_1"}

    result = run_analysis(clean, Settings(horizon=7))
    assert len(result.forecasts) == 7
    assert result.metrics["method"] == "Weekly average (fallback)"


def test_missing_required_column_raises():
    raw = pd.DataFrame({"date": ["2024-01-01"], "item_id": ["A"]})
    with pytest.raises(ValueError, match="Missing required"):
        clean_sales_data(raw)


def test_duplicate_rows_are_merged():
    raw = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01", "2024-01-02"],
            "store_id": ["S", "S", "S"],
            "item_id": ["A", "A", "A"],
            "units_sold": [2, 3, 4],
        }
    )
    clean, report = clean_sales_data(raw)
    assert report.duplicates == 1
    first = clean.iloc[0]
    assert first["units_sold"] == 5
