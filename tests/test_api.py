"""API and End-to-End Integration Tests for Restok."""
import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["series_tracked"] == 300


def test_kpis_endpoint(client):
    response = client.get("/api/kpis")
    assert response.status_code == 200
    data = response.json()
    assert "model_wape" in data
    assert "stockout_reduction_pct" in data
    assert data["total_series"] == 300


def test_stores_and_items_endpoints(client):
    stores_res = client.get("/api/stores")
    assert stores_res.status_code == 200
    stores = stores_res.json()
    assert len(stores) == 5

    items_res = client.get("/api/items?store_id=CA_1")
    assert items_res.status_code == 200
    items = items_res.json()
    assert len(items) == 60


def test_forecasts_endpoint(client):
    res = client.get("/api/forecasts?store_id=CA_1&item_id=FOODS_1_012")
    assert res.status_code == 200
    data = res.json()
    assert "meta" in data
    assert "history" in data
    assert "forecast" in data
    assert len(data["forecast"]) == 14
    assert "pred_q_90" in data["forecast"][0]


def test_policies_and_tradeoff_endpoints(client):
    pol_res = client.get("/api/policies")
    assert pol_res.status_code == 200
    pols = pol_res.json()
    assert len(pols) == 3

    tradeoff_res = client.get("/api/tradeoff")
    assert tradeoff_res.status_code == 200
    tradeoff = tradeoff_res.json()
    assert len(tradeoff) > 0


def test_purchase_orders_workflow(client):
    po_res = client.get("/api/purchase-orders")
    assert po_res.status_code == 200
    pos = po_res.json()
    assert len(pos) > 0

    first_po_id = pos[0]["po_id"]
    # Test update qty and status
    update_res = client.post(f"/api/purchase-orders/{first_po_id}/update", json={"order_qty": 36, "status": "Approved"})
    assert update_res.status_code == 200
    assert update_res.json()["po"]["order_qty"] == 36
    assert update_res.json()["po"]["status"] == "Approved"


def test_chatbot_kpi_query(client):
    res = client.post("/api/chat", json={"message": "What is our model WAPE and accuracy?"})
    assert res.status_code == 200
    data = res.json()
    assert "reply" in data
    assert "WAPE" in data["reply"]
    assert data["engine"] in ["local_fallback", "gemini-2.5-flash"]


def test_chatbot_censoring_query(client):
    res = client.post("/api/chat", json={"message": "Explain the bias in the censoring study"})
    assert res.status_code == 200
    data = res.json()
    assert "reply" in data
    assert "bias" in data["reply"].lower()


def test_chatbot_sku_query(client):
    res = client.post("/api/chat", json={"message": "What is the status of FOODS_1_012 in CA_1?"})
    assert res.status_code == 200
    data = res.json()
    assert "reply" in data
    assert "FOODS_1_012" in data["reply"]

