"""
Smoke integration tests for core CyberGuard AI endpoints:
- GET /health
- POST /ingest
- GET /risk-summary
- POST /optimize
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.database import init_db, SessionLocal
from backend.synthetic_generator import generate_synthetic_data


@pytest.fixture(scope="module")
def client():
    """Create test client and seed initial test database."""
    init_db()
    with SessionLocal() as db:
        generate_synthetic_data(db)
    with TestClient(app) as c:
        yield c


def test_health_smoke(client):
    """GET /health returns 200 and healthy status."""
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "database" in data
    assert "assets_registered" in data


def test_ingest_smoke(client):
    """POST /ingest with {"reseed_synthetic": true} returns 200 and expected counts."""
    res = client.post("/ingest", json={"reseed_synthetic": True})
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert "assets_ingested" in data
    assert "findings_ingested" in data
    assert "remediations_ingested" in data
    assert data["assets_ingested"] >= 20
    assert data["findings_ingested"] >= 100


def test_risk_summary_smoke(client):
    """GET /risk-summary returns 200 with formatted_total_risk, total_findings, total_assets."""
    res = client.get("/risk-summary")
    assert res.status_code == 200
    data = res.json()
    assert "formatted_total_risk" in data
    assert "total_findings" in data
    assert "total_assets" in data
    assert data["total_assets"] >= 20
    assert data["total_findings"] >= 100
    assert isinstance(data["formatted_total_risk"], str)
    assert data["formatted_total_risk"].startswith("$")


def test_optimize_smoke(client):
    """POST /optimize returns 200 with all required knapsack & ranking fields."""
    payload = {"budget": 50000.0, "strategy": "pulp"}
    res = client.post("/optimize", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "total_cost_allocated" in data
    assert "total_risk_reduction" in data
    assert "overall_roi" in data
    assert "selected_actions" in data
    assert "ranked_controls" in data
    assert data["total_cost_allocated"] <= 50000.0
    assert len(data["selected_actions"]) > 0
    assert len(data["ranked_controls"]) >= 100
