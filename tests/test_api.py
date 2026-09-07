"""
End-to-end integration tests for CyberGuard AI FastAPI endpoints.
Tests:
- GET /health
- GET /risk-summary
- POST /optimize (valid and invalid cases)
- GET /findings
- GET /assets
- GET /frameworks
- GET /executive-summary
- POST /ingest
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.database import init_db, SessionLocal
from backend.synthetic_generator import generate_synthetic_data


@pytest.fixture(scope="module")
def client():
    """Create test client and seed test database."""
    init_db()
    with SessionLocal() as db:
        generate_synthetic_data(db)
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    """GET /health returns healthy status and asset count."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["assets_registered"] >= 20


def test_risk_summary_endpoint(client):
    """GET /risk-summary returns total exposure and category breakdowns."""
    response = client.get("/risk-summary")
    assert response.status_code == 200
    data = response.json()

    assert "total_quantified_risk" in data
    assert data["total_quantified_risk"] > 0
    assert "formatted_total_risk" in data
    assert data["total_assets"] >= 20
    assert data["total_findings"] >= 50
    assert "risk_by_business_unit" in data
    assert len(data["risk_by_business_unit"]) >= 4
    assert "risk_by_asset_category" in data
    assert len(data["top_risks"]) <= 10
    assert len(data["top_risks"]) > 0


def test_optimize_endpoint_valid(client):
    """POST /optimize with budget returns optimal controls within budget."""
    payload = {"budget": 50000.0, "strategy": "pulp"}
    response = client.post("/optimize", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["budget"] == 50000.0
    assert data["total_cost_allocated"] <= 50000.0
    assert data["total_risk_reduction"] > 0
    assert data["overall_roi"] > 0
    assert len(data["selected_actions"]) > 0
    assert data["strategy_used"] == "pulp_knapsack_0_1"
    assert "sensitivity_analysis" in data


def test_optimize_endpoint_greedy(client):
    """POST /optimize with strategy=greedy works deterministically."""
    payload = {"budget": 25000.0, "strategy": "greedy"}
    response = client.post("/optimize", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_cost_allocated"] <= 25000.0
    assert data["strategy_used"] == "greedy_roi"


def test_optimize_endpoint_resilience(client):
    """POST /optimize handles empty body, string budget, and defaults gracefully without 422."""
    # 1. String budget with formatting
    res1 = client.post("/optimize", json={"budget": "$60,000", "strategy": "PULP"})
    assert res1.status_code == 200
    assert res1.json()["budget"] == 60000.0

    # 2. Empty payload
    res2 = client.post("/optimize", json={})
    assert res2.status_code == 200
    assert res2.json()["budget"] == 50000.0

    # 3. Zero / negative budget safely normalized
    res3 = client.post("/optimize", json={"budget": 0.0})
    assert res3.status_code == 200
    assert res3.json()["budget"] > 0


def test_ingest_called_twice_resilience(client):
    """POST /ingest can be called multiple times with or without payload without 422."""
    # First call with empty body
    res1 = client.post("/ingest", json={})
    assert res1.status_code == 200
    assert res1.json()["success"] is True

    # Second call with reseed flag and workflow metadata
    res2 = client.post("/ingest", json={"reseed_synthetic": True, "days_back": 14, "extra_workflow_id": "run-99"})
    assert res2.status_code == 200
    assert res2.json()["success"] is True


def test_findings_endpoint_filters(client):
    """GET /findings handles 'all', empty string, and search queries without 422."""
    # 1. Status 'all' and severity 'all'
    res1 = client.get("/findings?status=all&severity=all")
    assert res1.status_code == 200
    assert len(res1.json()) >= 50

    # 2. Empty query parameters
    res2 = client.get("/findings?status=&severity=&search=")
    assert res2.status_code == 200

    # 3. Search query
    res3 = client.get("/findings?search=Log4j")
    assert res3.status_code == 200
    assert len(res3.json()) >= 1


def test_assets_endpoint(client):
    """GET /assets returns list of 25 enterprise assets."""
    response = client.get("/assets")
    assert response.status_code == 200
    assets = response.json()
    assert len(assets) == 25
    assert all("business_unit" in a for a in assets)


def test_frameworks_endpoint(client):
    """GET /frameworks returns NIST CSF 2.0 and MITRE ATT&CK mappings."""
    response = client.get("/frameworks")
    assert response.status_code == 200
    data = response.json()
    assert "nist_csf_2_0" in data
    assert "mitre_attack" in data
    assert "fair_model" in data


def test_executive_summary_endpoint(client):
    """GET and POST /executive-summary handle zero, string, or missing budget without 422."""
    # Standard GET
    res1 = client.get("/executive-summary?budget=60000")
    assert res1.status_code == 200
    assert "CyberGuard AI quantifies" in res1.json()["executive_summary"]

    # GET with 0 or missing budget (gracefully defaulted, no 422)
    res2 = client.get("/executive-summary?budget=0")
    assert res2.status_code == 200

    # POST variant
    res3 = client.post("/executive-summary", json={"budget": "$80,000"})
    assert res3.status_code == 200
    assert res3.json()["allocated_budget"] == "$80,000"


def test_risk_calculator_endpoint(client):
    """POST /risk-calculator returns confidence-banded FAIR ranges and distribution bins."""
    payload = {
        "threat_event_frequency": 12.0,
        "vulnerability_score": 0.75,
        "primary_loss_usd": 150000.0,
        "secondary_loss_usd": 350000.0,
        "control_effectiveness_pct": 85.0,
        "risk_appetite_threshold_usd": 100000.0,
    }
    response = client.post("/risk-calculator", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "inherent_ale_band" in data
    assert "residual_ale_band" in data
    assert "lef_band" in data
    assert "reduction_band" in data
    assert "distribution_bins" in data
    assert data["inherent_ale_band"]["p50"] > data["residual_ale_band"]["p50"]
    assert "loss_event_frequency_range" in data["fair_breakdown"]


def test_scenario_analysis_endpoint(client):
    """POST /scenario-analysis returns Monte Carlo confidence bands and LEF/ALE ranges."""
    payload = {
        "scenarios": [
            {"name": "Low Budget", "budget": 25000.0},
            {"name": "Medium Budget", "budget": 75000.0},
        ]
    }
    response = client.post("/scenario-analysis", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["scenarios"]) == 2
    for s in data["scenarios"]:
        assert "confidence_band" in s
        assert "ale_band" in s
        assert "lef_band" in s
        assert "reduction_band" in s
        assert s["confidence_band"]["min"] <= s["confidence_band"]["p50"] <= s["confidence_band"]["max"]


def test_optimize_endpoint_ranked_pareto_and_audit(client):
    """POST /optimize returns ranked candidate controls, Pareto frontier, and logs to audit trail."""
    res = client.post("/optimize", json={"budget": 50000.0, "strategy": "pulp"})
    assert res.status_code == 200
    data = res.json()

    # Ranked candidate controls
    assert "ranked_controls" in data and data["ranked_controls"] is not None
    assert len(data["ranked_controls"]) > 0
    first_control = data["ranked_controls"][0]
    assert "rank" in first_control and first_control["rank"] == 1
    assert "delta_ale_cost_ratio" in first_control
    assert "is_selected" in first_control

    # Pareto frontier
    assert "pareto_frontier" in data and data["pareto_frontier"] is not None
    assert len(data["pareto_frontier"]) > 5
    assert any(pt["is_current_allocation"] for pt in data["pareto_frontier"])

    # Provenance
    assert "provenance" in data and data["provenance"] is not None
    assert "PuLP" in data["provenance"]["model_used"]

    # Check Audit Log endpoint
    audit_res = client.get("/audit-trail")
    assert audit_res.status_code == 200
    audit_logs = audit_res.json()
    assert any("AI Investment Optimizer" in log["actor"] for log in audit_logs)



