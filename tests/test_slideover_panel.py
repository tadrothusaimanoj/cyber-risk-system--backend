"""
Unit & Integration Tests for Phase 4: Reusable Slide-Over Drill-Down Panel.
Tests:
- Correct data rendering per type (Finding detail, Risk Calculator provenance)
- State preservation & read-only guarantee (zero data mutation)
- Conditional audit logging (Critical findings logged; routine views NOT logged)
- Error handling (404 for missing IDs, 400 for unsupported types)
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from datetime import datetime, timezone

from backend.main import app
from backend.database import get_db, SessionLocal, engine, Base
from backend.models import AssetModel, FindingModel, RemediationModel, AuditLogModel
from backend.synthetic_generator import generate_synthetic_data


@pytest.fixture(scope="module")
def client():
    # Ensure fresh test database schema
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    if db.query(FindingModel).count() == 0:
        generate_synthetic_data(db)
    db.close()
    return TestClient(app)


def test_slideover_finding_detail_rendering(client):
    """
    Asserts GET /drilldown/finding/{id} returns complete finding,
    asset context, remediation controls, and MITRE mapping.
    """
    db = SessionLocal()
    finding = db.query(FindingModel).first()
    assert finding is not None, "Test requires at least one seeded finding"
    finding_id = finding.finding_id
    db.close()

    res = client.get(f"/drilldown/finding/{finding_id}")
    assert res.status_code == 200
    data = res.json()

    assert data["entity_type"] == "finding"
    assert data["entity_id"] == finding_id
    detail = data["finding_detail"]
    assert detail is not None
    assert detail["finding"]["finding_id"] == finding_id
    assert detail["finding"]["title"] is not None
    assert detail["finding"]["quantified_risk_usd"] >= 0

    # Linked asset context
    assert detail["asset"] is not None
    assert detail["asset"]["asset_id"] == finding.asset_id

    # Candidate remediations
    assert isinstance(detail["remediations"], list)
    assert len(detail["remediations"]) > 0
    for r in detail["remediations"]:
        assert r["finding_id"] == finding_id
        assert r["cost"] >= 0

    # Linked audit logs array
    assert isinstance(detail["linked_audit_logs"], list)


def test_slideover_risk_calc_detail_rendering(client):
    """
    Asserts POST /drilldown/risk-calc returns scenario inputs,
    FAIR outputs, Monte Carlo confidence bands, and provenance.
    """
    payload = {
        "threat_event_frequency": 14.0,
        "vulnerability_score": 0.40,
        "primary_loss_usd": 200000.0,
        "secondary_loss_usd": 300000.0,
        "control_effectiveness_pct": 70.0,
        "risk_appetite_threshold_usd": 500000.0
    }

    res = client.post("/drilldown/risk-calc", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert data["entity_type"] == "risk_calc"
    assert data["risk_calc_detail"] is not None
    detail = data["risk_calc_detail"]

    # Scenario Inputs
    assert detail["scenario_inputs"]["threat_event_frequency"] == 14.0
    assert detail["scenario_inputs"]["primary_loss_usd"] == 200000.0

    # FAIR Outputs
    assert detail["fair_outputs"]["inherent_risk_usd"] > 0
    assert detail["fair_outputs"]["residual_risk_usd"] > 0
    assert detail["fair_outputs"]["loss_event_frequency"] > 0

    # Confidence Bands (Beta-PERT Monte Carlo percentiles)
    bands = detail["confidence_bands"]
    assert "inherent_ale_band" in bands
    inherent_band = bands["inherent_ale_band"]
    assert inherent_band["min"] <= inherent_band["p10"] <= inherent_band["p50"] <= inherent_band["p90"] <= inherent_band["max"]

    # Provenance
    assert "OpenFAIR" in detail["provenance"]["model_used"]
    assert detail["provenance"]["seed"] == 42


def test_slideover_conditional_audit_logging_sensitive(client):
    """
    Asserts that accessing a CRITICAL finding triggers an immutable audit log entry.
    """
    db = SessionLocal()
    critical_finding = db.query(FindingModel).filter(FindingModel.severity == "critical").first()
    if not critical_finding:
        # Create a critical finding for testing
        critical_finding = FindingModel(
            finding_id="FND-CRIT-TEST-001",
            asset_id="AST-001",
            title="Remote Code Execution Zero-Day",
            severity="critical",
            cvss_score=9.8,
            exploit_likelihood=0.85,
            exposure_factor=0.90,
            quantified_risk_usd=950000.0,
            status="open"
        )
        db.add(critical_finding)
        db.commit()

    crit_id = critical_finding.finding_id
    initial_audit_count = db.query(AuditLogModel).filter(
        AuditLogModel.action_type == "DRILLDOWN_ACCESS"
    ).count()
    db.close()

    res = client.get(f"/drilldown/finding/{crit_id}")
    assert res.status_code == 200
    data = res.json()

    assert data["audit_logged"] is True
    assert data["audit_log_id"] is not None

    db = SessionLocal()
    new_audit_count = db.query(AuditLogModel).filter(
        AuditLogModel.action_type == "DRILLDOWN_ACCESS"
    ).count()
    assert new_audit_count == initial_audit_count + 1

    entry = db.get(AuditLogModel, data["audit_log_id"])
    assert entry is not None
    assert "DRILLDOWN_ACCESS" == entry.action_type
    assert crit_id in entry.details
    db.close()


def test_slideover_conditional_audit_logging_routine(client):
    """
    Asserts that accessing a ROUTINE (medium/low) finding does NOT flood the audit trail.
    """
    db = SessionLocal()
    # Find a non-critical finding on a non-critical asset
    routine_finding = (
        db.query(FindingModel)
        .join(AssetModel, FindingModel.asset_id == AssetModel.asset_id)
        .filter(FindingModel.severity.in_(["medium", "low"]))
        .filter(AssetModel.criticality != "critical")
        .first()
    )
    if not routine_finding:
        routine_finding = db.query(FindingModel).filter(FindingModel.severity == "medium").first()

    assert routine_finding is not None
    routine_id = routine_finding.finding_id

    initial_audit_count = db.query(AuditLogModel).filter(
        AuditLogModel.action_type == "DRILLDOWN_ACCESS"
    ).count()
    db.close()

    res = client.get(f"/drilldown/finding/{routine_id}")
    assert res.status_code == 200
    data = res.json()

    # Must NOT log routine access
    assert data["audit_logged"] is False
    assert data["audit_log_id"] is None

    db = SessionLocal()
    new_audit_count = db.query(AuditLogModel).filter(
        AuditLogModel.action_type == "DRILLDOWN_ACCESS"
    ).count()
    assert new_audit_count == initial_audit_count, "Routine view must not increase audit log count"
    db.close()


def test_slideover_read_only_enforcement(client):
    """
    Asserts that opening drill-down views repeatedly strictly performs
    read-only operations with ZERO mutation to assets, findings, or remediations.
    """
    db = SessionLocal()
    f_before = [(f.finding_id, f.status, f.quantified_risk_usd) for f in db.query(FindingModel).all()]
    a_before = [(a.asset_id, a.asset_value) for a in db.query(AssetModel).all()]
    r_before = [(r.action_id, r.is_selected) for r in db.query(RemediationModel).all()]
    db.close()

    # Open multiple finding drilldowns
    for fid, _, _ in f_before[:5]:
        res = client.get(f"/drilldown/finding/{fid}")
        assert res.status_code == 200

    # Open generic endpoint
    client.get(f"/drilldown/finding/{f_before[0][0]}")

    db = SessionLocal()
    f_after = [(f.finding_id, f.status, f.quantified_risk_usd) for f in db.query(FindingModel).all()]
    a_after = [(a.asset_id, a.asset_value) for a in db.query(AssetModel).all()]
    r_after = [(r.action_id, r.is_selected) for r in db.query(RemediationModel).all()]
    db.close()

    assert f_before == f_after, "Findings mutated during drill-down view!"
    assert a_before == a_after, "Assets mutated during drill-down view!"
    assert r_before == r_after, "Remediations mutated during drill-down view!"


def test_slideover_error_handling(client):
    """
    Asserts 404 for non-existent finding ID and 400 for unsupported entity type.
    """
    res_404 = client.get("/drilldown/finding/NON-EXISTENT-ID-999")
    assert res_404.status_code == 404

    res_400 = client.get("/drilldown/invalid_type/ID-123")
    assert res_400.status_code == 400
