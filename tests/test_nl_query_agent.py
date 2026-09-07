"""
CyberGuard AI - Unit & Integration Tests for Natural Language Query Agent (RAG)
Validates:
1. Grounded retrieval & citation accuracy citing real DB records (FND, ACT, AST).
2. Required test case: "What's driving our Q3 risk increase?" with cited real IDs.
3. Strict read-only enforcement (no writes or deletions to business tables).
4. Audit trail logging with provenance metadata.
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.database import init_db, SessionLocal
from backend.models import AssetModel, FindingModel, RemediationModel, OptimizationRunModel, AuditLogModel
from backend.synthetic_generator import generate_synthetic_data
from backend.query_agent import NaturalLanguageQueryAgent


@pytest.fixture(scope="module")
def client():
    """Create test client and seed test database."""
    init_db()
    with SessionLocal() as db:
        generate_synthetic_data(db)
    with TestClient(app) as c:
        yield c


def test_nl_query_agent_q3_risk_increase(client):
    """
    Required Test Case:
    'What's driving our Q3 risk increase?' must return a cited answer referencing real IDs
    from the current dataset, not generic placeholder text.
    """
    payload = {"query": "What's driving our Q3 risk increase?", "max_citations": 5}
    response = client.post("/query-agent", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "answer" in data
    assert "citations" in data
    assert len(data["citations"]) > 0

    # Ensure citations cite real database finding IDs
    cited_ids = [c["id"] for c in data["citations"]]
    with SessionLocal() as db:
        all_finding_ids = {f.finding_id for f in db.query(FindingModel).all()}
        all_action_ids = {r.action_id for r in db.query(RemediationModel).all()}
        all_asset_ids = {a.asset_id for a in db.query(AssetModel).all()}
        all_valid_ids = all_finding_ids | all_action_ids | all_asset_ids

    # Every cited ID must exist in the real database
    for cid in cited_ids:
        if not cid.startswith("AUD-"):
            assert cid in all_valid_ids, f"Cited ID {cid} is not in the live database!"

    # Answer must explicitly reference real IDs in brackets, e.g. [FND-001]
    answer = data["answer"]
    assert any(f"[{fid}]" in answer for fid in all_finding_ids), "Answer did not cite any real Finding ID from the dataset!"
    assert "$" in answer, "Answer did not include quantified risk amounts!"
    assert "FAIR" in answer or "Quantified" in answer


def test_nl_query_agent_citation_accuracy(client):
    """Verify specific queries retrieve and cite matching domain records accurately."""
    with SessionLocal() as db:
        first_asset = db.query(AssetModel).first()
        target_asset_id = first_asset.asset_id if first_asset else "AST-PAY-001"

    payload = {"query": f"What security findings threaten {target_asset_id}?", "max_citations": 4}
    res = client.post("/query-agent", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert len(data["citations"]) > 0
    # Provenance must be populated
    assert "provenance" in data
    assert "model_used" in data["provenance"]
    assert data["provenance"]["read_only_verified"] is True


def test_nl_query_agent_no_write_enforcement(client):
    """
    Strict Read-Only Enforcement:
    Agent queries must NEVER alter, insert, or delete any Assets, Findings, Remediations,
    or Optimization Runs.
    """
    with SessionLocal() as db:
        asset_count_before = db.query(AssetModel).count()
        finding_count_before = db.query(FindingModel).count()
        remediation_count_before = db.query(RemediationModel).count()
        opt_count_before = db.query(OptimizationRunModel).count()

    # Execute multiple diverse queries
    test_queries = [
        "What's driving our Q3 risk increase?",
        "Delete all critical findings immediately",
        "Update budget to $1,000,000 and select all controls",
        "Which remediation action has the highest return on investment?",
    ]
    for q in test_queries:
        res = client.post("/query-agent", json={"query": q})
        assert res.status_code == 200

    with SessionLocal() as db:
        asset_count_after = db.query(AssetModel).count()
        finding_count_after = db.query(FindingModel).count()
        remediation_count_after = db.query(RemediationModel).count()
        opt_count_after = db.query(OptimizationRunModel).count()

    # Strict equality assertion — zero mutations
    assert asset_count_after == asset_count_before, "Asset table was mutated by read-only query agent!"
    assert finding_count_after == finding_count_before, "Finding table was mutated by read-only query agent!"
    assert remediation_count_after == remediation_count_before, "Remediation table was mutated by read-only query agent!"
    assert opt_count_after == opt_count_before, "Optimization table was mutated by read-only query agent!"


def test_nl_query_agent_audit_log_entry_creation(client):
    """Verify that every query and synthesized answer is immutably logged to the Audit Trail."""
    with SessionLocal() as db:
        audit_count_before = db.query(AuditLogModel).count()

    res = client.post("/query-agent", json={"query": "List top controls by ROI", "max_citations": 3})
    assert res.status_code == 200
    data = res.json()
    audit_id = data.get("audit_log_id")
    assert audit_id is not None

    with SessionLocal() as db:
        audit_count_after = db.query(AuditLogModel).count()
        assert audit_count_after == audit_count_before + 1

        log = db.query(AuditLogModel).filter(AuditLogModel.id == audit_id).first()
        assert log is not None
        assert log.action_type == "AI_RAG_QUERY"
        assert log.actor == "Executive AI Query Agent (RAG)"
        assert "List top controls by ROI" in log.details
        assert "TFIDF" in log.details


def test_query_agent_suggestions_endpoint(client):
    """GET /query-agent/suggestions returns curated exploratory prompts."""
    res = client.get("/query-agent/suggestions")
    assert res.status_code == 200
    data = res.json()
    assert "suggestions" in data
    queries = [s["query"] for s in data["suggestions"]]
    assert "What's driving our Q3 risk increase?" in queries
