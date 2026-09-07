"""
CyberGuard AI - FastAPI Main Application Entrypoint
Provides REST APIs for:
- Continuous Risk Quantification (GET /risk-summary)
- Investment Optimization (POST /optimize)
- Security Findings & Asset Ingestion (POST /ingest, GET /findings, GET /assets)
- Security Framework Mappings (GET /frameworks)
- Executive Decision Support (GET /executive-summary)
- GRC Analyst Tools: Risk Calculator, Compliance Gap, Audit Trail, Policy Mapping, Scenarios
- Serves interactive Executive Dashboard at /
"""

import os
from datetime import datetime, timezone
from contextlib import asynccontextmanager


def _get_utc_now():
    """Return timezone-naive UTC datetime for SQLite compatibility without deprecation warning."""
    return datetime.now(timezone.utc).replace(tzinfo=None)
from typing import List, Optional
from fastapi import FastAPI, Depends, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from sqlalchemy import func

from backend.config import settings
from backend.database import init_db, get_db, SessionLocal
from backend.models import AssetModel, FindingModel, RemediationModel, OptimizationRunModel, AuditLogModel
from backend.schemas import (
    RiskSummaryResponse,
    OptimizationRequest,
    OptimizationResult,
    FindingResponse,
    AssetResponse,
    RemediationResponse,
    IngestRequest,
    IngestResponse,
    RiskCalculatorRequest,
    RiskCalculatorResponse,
    AuditLogCreate,
    AuditLogResponse,
    ComplianceGapResponse,
    PolicyMappingRequest,
    PolicyMappingResponse,
    ScenarioAnalysisRequest,
    ScenarioAnalysisResponse,
    MitreEvalRequest,
    MitreEvalResponse,
    QueryAgentRequest,
    QueryAgentResponse,
    DrilldownResponse,
    DrilldownFindingDetail,
    DrilldownRiskCalcDetail,
)
from backend.risk_engine import (
    aggregate_risk_by_business_unit,
    aggregate_risk_by_asset_category,
    calculate_fair_monte_carlo
)
from backend.optimizer import optimize_investments, InvestmentOptimizer
from backend.synthetic_generator import generate_synthetic_data
from backend.executive_summary import generate_executive_summary
from backend.query_agent import NaturalLanguageQueryAgent
from backend.grc_engine import (
    calculate_fair_risk,
    analyze_compliance_gaps,
    map_policy_to_controls,
    run_scenario_comparison,
    evaluate_mitre_threat_response,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database and seed synthetic dataset if empty on startup."""
    init_db()
    with SessionLocal() as db:
        asset_count = db.query(AssetModel).count()
        if asset_count == 0:
            generate_synthetic_data(db)
    yield


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="AI-Powered Continuous Cyber Risk Quantification & Investment Optimization Platform",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static folder if exists
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", include_in_schema=False)
async def serve_dashboard():
    """Serve the Executive & Security Operator Dashboard."""
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse(
        content={
            "message": "CyberGuard AI API is operational. Visit /docs for Swagger specifications.",
            "endpoints": ["/risk-summary", "/optimize", "/findings", "/assets", "/frameworks"]
        }
    )


@app.get("/health", tags=["System"])
def health_check(db: Session = Depends(get_db)):
    """System health check endpoint."""
    try:
        asset_count = db.query(AssetModel).count()
        return {
            "status": "healthy",
            "version": settings.VERSION,
            "database": "connected",
            "assets_registered": asset_count
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database unreachable: {str(e)}"
        )


@app.get("/risk-summary", response_model=RiskSummaryResponse, tags=["Risk Quantification"])
def get_risk_summary(db: Session = Depends(get_db)):
    """
    GET /risk-summary
    Returns total quantified dollar-risk exposure, breakdown by Business Unit and Asset Category,
    and top prioritized risks.
    """
    findings = db.query(FindingModel).filter(FindingModel.status == "open").all()
    assets = db.query(AssetModel).all()
    assets_by_id = {a.asset_id: a for a in assets}

    total_risk = sum(f.quantified_risk_usd for f in findings)
    risk_by_bu = aggregate_risk_by_business_unit(findings, assets_by_id)
    risk_by_cat = aggregate_risk_by_asset_category(findings, assets_by_id)

    # Top 10 risks by quantified dollar loss
    top_findings = sorted(findings, key=lambda x: x.quantified_risk_usd, reverse=True)[:10]
    top_risks_dto = [FindingResponse.model_validate(f) for f in top_findings]

    avg_risk = round(total_risk / len(assets), 2) if assets else 0.0
    formatted_total = f"${total_risk:,.0f}" if total_risk >= 1000 else f"${total_risk:.2f}"

    return RiskSummaryResponse(
        total_quantified_risk=round(total_risk, 2),
        formatted_total_risk=formatted_total,
        total_assets=len(assets),
        total_findings=len(findings),
        risk_by_business_unit=risk_by_bu,
        risk_by_asset_category=risk_by_cat,
        top_risks=top_risks_dto,
        average_asset_risk=avg_risk
    )


@app.post("/optimize", response_model=OptimizationResult, tags=["Investment Optimization"])
def optimize_security_investments(
    request: Optional[OptimizationRequest] = None,
    db: Session = Depends(get_db)
):
    """
    POST /optimize
    Accepts a security budget and returns recommended actions maximizing risk reduction.
    Solves 0/1 knapsack using PuLP or Greedy ROI heuristic.
    Gracefully handles empty or string budget payloads.
    """
    if request is None:
        request = OptimizationRequest()

    remediations = db.query(RemediationModel).all()
    if not remediations:
        # If database is empty, seed it automatically
        generate_synthetic_data(db)
        remediations = db.query(RemediationModel).all()

    strategy_str = request.strategy.value if hasattr(request.strategy, "value") else str(request.strategy)

    result = optimize_investments(
        actions=remediations,
        budget=request.budget,
        strategy=strategy_str,
        min_roi_threshold=request.min_roi_threshold
    )

    # Add sensitivity analysis
    result.sensitivity_analysis = InvestmentOptimizer.sensitivity_analysis(
        actions=remediations,
        base_budget=request.budget
    )

    # Constraint: Log AI optimization & ranking decision to Audit Trail with provenance tag
    try:
        prov = result.provenance or {}
        model_name = prov.get("model_used", "PuLP-CBC-2.10 (0/1 Knapsack)")
        input_sum = prov.get("input_summary", f"Budget: ${request.budget:,.2f}")
        audit_entry = AuditLogModel(
            timestamp=_get_utc_now(),
            action_type="BUDGET_ALLOCATION",
            actor=f"AI Investment Optimizer ({model_name})",
            details=f"Optimized budget ${request.budget:,.0f} allocating ${result.total_cost_allocated:,.0f} across {len(result.selected_actions)} controls. ΔALE reduction: ${result.total_risk_reduction:,.0f} ({result.overall_roi:.2f}x ROI). Provenance: model='{model_name}', input='{input_sum}'",
            framework_ref="NIST CSF 2.0 / ISO 27001",
            impact_usd=result.total_risk_reduction,
        )
        db.add(audit_entry)
        db.commit()
    except Exception as e:
        db.rollback()

    return result


@app.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
def ingest_data(request: Optional[IngestRequest] = None, db: Session = Depends(get_db)):
    """
    POST /ingest
    Trigger data ingestion pipeline or reseed synthetic 25-asset dataset.
    Accepts empty body, null, or populated payload without 422 error.
    """
    if request is None:
        request = IngestRequest(reseed_synthetic=True, days_back=30)

    if request.reseed_synthetic:
        a_count, f_count, r_count = generate_synthetic_data(db)
        return IngestResponse(
            success=True,
            assets_ingested=a_count,
            findings_ingested=f_count,
            remediations_ingested=r_count,
            message=f"Successfully seeded {a_count} assets, {f_count} findings, and {r_count} remediation options."
        )
    return IngestResponse(
        success=True,
        assets_ingested=0,
        findings_ingested=0,
        remediations_ingested=0,
        message="No external feed specified. Reseed synthetic data by setting reseed_synthetic=true."
    )


@app.get("/findings", response_model=List[FindingResponse], tags=["Findings"])
def list_findings(
    severity: Optional[str] = None,
    asset_id: Optional[str] = None,
    status_filter: Optional[str] = Query(default=None, alias="status"),
    search: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """List normalized security findings with flexible filters (tolerates 'all' or empty strings)."""
    query = db.query(FindingModel)

    # Filter status if valid (ignore 'all', 'ALL', '')
    if status_filter and status_filter.strip().lower() not in ("all", ""):
        query = query.filter(FindingModel.status == status_filter.strip().lower())

    # Filter severity if valid (ignore 'all', 'ALL', '')
    if severity and severity.strip().lower() not in ("all", ""):
        query = query.filter(FindingModel.severity == severity.strip().lower())

    # Filter asset_id if valid
    if asset_id and asset_id.strip().lower() not in ("all", ""):
        query = query.filter(FindingModel.asset_id == asset_id.strip())

    # Search in title or cve_id
    if search and search.strip():
        term = f"%{search.strip()}%"
        query = query.filter(
            (FindingModel.title.ilike(term)) | (FindingModel.cve_id.ilike(term)) | (FindingModel.finding_id.ilike(term))
        )

    findings = query.all()
    return [FindingResponse.model_validate(f) for f in findings]


@app.get("/assets", response_model=List[AssetResponse], tags=["Assets"])
def list_assets(db: Session = Depends(get_db)):
    """List registered assets."""
    assets = db.query(AssetModel).all()
    return [AssetResponse.model_validate(a) for a in assets]


@app.get("/remediations", response_model=List[RemediationResponse], tags=["Remediations"])
def list_remediations(db: Session = Depends(get_db)):
    """List all available remediation actions."""
    remediations = db.query(RemediationModel).order_by(RemediationModel.roi.desc()).all()
    return [RemediationResponse.model_validate(r) for r in remediations]


@app.get("/frameworks", tags=["Compliance & Frameworks"])
def get_framework_mappings(db: Session = Depends(get_db)):
    """
    Returns framework alignment breakdown:
    - NIST CSF 2.0 (IDENTIFY, PROTECT, DETECT, RESPOND, RECOVER)
    - MITRE ATT&CK techniques
    - FAIR methodology grounding
    - CIS / NIST 800-53 controls
    """
    nist_counts = db.query(
        FindingModel.nist_csf_category, func.count(FindingModel.id)
    ).group_by(FindingModel.nist_csf_category).all()

    mitre_counts = db.query(
        FindingModel.mitre_technique_id, func.count(FindingModel.id)
    ).filter(FindingModel.mitre_technique_id != None).group_by(FindingModel.mitre_technique_id).all()

    return {
        "nist_csf_2_0": {
            cat: count for cat, count in nist_counts
        },
        "mitre_attack": {
            tech: count for tech, count in mitre_counts
        },
        "fair_model": {
            "loss_event_frequency_formula": "LEF = Threat Contact Frequency × Vulnerability Probability",
            "loss_magnitude_formula": "LM = Primary Loss (Asset Replacement + Direct) + Secondary Loss (Downtime + Fines)",
            "primary_framework_status": "Active"
        }
    }


def _build_executive_summary(budget_val: Optional[float], db: Session):
    safe_budget = 50000.0
    if budget_val is not None:
        try:
            safe_budget = max(1000.0, float(budget_val))
        except (ValueError, TypeError):
            safe_budget = 50000.0

    summary_dto = get_risk_summary(db)
    remediations = db.query(RemediationModel).all()
    opt_result = optimize_investments(remediations, safe_budget, strategy="pulp")

    text = generate_executive_summary(summary_dto, opt_result)
    return {
        "executive_summary": text,
        "total_risk_exposure": summary_dto.formatted_total_risk,
        "allocated_budget": f"${safe_budget:,.0f}",
        "projected_risk_reduction": f"${opt_result.total_risk_reduction:,.0f}",
        "roi": f"{opt_result.overall_roi:.1f}x"
    }


@app.get("/executive-summary", tags=["Executive Decision Support"])
def get_executive_summary_report(
    budget: Optional[float] = Query(default=50000.0),
    db: Session = Depends(get_db)
):
    """Generate a board-ready plain-English executive summary based on live metrics (GET)."""
    return _build_executive_summary(budget, db)


@app.post("/executive-summary", tags=["Executive Decision Support"])
def post_executive_summary_report(
    request: Optional[OptimizationRequest] = None,
    db: Session = Depends(get_db)
):
    """Generate a board-ready plain-English executive summary based on live metrics (POST)."""
    b = request.budget if request else 50000.0
    return _build_executive_summary(b, db)


# ==========================================
# GRC ANALYST TOOL ENDPOINTS
# ==========================================

@app.post("/risk-calculator", response_model=RiskCalculatorResponse, tags=["GRC Tools"])
def interactive_risk_calculator(request: RiskCalculatorRequest):
    """
    POST /risk-calculator
    Interactive FAIR risk calculator for GRC analysts.
    Computes inherent risk, applies control effectiveness, and returns residual risk
    with an actionable recommendation based on risk appetite comparison.
    """
    result = calculate_fair_risk(
        threat_event_frequency=request.threat_event_frequency,
        vulnerability_score=request.vulnerability_score,
        primary_loss_usd=request.primary_loss_usd,
        secondary_loss_usd=request.secondary_loss_usd,
        control_effectiveness_pct=request.control_effectiveness_pct,
        risk_appetite_threshold_usd=request.risk_appetite_threshold_usd,
    )
    return RiskCalculatorResponse(**result)


@app.get("/compliance-gap", response_model=ComplianceGapResponse, tags=["GRC Tools"])
def get_compliance_gap_analysis(db: Session = Depends(get_db)):
    """
    GET /compliance-gap
    Compliance gap analysis against NIST CSF 2.0 and ISO 27001:2022.
    Scans all open findings, maps them to framework functions/controls,
    computes coverage percentages, and generates a prioritized remediation roadmap.
    """
    findings = db.query(FindingModel).filter(FindingModel.status == "open").all()
    assets = db.query(AssetModel).all()

    result = analyze_compliance_gaps(findings, assets)
    return ComplianceGapResponse(**result)


@app.get("/audit-trail", response_model=List[AuditLogResponse], tags=["GRC Tools"])
def get_audit_trail(
    action_type: Optional[str] = Query(default=None, description="Filter by action type"),
    limit: int = Query(default=100, ge=1, le=500),
    db: Session = Depends(get_db)
):
    """
    GET /audit-trail
    Retrieve GRC decision audit trail with optional filtering.
    Returns timestamped entries with actor, justification, framework reference, and financial impact.
    """
    query = db.query(AuditLogModel).order_by(AuditLogModel.timestamp.desc())

    if action_type and action_type.strip().lower() not in ("all", ""):
        query = query.filter(AuditLogModel.action_type == action_type.strip().upper())

    logs = query.limit(limit).all()
    return [AuditLogResponse.model_validate(log) for log in logs]


@app.post("/audit-trail", response_model=AuditLogResponse, tags=["GRC Tools"])
def create_audit_entry(request: AuditLogCreate, db: Session = Depends(get_db)):
    """
    POST /audit-trail
    Record a GRC decision in the audit trail.
    Supports action types: RISK_ACCEPTED, BUDGET_ALLOCATED, CONTROL_EXCEPTION,
    POLICY_APPROVED, FINDING_ESCALATED, VENDOR_RISK_REVIEWED, RISK_ASSESSMENT.
    """
    log = AuditLogModel(
        timestamp=_get_utc_now(),
        action_type=request.action_type.strip().upper(),
        actor=request.actor.strip(),
        details=request.details.strip(),
        framework_ref=request.framework_ref.strip(),
        impact_usd=request.impact_usd,
    )
    db.add(log)
    db.commit()
    db.refresh(log)
    return AuditLogResponse.model_validate(log)


@app.post("/policy-mapping", response_model=PolicyMappingResponse, tags=["GRC Tools"])
def map_policy_controls(request: PolicyMappingRequest):
    """
    POST /policy-mapping
    Automated policy-to-control mapping engine.
    Accepts organizational policy text and maps it to NIST CSF 2.0 subcategories
    and ISO 27001:2022 Annex A controls using keyword-intelligence matching.
    Returns relevance scores, coverage assessment, and identified gaps.
    """
    result = map_policy_to_controls(request.policy_text)
    return PolicyMappingResponse(**result)


@app.post("/scenario-analysis", response_model=ScenarioAnalysisResponse, tags=["GRC Tools"])
def compare_budget_scenarios(
    request: ScenarioAnalysisRequest,
    db: Session = Depends(get_db)
):
    """
    POST /scenario-analysis
    What-if budget scenario comparator.
    Accepts 2-10 budget scenarios, runs PuLP optimization on each,
    and returns side-by-side comparison with marginal ROI analysis and recommendation.
    """
    remediations = db.query(RemediationModel).all()
    if not remediations:
        generate_synthetic_data(db)
        remediations = db.query(RemediationModel).all()

    findings = db.query(FindingModel).filter(FindingModel.status == "open").all()
    base_risk = sum(f.quantified_risk_usd for f in findings) if findings else 4579685.84
    base_lef = sum(f.exploit_likelihood for f in findings) if findings else 24.5

    scenarios_input = [{"name": s.name, "budget": s.budget} for s in request.scenarios]
    result = run_scenario_comparison(scenarios_input, remediations, base_risk=base_risk, base_lef=base_lef)
    return ScenarioAnalysisResponse(**result)


@app.post("/mitre-evaluation", response_model=MitreEvalResponse, tags=["GRC Tools", "AI Response"])
def evaluate_mitre_threat(
    request: MitreEvalRequest,
    db: Session = Depends(get_db)
):
    """
    POST /mitre-evaluation
    Evaluates an AI-detected MITRE TTP against the GRC Council mandates (FAIR & COSO ERM).
    Returns an automated response action and logs the decision immutably to the audit trail.
    """
    # 1. Evaluate logic using Council Safety Gate
    eval_result = evaluate_mitre_threat_response(
        threat_event=request.threat_event.model_dump(),
        asset_context=request.asset_context.model_dump(),
        coso_appetite_usd=request.coso_appetite_usd
    )
    
    # 2. Immutable Audit Trail (GRC Requirement)
    action_type_mapping = {
        "ISOLATE_ASSET": "CONTROL_EXCEPTION", # Extreme action logged as exception
        "ENRICH_AND_MONITOR": "RISK_ACCEPTED", # Risk within tolerance
        "ESCALATE_TO_HUMAN": "FINDING_ESCALATED" # Sent to human
    }
    
    audit_log = AuditLogModel(
        timestamp=_get_utc_now(),
        action_type=action_type_mapping.get(eval_result["action_taken"], "RISK_ASSESSMENT"),
        actor="AI Response Handler (Council Safety Gate)",
        details=f"MITRE TTP {request.threat_event.mitre_ttp} detected on Asset {request.asset_context.id}. "
                f"Action: {eval_result['action_taken']}. "
                f"Reason: {eval_result['reason']}",
        framework_ref=f"FAIR / COSO / MITRE {request.threat_event.mitre_ttp}",
        impact_usd=eval_result["fair_risk_usd"]
    )
    db.add(audit_log)
    db.commit()
    db.refresh(audit_log)

    return MitreEvalResponse(
        action_taken=eval_result["action_taken"],
        reason=eval_result["reason"],
        fair_risk_usd=eval_result["fair_risk_usd"],
        audit_id=audit_log.id
    )


@app.post("/query-agent", response_model=QueryAgentResponse, tags=["AI Query Agent"])
def query_agent_endpoint(
    request: QueryAgentRequest,
    db: Session = Depends(get_db)
):
    """
    POST /query-agent
    Read-only Natural Language Query Agent (RAG) over the continuous risk database.
    Retrieves grounded evidence from findings, remediations, assets, and audit logs.
    Strictly read-only: does not modify or commit any changes to risk records.
    Every query and response is logged immutably to the Audit Trail with model provenance.
    """
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # Ensure database has seeded data
    if db.query(FindingModel).count() == 0:
        generate_synthetic_data(db)

    result = NaturalLanguageQueryAgent.query(
        query_text=request.query,
        db=db,
        max_citations=request.max_citations or 5
    )
    return QueryAgentResponse(**result)


@app.get("/query-agent/suggestions", tags=["AI Query Agent"])
def get_query_agent_suggestions():
    """
    GET /query-agent/suggestions
    Returns curated executive queries to explore risk drivers, controls, and audit records.
    """
    return {
        "suggestions": [
            {"query": "What's driving our Q3 risk increase?", "category": "Risk Drivers"},
            {"query": "Which controls provide the highest ROI for our budget?", "category": "Optimization"},
            {"query": "What critical vulnerabilities affect FinTech / Payments?", "category": "Asset Risk"},
            {"query": "What recent actions were recorded in the audit trail?", "category": "Audit & Compliance"},
        ]
    }


# ==========================================
# PHASE 4: REUSABLE SLIDE-OVER DRILLDOWN API
# ==========================================

@app.get("/drilldown/finding/{finding_id}", response_model=DrilldownResponse, tags=["Drilldown"])
def get_finding_drilldown(finding_id: str, db: Session = Depends(get_db)):
    """
    GET /drilldown/finding/{finding_id}
    Retrieves complete finding details, linked asset context, candidate remediations,
    MITRE technique mapping, and linked audit logs.
    Logs read access to the Audit Trail if and only if the record is sensitive (Critical).
    """
    finding = db.query(FindingModel).filter(FindingModel.finding_id == finding_id).first()
    if not finding:
        raise HTTPException(status_code=404, detail=f"Finding '{finding_id}' not found.")

    asset = db.query(AssetModel).filter(AssetModel.asset_id == finding.asset_id).first()
    remediations = db.query(RemediationModel).filter(RemediationModel.finding_id == finding_id).all()

    # Search linked audit logs matching finding_id, asset_id, or cve_id
    search_terms = [finding_id, finding.asset_id]
    if finding.cve_id:
        search_terms.append(finding.cve_id)

    from sqlalchemy import or_
    audit_logs = db.query(AuditLogModel).filter(
        or_(*[AuditLogModel.details.ilike(f"%{t}%") for t in search_terms])
    ).order_by(AuditLogModel.timestamp.desc()).limit(10).all()

    # Conditional Audit Logging:
    # Log panel access to the existing Audit Trail only if the accessed record is sensitive/restricted (Critical)
    is_sensitive = (finding.severity.lower() == "critical") or (asset is not None and asset.criticality.lower() == "critical")
    audit_logged = False
    audit_id = None

    if is_sensitive:
        audit_entry = AuditLogModel(
            timestamp=_get_utc_now(),
            action_type="DRILLDOWN_ACCESS",
            actor="GRC Analyst (Drill-Down Viewer)",
            details=(
                f"Accessed restricted record detail for {finding.finding_id} "
                f"({finding.cve_id or finding.title}) on asset {finding.asset_id}. "
                f"Severity: {finding.severity.upper()}."
            ),
            framework_ref="NIST CSF 2.0 / ISO 27001",
            impact_usd=finding.quantified_risk_usd,
        )
        db.add(audit_entry)
        db.commit()
        db.refresh(audit_entry)
        audit_logged = True
        audit_id = audit_entry.id

    mitre_data = None
    if finding.mitre_technique_id:
        mitre_data = {
            "technique_id": finding.mitre_technique_id,
            "technique_name": finding.title,
            "tactic": "Initial Access / Execution",
            "framework_ref": f"MITRE ATT&CK {finding.mitre_technique_id}",
            "nist_alignment": finding.nist_csf_category
        }

    finding_resp = FindingResponse.model_validate(finding)
    asset_resp = AssetResponse.model_validate(asset) if asset else None
    remed_resps = [RemediationResponse.model_validate(r) for r in remediations]
    audit_resps = [AuditLogResponse.model_validate(a) for a in audit_logs]

    return DrilldownResponse(
        entity_type="finding",
        entity_id=finding_id,
        finding_detail=DrilldownFindingDetail(
            finding=finding_resp,
            asset=asset_resp,
            remediations=remed_resps,
            mitre_mapping=mitre_data,
            linked_audit_logs=audit_resps,
            is_sensitive=is_sensitive
        ),
        audit_logged=audit_logged,
        audit_log_id=audit_id
    )


@app.post("/drilldown/risk-calc", response_model=DrilldownResponse, tags=["Drilldown"])
def post_risk_calc_drilldown(request: RiskCalculatorRequest, db: Session = Depends(get_db)):
    """
    POST /drilldown/risk-calc
    Retrieves full scenario inputs, confidence band percentiles, and calculation provenance
    for the Risk Calculator output. Logs only when scenario exceeds high risk appetite.
    """
    result_dict = calculate_fair_risk(
        threat_event_frequency=request.threat_event_frequency,
        vulnerability_score=request.vulnerability_score,
        primary_loss_usd=request.primary_loss_usd,
        secondary_loss_usd=request.secondary_loss_usd,
        control_effectiveness_pct=request.control_effectiveness_pct,
        risk_appetite_threshold_usd=request.risk_appetite_threshold_usd,
    )
    result = RiskCalculatorResponse(**result_dict)

    is_sensitive = (result.residual_risk_usd > 1_000_000.0) or result.exceeds_appetite
    audit_logged = False
    audit_id = None

    if is_sensitive:
        audit_entry = AuditLogModel(
            timestamp=_get_utc_now(),
            action_type="DRILLDOWN_ACCESS",
            actor="GRC Analyst (Risk Calculator)",
            details=(
                f"Accessed restricted FAIR risk calculation drill-down. Residual ALE: "
                f"${result.residual_risk_usd:,.2f} exceeds risk appetite (${request.risk_appetite_threshold_usd:,.2f})."
            ),
            framework_ref="FAIR / COSO ERM",
            impact_usd=result.residual_risk_usd,
        )
        db.add(audit_entry)
        db.commit()
        db.refresh(audit_entry)
        audit_logged = True
        audit_id = audit_entry.id

    calc_id = "FAIR-CALC-" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    detail = DrilldownRiskCalcDetail(
        calculation_id=calc_id,
        scenario_inputs={
            "threat_event_frequency": request.threat_event_frequency,
            "vulnerability_score": request.vulnerability_score,
            "primary_loss_usd": request.primary_loss_usd,
            "secondary_loss_usd": request.secondary_loss_usd,
            "control_effectiveness_pct": request.control_effectiveness_pct,
            "risk_appetite_threshold_usd": request.risk_appetite_threshold_usd
        },
        fair_outputs={
            "inherent_risk_usd": result.inherent_risk_usd,
            "residual_risk_usd": result.residual_risk_usd,
            "risk_reduction_usd": result.risk_reduction_usd,
            "loss_event_frequency": result.fair_breakdown.get("loss_event_frequency", request.threat_event_frequency * request.vulnerability_score),
            "exceeds_appetite": result.exceeds_appetite,
            "appetite_delta_usd": round(result.inherent_risk_usd - request.risk_appetite_threshold_usd, 2)
        },
        confidence_bands={
            "inherent_ale_band": result.inherent_ale_band or {},
            "residual_ale_band": result.residual_ale_band or {},
            "lef_band": result.lef_band or {},
            "reduction_band": result.reduction_band or {}
        },
        provenance={
            "model_used": "OpenFAIR-Beta-PERT-MonteCarlo (5,000 iterations)",
            "methodology": "Open FAIR ISO/IEC 27005 Calibration with Beta-PERT distribution",
            "seed": 42,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "confidence_interval": "80% CI (P10 - P90)"
        },
        is_sensitive=is_sensitive
    )

    return DrilldownResponse(
        entity_type="risk_calc",
        entity_id=calc_id,
        risk_calc_detail=detail,
        audit_logged=audit_logged,
        audit_log_id=audit_id
    )


@app.get("/drilldown/{entity_type}/{entity_id}", response_model=DrilldownResponse, tags=["Drilldown"])
def get_drilldown_generic(entity_type: str, entity_id: str, db: Session = Depends(get_db)):
    """
    GET /drilldown/{entity_type}/{entity_id}
    Unified generic drill-down endpoint accepting any entity type + ID.
    """
    clean_type = entity_type.strip().lower()
    if clean_type == "finding":
        return get_finding_drilldown(entity_id, db)
    elif clean_type in ("risk_calc", "risk-calc"):
        req = RiskCalculatorRequest(
            threat_event_frequency=12.0,
            vulnerability_score=0.35,
            primary_loss_usd=150000.0,
            secondary_loss_usd=250000.0,
            control_effectiveness_pct=65.0,
            risk_appetite_threshold_usd=500000.0
        )
        return post_risk_calc_drilldown(req, db)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported entity type '{entity_type}'. Supported: 'finding', 'risk_calc'."
        )



