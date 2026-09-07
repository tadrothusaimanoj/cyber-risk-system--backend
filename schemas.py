"""
CyberGuard AI - Canonical Pydantic Schemas
Defines request, response, and domain models for the platform.
Includes pre-validators to eliminate 422 Unprocessable Entity errors from:
- String budget inputs (e.g. '$50,000')
- Missing or empty request bodies
- Strategy casing mismatches
- Loose framework and severity categorization
"""

from datetime import datetime, timezone
from enum import Enum
from typing import List, Dict, Optional, Any, Union
import re
from pydantic import BaseModel, Field, ConfigDict, model_validator


def _utc_now():
    """Return timezone-naive UTC datetime for compatibility without deprecation warning."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


# Enums
class AssetCategory(str, Enum):
    SERVER = "server"
    DATABASE = "database"
    WORKSTATION = "workstation"
    CLOUD_SERVICE = "cloud_service"
    NETWORK_DEVICE = "network_device"


class ExposureLevel(str, Enum):
    INTERNET_FACING = "internet_facing"
    DMZ = "dmz"
    INTERNAL = "internal"
    ISOLATED = "isolated"


class Criticality(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class NistCsfCategory(str, Enum):
    IDENTIFY = "IDENTIFY"
    PROTECT = "PROTECT"
    DETECT = "DETECT"
    RESPOND = "RESPOND"
    RECOVER = "RECOVER"


class RemediationType(str, Enum):
    PATCH = "patch"
    CONFIGURATION = "configuration"
    COMPENSATING_CONTROL = "compensating_control"
    ACCEPT_RISK = "accept_risk"


class OptimizationStrategy(str, Enum):
    PULP = "pulp"
    GREEDY = "greedy"
    HYBRID = "hybrid"


# Asset Schemas
class AssetBase(BaseModel):
    asset_id: str = Field(..., description="Unique asset identifier, e.g., AST-SRV-001")
    asset_name: str = Field(..., description="Descriptive asset name")
    asset_category: Union[AssetCategory, str]
    business_unit: str = Field(..., description="Owning business unit")
    asset_value: float = Field(..., gt=0, description="Estimated financial replacement/downtime value in USD")
    exposure: Union[ExposureLevel, str] = ExposureLevel.INTERNAL
    criticality: Union[Criticality, str] = Criticality.MEDIUM
    owner: Optional[str] = None


class AssetCreate(AssetBase):
    pass


class AssetResponse(AssetBase):
    id: Optional[int] = 0
    created_at: Optional[datetime] = Field(default_factory=_utc_now)
    model_config = ConfigDict(from_attributes=True, extra="ignore")


# Finding Schemas
class FindingBase(BaseModel):
    finding_id: str = Field(..., description="Unique finding ID, e.g. FIND-2024-001")
    asset_id: str = Field(..., description="Target asset ID")
    finding_type: str = Field(default="vulnerability", description="vulnerability | misconfiguration | threat_intelligence")
    cve_id: Optional[str] = None
    title: str
    description: Optional[str] = None
    severity: Union[Severity, str] = Severity.MEDIUM
    cvss_score: float = Field(default=5.0, ge=0.0, le=10.0)
    epss_score: Optional[float] = Field(default=None)
    exploit_likelihood: float = Field(default=0.1, ge=0.0, le=1.0)
    exposure_factor: float = Field(default=0.5, ge=0.0, le=1.0)
    quantified_risk_usd: float = Field(default=0.0, ge=0.0)
    nist_csf_category: Union[NistCsfCategory, str] = NistCsfCategory.PROTECT
    mitre_technique_id: Optional[str] = None
    status: str = Field(default="open")

    @model_validator(mode="before")
    @classmethod
    def normalize_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Clean severity
            if "severity" in data and isinstance(data["severity"], str):
                data["severity"] = data["severity"].lower()
            # Clean cvss
            if "cvss_score" in data and data["cvss_score"] is not None:
                try:
                    data["cvss_score"] = max(0.0, min(10.0, float(data["cvss_score"])))
                except (ValueError, TypeError):
                    data["cvss_score"] = 5.0
        return data


class FindingCreate(FindingBase):
    pass


class FindingResponse(FindingBase):
    id: Optional[int] = 0
    created_at: Optional[datetime] = Field(default_factory=_utc_now)
    model_config = ConfigDict(from_attributes=True, extra="ignore")


# Remediation Schemas
class RemediationBase(BaseModel):
    action_id: str = Field(..., description="Unique action ID, e.g. REM-001")
    finding_id: str = Field(..., description="Associated finding ID")
    asset_id: str = Field(..., description="Target asset ID")
    title: str
    remediation_type: Union[RemediationType, str] = RemediationType.PATCH
    cost: float = Field(..., ge=0.0, description="Cost to implement in USD")
    risk_reduction_usd: float = Field(..., ge=0.0, description="Expected risk reduction in USD")
    roi: float = Field(default=0.0, ge=0.0, description="Risk reduction per dollar spent (risk_reduction / cost)")
    effort_hours: float = Field(default=8.0, ge=0.0)
    nist_control_id: Optional[str] = None
    is_selected: bool = False


class RemediationCreate(RemediationBase):
    pass


class RemediationResponse(RemediationBase):
    id: Optional[int] = 0
    created_at: Optional[datetime] = Field(default_factory=_utc_now)
    model_config = ConfigDict(from_attributes=True, extra="ignore")


# Optimization Schemas
class OptimizationRequest(BaseModel):
    budget: float = Field(default=50000.0, description="Total remediation budget in USD")
    strategy: Union[OptimizationStrategy, str] = Field(default=OptimizationStrategy.PULP, description="pulp or greedy")
    min_roi_threshold: Optional[float] = Field(default=None, ge=0.0, description="Minimum ROI cutoff")

    model_config = ConfigDict(extra="ignore")

    @model_validator(mode="before")
    @classmethod
    def sanitize_inputs(cls, values: Any) -> Any:
        if not isinstance(values, dict):
            return {"budget": 50000.0, "strategy": OptimizationStrategy.PULP}

        # 1. Sanitize budget (handle '$50,000', commas, empty, or non-numeric)
        raw_budget = values.get("budget", values.get("remediationBudget", values.get("amount", 50000.0)))
        if isinstance(raw_budget, str):
            clean_str = re.sub(r"[^\d.]", "", raw_budget)
            try:
                values["budget"] = float(clean_str) if clean_str else 50000.0
            except ValueError:
                values["budget"] = 50000.0
        elif isinstance(raw_budget, (int, float)):
            values["budget"] = float(raw_budget)
        else:
            values["budget"] = 50000.0

        # Ensure budget is strictly positive to prevent 422
        if values["budget"] <= 0:
            values["budget"] = 50000.0

        # 2. Normalize strategy (handle 'PULP', 'greedy_roi', etc.)
        strat = str(values.get("strategy", "pulp")).lower().strip()
        if "greedy" in strat:
            values["strategy"] = OptimizationStrategy.GREEDY
        elif "hybrid" in strat:
            values["strategy"] = OptimizationStrategy.HYBRID
        else:
            values["strategy"] = OptimizationStrategy.PULP

        return values


class OptimizationResult(BaseModel):
    budget: float
    total_cost_allocated: float
    total_risk_reduction: float
    budget_utilization_pct: float
    overall_roi: float
    strategy_used: str
    selected_actions: List[RemediationResponse]
    sensitivity_analysis: Optional[Dict[str, Any]] = None
    ranked_controls: Optional[List[Dict[str, Any]]] = None
    pareto_frontier: Optional[List[Dict[str, Any]]] = None
    provenance: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(extra="ignore")


# Risk Summary Schemas
class RiskSummaryResponse(BaseModel):
    total_quantified_risk: float
    formatted_total_risk: str
    total_assets: int
    total_findings: int
    risk_by_business_unit: Dict[str, float]
    risk_by_asset_category: Dict[str, float]
    top_risks: List[FindingResponse]
    average_asset_risk: float
    timestamp: datetime = Field(default_factory=_utc_now)

    model_config = ConfigDict(extra="ignore")


# Ingestion Schemas
class IngestRequest(BaseModel):
    reseed_synthetic: bool = Field(default=True, description="Generate reproducible synthetic dataset")
    days_back: Optional[int] = Field(default=30, description="Days to look back for vulnerabilities")
    asset_mapping: Optional[Dict[str, Any]] = Field(default=None, alias="assetMapping")

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    @model_validator(mode="before")
    @classmethod
    def sanitize_ingest_inputs(cls, values: Any) -> Any:
        if not isinstance(values, dict):
            return {"reseed_synthetic": True, "days_back": 30}
        
        # Handle string booleans like 'true' / 'false'
        reseed = values.get("reseed_synthetic", values.get("reseedSynthetic", True))
        if isinstance(reseed, str):
            values["reseed_synthetic"] = reseed.lower() in ("true", "1", "yes")

        # Sanitize days_back
        days = values.get("days_back", values.get("daysBack", 30))
        try:
            values["days_back"] = max(1, min(365, int(days)))
        except (ValueError, TypeError):
            values["days_back"] = 30

        return values


class IngestResponse(BaseModel):
    success: bool
    assets_ingested: int
    findings_ingested: int
    remediations_ingested: int
    message: str

    model_config = ConfigDict(extra="ignore")


# Executive Summary Schemas
class ExecutiveSummaryRequest(BaseModel):
    budget: Optional[float] = Field(default=50000.0)

    @model_validator(mode="before")
    @classmethod
    def sanitize_budget(cls, values: Any) -> Any:
        if isinstance(values, dict):
            raw = values.get("budget", 50000.0)
            if isinstance(raw, str):
                clean = re.sub(r"[^\d.]", "", raw)
                values["budget"] = float(clean) if clean else 50000.0
            elif isinstance(raw, (int, float)):
                values["budget"] = max(1000.0, float(raw))
        return values


# ==========================================
# GRC ANALYST SCHEMAS
# ==========================================

class RiskCalculatorRequest(BaseModel):
    threat_event_frequency: float = Field(default=12.0, ge=0.1, description="Estimated threat contact events per year")
    vulnerability_score: float = Field(default=0.75, ge=0.01, le=1.0, description="Probability control fails (0.01 to 1.0)")
    primary_loss_usd: float = Field(default=150000.0, ge=0.0, description="Direct incident recovery/investigation cost")
    secondary_loss_usd: float = Field(default=350000.0, ge=0.0, description="Indirect fines, reputational damage, legal loss")
    control_effectiveness_pct: float = Field(default=85.0, ge=0.0, le=100.0, description="Expected control mitigation %")
    risk_appetite_threshold_usd: float = Field(default=100000.0, ge=0.0, description="Enterprise risk appetite tolerance")

    model_config = ConfigDict(extra="ignore")


class RiskCalculatorResponse(BaseModel):
    loss_magnitude_usd: float
    inherent_risk_usd: float
    residual_risk_usd: float
    risk_reduction_usd: float
    exceeds_appetite: bool
    action_recommendation: str
    fair_breakdown: Dict[str, Any]
    inherent_ale_band: Optional[Dict[str, float]] = None
    residual_ale_band: Optional[Dict[str, float]] = None
    lef_band: Optional[Dict[str, float]] = None
    reduction_band: Optional[Dict[str, float]] = None
    distribution_bins: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(extra="ignore")


class AuditLogCreate(BaseModel):
    action_type: str = Field(..., description="Action category, e.g. BUDGET_ALLOCATION, RISK_ACCEPTED")
    actor: str = Field(default="GRC Analyst / Risk Committee")
    details: str = Field(..., description="Decision justification and operational details")
    framework_ref: str = Field(default="NIST CSF 2.0 / ISO 27001")
    impact_usd: float = Field(default=0.0)

    model_config = ConfigDict(extra="ignore")


class AuditLogResponse(BaseModel):
    id: int
    timestamp: datetime
    action_type: str
    actor: str
    details: str
    framework_ref: str
    impact_usd: float

    model_config = ConfigDict(from_attributes=True, extra="ignore")


class ComplianceGapResponse(BaseModel):
    overall_compliance_score: float
    nist_csf_functions: Dict[str, Dict[str, Any]]
    iso_27001_controls: Dict[str, Dict[str, Any]]
    active_gaps_count: int
    remediation_roadmap: List[Dict[str, Any]]

    model_config = ConfigDict(extra="ignore")


class PolicyMappingRequest(BaseModel):
    policy_text: str = Field(..., min_length=10, description="Organizational policy text to map to framework controls")

    model_config = ConfigDict(extra="ignore")

    @model_validator(mode="before")
    @classmethod
    def sanitize_policy(cls, values: Any) -> Any:
        if isinstance(values, dict):
            text = values.get("policy_text", "")
            if isinstance(text, str):
                values["policy_text"] = text.strip()
        return values


class PolicyMappingResponse(BaseModel):
    nist_csf_mappings: List[Dict[str, Any]]
    iso_27001_mappings: List[Dict[str, Any]]
    overall_coverage_score: float
    unmapped_areas: List[str]
    policy_summary: str
    total_nist_controls_matched: int = 0
    total_iso_controls_matched: int = 0

    model_config = ConfigDict(extra="ignore")


class ScenarioInput(BaseModel):
    name: str = Field(default="Scenario", description="Scenario label")
    budget: float = Field(default=50000.0, gt=0, description="Budget for this scenario in USD")


class ScenarioAnalysisRequest(BaseModel):
    scenarios: List[ScenarioInput] = Field(..., min_length=1, max_length=10, description="Budget scenarios to compare")

    model_config = ConfigDict(extra="ignore")

    @model_validator(mode="before")
    @classmethod
    def sanitize_scenarios(cls, values: Any) -> Any:
        if isinstance(values, dict):
            scenarios = values.get("scenarios", [])
            sanitized = []
            for s in scenarios:
                if isinstance(s, dict):
                    budget = s.get("budget", 50000)
                    if isinstance(budget, str):
                        clean = re.sub(r"[^\d.]", "", budget)
                        budget = float(clean) if clean else 50000.0
                    s["budget"] = max(1000, float(budget))
                    sanitized.append(s)
            values["scenarios"] = sanitized
        return values


class ScenarioAnalysisResponse(BaseModel):
    scenarios: List[Dict[str, Any]]
    best_roi_scenario: Optional[str] = None
    best_efficiency_scenario: Optional[str] = None
    recommendation: str

    model_config = ConfigDict(extra="ignore")


# ==========================================
# MITRE ATT&CK AI INTEGRATION SCHEMAS
# ==========================================

class AssetContext(BaseModel):
    id: str = Field(..., description="Unique asset identifier")
    criticality: str = Field(..., description="Asset criticality (Low, Moderate, High)")
    availability_requirement: str = Field(..., description="CIA Availability requirement (Low, Moderate, High)")
    value_usd: float = Field(..., description="Financial value of the asset in USD")

class CouncilThreatEvent(BaseModel):
    event_id: str = Field(..., description="Unique event identifier from telemetry")
    mitre_ttp: str = Field(..., pattern=r"^T\d{4}(\.\d{3})?$", description="MITRE ATT&CK Technique ID (e.g., T1059.001)")
    ai_confidence_score: float = Field(..., ge=0.0, le=1.0, description="AI confidence in the TTP mapping")

class MitreEvalRequest(BaseModel):
    threat_event: CouncilThreatEvent
    asset_context: AssetContext
    coso_appetite_usd: float = Field(default=50000.0, description="COSO ERM risk appetite threshold for this business unit")

class MitreEvalResponse(BaseModel):
    action_taken: str = Field(..., description="Automated response action chosen by the Council Safety Gate")
    reason: str = Field(..., description="Justification based on FAIR risk, AI confidence, and asset criticality")
    fair_risk_usd: float = Field(..., description="Calculated FAIR residual risk in USD")
    audit_id: int = Field(..., description="ID of the immutable audit log entry created")


# ==========================================
# NATURAL LANGUAGE QUERY AGENT (RAG) SCHEMAS
# ==========================================

class QueryCitation(BaseModel):
    id: str = Field(..., description="Unique record identifier (e.g. FND-001, ACT-002, AST-003)")
    type: str = Field(..., description="Record entity type: finding, remediation, asset, audit_log, scenario")
    title: str = Field(..., description="Headline or title of the cited entity")
    relevance_score: float = Field(default=1.0, ge=0.0, le=1.0)
    details: Optional[str] = None

    model_config = ConfigDict(extra="ignore")


class QueryAgentRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language question about enterprise risk")
    max_citations: Optional[int] = Field(default=5, ge=1, le=20)

    model_config = ConfigDict(extra="ignore")


class QueryAgentResponse(BaseModel):
    query: str
    answer: str
    citations: List[QueryCitation] = []
    provenance: Dict[str, Any]
    audit_log_id: Optional[int] = None

    model_config = ConfigDict(extra="ignore")


# ==========================================
# SLIDE-OVER DRILL-DOWN PANEL SCHEMAS
# ==========================================

class DrilldownFindingDetail(BaseModel):
    finding: FindingResponse
    asset: Optional[AssetResponse] = None
    remediations: List[RemediationResponse] = []
    mitre_mapping: Optional[Dict[str, Any]] = None
    linked_audit_logs: List[AuditLogResponse] = []
    is_sensitive: bool = False

    model_config = ConfigDict(extra="ignore")


class DrilldownRiskCalcDetail(BaseModel):
    calculation_id: str
    scenario_inputs: Dict[str, Any]
    fair_outputs: Dict[str, Any]
    confidence_bands: Dict[str, Any]
    provenance: Dict[str, Any]
    is_sensitive: bool = False

    model_config = ConfigDict(extra="ignore")


class DrilldownResponse(BaseModel):
    entity_type: str
    entity_id: str
    finding_detail: Optional[DrilldownFindingDetail] = None
    risk_calc_detail: Optional[DrilldownRiskCalcDetail] = None
    audit_logged: bool = False
    audit_log_id: Optional[int] = None

    model_config = ConfigDict(extra="ignore")

