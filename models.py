"""
CyberGuard AI - SQLAlchemy ORM Models
Defines tables for Assets, Security Findings, Remediations, Optimization Runs, and Reports.
"""

from datetime import datetime, timezone
from sqlalchemy import (
    Column, Integer, String, Float, Text, Boolean, DateTime, ForeignKey, JSON
)
from sqlalchemy.orm import relationship
from backend.database import Base


def _get_utc_now():
    """Returns current UTC timestamp without deprecated datetime.utcnow() call."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class AssetModel(Base):
    __tablename__ = "assets"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    asset_id = Column(String(64), unique=True, index=True, nullable=False)
    asset_name = Column(String(255), nullable=False)
    asset_category = Column(String(64), nullable=False, index=True)
    business_unit = Column(String(128), nullable=False, index=True)
    asset_value = Column(Float, nullable=False, default=100000.0)
    exposure = Column(String(64), nullable=False, default="internal")
    criticality = Column(String(32), nullable=False, default="medium")
    owner = Column(String(128), nullable=True)
    created_at = Column(DateTime, default=_get_utc_now)

    # Relationships
    findings = relationship("FindingModel", back_populates="asset", cascade="all, delete-orphan")


class FindingModel(Base):
    __tablename__ = "findings"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    finding_id = Column(String(64), unique=True, index=True, nullable=False)
    asset_id = Column(String(64), ForeignKey("assets.asset_id"), nullable=False, index=True)
    finding_type = Column(String(64), nullable=False, default="vulnerability")
    cve_id = Column(String(64), nullable=True, index=True)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    severity = Column(String(32), nullable=False, default="medium")
    cvss_score = Column(Float, nullable=False, default=5.0)
    epss_score = Column(Float, nullable=True, default=0.05)
    exploit_likelihood = Column(Float, nullable=False, default=0.1)
    exposure_factor = Column(Float, nullable=False, default=0.5)
    quantified_risk_usd = Column(Float, nullable=False, default=0.0)
    nist_csf_category = Column(String(64), nullable=False, default="PROTECT")
    mitre_technique_id = Column(String(64), nullable=True)
    status = Column(String(32), nullable=False, default="open")
    created_at = Column(DateTime, default=_get_utc_now)

    # Relationships
    asset = relationship("AssetModel", back_populates="findings")
    remediations = relationship("RemediationModel", back_populates="finding", cascade="all, delete-orphan")


class RemediationModel(Base):
    __tablename__ = "remediations"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    action_id = Column(String(64), unique=True, index=True, nullable=False)
    finding_id = Column(String(64), ForeignKey("findings.finding_id"), nullable=False, index=True)
    asset_id = Column(String(64), nullable=False, index=True)
    title = Column(String(255), nullable=False)
    remediation_type = Column(String(64), nullable=False, default="patch")
    cost = Column(Float, nullable=False, default=1000.0)
    risk_reduction_usd = Column(Float, nullable=False, default=0.0)
    roi = Column(Float, nullable=False, default=0.0)
    effort_hours = Column(Float, nullable=False, default=8.0)
    nist_control_id = Column(String(64), nullable=True)
    is_selected = Column(Boolean, default=False)
    created_at = Column(DateTime, default=_get_utc_now)

    # Relationships
    finding = relationship("FindingModel", back_populates="remediations")


class OptimizationRunModel(Base):
    __tablename__ = "optimization_runs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    run_id = Column(String(64), unique=True, index=True, nullable=False)
    budget = Column(Float, nullable=False)
    total_cost_allocated = Column(Float, nullable=False)
    total_risk_reduction = Column(Float, nullable=False)
    budget_utilization_pct = Column(Float, nullable=False)
    overall_roi = Column(Float, nullable=False)
    strategy = Column(String(32), nullable=False, default="pulp")
    selected_action_ids = Column(JSON, nullable=False)
    run_date = Column(DateTime, default=_get_utc_now)


class ReportModel(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    report_date = Column(DateTime, default=_get_utc_now)
    total_assets = Column(Integer, nullable=False, default=0)
    total_findings = Column(Integer, nullable=False, default=0)
    total_quantified_risk = Column(Float, nullable=False, default=0.0)
    top_risks_snapshot = Column(JSON, nullable=True)
    executive_summary = Column(Text, nullable=False)


class AuditLogModel(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    timestamp = Column(DateTime, default=_get_utc_now)
    action_type = Column(String(64), nullable=False)  # BUDGET_ALLOCATION, RISK_ASSESSMENT, EXCEPTION_APPROVED, CONTROL_SELECTED
    actor = Column(String(128), default="GRC Analyst / Risk Committee")
    details = Column(Text, nullable=False)
    framework_ref = Column(String(64), default="NIST CSF 2.0 / ISO 27001")
    impact_usd = Column(Float, default=0.0)

