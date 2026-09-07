"""
CyberGuard AI - GRC (Governance, Risk & Compliance) Computation Engine
Provides interactive GRC workflow tools for daily analyst operations:
  1. FAIR Risk Calculator - inherent/residual risk with control effectiveness
  2. Compliance Gap Analyzer - NIST CSF 2.0 & ISO 27001 coverage scoring
  3. Policy-to-Control Mapper - maps policy text to framework controls
  4. Scenario Comparator - multi-budget what-if analysis
"""

from typing import List, Dict, Any, Optional, Tuple
import re
import numpy as np
from backend.optimizer import optimize_investments


# =============================================================================
# NIST CSF 2.0 KNOWLEDGE BASE
# =============================================================================
NIST_CSF_SUBCATEGORIES: Dict[str, Dict[str, str]] = {
    "IDENTIFY": {
        "ID.AM-1": "Physical devices and systems are inventoried",
        "ID.AM-2": "Software platforms and applications are inventoried",
        "ID.AM-3": "Organizational communication and data flows are mapped",
        "ID.AM-4": "External information systems are catalogued",
        "ID.AM-5": "Resources are prioritized based on classification and business value",
        "ID.BE-1": "Organization's role in supply chain is identified and communicated",
        "ID.BE-2": "Organization's place in critical infrastructure is identified",
        "ID.BE-3": "Priorities for organizational mission and activities are established",
        "ID.BE-4": "Dependencies and critical functions are established",
        "ID.BE-5": "Resilience requirements are established for delivery of critical services",
        "ID.GV-1": "Organizational cybersecurity policy is established",
        "ID.GV-2": "Cybersecurity roles and responsibilities are coordinated",
        "ID.GV-3": "Legal and regulatory requirements are understood and managed",
        "ID.GV-4": "Governance and risk management processes address cybersecurity risks",
        "ID.RA-1": "Asset vulnerabilities are identified and documented",
        "ID.RA-2": "Cyber threat intelligence is received from information sharing forums",
        "ID.RA-3": "Threats internal and external are identified and documented",
        "ID.RA-4": "Potential business impacts and likelihoods are identified",
        "ID.RA-5": "Threats, vulnerabilities, likelihoods, and impacts determine risk",
        "ID.RA-6": "Risk responses are identified and prioritized",
        "ID.RM-1": "Risk management processes are established and managed",
        "ID.RM-2": "Organizational risk tolerance is determined and expressed",
        "ID.RM-3": "Risk determination includes consideration of role in critical infrastructure",
        "ID.SC-1": "Supply chain risk management processes are established",
        "ID.SC-2": "Suppliers and third-party partners are assessed using risk assessment",
        "ID.SC-3": "Contracts with suppliers include security requirements",
        "ID.SC-4": "Suppliers and third-party partners are routinely assessed",
        "ID.SC-5": "Response and recovery planning and testing are conducted with suppliers",
    },
    "PROTECT": {
        "PR.AC-1": "Identities and credentials are issued, managed, verified, and revoked",
        "PR.AC-2": "Physical access to assets is managed and protected",
        "PR.AC-3": "Remote access is managed",
        "PR.AC-4": "Access permissions and authorizations are managed with least privilege",
        "PR.AC-5": "Network integrity is protected incorporating segregation",
        "PR.AT-1": "All users are informed and trained on cybersecurity",
        "PR.AT-2": "Privileged users understand roles and responsibilities",
        "PR.AT-3": "Third-party stakeholders understand roles and responsibilities",
        "PR.AT-4": "Senior executives understand roles and responsibilities",
        "PR.AT-5": "Physical and cybersecurity personnel understand roles and responsibilities",
        "PR.DS-1": "Data at rest is protected",
        "PR.DS-2": "Data in transit is protected",
        "PR.DS-3": "Assets are managed throughout removal, transfer, and disposition",
        "PR.DS-4": "Adequate capacity to ensure availability is maintained",
        "PR.DS-5": "Protections against data leaks are implemented",
        "PR.DS-6": "Integrity checking mechanisms verify software and information integrity",
        "PR.DS-7": "Development and testing environments are separate from production",
        "PR.IP-1": "Configuration baselines are created and maintained",
        "PR.IP-2": "System development life cycle is implemented to manage systems",
        "PR.IP-3": "Configuration change control processes are in place",
        "PR.IP-4": "Backups of information are conducted and maintained",
        "PR.IP-5": "Policy and regulations regarding physical operating environment are met",
        "PR.IP-6": "Data is destroyed according to policy",
        "PR.IP-7": "Protection processes are improved based on lessons learned",
        "PR.IP-8": "Effectiveness of protection technologies is shared",
        "PR.IP-9": "Response and recovery plans are in place and managed",
        "PR.IP-10": "Response and recovery plans are tested",
        "PR.IP-11": "Cybersecurity is included in human resources practices",
        "PR.IP-12": "A vulnerability management plan is developed and implemented",
        "PR.MA-1": "Maintenance and repair is performed and logged in a timely manner",
        "PR.MA-2": "Remote maintenance is approved, logged, and performed securely",
        "PR.PT-1": "Audit and log records are maintained and reviewed",
        "PR.PT-2": "Removable media is protected and use restricted per policy",
        "PR.PT-3": "Principle of least functionality is incorporated",
        "PR.PT-4": "Communications and control networks are protected",
        "PR.PT-5": "Mechanisms are implemented to achieve resilience requirements",
    },
    "DETECT": {
        "DE.AE-1": "A baseline of network operations and data flows is established",
        "DE.AE-2": "Detected events are analyzed to understand attack targets and methods",
        "DE.AE-3": "Event data are collected and correlated from multiple sources",
        "DE.AE-4": "Impact of events is determined",
        "DE.AE-5": "Incident alert thresholds are established",
        "DE.CM-1": "The network is monitored to detect potential cybersecurity events",
        "DE.CM-2": "The physical environment is monitored for cybersecurity events",
        "DE.CM-3": "Personnel activity is monitored for cybersecurity events",
        "DE.CM-4": "Malicious code is detected",
        "DE.CM-5": "Unauthorized mobile code is detected",
        "DE.CM-6": "External service provider activity is monitored",
        "DE.CM-7": "Monitoring for unauthorized personnel, connections, devices, and software",
        "DE.CM-8": "Vulnerability scans are performed",
        "DE.DP-1": "Roles and responsibilities for detection are well defined",
        "DE.DP-2": "Detection activities comply with all applicable requirements",
        "DE.DP-3": "Detection processes are tested",
        "DE.DP-4": "Event detection information is communicated",
        "DE.DP-5": "Detection processes are continuously improved",
    },
    "RESPOND": {
        "RS.RP-1": "Response plan is executed during or after an incident",
        "RS.CO-1": "Personnel know their roles and order of operations in response",
        "RS.CO-2": "Incidents are reported consistent with established criteria",
        "RS.CO-3": "Information is shared consistent with response plans",
        "RS.CO-4": "Coordination with stakeholders occurs consistent with response plans",
        "RS.CO-5": "Voluntary information sharing occurs with external stakeholders",
        "RS.AN-1": "Notifications from detection systems are investigated",
        "RS.AN-2": "Impact of the incident is understood",
        "RS.AN-3": "Forensics are performed",
        "RS.AN-4": "Incidents are categorized consistent with response plans",
        "RS.AN-5": "Processes are established to receive and analyze vulnerability disclosures",
        "RS.MI-1": "Incidents are contained",
        "RS.MI-2": "Incidents are mitigated",
        "RS.MI-3": "Newly identified vulnerabilities are mitigated or documented as accepted",
        "RS.IM-1": "Response plans incorporate lessons learned",
        "RS.IM-2": "Response strategies are updated",
    },
    "RECOVER": {
        "RC.RP-1": "Recovery plan is executed during or after a cybersecurity incident",
        "RC.IM-1": "Recovery plans incorporate lessons learned",
        "RC.IM-2": "Recovery strategies are updated",
        "RC.CO-1": "Public relations are managed",
        "RC.CO-2": "Reputation is repaired after an incident",
        "RC.CO-3": "Recovery activities are communicated to stakeholders",
    },
}

# =============================================================================
# ISO 27001:2022 ANNEX A KNOWLEDGE BASE
# =============================================================================
ISO_27001_CONTROLS: Dict[str, Dict[str, str]] = {
    "A.5 Organizational Controls": {
        "A.5.1": "Policies for information security",
        "A.5.2": "Information security roles and responsibilities",
        "A.5.3": "Segregation of duties",
        "A.5.4": "Management responsibilities",
        "A.5.5": "Contact with authorities",
        "A.5.6": "Contact with special interest groups",
        "A.5.7": "Threat intelligence",
        "A.5.8": "Information security in project management",
        "A.5.9": "Inventory of information and associated assets",
        "A.5.10": "Acceptable use of information and associated assets",
        "A.5.11": "Return of assets",
        "A.5.12": "Classification of information",
        "A.5.13": "Labelling of information",
        "A.5.14": "Information transfer",
        "A.5.15": "Access control",
        "A.5.16": "Identity management",
        "A.5.17": "Authentication information",
        "A.5.18": "Access rights",
        "A.5.19": "Information security in supplier relationships",
        "A.5.20": "Addressing security within supplier agreements",
        "A.5.21": "Managing security in the ICT supply chain",
        "A.5.22": "Monitoring and review of supplier services",
        "A.5.23": "Information security for use of cloud services",
        "A.5.24": "Information security incident management planning",
        "A.5.25": "Assessment and decision on information security events",
        "A.5.26": "Response to information security incidents",
        "A.5.27": "Learning from information security incidents",
        "A.5.28": "Collection of evidence",
        "A.5.29": "Information security during disruption",
        "A.5.30": "ICT readiness for business continuity",
        "A.5.31": "Legal and regulatory requirements",
        "A.5.32": "Intellectual property rights",
        "A.5.33": "Protection of records",
        "A.5.34": "Privacy and protection of PII",
        "A.5.35": "Independent review of information security",
        "A.5.36": "Compliance with policies and standards",
        "A.5.37": "Documented operating procedures",
    },
    "A.6 People Controls": {
        "A.6.1": "Screening",
        "A.6.2": "Terms and conditions of employment",
        "A.6.3": "Information security awareness, education and training",
        "A.6.4": "Disciplinary process",
        "A.6.5": "Responsibilities after termination or change of employment",
        "A.6.6": "Confidentiality or non-disclosure agreements",
        "A.6.7": "Remote working",
        "A.6.8": "Information security event reporting",
    },
    "A.7 Physical Controls": {
        "A.7.1": "Physical security perimeters",
        "A.7.2": "Physical entry",
        "A.7.3": "Securing offices, rooms and facilities",
        "A.7.4": "Physical security monitoring",
        "A.7.5": "Protecting against physical and environmental threats",
        "A.7.6": "Working in secure areas",
        "A.7.7": "Clear desk and clear screen",
        "A.7.8": "Equipment siting and protection",
        "A.7.9": "Security of assets off-premises",
        "A.7.10": "Storage media",
        "A.7.11": "Supporting utilities",
        "A.7.12": "Cabling security",
        "A.7.13": "Equipment maintenance",
        "A.7.14": "Secure disposal or re-use of equipment",
    },
    "A.8 Technological Controls": {
        "A.8.1": "User endpoint devices",
        "A.8.2": "Privileged access rights",
        "A.8.3": "Information access restriction",
        "A.8.4": "Access to source code",
        "A.8.5": "Secure authentication",
        "A.8.6": "Capacity management",
        "A.8.7": "Protection against malware",
        "A.8.8": "Management of technical vulnerabilities",
        "A.8.9": "Configuration management",
        "A.8.10": "Information deletion",
        "A.8.11": "Data masking",
        "A.8.12": "Data leakage prevention",
        "A.8.13": "Information backup",
        "A.8.14": "Redundancy of information processing facilities",
        "A.8.15": "Logging",
        "A.8.16": "Monitoring activities",
        "A.8.17": "Clock synchronization",
        "A.8.18": "Use of privileged utility programs",
        "A.8.19": "Installation of software on operational systems",
        "A.8.20": "Networks security",
        "A.8.21": "Security of network services",
        "A.8.22": "Segregation of networks",
        "A.8.23": "Web filtering",
        "A.8.24": "Use of cryptography",
        "A.8.25": "Secure development life cycle",
        "A.8.26": "Application security requirements",
        "A.8.27": "Secure system architecture and engineering principles",
        "A.8.28": "Secure coding",
        "A.8.29": "Security testing in development and acceptance",
        "A.8.30": "Outsourced development",
        "A.8.31": "Separation of development, test and production environments",
        "A.8.32": "Change management",
        "A.8.33": "Test information",
        "A.8.34": "Protection of information systems during audit testing",
    },
}

# =============================================================================
# KEYWORD-TO-FRAMEWORK MAPPING INDEX
# =============================================================================
_KEYWORD_MAP: Dict[str, List[str]] = {
    # Vulnerability management
    "vulnerability": ["ID.RA-1", "ID.RA-5", "PR.IP-12", "DE.CM-8", "RS.AN-5", "RS.MI-3", "A.8.8"],
    "scanning": ["DE.CM-8", "DE.CM-1", "PR.IP-12", "A.8.8", "A.8.16"],
    "patch": ["PR.IP-12", "PR.IP-3", "RS.MI-2", "RS.MI-3", "A.8.8", "A.8.9", "A.8.32"],
    "remediation": ["RS.MI-2", "RS.MI-3", "ID.RA-6", "PR.IP-12", "A.8.8"],
    "remediate": ["RS.MI-2", "RS.MI-3", "ID.RA-6", "PR.IP-12", "A.8.8"],
    # Access control
    "access control": ["PR.AC-1", "PR.AC-3", "PR.AC-4", "PR.AC-5", "A.5.15", "A.5.18", "A.8.3"],
    "authentication": ["PR.AC-1", "PR.AC-7", "A.5.17", "A.8.5"],
    "password": ["PR.AC-1", "A.5.17", "A.8.5"],
    "mfa": ["PR.AC-1", "A.5.17", "A.8.5"],
    "multi-factor": ["PR.AC-1", "A.5.17", "A.8.5"],
    "least privilege": ["PR.AC-4", "PR.PT-3", "A.5.15", "A.8.2", "A.8.3"],
    "privileged": ["PR.AC-4", "PR.AT-2", "A.8.2", "A.8.18"],
    "identity": ["PR.AC-1", "A.5.16"],
    # Network security
    "network": ["PR.AC-5", "PR.PT-4", "DE.CM-1", "A.8.20", "A.8.21", "A.8.22"],
    "segmentation": ["PR.AC-5", "A.8.22"],
    "segregation": ["PR.AC-5", "A.5.3", "A.8.22"],
    "firewall": ["PR.AC-5", "PR.PT-4", "A.8.20", "A.8.21"],
    # Data protection
    "encryption": ["PR.DS-1", "PR.DS-2", "A.8.24"],
    "cryptography": ["PR.DS-1", "PR.DS-2", "A.8.24"],
    "data at rest": ["PR.DS-1", "A.8.24"],
    "data in transit": ["PR.DS-2", "A.8.24"],
    "data protection": ["PR.DS-1", "PR.DS-2", "PR.DS-5", "A.8.10", "A.8.11", "A.8.12"],
    "data loss": ["PR.DS-5", "A.8.12"],
    "dlp": ["PR.DS-5", "A.8.12"],
    "backup": ["PR.IP-4", "A.8.13"],
    "privacy": ["A.5.34"],
    "pii": ["A.5.34", "PR.DS-5"],
    # Monitoring & detection
    "monitoring": ["DE.CM-1", "DE.CM-3", "DE.CM-6", "DE.CM-7", "A.8.16"],
    "logging": ["PR.PT-1", "DE.AE-3", "A.8.15"],
    "audit log": ["PR.PT-1", "A.8.15"],
    "siem": ["DE.AE-3", "DE.AE-1", "A.8.16"],
    "detection": ["DE.AE-1", "DE.CM-1", "DE.CM-4", "DE.DP-1", "A.8.7", "A.8.16"],
    "malware": ["DE.CM-4", "A.8.7"],
    "antivirus": ["DE.CM-4", "A.8.7"],
    "intrusion": ["DE.CM-1", "DE.AE-2", "A.8.16"],
    # Incident response
    "incident": ["RS.RP-1", "RS.CO-1", "RS.AN-1", "RS.MI-1", "A.5.24", "A.5.25", "A.5.26"],
    "incident response": ["RS.RP-1", "RS.CO-1", "RS.AN-1", "RS.MI-1", "RS.IM-1", "A.5.24", "A.5.26"],
    "forensic": ["RS.AN-3", "A.5.28"],
    "containment": ["RS.MI-1", "A.5.26"],
    # Business continuity
    "recovery": ["RC.RP-1", "RC.IM-1", "A.5.29", "A.5.30"],
    "business continuity": ["RC.RP-1", "PR.IP-9", "A.5.29", "A.5.30"],
    "disaster recovery": ["RC.RP-1", "PR.IP-9", "PR.IP-10", "A.5.29", "A.5.30"],
    "resilience": ["PR.PT-5", "ID.BE-5", "A.5.30"],
    # Risk management
    "risk assessment": ["ID.RA-1", "ID.RA-4", "ID.RA-5", "ID.RA-6", "ID.RM-1"],
    "risk management": ["ID.RM-1", "ID.RM-2", "ID.GV-4"],
    "risk tolerance": ["ID.RM-2"],
    "risk appetite": ["ID.RM-2"],
    "threat intelligence": ["ID.RA-2", "ID.RA-3", "A.5.7"],
    "threat": ["ID.RA-2", "ID.RA-3", "A.5.7"],
    # Governance & policy
    "policy": ["ID.GV-1", "A.5.1", "A.5.36"],
    "governance": ["ID.GV-1", "ID.GV-4", "A.5.1", "A.5.2"],
    "compliance": ["ID.GV-3", "DE.DP-2", "A.5.31", "A.5.36"],
    "regulatory": ["ID.GV-3", "A.5.31"],
    "legal": ["ID.GV-3", "A.5.31"],
    # Training & awareness
    "training": ["PR.AT-1", "PR.AT-2", "A.6.3"],
    "awareness": ["PR.AT-1", "A.6.3"],
    # Asset management
    "asset inventory": ["ID.AM-1", "ID.AM-2", "A.5.9"],
    "asset management": ["ID.AM-1", "ID.AM-5", "A.5.9", "A.5.10"],
    "classification": ["ID.AM-5", "A.5.12"],
    # Supply chain
    "supply chain": ["ID.SC-1", "ID.SC-2", "ID.SC-3", "A.5.19", "A.5.20", "A.5.21"],
    "supplier": ["ID.SC-2", "ID.SC-4", "A.5.19", "A.5.20", "A.5.22"],
    "third-party": ["ID.SC-2", "PR.AT-3", "A.5.19", "A.5.21"],
    "vendor": ["ID.SC-2", "ID.SC-4", "A.5.19", "A.5.22"],
    "cloud": ["A.5.23", "ID.AM-4"],
    # Change management
    "change management": ["PR.IP-3", "A.8.32"],
    "configuration": ["PR.IP-1", "PR.IP-3", "A.8.9"],
    "baseline": ["PR.IP-1", "DE.AE-1", "A.8.9"],
    # Development security
    "secure development": ["PR.IP-2", "A.8.25", "A.8.26", "A.8.28"],
    "sdlc": ["PR.IP-2", "A.8.25"],
    "code review": ["A.8.28", "A.8.4"],
    "source code": ["A.8.4", "A.8.28"],
    "testing": ["PR.IP-10", "DE.DP-3", "A.8.29"],
    "penetration": ["DE.CM-8", "A.8.29"],
    # Physical security
    "physical": ["PR.AC-2", "DE.CM-2", "A.7.1", "A.7.2", "A.7.4"],
    # Specific compliance items
    "internet-facing": ["PR.AC-5", "PR.PT-4", "DE.CM-1", "A.8.20"],
    "critical": ["ID.AM-5", "ID.BE-4", "A.5.12"],
    "quarterly": ["DE.CM-8", "ID.SC-4", "A.5.35"],
    "30 days": ["RS.MI-3", "PR.IP-12", "A.8.8"],
    "annual": ["PR.AT-1", "ID.SC-4", "A.5.35", "PR.IP-10"],
}


# =============================================================================
# 1. FAIR RISK CALCULATOR
# =============================================================================

def calculate_fair_risk(
    threat_event_frequency: float,
    vulnerability_score: float,
    primary_loss_usd: float,
    secondary_loss_usd: float,
    control_effectiveness_pct: float,
    risk_appetite_threshold_usd: float,
) -> Dict[str, Any]:
    """
    Calculate inherent and residual risk using FAIR methodology.

    FAIR Formula:
      Loss Event Frequency (LEF) = Threat Event Frequency × Vulnerability
      Loss Magnitude (LM) = Primary Loss + Secondary Loss
      Inherent Risk = LEF × LM
      Residual Risk = Inherent Risk × (1 - Control Effectiveness)

    Returns a comprehensive breakdown with action recommendation.
    """
    # Bound inputs
    tef = max(0.1, min(365.0, threat_event_frequency))
    vuln = max(0.01, min(1.0, vulnerability_score))
    primary = max(0.0, primary_loss_usd)
    secondary = max(0.0, secondary_loss_usd)
    effectiveness = max(0.0, min(100.0, control_effectiveness_pct)) / 100.0
    appetite = max(0.0, risk_appetite_threshold_usd)

    # FAIR calculations
    lef = tef * vuln  # Loss Event Frequency (events/year)
    loss_magnitude = primary + secondary
    inherent_risk = lef * loss_magnitude
    risk_reduction = inherent_risk * effectiveness
    residual_risk = inherent_risk - risk_reduction
    exceeds_appetite = residual_risk > appetite

    # Generate action recommendation
    if residual_risk <= 0:
        recommendation = (
            "Controls fully mitigate this risk scenario. No further action required. "
            "Continue monitoring and periodic reassessment per ID.RA-5."
        )
    elif not exceeds_appetite:
        recommendation = (
            f"Residual risk of ${residual_risk:,.0f} is WITHIN the organizational risk appetite "
            f"threshold of ${appetite:,.0f}. Risk can be formally accepted per NIST CSF ID.RA-6. "
            f"Document the risk acceptance decision in the audit trail with CISO approval."
        )
    elif residual_risk <= appetite * 1.5:
        recommendation = (
            f"Residual risk of ${residual_risk:,.0f} EXCEEDS the risk appetite by "
            f"${residual_risk - appetite:,.0f} ({((residual_risk / appetite - 1) * 100):.0f}% over threshold). "
            f"RECOMMENDATION: Increase control effectiveness to {min(99, control_effectiveness_pct + 10):.0f}% "
            f"or implement a compensating control to reduce exposure. Consider PR.IP-12 vulnerability "
            f"management improvements and DE.CM-8 enhanced scanning frequency."
        )
    else:
        recommendation = (
            f"CRITICAL: Residual risk of ${residual_risk:,.0f} significantly exceeds the risk appetite "
            f"of ${appetite:,.0f} by {((residual_risk / appetite - 1) * 100):.0f}%. "
            f"IMMEDIATE ACTION REQUIRED: Escalate to Risk Committee for emergency remediation budget. "
            f"Prioritize patch deployment (PR.IP-12), network segmentation (PR.AC-5), and implement "
            f"compensating controls. Target residual risk below ${appetite:,.0f} within 30 days."
        )

    # 5,000-run Beta-PERT Monte Carlo simulation for empirical confidence bands
    rng = np.random.default_rng(42)
    runs = 5000

    def _pert_samples(low: float, mode: float, high: float, size: int) -> np.ndarray:
        if high <= low or mode < low or mode > high:
            return np.full(size, mode)
        lam = 4.0
        alpha = 1.0 + lam * (mode - low) / (high - low)
        beta = 1.0 + lam * (high - mode) / (high - low)
        return low + rng.beta(alpha, beta, size) * (high - low)

    tef_low = max(0.05, tef * 0.5)
    tef_high = max(tef_low + 0.1, tef * 2.0)
    tef_sim = _pert_samples(tef_low, tef, tef_high, runs)

    v_low = max(0.01, vuln * 0.6)
    v_high = min(1.0, max(v_low + 0.01, vuln * 1.4))
    vuln_sim = _pert_samples(v_low, vuln, v_high, runs)

    lef_sim = tef_sim * vuln_sim

    lm_low = max(100.0, loss_magnitude * 0.4)
    lm_high = max(lm_low + 100.0, loss_magnitude * 2.2)
    lm_sim = _pert_samples(lm_low, loss_magnitude, lm_high, runs)

    inherent_sim = lef_sim * lm_sim
    residual_sim = inherent_sim * (1.0 - effectiveness)
    reduction_sim = inherent_sim - residual_sim

    # Percentiles for Confidence Bands: [min, p10, p50, p90, max]
    inh_pct = np.percentile(inherent_sim, [0, 10, 50, 90, 100])
    res_pct = np.percentile(residual_sim, [0, 10, 50, 90, 100])
    lef_pct = np.percentile(lef_sim, [0, 10, 50, 90, 100])
    red_pct = np.percentile(reduction_sim, [0, 10, 50, 90, 100])

    inherent_ale_band = {
        "min": round(float(inh_pct[0]), 2),
        "p10": round(float(inh_pct[1]), 2),
        "p50": round(float(inh_pct[2]), 2),
        "p90": round(float(inh_pct[3]), 2),
        "max": round(float(inh_pct[4]), 2),
        "mean": round(float(np.mean(inherent_sim)), 2),
    }

    residual_ale_band = {
        "min": round(float(res_pct[0]), 2),
        "p10": round(float(res_pct[1]), 2),
        "p50": round(float(res_pct[2]), 2),
        "p90": round(float(res_pct[3]), 2),
        "max": round(float(res_pct[4]), 2),
        "mean": round(float(np.mean(residual_sim)), 2),
    }

    lef_band = {
        "min": round(float(lef_pct[0]), 4),
        "p10": round(float(lef_pct[1]), 4),
        "p50": round(float(lef_pct[2]), 4),
        "p90": round(float(lef_pct[3]), 4),
        "max": round(float(lef_pct[4]), 4),
        "mean": round(float(np.mean(lef_sim)), 4),
    }

    reduction_band = {
        "min": round(float(red_pct[0]), 2),
        "p10": round(float(red_pct[1]), 2),
        "p50": round(float(red_pct[2]), 2),
        "p90": round(float(red_pct[3]), 2),
        "max": round(float(red_pct[4]), 2),
        "mean": round(float(np.mean(reduction_sim)), 2),
    }

    # Probability density bins for Chart.js confidence band rendering
    bin_count = 14
    combined_max = max(float(inh_pct[3]) * 1.2, float(res_pct[3]) * 1.5, appetite * 1.3)
    bin_edges = np.linspace(0, combined_max, bin_count + 1)
    inh_counts, _ = np.histogram(inherent_sim, bins=bin_edges)
    res_counts, _ = np.histogram(residual_sim, bins=bin_edges)

    bin_labels = []
    for i in range(bin_count):
        mid = (bin_edges[i] + bin_edges[i+1]) / 2.0
        if mid >= 1_000_000:
            bin_labels.append(f"${mid/1_000_000:.1f}M")
        elif mid >= 1_000:
            bin_labels.append(f"${mid/1_000:.0f}K")
        else:
            bin_labels.append(f"${mid:.0f}")

    distribution_bins = {
        "labels": bin_labels,
        "bin_edges": [round(float(b), 2) for b in bin_edges],
        "inherent_density": [round(float(c / runs * 100), 2) for c in inh_counts],
        "residual_density": [round(float(c / runs * 100), 2) for c in res_counts],
        "appetite_threshold_usd": appetite,
    }

    return {
        "loss_magnitude_usd": round(loss_magnitude, 2),
        "inherent_risk_usd": round(inherent_risk, 2),
        "residual_risk_usd": round(residual_risk, 2),
        "risk_reduction_usd": round(risk_reduction, 2),
        "exceeds_appetite": exceeds_appetite,
        "action_recommendation": recommendation,
        "fair_breakdown": {
            "threat_event_frequency": round(tef, 2),
            "vulnerability_probability": round(vuln, 4),
            "loss_event_frequency": round(lef, 4),
            "primary_loss_usd": round(primary, 2),
            "secondary_loss_usd": round(secondary, 2),
            "total_loss_magnitude_usd": round(loss_magnitude, 2),
            "control_effectiveness_pct": round(effectiveness * 100, 1),
            "risk_reduction_pct": round(effectiveness * 100, 1),
            "inherent_risk_usd": round(inherent_risk, 2),
            "residual_risk_usd": round(residual_risk, 2),
            "risk_appetite_threshold_usd": round(appetite, 2),
            "appetite_utilization_pct": round(
                (residual_risk / appetite * 100) if appetite > 0 else 0, 1
            ),
            "loss_event_frequency_range": f"{lef_band['min']:.2f} – {lef_band['p50']:.2f} – {lef_band['max']:.2f}/yr",
            "inherent_ale_range": f"${inherent_ale_band['min']:,.0f} – ${inherent_ale_band['p50']:,.0f} – ${inherent_ale_band['max']:,.0f}",
            "residual_ale_range": f"${residual_ale_band['min']:,.0f} – ${residual_ale_band['p50']:,.0f} – ${residual_ale_band['max']:,.0f}",
            "risk_reduction_range": f"${reduction_band['min']:,.0f} – ${reduction_band['p50']:,.0f} – ${reduction_band['max']:,.0f}",
        },
        "inherent_ale_band": inherent_ale_band,
        "residual_ale_band": residual_ale_band,
        "lef_band": lef_band,
        "reduction_band": reduction_band,
        "distribution_bins": distribution_bins,
    }


# =============================================================================
# 2. COMPLIANCE GAP ANALYZER
# =============================================================================

def analyze_compliance_gaps(
    findings: list,
    assets: list,
) -> Dict[str, Any]:
    """
    Analyze compliance posture against NIST CSF 2.0 and ISO 27001.
    Uses actual finding data to determine which controls are addressed
    and which have gaps requiring remediation.
    """
    # Count findings per NIST CSF function
    nist_finding_counts: Dict[str, int] = {
        "IDENTIFY": 0, "PROTECT": 0, "DETECT": 0, "RESPOND": 0, "RECOVER": 0
    }
    nist_risk_totals: Dict[str, float] = {
        "IDENTIFY": 0, "PROTECT": 0, "DETECT": 0, "RESPOND": 0, "RECOVER": 0
    }

    for f in findings:
        cat = getattr(f, "nist_csf_category", "PROTECT")
        if cat in nist_finding_counts:
            nist_finding_counts[cat] += 1
            nist_risk_totals[cat] += getattr(f, "quantified_risk_usd", 0.0)

    # Determine coverage per NIST CSF function
    # Coverage is based on: having active findings = controls being assessed/addressed
    total_findings = len(findings)
    nist_functions: Dict[str, Dict[str, Any]] = {}

    # Expected minimum control coverage per function
    expected_controls = {
        "IDENTIFY": len(NIST_CSF_SUBCATEGORIES.get("IDENTIFY", {})),
        "PROTECT": len(NIST_CSF_SUBCATEGORIES.get("PROTECT", {})),
        "DETECT": len(NIST_CSF_SUBCATEGORIES.get("DETECT", {})),
        "RESPOND": len(NIST_CSF_SUBCATEGORIES.get("RESPOND", {})),
        "RECOVER": len(NIST_CSF_SUBCATEGORIES.get("RECOVER", {})),
    }

    for func_name in ["IDENTIFY", "PROTECT", "DETECT", "RESPOND", "RECOVER"]:
        finding_count = nist_finding_counts[func_name]
        total_controls = expected_controls[func_name]

        # Coverage heuristic: findings being tracked = controls being monitored
        # More findings = more controls being assessed (up to a cap)
        addressed = min(total_controls, max(1, int(finding_count * 0.6)))
        if finding_count == 0:
            addressed = max(1, int(total_controls * 0.15))  # baseline coverage

        coverage_pct = round((addressed / total_controls) * 100, 1) if total_controls > 0 else 0
        gap_count = total_controls - addressed

        if coverage_pct >= 80:
            status = "COMPLIANT"
        elif coverage_pct >= 50:
            status = "PARTIAL"
        else:
            status = "GAP"

        nist_functions[func_name] = {
            "total_controls": total_controls,
            "controls_addressed": addressed,
            "coverage_pct": coverage_pct,
            "findings_mapped": finding_count,
            "risk_exposure_usd": round(nist_risk_totals[func_name], 2),
            "gap_count": gap_count,
            "status": status,
        }

    # ISO 27001 analysis based on finding types and remediation coverage
    iso_controls: Dict[str, Dict[str, Any]] = {}
    finding_types = set()
    severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}

    for f in findings:
        finding_types.add(getattr(f, "finding_type", "vulnerability"))
        sev = getattr(f, "severity", "medium").lower()
        if sev in severity_counts:
            severity_counts[sev] += 1

    for domain_name, controls in ISO_27001_CONTROLS.items():
        total = len(controls)
        # Estimate coverage based on domain relevance to findings
        if "Technological" in domain_name:
            addressed = min(total, max(3, int(total * (total_findings / 100.0))))
        elif "Organizational" in domain_name:
            addressed = min(total, max(5, int(total * 0.40)))
        elif "People" in domain_name:
            addressed = min(total, max(2, int(total * 0.35)))
        else:  # Physical
            addressed = min(total, max(1, int(total * 0.25)))

        coverage_pct = round((addressed / total) * 100, 1) if total > 0 else 0

        iso_controls[domain_name] = {
            "total_controls": total,
            "controls_addressed": addressed,
            "coverage_pct": coverage_pct,
            "gap_count": total - addressed,
            "status": "COMPLIANT" if coverage_pct >= 80 else ("PARTIAL" if coverage_pct >= 50 else "GAP"),
        }

    # Overall compliance score (weighted average)
    nist_score = sum(v["coverage_pct"] for v in nist_functions.values()) / 5
    iso_score = sum(v["coverage_pct"] for v in iso_controls.values()) / len(iso_controls)
    overall_score = round((nist_score * 0.6 + iso_score * 0.4), 1)

    # Active gaps count
    active_gaps = sum(v["gap_count"] for v in nist_functions.values()) + \
                  sum(v["gap_count"] for v in iso_controls.values())

    # Generate prioritized remediation roadmap
    roadmap = _generate_remediation_roadmap(nist_functions, iso_controls, severity_counts)

    return {
        "overall_compliance_score": overall_score,
        "nist_csf_functions": nist_functions,
        "iso_27001_controls": iso_controls,
        "active_gaps_count": active_gaps,
        "remediation_roadmap": roadmap,
    }


def _generate_remediation_roadmap(
    nist_functions: Dict, iso_controls: Dict, severity_counts: Dict
) -> List[Dict[str, Any]]:
    """Generate a prioritized remediation roadmap based on gap analysis."""
    roadmap = []
    priority = 1

    # Sort NIST functions by gap severity (lowest coverage first)
    sorted_nist = sorted(nist_functions.items(), key=lambda x: x[1]["coverage_pct"])
    for func_name, data in sorted_nist:
        if data["status"] in ("GAP", "PARTIAL"):
            roadmap.append({
                "priority": priority,
                "framework": "NIST CSF 2.0",
                "area": func_name,
                "current_coverage": f"{data['coverage_pct']}%",
                "gaps_to_close": data["gap_count"],
                "risk_exposure": f"${data['risk_exposure_usd']:,.0f}",
                "recommended_action": _get_nist_remediation_action(func_name),
                "effort_estimate": _estimate_effort(data["gap_count"]),
                "target_coverage": "80%+",
            })
            priority += 1

    # Sort ISO domains by gap severity
    sorted_iso = sorted(iso_controls.items(), key=lambda x: x[1]["coverage_pct"])
    for domain_name, data in sorted_iso:
        if data["status"] in ("GAP", "PARTIAL"):
            roadmap.append({
                "priority": priority,
                "framework": "ISO 27001:2022",
                "area": domain_name,
                "current_coverage": f"{data['coverage_pct']}%",
                "gaps_to_close": data["gap_count"],
                "risk_exposure": "Assessment required",
                "recommended_action": f"Implement remaining {data['gap_count']} controls in {domain_name}",
                "effort_estimate": _estimate_effort(data["gap_count"]),
                "target_coverage": "80%+",
            })
            priority += 1

    return roadmap


def _get_nist_remediation_action(function_name: str) -> str:
    """Get recommended remediation action for a NIST CSF function gap."""
    actions = {
        "IDENTIFY": "Conduct comprehensive asset discovery, update risk register, establish governance framework",
        "PROTECT": "Implement access controls, deploy data protection, establish vulnerability management program",
        "DETECT": "Deploy SIEM/monitoring, establish anomaly detection baselines, implement vulnerability scanning",
        "RESPOND": "Develop incident response playbooks, conduct tabletop exercises, establish communication plans",
        "RECOVER": "Create disaster recovery plans, test backup restoration, establish recovery time objectives",
    }
    return actions.get(function_name, "Conduct gap assessment and implement missing controls")


def _estimate_effort(gap_count: int) -> str:
    """Estimate effort to close compliance gaps."""
    if gap_count <= 3:
        return "1-2 weeks"
    elif gap_count <= 8:
        return "2-4 weeks"
    elif gap_count <= 15:
        return "1-2 months"
    else:
        return "2-4 months"


# =============================================================================
# 3. POLICY-TO-CONTROL MAPPER
# =============================================================================

def map_policy_to_controls(policy_text: str) -> Dict[str, Any]:
    """
    Map free-text policy description to relevant framework controls.
    Uses keyword matching against the built-in knowledge base.
    """
    if not policy_text or not policy_text.strip():
        return {
            "nist_csf_mappings": [],
            "iso_27001_mappings": [],
            "overall_coverage_score": 0,
            "unmapped_areas": list(NIST_CSF_SUBCATEGORIES.keys()),
            "policy_summary": "No policy text provided.",
        }

    text_lower = policy_text.lower().strip()

    # Collect matched control IDs with scores
    nist_scores: Dict[str, float] = {}
    iso_scores: Dict[str, float] = {}

    for keyword, control_ids in _KEYWORD_MAP.items():
        if keyword in text_lower:
            # Longer keyword matches get higher relevance
            relevance = min(100, 40 + len(keyword) * 5)
            for cid in control_ids:
                if cid.startswith(("ID.", "PR.", "DE.", "RS.", "RC.")):
                    nist_scores[cid] = max(nist_scores.get(cid, 0), relevance)
                elif cid.startswith("A."):
                    iso_scores[cid] = max(iso_scores.get(cid, 0), relevance)

    # Build NIST mappings with descriptions
    nist_mappings = []
    for cid, score in sorted(nist_scores.items(), key=lambda x: x[1], reverse=True):
        desc = _lookup_nist_description(cid)
        if desc:
            function = cid.split(".")[0]
            function_map = {
                "ID": "IDENTIFY", "PR": "PROTECT", "DE": "DETECT",
                "RS": "RESPOND", "RC": "RECOVER"
            }
            nist_mappings.append({
                "control_id": cid,
                "function": function_map.get(function, function),
                "description": desc,
                "relevance_score": score,
            })

    # Build ISO mappings with descriptions
    iso_mappings = []
    for cid, score in sorted(iso_scores.items(), key=lambda x: x[1], reverse=True):
        desc = _lookup_iso_description(cid)
        if desc:
            domain = _get_iso_domain(cid)
            iso_mappings.append({
                "control_id": cid,
                "domain": domain,
                "description": desc,
                "relevance_score": score,
            })

    # Calculate overall coverage
    nist_functions_covered = set()
    for m in nist_mappings:
        nist_functions_covered.add(m["function"])

    coverage = round(len(nist_functions_covered) / 5 * 100, 1)

    # Identify unmapped NIST functions
    all_functions = {"IDENTIFY", "PROTECT", "DETECT", "RESPOND", "RECOVER"}
    unmapped = sorted(all_functions - nist_functions_covered)

    # Generate policy summary
    policy_summary = (
        f"Policy maps to {len(nist_mappings)} NIST CSF 2.0 controls across "
        f"{len(nist_functions_covered)} functions, and {len(iso_mappings)} ISO 27001 Annex A controls. "
    )
    if unmapped:
        policy_summary += f"GAP: No coverage for NIST CSF functions: {', '.join(unmapped)}. "
    else:
        policy_summary += "Full NIST CSF function coverage achieved. "

    return {
        "nist_csf_mappings": nist_mappings[:20],  # Top 20
        "iso_27001_mappings": iso_mappings[:15],   # Top 15
        "overall_coverage_score": coverage,
        "unmapped_areas": unmapped,
        "policy_summary": policy_summary,
        "total_nist_controls_matched": len(nist_mappings),
        "total_iso_controls_matched": len(iso_mappings),
    }


def _lookup_nist_description(control_id: str) -> Optional[str]:
    """Look up NIST CSF subcategory description."""
    for func_controls in NIST_CSF_SUBCATEGORIES.values():
        if control_id in func_controls:
            return func_controls[control_id]
    return None


def _lookup_iso_description(control_id: str) -> Optional[str]:
    """Look up ISO 27001 Annex A control description."""
    for domain_controls in ISO_27001_CONTROLS.values():
        if control_id in domain_controls:
            return domain_controls[control_id]
    return None


def _get_iso_domain(control_id: str) -> str:
    """Get the ISO 27001 domain for a control ID."""
    for domain_name, controls in ISO_27001_CONTROLS.items():
        if control_id in controls:
            return domain_name
    return "Unknown"


# =============================================================================
# 4. SCENARIO COMPARATOR
# =============================================================================

def run_scenario_comparison(
    scenarios: List[Dict[str, Any]],
    remediations: list,
    base_risk: float = 4579685.84,
    base_lef: float = 24.5,
) -> Dict[str, Any]:
    """
    Run PuLP optimization across multiple budget scenarios and compare results with
    empirical Monte Carlo confidence bands (min, p10, p50, p90, max) for both ALE and LEF.
    Each scenario = { "name": str, "budget": float }
    """
    if not scenarios:
        return {"scenarios": [], "recommendation": "No scenarios provided."}

    results = []
    best_roi_scenario = None
    best_roi = 0.0
    best_efficiency_scenario = None
    best_efficiency = 0.0

    for scenario in scenarios:
        name = scenario.get("name", f"Scenario {len(results) + 1}")
        budget = float(scenario.get("budget", 50000))

        if budget <= 0:
            results.append({
                "name": name,
                "budget": budget,
                "error": "Budget must be positive",
            })
            continue

        try:
            opt_result = optimize_investments(
                actions=remediations,
                budget=budget,
                strategy="pulp",
            )
            # Empirical Monte Carlo simulation for scenario residual risk and LEF
            rng = np.random.default_rng(42 + len(results))
            runs = 2500

            def _sim_pert(low: float, mode: float, high: float, size: int) -> np.ndarray:
                if high <= low or mode < low or mode > high:
                    return np.full(size, mode)
                lam = 4.0
                alpha = 1.0 + lam * (mode - low) / (high - low)
                beta = 1.0 + lam * (high - mode) / (high - low)
                return low + rng.beta(alpha, beta, size) * (high - low)

            eff_base_risk = max(10000.0, float(base_risk))
            eff_base_lef = max(0.5, float(base_lef))
            base_lef_sim = _sim_pert(max(0.1, eff_base_lef * 0.5), eff_base_lef, eff_base_lef * 1.8, runs)
            base_lm_sim = _sim_pert(max(1000.0, (eff_base_risk / eff_base_lef) * 0.4), eff_base_risk / eff_base_lef, (eff_base_risk / eff_base_lef) * 2.2, runs)
            base_ale_sim = base_lef_sim * base_lm_sim

            mitigation_ratio = min(0.99, opt_result.total_risk_reduction / eff_base_risk) if eff_base_risk > 0 else 0.0
            residual_ale_sim = np.maximum(0.0, base_ale_sim * (1.0 - mitigation_ratio))
            residual_lef_sim = np.maximum(0.05, base_lef_sim * (1.0 - mitigation_ratio * 0.60))
            reduction_sim = np.maximum(0.0, base_ale_sim - residual_ale_sim)

            ale_pct = np.percentile(residual_ale_sim, [0, 10, 50, 90, 100])
            lef_pct = np.percentile(residual_lef_sim, [0, 10, 50, 90, 100])
            red_pct_band = np.percentile(reduction_sim, [0, 10, 50, 90, 100])

            ale_band = {
                "min": round(float(ale_pct[0]), 2),
                "p10": round(float(ale_pct[1]), 2),
                "p50": round(float(ale_pct[2]), 2),
                "p90": round(float(ale_pct[3]), 2),
                "max": round(float(ale_pct[4]), 2),
                "mean": round(float(np.mean(residual_ale_sim)), 2),
            }
            lef_band = {
                "min": round(float(lef_pct[0]), 2),
                "p10": round(float(lef_pct[1]), 2),
                "p50": round(float(lef_pct[2]), 2),
                "p90": round(float(lef_pct[3]), 2),
                "max": round(float(lef_pct[4]), 2),
                "mean": round(float(np.mean(residual_lef_sim)), 2),
            }
            reduction_band = {
                "min": round(float(red_pct_band[0]), 2),
                "p10": round(float(red_pct_band[1]), 2),
                "p50": round(float(red_pct_band[2]), 2),
                "p90": round(float(red_pct_band[3]), 2),
                "max": round(float(red_pct_band[4]), 2),
                "mean": round(float(np.mean(reduction_sim)), 2),
            }

            confidence_band = {
                "min": ale_band["min"],
                "p10": ale_band["p10"],
                "p50": ale_band["p50"],
                "p90": ale_band["p90"],
                "max": ale_band["max"],
                "reduction_p10": reduction_band["p10"],
                "reduction_p50": reduction_band["p50"],
                "reduction_p90": reduction_band["p90"],
                "ale_band": ale_band,
                "lef_band": lef_band,
                "reduction_band": reduction_band,
            }
            result = {
                "name": name,
                "budget": round(budget, 2),
                "total_cost_allocated": opt_result.total_cost_allocated,
                "total_risk_reduction": opt_result.total_risk_reduction,
                "budget_utilization_pct": opt_result.budget_utilization_pct,
                "overall_roi": opt_result.overall_roi,
                "controls_selected": len(opt_result.selected_actions),
                "strategy_used": opt_result.strategy_used,
                "confidence_band": confidence_band,
                "ale_band": ale_band,
                "lef_band": lef_band,
                "reduction_band": reduction_band,
            }
            results.append(result)

            if opt_result.overall_roi > best_roi:
                best_roi = opt_result.overall_roi
                best_roi_scenario = name

            # Marginal efficiency = risk reduction per dollar of budget
            efficiency = opt_result.total_risk_reduction / budget if budget > 0 else 0
            if efficiency > best_efficiency:
                best_efficiency = efficiency
                best_efficiency_scenario = name

        except Exception as e:
            results.append({
                "name": name,
                "budget": round(budget, 2),
                "error": str(e),
            })

    # Generate recommendation
    recommendation = _generate_scenario_recommendation(results, best_roi_scenario, best_efficiency_scenario)

    return {
        "scenarios": results,
        "best_roi_scenario": best_roi_scenario,
        "best_efficiency_scenario": best_efficiency_scenario,
        "recommendation": recommendation,
    }


def _generate_scenario_recommendation(
    results: List[Dict], best_roi: Optional[str], best_efficiency: Optional[str]
) -> str:
    """Generate a recommendation based on scenario comparison."""
    valid = [r for r in results if "error" not in r]
    if not valid:
        return "No valid scenarios to compare. Please provide scenarios with positive budgets."

    if len(valid) == 1:
        r = valid[0]
        return (
            f"Single scenario '{r['name']}' achieves {r['overall_roi']}x ROI with "
            f"${r['total_risk_reduction']:,.0f} risk reduction at ${r['total_cost_allocated']:,.0f} cost. "
            f"Consider adding more budget scenarios for comparison."
        )

    # Find diminishing returns point
    sorted_by_budget = sorted(valid, key=lambda x: x["budget"])
    marginal_rois = []
    for i in range(1, len(sorted_by_budget)):
        prev = sorted_by_budget[i - 1]
        curr = sorted_by_budget[i]
        delta_budget = curr["budget"] - prev["budget"]
        delta_reduction = curr["total_risk_reduction"] - prev["total_risk_reduction"]
        marginal_roi = delta_reduction / delta_budget if delta_budget > 0 else 0
        marginal_rois.append((curr["name"], marginal_roi))

    recommendation = (
        f"Best overall ROI: '{best_roi}'. "
        f"Best cost efficiency: '{best_efficiency}'. "
    )

    # Check for diminishing returns
    if marginal_rois:
        worst_marginal = min(marginal_rois, key=lambda x: x[1])
        best_marginal = max(marginal_rois, key=lambda x: x[1])
        if worst_marginal[1] < 1.0:
            recommendation += (
                f"Diminishing returns detected beyond '{worst_marginal[0]}' "
                f"(marginal ROI: {worst_marginal[1]:.2f}x). "
            )
        recommendation += (
            f"Highest marginal efficiency stepping up to '{best_marginal[0]}' "
            f"(marginal ROI: {best_marginal[1]:.2f}x)."
        )

    return recommendation


# =============================================================================
# 5. MITRE ATT&CK AI RESPONSE HANDLER (FAIR & COUNCIL SAFETY GATE)
# =============================================================================

# Default FAIR modifiers for MITRE Tactics/Techniques based on historical impact
MITRE_FAIR_MODIFIERS = {
    # Tactics
    "Initial Access": {"tef_multiplier": 1.5, "lm_multiplier": 0.5},
    "Execution": {"tef_multiplier": 1.2, "lm_multiplier": 0.8},
    "Persistence": {"tef_multiplier": 0.8, "lm_multiplier": 1.0},
    "Privilege Escalation": {"tef_multiplier": 0.5, "lm_multiplier": 1.5},
    "Defense Evasion": {"tef_multiplier": 1.0, "lm_multiplier": 1.2},
    "Credential Access": {"tef_multiplier": 1.5, "lm_multiplier": 1.2},
    "Discovery": {"tef_multiplier": 2.0, "lm_multiplier": 0.2},
    "Lateral Movement": {"tef_multiplier": 0.8, "lm_multiplier": 1.5},
    "Collection": {"tef_multiplier": 0.6, "lm_multiplier": 1.8},
    "Command and Control": {"tef_multiplier": 0.8, "lm_multiplier": 1.2},
    "Exfiltration": {"tef_multiplier": 0.3, "lm_multiplier": 2.5},
    "Impact": {"tef_multiplier": 0.2, "lm_multiplier": 3.0}, # e.g. Ransomware
    
    # Specific Techniques (overrides tactics if matched)
    "T1059.001": {"tef_multiplier": 1.5, "lm_multiplier": 0.8}, # PowerShell
    "T1486": {"tef_multiplier": 0.2, "lm_multiplier": 3.5}, # Data Encrypted for Impact (Ransomware)
    "T1078": {"tef_multiplier": 1.0, "lm_multiplier": 2.0}, # Valid Accounts
}

def calculate_mitre_fair_risk(mitre_ttp: str, asset_value_usd: float) -> float:
    """
    Quantifies the financial risk (Probable Loss) of a specific MITRE TTP against an asset
    using the FAIR framework methodology.
    """
    # Base baseline metrics for a generic threat event
    base_tef = 12.0 # Threat Event Frequency: 12 times a year
    base_lm_pct = 0.10 # Loss Magnitude: 10% of asset value
    
    # Fetch modifiers from Knowledge Base (default to 1.0 if unknown)
    modifiers = MITRE_FAIR_MODIFIERS.get(mitre_ttp, {"tef_multiplier": 1.0, "lm_multiplier": 1.0})
    
    # Calculate Threat Event Frequency (TEF) and Loss Magnitude (LM)
    adjusted_tef = base_tef * modifiers["tef_multiplier"]
    adjusted_lm = (asset_value_usd * base_lm_pct) * modifiers["lm_multiplier"]
    
    # Probable Loss (Risk) = TEF * LM (simplified FAIR annualized loss expectancy)
    inherent_risk_usd = (adjusted_tef / 12.0) * adjusted_lm # Normalized for a single occurrence context
    
    return inherent_risk_usd

def evaluate_mitre_threat_response(threat_event: dict, asset_context: dict, coso_appetite_usd: float) -> dict:
    """
    Implements the 'Council Safety Gate' logic for AI-driven automated responses.
    Evaluates the detected MITRE TTP against FAIR risk and COSO ERM tolerances.
    """
    ttp = threat_event.get("mitre_ttp", "Unknown")
    confidence = threat_event.get("ai_confidence_score", 0.0)
    asset_id = asset_context.get("id", "Unknown")
    criticality = asset_context.get("criticality", "Moderate")
    availability_req = asset_context.get("availability_requirement", "Moderate")
    asset_value = asset_context.get("value_usd", 0.0)
    
    # 1. FAIR: Quantify the risk of the TTP succeeding
    risk_usd = calculate_mitre_fair_risk(ttp, asset_value)
    
    # 2. Council Safety Gate (CIA Triad & GRC Tolerance Checking)
    if risk_usd > coso_appetite_usd:
        # ISO 27005 / NIST RMF / NIST CSF Check: Prevent disruption of critical operations on low confidence
        if availability_req == "High" and confidence < 0.95:
            action = "ESCALATE_TO_HUMAN"
            reason = f"High asset availability requirement. AI confidence ({confidence:.2f}) insufficient for automated disruption despite risk exceeding COSO appetite."
        else:
            # NIST CSF (Respond): High risk, high confidence -> Automate containment
            action = "ISOLATE_ASSET"
            reason = f"FAIR risk (${risk_usd:,.2f}) exceeds COSO appetite (${coso_appetite_usd:,.2f}). Automated containment authorized by Council Safety Gate."
    else:
        # Risk is within appetite, monitor and log
        action = "ENRICH_AND_MONITOR"
        reason = f"FAIR risk (${risk_usd:,.2f}) is within COSO appetite (${coso_appetite_usd:,.2f}). Proceeding with standard monitoring."

    return {
        "action_taken": action,
        "reason": reason,
        "fair_risk_usd": round(risk_usd, 2)
    }
