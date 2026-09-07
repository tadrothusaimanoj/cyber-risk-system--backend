"""
CyberGuard AI - Deterministic Synthetic Enterprise Data Generator
Generates a reproducible corporate environment with 25 realistic assets,
65+ security findings (CVEs, misconfigurations, threat intel), and corresponding
remediation options across 5 core business units.
Fixed seed ensures 100% determinism.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from sqlalchemy.orm import Session
from backend.config import settings
from backend.models import AssetModel, FindingModel, RemediationModel
from backend.risk_engine import calculate_finding_risk, calculate_action_risk_reduction
from backend.ml_prioritization import predict_exploit_likelihood


def generate_synthetic_data(db: Session) -> Tuple[int, int, int]:
    """
    Generate and populate 25 assets, 65+ findings, and remediations.
    Returns: (asset_count, finding_count, remediation_count)
    """
    # Use fixed seed
    rng = np.random.default_rng(settings.RANDOM_SEED)

    # 1. Clear existing data
    db.query(RemediationModel).delete()
    db.query(FindingModel).delete()
    db.query(AssetModel).delete()
    db.commit()

    # 2. Define 25 realistic enterprise assets with calibrated values
    raw_assets = [
        # FinTech / Payments
        ("AST-PAY-001", "Core Payment Gateway Primary", "server", "FinTech / Payments", 130000.0, "internet_facing", "critical", "sre-payments@corp.internal"),
        ("AST-PAY-002", "PCI Cardholder Data Vault", "database", "FinTech / Payments", 160000.0, "isolated", "critical", "secops-compliance@corp.internal"),
        ("AST-PAY-003", "SWIFT Clearing Settlement Node", "server", "FinTech / Payments", 110000.0, "dmz", "critical", "banking-ops@corp.internal"),
        ("AST-PAY-004", "Merchant Partner REST Gateway", "cloud_service", "FinTech / Payments", 75000.0, "internet_facing", "high", "api-gateway-team@corp.internal"),
        ("AST-PAY-005", "Core Transaction Ledger DB", "database", "FinTech / Payments", 95000.0, "internal", "critical", "dba-team@corp.internal"),

        # Core Engineering
        ("AST-ENG-001", "Production Kubernetes Master Cluster", "cloud_service", "Core Engineering", 100000.0, "dmz", "critical", "platform-eng@corp.internal"),
        ("AST-ENG-002", "Production Bastion Jump Host", "server", "Core Engineering", 45000.0, "internet_facing", "high", "devops-sec@corp.internal"),
        ("AST-ENG-003", "GitLab CI/CD Runner Farm", "server", "Core Engineering", 40000.0, "internal", "medium", "devops-team@corp.internal"),
        ("AST-ENG-004", "Docker Artifact Registry", "cloud_service", "Core Engineering", 35000.0, "dmz", "medium", "devops-team@corp.internal"),
        ("AST-ENG-005", "HashiCorp Vault Production Cluster", "server", "Core Engineering", 85000.0, "internal", "critical", "secops@corp.internal"),

        # E-Commerce
        ("AST-ECOM-001", "Customer Storefront Web Frontend", "cloud_service", "E-Commerce", 80000.0, "internet_facing", "high", "frontend-leads@corp.internal"),
        ("AST-ECOM-002", "Product Catalog Redis Cache Cluster", "database", "E-Commerce", 30000.0, "internal", "medium", "ecom-backend@corp.internal"),
        ("AST-ECOM-003", "Checkout & Cart Microservice", "cloud_service", "E-Commerce", 75000.0, "internet_facing", "critical", "checkout-eng@corp.internal"),
        ("AST-ECOM-004", "ElasticSearch Search & Query Node", "server", "E-Commerce", 40000.0, "internal", "medium", "search-team@corp.internal"),
        ("AST-ECOM-005", "Customer Recommendations API", "cloud_service", "E-Commerce", 35000.0, "internal", "low", "ml-recs@corp.internal"),

        # Corporate IT
        ("AST-CORP-001", "Active Directory Primary DC", "server", "Corporate IT", 90000.0, "internal", "critical", "it-sysadmin@corp.internal"),
        ("AST-CORP-002", "Enterprise SSO & Okta Proxy", "cloud_service", "Corporate IT", 65000.0, "internet_facing", "high", "identity-team@corp.internal"),
        ("AST-CORP-003", "Executive Laptop Fleet (50 Hosts)", "workstation", "Corporate IT", 25000.0, "internal", "high", "helpdesk@corp.internal"),
        ("AST-CORP-004", "Corporate Mail Gateway (Exchange)", "server", "Corporate IT", 50000.0, "internet_facing", "high", "mailadmin@corp.internal"),
        ("AST-CORP-005", "Cisco Core VPN Gateway", "network_device", "Corporate IT", 60000.0, "internet_facing", "critical", "netops@corp.internal"),

        # Data Science
        ("AST-DATA-001", "GPU ML Model Training Cluster", "server", "Data Science", 75000.0, "internal", "high", "ml-platform@corp.internal"),
        ("AST-DATA-002", "Customer Analytics S3 Data Lake", "cloud_service", "Data Science", 85000.0, "internal", "critical", "data-infra@corp.internal"),
        ("AST-DATA-003", "Kafka Real-Time Event Pipeline", "server", "Data Science", 55000.0, "internal", "medium", "streaming-team@corp.internal"),
        ("AST-DATA-004", "Snowflake Analytics ETL Worker", "cloud_service", "Data Science", 45000.0, "internal", "medium", "analytics-eng@corp.internal"),
        ("AST-DATA-005", "Internal BI Metabase Server", "server", "Data Science", 20000.0, "internal", "low", "bi-admin@corp.internal"),
    ]

    asset_models = []
    assets_dict = {}
    for item in raw_assets:
        a = AssetModel(
            asset_id=item[0],
            asset_name=item[1],
            asset_category=item[2],
            business_unit=item[3],
            asset_value=item[4],
            exposure=item[5],
            criticality=item[6],
            owner=item[7]
        )
        asset_models.append(a)
        assets_dict[item[0]] = a

    db.add_all(asset_models)
    db.flush()

    # 3. Known Vulnerability & Misconfiguration Catalog (Critical, High, Medium, Low)
    vulnerability_templates = [
        # Critical Tier
        ("CVE-2024-21626", "runc Container Breakout via File Descriptor Leak", 8.6, 0.72, "critical", "vulnerability", "PROTECT", "T1611"),
        ("CVE-2021-44228", "Apache Log4j Remote Code Execution (Log4Shell)", 10.0, 0.94, "critical", "vulnerability", "PROTECT", "T1190"),
        ("CVE-2024-3094", "XZ Utils Liblzma Malicious Backdoor", 10.0, 0.88, "critical", "threat_intelligence", "DETECT", "T1554"),
        ("CVE-2024-23897", "Jenkins CLI Arbitrary File Read via args4j", 9.8, 0.82, "critical", "vulnerability", "PROTECT", "T1213"),
        ("CVE-2023-22515", "Confluence Data Center Broken Access Control", 9.8, 0.89, "critical", "vulnerability", "PROTECT", "T1190"),
        ("CVE-2023-3519", "Citrix ADC Unauthenticated Remote Code Execution", 9.8, 0.91, "critical", "threat_intelligence", "PROTECT", "T1190"),
        ("MISC-K8S-002", "Kubernetes API Server Allows Anonymous Authentication", 9.1, 0.62, "critical", "misconfiguration", "PROTECT", "T1078"),

        # High Tier
        ("CVE-2023-44487", "HTTP/2 Rapid Reset Denial of Service", 7.5, 0.65, "high", "vulnerability", "RESPOND", "T1499"),
        ("CVE-2024-0001", "PostgreSQL Buffer Overflow & Privilege Escalation", 8.8, 0.45, "high", "vulnerability", "PROTECT", "T1068"),
        ("CVE-2023-38606", "Kernel Memory Corruption & Jailbreak", 7.8, 0.32, "high", "vulnerability", "PROTECT", "T1068"),
        ("MISC-S3-001", "AWS S3 Data Lake Bucket Publicly Accessible", 8.2, 0.70, "high", "misconfiguration", "IDENTIFY", "T1530"),
        ("MISC-SSH-004", "SSH Root Login & Password Auth Permitted", 7.2, 0.55, "high", "misconfiguration", "PROTECT", "T1078"),
        ("MISC-IAM-006", "Over-Permissive IAM Role Attached to Compute Instance", 7.9, 0.48, "high", "misconfiguration", "IDENTIFY", "T1078"),
        ("CVE-2024-21413", "Microsoft Outlook Remote Code Execution Flaw", 7.8, 0.58, "high", "vulnerability", "PROTECT", "T1204"),

        # Medium Tier
        ("CVE-2023-48795", "Terrapin SSH Protocol Prefix Truncation Attack", 5.9, 0.28, "medium", "vulnerability", "PROTECT", "T1557"),
        ("MISC-DB-005", "Database Allows Cleartext Connections Without TLS", 6.5, 0.38, "medium", "misconfiguration", "PROTECT", "T1040"),
        ("CVE-2023-32463", "Alteryx Analytics Insecure Library Loading", 5.5, 0.22, "medium", "vulnerability", "PROTECT", "T1574"),
        ("MISC-CORS-007", "Overly Permissive Cross-Origin Resource Sharing (CORS)", 5.2, 0.25, "medium", "misconfiguration", "PROTECT", "T1189"),
        ("MISC-SNMP-008", "Default Public Community String Enabled on Gateway", 6.1, 0.30, "medium", "misconfiguration", "IDENTIFY", "T1046"),

        # Low Tier (Security Hygiene & Informational Gaps)
        ("MISC-TLS-003", "Legacy TLS 1.0/1.1 Protocols Enabled on Ingress", 3.8, 0.12, "low", "misconfiguration", "PROTECT", "T1040"),
        ("MISC-HDR-009", "Missing Strict-Transport-Security (HSTS) and CSP Headers", 3.2, 0.08, "low", "misconfiguration", "PROTECT", "T1189"),
        ("MISC-BANNER-010", "Verbose Server & Framework Version Headers Disclosed", 2.7, 0.05, "low", "misconfiguration", "IDENTIFY", "T1592"),
        ("MISC-PATH-011", "Unquoted Windows Service Path with Spaces in Binary Path", 3.5, 0.11, "low", "misconfiguration", "PROTECT", "T1574"),
        ("MISC-PWD-012", "Password Minimum Age & History Reuse Policy Not Enforced", 3.1, 0.06, "low", "misconfiguration", "PROTECT", "T1078"),
        ("MISC-COOKIE-013", "Session Cookie Missing Secure & SameSite Attributes", 3.7, 0.14, "low", "misconfiguration", "PROTECT", "T1539"),
        ("MISC-DNS-014", "Dangling Subdomain DNS CNAME Entry Detected", 3.4, 0.09, "low", "misconfiguration", "IDENTIFY", "T1584"),
    ]

    findings_to_insert = []
    remediations_to_insert = []
    finding_counter = 1
    action_counter = 1

    # Distribute findings deterministically across all 25 assets to reach ~115-125 findings
    for global_asset_idx, asset in enumerate(asset_models):
        # Assign 4 to 6 findings per asset to reach 115-125 realistic findings
        if asset.criticality == "critical" or asset.exposure == "internet_facing":
            num_findings = 6
        elif asset.criticality == "high":
            num_findings = 5
        else:
            num_findings = 4

        # Pick distinct templates deterministically across the full catalog
        catalog_len = len(vulnerability_templates)
        selected_templates = [
            vulnerability_templates[(global_asset_idx * 3 + offset) % catalog_len]
            for offset in range(num_findings)
        ]

        for t in selected_templates:
            fid = f"FIND-2024-{finding_counter:03d}"
            cve_id, title, cvss, base_epss, severity, f_type, nist_cat, mitre_id = t

            # Slight deterministic jitter to CVSS/EPSS
            jitter = ((finding_counter * 7) % 10 - 5) / 50.0  # -0.1 to +0.1
            cvss_final = max(1.0, min(10.0, round(cvss + jitter, 1)))
            epss_final = max(0.01, min(0.99, round(base_epss + (jitter * 0.5), 3)))

            # Mock finding object for ML prediction
            class MockFinding:
                def __init__(self, cvss_val, epss_val, exp_val, sev_val, cve_val):
                    self.cvss_score = cvss_val
                    self.epss_score = epss_val
                    self.exposure = exp_val
                    self.severity = sev_val
                    self.cve_id = cve_val

            mock_f = MockFinding(cvss_final, epss_final, asset.exposure, severity, cve_id)
            exploit_prob = predict_exploit_likelihood(mock_f)

            # Exposure factor
            exp_factor = settings.EXPOSURE_MULTIPLIERS.get(asset.exposure, 0.5)

            # Calculate dollar-quantified risk
            risk_usd = calculate_finding_risk(
                cvss_score=cvss_final,
                exploit_likelihood=exploit_prob,
                exposure=asset.exposure,
                criticality=asset.criticality,
                asset_value=asset.asset_value
            )

            finding = FindingModel(
                finding_id=fid,
                asset_id=asset.asset_id,
                finding_type=f_type,
                cve_id=cve_id if cve_id.startswith("CVE-") else None,
                title=f"{title} on {asset.asset_name}",
                description=f"Automated scan detected {title}. Affects {asset.asset_name} in {asset.business_unit}.",
                severity=severity,
                cvss_score=cvss_final,
                epss_score=epss_final,
                exploit_likelihood=exploit_prob,
                exposure_factor=exp_factor,
                quantified_risk_usd=risk_usd,
                nist_csf_category=nist_cat,
                mitre_technique_id=mitre_id,
                status="open"
            )
            findings_to_insert.append(finding)

            # Generate remediation option for this finding
            act_id = f"REM-{action_counter:03d}"
            rem_type = "patch" if f_type == "vulnerability" else "configuration"
            
            # Realistic remediation cost: $800 to $25,000 depending on severity and complexity
            if severity == "low":
                base_cost = 800.0 if rem_type == "configuration" else 1500.0
                remediation_cost = round(base_cost + ((finding_counter * 137) % 800), 2)
            else:
                base_cost = 2500.0 if rem_type == "configuration" else 5000.0
                cost_factor = 1.8 if asset.criticality == "critical" else (1.3 if asset.criticality == "high" else 1.0)
                remediation_cost = round(base_cost * cost_factor + ((finding_counter * 311) % 4000), 2)

            risk_reduction = calculate_action_risk_reduction(risk_usd, rem_type)
            roi = round(risk_reduction / remediation_cost, 2) if remediation_cost > 0 else 0.0
            effort = round(4.0 + (remediation_cost / 500.0), 1)

            remediation = RemediationModel(
                action_id=act_id,
                finding_id=fid,
                asset_id=asset.asset_id,
                title=f"Deploy {rem_type.replace('_', ' ').capitalize()} for {cve_id}",
                remediation_type=rem_type,
                cost=remediation_cost,
                risk_reduction_usd=risk_reduction,
                roi=roi,
                effort_hours=effort,
                nist_control_id="PR.IP-12" if rem_type == "patch" else "A.8.9",
                is_selected=False
            )
            remediations_to_insert.append(remediation)

            finding_counter += 1
            action_counter += 1

    db.add_all(findings_to_insert)
    db.add_all(remediations_to_insert)
    db.commit()

    return len(asset_models), len(findings_to_insert), len(remediations_to_insert)
