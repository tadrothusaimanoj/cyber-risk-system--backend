"""
CyberGuard AI - Continuous Risk Quantification Engine
Implements the core hackathon formula:
    Risk($) = Threat Likelihood × Vulnerability × Asset Value (enriched with Exposure & Criticality)
Also provides FAIR Monte Carlo statistical verification.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from backend.config import settings


def calculate_finding_risk(
    cvss_score: float,
    exploit_likelihood: float,
    exposure: str,
    criticality: str,
    asset_value: float
) -> float:
    """
    Calculate dollar-quantified risk for an individual finding.
    
    Formula:
        Risk($) = Threat Likelihood × Vulnerability Factor × Exposure Factor × Criticality Multiplier × Asset Value
        
    Args:
        cvss_score: CVSS base score [0.0 - 10.0]
        exploit_likelihood: Probability of exploit in the wild [0.0 - 1.0]
        exposure: Exposure level ('internet_facing', 'dmz', 'internal', 'isolated')
        criticality: Asset criticality ('critical', 'high', 'medium', 'low')
        asset_value: Asset value in USD (> 0)
        
    Returns:
        Quantified dollar risk in USD (rounded to 2 decimal places)
    """
    if asset_value <= 0:
        return 0.0

    # 1. Threat Likelihood (bounded between 0.01 and 1.0)
    threat_likelihood = max(0.01, min(1.0, float(exploit_likelihood)))

    # 2. Vulnerability Severity Factor (CVSS normalized: 0.0 to 1.0)
    vuln_factor = max(0.05, min(1.0, float(cvss_score) / 10.0))

    # 3. Exposure Multiplier from centralized config
    exposure_multiplier = settings.EXPOSURE_MULTIPLIERS.get(
        exposure.lower(), settings.EXPOSURE_MULTIPLIERS["internal"]
    )

    # 4. Criticality Multiplier from centralized config
    criticality_multiplier = settings.ASSET_CRITICALITY_MULTIPLIERS.get(
        criticality.lower(), settings.ASSET_CRITICALITY_MULTIPLIERS["medium"]
    )

    # Calculate quantified risk
    risk_usd = threat_likelihood * vuln_factor * exposure_multiplier * criticality_multiplier * asset_value

    # Cap risk at asset value * criticality multiplier to prevent unbounded numbers
    max_allowable_risk = asset_value * criticality_multiplier
    final_risk = min(risk_usd, max_allowable_risk)

    return round(final_risk, 2)


def calculate_action_risk_reduction(
    finding_risk_usd: float,
    remediation_type: str
) -> float:
    """
    Calculate the risk reduction achieved by applying a remediation control.
    
    Effectiveness:
    - patch: 95% risk reduction (virtually eliminates the vulnerability)
    - configuration: 80% risk reduction
    - compensating_control: 65% risk reduction
    - accept_risk: 0% risk reduction
    """
    effectiveness_map = {
        "patch": 0.95,
        "configuration": 0.80,
        "compensating_control": 0.65,
        "accept_risk": 0.0
    }
    effectiveness = effectiveness_map.get(remediation_type.lower(), 0.70)
    return round(finding_risk_usd * effectiveness, 2)


def calculate_fair_monte_carlo(
    cvss_score: float,
    epss_score: Optional[float],
    asset_value: float,
    criticality: str,
    runs: int = 5000
) -> Dict[str, float]:
    """
    FAIR (Factor Analysis of Information Risk) Monte Carlo simulation.
    Used for statistical confidence validation (LEF × LM).
    
    LEF (Loss Event Frequency) = Threat Contact × Vulnerability Probability
    LM (Loss Magnitude) = Primary Loss + Secondary Loss
    """
    # Fix seed for reproducibility
    rng = np.random.default_rng(settings.RANDOM_SEED)

    # Map CVSS to vulnerability failure probability PERT parameters
    if cvss_score >= 9.0:
        v_min, v_mode, v_max = 0.70, 0.85, 0.95
    elif cvss_score >= 7.0:
        v_min, v_mode, v_max = 0.50, 0.70, 0.85
    elif cvss_score >= 4.0:
        v_min, v_mode, v_max = 0.20, 0.40, 0.60
    else:
        v_min, v_mode, v_max = 0.05, 0.15, 0.30

    # Map EPSS to Contact Frequency (events/year)
    epss = epss_score if epss_score is not None else 0.05
    tef_mode = max(1.0, min(100.0, epss * 50))
    tef_min, tef_max = tef_mode * 0.5, tef_mode * 2.0

    # Beta-PERT helper
    def pert_samples(low, mode, high, size):
        if high == low:
            return np.full(size, mode)
        lam = 4.0
        alpha = 1.0 + lam * (mode - low) / (high - low)
        beta = 1.0 + lam * (high - mode) / (high - low)
        return low + rng.beta(alpha, beta, size) * (high - low)

    vuln_samples = pert_samples(v_min, v_mode, v_max, runs)
    tef_samples = pert_samples(tef_min, tef_mode, tef_max, runs)
    lef = tef_samples * vuln_samples

    # Loss Magnitude (LM)
    crit_mult = settings.ASSET_CRITICALITY_MULTIPLIERS.get(criticality.lower(), 1.0)
    loss_mode = asset_value * crit_mult * 0.5
    loss_min = loss_mode * 0.2
    loss_max = loss_mode * 2.5

    lm = pert_samples(loss_min, loss_mode, loss_max, runs)
    eal_distribution = lef * lm

    percentiles = np.percentile(eal_distribution, [0, 10, 50, 90, 95, 100])
    lef_pct = np.percentile(lef, [0, 10, 50, 90, 95, 100])
    lm_pct = np.percentile(lm, [0, 10, 50, 90, 95, 100])

    return {
        "eal_min": round(float(percentiles[0]), 2),
        "eal_p10": round(float(percentiles[1]), 2),
        "eal_p50": round(float(percentiles[2]), 2),
        "eal_mode": round(float(percentiles[2]), 2),
        "eal_mean": round(float(np.mean(eal_distribution)), 2),
        "eal_p90": round(float(percentiles[3]), 2),
        "eal_95th_percentile": round(float(percentiles[4]), 2),
        "eal_max": round(float(percentiles[5]), 2),
        "lef_min": round(float(lef_pct[0]), 4),
        "lef_p10": round(float(lef_pct[1]), 4),
        "lef_p50": round(float(lef_pct[2]), 4),
        "lef_p90": round(float(lef_pct[3]), 4),
        "lef_max": round(float(lef_pct[5]), 4),
        "lef_mean": round(float(np.mean(lef)), 4),
        "lm_min": round(float(lm_pct[0]), 2),
        "lm_p10": round(float(lm_pct[1]), 2),
        "lm_p50": round(float(lm_pct[2]), 2),
        "lm_p90": round(float(lm_pct[3]), 2),
        "lm_max": round(float(lm_pct[5]), 2),
        "lm_mean": round(float(np.mean(lm)), 2),
    }


def aggregate_risk_by_business_unit(findings: List[Any], assets_by_id: Dict[str, Any]) -> Dict[str, float]:
    """Aggregate quantified dollar risk grouped by Business Unit."""
    agg: Dict[str, float] = {}
    for f in findings:
        asset = assets_by_id.get(f.asset_id)
        bu = asset.business_unit if asset else "Unassigned"
        agg[bu] = round(agg.get(bu, 0.0) + getattr(f, "quantified_risk_usd", 0.0), 2)
    return agg


def aggregate_risk_by_asset_category(findings: List[Any], assets_by_id: Dict[str, Any]) -> Dict[str, float]:
    """Aggregate quantified dollar risk grouped by Asset Category."""
    agg: Dict[str, float] = {}
    for f in findings:
        asset = assets_by_id.get(f.asset_id)
        cat = str(asset.asset_category).replace("AssetCategory.", "") if asset else "Other"
        agg[cat] = round(agg.get(cat, 0.0) + getattr(f, "quantified_risk_usd", 0.0), 2)
    return agg
