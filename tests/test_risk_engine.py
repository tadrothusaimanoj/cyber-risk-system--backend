"""
Unit tests for Continuous Cyber Risk Quantification Engine.
Verifies:
- Formula: Risk($) = Threat Likelihood × Vulnerability × Asset Value × Exposure
- Boundary and zero values
- Business Unit and Asset Category aggregations
- Action risk reduction calculations
- FAIR Monte Carlo simulation
"""

import pytest
from backend.risk_engine import (
    calculate_finding_risk,
    calculate_action_risk_reduction,
    calculate_fair_monte_carlo,
    aggregate_risk_by_business_unit,
    aggregate_risk_by_asset_category,
)
from backend.schemas import AssetCategory


def test_calculate_finding_risk_standard():
    """Verify standard risk calculation with known inputs."""
    # cvss=10.0 (factor=1.0), exploit=0.8, exposure='internet_facing' (1.0), criticality='critical' (1.5), value=$1,000,000
    # Expected: 0.8 * 1.0 * 1.0 * 1.5 * 1,000,000 = $1,200,000
    risk = calculate_finding_risk(
        cvss_score=10.0,
        exploit_likelihood=0.8,
        exposure="internet_facing",
        criticality="critical",
        asset_value=1000000.0
    )
    assert risk == 1200000.0


def test_calculate_finding_risk_zero_asset_value():
    """Zero or negative asset value must result in zero risk."""
    risk = calculate_finding_risk(
        cvss_score=9.5,
        exploit_likelihood=0.9,
        exposure="internet_facing",
        criticality="critical",
        asset_value=0.0
    )
    assert risk == 0.0

    risk_neg = calculate_finding_risk(
        cvss_score=9.5,
        exploit_likelihood=0.9,
        exposure="internet_facing",
        criticality="critical",
        asset_value=-50000.0
    )
    assert risk_neg == 0.0


def test_calculate_finding_risk_exposure_isolation():
    """Isolated assets must have significantly lower exposure risk than internet-facing."""
    risk_internet = calculate_finding_risk(
        cvss_score=8.0,
        exploit_likelihood=0.5,
        exposure="internet_facing",
        criticality="high",
        asset_value=500000.0
    )
    risk_isolated = calculate_finding_risk(
        cvss_score=8.0,
        exploit_likelihood=0.5,
        exposure="isolated",
        criticality="high",
        asset_value=500000.0
    )
    assert risk_isolated < risk_internet
    assert risk_isolated > 0.0


def test_calculate_finding_risk_boundary_caps():
    """Exploit likelihood and CVSS inputs should be bounded safely."""
    # Exceeding 1.0 exploit likelihood
    risk_high = calculate_finding_risk(
        cvss_score=12.0,
        exploit_likelihood=2.5,
        exposure="internal",
        criticality="medium",
        asset_value=100000.0
    )
    # Under 0.0
    risk_low = calculate_finding_risk(
        cvss_score=-2.0,
        exploit_likelihood=-0.5,
        exposure="internal",
        criticality="medium",
        asset_value=100000.0
    )
    assert risk_high > 0.0
    assert risk_low > 0.0
    assert risk_high > risk_low


def test_calculate_action_risk_reduction():
    """Patch should reduce 95% of risk, config 80%, accept risk 0%."""
    base_risk = 100000.0
    patch_red = calculate_action_risk_reduction(base_risk, "patch")
    config_red = calculate_action_risk_reduction(base_risk, "configuration")
    accept_red = calculate_action_risk_reduction(base_risk, "accept_risk")

    assert patch_red == 95000.0
    assert config_red == 80000.0
    assert accept_red == 0.0


def test_calculate_fair_monte_carlo():
    """FAIR Monte Carlo simulation returns valid percentiles and positive EAL."""
    result = calculate_fair_monte_carlo(
        cvss_score=9.0,
        epss_score=0.75,
        asset_value=500000.0,
        criticality="critical",
        runs=1000
    )
    assert "eal_min" in result
    assert "eal_mode" in result
    assert "eal_mean" in result
    assert "eal_95th_percentile" in result
    assert "eal_max" in result
    assert result["eal_min"] <= result["eal_mode"] <= result["eal_95th_percentile"] <= result["eal_max"]
    assert result["eal_mean"] > 0


def test_aggregations():
    """Test business unit and category risk aggregations."""
    class MockAsset:
        def __init__(self, bu, cat):
            self.business_unit = bu
            self.asset_category = cat

    class MockFinding:
        def __init__(self, asset_id, risk):
            self.asset_id = asset_id
            self.quantified_risk_usd = risk

    assets = {
        "A1": MockAsset("FinTech", "server"),
        "A2": MockAsset("FinTech", "database"),
        "A3": MockAsset("Corporate", "workstation")
    }

    findings = [
        MockFinding("A1", 50000.0),
        MockFinding("A2", 30000.0),
        MockFinding("A3", 20000.0)
    ]

    bu_agg = aggregate_risk_by_business_unit(findings, assets)
    cat_agg = aggregate_risk_by_asset_category(findings, assets)

    assert bu_agg["FinTech"] == 80000.0
    assert bu_agg["Corporate"] == 20000.0
    assert cat_agg["server"] == 50000.0
    assert cat_agg["database"] == 30000.0
    assert cat_agg["workstation"] == 20000.0


def test_calculate_fair_monte_carlo_confidence_bands():
    """Verify that Monte Carlo outputs complete P10/P50/P90 confidence intervals."""
    res = calculate_fair_monte_carlo(
        cvss_score=8.5,
        epss_score=0.75,
        asset_value=1000000.0,
        criticality="critical",
        runs=2000
    )
    assert "eal_p10" in res and "eal_p50" in res and "eal_p90" in res
    assert res["eal_min"] <= res["eal_p10"] <= res["eal_p50"] <= res["eal_p90"] <= res["eal_max"]
    assert "lef_p10" in res and "lef_p50" in res and "lef_p90" in res
    assert res["lef_min"] <= res["lef_p10"] <= res["lef_p50"] <= res["lef_p90"] <= res["lef_max"]


def test_calculate_fair_risk_confidence_bands():
    """Verify that GRC calculate_fair_risk returns empirical confidence bands and distribution bins."""
    from backend.grc_engine import calculate_fair_risk
    res = calculate_fair_risk(
        threat_event_frequency=12,
        vulnerability_score=0.75,
        primary_loss_usd=150000,
        secondary_loss_usd=350000,
        control_effectiveness_pct=85,
        risk_appetite_threshold_usd=100000
    )
    assert "inherent_ale_band" in res
    assert "residual_ale_band" in res
    assert "lef_band" in res
    assert "distribution_bins" in res

    inh = res["inherent_ale_band"]
    res_ale = res["residual_ale_band"]
    assert inh["min"] <= inh["p10"] <= inh["p50"] <= inh["p90"] <= inh["max"]
    assert res_ale["min"] <= res_ale["p10"] <= res_ale["p50"] <= res_ale["p90"] <= res_ale["max"]
    assert res_ale["p50"] < inh["p50"]

    bins = res["distribution_bins"]
    assert len(bins["labels"]) > 5
    assert len(bins["inherent_density"]) == len(bins["labels"])
    assert len(bins["residual_density"]) == len(bins["labels"])

    # Verify reduction_band and range summaries
    assert "reduction_band" in res
    red = res["reduction_band"]
    assert red["min"] <= red["p10"] <= red["p50"] <= red["p90"] <= red["max"]
    assert "loss_event_frequency_range" in res["fair_breakdown"]
    assert "inherent_ale_range" in res["fair_breakdown"]
    assert "residual_ale_range" in res["fair_breakdown"]


def test_scenario_comparison_confidence_bands():
    """Verify that run_scenario_comparison outputs empirical ALE and LEF confidence bands."""
    from backend.grc_engine import run_scenario_comparison
    from backend.models import RemediationModel

    dummy_remediations = [
        RemediationModel(
            action_id=f"ACT-{i}",
            finding_id=f"FND-{i}",
            asset_id=f"AST-{i}",
            title=f"Patch {i}",
            remediation_type="patch",
            cost=5000.0 * i,
            risk_reduction_usd=25000.0 * i,
            roi=5.0,
            effort_hours=10.0,
            nist_control_id="PR.IP-12",
            is_selected=False,
        )
        for i in range(1, 6)
    ]

    scenarios = [
        {"name": "Conservative", "budget": 10000.0},
        {"name": "Aggressive", "budget": 30000.0},
    ]

    res = run_scenario_comparison(scenarios, dummy_remediations, base_risk=500000.0, base_lef=15.0)
    assert len(res["scenarios"]) == 2

    for s in res["scenarios"]:
        assert "confidence_band" in s
        cb = s["confidence_band"]
        assert "min" in cb and "p10" in cb and "p50" in cb and "p90" in cb and "max" in cb
        assert cb["min"] <= cb["p10"] <= cb["p50"] <= cb["p90"] <= cb["max"]

        assert "ale_band" in s
        assert "lef_band" in s
        assert "reduction_band" in s
        assert s["lef_band"]["min"] <= s["lef_band"]["p10"] <= s["lef_band"]["p50"] <= s["lef_band"]["p90"] <= s["lef_band"]["max"]

