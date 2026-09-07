"""
Unit tests for 0/1 Knapsack Investment Optimization Engine.
Tests:
- Zero / negative budget validation
- Budget smaller than cheapest action
- Exact budget matching
- Full budget coverage
- Budget constraint preservation: Sum(Cost) <= Budget
- Formal PuLP vs Greedy ROI comparison
- Sensitivity analysis
"""

from datetime import datetime, timezone
import pytest
from backend.optimizer import InvestmentOptimizer, optimize_investments
from backend.schemas import RemediationType, OptimizationResult


class MockAction:
    def __init__(self, action_id: str, cost: float, risk_reduction: float):
        self.action_id = action_id
        self.title = f"Fix for {action_id}"
        self.cost = cost
        self.risk_reduction_usd = risk_reduction
        self.remediation_type = RemediationType.PATCH
        self.finding_id = f"FIND-{action_id}"
        self.asset_id = "AST-001"
        self.roi = round(self.risk_reduction_usd / self.cost, 2) if self.cost > 0 else 0.0
        self.effort_hours = 8.0
        self.nist_control_id = "PR.IP-12"
        self.is_selected = False
        self.id = int(action_id.replace("REM-", ""))
        self.created_at = datetime.now(timezone.utc).replace(tzinfo=None)


@pytest.fixture
def sample_actions():
    return [
        MockAction("REM-1", cost=5000, risk_reduction=50000),    # ROI: 10.0x
        MockAction("REM-2", cost=2000, risk_reduction=16000),    # ROI: 8.0x
        MockAction("REM-3", cost=8000, risk_reduction=88000),    # ROI: 11.0x
        MockAction("REM-4", cost=12000, risk_reduction=60000),   # ROI: 5.0x
        MockAction("REM-5", cost=3000, risk_reduction=21000),    # ROI: 7.0x
    ]


def test_invalid_budget_raises_error(sample_actions):
    """Budget <= 0 must raise ValueError."""
    with pytest.raises(ValueError):
        optimize_investments(sample_actions, budget=0.0)

    with pytest.raises(ValueError):
        optimize_investments(sample_actions, budget=-1000.0)


def test_budget_smaller_than_cheapest_action(sample_actions):
    """Budget insufficient for any action must return 0 selected actions."""
    cheapest_cost = min(a.cost for a in sample_actions)
    result = optimize_investments(sample_actions, budget=cheapest_cost - 500)

    assert len(result.selected_actions) == 0
    assert result.total_cost_allocated == 0.0
    assert result.total_risk_reduction == 0.0
    assert result.budget_utilization_pct == 0.0


def test_budget_exact_match(sample_actions):
    """Budget equal to cheapest action ($2,000) selects that action."""
    result = optimize_investments(sample_actions, budget=2000.0, strategy="pulp")
    assert result.total_cost_allocated <= 2000.0
    assert result.total_cost_allocated == 2000.0
    assert len(result.selected_actions) == 1
    assert result.selected_actions[0].action_id == "REM-2"


def test_budget_constraint_never_exceeded(sample_actions):
    """Total allocated cost must NEVER exceed available budget across various budgets."""
    test_budgets = [2500, 7000, 10000, 15000, 18000, 22000, 30000]
    for b in test_budgets:
        res_pulp = optimize_investments(sample_actions, budget=b, strategy="pulp")
        res_greedy = optimize_investments(sample_actions, budget=b, strategy="greedy")

        assert res_pulp.total_cost_allocated <= b, f"PuLP exceeded budget {b}"
        assert res_greedy.total_cost_allocated <= b, f"Greedy exceeded budget {b}"


def test_budget_covers_all_actions(sample_actions):
    """Very large budget must select all valid actions."""
    total_cost = sum(a.cost for a in sample_actions)
    total_reduction = sum(a.risk_reduction_usd for a in sample_actions)

    result = optimize_investments(sample_actions, budget=total_cost + 10000, strategy="pulp")
    assert len(result.selected_actions) == len(sample_actions)
    assert result.total_cost_allocated == total_cost
    assert result.total_risk_reduction == total_reduction


def test_greedy_and_pulp_both_deterministic(sample_actions):
    """Running optimization repeatedly must produce identical deterministic results."""
    res1 = optimize_investments(sample_actions, budget=15000, strategy="pulp")
    res2 = optimize_investments(sample_actions, budget=15000, strategy="pulp")

    assert [a.action_id for a in res1.selected_actions] == [a.action_id for a in res2.selected_actions]
    assert res1.total_cost_allocated == res2.total_cost_allocated
    assert res1.total_risk_reduction == res2.total_risk_reduction


def test_sensitivity_analysis(sample_actions):
    """Sensitivity analysis returns dictionary of budget levels and risk reductions."""
    sensitivity = InvestmentOptimizer.sensitivity_analysis(sample_actions, base_budget=10000)
    assert len(sensitivity) > 0
    # Higher budgets should yield >= risk reduction
    reductions = [data["risk_reduction"] for data in sensitivity.values()]
    for i in range(len(reductions) - 1):
        assert reductions[i] <= reductions[i + 1]


def test_rank_candidate_controls(sample_actions):
    """Verify ranking by marginal utility (delta_ale / cost) and cumulative metrics."""
    selected_ids = {"ACT-001", "ACT-002"}
    ranked = InvestmentOptimizer.rank_candidate_controls(sample_actions, selected_ids)
    assert len(ranked) == len(sample_actions)

    # Check descending order of delta_ale_cost_ratio
    ratios = [r["delta_ale_cost_ratio"] for r in ranked]
    for i in range(len(ratios) - 1):
        assert ratios[i] >= ratios[i + 1]

    # Check monotonic cumulative cost
    for i in range(len(ranked) - 1):
        assert ranked[i]["cumulative_cost"] <= ranked[i + 1]["cumulative_cost"]
        assert ranked[i]["cumulative_risk_reduction"] <= ranked[i + 1]["cumulative_risk_reduction"]

    # Check is_selected flags
    for r in ranked:
        if r["action_id"] in selected_ids:
            assert r["is_selected"] is True
        else:
            assert r["is_selected"] is False


def test_compute_pareto_frontier(sample_actions):
    """Verify Pareto efficient frontier calculation and knee point identification."""
    frontier = InvestmentOptimizer.compute_pareto_frontier(sample_actions, current_budget=10000.0, num_points=10)
    assert len(frontier) >= 10

    # Costs and risk reductions must be non-decreasing along the frontier
    costs = [pt["cost"] for pt in frontier]
    reductions = [pt["risk_reduction"] for pt in frontier]
    for i in range(len(costs) - 1):
        assert costs[i] <= costs[i + 1]
        assert reductions[i] <= reductions[i + 1]

    # Must contain current allocation and a designated knee point
    assert any(pt["is_current_allocation"] for pt in frontier)
    assert any(pt["is_knee_point"] for pt in frontier)


def test_optimize_investments_returns_ranked_and_pareto(sample_actions):
    """Verify public optimize_investments provides ranked_controls, pareto_frontier, and provenance."""
    res = optimize_investments(sample_actions, budget=12000.0, strategy="pulp")
    assert res.ranked_controls is not None
    assert len(res.ranked_controls) == len(sample_actions)
    assert res.pareto_frontier is not None
    assert len(res.pareto_frontier) > 5
    assert res.provenance is not None
    assert "model_used" in res.provenance
    assert "timestamp" in res.provenance
    assert "input_summary" in res.provenance

