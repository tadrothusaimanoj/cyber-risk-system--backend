"""
CyberGuard AI - Investment Optimization Engine
Implements 0/1 Knapsack optimization using:
1. Formal integer programming via PuLP (CBC Solver)
2. Greedy ROI ranking (Risk Reduction / Cost)
Guarantees: Total Allocated Cost <= Available Budget.
"""

from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Set
import pulp
from backend.schemas import RemediationResponse, OptimizationResult, OptimizationStrategy


class InvestmentOptimizer:
    """
    Solves security investment resource allocation as a 0/1 Knapsack problem.
    """

    @staticmethod
    def _validate_budget(budget: float) -> None:
        if budget is None or budget <= 0:
            raise ValueError(f"Invalid budget: {budget}. Budget must be greater than 0.")

    @classmethod
    def optimize_greedy(
        cls,
        actions: List[Any],
        budget: float,
        min_roi_threshold: Optional[float] = None
    ) -> OptimizationResult:
        """
        Greedy 0/1 knapsack approximation:
        Sort actions descending by ROI = (risk_reduction / cost).
        Deterministically break ties by cost (cheaper first) and action_id.
        """
        cls._validate_budget(budget)

        # Filter valid candidates (cost > 0, risk_reduction > 0)
        candidates = [
            a for a in actions
            if getattr(a, "cost", 0.0) > 0 and getattr(a, "risk_reduction_usd", 0.0) > 0
        ]

        if min_roi_threshold is not None:
            candidates = [c for c in candidates if (c.risk_reduction_usd / c.cost) >= min_roi_threshold]

        # Sort descending by ROI, ascending by cost, ascending by action_id for determinism
        candidates.sort(
            key=lambda x: (
                round(x.risk_reduction_usd / x.cost, 6),
                -x.cost,
                x.action_id
            ),
            reverse=True
        )

        selected: List[Any] = []
        current_cost = 0.0
        current_risk_reduction = 0.0

        for action in candidates:
            if current_cost + action.cost <= budget:
                selected.append(action)
                current_cost += action.cost
                current_risk_reduction += action.risk_reduction_usd

        utilization = (current_cost / budget * 100.0) if budget > 0 else 0.0
        overall_roi = (current_risk_reduction / current_cost) if current_cost > 0 else 0.0

        # Convert to RemediationResponse format if needed
        formatted_selected = [
            RemediationResponse.model_validate(a) if hasattr(a, "__table__") else a
            for a in selected
        ]

        return OptimizationResult(
            budget=round(budget, 2),
            total_cost_allocated=round(current_cost, 2),
            total_risk_reduction=round(current_risk_reduction, 2),
            budget_utilization_pct=round(utilization, 2),
            overall_roi=round(overall_roi, 2),
            strategy_used="greedy_roi",
            selected_actions=formatted_selected
        )

    @classmethod
    def optimize_pulp(
        cls,
        actions: List[Any],
        budget: float,
        min_roi_threshold: Optional[float] = None
    ) -> OptimizationResult:
        """
        Formal 0/1 Knapsack solution using PuLP (CBC Solver).
        
        Maximize: Σ (x_i * risk_reduction_i)
        Subject to: Σ (x_i * cost_i) <= budget
                    x_i ∈ {0, 1}
        """
        cls._validate_budget(budget)

        # Filter candidates
        candidates = [
            a for a in actions
            if getattr(a, "cost", 0.0) > 0 and getattr(a, "risk_reduction_usd", 0.0) > 0
        ]

        if min_roi_threshold is not None:
            candidates = [c for c in candidates if (c.risk_reduction_usd / c.cost) >= min_roi_threshold]

        if not candidates:
            return OptimizationResult(
                budget=round(budget, 2),
                total_cost_allocated=0.0,
                total_risk_reduction=0.0,
                budget_utilization_pct=0.0,
                overall_roi=0.0,
                strategy_used="pulp_knapsack_0_1",
                selected_actions=[]
            )

        # Create LP problem
        problem = pulp.LpProblem("CyberGuard_Investment_Optimization", pulp.LpMaximize)

        # Decision variables: binary x_i
        decision_vars = {
            i: problem.add_variable(f"x_{i}", cat=pulp.LpBinary) if hasattr(problem, "add_variable") else pulp.LpVariable(f"x_{i}", cat=pulp.LpBinary)
            for i in range(len(candidates))
        }

        # Objective: Maximize total risk reduction
        problem += pulp.lpSum([
            decision_vars[i] * candidates[i].risk_reduction_usd
            for i in range(len(candidates))
        ]), "Total_Risk_Reduction"

        # Constraint: Cost <= Budget
        problem += pulp.lpSum([
            decision_vars[i] * candidates[i].cost
            for i in range(len(candidates))
        ]) <= budget, "Budget_Constraint"

        # Solve silently
        solver = pulp.PULP_CBC_CMD(msg=False)
        problem.solve(solver)

        selected_indices = [
            i for i, var in decision_vars.items()
            if var.varValue is not None and var.varValue > 0.5
        ]

        # Deterministic sorting: sort selected actions by ROI descending
        selected = [candidates[i] for i in selected_indices]
        selected.sort(
            key=lambda x: (
                round(x.risk_reduction_usd / x.cost, 6),
                -x.cost,
                x.action_id
            ),
            reverse=True
        )

        total_cost = sum(a.cost for a in selected)
        total_risk_reduction = sum(a.risk_reduction_usd for a in selected)
        utilization = (total_cost / budget * 100.0) if budget > 0 else 0.0
        overall_roi = (total_risk_reduction / total_cost) if total_cost > 0 else 0.0

        formatted_selected = [
            RemediationResponse.model_validate(a) if hasattr(a, "__table__") else a
            for a in selected
        ]

        return OptimizationResult(
            budget=round(budget, 2),
            total_cost_allocated=round(total_cost, 2),
            total_risk_reduction=round(total_risk_reduction, 2),
            budget_utilization_pct=round(utilization, 2),
            overall_roi=round(overall_roi, 2),
            strategy_used="pulp_knapsack_0_1",
            selected_actions=formatted_selected
        )

    @classmethod
    def rank_candidate_controls(
        cls,
        actions: List[Any],
        selected_action_ids: Optional[Set[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Ranks candidate controls by (ΔALE reduction / cost) using existing risk register and control cost fields.
        Returns full list of controls with marginal utility (ΔALE/cost), cumulative metrics, and selection status.
        """
        selected_ids = selected_action_ids or set()
        candidates = [
            a for a in actions
            if getattr(a, "cost", 0.0) > 0 and getattr(a, "risk_reduction_usd", 0.0) > 0
        ]

        # Deterministic sort descending by marginal utility (delta_ale / cost)
        candidates.sort(
            key=lambda x: (
                round(x.risk_reduction_usd / x.cost, 6),
                -x.cost,
                x.action_id
            ),
            reverse=True
        )

        ranked: List[Dict[str, Any]] = []
        cum_cost = 0.0
        cum_reduction = 0.0

        for idx, act in enumerate(candidates, start=1):
            cum_cost += act.cost
            cum_reduction += act.risk_reduction_usd
            is_sel = act.action_id in selected_ids
            ratio = act.risk_reduction_usd / act.cost if act.cost > 0 else 0.0

            ranked.append({
                "rank": idx,
                "action_id": act.action_id,
                "finding_id": getattr(act, "finding_id", ""),
                "asset_id": getattr(act, "asset_id", ""),
                "title": act.title,
                "remediation_type": getattr(act, "remediation_type", "patch"),
                "cost": round(float(act.cost), 2),
                "risk_reduction_usd": round(float(act.risk_reduction_usd), 2),
                "delta_ale_cost_ratio": round(float(ratio), 2),
                "is_selected": is_sel,
                "cumulative_cost": round(cum_cost, 2),
                "cumulative_risk_reduction": round(cum_reduction, 2),
                "effort_hours": getattr(act, "effort_hours", 8.0),
                "nist_control_id": getattr(act, "nist_control_id", "PR.IP-12") or "PR.IP-12",
            })

        return ranked

    @classmethod
    def compute_pareto_frontier(
        cls,
        actions: List[Any],
        current_budget: float,
        num_points: int = 15
    ) -> List[Dict[str, Any]]:
        """
        Computes the Pareto efficient frontier curve (optimal risk reduction vs cost)
        across budget intervals using formal PuLP / greedy optimization.
        """
        valid = [
            a for a in actions
            if getattr(a, "cost", 0.0) > 0 and getattr(a, "risk_reduction_usd", 0.0) > 0
        ]
        if not valid:
            return []

        total_portfolio_cost = sum(a.cost for a in valid)
        max_budget = max(total_portfolio_cost * 1.05, current_budget * 1.5)

        budget_points = [0.0]
        step = max_budget / max(1, num_points)
        for i in range(1, num_points + 1):
            budget_points.append(round(step * i, 2))

        if current_budget > 0 and not any(abs(b - current_budget) < 1.0 for b in budget_points):
            budget_points.append(round(current_budget, 2))
            budget_points.sort()

        frontier: List[Dict[str, Any]] = []
        prev_cost = 0.0
        prev_reduction = 0.0

        for b in budget_points:
            if b <= 0:
                frontier.append({
                    "budget": 0.0,
                    "cost": 0.0,
                    "risk_reduction": 0.0,
                    "controls_selected": 0,
                    "overall_roi": 0.0,
                    "marginal_roi": 0.0,
                    "is_current_allocation": False,
                    "is_knee_point": False
                })
                continue

            opt = cls.optimize_pulp(valid, b)
            cost = opt.total_cost_allocated
            reduction = opt.total_risk_reduction

            delta_c = cost - prev_cost
            delta_r = reduction - prev_reduction
            marginal_roi = (delta_r / delta_c) if delta_c > 0 else 0.0

            is_curr = abs(b - current_budget) < 1.0

            frontier.append({
                "budget": round(b, 2),
                "cost": round(cost, 2),
                "risk_reduction": round(reduction, 2),
                "controls_selected": len(opt.selected_actions),
                "overall_roi": round(opt.overall_roi, 2),
                "marginal_roi": round(marginal_roi, 2),
                "is_current_allocation": is_curr,
                "is_knee_point": False
            })

            prev_cost = cost
            prev_reduction = reduction

        # Mark diminishing returns knee point
        if len(frontier) > 2:
            knee_idx = 1
            max_eff = -1.0
            for idx, pt in enumerate(frontier[1:], start=1):
                eff = pt["risk_reduction"] / (pt["cost"] + 1e-6)
                if eff > max_eff:
                    max_eff = eff
                if eff < max_eff * 0.55 and knee_idx == 1:
                    knee_idx = idx
            frontier[knee_idx]["is_knee_point"] = True

        return frontier

    @classmethod
    def sensitivity_analysis(
        cls,
        actions: List[Any],
        base_budget: float,
        increments: List[float] = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculates risk reduction across multiple budget thresholds to plot the ROI curve.
        """
        results: Dict[str, Dict[str, float]] = {}
        for inc in increments:
            test_budget = round(base_budget * inc, 2)
            res = cls.optimize_pulp(actions, test_budget)
            results[f"${test_budget:,.0f}"] = {
                "budget": test_budget,
                "cost_allocated": res.total_cost_allocated,
                "risk_reduction": res.total_risk_reduction,
                "actions_selected": len(res.selected_actions),
                "roi": res.overall_roi
            }
        return results


def optimize_investments(
    actions: List[Any],
    budget: float,
    strategy: str = "pulp",
    min_roi_threshold: Optional[float] = None
) -> OptimizationResult:
    """
    Public entry point for investment optimization.
    Returns optimal subset, ranked candidate controls, Pareto frontier, and provenance.
    """
    if strategy.lower() in ("pulp", "pulp_knapsack_0_1"):
        result = InvestmentOptimizer.optimize_pulp(actions, budget, min_roi_threshold)
    else:
        result = InvestmentOptimizer.optimize_greedy(actions, budget, min_roi_threshold)

    # Attach ranked candidate controls and Pareto frontier
    selected_ids = {a.action_id for a in result.selected_actions}
    result.ranked_controls = InvestmentOptimizer.rank_candidate_controls(actions, selected_ids)
    result.pareto_frontier = InvestmentOptimizer.compute_pareto_frontier(actions, current_budget=budget)
    result.provenance = {
        "model_used": "PuLP-CBC-2.10 (0/1 Knapsack Branch-and-Cut)" if "pulp" in strategy.lower() else "Greedy-Marginal-Utility-Ranker",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_summary": f"Budget: ${budget:,.2f}, Strategy: {strategy}, Candidates: {len(actions)}",
        "algorithm": "Integer Linear Programming (ILP) 0/1 Knapsack",
        "objective": "Maximize Risk Reduction (ΔALE) under Budget Constraint",
    }

    return result
