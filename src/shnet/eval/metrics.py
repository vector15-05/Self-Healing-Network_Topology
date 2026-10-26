from __future__ import annotations

from typing import Any

from shnet.baselines.strategies import NoOpBaseline, ReactiveShortestPathBaseline
from shnet.failures.scenarios import FailureScenario, inject_failure
from shnet.sim.graph_sim import GraphSimulator
from shnet.topology.generators import generate_topology


def mttr(historical_rewards: list[float]) -> float:
    positive_steps = [step for step in historical_rewards if step > 0.0]
    return sum(positive_steps) / max(1, len(positive_steps)) if positive_steps else 0.0


def compute_experiment_metrics(strategy_name: str, steps: list[dict[str, Any]]) -> dict[str, float]:
    costs = [float(step.get("reconfig_cost", 0.0)) for step in steps]
    rewards = [float(step.get("reward", 0.0)) for step in steps]
    mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
    total_cost = sum(costs)
    return {
        "strategy": strategy_name,
        "mean_reward": round(mean_reward, 2),
        "total_cost": round(total_cost, 2),
        "mttr": round(mttr(rewards), 2),
        "steps": len(steps),
    }


def run_policy_experiment(strategy_name: str = "REACTIVE", *, topology_kind: str = "ring", seed: int = 7) -> dict[str, Any]:
    graph = generate_topology(topology_kind, 12, seed=seed)
    simulation = GraphSimulator(graph)
    scenario = FailureScenario("failure", "hard_link_failure", target=(0, 1), severity=1.1, timestamp=0.0)
    simulation.graph = inject_failure(simulation.graph, scenario)

    strategy = {
        "NO_OP": NoOpBaseline(),
        "REACTIVE": ReactiveShortestPathBaseline(),
    }.get(strategy_name, NoOpBaseline())

    steps: list[dict[str, Any]] = []
    for _ in range(3):
        action = strategy.choose_action(simulation)
        step = simulation.simulate_step(action)
        steps.append(step)
    return compute_experiment_metrics(strategy_name, steps)
