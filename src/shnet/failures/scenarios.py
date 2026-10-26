from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import networkx as nx


@dataclass
class FailureScenario:
    name: str
    kind: str
    target: Any | None = None
    severity: float = 1.0
    timestamp: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


def build_failure_schedule(num_scenarios: int = 5, *, seed: int = 7) -> list[FailureScenario]:
    scenario_pool = [
        FailureScenario("hard_link_failure", "hard_link_failure", None, 1.0, 0.0),
        FailureScenario("node_failure", "node_failure", None, 1.0, 1.0),
        FailureScenario("gradual_degradation", "gradual_degradation", None, 0.6, 2.0),
        FailureScenario("congestion", "congestion", None, 0.8, 3.0),
        FailureScenario("cascading", "cascading", None, 1.2, 4.0),
    ]
    schedule: list[FailureScenario] = []
    for idx in range(num_scenarios):
        base = scenario_pool[idx % len(scenario_pool)]
        schedule.append(
            FailureScenario(
                name=f"{base.name}_{idx}",
                kind=base.kind,
                target=base.target,
                severity=round(base.severity + (idx * 0.15), 2),
                timestamp=float(base.timestamp + idx),
                metadata={"seed": seed + idx},
            )
        )
    return schedule


def inject_failure(graph: nx.Graph, scenario: FailureScenario) -> nx.Graph:
    cloned = graph.copy()
    if scenario.kind == "hard_link_failure":
        if scenario.target is None:
            edges = list(cloned.edges)
            if not edges:
                return cloned
            target = edges[0]
        else:
            target = scenario.target
        if target in cloned.edges:
            cloned.remove_edge(*target)
        elif len(target) == 2 and target[0] in cloned and target[1] in cloned:
            cloned.remove_edge(*target)
        return cloned

    if scenario.kind == "node_failure":
        if scenario.target is None:
            targets = list(cloned.nodes)
            if not targets:
                return cloned
            target = targets[0]
        else:
            target = scenario.target
        if target in cloned:
            cloned.remove_node(target)
        return cloned

    if scenario.kind == "gradual_degradation":
        for _, _, edge_data in cloned.edges(data=True):
            edge_data["delay_ms"] = float(edge_data.get("delay_ms", 0.0)) + 5.0 * scenario.severity
            edge_data["loss_pct"] = float(edge_data.get("loss_pct", 0.0)) + 0.8 * scenario.severity
        return cloned

    if scenario.kind == "congestion":
        for _, _, edge_data in cloned.edges(data=True):
            edge_data["utilization"] = min(1.0, float(edge_data.get("utilization", 0.0)) + 0.25 * scenario.severity)
            edge_data["bandwidth_mbps"] = max(10.0, float(edge_data.get("bandwidth_mbps", 100.0)) * (1.0 - 0.1 * scenario.severity))
        return cloned

    if scenario.kind == "cascading":
        for edge in list(cloned.edges):
            u, v = edge
            cloned[u][v]["utilization"] = min(1.0, cloned[u][v].get("utilization", 0.0) + 0.2)
        if cloned.number_of_edges() > 0:
            u, v = next(iter(cloned.edges))
            cloned.remove_edge(u, v)
        return cloned

    raise ValueError(f"Unsupported failure scenario kind: {scenario.kind}")


def failure_labels(graph: nx.Graph, failure_scenarios: list[FailureScenario]) -> dict[str, float]:
    labels: dict[str, float] = {}
    for edge in graph.edges:
        labels[f"{edge[0]}->{edge[1]}"] = 0.0
    for failure in failure_scenarios:
        if failure.kind in {"hard_link_failure", "cascading"}:
            target = failure.target
            if target is None and graph.number_of_edges() > 0:
                target = next(iter(graph.edges))
            if target is not None:
                key = f"{target[0]}->{target[1]}"
                labels[key] = 1.0
    return labels
