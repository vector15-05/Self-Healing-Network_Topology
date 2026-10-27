from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import networkx as nx

SCENARIO_KINDS = (
    "hard_link_failure",
    "node_failure",
    "gradual_degradation",
    "congestion",
    "cascading",
)


@dataclass
class FailureScenario:
    name: str
    kind: str
    target: Any | None = None
    severity: float = 1.0
    timestamp: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


def build_failure_schedule(
    num_scenarios: int = 5,
    *,
    seed: int = 7,
    rates: dict[str, float] | None = None,
    interval_seconds: float = 1.0,
) -> list[FailureScenario]:
    if num_scenarios < 0:
        raise ValueError("num_scenarios must be non-negative")
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")
    rng = random.Random(seed)
    if rates is not None:
        unknown = set(rates) - set(SCENARIO_KINDS)
        if unknown:
            raise ValueError(f"Unknown failure kinds in rates: {sorted(unknown)}")
        if not rates or any(weight < 0 for weight in rates.values()) or sum(rates.values()) <= 0:
            raise ValueError("Failure rates must contain non-negative weights with a positive total")
        choices = list(rates)
        weights = [rates[kind] for kind in choices]
    else:
        choices = list(SCENARIO_KINDS)
        weights = [1.0] * len(choices)
    schedule: list[FailureScenario] = []
    for idx in range(num_scenarios):
        kind = rng.choices(choices, weights=weights, k=1)[0]
        schedule.append(
            FailureScenario(
                name=f"{kind}_{idx}",
                kind=kind,
                severity=round(rng.uniform(0.6, 1.3), 2),
                timestamp=idx * interval_seconds,
                metadata={"seed": seed + idx},
            )
        )
    return schedule


def _choose_edge(graph: nx.Graph, scenario: FailureScenario) -> tuple[Any, Any] | None:
    if scenario.target is not None:
        if not isinstance(scenario.target, (tuple, list)) or len(scenario.target) != 2:
            raise ValueError("A link failure target must be a two-node tuple or list")
        u, v = scenario.target
        if not graph.has_edge(u, v):
            raise ValueError(f"Failure target edge does not exist: {(u, v)!r}")
        return u, v
    edges = list(graph.edges)
    if not edges:
        return None
    rng = random.Random(int(scenario.metadata.get("seed", 0)))
    return edges[rng.randrange(len(edges))]


def inject_failure(graph: nx.Graph, scenario: FailureScenario) -> nx.Graph:
    """Apply a deterministic failure while retaining the physical topology."""
    if scenario.kind not in SCENARIO_KINDS:
        raise ValueError(f"Unsupported failure scenario kind: {scenario.kind}")
    if scenario.severity < 0:
        raise ValueError("Failure severity must be non-negative")

    cloned = graph.copy()
    if scenario.kind in {"hard_link_failure", "cascading"}:
        edge = _choose_edge(cloned, scenario)
        if edge is None:
            return cloned
        u, v = edge
        cloned[u][v]["active"] = False
        cloned[u][v]["failure_kind"] = scenario.kind
        cloned[u][v]["failure_timestamp"] = scenario.timestamp
        if scenario.kind == "cascading":
            # Model the shifted load on the remaining links incident to the endpoints.
            for endpoint in edge:
                for neighbor in cloned.neighbors(endpoint):
                    if neighbor == (v if endpoint == u else u):
                        continue
                    attrs = cloned[endpoint][neighbor]
                    attrs["utilization"] = min(
                        1.0,
                        float(attrs.get("utilization", 0.0)) + 0.25 * scenario.severity,
                    )
        return cloned

    if scenario.kind == "node_failure":
        target = scenario.target
        if target is None:
            nodes = list(cloned.nodes)
            if not nodes:
                return cloned
            rng = random.Random(int(scenario.metadata.get("seed", 0)))
            target = nodes[rng.randrange(len(nodes))]
        if target not in cloned:
            raise ValueError(f"Failure target node does not exist: {target!r}")
        cloned.nodes[target]["active"] = False
        cloned.nodes[target]["failure_kind"] = scenario.kind
        cloned.nodes[target]["failure_timestamp"] = scenario.timestamp
        return cloned

    if scenario.kind == "gradual_degradation":
        for _, _, attrs in cloned.edges(data=True):
            attrs["delay_ms"] = float(attrs.get("delay_ms", 0.0)) + 5.0 * scenario.severity
            attrs["loss_pct"] = min(
                100.0,
                float(attrs.get("loss_pct", 0.0)) + 0.8 * scenario.severity,
            )
            attrs["failure_kind"] = scenario.kind
            attrs["failure_timestamp"] = scenario.timestamp
        return cloned

    if scenario.kind == "congestion":
        for _, _, attrs in cloned.edges(data=True):
            attrs["utilization"] = min(
                1.0,
                float(attrs.get("utilization", 0.0)) + 0.25 * scenario.severity,
            )
            attrs["failure_kind"] = scenario.kind
            attrs["failure_timestamp"] = scenario.timestamp
        return cloned

    raise AssertionError("Scenario kind validation should make this unreachable")


def failure_labels(
    graph: nx.Graph,
    failure_scenarios: list[FailureScenario],
    *,
    horizon_seconds: float = 0.0,
    timestamp: float = 0.0,
) -> dict[str, float]:
    """Label existing links that fail in the requested future horizon."""
    if horizon_seconds < 0:
        raise ValueError("horizon_seconds must be non-negative")
    labels = {
        f"{u}->{v}": 0.0
        for u, v in graph.edges
    }
    for failure in failure_scenarios:
        if failure.kind not in {"hard_link_failure", "cascading", "node_failure"}:
            continue
        occurs_in_horizon = timestamp <= failure.timestamp <= timestamp + horizon_seconds
        if not occurs_in_horizon:
            continue
        if failure.kind == "node_failure":
            target_node = failure.target
            if target_node is None:
                nodes = list(graph.nodes)
                if nodes:
                    rng = random.Random(int(failure.metadata.get("seed", 0)))
                    target_node = nodes[rng.randrange(len(nodes))]
            for u, v in graph.edges:
                if u == target_node or v == target_node:
                    labels[f"{u}->{v}"] = 1.0
            continue
        target = failure.target or _choose_edge(graph, failure)
        if target is None:
            continue
        u, v = target
        if graph.has_edge(u, v):
            labels[f"{u}->{v}"] = 1.0
        elif graph.has_edge(v, u):
            labels[f"{v}->{u}"] = 1.0
    return labels
