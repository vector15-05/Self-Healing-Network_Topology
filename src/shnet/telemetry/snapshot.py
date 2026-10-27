from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import Event
from typing import Any, Callable, Iterator

import networkx as nx


@dataclass
class GraphSnapshot:
    timestamp: float
    node_names: list[str]
    node_features: list[list[float]] = field(default_factory=list)
    edge_index: list[list[int]] = field(default_factory=list)
    edge_features: list[list[float]] = field(default_factory=list)
    flow_summary: dict[str, Any] = field(default_factory=dict)
    labels: dict[str, float] = field(default_factory=dict)


def _node_feature_vector(graph: nx.Graph, node: Any) -> list[float]:
    node_data = graph.nodes[node]
    return [
        float(graph.degree(node)),
        float(node_data.get("capacity", 0.0)),
        float(node_data.get("load", 0.0)),
        float(node_data.get("type") == "host"),
        float(node_data.get("active", True)),
        float(node_data.get("flow_count", 0.0)),
    ]


def _edge_feature_vector(graph: nx.Graph, u: Any, v: Any) -> list[float]:
    edge_data = graph[u][v]
    return [
        float(edge_data.get("bandwidth_mbps", 0.0)),
        float(edge_data.get("delay_ms", 0.0)),
        float(edge_data.get("loss_pct", 0.0)),
        float(edge_data.get("utilization", 0.0)),
        float(edge_data.get("cost", 0.0)),
        float(edge_data.get("active", True)),
    ]


def graph_to_snapshot(
    graph: nx.Graph,
    timestamp: float | None = None,
    *,
    flow_summary: dict[str, Any] | None = None,
    labels: dict[str, float] | None = None,
) -> GraphSnapshot:
    node_names = sorted(graph.nodes, key=lambda item: str(item))
    node_index = {node: index for index, node in enumerate(node_names)}
    edge_index: list[list[int]] = []
    edge_features: list[list[float]] = []
    for u, v in graph.edges:
        edge_index.append([node_index[u], node_index[v]])
        edge_features.append(_edge_feature_vector(graph, u, v))

    active_edges = [
        (u, v) for u, v, attrs in graph.edges(data=True) if attrs.get("active", True)
    ]
    mean_utilization = (
        sum(float(graph[u][v].get("utilization", 0.0)) for u, v in active_edges)
        / len(active_edges)
        if active_edges
        else 0.0
    )
    return GraphSnapshot(
        timestamp=time.time() if timestamp is None else float(timestamp),
        node_names=[str(node) for node in node_names],
        node_features=[_node_feature_vector(graph, node) for node in node_names],
        edge_index=edge_index,
        edge_features=edge_features,
        flow_summary=flow_summary
        or {
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "active_edges": len(active_edges),
            "mean_utilization": mean_utilization,
        },
        labels=labels or {},
    )


class SnapshotStore:
    """Append-only JSONL storage for reproducible graph time series."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, snapshot: GraphSnapshot) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(snapshot), separators=(",", ":"), default=str))
            handle.write("\n")

    def read_all(self) -> list[GraphSnapshot]:
        if not self.path.exists():
            return []
        snapshots: list[GraphSnapshot] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid snapshot JSON on line {line_number}") from exc
                snapshots.append(GraphSnapshot(**row))
        return snapshots


class TelemetryCollector:
    def __init__(self, store: SnapshotStore | None = None) -> None:
        self.store = store

    def collect(
        self,
        graph: nx.Graph,
        *,
        timestamp: float | None = None,
        flow_summary: dict[str, Any] | None = None,
        labels: dict[str, float] | None = None,
    ) -> GraphSnapshot:
        snapshot = graph_to_snapshot(
            graph,
            timestamp,
            flow_summary=flow_summary,
            labels=labels,
        )
        if self.store is not None:
            self.store.append(snapshot)
        return snapshot

    def collect_periodically(
        self,
        graph_provider: Callable[[], nx.Graph],
        *,
        interval_seconds: float = 1.0,
        stop_event: Event,
        max_ticks: int | None = None,
    ) -> Iterator[GraphSnapshot]:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        if max_ticks is not None and max_ticks <= 0:
            raise ValueError("max_ticks must be positive when provided")
        ticks = 0
        deadline = time.monotonic()
        while not stop_event.is_set() and (max_ticks is None or ticks < max_ticks):
            now = time.time()
            yield self.collect(graph_provider(), timestamp=now)
            ticks += 1
            if max_ticks is not None and ticks >= max_ticks:
                break
            deadline += interval_seconds
            stop_event.wait(max(0.0, deadline - time.monotonic()))
