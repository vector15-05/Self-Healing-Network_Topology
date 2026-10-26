from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import networkx as nx


@dataclass
class GraphSnapshot:
    timestamp: float
    node_names: list[str]
    node_features: list[list[float]] = field(default_factory=list)
    edge_index: list[list[int]] = field(default_factory=list)
    edge_features: list[list[float]] = field(default_factory=list)
    flow_summary: dict[str, Any] = field(default_factory=dict)


def _node_feature_vector(graph: nx.Graph, node: Any) -> list[float]:
    degree = graph.degree(node)
    node_data = graph.nodes[node]
    return [
        float(degree),
        float(node_data.get("capacity", 0.0)),
        float(node_data.get("load", 0.0)),
        float(node_data.get("type") == "host"),
    ]


def _edge_feature_vector(graph: nx.Graph, u: Any, v: Any) -> list[float]:
    edge_data = graph[u][v]
    return [
        float(edge_data.get("bandwidth_mbps", 0.0)),
        float(edge_data.get("delay_ms", 0.0)),
        float(edge_data.get("loss_pct", 0.0)),
        float(edge_data.get("utilization", 0.0)),
        float(edge_data.get("cost", 0.0)),
    ]


def graph_to_snapshot(graph: nx.Graph, timestamp: float = 0.0) -> GraphSnapshot:
    node_names = sorted(graph.nodes, key=lambda item: str(item))
    node_features = [_node_feature_vector(graph, node) for node in node_names]
    edge_index: list[list[int]] = []
    edge_features: list[list[float]] = []
    for u, v in graph.edges:
        u_index = node_names.index(u)
        v_index = node_names.index(v)
        edge_index.append([u_index, v_index])
        edge_features.append(_edge_feature_vector(graph, u, v))
    return GraphSnapshot(
        timestamp=timestamp,
        node_names=[str(node) for node in node_names],
        node_features=node_features,
        edge_index=edge_index,
        edge_features=edge_features,
        flow_summary={
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "mean_utilization": sum(float(graph[u][v].get("utilization", 0.0)) for u, v in graph.edges) / max(1, graph.number_of_edges()),
        },
    )
