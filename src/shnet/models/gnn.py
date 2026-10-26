from __future__ import annotations

from typing import Any

import networkx as nx


class GraphFailurePredictor:
    def __init__(self, hidden_dim: int = 16, *, seed: int = 7) -> None:
        self.hidden_dim = hidden_dim
        self.seed = seed

    def encode(self, graph: nx.Graph) -> dict[str, Any]:
        node_embeddings: dict[Any, list[float]] = {}
        for node in graph.nodes:
            degree = float(graph.degree(node))
            node_embeddings[node] = [degree, float(graph.nodes[node].get("capacity", 0.0)), float(graph.nodes[node].get("load", 0.0))]

        edge_embeddings: dict[tuple[Any, Any], list[float]] = {}
        for u, v in graph.edges:
            edge_data = graph[u][v]
            edge_embeddings[(u, v)] = [
                float(edge_data.get("bandwidth_mbps", 0.0)),
                float(edge_data.get("delay_ms", 0.0)),
                float(edge_data.get("loss_pct", 0.0)),
                float(edge_data.get("utilization", 0.0)),
            ]
        return {"node_embeddings": node_embeddings, "edge_embeddings": edge_embeddings}

    def predict(self, graph: nx.Graph) -> dict[str, float]:
        scores: dict[str, float] = {}
        for u, v in graph.edges:
            edge_data = graph[u][v]
            risk = 0.1
            risk += float(edge_data.get("loss_pct", 0.0)) / 20.0
            risk += float(edge_data.get("utilization", 0.0)) * 0.6
            risk += float(edge_data.get("delay_ms", 0.0)) / 50.0
            scores[f"{u}->{v}"] = min(1.0, round(risk, 4))
        return scores

    def fit(self, graphs: list[nx.Graph]) -> "GraphFailurePredictor":
        self._graph_count = len(graphs)
        return self
