from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

import networkx as nx


def _apply_edge_attributes(graph: nx.Graph, min_bandwidth: float, max_bandwidth: float, min_delay: float, max_delay: float, min_loss: float, max_loss: float, min_capacity: float, max_capacity: float, seed: int) -> nx.Graph:
    rng = random.Random(seed)
    for node in graph.nodes:
        graph.nodes[node].setdefault("type", "switch")
        graph.nodes[node].setdefault("capacity", rng.uniform(min_capacity, max_capacity))
    for u, v in graph.edges:
        graph[u][v].update(
            {
                "bandwidth_mbps": round(rng.uniform(min_bandwidth, max_bandwidth), 2),
                "delay_ms": round(rng.uniform(min_delay, max_delay), 2),
                "loss_pct": round(rng.uniform(min_loss, max_loss), 2),
                "utilization": 0.0,
                "cost": round(rng.uniform(1.0, 5.0), 2),
            }
        )
    return graph


def _ensure_connected(graph: nx.Graph) -> nx.Graph:
    if nx.is_connected(graph):
        return graph
    components = list(nx.connected_components(graph))
    for idx in range(1, len(components)):
        source = next(iter(components[idx - 1]))
        target = next(iter(components[idx]))
        graph.add_edge(source, target, bandwidth_mbps=100.0, delay_ms=10.0, loss_pct=0.5, utilization=0.0, cost=1.0)
    return graph


def generate_topology(
    kind: str,
    num_nodes: int = 10,
    *,
    seed: int = 7,
    min_bandwidth: float = 20.0,
    max_bandwidth: float = 120.0,
    min_delay: float = 2.0,
    max_delay: float = 12.0,
    min_loss: float = 0.1,
    max_loss: float = 1.5,
    min_capacity: float = 100.0,
    max_capacity: float = 500.0,
) -> nx.Graph:
    kind = kind.lower()
    if kind == "barabasi_albert":
        graph = nx.barabasi_albert_graph(num_nodes, min(3, max(2, num_nodes // 5)), seed=seed)
    elif kind == "erdos_renyi":
        probability = 0.25 if num_nodes <= 20 else 0.15
        graph = nx.erdos_renyi_graph(num_nodes, probability, seed=seed)
        if not nx.is_connected(graph):
            graph = nx.gnp_random_graph(num_nodes, 0.5, seed=seed)
    elif kind == "ring":
        graph = nx.cycle_graph(num_nodes)
    elif kind == "grid":
        side = int(round(num_nodes ** 0.5))
        if side * side < num_nodes:
            side += 1
        graph = nx.grid_2d_graph(side, side)
        graph = nx.convert_node_labels_to_integers(graph)
    elif kind == "waxman":
        graph = nx.waxman_graph(num_nodes, alpha=0.4, beta=0.5, seed=seed)
    elif kind == "fat_tree":
        pod_count = max(2, min(4, num_nodes // 4))
        graph = nx.Graph()
        for pod in range(pod_count):
            for leaf in range(2):
                node = f"pod{pod}-leaf{leaf}"
                graph.add_node(node, type="switch", capacity=200.0)
            for edge in range(2):
                node = f"core{pod}-{edge}"
                graph.add_node(node, type="switch", capacity=250.0)
        graph.add_edges_from([("pod0-leaf0", "pod0-leaf1"), ("pod0-leaf0", "core0-0"), ("pod0-leaf1", "core0-1")])
        for pod in range(1, pod_count):
            graph.add_edge(f"pod{pod}-leaf0", f"core{pod % 2}-0")
            graph.add_edge(f"pod{pod}-leaf1", f"core{pod % 2}-1")
    else:
        raise ValueError(f"Unsupported topology kind: {kind}")

    if isinstance(graph, nx.Graph):
        graph = _ensure_connected(graph)
        graph = _apply_edge_attributes(graph, min_bandwidth, max_bandwidth, min_delay, max_delay, min_loss, max_loss, min_capacity, max_capacity, seed)
        for node in graph.nodes:
            graph.nodes[node].setdefault("type", "switch")
            graph.nodes[node].setdefault("capacity", max_capacity)
        return graph
    raise TypeError("Expected a NetworkX graph")


def generate_train_test_sets(num_train: int = 3, num_test: int = 2, *, seed: int = 7) -> tuple[list[nx.Graph], list[nx.Graph]]:
    train_graphs = [generate_topology("barabasi_albert", 12 + idx * 4, seed=seed + idx) for idx in range(num_train)]
    test_graphs = [generate_topology("erdos_renyi", 15 + idx * 5, seed=seed + 100 + idx) for idx in range(num_test)]
    return train_graphs, test_graphs


def load_graphml(path: str | Path) -> nx.Graph:
    path = Path(path)
    if path.suffix == ".gml":
        return nx.read_gml(path)
    if path.suffix == ".graphml":
        return nx.read_graphml(path)
    raise ValueError(f"Unsupported graph format for {path}")
