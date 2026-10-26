from __future__ import annotations

from typing import Any

import networkx as nx

from shnet.topology.generators import generate_topology


class GraphSimulator:
    def __init__(self, graph: nx.Graph | None = None):
        self.graph = graph.copy() if graph is not None else generate_topology("barabasi_albert", 10, seed=7)
        self.history: list[dict[str, Any]] = []

    def shortest_path(self, source: Any, target: Any) -> list[Any]:
        try:
            return nx.shortest_path(self.graph, source, target, weight="delay_ms")
        except nx.NetworkXNoPath:
            return []

    def connected_ratio(self) -> float:
        if self.graph.number_of_nodes() == 0:
            return 0.0
        components = list(nx.connected_components(self.graph))
        if not components:
            return 0.0
        largest_component = max(components, key=len)
        return len(largest_component) / self.graph.number_of_nodes()

    def action_mask(self) -> list[str]:
        valid_actions = ["NO_OP"]
        for u, v in self.graph.edges:
            valid_actions.append(f"REPAIR:{u}->{v}")
        return valid_actions

    def apply_action(self, action: str) -> tuple[nx.Graph, dict[str, Any]]:
        candidate = self.graph.copy()
        if action == "NO_OP":
            return candidate, {"cost": 0.0, "reason": "no intervention"}
        if not action.startswith("REPAIR:"):
            return candidate, {"cost": 0.0, "reason": "ignored"}

        target = action.split(":", 1)[1]
        if "->" not in target:
            return candidate, {"cost": 0.0, "reason": "invalid target"}

        u_str, v_str = target.split("->", 1)
        if candidate.has_edge(u_str, v_str):
            candidate[u_str][v_str]["loss_pct"] = max(0.0, candidate[u_str][v_str].get("loss_pct", 0.0) - 0.5)
            candidate[u_str][v_str]["delay_ms"] = max(0.1, candidate[u_str][v_str].get("delay_ms", 0.1) - 2.0)
            candidate[u_str][v_str]["utilization"] = max(0.0, candidate[u_str][v_str].get("utilization", 0.0) - 0.15)
            return candidate, {"cost": 1.0, "reason": "repair"}
        return candidate, {"cost": 0.0, "reason": "invalid edge"}

    def build_flow_summary(self) -> dict[str, Any]:
        flows: dict[str, float] = {}
        latencies: list[float] = []
        for source in self.graph.nodes:
            for target in self.graph.nodes:
                if source == target:
                    continue
                path = self.shortest_path(source, target)
                if not path:
                    continue
                route_latency = sum(float(self.graph[u][v].get("delay_ms", 0.0)) for u, v in zip(path, path[1:]))
                flows[f"{source}->{target}"] = round(route_latency, 2)
                latencies.append(route_latency)

        mean_latency = sum(latencies) / len(latencies) if latencies else 0.0
        throughput = sum(float(self.graph[u][v].get("bandwidth_mbps", 0.0)) for u, v in self.graph.edges) / max(1, self.graph.number_of_edges())
        return {
            "num_flows": len(flows),
            "mean_latency_ms": round(mean_latency, 2),
            "throughput_mbps": round(throughput, 2),
            "connected_ratio": self.connected_ratio(),
            "flows": flows,
        }

    def simulate_step(self, action: str = "NO_OP") -> dict[str, Any]:
        candidate, meta = self.apply_action(action)
        self.graph = candidate
        summary = self.build_flow_summary()
        reward = (summary["connected_ratio"] * 10.0) - (summary["mean_latency_ms"] * 0.05) - (meta["cost"] * 2.0)
        packet = {
            "action": action,
            "reconfig_cost": meta["cost"],
            "summary": summary,
            "reward": round(reward, 2),
        }
        self.history.append(packet)
        return packet
