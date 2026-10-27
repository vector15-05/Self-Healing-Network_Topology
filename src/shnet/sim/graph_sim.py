from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, islice
from typing import Any

import networkx as nx

from shnet.sim.reward import RewardWeights, calculate_reward, is_recovered
from shnet.topology.generators import generate_topology


@dataclass(frozen=True)
class FlowDemand:
    source: Any
    target: Any
    demand_mbps: float = 5.0
    flow_id: str | None = None


class GraphSimulator:
    """Fast, deterministic capacity-aware graph simulator.

    Failed links and nodes remain in the physical topology with ``active=False``.
    This lets recovery actions restore them and keeps action identifiers stable.
    """

    def __init__(
        self,
        graph: nx.Graph | None = None,
        flows: list[FlowDemand] | None = None,
        *,
        max_paths: int = 3,
        recovery_target: dict[str, Any] | None = None,
        reward_weights: RewardWeights | None = None,
    ) -> None:
        if max_paths <= 0:
            raise ValueError("max_paths must be positive")
        self.graph = graph.copy() if graph is not None else generate_topology("barabasi_albert", 10, seed=7)
        for _, _, attrs in self.graph.edges(data=True):
            attrs.setdefault("active", True)
            attrs.setdefault("weight", float(attrs.get("cost", 1.0)))
            attrs.setdefault("utilization", 0.0)
        for _, attrs in self.graph.nodes(data=True):
            attrs.setdefault("active", True)
        self.edge_inventory = sorted(
            [(u, v) for u, v in self.graph.edges],
            key=lambda edge: (str(edge[0]), str(edge[1])),
        )
        self.edge_ids = {edge: idx for idx, edge in enumerate(self.edge_inventory)}
        self.node_inventory = list(self.graph.nodes)
        self.max_paths = max_paths
        self.flows = flows if flows is not None else self._default_flows()
        self.reward_weights = reward_weights or RewardWeights()
        self.risk_scores: dict[str, float] = {}
        self.history: list[dict[str, Any]] = []
        self.reconfigurations = 0
        self._last_summary: dict[str, Any] | None = None
        if recovery_target is None:
            current_graph = self.graph
            self.graph = current_graph.copy()
            nx.set_node_attributes(self.graph, True, "active")
            for _, _, attrs in self.graph.edges(data=True):
                attrs["active"] = True
            self.recovery_target = self.build_flow_summary()
            self.graph = current_graph
            self._last_summary = None
        else:
            self.recovery_target = recovery_target.copy()

    def _default_flows(self) -> list[FlowDemand]:
        nodes = [node for node, data in self.graph.nodes(data=True) if data.get("active", True)]
        if len(nodes) < 2:
            return []
        # A reproducible sparse workload keeps each simulator step cheap.
        pairs = list(combinations(nodes, 2))
        stride = max(1, len(pairs) // min(24, len(pairs)))
        selected = pairs[::stride][:24]
        return [
            FlowDemand(source, target, 5.0, f"flow-{index}")
            for index, (source, target) in enumerate(selected)
        ]

    def _active_graph(self) -> nx.Graph:
        active = nx.Graph()
        active.add_nodes_from(
            (node, attrs.copy())
            for node, attrs in self.graph.nodes(data=True)
            if attrs.get("active", True)
        )
        active.add_edges_from(
            (u, v, attrs.copy())
            for u, v, attrs in self.graph.edges(data=True)
            if attrs.get("active", True)
            and self.graph.nodes[u].get("active", True)
            and self.graph.nodes[v].get("active", True)
        )
        return active

    @staticmethod
    def _path_cost(_u: Any, _v: Any, attrs: dict[str, Any]) -> float:
        delay = max(0.01, float(attrs.get("delay_ms", 1.0)))
        loss = min(99.9, max(0.0, float(attrs.get("loss_pct", 0.0))))
        utilization = min(1.0, max(0.0, float(attrs.get("utilization", 0.0))))
        bandwidth = max(0.01, float(attrs.get("bandwidth_mbps", 1.0)))
        route_weight = max(0.0, float(attrs.get("weight", attrs.get("cost", 1.0))))
        return (
            delay * (1.0 + utilization * 4.0 + loss / 10.0)
            + 10.0 / bandwidth
            + 0.1 * route_weight
        )

    def _flow_paths(self, graph: nx.Graph, flow: FlowDemand) -> list[list[Any]]:
        if flow.source not in graph or flow.target not in graph:
            return []
        try:
            return [nx.shortest_path(graph, flow.source, flow.target, weight=self._path_cost)]
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    @staticmethod
    def _available_capacity(graph: nx.Graph, path: list[Any]) -> float:
        return min(
            (
                max(
                    0.0,
                    float(graph[u][v].get("bandwidth_mbps", 0.0))
                    * (1.0 - min(1.0, float(graph[u][v].get("utilization", 0.0)))),
                )
                for u, v in zip(path, path[1:])
            ),
            default=float("inf"),
        )

    def build_flow_summary(self) -> dict[str, Any]:
        for _, _, attrs in self.graph.edges(data=True):
            attrs["utilization"] = 0.0
        active = self._active_graph()

        flow_records: dict[str, dict[str, Any]] = {}
        latencies: list[float] = []
        delivered_total = 0.0
        demand_total = sum(max(0.0, flow.demand_mbps) for flow in self.flows)
        for index, flow in enumerate(self.flows):
            flow_id = flow.flow_id or f"{flow.source}->{flow.target}"
            chosen_path: list[Any] | None = None
            delivered = 0.0
            for path in self._flow_paths(active, flow):
                chosen_path = path
                available = self._available_capacity(active, path)
                if available < flow.demand_mbps and self.max_paths > 1:
                    try:
                        alternatives = nx.shortest_simple_paths(
                            active,
                            flow.source,
                            flow.target,
                            weight=self._path_cost,
                        )
                        for alternate in islice(alternatives, 1, self.max_paths):
                            alternate_capacity = self._available_capacity(active, alternate)
                            if alternate_capacity > available:
                                chosen_path = alternate
                                available = alternate_capacity
                            if available >= flow.demand_mbps:
                                break
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        pass
                if available > 0.0:
                    delivered = min(max(0.0, flow.demand_mbps), available)
                else:
                    chosen_path = None
                break

            if chosen_path is None:
                flow_records[flow_id] = {
                    "source": flow.source,
                    "target": flow.target,
                    "demand_mbps": flow.demand_mbps,
                    "delivered_mbps": 0.0,
                    "path": [],
                    "latency_ms": None,
                    "connected": False,
                }
                continue

            latency = 0.0
            path_loss_survival = 1.0
            for u, v in zip(chosen_path, chosen_path[1:]):
                attrs = active[u][v]
                capacity = max(0.01, float(attrs.get("bandwidth_mbps", 1.0)))
                attrs["utilization"] = min(
                    1.0,
                    float(attrs.get("utilization", 0.0)) + delivered / capacity,
                )
                self.graph[u][v]["utilization"] = attrs["utilization"]
                latency += float(attrs.get("delay_ms", 0.0))
                path_loss_survival *= 1.0 - min(1.0, float(attrs.get("loss_pct", 0.0)) / 100.0)
            delivered_total += delivered
            latencies.append(latency)
            flow_records[flow_id] = {
                "source": flow.source,
                "target": flow.target,
                "demand_mbps": flow.demand_mbps,
                "delivered_mbps": round(delivered, 3),
                "path": chosen_path,
                "latency_ms": round(latency, 3),
                "loss_pct": round((1.0 - path_loss_survival) * 100.0, 4),
                "connected": True,
            }

        connected_flows = sum(record["connected"] for record in flow_records.values())
        flow_count = len(flow_records)
        edge_count = sum(
            1 for _, _, attrs in self.graph.edges(data=True) if attrs.get("active", True)
        )
        active_nodes = sum(
            1 for _, attrs in self.graph.nodes(data=True) if attrs.get("active", True)
        )
        summary = {
            "num_flows": flow_count,
            "connected_flows": connected_flows,
            "connected_ratio": connected_flows / flow_count if flow_count else 1.0,
            "mean_latency_ms": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
            "throughput_mbps": round(delivered_total, 3),
            "demand_mbps": round(demand_total, 3),
            "throughput_ratio": delivered_total / demand_total if demand_total else 1.0,
            "mean_loss_pct": round(
                sum(float(record.get("loss_pct", 0.0)) for record in flow_records.values())
                / max(1, connected_flows),
                4,
            ),
            "active_nodes": active_nodes,
            "active_edges": edge_count,
            "flows": flow_records,
        }
        self._last_summary = summary
        return summary

    def connected_ratio(self) -> float:
        return float(self.build_flow_summary()["connected_ratio"])

    def action_mask(self) -> list[str]:
        actions = ["NO_OP"]
        for edge_id, (u, v) in enumerate(self.edge_inventory):
            if not self.graph.has_edge(u, v):
                continue
            attrs = self.graph[u][v]
            if not attrs.get("active", True):
                actions.append(f"ACTIVATE_BACKUP_LINK:{edge_id}")
        for node_id, node in enumerate(self.node_inventory):
            if not self.graph.nodes[node].get("active", True):
                actions.append(f"ACTIVATE_NODE:{node_id}")
        actions.append("REBALANCE_WEIGHTS")
        return actions

    def _resolve_edge_id(self, action: str) -> tuple[Any, Any] | None:
        try:
            edge_id = int(action.split(":", 1)[1])
        except (IndexError, ValueError):
            return None
        if not 0 <= edge_id < len(self.edge_inventory):
            return None
        return self.edge_inventory[edge_id]

    def apply_action(self, action: str) -> tuple[nx.Graph, dict[str, Any]]:
        if action not in self.action_mask():
            raise ValueError(f"Invalid or currently masked action: {action}")
        candidate = self.graph.copy()
        if action == "NO_OP":
            return candidate, {"cost": 0.0, "reason": "no intervention"}
        if action.startswith("ACTIVATE_BACKUP_LINK:"):
            edge = self._resolve_edge_id(action)
            if edge is None:
                raise ValueError(f"Invalid link action: {action}")
            u, v = edge
            candidate[u][v]["active"] = True
            candidate[u][v]["weight"] = float(candidate[u][v].get("cost", 1.0))
            return candidate, {"cost": 1.0, "reason": "link restored", "edge": edge}
        if action.startswith("ACTIVATE_NODE:"):
            try:
                node_id = int(action.split(":", 1)[1])
                node = self.node_inventory[node_id]
            except (IndexError, ValueError) as exc:
                raise ValueError(f"Invalid node action: {action}") from exc
            candidate.nodes[node]["active"] = True
            return candidate, {"cost": 1.0, "reason": "node restored", "node": node}
        if action.startswith("REPAIR:"):
            target = action.split(":", 1)[1]
            edge = next(
                ((u, v) for u, v in self.edge_inventory if f"{u}->{v}" == target or f"{v}->{u}" == target),
                None,
            )
            if edge is None:
                raise ValueError(f"Unknown link: {target}")
            u, v = edge
            candidate[u][v]["active"] = True
            return candidate, {"cost": 1.0, "reason": "link restored", "edge": edge}
        if action == "REBALANCE_WEIGHTS":
            for u, v, attrs in candidate.edges(data=True):
                if attrs.get("active", True):
                    risk = max(
                        0.0,
                        float(
                            self.risk_scores.get(
                                f"{u}->{v}",
                                self.risk_scores.get(f"{v}->{u}", 0.0),
                            )
                        ),
                    )
                    attrs["weight"] = max(
                        0.01,
                        float(attrs.get("delay_ms", 1.0))
                        * (1.0 + 4.0 * float(attrs.get("utilization", 0.0)) + 10.0 * risk),
                    )
            return candidate, {"cost": 1.0, "reason": "routing weights rebalanced"}
        raise ValueError(f"Unsupported action: {action}")

    def simulate_step(self, action: str = "NO_OP") -> dict[str, Any]:
        before = self.build_flow_summary()
        candidate, meta = self.apply_action(action)
        self.graph = candidate
        after = self.build_flow_summary()
        reward, restored = calculate_reward(
            after,
            reconfiguration_cost=float(meta["cost"]),
            timestep=len(self.history) + 1,
            recovery_target=self.recovery_target,
            weights=self.reward_weights,
        )
        was_recovered = is_recovered(before, self.recovery_target)
        if float(meta["cost"]) > 0:
            self.reconfigurations += 1
        packet = {
            "action": action,
            "reconfig_cost": float(meta["cost"]),
            "meta": meta,
            "before": before,
            "summary": after,
            "recovered": restored and not was_recovered,
            "reward": reward,
        }
        self.history.append(packet)
        return packet

    def counterfactual(self, action: str) -> dict[str, Any]:
        candidate, _ = self.apply_action(action)
        simulation = GraphSimulator(
            candidate,
            self.flows,
            max_paths=self.max_paths,
            recovery_target=self.recovery_target,
            reward_weights=self.reward_weights,
        )
        simulation.risk_scores = self.risk_scores.copy()
        return simulation.build_flow_summary()
