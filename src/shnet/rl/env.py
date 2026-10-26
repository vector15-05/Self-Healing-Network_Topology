from __future__ import annotations

from typing import Any

import networkx as nx

from shnet.sim.graph_sim import GraphSimulator
from shnet.topology.generators import generate_topology


class SelfHealingEnv:
    def __init__(self, graph: nx.Graph | None = None, horizon: int = 5) -> None:
        self.initial_graph = graph.copy() if graph is not None else generate_topology("ring", 8, seed=11)
        self.horizon = horizon
        self.simulator = GraphSimulator(self.initial_graph)
        self.timestep = 0

    def reset(self) -> dict[str, Any]:
        self.simulator = GraphSimulator(self.initial_graph)
        self.timestep = 0
        return self._observation()

    def _observation(self) -> dict[str, Any]:
        summary = self.simulator.build_flow_summary()
        return {
            "step": self.timestep,
            "nodes": list(self.simulator.graph.nodes),
            "edges": list(self.simulator.graph.edges),
            "connected_ratio": summary["connected_ratio"],
            "mean_latency_ms": summary["mean_latency_ms"],
            "throughput_mbps": summary["throughput_mbps"],
            "action_mask": self.simulator.action_mask(),
        }

    def step(self, action: str) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        packet = self.simulator.simulate_step(action)
        self.timestep += 1
        summary = packet["summary"]
        reward = float(packet["reward"]) - (0.2 * self.timestep)
        done = self.timestep >= self.horizon or summary["connected_ratio"] >= 0.98
        return self._observation(), reward, done, {"meta": packet}

    @property
    def action_space(self) -> list[str]:
        return self.simulator.action_mask()
