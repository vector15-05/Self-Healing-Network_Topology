from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:
    raise ImportError(
        "SelfHealingEnv requires Gymnasium. Install the project with `pip install -e '.[rl]'`."
    ) from exc

from shnet.sim.graph_sim import GraphSimulator
from shnet.sim.reward import RewardWeights, is_recovered
from shnet.topology.generators import generate_topology


class SelfHealingEnv(gym.Env[dict[str, np.ndarray], int]):
    """Gymnasium environment with a fixed, topology-size-agnostic graph observation."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        graph: nx.Graph | None = None,
        horizon: int = 5,
        *,
        failure_scenario: Any | None = None,
        randomize_failure: bool = False,
        flows: list[Any] | None = None,
        max_nodes: int = 50,
        max_edges: int = 512,
        max_flows: int = 128,
        reward_weights: RewardWeights | None = None,
    ) -> None:
        super().__init__()
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        self.initial_graph = graph.copy() if graph is not None else generate_topology("ring", 8, seed=11)
        self.horizon = horizon
        self.failure_scenario = failure_scenario
        self.randomize_failure = randomize_failure
        self.flows = flows
        self.reward_weights = reward_weights or RewardWeights()
        if max_nodes <= 0 or max_edges <= 0 or max_flows <= 0:
            raise ValueError("max_nodes, max_edges, and max_flows must be positive")
        self.simulator = GraphSimulator(
            self.initial_graph,
            flows,
            reward_weights=self.reward_weights,
        )
        self.flows = self.simulator.flows
        self.recovery_target = self.simulator.build_flow_summary()
        self.timestep = 0
        self._nodes = list(self.simulator.graph.nodes)
        self._node_index = {node: index for index, node in enumerate(self._nodes)}
        self._edges = list(self.simulator.edge_inventory)
        self._edge_index = {edge: index for index, edge in enumerate(self._edges)}
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.max_flows = max_flows
        self._check_topology_capacity()
        self.action_space = spaces.Discrete(max_edges + max_nodes + 2)
        self.observation_space = spaces.Dict(
            {
                "node_features": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(max_nodes, 6), dtype=np.float32
                ),
                "node_mask": spaces.Box(low=0.0, high=1.0, shape=(max_nodes,), dtype=np.float32),
                "edge_index": spaces.Box(
                    low=0, high=max_nodes - 1, shape=(2, max_edges), dtype=np.int64
                ),
                "edge_features": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(max_edges, 6), dtype=np.float32
                ),
                "edge_mask": spaces.Box(low=0.0, high=1.0, shape=(max_edges,), dtype=np.float32),
                "flow_features": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(max_flows, 5), dtype=np.float32
                ),
                "flow_mask": spaces.Box(low=0.0, high=1.0, shape=(max_flows,), dtype=np.float32),
                "metrics": spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
            }
        )

    def _check_topology_capacity(self) -> None:
        if len(self._nodes) > self.max_nodes or len(self._edges) > self.max_edges:
            raise ValueError(
                f"Graph has {len(self._nodes)} nodes and {len(self._edges)} edges; "
                f"observation capacity is {self.max_nodes} nodes and {self.max_edges} edges"
            )
        if len(self.simulator.flows) > self.max_flows:
            raise ValueError(
                f"Graph has {len(self.simulator.flows)} flows; observation capacity is {self.max_flows}"
            )

    def _make_observation(self) -> dict[str, np.ndarray]:
        graph = self.simulator.graph
        summary = self.simulator.build_flow_summary()
        self._check_topology_capacity()
        node_features = np.zeros((self.max_nodes, 6), dtype=np.float32)
        for index, node in enumerate(self._nodes):
            node_features[index] = [
                float(graph.degree(node)),
                float(graph.nodes[node].get("capacity", 0.0)),
                float(graph.nodes[node].get("load", 0.0)),
                float(graph.nodes[node].get("type") == "host"),
                float(graph.nodes[node].get("active", True)),
                float(graph.nodes[node].get("flow_count", 0.0)),
            ]
        node_mask = np.zeros(self.max_nodes, dtype=np.float32)
        node_mask[: len(self._nodes)] = 1.0
        edge_index = np.zeros((2, self.max_edges), dtype=np.int64)
        edge_features = np.zeros((self.max_edges, 6), dtype=np.float32)
        edge_mask = np.zeros(self.max_edges, dtype=np.float32)
        for index, (u, v) in enumerate(self._edges):
            edge_index[:, index] = [self._node_index[u], self._node_index[v]]
            edge_features[index] = [
                float(graph[u][v].get("bandwidth_mbps", 0.0)),
                float(graph[u][v].get("delay_ms", 0.0)),
                float(graph[u][v].get("loss_pct", 0.0)),
                float(graph[u][v].get("utilization", 0.0)),
                float(graph[u][v].get("cost", 0.0)),
                float(graph[u][v].get("active", True)),
            ]
            edge_mask[index] = 1.0
        flow_features = np.zeros((self.max_flows, 5), dtype=np.float32)
        flow_mask = np.zeros(self.max_flows, dtype=np.float32)
        for index, flow in enumerate(self.simulator.flows):
            record = summary["flows"].get(flow.flow_id or f"{flow.source}->{flow.target}", {})
            path = record.get("path", [])
            flow_features[index] = [
                float(self._node_index.get(flow.source, -1)),
                float(self._node_index.get(flow.target, -1)),
                float(flow.demand_mbps),
                float(record.get("delivered_mbps", 0.0)),
                float(len(path) - 1),
            ]
            flow_mask[index] = 1.0
        metrics = np.asarray(
            [
                float(summary["connected_ratio"]),
                float(summary["throughput_ratio"]),
                float(summary["mean_latency_ms"]),
                float(summary["mean_loss_pct"]),
                float(summary["throughput_mbps"]),
                float(self.timestep),
            ],
            dtype=np.float32,
        )
        return {
            "node_features": node_features,
            "node_mask": node_mask,
            "edge_index": edge_index,
            "edge_features": edge_features,
            "edge_mask": edge_mask,
            "flow_features": flow_features,
            "flow_mask": flow_mask,
            "metrics": metrics,
        }

    def _index_action(self, action: int) -> str:
        if action == 0:
            return "NO_OP"
        if 1 <= action <= self.max_edges:
            if action - 1 >= len(self._edges):
                raise ValueError(f"Action index {action} refers to an unused padded link slot")
            return f"ACTIVATE_BACKUP_LINK:{action - 1}"
        node_offset = self.max_edges + 1
        if node_offset <= action < node_offset + self.max_nodes:
            if action - node_offset >= len(self._nodes):
                raise ValueError(f"Action index {action} refers to an unused padded node slot")
            return f"ACTIVATE_NODE:{action - node_offset}"
        if action == self.max_edges + self.max_nodes + 1:
            return "REBALANCE_WEIGHTS"
        raise ValueError(f"Unknown action index: {action}")

    def action_masks(self) -> np.ndarray:
        valid = set(self.simulator.action_mask())
        mask = np.zeros(self.action_space.n, dtype=bool)
        mask[0] = True
        for edge_id in range(len(self._edges)):
            mask[edge_id + 1] = f"ACTIVATE_BACKUP_LINK:{edge_id}" in valid
        node_offset = self.max_edges + 1
        for node_id in range(len(self._nodes)):
            mask[node_offset + node_id] = f"ACTIVATE_NODE:{node_id}" in valid
        mask[-1] = "REBALANCE_WEIGHTS" in valid
        return mask

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        graph = self.initial_graph.copy()
        scenario = self.failure_scenario
        if self.randomize_failure:
            from shnet.failures.scenarios import FailureScenario

            kinds = ("hard_link_failure", "node_failure", "gradual_degradation", "congestion")
            kind = kinds[int(self.np_random.integers(len(kinds)))]
            if kind in {"hard_link_failure"}:
                target = list(graph.edges)[int(self.np_random.integers(graph.number_of_edges()))]
            elif kind == "node_failure":
                target = list(graph.nodes)[int(self.np_random.integers(graph.number_of_nodes()))]
            else:
                target = None
            scenario = FailureScenario(
                f"episode-{self.timestep}-{kind}",
                kind,
                target=target,
                severity=1.0,
                timestamp=0.0,
            )
        if scenario is not None:
            from shnet.failures.scenarios import inject_failure

            graph = inject_failure(graph, scenario)
        self.simulator = GraphSimulator(
            graph,
            self.flows,
            recovery_target=self.recovery_target,
            reward_weights=self.reward_weights,
        )
        self.timestep = 0
        observation = self._make_observation()
        summary = self.simulator.build_flow_summary()
        return observation, {
            "action_mask": self.action_masks(),
            "recovery_complete": is_recovered(summary, self.recovery_target),
        }

    def step(
        self, action: int | str
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        action_name = action if isinstance(action, str) else self._index_action(int(action))
        if action_name not in self.simulator.action_mask():
            raise ValueError(f"Invalid or currently masked action: {action_name}")
        packet = self.simulator.simulate_step(action_name)
        self.timestep += 1
        summary = packet["summary"]
        success = is_recovered(summary, self.recovery_target)
        terminated = bool(success and self.timestep > 0)
        truncated = self.timestep >= self.horizon and not terminated
        info = {
            "action": action_name,
            "action_mask": self.action_masks(),
            "recovered": packet["recovered"],
            "recovery_complete": success,
            "reconfig_cost": packet["reconfig_cost"],
            "summary": summary,
            "meta": packet["meta"],
        }
        return (
            self._make_observation(),
            float(packet["reward"]),
            terminated,
            truncated,
            info,
        )

    def render(self) -> None:
        summary = self.simulator.build_flow_summary()
        print(
            f"step={self.timestep} connected={summary['connected_ratio']:.3f} "
            f"throughput={summary['throughput_mbps']:.2f}Mbps"
        )
