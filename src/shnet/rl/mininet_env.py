from __future__ import annotations

import time
from typing import Any

from shnet.controller.ryu_client import RyuControllerClient
from shnet.emulation.mininet_builder import MininetHarness
from shnet.rl.env import SelfHealingEnv
from shnet.telemetry.ryu_stats import RyuStatsCollector


class MininetSelfHealingEnv(SelfHealingEnv):
    """Gymnasium adapter that applies the same recovery actions to live Mininet."""

    def __init__(
        self,
        harness: MininetHarness,
        *,
        controller_client: RyuControllerClient | None = None,
        horizon: int = 5,
        traffic_duration_seconds: int = 1,
        traffic_rate_mbps: float = 5.0,
        **env_kwargs: Any,
    ) -> None:
        super().__init__(harness.graph, horizon=horizon, **env_kwargs)
        self.harness = harness
        self.controller_client = controller_client or RyuControllerClient(
            host=harness.controller_ip,
            port=harness.rest_port,
        )
        self.stats_collector = RyuStatsCollector(self.controller_client)
        self.traffic_duration_seconds = traffic_duration_seconds
        self.traffic_rate_mbps = traffic_rate_mbps

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if self.harness.net is not None:
            self.harness.teardown()
        self.harness.graph = self.initial_graph.copy()
        observation, info = super().reset(seed=seed, options=options)
        self.harness.start()
        try:
            for u, v, attrs in self.simulator.graph.edges(data=True):
                if not attrs.get("active", True):
                    self.harness.set_link_status(u, v, up=False)
            for node, attrs in self.simulator.graph.nodes(data=True):
                if not attrs.get("active", True):
                    self.harness.set_node_status(node, up=False)
        except Exception:
            self.harness.teardown()
            raise
        return observation, info

    def step(self, action: int | str):
        action_name = action if isinstance(action, str) else self._index_action(int(action))
        rule_changes_before = self.controller_client.flow_rule_changes
        observation, reward, terminated, truncated, info = super().step(action)
        if action_name.startswith(("ACTIVATE_BACKUP_LINK:", "REPAIR:")):
            if action_name.startswith("ACTIVATE_BACKUP_LINK:"):
                edge_id = int(action_name.split(":", 1)[1])
                u, v = self._edges[edge_id]
            else:
                target = action_name.split(":", 1)[1]
                u, v = next(
                    (edge for edge in self._edges if f"{edge[0]}->{edge[1]}" == target or f"{edge[1]}->{edge[0]}" == target)
                )
            self.harness.set_link_status(u, v, up=True)
        elif action_name.startswith("ACTIVATE_NODE:"):
            node_id = int(action_name.split(":", 1)[1])
            self.harness.set_node_status(self._nodes[node_id], up=True)
        elif action_name == "REBALANCE_WEIGHTS":
            weights = {
                f"{u}-{v}": float(attrs.get("weight", attrs.get("cost", 1.0)))
                for u, v, attrs in self.simulator.graph.edges(data=True)
                if attrs.get("active", True)
            }
            self.controller_client.set_link_weights(weights)

        if action_name != "NO_OP":
            summary = self.simulator.build_flow_summary()
            for flow in self.simulator.flows:
                record = summary["flows"].get(
                    flow.flow_id or f"{flow.source}->{flow.target}",
                    {},
                )
                switch_path = [
                    self.harness.switch_dpids[node]
                    for node in record.get("path", [])
                    if node in self.harness.switch_dpids
                ]
                if len(switch_path) < 2:
                    continue
                destination_ip = self.harness.host_ip_for_switch(flow.target)
                self.controller_client.set_path(
                    switch_path,
                    destination_ip=destination_ip,
                )
        rule_changes = self.controller_client.flow_rule_changes - rule_changes_before
        rule_change_adjustment = -self.reward_weights.reconfiguration_cost * rule_changes
        reward += rule_change_adjustment
        info["flow_rule_changes"] = rule_changes
        info["reconfiguration_count"] = self.simulator.reconfigurations + rule_changes
        info["measured_reward_adjustment"] = rule_change_adjustment

        measured = self.harness.verify_connectivity()
        info["measured_connectivity"] = measured
        loss_adjustment = -self.reward_weights.qos_degradation * measured["packet_loss_pct"] * 0.15
        reward += loss_adjustment
        info["measured_reward_adjustment"] += loss_adjustment
        traffic = self.harness.run_traffic(
            duration_seconds=self.traffic_duration_seconds,
            flows=1,
            protocol="tcp",
            rate_mbps=self.traffic_rate_mbps,
        )
        throughputs = [
            float(record.get("result", {}).get("sum_received", {}).get("bits_per_second", 0.0))
            / 1_000_000.0
            for record in traffic
        ]
        measured_throughput = sum(throughputs) / max(1, len(throughputs))
        shortfall = max(
            0.0,
            (self.traffic_rate_mbps - measured_throughput) / self.traffic_rate_mbps,
        )
        throughput_adjustment = -self.reward_weights.qos_degradation * shortfall * 5.0
        reward += throughput_adjustment
        info["measured_traffic"] = traffic
        info["measured_throughput_mbps"] = measured_throughput
        info["measured_reward_adjustment"] += throughput_adjustment
        measured_recovered = bool(measured["connected"] and shortfall <= 0.02)
        info["measured_recovery_complete"] = measured_recovered
        if terminated and not measured_recovered:
            terminated = False
            info["recovery_complete"] = False
        if self.harness.switch_dpids:
            snapshot = self.stats_collector.collect(
                self.simulator.graph,
                self.harness.switch_dpids,
                timestamp=time.time(),
            )
            info["telemetry_timestamp"] = snapshot.timestamp
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        self.harness.teardown()
