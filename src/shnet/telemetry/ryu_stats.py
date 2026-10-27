from __future__ import annotations

import time
from typing import Any

import networkx as nx

from shnet.controller.ryu_client import RyuControllerClient
from shnet.telemetry.snapshot import GraphSnapshot, TelemetryCollector


class RyuStatsCollector:
    """Derive per-link rates and loss estimates from Ryu OpenFlow port counters."""

    def __init__(
        self,
        client: RyuControllerClient,
        *,
        snapshot_collector: TelemetryCollector | None = None,
    ) -> None:
        self.client = client
        self.snapshot_collector = snapshot_collector or TelemetryCollector()
        self._previous: dict[tuple[int, int], dict[str, float]] = {}
        self._last_timestamp: float | None = None

    @staticmethod
    def _rows(response: Any, dpid: int) -> list[dict[str, Any]]:
        if isinstance(response, list):
            return response
        if isinstance(response, dict):
            return response.get(str(dpid), response.get(dpid, []))
        raise ValueError(f"Unexpected Ryu stats response for switch {dpid}")

    def collect(
        self,
        graph: nx.Graph,
        dpid_by_node: dict[Any, int],
        *,
        timestamp: float | None = None,
        latency_by_edge: dict[tuple[Any, Any], float] | None = None,
        labels: dict[str, float] | None = None,
    ) -> GraphSnapshot:
        now = time.time() if timestamp is None else float(timestamp)
        elapsed = max(1e-6, now - self._last_timestamp) if self._last_timestamp is not None else 0.0
        self._last_timestamp = now
        topology = self.client.get_topology()
        links = topology.get("links", [])
        ports: dict[tuple[int, int], dict[str, float]] = {}
        flow_counts: dict[Any, int] = {}

        for node, dpid in dpid_by_node.items():
            port_response = self.client.get_port_stats(dpid)
            for row in self._rows(port_response, dpid):
                port_number = int(row["port_no"])
                ports[(int(dpid), port_number)] = {
                    field: float(row.get(field, 0.0))
                    for field in ("tx_bytes", "rx_bytes", "tx_packets", "rx_packets", "tx_dropped", "rx_dropped", "tx_errors", "rx_errors")
                }
            flow_response = self.client.get_flow_stats(dpid)
            flow_rows = self._rows(flow_response, dpid)
            flow_counts[node] = sum(1 for row in flow_rows if int(row.get("priority", 0)) > 0)
            graph.nodes[node]["flow_count"] = flow_counts[node]

        port_links: dict[tuple[int, int], tuple[int, int]] = {}
        for link in links:
            source = link.get("src", {})
            target = link.get("dst", {})
            try:
                left = (int(source["dpid"]), int(source["port_no"]))
                right = (int(target["dpid"]), int(target["port_no"]))
            except (KeyError, TypeError, ValueError):
                continue
            port_links[left] = right

        for u, v, attrs in graph.edges(data=True):
            dpid_u = dpid_by_node.get(u)
            dpid_v = dpid_by_node.get(v)
            if dpid_u is None or dpid_v is None:
                continue
            matching_ports: list[tuple[int, int]] = []
            for (switch_id, port_number), neighbor in port_links.items():
                if switch_id == dpid_u and neighbor[0] == dpid_v:
                    matching_ports.append((switch_id, port_number))
                elif switch_id == dpid_v and neighbor[0] == dpid_u:
                    matching_ports.append((switch_id, port_number))

            throughput_candidates: list[float] = []
            lost_packets = sent_packets = 0.0
            for port_key in matching_ports:
                current = ports.get(port_key)
                previous = self._previous.get(port_key)
                if current is None:
                    continue
                if previous is not None and elapsed > 0:
                    byte_delta = max(0.0, current["tx_bytes"] - previous["tx_bytes"])
                    throughput_candidates.append(byte_delta * 8.0 / elapsed / 1_000_000.0)
                    sent_packets += max(0.0, current["tx_packets"] - previous["tx_packets"])
                    lost_packets += sum(
                        max(0.0, current[field] - previous[field])
                        for field in ("tx_dropped", "rx_dropped", "tx_errors", "rx_errors")
                    )
                self._previous[port_key] = current

            throughput = max(throughput_candidates, default=0.0)
            capacity = max(0.01, float(attrs.get("bandwidth_mbps", 100.0)))
            attrs["throughput_mbps"] = throughput
            attrs["utilization"] = min(1.0, throughput / capacity)
            attrs["loss_pct"] = (
                min(100.0, 100.0 * lost_packets / max(1.0, sent_packets))
                if sent_packets > 0.0
                else float(attrs.get("loss_pct", 0.0))
            )
            if latency_by_edge:
                latency = latency_by_edge.get((u, v), latency_by_edge.get((v, u)))
                if latency is not None:
                    attrs["delay_ms"] = float(latency)

        flow_summary = {
            "timestamp": now,
            "switch_flow_counts": {str(node): count for node, count in flow_counts.items()},
            "port_stats_sampled": len(ports),
        }
        return self.snapshot_collector.collect(
            graph,
            timestamp=now,
            flow_summary=flow_summary,
            labels=labels,
        )
