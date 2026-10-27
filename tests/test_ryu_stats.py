import networkx as nx

from shnet.telemetry.ryu_stats import RyuStatsCollector


class FakeRyuClient:
    def __init__(self) -> None:
        self.sample = 0

    def get_topology(self):
        return {
            "links": [
                {"src": {"dpid": "1", "port_no": 2}, "dst": {"dpid": "2", "port_no": 3}},
                {"src": {"dpid": "2", "port_no": 3}, "dst": {"dpid": "1", "port_no": 2}},
            ]
        }

    def get_port_stats(self, dpid: int):
        self.sample += 1
        tx_bytes = 0 if self.sample <= 2 else 100_000
        return {
            str(dpid): [
                {
                    "port_no": 2 if dpid == 1 else 3,
                    "tx_bytes": tx_bytes,
                    "rx_bytes": tx_bytes,
                    "tx_packets": tx_bytes // 100,
                    "rx_packets": tx_bytes // 100,
                    "tx_dropped": 0,
                    "rx_dropped": 0,
                    "tx_errors": 0,
                    "rx_errors": 0,
                }
            ]
        }

    def get_flow_stats(self, dpid: int):
        return {str(dpid): [{"priority": 1}, {"priority": 0}]}


def test_ryu_collector_derives_link_utilization_and_flow_counts() -> None:
    graph = nx.Graph()
    graph.add_edge(0, 1, bandwidth_mbps=10.0, delay_ms=1.0, loss_pct=0.0)
    collector = RyuStatsCollector(FakeRyuClient())

    collector.collect(graph, {0: 1, 1: 2}, timestamp=0.0)
    snapshot = collector.collect(graph, {0: 1, 1: 2}, timestamp=1.0)

    assert graph[0][1]["throughput_mbps"] > 0.0
    assert 0.0 < graph[0][1]["utilization"] <= 1.0
    assert graph.nodes[0]["flow_count"] == 1
    assert snapshot.flow_summary["port_stats_sampled"] == 2
