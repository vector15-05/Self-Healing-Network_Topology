import networkx as nx

from shnet.telemetry.snapshot import SnapshotStore, TelemetryCollector, graph_to_snapshot


def test_snapshot_contains_node_and_edge_features(tmp_path) -> None:
    graph = nx.path_graph(3)
    nx.set_node_attributes(graph, "switch", "type")
    nx.set_node_attributes(graph, 100.0, "capacity")
    for u, v in graph.edges:
        graph[u][v].update(
            bandwidth_mbps=50.0,
            delay_ms=3.0,
            loss_pct=0.1,
            utilization=0.2,
            active=True,
        )
    store = SnapshotStore(tmp_path / "sequence.jsonl")
    collector = TelemetryCollector(store)
    snapshot = collector.collect(
        graph,
        timestamp=1.5,
        flow_summary={"connected_ratio": 1.0},
        labels={"0->1": 1.0},
    )

    assert len(snapshot.node_features) == 3
    assert len(snapshot.node_features[0]) == 6
    assert len(snapshot.edge_index) == 2
    assert len(snapshot.edge_features[0]) == 6
    assert SnapshotStore(store.path).read_all()[0].labels == {"0->1": 1.0}


def test_empty_graph_snapshot_is_supported() -> None:
    snapshot = graph_to_snapshot(nx.Graph(), timestamp=3.0)
    assert snapshot.timestamp == 3.0
    assert snapshot.node_features == []
    assert snapshot.edge_features == []
