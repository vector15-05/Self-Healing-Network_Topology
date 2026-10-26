import networkx as nx

from shnet.topology.generators import generate_topology


def test_barabasi_topology_connectivity_and_attributes() -> None:
    graph = generate_topology("barabasi_albert", 12, seed=42)

    assert graph.number_of_nodes() == 12
    assert nx.is_connected(graph)
    assert graph.number_of_edges() >= 11
    for _, _, edge_data in graph.edges(data=True):
        assert edge_data["bandwidth_mbps"] > 0
        assert edge_data["delay_ms"] > 0
        assert edge_data["loss_pct"] >= 0
