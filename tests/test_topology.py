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


def test_generators_return_exact_seeded_node_count() -> None:
    for kind in ("barabasi_albert", "erdos_renyi", "ring", "grid", "waxman", "fat_tree"):
        graph_a = generate_topology(kind, 11, seed=13)
        graph_b = generate_topology(kind, 11, seed=13)
        assert graph_a.number_of_nodes() == 11
        assert nx.is_connected(graph_a)
        assert set(graph_a.edges) == set(graph_b.edges)
