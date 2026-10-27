import networkx as nx

from shnet.models.gnn import GraphFailurePredictor


def _graph(delay: float, loss: float, active: bool) -> nx.Graph:
    graph = nx.Graph()
    graph.add_node(0, capacity=100.0)
    graph.add_node(1, capacity=100.0)
    graph.add_edge(
        0,
        1,
        bandwidth_mbps=50.0,
        delay_ms=delay,
        loss_pct=loss,
        utilization=0.1,
        active=active,
    )
    return graph


def test_predictor_fits_saves_and_loads(tmp_path) -> None:
    graphs = [_graph(2.0, 0.1, True), _graph(10.0, 4.0, False)]
    labels = [{(0, 1): 0.0}, {(0, 1): 1.0}]
    predictor = GraphFailurePredictor(epochs=100).fit(graphs, labels)
    prediction = predictor.predict(graphs[1])
    assert 0.0 <= prediction["0->1"] <= 1.0
    assert predictor.training_history[-1] < predictor.training_history[0]

    checkpoint = tmp_path / "predictor.npz"
    predictor.save(checkpoint)
    restored = GraphFailurePredictor.load(checkpoint)
    assert restored.predict(graphs[1]) == prediction
