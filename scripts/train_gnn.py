#!/usr/bin/env python3
from __future__ import annotations

from shnet.models.gnn import GraphFailurePredictor
from shnet.topology.generators import generate_topology


def main() -> None:
    predictor = GraphFailurePredictor(hidden_dim=16)
    graphs = [generate_topology("ring", 10 + idx, seed=idx) for idx in range(3)]
    predictor.fit(graphs)
    for graph in graphs:
        risk = predictor.predict(graph)
    print({"models_loaded": 1, "edge_predictions": len(risk) if 'risk' in locals() else 0})


if __name__ == "__main__":
    main()
