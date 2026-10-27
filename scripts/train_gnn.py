#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from shnet.failures.scenarios import FailureScenario, failure_labels, inject_failure
from shnet.models.gnn import GraphFailurePredictor
from shnet.telemetry.snapshot import graph_to_snapshot
from shnet.topology.generators import generate_topology
from shnet.utils.config import load_config


def build_dataset(
    samples: int,
    seed: int,
    kinds: tuple[str, ...] = ("barabasi_albert", "grid", "erdos_renyi"),
) -> tuple[list[nx.Graph], list[dict[str, float]]]:
    graphs: list[nx.Graph] = []
    labels: list[dict[str, float]] = []
    for index in range(samples):
        graph = generate_topology(kinds[index % len(kinds)], 10 + index % 8, seed=seed + index)
        centrality = nx.edge_betweenness_centrality(graph)
        edge = max(
            graph.edges,
            key=lambda item: centrality.get(item, centrality.get((item[1], item[0]), 0.0)),
        )
        scenario = FailureScenario(
            name=f"link_failure_{index}",
            kind="hard_link_failure",
            target=edge,
            timestamp=0.0,
            metadata={"seed": seed + index},
        )
        labels.append(failure_labels(graph, [scenario], horizon_seconds=1.0, timestamp=0.0))
        graphs.append(inject_failure(graph, scenario))
    return graphs, labels


def build_temporal_dataset(
    samples: int,
    seed: int,
    window_size: int,
    kinds: tuple[str, ...] = ("barabasi_albert", "grid", "erdos_renyi"),
) -> tuple[list[list[nx.Graph]], list[dict[str, float]]]:
    windows: list[list[nx.Graph]] = []
    labels: list[dict[str, float]] = []
    for index in range(samples):
        graph = generate_topology(kinds[index % len(kinds)], 10 + index % 8, seed=seed + index)
        centrality = nx.edge_betweenness_centrality(graph)
        target = max(
            graph.edges,
            key=lambda edge: centrality.get(edge, centrality.get((edge[1], edge[0]), 0.0)),
        )
        scenario = FailureScenario(
            name=f"temporal_link_failure_{index}",
            kind="hard_link_failure",
            target=target,
            timestamp=float(window_size),
            metadata={"seed": seed + index},
        )
        labels.append(
            failure_labels(
                graph,
                [scenario],
                horizon_seconds=float(window_size),
                timestamp=0.0,
            )
        )
        sequence: list[nx.Graph] = []
        original_delay = float(graph[target[0]][target[1]].get("delay_ms", 0.0))
        original_loss = float(graph[target[0]][target[1]].get("loss_pct", 0.0))
        for tick in range(window_size):
            step_graph = graph.copy()
            progress = (tick + 1) / window_size
            attrs = step_graph[target[0]][target[1]]
            attrs["delay_ms"] = original_delay + 5.0 * progress
            attrs["loss_pct"] = min(100.0, original_loss + 0.8 * progress)
            attrs["utilization"] = min(1.0, progress * 0.7)
            sequence.append(step_graph)
        windows.append(sequence)
    return windows, labels


def _classification_metrics(labels: list[int], scores: list[float]) -> dict[str, float]:
    if not labels:
        return {"pr_auc": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}
    ordered = sorted(zip(scores, labels), reverse=True)
    positives = max(1, sum(labels))
    tp = fp = 0
    prev_recall = prev_precision = pr_auc = 0.0
    for _, label in ordered:
        if label:
            tp += 1
        else:
            fp += 1
        precision_at_rank = tp / max(1, tp + fp)
        recall_at_rank = tp / positives
        pr_auc += (recall_at_rank - prev_recall) * (
            precision_at_rank + prev_precision
        ) / 2.0
        prev_recall, prev_precision = recall_at_rank, precision_at_rank
    predicted = [score >= 0.5 for score in scores]
    true_positive = sum(actual == 1 and guess for actual, guess in zip(labels, predicted))
    false_positive = sum(actual == 0 and guess for actual, guess in zip(labels, predicted))
    false_negative = sum(actual == 1 and not guess for actual, guess in zip(labels, predicted))
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    return {"pr_auc": pr_auc, "f1": f1, "precision": precision, "recall": recall}


def train_numpy(
    samples: int,
    seed: int,
    output: Path,
    *,
    epochs: int,
    learning_rate: float,
    hidden_dim: int,
) -> None:
    train_graphs, train_labels = build_dataset(samples, seed)
    test_graphs, test_labels = build_dataset(
        max(3, samples // 3),
        seed + 1000,
        ("waxman", "fat_tree"),
    )
    predictor = GraphFailurePredictor(
        seed=seed,
        epochs=epochs,
        learning_rate=learning_rate,
        hidden_dim=hidden_dim,
    )
    predictor.fit(train_graphs, train_labels)
    metrics = predictor.evaluate(test_graphs, test_labels)
    predictor.save(output)
    print({"backend": "numpy-message-passing", "holdout": metrics, "checkpoint": str(output)})


def train_torch(
    samples: int,
    seed: int,
    output: Path,
    *,
    epochs: int,
    learning_rate: float,
    hidden_dim: int,
    temporal: bool,
    window_size: int,
) -> None:
    try:
        import torch

        from shnet.models.torch_gnn import EdgeMessagePassingNetwork, train_edge_gnn
    except ImportError as exc:
        raise SystemExit(
            "The torch backend requires PyTorch. Install optional dependencies with "
            "`python -m pip install -e '.[rl]'`."
        ) from exc

    if temporal:
        train_windows, labels = build_temporal_dataset(samples, seed, window_size)
        test_windows, test_labels = build_temporal_dataset(
            max(3, samples // 3),
            seed + 1000,
            window_size,
            ("waxman", "fat_tree"),
        )
        train_graphs = [window[-1] for window in train_windows]
        test_graphs = [window[-1] for window in test_windows]
    else:
        train_graphs, labels = build_dataset(samples, seed)
        test_graphs, test_labels = build_dataset(
            max(3, samples // 3),
            seed + 1000,
            ("waxman", "fat_tree"),
        )
        train_windows = [[graph] for graph in train_graphs]
        test_windows = [[graph] for graph in test_graphs]

    def tensor_rows(windows: list[list[nx.Graph]], edge_labels: list[dict[str, float]]):
        rows = []
        for window, label_map in zip(windows, edge_labels):
            snapshots = [graph_to_snapshot(graph) for graph in window]
            edges = list(window[-1].edges)
            labels_for_edges = [
                label_map.get(f"{u}->{v}", label_map.get(f"{v}->{u}", 0.0))
                for u, v in edges
            ]
            row: dict[str, Any] = {
                "labels": torch.tensor(labels_for_edges, dtype=torch.float32),
            }
            if temporal:
                row["window"] = {
                    "node_features": torch.tensor(
                        np.asarray([snapshot.node_features for snapshot in snapshots]),
                        dtype=torch.float32,
                    ),
                    "edge_index": torch.tensor(
                        np.asarray(snapshots[-1].edge_index).T,
                        dtype=torch.long,
                    ),
                    "edge_features": torch.tensor(
                        np.asarray([snapshot.edge_features for snapshot in snapshots]),
                        dtype=torch.float32,
                    ),
                }
            else:
                row.update(
                    {
                        "node_features": torch.tensor(
                            np.asarray(snapshots[-1].node_features),
                            dtype=torch.float32,
                        ),
                        "edge_index": torch.tensor(
                            np.asarray(snapshots[-1].edge_index).T,
                            dtype=torch.long,
                        ),
                        "edge_features": torch.tensor(
                            np.asarray(snapshots[-1].edge_features),
                            dtype=torch.float32,
                        ),
                    }
                )
            rows.append(row)
        return rows

    training_rows = tensor_rows(train_windows, labels)
    holdout_rows = tensor_rows(test_windows, test_labels)
    model = EdgeMessagePassingNetwork(hidden_dim=hidden_dim)
    history = train_edge_gnn(
        model,
        training_rows,
        epochs=epochs,
        learning_rate=learning_rate,
    )
    model.eval()
    actual: list[int] = []
    predicted: list[float] = []
    with torch.no_grad():
        for row in holdout_rows:
            if temporal:
                output_values = model.forward_window(
                    row["window"]["node_features"],
                    row["window"]["edge_index"],
                    row["window"]["edge_features"],
                )
            else:
                output_values = model(
                    row["node_features"],
                    row["edge_index"],
                    row["edge_features"],
                )
            probabilities = torch.sigmoid(output_values["link_failure_logits"])
            actual.extend(int(value) for value in row["labels"].tolist())
            predicted.extend(float(value) for value in probabilities.tolist())
    metrics = _classification_metrics(actual, predicted)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "seed": seed,
            "hidden_dim": hidden_dim,
            "temporal": temporal,
            "window_size": window_size if temporal else 1,
        },
        output,
    )
    print(
        {
            "backend": "torch-temporal-edge-message-passing" if temporal else "torch-edge-message-passing",
            "graphs": len(training_rows),
            "epochs": len(history),
            "final_loss": history[-1],
            "holdout": metrics,
            "checkpoint": str(output),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the topology-aware link-risk predictor.")
    parser.add_argument("--config", type=Path, default=Path("configs/training.yaml"))
    parser.add_argument(
        "--backend",
        choices=("numpy", "torch", "torch-temporal"),
        default="numpy",
    )
    parser.add_argument("--samples", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path, default=Path("results/gnn_model.npz"))
    args = parser.parse_args()
    config = load_config(args.config)
    training = config.get("training", {})
    model_config = config.get("model", {})
    samples = args.samples if args.samples is not None else int(training.get("samples", 24))
    epochs = args.epochs if args.epochs is not None else int(training.get("epochs", 100))
    learning_rate = (
        args.learning_rate
        if args.learning_rate is not None
        else float(training.get("learning_rate", 0.05))
    )
    hidden_dim = args.hidden_dim if args.hidden_dim is not None else int(model_config.get("hidden_dim", 16))
    window_size = int(model_config.get("max_window", 3))
    seed = args.seed if args.seed is not None else int(training.get("seed", 7))
    if samples < 3 or epochs <= 0 or learning_rate <= 0 or hidden_dim <= 0:
        parser.error(
            "--samples must be >= 3; --epochs, --learning-rate, and --hidden-dim must be positive"
        )
    if args.backend == "numpy":
        train_numpy(
            samples,
            seed,
            args.output,
            epochs=epochs,
            learning_rate=learning_rate,
            hidden_dim=hidden_dim,
        )
    else:
        train_torch(
            samples,
            seed,
            args.output.with_suffix(".pt"),
            epochs=epochs,
            learning_rate=learning_rate,
            hidden_dim=hidden_dim,
            temporal=args.backend == "torch-temporal",
            window_size=window_size,
        )


if __name__ == "__main__":
    main()
