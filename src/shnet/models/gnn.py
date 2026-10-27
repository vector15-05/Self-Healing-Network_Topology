from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np


class GraphFailurePredictor:
    """Edge-level message-passing feature model with a weighted MLP head.

    The implementation is NumPy-only so it can be trained on the fast simulator
    without requiring a GPU. It aggregates incident-link and endpoint features
    before classification, making its input independent of graph size.
    """

    def __init__(
        self,
        hidden_dim: int = 16,
        *,
        seed: int = 7,
        learning_rate: float = 0.05,
        epochs: int = 250,
        l2: float = 1e-3,
    ) -> None:
        if hidden_dim <= 0 or learning_rate <= 0 or epochs <= 0:
            raise ValueError("hidden_dim, learning_rate, and epochs must be positive")
        self.hidden_dim = hidden_dim
        self.seed = seed
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2 = l2
        self._weights_in: np.ndarray | None = None
        self._bias_in: np.ndarray | None = None
        self._weights_out: np.ndarray | None = None
        self._bias_out: float = 0.0
        self._mean: np.ndarray | None = None
        self._scale: np.ndarray | None = None
        self.training_history: list[float] = []

    @staticmethod
    def _edge_rows(graph: nx.Graph) -> tuple[list[tuple[Any, Any]], np.ndarray]:
        edges = list(graph.edges)
        if not edges:
            return [], np.empty((0, 12), dtype=np.float64)
        incident_utilization: dict[Any, list[float]] = {node: [] for node in graph.nodes}
        for u, v, attrs in graph.edges(data=True):
            utilization = float(attrs.get("utilization", 0.0))
            incident_utilization[u].append(utilization)
            incident_utilization[v].append(utilization)

        rows: list[list[float]] = []
        for u, v in edges:
            attrs = graph[u][v]
            left = graph.nodes[u]
            right = graph.nodes[v]
            neighbors = incident_utilization[u] + incident_utilization[v]
            neighborhood_mean = sum(neighbors) / max(1, len(neighbors))
            rows.append(
                [
                    float(attrs.get("bandwidth_mbps", 0.0)),
                    float(attrs.get("delay_ms", 0.0)),
                    float(attrs.get("loss_pct", 0.0)),
                    float(attrs.get("utilization", 0.0)),
                    float(attrs.get("cost", 0.0)),
                    float(attrs.get("active", True)),
                    float(graph.degree(u)),
                    float(graph.degree(v)),
                    float(left.get("capacity", 0.0)),
                    float(right.get("capacity", 0.0)),
                    neighborhood_mean,
                    float(left.get("load", 0.0)) + float(right.get("load", 0.0)),
                ]
            )
        return edges, np.asarray(rows, dtype=np.float64)

    @staticmethod
    def _edge_key(edge: tuple[Any, Any]) -> tuple[str, str]:
        return str(edge[0]), str(edge[1])

    def fit(
        self,
        graphs: list[nx.Graph],
        labels: list[dict[Any, float]] | None = None,
        *,
        epochs: int | None = None,
        learning_rate: float | None = None,
    ) -> GraphFailurePredictor:
        if not graphs:
            raise ValueError("At least one graph is required for training")
        if labels is not None and len(labels) != len(graphs):
            raise ValueError("labels must contain one edge-label mapping per graph")

        features: list[np.ndarray] = []
        targets: list[float] = []
        for graph_index, graph in enumerate(graphs):
            edges, rows = self._edge_rows(graph)
            if labels is None:
                edge_labels = {
                    edge: float(
                        not graph[u][v].get("active", True)
                        or graph[u][v].get("failure_kind") is not None
                    )
                    for edge, (u, v) in zip(edges, edges)
                }
            else:
                edge_labels = labels[graph_index]
            for edge, row in zip(edges, rows):
                u, v = edge
                label = edge_labels.get(
                    edge,
                    edge_labels.get(
                        self._edge_key(edge),
                        edge_labels.get(f"{u}->{v}", edge_labels.get(f"{v}->{u}", 0.0)),
                    ),
                )
                features.append(row)
                targets.append(float(label))
        if not features:
            raise ValueError("Training graphs must contain at least one link")

        matrix = np.asarray(features, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        if np.any((y < 0.0) | (y > 1.0)):
            raise ValueError("Training labels must be probabilities in [0, 1]")
        self._mean = matrix.mean(axis=0)
        self._scale = matrix.std(axis=0)
        self._scale[self._scale < 1e-8] = 1.0
        x = (matrix - self._mean) / self._scale

        positive = max(1.0, float(y.sum()))
        negative = max(1.0, float(len(y) - y.sum()))
        sample_weights = np.where(y > 0.0, len(y) / (2.0 * positive), len(y) / (2.0 * negative))
        step_size = learning_rate or self.learning_rate
        training_epochs = epochs or self.epochs
        rng = np.random.default_rng(self.seed)
        self._weights_in = rng.normal(
            0.0,
            np.sqrt(2.0 / x.shape[1]),
            size=(x.shape[1], self.hidden_dim),
        )
        self._bias_in = np.zeros(self.hidden_dim, dtype=np.float64)
        self._weights_out = rng.normal(
            0.0,
            np.sqrt(1.0 / self.hidden_dim),
            size=self.hidden_dim,
        )
        self._bias_out = 0.0
        self.training_history = []
        for _ in range(training_epochs):
            hidden = np.tanh(x @ self._weights_in + self._bias_in)
            probabilities = self._sigmoid(hidden @ self._weights_out + self._bias_out)
            output_gradient = sample_weights * (probabilities - y) / len(y)
            gradient_out = hidden.T @ output_gradient + self.l2 * self._weights_out
            gradient_bias_out = float(output_gradient.sum())
            hidden_gradient = (
                output_gradient[:, None]
                * self._weights_out[None, :]
                * (1.0 - hidden**2)
            )
            gradient_in = x.T @ hidden_gradient + self.l2 * self._weights_in
            gradient_bias_in = hidden_gradient.sum(axis=0)
            self._weights_out -= step_size * gradient_out
            self._bias_out -= step_size * gradient_bias_out
            self._weights_in -= step_size * gradient_in
            self._bias_in -= step_size * gradient_bias_in
            probabilities = self._sigmoid(
                np.tanh(x @ self._weights_in + self._bias_in) @ self._weights_out
                + self._bias_out
            )
            clipped = np.clip(probabilities, 1e-9, 1.0 - 1e-9)
            loss = -np.average(
                y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped),
                weights=sample_weights,
            )
            self.training_history.append(float(loss))
        return self

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))

    def predict(self, graph: nx.Graph) -> dict[str, float]:
        edges, rows = self._edge_rows(graph)
        if (
            self._weights_in is None
            or self._bias_in is None
            or self._weights_out is None
            or self._mean is None
            or self._scale is None
        ):
            # Explicit, deterministic stress prior for inference before training.
            scores = {}
            for (u, v), row in zip(edges, rows):
                _, delay, loss, utilization, *_ = row
                score = min(1.0, max(0.0, 0.02 + delay / 100.0 + loss / 20.0 + utilization * 0.65))
                scores[f"{u}->{v}"] = round(float(score), 6)
            return scores
        x = (rows - self._mean) / self._scale
        hidden = np.tanh(x @ self._weights_in + self._bias_in)
        probabilities = self._sigmoid(hidden @ self._weights_out + self._bias_out)
        return {
            f"{u}->{v}": round(float(probability), 6)
            for (u, v), probability in zip(edges, probabilities)
        }

    def save(self, path: str | Path) -> None:
        if (
            self._weights_in is None
            or self._bias_in is None
            or self._weights_out is None
            or self._mean is None
            or self._scale is None
        ):
            raise RuntimeError("Cannot save an untrained predictor")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            np.savez_compressed(
                handle,
                weights_in=self._weights_in,
                bias_in=self._bias_in,
                weights_out=self._weights_out,
                bias_out=np.asarray([self._bias_out], dtype=np.float64),
                mean=self._mean,
                scale=self._scale,
                hidden_dim=np.asarray([self.hidden_dim], dtype=np.int64),
                seed=np.asarray([self.seed], dtype=np.int64),
            )

    @classmethod
    def load(cls, path: str | Path) -> GraphFailurePredictor:
        with np.load(Path(path), allow_pickle=False) as archive:
            model = cls(
                hidden_dim=int(archive["hidden_dim"][0]),
                seed=int(archive["seed"][0]),
            )
            model._weights_in = archive["weights_in"].copy()
            model._bias_in = archive["bias_in"].copy()
            model._weights_out = archive["weights_out"].copy()
            model._bias_out = float(archive["bias_out"][0])
            model._mean = archive["mean"].copy()
            model._scale = archive["scale"].copy()
        return model

    def evaluate(
        self,
        graphs: list[nx.Graph],
        labels: list[dict[Any, float]],
        *,
        threshold: float = 0.5,
    ) -> dict[str, float]:
        actual: list[int] = []
        predicted: list[float] = []
        for graph, graph_labels in zip(graphs, labels):
            scores = self.predict(graph)
            for u, v in graph.edges:
                actual.append(
                    int(
                        graph_labels.get(
                            (u, v),
                            graph_labels.get(
                                f"{u}->{v}", graph_labels.get(f"{v}->{u}", 0.0)
                            ),
                        )
                        >= 0.5
                    )
                )
                predicted.append(scores.get(f"{u}->{v}", scores.get(f"{v}->{u}", 0.0)))

        if not actual:
            return {"pr_auc": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}
        labels_array = np.asarray(actual, dtype=np.int64)
        scores_array = np.asarray(predicted, dtype=np.float64)
        binary = scores_array >= threshold
        true_positive = int(np.sum((labels_array == 1) & binary))
        false_positive = int(np.sum((labels_array == 0) & binary))
        false_negative = int(np.sum((labels_array == 1) & ~binary))
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
        order = np.argsort(-scores_array)
        sorted_labels = labels_array[order]
        positives = max(1, int(labels_array.sum()))
        cumulative_tp = np.cumsum(sorted_labels == 1)
        cumulative_fp = np.cumsum(sorted_labels == 0)
        precisions = cumulative_tp / np.maximum(1, cumulative_tp + cumulative_fp)
        recalls = cumulative_tp / positives
        pr_auc = float(np.sum((recalls[1:] - recalls[:-1]) * precisions[1:])) if len(recalls) > 1 else float(labels_array[0])
        return {
            "pr_auc": pr_auc,
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
        }
