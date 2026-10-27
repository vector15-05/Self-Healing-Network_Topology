from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path
from typing import Any

import networkx as nx

from shnet.baselines.strategies import (
    FixedFeatureGreedyBaseline,
    GNNHeuristicBaseline,
    LoadAwareBaseline,
    NoOpBaseline,
    ReactiveShortestPathBaseline,
)
from shnet.failures.scenarios import FailureScenario, inject_failure
from shnet.models.gnn import GraphFailurePredictor
from shnet.sim.graph_sim import GraphSimulator
from shnet.sim.reward import is_recovered
from shnet.topology.generators import generate_topology

STRATEGIES = {
    "NO_OP": NoOpBaseline,
    "REACTIVE": ReactiveShortestPathBaseline,
    "REACTIVE_SHORTEST_PATH": ReactiveShortestPathBaseline,
    "LOAD_AWARE": LoadAwareBaseline,
    "GNN_HEURISTIC": GNNHeuristicBaseline,
    "FIXED_FEATURE_GREEDY": FixedFeatureGreedyBaseline,
}


def _restored(summary: dict[str, Any], baseline: dict[str, Any]) -> bool:
    return is_recovered(summary, baseline)


def mttr(recovery_ticks: list[float | None], *, tick_seconds: float = 1.0) -> float | None:
    recovered = [float(ticks) * tick_seconds for ticks in recovery_ticks if ticks is not None]
    return sum(recovered) / len(recovered) if recovered else None


def evaluate_failure_predictions(
    snapshots: list[Any],
    predictions: list[dict[str, float]],
    failure_timestamps: dict[str, float],
    *,
    threshold: float = 0.5,
) -> dict[str, float | None]:
    if len(snapshots) != len(predictions):
        raise ValueError("Provide one prediction mapping per snapshot")
    labels: list[int] = []
    scores: list[float] = []
    first_prediction: dict[str, float] = {}
    for snapshot, prediction in zip(snapshots, predictions):
        for link, label in snapshot.labels.items():
            labels.append(int(float(label) >= 0.5))
            scores.append(float(prediction.get(link, 0.0)))
            if prediction.get(link, 0.0) >= threshold:
                first_prediction.setdefault(link, float(snapshot.timestamp))
    if not labels:
        return {"pr_auc": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0, "lead_time_seconds": None}

    ordered = sorted(zip(scores, labels), reverse=True)
    positives = max(1, sum(labels))
    true_positive = false_positive = 0
    previous_recall = previous_precision = 0.0
    pr_auc = 0.0
    for _, label in ordered:
        if label:
            true_positive += 1
        else:
            false_positive += 1
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / positives
        pr_auc += (recall - previous_recall) * (precision + previous_precision) / 2.0
        previous_recall, previous_precision = recall, precision

    predicted_labels = [score >= threshold for score in scores]
    tp = sum(actual == 1 and predicted for actual, predicted in zip(labels, predicted_labels))
    fp = sum(actual == 0 and predicted for actual, predicted in zip(labels, predicted_labels))
    fn = sum(actual == 1 and not predicted for actual, predicted in zip(labels, predicted_labels))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    lead_times = [
        failure_time - first_prediction[link]
        for link, failure_time in failure_timestamps.items()
        if link in first_prediction and first_prediction[link] < failure_time
    ]
    return {
        "pr_auc": pr_auc,
        "f1": 2.0 * precision * recall / max(1e-12, precision + recall),
        "precision": precision,
        "recall": recall,
        "lead_time_seconds": statistics.mean(lead_times) if lead_times else None,
    }


def _confidence_interval(values: list[float]) -> tuple[float | None, float | None, float | None]:
    if not values:
        return None, None, None
    mean = statistics.mean(values)
    if len(values) == 1:
        return mean, 0.0, 0.0
    std = statistics.stdev(values)
    margin = 1.96 * std / math.sqrt(len(values))
    return mean, std, margin


def compute_experiment_metrics(
    strategy_name: str,
    trials: list[dict[str, Any]],
) -> dict[str, Any]:
    def numeric(key: str) -> list[float]:
        return [float(trial[key]) for trial in trials if trial.get(key) is not None]

    metrics: dict[str, Any] = {"strategy": strategy_name, "trials": len(trials)}
    for metric in ("mttr_seconds", "reconfigurations", "restoration_rate", "throughput_ratio", "mean_latency_ms", "mean_loss_pct"):
        values = numeric(metric)
        mean, std, margin = _confidence_interval(values)
        metrics[f"{metric}_mean"] = mean
        metrics[f"{metric}_std"] = std
        metrics[f"{metric}_ci95"] = margin
    return metrics


def _failure_edge(graph: nx.Graph, seed: int) -> tuple[Any, Any] | None:
    if not graph.edges:
        return None
    bridge_edges = list(nx.bridges(graph))
    edges = bridge_edges or list(graph.edges)
    centrality = nx.edge_betweenness_centrality(graph)
    return max(edges, key=lambda edge: (centrality.get(edge, centrality.get((edge[1], edge[0]), 0.0)), str(edge)))


def _scenario_for(graph: nx.Graph, kind: str, seed: int) -> FailureScenario:
    kind = {
        "degradation": "gradual_degradation",
        "hard_failure": "hard_link_failure",
        "node": "node_failure",
    }.get(kind, kind)
    if kind in {"hard_link_failure", "cascading"}:
        target = _failure_edge(graph, seed)
    elif kind == "node_failure":
        target = max(graph.nodes, key=lambda node: graph.degree(node), default=None)
    else:
        target = None
    return FailureScenario(
        name=f"{kind}-{seed}",
        kind=kind,
        target=target,
        severity=1.0,
        timestamp=0.0,
        metadata={"seed": seed},
    )


def run_policy_experiment(
    strategy_name: str = "REACTIVE",
    *,
    topology_kind: str = "ring",
    seed: int = 7,
    failure_kind: str = "hard_link_failure",
    num_nodes: int = 12,
    horizon: int = 10,
    predictor: GraphFailurePredictor | None = None,
) -> dict[str, Any]:
    if strategy_name not in STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy_name!r}; choose from {sorted(STRATEGIES)}")
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    graph = generate_topology(topology_kind, num_nodes, seed=seed)
    healthy = GraphSimulator(graph)
    baseline_summary = healthy.build_flow_summary()
    scenario = _scenario_for(graph, failure_kind, seed)
    failure_kind = scenario.kind
    failed_graph = inject_failure(graph, scenario)
    simulator = GraphSimulator(failed_graph, healthy.flows)
    strategy = (
        GNNHeuristicBaseline(predictor)
        if strategy_name == "GNN_HEURISTIC"
        else STRATEGIES[strategy_name]()
    )
    recovered_at: int | None = None
    steps: list[dict[str, Any]] = []
    for tick in range(1, horizon + 1):
        action = strategy.choose_action(simulator)
        result = simulator.simulate_step(action)
        result["tick"] = tick
        result["action"] = action
        steps.append(result)
        if _restored(result["summary"], baseline_summary):
            recovered_at = tick
            break
    final = steps[-1]["summary"] if steps else simulator.build_flow_summary()
    return {
        "strategy": strategy_name,
        "topology": topology_kind,
        "failure_kind": failure_kind,
        "seed": seed,
        "mttr_seconds": float(recovered_at) if recovered_at is not None else None,
        "reconfigurations": simulator.reconfigurations,
        "restored": recovered_at is not None,
        "restoration_rate": 1.0 if recovered_at is not None else 0.0,
        "throughput_ratio": final["throughput_ratio"],
        "mean_latency_ms": final["mean_latency_ms"],
        "mean_loss_pct": final["mean_loss_pct"],
        "steps": steps,
    }


def run_experiment_matrix(
    *,
    topology_kinds: list[str] | None = None,
    failure_kinds: list[str] | None = None,
    strategies: list[str] | None = None,
    seeds: list[int] | None = None,
    num_nodes: int = 12,
    horizon: int = 10,
    output_dir: str | Path | None = None,
    predictor: GraphFailurePredictor | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    topologies = topology_kinds or ["ring", "grid", "barabasi_albert"]
    failures = failure_kinds or ["hard_link_failure", "degradation", "congestion"]
    policy_names = strategies or ["NO_OP", "REACTIVE", "LOAD_AWARE", "GNN_HEURISTIC", "FIXED_FEATURE_GREEDY"]
    trial_seeds = seeds or [1, 2, 3]
    trials: list[dict[str, Any]] = []
    aggregates: list[dict[str, Any]] = []
    for topology in topologies:
        for failure in failures:
            for strategy in policy_names:
                group: list[dict[str, Any]] = []
                for seed in trial_seeds:
                    result = run_policy_experiment(
                        strategy,
                        topology_kind=topology,
                        failure_kind=failure,
                        seed=seed,
                        num_nodes=num_nodes,
                        horizon=horizon,
                        predictor=predictor,
                    )
                    trials.append(result)
                    group.append(result)
                aggregate = compute_experiment_metrics(strategy, group)
                aggregate.update({"topology": topology, "failure_kind": failure})
                aggregates.append(aggregate)
    if output_dir is not None:
        save_experiment_results(trials, aggregates, output_dir)
    return trials, aggregates


def save_experiment_results(
    trials: list[dict[str, Any]],
    aggregates: list[dict[str, Any]],
    output_dir: str | Path,
) -> None:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    for filename, rows in (("trials.csv", trials), ("summary.csv", aggregates)):
        if not rows:
            continue
        fields = sorted({key for row in rows for key in row if key != "steps"})
        with (directory / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows({key: value for key, value in row.items() if key != "steps"} for row in rows)


def save_evaluation_plots(
    trials: list[dict[str, Any]],
    aggregates: list[dict[str, Any]],
    output_dir: str | Path,
) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Plot generation requires matplotlib. Install it with `pip install -e '.[plot]'`."
        ) from exc

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    strategies = sorted({str(row["strategy"]) for row in trials})
    mttr_data = [
        values
        for strategy in strategies
        if (
            values := [
                float(row["mttr_seconds"])
                for row in trials
                if row["strategy"] == strategy and row["mttr_seconds"] is not None
            ]
        )
    ]
    figure, axis = plt.subplots(figsize=(max(7, len(strategies) * 1.2), 4))
    if mttr_data:
        mttr_strategies = [
            strategy
            for strategy in strategies
            if any(row["strategy"] == strategy and row["mttr_seconds"] is not None for row in trials)
        ]
        axis.boxplot(mttr_data, labels=mttr_strategies)
    else:
        axis.text(0.5, 0.5, "No trial recovered within the configured horizon", ha="center")
    axis.set_ylabel("MTTR (seconds; recovered trials)")
    axis.set_title("Recovery time by strategy")
    axis.tick_params(axis="x", labelrotation=25)
    figure.tight_layout()
    path = directory / "mttr_by_strategy.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    saved.append(path)

    scatter_rows = [row for row in trials if row["mttr_seconds"] is not None]
    figure, axis = plt.subplots(figsize=(6, 4))
    for strategy in strategies:
        selected = [row for row in scatter_rows if row["strategy"] == strategy]
        axis.scatter(
            [row["reconfigurations"] for row in selected],
            [row["mttr_seconds"] for row in selected],
            label=strategy,
            alpha=0.75,
        )
    axis.set_xlabel("Reconfiguration operations")
    axis.set_ylabel("MTTR (seconds)")
    axis.set_title("Reconfiguration cost vs recovery time")
    axis.legend(fontsize="small")
    figure.tight_layout()
    path = directory / "reconfigs_vs_mttr.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    saved.append(path)

    topology_names = sorted({str(row["topology"]) for row in aggregates})
    figure, axis = plt.subplots(figsize=(max(7, len(topology_names) * 1.2), 4))
    for strategy in strategies:
        selected = [
            row for row in aggregates
            if row["strategy"] == strategy and row.get("mttr_seconds_mean") is not None
        ]
        selected.sort(key=lambda row: row["topology"])
        axis.plot(
            [row["topology"] for row in selected],
            [row["mttr_seconds_mean"] for row in selected],
            marker="o",
            label=strategy,
        )
    axis.set_ylabel("Mean MTTR (seconds)")
    axis.set_title("Recovery generalization by topology family")
    axis.tick_params(axis="x", labelrotation=25)
    axis.legend(fontsize="small")
    figure.tight_layout()
    path = directory / "generalization_by_topology.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    saved.append(path)
    return saved
