#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from shnet.eval.metrics import (
    run_experiment_matrix,
    save_evaluation_plots,
)
from shnet.models.gnn import GraphFailurePredictor
from shnet.utils.config import load_config


def _as_int_list(values: list[Any], repetitions: int) -> list[int]:
    base = [int(value) for value in values] or [1]
    return (base + list(range(max(base) + 1, max(base) + 1 + repetitions)))[:repetitions]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run seeded baseline comparison experiments.")
    parser.add_argument("--config", type=Path, default=Path("configs/evaluation.yaml"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--nodes", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument(
        "--predictor",
        type=Path,
        help="Optional trained NumPy edge predictor checkpoint for the GNN_HEURISTIC strategy.",
    )
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config).get("evaluation", {})
    output_dir = args.output or Path(config.get("report_dir", "results"))
    predictor = GraphFailurePredictor.load(args.predictor) if args.predictor else None
    repetitions = int(config.get("repeat_runs", 3))
    seeds = _as_int_list(config.get("seeds", [1]), repetitions)
    trials, aggregates = run_experiment_matrix(
        topology_kinds=list(config.get("topology_kinds", ["ring", "grid", "barabasi_albert"])),
        failure_kinds=list(config.get("failure_scenarios", ["hard_link_failure", "gradual_degradation", "congestion"])),
        strategies=list(config.get("strategies", ["NO_OP", "REACTIVE", "LOAD_AWARE", "GNN_HEURISTIC", "FIXED_FEATURE_GREEDY"])),
        seeds=seeds,
        num_nodes=args.nodes,
        horizon=args.horizon,
        output_dir=output_dir,
        predictor=predictor,
    )
    print({"trials": len(trials), "groups": len(aggregates), "output": str(output_dir)})
    if not args.no_plots:
        try:
            plots = save_evaluation_plots(trials, aggregates, output_dir)
            print({"plots": [str(path) for path in plots]})
        except RuntimeError as exc:
            print(f"Plot generation skipped: {exc}")


if __name__ == "__main__":
    main()
