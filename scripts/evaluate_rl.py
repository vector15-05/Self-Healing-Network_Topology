#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import networkx as nx

from shnet.failures.scenarios import FailureScenario
from shnet.rl.env import SelfHealingEnv
from shnet.topology.generators import generate_topology


def _failure(graph: nx.Graph, seed: int, kind: str) -> FailureScenario:
    target: Any | None = None
    if kind == "hard_link_failure":
        centrality = nx.edge_betweenness_centrality(graph)
        target = max(
            graph.edges,
            key=lambda edge: centrality.get(edge, centrality.get((edge[1], edge[0]), 0.0)),
        )
    elif kind == "node_failure":
        target = max(graph.nodes, key=lambda node: graph.degree(node))
    return FailureScenario(
        f"evaluation-{seed}-{kind}",
        kind,
        target=target,
        timestamp=0.0,
        metadata={"seed": seed},
    )


def evaluate_policy(
    model: Any,
    *,
    topology: str,
    nodes: int,
    failure_kind: str,
    seed: int,
    horizon: int,
) -> dict[str, Any]:
    graph = generate_topology(topology, nodes, seed=seed)
    env = SelfHealingEnv(
        graph,
        horizon=horizon,
        failure_scenario=_failure(graph, seed, failure_kind),
    )
    observation, info = env.reset(seed=seed)
    total_reward = 0.0
    recovery_tick: int | None = None
    for tick in range(1, horizon + 1):
        action, _ = model.predict(
            observation,
            action_masks=info["action_mask"],
            deterministic=True,
        )
        observation, reward, terminated, truncated, info = env.step(int(action))
        total_reward += reward
        if info["recovered"]:
            recovery_tick = tick
        if terminated or truncated:
            break
    summary = info["summary"]
    return {
        "topology": topology,
        "nodes": nodes,
        "failure_kind": failure_kind,
        "seed": seed,
        "mttr_seconds": float(recovery_tick) if recovery_tick is not None else None,
        "reconfigurations": env.simulator.reconfigurations,
        "restored": recovery_tick is not None,
        "throughput_ratio": summary["throughput_ratio"],
        "mean_latency_ms": summary["mean_latency_ms"],
        "mean_loss_pct": summary["mean_loss_pct"],
        "total_reward": total_reward,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained MaskablePPO recovery policy.")
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/rl_policy_trials.csv"))
    parser.add_argument("--topologies", nargs="+", default=["ring", "grid", "barabasi_albert"])
    parser.add_argument("--nodes", type=int, default=12)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument(
        "--failures",
        nargs="+",
        default=["hard_link_failure", "gradual_degradation", "congestion", "node_failure"],
    )
    parser.add_argument("--horizon", type=int, default=10)
    args = parser.parse_args()
    try:
        from sb3_contrib import MaskablePPO
    except ImportError as exc:
        raise SystemExit(
            "Policy evaluation requires sb3-contrib and PyTorch. "
            "Install them with `python -m pip install -e '.[rl]'`."
        ) from exc
    if not args.policy.exists():
        parser.error(f"Policy checkpoint does not exist: {args.policy}")
    model = MaskablePPO.load(args.policy)
    rows = [
        evaluate_policy(
            model,
            topology=topology,
            nodes=args.nodes,
            failure_kind=failure,
            seed=seed,
            horizon=args.horizon,
        )
        for topology in args.topologies
        for failure in args.failures
        for seed in args.seeds
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print({"trials": len(rows), "output": str(args.output)})


if __name__ == "__main__":
    main()
