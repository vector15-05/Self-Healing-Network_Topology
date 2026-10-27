#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx

from shnet.failures.scenarios import FailureScenario, failure_labels, inject_failure
from shnet.sim.graph_sim import GraphSimulator
from shnet.telemetry.snapshot import SnapshotStore, TelemetryCollector
from shnet.topology.generators import generate_topology
from shnet.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect a reproducible simulator telemetry sequence.")
    parser.add_argument("--config", type=Path, default=Path("configs/topology.yaml"))
    parser.add_argument("--topology")
    parser.add_argument("--nodes", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--ticks", type=int)
    parser.add_argument("--output", type=Path, default=Path("results/snapshots.jsonl"))
    args = parser.parse_args()
    config = load_config(args.config)
    topology_config = config.get("topology", {})
    simulation_config = config.get("simulation", {})
    topology = args.topology or topology_config.get("kind", "barabasi_albert")
    nodes = (
        args.nodes
        if args.nodes is not None
        else int(topology_config.get("num_nodes", 12))
    )
    seed = args.seed if args.seed is not None else int(topology_config.get("seed", 7))
    ticks = (
        args.ticks
        if args.ticks is not None
        else int(simulation_config.get("horizon", 5))
    )
    if ticks <= 0:
        parser.error("--ticks must be positive")

    graph = generate_topology(topology, nodes, seed=seed)
    centrality = nx.edge_betweenness_centrality(graph)
    target = max(
        graph.edges,
        key=lambda edge: centrality.get(edge, centrality.get((edge[1], edge[0]), 0.0)),
    )
    scenario = FailureScenario(
        "hard_link_failure",
        "hard_link_failure",
        target=target,
        severity=1.0,
        timestamp=1.0,
        metadata={"seed": seed},
    )
    labels = failure_labels(graph, [scenario], horizon_seconds=1.0, timestamp=0.0)
    collector = TelemetryCollector(SnapshotStore(args.output))
    simulator = GraphSimulator(graph)
    collector.collect(
        graph,
        timestamp=0.0,
        flow_summary=simulator.build_flow_summary(),
        labels=labels,
    )

    failed_graph = inject_failure(graph, scenario)
    simulator = GraphSimulator(failed_graph, simulator.flows)
    for tick in range(1, ticks + 1):
        summary = simulator.build_flow_summary()
        collector.collect(
            simulator.graph,
            timestamp=float(tick),
            flow_summary=summary,
            labels={key: 0.0 for key in labels},
        )
    print(
        {
            "snapshots": ticks + 1,
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "failure_edge": [str(node) for node in target],
            "output": str(args.output),
        }
    )


if __name__ == "__main__":
    main()
