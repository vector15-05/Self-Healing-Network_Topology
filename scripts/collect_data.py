#!/usr/bin/env python3
from __future__ import annotations

from shnet.failures.scenarios import FailureScenario, inject_failure
from shnet.sim.graph_sim import GraphSimulator
from shnet.telemetry.snapshot import graph_to_snapshot
from shnet.topology.generators import generate_topology


def main() -> None:
    graph = generate_topology("barabasi_albert", 12, seed=3)
    scenario = FailureScenario("hard_link_failure", "hard_link_failure", target=(0, 1), severity=1.0)
    graph = inject_failure(graph, scenario)
    simulator = GraphSimulator(graph)
    step = simulator.simulate_step("NO_OP")
    snapshot = graph_to_snapshot(simulator.graph, timestamp=1.0)
    print({
        "nodes": simulator.graph.number_of_nodes(),
        "edges": simulator.graph.number_of_edges(),
        "mean_latency_ms": step["summary"]["mean_latency_ms"],
        "snapshot_nodes": len(snapshot.node_names),
    })


if __name__ == "__main__":
    main()
