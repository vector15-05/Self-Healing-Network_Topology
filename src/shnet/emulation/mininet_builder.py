from __future__ import annotations

from typing import Any

from shnet.topology.generators import generate_topology


class MininetHarness:
    def __init__(self, graph: Any | None = None, *, topology_kind: str = "ring", num_nodes: int = 10) -> None:
        self.graph = graph if graph is not None else generate_topology(topology_kind, num_nodes)
        self.running = False

    def start(self) -> dict[str, Any]:
        self.running = True
        return {"status": "started", "nodes": list(self.graph.nodes), "edges": list(self.graph.edges)}

    def run_traffic(self, *, duration_seconds: int = 5) -> dict[str, Any]:
        self.running = True
        return {"status": "traffic_generated", "duration_seconds": duration_seconds, "edges": len(self.graph.edges)}

    def teardown(self) -> dict[str, Any]:
        self.running = False
        return {"status": "teardown_complete"}
