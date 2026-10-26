from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from shnet.sim.graph_sim import GraphSimulator


@dataclass
class BaselineStrategy:
    name: str

    def choose_action(self, simulator: GraphSimulator, **kwargs: Any) -> str:
        raise NotImplementedError


class NoOpBaseline(BaselineStrategy):
    def __init__(self) -> None:
        super().__init__("NO_OP")

    def choose_action(self, simulator: GraphSimulator, **kwargs: Any) -> str:
        return "NO_OP"


class ReactiveShortestPathBaseline(BaselineStrategy):
    def __init__(self) -> None:
        super().__init__("REACTIVE_SHORTEST_PATH")

    def choose_action(self, simulator: GraphSimulator, **kwargs: Any) -> str:
        if simulator.graph.number_of_edges() == 0:
            return "NO_OP"
        best_edge = min(
            simulator.graph.edges,
            key=lambda edge: simulator.graph[edge[0]][edge[1]].get("delay_ms", 0.0),
        )
        return f"REPAIR:{best_edge[0]}->{best_edge[1]}"


class LoadAwareBaseline(BaselineStrategy):
    def __init__(self) -> None:
        super().__init__("LOAD_AWARE")

    def choose_action(self, simulator: GraphSimulator, **kwargs: Any) -> str:
        if simulator.graph.number_of_edges() == 0:
            return "NO_OP"
        best_edge = max(
            simulator.graph.edges,
            key=lambda edge: float(simulator.graph[edge[0]][edge[1]].get("utilization", 0.0)),
        )
        return f"REPAIR:{best_edge[0]}->{best_edge[1]}"
