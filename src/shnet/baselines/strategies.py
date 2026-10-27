from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from shnet.models.gnn import GraphFailurePredictor
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


class _RecoveryBaseline(BaselineStrategy):
    def _candidate_score(
        self, simulator: GraphSimulator, action: str, *, load_aware: bool = False
    ) -> tuple[float, float]:
        current = simulator.build_flow_summary()
        candidate_graph, metadata = simulator.apply_action(action)
        candidate = GraphSimulator(candidate_graph, simulator.flows, max_paths=simulator.max_paths)
        candidate.risk_scores = simulator.risk_scores.copy()
        projected = candidate.build_flow_summary()
        connectivity_gain = projected["connected_ratio"] - current["connected_ratio"]
        throughput_gain = projected["throughput_ratio"] - current["throughput_ratio"]
        loss_gain = current["mean_loss_pct"] - projected["mean_loss_pct"]
        latency_gain = current["mean_latency_ms"] - projected["mean_latency_ms"]
        load_gain = projected["throughput_ratio"] - current["throughput_ratio"]
        benefit = (
            100.0 * connectivity_gain
            + 20.0 * throughput_gain
            + 0.2 * loss_gain
            + 0.02 * latency_gain
        )
        if load_aware:
            benefit += 5.0 * load_gain
        cost = float(metadata["cost"])
        return benefit - cost * 0.05, benefit

    def _best_action(
        self, simulator: GraphSimulator, *, load_aware: bool = False
    ) -> str:
        choices = [
            action
            for action in simulator.action_mask()
            if action != "NO_OP"
        ]
        if not choices:
            return "NO_OP"
        ranked = sorted(
            (
                self._candidate_score(simulator, action, load_aware=load_aware)[0],
                action,
            )
            for action in choices
        )
        best_score, best_action = ranked[-1]
        # Avoid needless modifications when QoS is already healthy or no action helps.
        return best_action if best_score > 1e-9 else "NO_OP"


class ReactiveShortestPathBaseline(_RecoveryBaseline):
    def __init__(self) -> None:
        super().__init__("REACTIVE_SHORTEST_PATH")

    def choose_action(self, simulator: GraphSimulator, **kwargs: Any) -> str:
        return self._best_action(simulator)


class LoadAwareBaseline(_RecoveryBaseline):
    def __init__(self) -> None:
        super().__init__("LOAD_AWARE")

    def choose_action(self, simulator: GraphSimulator, **kwargs: Any) -> str:
        return self._best_action(simulator, load_aware=True)


class GNNHeuristicBaseline(_RecoveryBaseline):
    def __init__(self, predictor: GraphFailurePredictor | None = None) -> None:
        super().__init__("GNN_HEURISTIC")
        self.predictor = predictor or GraphFailurePredictor()

    def choose_action(self, simulator: GraphSimulator, **kwargs: Any) -> str:
        # Prediction risk is used to prioritize recovery only when an intervention
        # also improves the counterfactual QoS outcome.
        risks = self.predictor.predict(simulator.graph)
        simulator.risk_scores = risks
        action = self._best_action(simulator, load_aware=True)
        if action.startswith("ACTIVATE_BACKUP_LINK:"):
            edge_id = int(action.split(":", 1)[1])
            u, v = simulator.edge_inventory[edge_id]
            risk = risks.get(f"{u}->{v}", risks.get(f"{v}->{u}", 0.0))
            if risk <= 0.0 and simulator.build_flow_summary()["connected_ratio"] >= 0.999:
                return "NO_OP"
        return action


class FixedFeatureGreedyBaseline(_RecoveryBaseline):
    """Non-learning counterfactual heuristic for quick, reproducible comparison."""

    def __init__(self) -> None:
        super().__init__("FIXED_FEATURE_GREEDY")

    def choose_action(self, simulator: GraphSimulator, **kwargs: Any) -> str:
        return self._best_action(simulator, load_aware=True)
