from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


@dataclass(frozen=True)
class RewardWeights:
    disconnected_flows: float = 10.0
    qos_degradation: float = 1.0
    reconfiguration_cost: float = 2.0
    recovery_time: float = 0.2
    terminal_bonus: float = 10.0

    def __post_init__(self) -> None:
        if any(
            not math.isfinite(value) or value < 0
            for value in (
                self.disconnected_flows,
                self.qos_degradation,
                self.reconfiguration_cost,
                self.recovery_time,
                self.terminal_bonus,
            )
        ):
            raise ValueError("Reward weights must be finite and non-negative")

    @classmethod
    def from_mapping(cls, values: dict[str, Any] | None) -> RewardWeights:
        if values is None:
            return cls()
        return cls(
            disconnected_flows=float(values.get("disconnected_flows", 10.0)),
            qos_degradation=float(values.get("qos_degradation", 1.0)),
            reconfiguration_cost=float(values.get("reconfiguration_cost", 2.0)),
            recovery_time=float(values.get("recovery_time", 0.2)),
            terminal_bonus=float(values.get("terminal_bonus", 10.0)),
        )


def is_recovered(
    summary: dict[str, Any],
    target: dict[str, Any],
    *,
    latency_tolerance: float = 0.15,
    loss_tolerance_pct: float = 0.05,
    throughput_tolerance: float = 0.02,
) -> bool:
    return (
        summary["connected_ratio"] >= target["connected_ratio"] - 1e-9
        and summary["throughput_ratio"] >= target["throughput_ratio"] - throughput_tolerance
        and summary["mean_loss_pct"] <= target["mean_loss_pct"] + loss_tolerance_pct
        and summary["mean_latency_ms"]
        <= target["mean_latency_ms"] * (1.0 + latency_tolerance) + 0.1
    )


def calculate_reward(
    summary: dict[str, Any],
    *,
    reconfiguration_cost: float,
    timestep: int,
    recovery_target: dict[str, Any],
    weights: RewardWeights | None = None,
) -> tuple[float, bool]:
    weights = weights or RewardWeights()
    disconnected_flows = summary["num_flows"] - summary["connected_flows"]
    qos_degradation = (
        (1.0 - summary["throughput_ratio"]) * 5.0
        + summary["mean_loss_pct"] * 0.15
        + summary["mean_latency_ms"] * 0.01
    )
    recovered = is_recovered(summary, recovery_target)
    reward = (
        -weights.disconnected_flows * disconnected_flows
        -weights.qos_degradation * qos_degradation
        -weights.reconfiguration_cost * reconfiguration_cost
        -weights.recovery_time * timestep
        + (weights.terminal_bonus if recovered else 0.0)
    )
    return round(reward, 4), recovered
