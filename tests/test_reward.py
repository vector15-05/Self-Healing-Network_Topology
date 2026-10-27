import pytest

from shnet.sim.reward import RewardWeights, calculate_reward, is_recovered


def _summary(*, connected: int, throughput: float, latency: float, loss: float) -> dict:
    return {
        "num_flows": 2,
        "connected_flows": connected,
        "connected_ratio": connected / 2,
        "throughput_ratio": throughput,
        "mean_latency_ms": latency,
        "mean_loss_pct": loss,
    }


def test_reward_includes_recovery_time_and_terminal_bonus() -> None:
    target = _summary(connected=2, throughput=1.0, latency=10.0, loss=0.0)
    restored = _summary(connected=2, throughput=1.0, latency=11.0, loss=0.0)
    weights = RewardWeights(
        disconnected_flows=10.0,
        qos_degradation=1.0,
        reconfiguration_cost=2.0,
        recovery_time=0.5,
        terminal_bonus=10.0,
    )

    reward, recovered = calculate_reward(
        restored,
        reconfiguration_cost=1.0,
        timestep=2,
        recovery_target=target,
        weights=weights,
    )

    assert recovered
    assert reward == 6.89
    assert is_recovered(restored, target)


def test_qos_degradation_prevents_false_recovery() -> None:
    target = _summary(connected=2, throughput=1.0, latency=10.0, loss=0.0)
    degraded = _summary(connected=2, throughput=0.7, latency=30.0, loss=1.0)

    assert not is_recovered(degraded, target)


def test_reward_weights_must_be_finite_and_non_negative() -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        RewardWeights(recovery_time=-0.1)
    with pytest.raises(ValueError, match="finite and non-negative"):
        RewardWeights(terminal_bonus=float("inf"))
