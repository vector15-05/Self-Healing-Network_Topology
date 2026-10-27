import networkx as nx
import numpy as np
import pytest

from shnet.failures.scenarios import FailureScenario
from shnet.rl.env import SelfHealingEnv
from shnet.sim.graph_sim import FlowDemand


def _graph() -> nx.Graph:
    graph = nx.path_graph(4)
    nx.set_node_attributes(graph, "switch", "type")
    nx.set_node_attributes(graph, 100.0, "capacity")
    for u, v in graph.edges:
        graph[u][v].update(
            bandwidth_mbps=100.0,
            delay_ms=2.0,
            loss_pct=0.0,
            utilization=0.0,
            cost=1.0,
        )
    return graph


def test_gymnasium_environment_observations_and_masked_recovery() -> None:
    env = SelfHealingEnv(
        _graph(),
        horizon=3,
        failure_scenario=FailureScenario(
            "outage",
            "hard_link_failure",
            target=(1, 2),
        ),
        flows=[FlowDemand(0, 3, 10.0, "test-flow")],
    )
    observation, info = env.reset(seed=17)

    assert env.observation_space.contains(observation)
    assert info["action_mask"].dtype == np.bool_
    assert observation["metrics"][0] == 0.0
    assert env.action_masks()[2]

    next_observation, reward, terminated, truncated, info = env.step(2)
    assert env.observation_space.contains(next_observation)
    assert isinstance(reward, float)
    assert terminated
    assert not truncated
    assert info["recovered"]
    assert info["reconfig_cost"] == 1.0


def test_randomized_failure_reset_is_seeded() -> None:
    env = SelfHealingEnv(_graph(), randomize_failure=True)
    first, _ = env.reset(seed=23)
    second, _ = env.reset(seed=23)

    assert np.array_equal(first["edge_features"], second["edge_features"])


def test_degraded_qos_does_not_count_as_recovered() -> None:
    env = SelfHealingEnv(
        _graph(),
        horizon=2,
        failure_scenario=FailureScenario("degraded", "gradual_degradation"),
        flows=[FlowDemand(0, 3, 10.0, "test-flow")],
    )
    env.reset(seed=1)
    _, _, terminated, truncated, info = env.step(0)

    assert not terminated
    assert not truncated
    assert not info["recovery_complete"]


def test_observation_shapes_are_stable_across_topology_sizes() -> None:
    small = SelfHealingEnv(_graph())
    larger = SelfHealingEnv(nx.cycle_graph(8))

    assert small.action_space.n == larger.action_space.n
    small_observation = small.reset(seed=1)[0]
    larger_observation = larger.reset(seed=1)[0]
    assert small_observation["node_features"].shape == (50, 6)
    assert larger_observation["node_features"].shape == (50, 6)
    assert small.observation_space.contains(small_observation)
    assert larger.observation_space.contains(larger_observation)


def test_graph_feature_extractor_returns_declared_feature_dimension() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("stable_baselines3")
    from shnet.rl.features_extractor import GraphPolicyFeatureExtractor

    env = SelfHealingEnv(_graph())
    observation, _ = env.reset(seed=2)
    batched = {
        key: torch.as_tensor(value).unsqueeze(0)
        for key, value in observation.items()
    }
    extractor = GraphPolicyFeatureExtractor(env.observation_space)

    features = extractor(batched)

    assert features.shape == (1, extractor.features_dim)
