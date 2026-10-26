from shnet.failures.scenarios import FailureScenario, build_failure_schedule, inject_failure
from shnet.topology.generators import generate_topology


def test_failure_schedule_and_injection() -> None:
    schedule = build_failure_schedule(3, seed=2)
    assert len(schedule) == 3
    assert schedule[0].kind in {"hard_link_failure", "node_failure", "gradual_degradation", "congestion", "cascading"}

    graph = generate_topology("ring", 8, seed=4)
    scenario = FailureScenario("gradual_degradation", "gradual_degradation", severity=0.7)
    degraded = inject_failure(graph, scenario)
    for _, _, edge_data in degraded.edges(data=True):
        assert edge_data["delay_ms"] >= 2.0
        assert edge_data["loss_pct"] >= 0.0
