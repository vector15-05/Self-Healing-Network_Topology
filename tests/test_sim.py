import networkx as nx

from shnet.baselines.strategies import ReactiveShortestPathBaseline
from shnet.failures.scenarios import FailureScenario, inject_failure
from shnet.sim.graph_sim import FlowDemand, GraphSimulator


def _path_graph() -> nx.Graph:
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


def test_failure_action_restores_connectivity_and_is_minimal() -> None:
    graph = _path_graph()
    flow = FlowDemand(0, 3, 10.0, "test-flow")
    scenario = FailureScenario("outage", "hard_link_failure", target=(1, 2))
    failed_graph = inject_failure(graph, scenario)
    simulator = GraphSimulator(failed_graph, [flow])

    before = simulator.build_flow_summary()
    assert before["connected_ratio"] == 0.0
    valid_actions = simulator.action_mask()
    assert "NO_OP" in valid_actions
    assert "ACTIVATE_BACKUP_LINK:1" in valid_actions

    baseline = ReactiveShortestPathBaseline()
    action = baseline.choose_action(simulator)
    assert action == "ACTIVATE_BACKUP_LINK:1"
    counterfactual = simulator.counterfactual(action)
    assert counterfactual["connected_ratio"] == 1.0
    assert simulator.graph[1][2]["active"] is False

    packet = simulator.simulate_step(action)
    assert packet["summary"]["connected_ratio"] == 1.0
    assert packet["reconfig_cost"] == 1.0
    assert packet["recovered"] is True


def test_action_mask_rejects_repair_on_healthy_graph() -> None:
    simulator = GraphSimulator(_path_graph(), [FlowDemand(0, 3, 10.0)])
    actions = simulator.action_mask()
    assert "NO_OP" in actions
    assert not any(action.startswith("ACTIVATE_BACKUP_LINK:") for action in actions)


def test_flow_assignment_uses_alternate_path_when_shortest_is_saturated() -> None:
    graph = nx.Graph()
    graph.add_edges_from(
        [
            (0, 1, {"bandwidth_mbps": 5.0, "delay_ms": 1.0}),
            (1, 3, {"bandwidth_mbps": 5.0, "delay_ms": 1.0}),
            (0, 2, {"bandwidth_mbps": 5.0, "delay_ms": 3.0}),
            (2, 3, {"bandwidth_mbps": 5.0, "delay_ms": 3.0}),
        ]
    )
    flows = [
        FlowDemand(0, 3, 5.0, "first"),
        FlowDemand(0, 3, 5.0, "second"),
    ]

    summary = GraphSimulator(graph, flows).build_flow_summary()

    assert summary["throughput_mbps"] == 10.0
    assert summary["flows"]["first"]["path"] != summary["flows"]["second"]["path"]
