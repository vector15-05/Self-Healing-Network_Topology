from shnet.sim.graph_sim import GraphSimulator
from shnet.topology.generators import generate_topology


def test_graph_simulator_has_valid_mask_and_step() -> None:
    graph = generate_topology("grid", 9, seed=9)
    simulator = GraphSimulator(graph)

    mask = simulator.action_mask()
    assert "NO_OP" in mask
    assert len(mask) >= 2

    step = simulator.simulate_step("NO_OP")
    assert "summary" in step
    assert step["reconfig_cost"] == 0
    assert step["summary"]["connected_ratio"] > 0
