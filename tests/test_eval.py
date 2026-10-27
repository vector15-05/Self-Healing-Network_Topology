from shnet.eval.metrics import run_experiment_matrix, run_policy_experiment


def test_experiment_metrics_are_measured_and_seeded() -> None:
    report = run_policy_experiment(
        "REACTIVE",
        topology_kind="ring",
        seed=3,
        num_nodes=6,
        horizon=5,
    )

    assert report["strategy"] == "REACTIVE"
    assert report["failure_kind"] == "hard_link_failure"
    assert report["mttr_seconds"] is not None
    assert report["reconfigurations"] >= 1
    assert report["steps"]


def test_experiment_matrix_returns_trial_and_aggregate_rows(tmp_path) -> None:
    trials, aggregates = run_experiment_matrix(
        topology_kinds=["ring"],
        failure_kinds=["hard_link_failure"],
        strategies=["NO_OP", "REACTIVE"],
        seeds=[1, 2],
        num_nodes=6,
        horizon=3,
        output_dir=tmp_path,
    )

    assert len(trials) == 4
    assert len(aggregates) == 2
    assert (tmp_path / "trials.csv").exists()
    assert (tmp_path / "summary.csv").exists()
    assert all("restoration_rate_mean" in row for row in aggregates)
