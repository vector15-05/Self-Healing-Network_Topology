from shnet.eval.metrics import run_policy_experiment


def test_experiment_metrics_are_reported() -> None:
    report = run_policy_experiment("NO_OP", topology_kind="ring", seed=3)

    assert report["strategy"] == "NO_OP"
    assert "mean_reward" in report
    assert "steps" in report
    assert report["steps"] >= 1
