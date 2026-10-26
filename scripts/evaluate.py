#!/usr/bin/env python3
from __future__ import annotations

from shnet.eval.metrics import run_policy_experiment


def main() -> None:
    report = run_policy_experiment("NO_OP", topology_kind="grid", seed=4)
    print(report)


if __name__ == "__main__":
    main()
