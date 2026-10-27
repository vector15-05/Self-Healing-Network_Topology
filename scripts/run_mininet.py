#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import networkx as nx

from shnet.baselines.strategies import ReactiveShortestPathBaseline
from shnet.emulation.mininet_builder import MininetHarness
from shnet.failures.scenarios import FailureScenario, inject_failure
from shnet.sim.graph_sim import FlowDemand, GraphSimulator
from shnet.topology.generators import generate_topology


def wait_for_port(host: str, port: int, timeout: float, process: subprocess.Popen[bytes] | None) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"Ryu exited early with status {process.returncode}")
        try:
            with socket.create_connection((host, port), timeout=0.25):
                return
        except OSError:
            time.sleep(0.2)
    raise TimeoutError(f"Timed out waiting for {host}:{port}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the real Mininet + Ryu + iperf3 smoke test.")
    parser.add_argument("--nodes", type=int, default=10)
    parser.add_argument("--topology", default="ring")
    parser.add_argument("--duration", type=int, default=3)
    parser.add_argument("--flows", type=int, default=2)
    parser.add_argument("--controller-ip", default="127.0.0.1")
    parser.add_argument("--openflow-port", type=int, default=6653)
    parser.add_argument("--rest-port", type=int, default=8080)
    parser.add_argument("--external-controller", action="store_true")
    parser.add_argument(
        "--recovery-smoke",
        action="store_true",
        help="Run the five-switch link-failure and baseline-recovery acceptance smoke test.",
    )
    parser.add_argument("--output", type=Path, default=Path("results/mininet_smoke.json"))
    args = parser.parse_args()

    if os.geteuid() != 0:
        raise SystemExit("Mininet requires root privileges; run this script with sudo.")
    controller_process: subprocess.Popen[bytes] | None = None
    if args.recovery_smoke:
        graph = nx.path_graph(5)
        nx.set_node_attributes(graph, "switch", "type")
        nx.set_node_attributes(graph, 200.0, "capacity")
        for u, v in graph.edges:
            graph[u][v].update(
                bandwidth_mbps=100.0,
                delay_ms=2.0,
                loss_pct=0.0,
                utilization=0.0,
                active=True,
                cost=1.0,
            )
    else:
        graph = generate_topology(args.topology, args.nodes)
    harness = MininetHarness(
        graph=graph,
        controller_ip=args.controller_ip,
        controller_port=args.openflow_port,
        rest_port=args.rest_port,
    )
    try:
        if not args.external_controller:
            ryu_manager = shutil.which("ryu-manager")
            if ryu_manager is None:
                candidate = Path(sys.executable).with_name("ryu-manager")
                if candidate.is_file():
                    ryu_manager = str(candidate)
            if ryu_manager is None:
                raise RuntimeError(
                    "ryu-manager is not installed. Install Ryu or pass --external-controller "
                    "to use an already-running Ryu instance."
                )
            controller_process = subprocess.Popen(
                [
                    ryu_manager,
                    "--ofp-tcp-listen-port",
                    str(args.openflow_port),
                    "--wsapi-port",
                    str(args.rest_port),
                    "ryu.app.simple_switch_13",
                    "ryu.app.ofctl_rest",
                    "ryu.app.rest_topology",
                    "shnet.controller.ryu_app",
                ]
            )
        wait_for_port(
            args.controller_ip,
            args.openflow_port,
            timeout=20.0,
            process=controller_process,
        )
        wait_for_port(args.controller_ip, args.rest_port, timeout=20.0, process=controller_process)
        started = harness.start()
        if args.recovery_smoke:
            scenario = FailureScenario(
                "smoke-link-failure",
                "hard_link_failure",
                target=(2, 3),
                timestamp=0.0,
            )
            simulator = GraphSimulator(
                inject_failure(graph, scenario),
                [FlowDemand(0, 4, 10.0, "smoke-flow")],
            )
            baseline = ReactiveShortestPathBaseline()
            action = baseline.choose_action(simulator)
            if not action.startswith("ACTIVATE_BACKUP_LINK:"):
                raise RuntimeError(f"Recovery baseline did not select the failed link: {action}")
            harness.verify_connectivity()
            harness.set_link_status(2, 3, up=False)
            failed = harness.verify_connectivity()
            simulator.simulate_step(action)
            harness.set_link_status(2, 3, up=True)
            recovered = harness.verify_connectivity()
            if recovered["packet_loss_pct"] >= 100.0:
                raise RuntimeError("Mininet connectivity did not recover after the baseline action")
            traffic = harness.run_traffic(
                duration_seconds=args.duration,
                flows=args.flows,
            )
            report = {
                "network": started,
                "injected": {"edge": [2, 3], "action": action},
                "failed_connectivity": failed,
                "recovered_connectivity": recovered,
                "flows": traffic,
            }
        else:
            traffic = harness.run_traffic(
                duration_seconds=args.duration,
                flows=args.flows,
            )
            report = {"network": started, "flows": traffic}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
        print({"status": "passed", "report": str(args.output), "flows": len(traffic)})
    finally:
        harness.teardown()
        if controller_process is not None:
            controller_process.terminate()
            try:
                controller_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                controller_process.kill()
                controller_process.wait()


if __name__ == "__main__":
    main()
