from __future__ import annotations

import json
import random
import shutil
import subprocess
import time
from typing import Any

import networkx as nx

from shnet.topology.generators import generate_topology


class MininetHarness:
    """Optional real Mininet backend; importing this module does not require Mininet."""

    def __init__(
        self,
        graph: nx.Graph | None = None,
        *,
        topology_kind: str = "ring",
        num_nodes: int = 10,
        controller_ip: str = "127.0.0.1",
        controller_port: int = 6653,
        rest_port: int = 8080,
        seed: int = 7,
    ) -> None:
        self.graph = graph.copy() if graph is not None else generate_topology(topology_kind, num_nodes, seed=seed)
        self.controller_ip = controller_ip
        self.controller_port = controller_port
        self.rest_port = rest_port
        self.seed = seed
        self.net: Any | None = None
        self.node_names: dict[Any, str] = {}
        self.switch_dpids: dict[Any, int] = {}
        self.host_names: list[str] = []
        self.host_names_by_node: dict[Any, str] = {}
        self.host_names_by_switch: dict[Any, str] = {}
        self._servers: dict[str, Any] = {}
        self.flow_demands: list[dict[str, Any]] = []

    @staticmethod
    def _mininet_imports() -> tuple[Any, Any, Any, Any]:
        try:
            from mininet.link import TCLink
            from mininet.net import Mininet
            from mininet.node import OVSSwitch, RemoteController
            from mininet.topo import Topo
        except ImportError as exc:
            raise RuntimeError(
                "Mininet is not installed in this Python environment. Install Mininet and "
                "Open vSwitch on a supported Linux host, then run with root privileges."
            ) from exc
        return Mininet, OVSSwitch, RemoteController, (Topo, TCLink)

    def build_topology(self) -> Any:
        Mininet, _, _, components = self._mininet_imports()
        del Mininet
        Topo, _ = components
        graph = self.graph
        switch_nodes = [node for node, attrs in graph.nodes(data=True) if attrs.get("type") != "host"]
        if not switch_nodes:
            raise ValueError("Mininet topology must contain at least one switch node")
        self.node_names = {node: f"s{index + 1}" for index, node in enumerate(switch_nodes)}
        self.switch_dpids = {node: index + 1 for index, node in enumerate(switch_nodes)}
        host_nodes = [node for node, attrs in graph.nodes(data=True) if attrs.get("type") == "host"]
        self.host_names_by_node = {node: f"h{index + 1}" for index, node in enumerate(host_nodes)}
        auto_hosts = not host_nodes
        self.host_names_by_switch = (
            {node: f"h{index + 1}" for index, node in enumerate(switch_nodes)}
            if auto_hosts
            else {}
        )

        class GraphTopo(Topo):
            def build(inner_self) -> None:
                switch_map: dict[Any, str] = {}
                for node in switch_nodes:
                    name = self.node_names[node]
                    inner_self.addSwitch(
                        name,
                        protocols="OpenFlow13",
                        dpid=f"{self.switch_dpids[node]:016x}",
                    )
                    switch_map[node] = name

                host_map: dict[Any, str] = {}
                for index, node in enumerate(host_nodes):
                    name = self.host_names_by_node[node]
                    inner_self.addHost(name)
                    host_map[node] = name
                if auto_hosts:
                    for index, node in enumerate(switch_nodes):
                        host = self.host_names_by_switch[node]
                        inner_self.addHost(host)
                        inner_self.addLink(host, switch_map[node])

                for u, v, attrs in graph.edges(data=True):
                    left = switch_map.get(u, host_map.get(u))
                    right = switch_map.get(v, host_map.get(v))
                    if left is None or right is None:
                        raise ValueError(f"Unable to map topology edge {(u, v)!r} to Mininet nodes")
                    params: dict[str, Any] = {
                        "bw": max(0.01, float(attrs.get("bandwidth_mbps", 100.0))),
                        "delay": f"{max(0.0, float(attrs.get('delay_ms', 1.0)))}ms",
                        "loss": min(100.0, max(0.0, float(attrs.get("loss_pct", 0.0)))),
                    }
                    inner_self.addLink(left, right, **params)

        topology = GraphTopo()
        self.host_names = [f"h{index + 1}" for index in range(len(host_nodes) if host_nodes else len(switch_nodes))]
        return topology

    def start(self) -> dict[str, Any]:
        if self.net is not None:
            raise RuntimeError("Mininet is already running")
        Mininet, OVSSwitch, RemoteController, components = self._mininet_imports()
        _, TCLink = components
        topology = self.build_topology()
        net = Mininet(
            topo=topology,
            controller=None,
            switch=OVSSwitch,
            link=TCLink,
            autoSetMacs=True,
            build=True,
        )
        try:
            net.addController(
                "c0",
                controller=RemoteController,
                ip=self.controller_ip,
                port=self.controller_port,
            )
            net.start()
        except Exception:
            net.stop()
            raise
        self.net = net
        try:
            for u, v, attrs in self.graph.edges(data=True):
                if not attrs.get("active", True):
                    self.set_link_status(u, v, up=False)
            for node, attrs in self.graph.nodes(data=True):
                if not attrs.get("active", True):
                    self.set_node_status(node, up=False)
        except Exception:
            self.teardown()
            raise
        return {
            "status": "started",
            "switches": len(self.node_names),
            "hosts": len(self.host_names),
            "edges": self.graph.number_of_edges(),
            "controller": f"{self.controller_ip}:{self.controller_port}",
        }

    def run_traffic(
        self,
        *,
        duration_seconds: int = 5,
        flows: int = 1,
        protocol: str = "tcp",
        rate_mbps: float = 5.0,
    ) -> list[dict[str, Any]]:
        if self.net is None:
            raise RuntimeError("Start Mininet before generating traffic")
        if duration_seconds <= 0 or flows <= 0 or rate_mbps <= 0:
            raise ValueError("duration_seconds, flows, and rate_mbps must be positive")
        protocol = protocol.lower()
        if protocol not in {"tcp", "udp"}:
            raise ValueError("protocol must be 'tcp' or 'udp'")
        if shutil.which("iperf3") is None:
            raise RuntimeError("iperf3 is not installed on the host")
        if len(self.host_names) < 2:
            raise RuntimeError("At least two Mininet hosts are required for traffic generation")

        hosts = [self.net.get(name) for name in self.host_names]
        for host in hosts:
            process = self._servers.get(host.name)
            if process is None or process.poll() is not None:
                self._servers[host.name] = host.popen(
                    ["iperf3", "-s"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
        time.sleep(0.2)
        rng = random.Random(self.seed + len(self.flow_demands))
        results: list[dict[str, Any]] = []
        for index in range(flows):
            source_index = rng.randrange(len(hosts))
            destination_index = (source_index + rng.randrange(1, len(hosts))) % len(hosts)
            source = hosts[source_index]
            destination = hosts[destination_index]
            command = [
                "iperf3",
                "-c",
                destination.IP(),
                "-J",
                "-t",
                str(duration_seconds),
                "-b",
                f"{rate_mbps}M",
            ]
            if protocol == "udp":
                command.append("-u")
            client = source.popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                stdout, stderr = client.communicate(timeout=duration_seconds + 15)
            except subprocess.TimeoutExpired:
                client.terminate()
                client.wait(timeout=2)
                raise RuntimeError(f"iperf3 flow {index} timed out")
            stdout = stdout or b""
            stderr = stderr or b""
            if client.returncode != 0:
                raise RuntimeError(
                    f"iperf3 flow {index} failed ({client.returncode}): "
                    f"{stderr.decode(errors='replace').strip()}"
                )
            try:
                measurement = json.loads(stdout.decode())
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"iperf3 returned invalid JSON for flow {index}") from exc
            record = {
                "flow_id": f"flow-{len(self.flow_demands)}",
                "source": source.name,
                "destination": destination.name,
                "protocol": protocol,
                "demand_mbps": rate_mbps,
                "duration_seconds": duration_seconds,
                "result": measurement.get("end", {}),
            }
            self.flow_demands.append(record)
            results.append(record)
        return results

    def set_link_status(self, node_a: Any, node_b: Any, *, up: bool) -> None:
        if self.net is None:
            raise RuntimeError("Start Mininet before changing link status")
        left = self.node_names.get(node_a, self.host_names_by_node.get(node_a, str(node_a)))
        right = self.node_names.get(node_b, self.host_names_by_node.get(node_b, str(node_b)))
        status = "up" if up else "down"
        self.net.configLinkStatus(left, right, status)
        if self.graph.has_edge(node_a, node_b):
            self.graph[node_a][node_b]["active"] = up

    def set_node_status(self, node: Any, *, up: bool) -> None:
        if self.net is None:
            raise RuntimeError("Start Mininet before changing node status")
        switch_name = self.node_names.get(node)
        if switch_name is not None:
            switch = self.net.get(switch_name)
            if up:
                switch.start(self.net.controllers)
            else:
                switch.stop()
        else:
            host_name = self.host_names_by_node.get(node)
            if host_name is None:
                raise ValueError(f"{node!r} is not a node in the Mininet topology")
            for neighbor in self.graph.neighbors(node):
                neighbor_name = self.node_names.get(
                    neighbor,
                    self.host_names_by_node.get(neighbor),
                )
                if neighbor_name is not None:
                    self.net.configLinkStatus(
                        host_name,
                        neighbor_name,
                        "up" if up else "down",
                    )
        self.graph.nodes[node]["active"] = up

    def set_link_impairment(
        self,
        node_a: Any,
        node_b: Any,
        *,
        delay_ms: float | None = None,
        loss_pct: float | None = None,
        bandwidth_mbps: float | None = None,
    ) -> None:
        if self.net is None:
            raise RuntimeError("Start Mininet before changing link impairment")
        if not self.graph.has_edge(node_a, node_b):
            raise ValueError(f"Unknown link {(node_a, node_b)!r}")
        left = self.node_names.get(node_a, self.host_names_by_node.get(node_a, str(node_a)))
        right = self.node_names.get(node_b, self.host_names_by_node.get(node_b, str(node_b)))
        links = self.net.linksBetween(left, right)
        if not links:
            raise RuntimeError(f"Mininet does not contain link {left}-{right}")
        for link in links:
            for interface in (link.intf1, link.intf2):
                config: dict[str, Any] = {}
                if delay_ms is not None:
                    config["delay"] = f"{max(0.0, delay_ms)}ms"
                if loss_pct is not None:
                    config["loss"] = min(100.0, max(0.0, loss_pct))
                if bandwidth_mbps is not None:
                    config["bw"] = max(0.01, bandwidth_mbps)
                interface.config(**config)
        attrs = self.graph[node_a][node_b]
        if delay_ms is not None:
            attrs["delay_ms"] = delay_ms
        if loss_pct is not None:
            attrs["loss_pct"] = loss_pct
        if bandwidth_mbps is not None:
            attrs["bandwidth_mbps"] = bandwidth_mbps

    def verify_connectivity(self, *, timeout_seconds: int = 1) -> dict[str, Any]:
        if self.net is None:
            raise RuntimeError("Start Mininet before verifying connectivity")
        hosts = [self.net.get(name) for name in self.host_names]
        packet_loss_pct = float(self.net.ping(hosts, timeout=timeout_seconds))
        return {
            "packet_loss_pct": packet_loss_pct,
            "connected": packet_loss_pct < 100.0,
            "hosts": len(hosts),
        }

    def host_ip_for_switch(self, switch_node: Any) -> str:
        if self.net is None:
            raise RuntimeError("Start Mininet before looking up host addresses")
        host_name = self.host_names_by_switch.get(
            switch_node,
            self.host_names_by_node.get(switch_node),
        )
        if host_name is None:
            raise ValueError(f"No attached host is configured for switch {switch_node!r}")
        return str(self.net.get(host_name).IP())

    def teardown(self) -> dict[str, Any]:
        for server in self._servers.values():
            if server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()
        self._servers.clear()
        if self.net is not None:
            self.net.stop()
            self.net = None
        return {"status": "teardown_complete"}

    def __enter__(self) -> MininetHarness:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.teardown()
