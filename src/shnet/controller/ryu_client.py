from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class RyuControllerError(RuntimeError):
    pass


@dataclass
class RyuControllerClient:
    """Client for Ryu's REST topology and ofctl REST applications."""

    host: str = "127.0.0.1"
    port: int = 8080
    timeout_seconds: float = 3.0
    flow_rule_changes: int = 0

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def _request(
        self,
        path: str,
        *,
        method: str = "GET",
        payload: dict[str, Any] | None = None,
    ) -> Any:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            f"{self.base_url}{path}",
            data=body,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                content = response.read()
        except (HTTPError, URLError, TimeoutError) as exc:
            raise RyuControllerError(f"Ryu request {method} {path} failed: {exc}") from exc
        if not content:
            return {}
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise RyuControllerError(f"Ryu returned invalid JSON for {path}") from exc

    def get_topology(self) -> dict[str, Any]:
        switches = self._request("/v1.0/topology/switches")
        links = self._request("/v1.0/topology/links")
        return {"switches": switches, "links": links}

    def get_switches(self) -> list[int]:
        switch_ids = self._request("/stats/switches")
        return [int(switch_id) for switch_id in switch_ids]

    def get_port_stats(self, dpid: int) -> Any:
        return self._request(f"/stats/port/{int(dpid)}")

    def get_flow_stats(self, dpid: int) -> Any:
        return self._request(f"/stats/flow/{int(dpid)}")

    def install_flow(self, dpid: int, flow: dict[str, Any]) -> Any:
        result = self._request(
            "/stats/flowentry/add",
            method="POST",
            payload={"dpid": int(dpid), **flow},
        )
        self.flow_rule_changes += 1
        return result

    def remove_flow(self, dpid: int, flow: dict[str, Any]) -> Any:
        result = self._request(
            "/stats/flowentry/delete",
            method="POST",
            payload={"dpid": int(dpid), **flow},
        )
        self.flow_rule_changes += 1
        return result

    def set_path(
        self,
        path: list[int],
        *,
        destination_ip: str,
        priority: int = 100,
    ) -> list[Any]:
        if len(path) < 2:
            raise ValueError("A path must contain at least two switches")
        topology = self.get_topology()
        links = topology["links"]
        installed: list[Any] = []
        for current, following in zip(path, path[1:]):
            output_port = None
            for link in links:
                source = link.get("src", {})
                target = link.get("dst", {})
                if int(source.get("dpid", -1)) == int(current) and int(target.get("dpid", -1)) == int(following):
                    output_port = int(source["port_no"])
                    break
                if int(source.get("dpid", -1)) == int(following) and int(target.get("dpid", -1)) == int(current):
                    output_port = int(target["port_no"])
                    break
            if output_port is None:
                raise RyuControllerError(f"Topology has no link between switches {current} and {following}")
            installed.append(
                self.install_flow(
                    int(current),
                    {
                        "priority": priority,
                        "match": {"eth_type": 0x0800, "ipv4_dst": destination_ip},
                        "actions": [{"type": "OUTPUT", "port": output_port}],
                    },
                )
            )
        return installed

    def set_link_weights(self, weights: dict[str, float]) -> Any:
        """Use an explicitly installed project Ryu app; stock ofctl REST has no weight API."""
        return self._request("/shnet/link-weights", method="POST", payload={"weights": weights})
