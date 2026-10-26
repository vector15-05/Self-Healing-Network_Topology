from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RyuControllerClient:
    host: str = "127.0.0.1"
    port: int = 6653

    def get_topology(self) -> dict[str, Any]:
        return {"host": self.host, "port": self.port, "topology": []}

    def install_flow(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"status": "dry_run", "args": args, "kwargs": kwargs}

    def remove_flow(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"status": "dry_run", "args": args, "kwargs": kwargs}

    def set_path(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"status": "dry_run", "args": args, "kwargs": kwargs}
