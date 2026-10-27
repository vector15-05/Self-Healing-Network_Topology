"""Optional Ryu REST endpoint for project-specific route weights.

Run alongside ``ryu.app.ofctl_rest`` and ``ryu.app.rest_topology`` with
``ryu-manager``. The standard applications remain the source of topology,
port-statistics, and flow-entry endpoints.
"""

from __future__ import annotations

import json

from ryu.app.wsgi import ControllerBase, WSGIApplication, route
from ryu.base import app_manager
from webob import Response


class ShnetRouteController(ControllerBase):
    def __init__(self, req, link, data, **config):
        super().__init__(req, link, data, **config)
        self.state = data["shnet_route_state"]

    @route("shnet", "/shnet/link-weights", methods=["POST"])
    def set_link_weights(self, req, **kwargs):
        try:
            payload = req.json
            weights = payload["weights"]
            if not isinstance(weights, dict):
                raise ValueError("weights must be an object")
            normalized = {str(key): float(value) for key, value in weights.items()}
            if any(value < 0 for value in normalized.values()):
                raise ValueError("link weights must be non-negative")
        except (KeyError, TypeError, ValueError) as exc:
            return Response(
                status=400,
                content_type="application/json",
                body=json.dumps({"error": str(exc)}).encode("utf-8"),
            )
        self.state["weights"] = normalized
        return Response(
            content_type="application/json",
            body=json.dumps({"status": "updated", "weights": normalized}).encode("utf-8"),
        )

    @route("shnet", "/shnet/link-weights", methods=["GET"])
    def get_link_weights(self, req, **kwargs):
        return Response(
            content_type="application/json",
            body=json.dumps({"weights": self.state["weights"]}).encode("utf-8"),
        )


class ShnetRouteWeightApp(app_manager.RyuApp):
    """Stores policy link weights for the recovery controller integration."""

    _CONTEXTS = {"wsgi": WSGIApplication}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.route_state = {"weights": {}}
        kwargs["wsgi"].register(ShnetRouteController, {"shnet_route_state": self.route_state})
