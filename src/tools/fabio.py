"""Fabio route validation client and LangChain tools.

This module provides:
- FabioClient for querying the Fabio routing table
- Route conflict detection in strict mode
- LangChain tools for LLM-driven route validation
"""

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any

import httpx
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@dataclass
class FabioRouteEntry:
    """An entry in the Fabio routing table."""

    service: str
    src: str  # e.g., "myapp.cluster:9999/"
    dst: str  # e.g., "http://10.0.1.12:32456/"
    weight: float
    tags: list[str]


@dataclass
class RouteConflict:
    """Details about a route conflict."""

    existing_route: FabioRouteEntry
    proposed_route: str
    conflict_type: str  # exact_match, path_overlap, hostname_collision
    severity: str  # error, warning


class FabioRouteConflictError(Exception):
    """Raised when strict mode detects a route conflict that should block deployment."""

    def __init__(self, conflict: RouteConflict):
        self.conflict = conflict
        super().__init__(
            f"Route conflict detected ({conflict.conflict_type}): "
            f"'{conflict.proposed_route}' conflicts with existing route "
            f"'{conflict.existing_route.src}' -> '{conflict.existing_route.service}'"
        )


class FabioClient:
    """Client for querying Fabio routing table and validating routes.

    Fabio exposes a routing table API at /api/routes that returns CSV data
    with the current routing configuration.
    """

    def __init__(
        self,
        addr: str | None = None,
        timeout: float = 10.0,
    ):
        """Initialize Fabio client.

        Args:
            addr: Fabio admin address. Defaults to FABIO_ADMIN_ADDR env var.
            timeout: HTTP request timeout in seconds.
        """
        self.addr = addr or os.environ.get("FABIO_ADMIN_ADDR", "http://localhost:9998")
        self.timeout = timeout
        self._routes_cache: list[FabioRouteEntry] | None = None
        self._cache_ttl = 5.0  # Cache routes for 5 seconds

    def get_routes(self, force_refresh: bool = False) -> list[FabioRouteEntry]:
        """Get all routes from Fabio routing table.

        Args:
            force_refresh: Force refresh of cached routes.

        Returns:
            List of FabioRouteEntry objects.
        """
        if self._routes_cache is not None and not force_refresh:
            return self._routes_cache

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(f"{self.addr}/api/routes")
                response.raise_for_status()

                routes = self._parse_routes_csv(response.text)
                self._routes_cache = routes
                return routes

        except httpx.HTTPError as e:
            logger.error(f"Failed to fetch Fabio routes: {e}")
            return []

    def _parse_routes_csv(self, csv_data: str) -> list[FabioRouteEntry]:
        """Parse Fabio CSV route data.

        Format: service,src,dst,weight,tags (no header)
        Example: myapp,myapp.cluster:9999/,http://10.0.1.12:32456/,1.0,
        """
        routes = []
        for line in csv_data.strip().split("\n"):
            if not line:
                continue

            parts = line.split(",")
            if len(parts) >= 4:
                routes.append(
                    FabioRouteEntry(
                        service=parts[0],
                        src=parts[1],
                        dst=parts[2],
                        weight=float(parts[3]) if parts[3] else 1.0,
                        tags=parts[4].split(" ") if len(parts) > 4 and parts[4] else [],
                    )
                )

        return routes

    def check_route_conflict(
        self,
        hostname: str | None = None,
        path: str | None = None,
        port: int = 9999,
    ) -> RouteConflict | None:
        """Check if a proposed route conflicts with existing routes.

        Args:
            hostname: Proposed hostname (e.g., "myapp.cluster").
            path: Proposed path (e.g., "/myapp").
            port: Fabio port (default 9999).

        Returns:
            RouteConflict if conflict found, None otherwise.
        """
        if not hostname and not path:
            return None

        routes = self.get_routes()

        # Build proposed route string
        if hostname:
            proposed = f"{hostname}:{port}/"
            proposed_pattern = proposed
        else:
            proposed = path or "/"
            proposed_pattern = proposed

        for route in routes:
            conflict = self._check_single_route_conflict(route, hostname, path, port)
            if conflict:
                return conflict

        return None

    def _check_single_route_conflict(
        self,
        route: FabioRouteEntry,
        hostname: str | None,
        path: str | None,
        port: int,
    ) -> RouteConflict | None:
        """Check if a single route conflicts with the proposed route."""
        src = route.src

        if hostname:
            proposed = f"{hostname}:{port}/"

            # Exact match
            if src == proposed:
                return RouteConflict(
                    existing_route=route,
                    proposed_route=proposed,
                    conflict_type="exact_match",
                    severity="error",
                )

            # Hostname collision (same hostname, different paths might still work)
            if src.startswith(f"{hostname}:"):
                return RouteConflict(
                    existing_route=route,
                    proposed_route=proposed,
                    conflict_type="hostname_collision",
                    severity="warning",
                )

        if path:
            # Check path-based routing conflicts
            # Extract path from src if it's path-based (no hostname)
            if ":" not in src.split("/")[0]:
                existing_path = "/" + src.lstrip("/")

                # Exact path match
                if existing_path == path:
                    return RouteConflict(
                        existing_route=route,
                        proposed_route=path,
                        conflict_type="exact_match",
                        severity="error",
                    )

                # Path prefix overlap
                if existing_path.startswith(path) or path.startswith(existing_path):
                    return RouteConflict(
                        existing_route=route,
                        proposed_route=path,
                        conflict_type="path_overlap",
                        severity="warning",
                    )

        return None

    def validate_route_strict(
        self,
        hostname: str | None = None,
        path: str | None = None,
        port: int = 9999,
    ) -> None:
        """Validate route in strict mode - raises exception on conflict.

        Args:
            hostname: Proposed hostname.
            path: Proposed path.
            port: Fabio port.

        Raises:
            FabioRouteConflictError: If route conflict detected.
        """
        conflict = self.check_route_conflict(hostname, path, port)
        if conflict and conflict.severity == "error":
            raise FabioRouteConflictError(conflict)

    def get_service_routes(self, service_name: str) -> list[FabioRouteEntry]:
        """Get all routes for a specific service.

        Args:
            service_name: Name of the service.

        Returns:
            List of routes for the service.
        """
        routes = self.get_routes()
        return [r for r in routes if r.service == service_name]

    def get_existing_hostnames(self) -> list[str]:
        """Get all hostnames currently in the routing table.

        Returns:
            List of unique hostnames.
        """
        routes = self.get_routes()
        hostnames = set()

        for route in routes:
            # Extract hostname from src like "myapp.cluster:9999/"
            match = re.match(r"([^:]+):\d+/?", route.src)
            if match:
                hostnames.add(match.group(1))

        return sorted(hostnames)

    def get_existing_paths(self) -> list[str]:
        """Get all path-based routes currently in the routing table.

        Returns:
            List of unique paths.
        """
        routes = self.get_routes()
        paths = set()

        for route in routes:
            # Path-based routes don't have hostname (no colon before first slash)
            if ":" not in route.src.split("/")[0]:
                paths.add("/" + route.src.lstrip("/").rstrip("/"))

        return sorted(paths)

    def suggest_hostname(self, app_name: str, suffix: str = ".cluster") -> str:
        """Suggest a hostname for an application.

        Checks existing hostnames to avoid conflicts.

        Args:
            app_name: Application name.
            suffix: Hostname suffix (e.g., ".cluster").

        Returns:
            Suggested hostname.
        """
        base = f"{app_name}{suffix}"
        existing = set(self.get_existing_hostnames())

        if base not in existing:
            return base

        # Try numbered variants
        for i in range(2, 100):
            candidate = f"{app_name}-{i}{suffix}"
            if candidate not in existing:
                return candidate

        return base  # Fall back to base, will show conflict


# Global client instance (lazy initialization)
_fabio_client: FabioClient | None = None


def get_fabio_client() -> FabioClient:
    """Get or create the global Fabio client instance."""
    global _fabio_client
    if _fabio_client is None:
        _fabio_client = FabioClient()
    return _fabio_client


def set_fabio_client(client: FabioClient) -> None:
    """Set the global Fabio client instance."""
    global _fabio_client
    _fabio_client = client


# LangChain Tools


@tool
def list_fabio_routes() -> str:
    """List all current routes in the Fabio routing table.

    Use this to understand what routes are already configured
    before proposing new routes for an application.

    Returns:
        JSON string with all routing entries.
    """
    client = get_fabio_client()
    routes = client.get_routes(force_refresh=True)

    return json.dumps(
        {
            "total_routes": len(routes),
            "routes": [
                {
                    "service": r.service,
                    "source": r.src,
                    "destination": r.dst,
                    "weight": r.weight,
                }
                for r in routes
            ],
        },
        indent=2,
    )


@tool
def validate_fabio_route(hostname: str | None = None, path: str | None = None) -> str:
    """Validate that a proposed Fabio route doesn't conflict with existing routes.

    Use this before generating a job spec to ensure the proposed
    routing won't cause conflicts.

    Args:
        hostname: Proposed hostname (e.g., "myapp.cluster").
        path: Proposed path for path-based routing (e.g., "/myapp").

    Returns:
        JSON string with validation result.
    """
    if not hostname and not path:
        return json.dumps(
            {"error": "Must provide either hostname or path to validate"},
            indent=2,
        )

    client = get_fabio_client()
    conflict = client.check_route_conflict(hostname=hostname, path=path)

    if conflict:
        return json.dumps(
            {
                "valid": False,
                "conflict": {
                    "type": conflict.conflict_type,
                    "severity": conflict.severity,
                    "existing_service": conflict.existing_route.service,
                    "existing_route": conflict.existing_route.src,
                    "proposed_route": conflict.proposed_route,
                },
                "recommendation": (
                    "Choose a different hostname/path"
                    if conflict.severity == "error"
                    else "Route may work but could cause unexpected behavior"
                ),
            },
            indent=2,
        )
    else:
        route_type = "hostname" if hostname else "path"
        route_value = hostname or path
        return json.dumps(
            {
                "valid": True,
                "route_type": route_type,
                "route_value": route_value,
                "message": f"No conflicts found for {route_type}-based routing",
            },
            indent=2,
        )


@tool
def get_existing_hostnames() -> str:
    """Get all hostnames currently registered in Fabio.

    Use this to see what hostnames are taken and to suggest
    a unique hostname for a new application.

    Returns:
        JSON string with list of existing hostnames.
    """
    client = get_fabio_client()
    hostnames = client.get_existing_hostnames()

    return json.dumps(
        {
            "total_hostnames": len(hostnames),
            "hostnames": hostnames,
        },
        indent=2,
    )


@tool
def suggest_fabio_hostname(app_name: str, suffix: str = ".cluster") -> str:
    """Suggest a unique hostname for an application.

    Checks existing Fabio routes and suggests a hostname that
    doesn't conflict with existing routes.

    Args:
        app_name: Application name to base hostname on.
        suffix: Hostname suffix (default: ".cluster").

    Returns:
        JSON string with suggested hostname.
    """
    client = get_fabio_client()
    suggested = client.suggest_hostname(app_name, suffix)
    existing = client.get_existing_hostnames()

    return json.dumps(
        {
            "app_name": app_name,
            "suggested_hostname": suggested,
            "fabio_tag": f"urlprefix-{suggested}:9999/",
            "is_available": suggested not in existing,
        },
        indent=2,
    )


@tool
def get_service_routes(service_name: str) -> str:
    """Get all Fabio routes for a specific service.

    Use this to see what routes are already configured for
    a service that might need updating.

    Args:
        service_name: Name of the service.

    Returns:
        JSON string with routes for the service.
    """
    client = get_fabio_client()
    routes = client.get_service_routes(service_name)

    if routes:
        return json.dumps(
            {
                "service": service_name,
                "route_count": len(routes),
                "routes": [
                    {
                        "source": r.src,
                        "destination": r.dst,
                        "weight": r.weight,
                    }
                    for r in routes
                ],
            },
            indent=2,
        )
    else:
        return json.dumps(
            {
                "service": service_name,
                "route_count": 0,
                "message": "No existing routes found for this service",
            },
            indent=2,
        )
