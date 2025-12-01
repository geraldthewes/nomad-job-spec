"""Consul API client and LangChain tools for KV and service discovery.

This module provides:
- ConsulClient for querying Consul KV store and service catalog
- Convention storage and retrieval from Consul KV
- LangChain tools for LLM-driven infrastructure queries
"""

import base64
import json
import logging
import os
from dataclasses import dataclass
from typing import Any

import consul
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@dataclass
class ConsulService:
    """Information about a Consul-registered service."""

    name: str
    address: str
    port: int
    tags: list[str]
    health_status: str  # passing, warning, critical
    node: str


@dataclass
class ConsulKVEntry:
    """A Consul KV store entry."""

    key: str
    value: str
    flags: int
    modify_index: int


class ConsulClient:
    """Client for interacting with Consul KV and service catalog.

    Provides access to:
    - KV store for configuration and conventions
    - Service catalog for discovery
    - Health checks for service status
    """

    def __init__(
        self,
        addr: str | None = None,
        token: str | None = None,
        conventions_path: str = "config/nomad-agent/conventions",
    ):
        """Initialize Consul client.

        Args:
            addr: Consul HTTP address. Defaults to CONSUL_HTTP_ADDR env var.
            token: Consul ACL token. Defaults to CONSUL_HTTP_TOKEN env var.
            conventions_path: KV path for agent conventions.
        """
        self.addr = addr or os.environ.get("CONSUL_HTTP_ADDR", "http://localhost:8500")
        self.token = token or os.environ.get("CONSUL_HTTP_TOKEN")
        self.conventions_path = conventions_path

        # Parse host, port, and scheme from address
        # Handle formats like "http://localhost:8500", "localhost:8500", "localhost"
        scheme = "http"
        addr_clean = self.addr

        if addr_clean.startswith("https://"):
            scheme = "https"
            addr_clean = addr_clean.replace("https://", "")
        elif addr_clean.startswith("http://"):
            scheme = "http"
            addr_clean = addr_clean.replace("http://", "")

        # Handle trailing slash
        addr_clean = addr_clean.rstrip("/")

        # Split host and port
        if ":" in addr_clean:
            host, port_str = addr_clean.split(":", 1)
            port = int(port_str)
        else:
            host = addr_clean
            port = 8500

        # python-consul reads CONSUL_HTTP_ADDR internally and expects host:port format,
        # but HashiCorp's standard allows URLs with scheme (e.g., http://host:port).
        # Temporarily set the env var to host:port format for python-consul compatibility.
        original_addr = os.environ.get("CONSUL_HTTP_ADDR")
        os.environ["CONSUL_HTTP_ADDR"] = f"{host}:{port}"
        try:
            self._client = consul.Consul(
                host=host,
                port=port,
                token=self.token,
                scheme=scheme,
            )
        finally:
            # Restore the original environment variable
            if original_addr is not None:
                os.environ["CONSUL_HTTP_ADDR"] = original_addr
            else:
                os.environ.pop("CONSUL_HTTP_ADDR", None)
        self._conventions_cache: dict[str, Any] | None = None

    def get_kv(self, key: str) -> ConsulKVEntry | None:
        """Get a value from Consul KV store.

        Args:
            key: Key path in KV store.

        Returns:
            ConsulKVEntry or None if not found.
        """
        try:
            index, data = self._client.kv.get(key)
            if data is None:
                return None

            # Decode base64 value
            value = data.get("Value")
            if value:
                value = base64.b64decode(value).decode("utf-8")
            else:
                value = ""

            return ConsulKVEntry(
                key=data.get("Key", key),
                value=value,
                flags=data.get("Flags", 0),
                modify_index=data.get("ModifyIndex", 0),
            )
        except consul.ConsulException as e:
            logger.error(f"Consul error getting key {key}: {e}")
            return None

    def get_kv_json(self, key: str) -> dict[str, Any] | None:
        """Get a JSON value from Consul KV store.

        Args:
            key: Key path in KV store.

        Returns:
            Parsed JSON dict or None if not found/invalid.
        """
        entry = self.get_kv(key)
        if entry is None:
            return None

        try:
            return json.loads(entry.value)
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON at key {key}")
            return None

    def list_kv(self, prefix: str) -> list[ConsulKVEntry]:
        """List all keys under a prefix.

        Args:
            prefix: Key prefix to list.

        Returns:
            List of ConsulKVEntry objects.
        """
        try:
            index, data = self._client.kv.get(prefix, recurse=True)
            if data is None:
                return []

            entries = []
            for item in data:
                value = item.get("Value")
                if value:
                    value = base64.b64decode(value).decode("utf-8")
                else:
                    value = ""

                entries.append(
                    ConsulKVEntry(
                        key=item.get("Key", ""),
                        value=value,
                        flags=item.get("Flags", 0),
                        modify_index=item.get("ModifyIndex", 0),
                    )
                )
            return entries
        except consul.ConsulException as e:
            logger.error(f"Consul error listing prefix {prefix}: {e}")
            return []

    def put_kv(self, key: str, value: str) -> bool:
        """Put a value in Consul KV store.

        Args:
            key: Key path in KV store.
            value: Value to store (string).

        Returns:
            True if successful.
        """
        try:
            return self._client.kv.put(key, value)
        except consul.ConsulException as e:
            logger.error(f"Consul error putting key {key}: {e}")
            return False

    def put_kv_json(self, key: str, data: dict[str, Any]) -> bool:
        """Put a JSON value in Consul KV store.

        Args:
            key: Key path in KV store.
            data: Dict to store as JSON.

        Returns:
            True if successful.
        """
        return self.put_kv(key, json.dumps(data, indent=2))

    def get_conventions(self) -> dict[str, Any]:
        """Get agent conventions from Consul KV.

        Returns cached conventions if available.

        Returns:
            Conventions dict or default if not found.
        """
        if self._conventions_cache is not None:
            return self._conventions_cache

        conventions = self.get_kv_json(self.conventions_path)
        if conventions:
            self._conventions_cache = conventions
            return conventions

        # Return defaults if not in Consul
        return self._default_conventions()

    def clear_conventions_cache(self) -> None:
        """Clear the cached conventions."""
        self._conventions_cache = None

    def _default_conventions(self) -> dict[str, Any]:
        """Return default conventions structure."""
        return {
            "vault": {
                "path_patterns": {
                    "database": "secret/data/{app_name}/db",
                    "redis": "secret/data/{app_name}/redis",
                    "api_keys": "secret/data/{app_name}/app",
                    "aws": "secret/data/aws/{app_name}",
                },
                "env_var_mappings": {
                    "DB_PASSWORD": {"path": "db", "key": "password"},
                    "AWS_ACCESS_KEY_ID": {"path": "aws", "key": "access_key"},
                    "AWS_SECRET_ACCESS_KEY": {"path": "aws", "key": "secret_key"},
                },
            },
            "fabio": {
                "default_port": 9999,
                "hostname_suffix": ".cluster",
                "reserved_hostnames": ["registry", "vault", "consul"],
            },
        }

    def list_services(self) -> dict[str, list[str]]:
        """List all registered services with their tags.

        Returns:
            Dict mapping service name to list of tags.
        """
        try:
            index, services = self._client.catalog.services()
            return services
        except consul.ConsulException as e:
            logger.error(f"Consul error listing services: {e}")
            return {}

    def get_service(self, name: str) -> list[ConsulService]:
        """Get all instances of a service.

        Args:
            name: Service name.

        Returns:
            List of ConsulService instances.
        """
        try:
            index, nodes = self._client.catalog.service(name)
            return [
                ConsulService(
                    name=node.get("ServiceName", name),
                    address=node.get("ServiceAddress") or node.get("Address", ""),
                    port=node.get("ServicePort", 0),
                    tags=node.get("ServiceTags", []),
                    health_status="unknown",  # Catalog doesn't include health
                    node=node.get("Node", ""),
                )
                for node in nodes
            ]
        except consul.ConsulException as e:
            logger.error(f"Consul error getting service {name}: {e}")
            return []

    def get_service_health(self, name: str) -> list[ConsulService]:
        """Get service instances with health status.

        Args:
            name: Service name.

        Returns:
            List of ConsulService instances with health info.
        """
        try:
            index, health_data = self._client.health.service(name)

            services = []
            for entry in health_data:
                node = entry.get("Node", {})
                service = entry.get("Service", {})
                checks = entry.get("Checks", [])

                # Determine overall health status
                health_status = "passing"
                for check in checks:
                    status = check.get("Status", "passing")
                    if status == "critical":
                        health_status = "critical"
                        break
                    elif status == "warning" and health_status != "critical":
                        health_status = "warning"

                services.append(
                    ConsulService(
                        name=service.get("Service", name),
                        address=service.get("Address") or node.get("Address", ""),
                        port=service.get("Port", 0),
                        tags=service.get("Tags", []),
                        health_status=health_status,
                        node=node.get("Node", ""),
                    )
                )

            return services
        except consul.ConsulException as e:
            logger.error(f"Consul error getting health for {name}: {e}")
            return []

    def service_exists(self, name: str) -> bool:
        """Check if a service is registered.

        Args:
            name: Service name.

        Returns:
            True if service exists.
        """
        services = self.list_services()
        return name in services


# Global client instance (lazy initialization)
_consul_client: ConsulClient | None = None


def get_consul_client() -> ConsulClient:
    """Get or create the global Consul client instance."""
    global _consul_client
    if _consul_client is None:
        _consul_client = ConsulClient()
    return _consul_client


def set_consul_client(client: ConsulClient) -> None:
    """Set the global Consul client instance."""
    global _consul_client
    _consul_client = client


# LangChain Tools


@tool
def query_consul_kv(key: str) -> str:
    """Query a value from Consul KV store.

    Use this to retrieve configuration values or conventions
    stored in Consul.

    Args:
        key: Key path in Consul KV store.

    Returns:
        JSON string with the value or error.
    """
    client = get_consul_client()
    entry = client.get_kv(key)

    if entry:
        # Try to parse as JSON for pretty output
        try:
            parsed = json.loads(entry.value)
            return json.dumps(
                {
                    "key": entry.key,
                    "value": parsed,
                    "modify_index": entry.modify_index,
                },
                indent=2,
            )
        except json.JSONDecodeError:
            return json.dumps(
                {
                    "key": entry.key,
                    "value": entry.value,
                    "modify_index": entry.modify_index,
                },
                indent=2,
            )
    else:
        return json.dumps({"key": key, "error": "Key not found"}, indent=2)


@tool
def list_consul_services() -> str:
    """List all services registered in Consul.

    Use this to discover what services are available in the cluster
    for service discovery and dependency resolution.

    Returns:
        JSON string with service names and their tags.
    """
    client = get_consul_client()
    services = client.list_services()

    return json.dumps(
        {
            "total_services": len(services),
            "services": [
                {"name": name, "tags": tags} for name, tags in services.items()
            ],
        },
        indent=2,
    )


@tool
def get_service_endpoints(service_name: str) -> str:
    """Get the endpoints (address:port) for a Consul service.

    Use this to find how to connect to a service dependency,
    including health status of each instance.

    Args:
        service_name: Name of the service to look up.

    Returns:
        JSON string with service instances and their health.
    """
    client = get_consul_client()
    services = client.get_service_health(service_name)

    if services:
        return json.dumps(
            {
                "service": service_name,
                "instance_count": len(services),
                "instances": [
                    {
                        "address": f"{s.address}:{s.port}",
                        "node": s.node,
                        "tags": s.tags,
                        "health": s.health_status,
                    }
                    for s in services
                ],
                "healthy_count": sum(1 for s in services if s.health_status == "passing"),
            },
            indent=2,
        )
    else:
        return json.dumps(
            {
                "service": service_name,
                "error": "Service not found",
                "instances": [],
            },
            indent=2,
        )


@tool
def get_agent_conventions() -> str:
    """Get the naming conventions for this Nomad agent.

    Retrieves the conventions from Consul KV that define:
    - Vault path patterns for secrets
    - Environment variable to Vault path mappings
    - Fabio routing conventions

    Returns:
        JSON string with conventions configuration.
    """
    client = get_consul_client()
    conventions = client.get_conventions()

    return json.dumps(
        {
            "source": "consul_kv" if client._conventions_cache else "defaults",
            "conventions": conventions,
        },
        indent=2,
    )


@tool
def check_service_dependency(service_name: str) -> str:
    """Check if a service dependency exists and is healthy.

    Use this before generating a job spec to verify that
    required dependencies (like databases) are available.

    Args:
        service_name: Name of the dependency service.

    Returns:
        JSON string with availability status.
    """
    client = get_consul_client()
    services = client.get_service_health(service_name)

    healthy = [s for s in services if s.health_status == "passing"]

    return json.dumps(
        {
            "service": service_name,
            "available": len(healthy) > 0,
            "total_instances": len(services),
            "healthy_instances": len(healthy),
            "consul_dns": f"{service_name}.service.consul",
            "recommendation": (
                f"Use '{service_name}.service.consul' for service discovery"
                if healthy
                else f"Warning: No healthy instances of {service_name} found"
            ),
        },
        indent=2,
    )
