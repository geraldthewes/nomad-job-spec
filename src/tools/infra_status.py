"""Infrastructure health checking and status reporting.

This module provides centralized health checking for infrastructure services
(Nomad, Consul, Fabio, Vault) with actionable error messages and fix suggestions.
"""

import logging
import os
import socket
from dataclasses import dataclass, field
from typing import Callable

import consul
import httpx
import nomad

logger = logging.getLogger(__name__)


@dataclass
class InfraStatus:
    """Status of a single infrastructure service."""

    service: str  # "nomad", "consul", "fabio", "vault"
    available: bool
    address: str  # What address was attempted
    error: str | None = None  # Error message if failed
    suggestion: str | None = None  # Fix suggestion for the user


@dataclass
class InfraHealthReport:
    """Aggregated health report for all infrastructure services."""

    statuses: list[InfraStatus] = field(default_factory=list)

    @property
    def all_healthy(self) -> bool:
        """Check if all services are available."""
        return all(s.available for s in self.statuses)

    def get_failures(self) -> list[InfraStatus]:
        """Get list of failed services."""
        return [s for s in self.statuses if not s.available]

    def get_healthy(self) -> list[InfraStatus]:
        """Get list of healthy services."""
        return [s for s in self.statuses if s.available]


def _parse_connection_error(error: Exception, service: str, addr: str) -> tuple[str, str]:
    """Parse a connection error and return (error_msg, suggestion).

    Args:
        error: The exception that occurred.
        service: Service name (nomad, consul, etc.).
        addr: Address that was attempted.

    Returns:
        Tuple of (error message, suggestion).
    """
    error_str = str(error).lower()

    # Connection refused
    if "connection refused" in error_str or "[errno 111]" in error_str:
        return (
            "Connection refused",
            f"Check if {service.title()} is running at {addr}",
        )

    # Timeout
    if "timeout" in error_str or "timed out" in error_str:
        return (
            "Connection timeout",
            f"Service at {addr} not responding - check network/firewall",
        )

    # DNS resolution failure
    if "name or service not known" in error_str or "nodename nor servname" in error_str:
        host = addr.replace("http://", "").replace("https://", "").split(":")[0]
        return (
            f"DNS resolution failed for '{host}'",
            "Check DNS settings or use IP address instead",
        )

    # SSL/TLS errors
    if "ssl" in error_str or "certificate" in error_str:
        return (
            "SSL/TLS error",
            "Check certificate configuration or use http:// instead of https://",
        )

    # Generic network error
    if "network" in error_str or "unreachable" in error_str:
        return (
            "Network unreachable",
            f"Check network connectivity to {addr}",
        )

    # Default fallback
    return (
        str(error)[:100],  # Truncate long error messages
        f"Check {service.title()} service and network configuration",
    )


def _parse_http_error(status_code: int, service: str, addr: str) -> tuple[str, str]:
    """Parse an HTTP error status code and return (error_msg, suggestion).

    Args:
        status_code: HTTP status code.
        service: Service name.
        addr: Address that was attempted.

    Returns:
        Tuple of (error message, suggestion).
    """
    token_env_vars = {
        "nomad": "NOMAD_TOKEN",
        "consul": "CONSUL_HTTP_TOKEN",
        "vault": "VAULT_TOKEN",
        "fabio": None,  # Fabio typically doesn't need auth
    }

    token_var = token_env_vars.get(service.lower())

    if status_code == 401:
        if token_var:
            return (
                "Authentication required",
                f"Set {token_var} environment variable with valid ACL token",
            )
        return ("Authentication required", "Configure authentication credentials")

    if status_code == 403:
        if token_var:
            return (
                "Access denied (insufficient permissions)",
                f"Check {token_var} has required permissions",
            )
        return ("Access denied", "Check token permissions")

    if status_code == 404:
        return (
            "API endpoint not found",
            f"Check {service.title()} version compatibility or address configuration",
        )

    if status_code == 500:
        return (
            "Internal server error",
            f"Check {service.title()} server logs for details",
        )

    if status_code == 502 or status_code == 503:
        return (
            f"Service unavailable (HTTP {status_code})",
            f"{service.title()} may be starting up or overloaded",
        )

    return (
        f"HTTP error {status_code}",
        f"Check {service.title()} server status",
    )


def check_nomad_health(
    addr: str | None = None,
    token: str | None = None,
    timeout: float = 5.0,
) -> InfraStatus:
    """Check Nomad server health.

    Args:
        addr: Nomad server address. Defaults to NOMAD_ADDR env var.
        token: Nomad ACL token. Defaults to NOMAD_TOKEN env var.
        timeout: Connection timeout in seconds.

    Returns:
        InfraStatus with availability and any error details.
    """
    addr = addr or os.environ.get("NOMAD_ADDR", "http://localhost:4646")
    token = token or os.environ.get("NOMAD_TOKEN")

    try:
        # Parse address for nomad client
        host = addr.replace("http://", "").replace("https://", "").split(":")[0]
        port_str = addr.split(":")[-1] if ":" in addr.split("/")[-1] else "4646"
        port = int(port_str.split("/")[0])  # Handle trailing paths

        client = nomad.Nomad(
            host=host,
            port=port,
            secure=addr.startswith("https"),
            token=token,
            timeout=timeout,
        )

        # Try to get agent info - this validates connectivity
        agent_info = client.agent.self()

        # Check if we got valid data
        if not agent_info or "config" not in agent_info:
            return InfraStatus(
                service="Nomad",
                available=False,
                address=addr,
                error="Invalid response from agent",
                suggestion="Check Nomad server health and version",
            )

        return InfraStatus(
            service="Nomad",
            available=True,
            address=addr,
        )

    except Exception as e:
        error_str = str(e)

        # Handle specific nomad-python errors
        if "self does not exist" in error_str:
            return InfraStatus(
                service="Nomad",
                available=False,
                address=addr,
                error="Agent API returned unexpected response",
                suggestion="Check Nomad server is healthy and API is accessible",
            )

        # Handle HTTP status codes if available
        if hasattr(e, "response") and hasattr(e.response, "status_code"):
            error_msg, suggestion = _parse_http_error(e.response.status_code, "nomad", addr)
        else:
            error_msg, suggestion = _parse_connection_error(e, "nomad", addr)

        logger.debug(f"Nomad health check failed: {e}")
        return InfraStatus(
            service="Nomad",
            available=False,
            address=addr,
            error=error_msg,
            suggestion=suggestion,
        )


def check_consul_health(
    addr: str | None = None,
    token: str | None = None,
    timeout: float = 5.0,
) -> InfraStatus:
    """Check Consul server health.

    Args:
        addr: Consul HTTP address. Defaults to CONSUL_HTTP_ADDR env var.
        token: Consul ACL token. Defaults to CONSUL_HTTP_TOKEN env var.
        timeout: Connection timeout in seconds.

    Returns:
        InfraStatus with availability and any error details.
    """
    addr = addr or os.environ.get("CONSUL_HTTP_ADDR", "http://localhost:8500")
    token = token or os.environ.get("CONSUL_HTTP_TOKEN")

    # Parse address (handle URL format)
    scheme = "http"
    addr_clean = addr

    if addr_clean.startswith("https://"):
        scheme = "https"
        addr_clean = addr_clean.replace("https://", "")
    elif addr_clean.startswith("http://"):
        addr_clean = addr_clean.replace("http://", "")

    addr_clean = addr_clean.rstrip("/")

    if ":" in addr_clean:
        host, port_str = addr_clean.split(":", 1)
        port = int(port_str)
    else:
        host = addr_clean
        port = 8500

    try:
        # Use httpx for direct health check (simpler than consul library)
        health_url = f"{scheme}://{host}:{port}/v1/status/leader"
        headers = {"X-Consul-Token": token} if token else {}

        with httpx.Client(timeout=timeout) as client:
            response = client.get(health_url, headers=headers)
            response.raise_for_status()

            # Should return the leader address as a string
            leader = response.text.strip('"')
            if not leader:
                return InfraStatus(
                    service="Consul",
                    available=False,
                    address=addr,
                    error="No leader elected",
                    suggestion="Consul cluster may not be healthy - check cluster status",
                )

        return InfraStatus(
            service="Consul",
            available=True,
            address=addr,
        )

    except httpx.HTTPStatusError as e:
        error_msg, suggestion = _parse_http_error(e.response.status_code, "consul", addr)
        logger.debug(f"Consul health check failed: {e}")
        return InfraStatus(
            service="Consul",
            available=False,
            address=addr,
            error=error_msg,
            suggestion=suggestion,
        )

    except Exception as e:
        error_msg, suggestion = _parse_connection_error(e, "consul", addr)
        logger.debug(f"Consul health check failed: {e}")
        return InfraStatus(
            service="Consul",
            available=False,
            address=addr,
            error=error_msg,
            suggestion=suggestion,
        )


def check_fabio_health(
    addr: str | None = None,
    timeout: float = 5.0,
) -> InfraStatus:
    """Check Fabio load balancer health.

    Args:
        addr: Fabio admin address. Defaults to FABIO_ADMIN_ADDR env var.
        timeout: Connection timeout in seconds.

    Returns:
        InfraStatus with availability and any error details.
    """
    addr = addr or os.environ.get("FABIO_ADMIN_ADDR", "http://localhost:9998")

    try:
        # Check the routes API endpoint
        routes_url = f"{addr}/api/routes"

        with httpx.Client(timeout=timeout) as client:
            response = client.get(routes_url)
            response.raise_for_status()

        return InfraStatus(
            service="Fabio",
            available=True,
            address=addr,
        )

    except httpx.HTTPStatusError as e:
        error_msg, suggestion = _parse_http_error(e.response.status_code, "fabio", addr)
        logger.debug(f"Fabio health check failed: {e}")
        return InfraStatus(
            service="Fabio",
            available=False,
            address=addr,
            error=error_msg,
            suggestion=suggestion,
        )

    except Exception as e:
        error_msg, suggestion = _parse_connection_error(e, "fabio", addr)
        logger.debug(f"Fabio health check failed: {e}")
        return InfraStatus(
            service="Fabio",
            available=False,
            address=addr,
            error=error_msg,
            suggestion=suggestion,
        )


def check_vault_health(
    addr: str | None = None,
    token: str | None = None,
    timeout: float = 5.0,
) -> InfraStatus:
    """Check Vault server health.

    Args:
        addr: Vault server address. Defaults to VAULT_ADDR env var.
        token: Vault token. Defaults to VAULT_TOKEN env var.
        timeout: Connection timeout in seconds.

    Returns:
        InfraStatus with availability and any error details.
    """
    addr = addr or os.environ.get("VAULT_ADDR", "http://localhost:8200")
    token = token or os.environ.get("VAULT_TOKEN")

    try:
        # Check the health endpoint (doesn't require auth)
        health_url = f"{addr}/v1/sys/health"

        with httpx.Client(timeout=timeout) as client:
            # Note: Vault returns different status codes for different states
            # 200 = initialized, unsealed, active
            # 429 = unsealed, standby
            # 472 = disaster recovery mode
            # 473 = performance standby
            # 501 = not initialized
            # 503 = sealed
            response = client.get(health_url)

            if response.status_code == 200 or response.status_code == 429:
                # Check if token is needed and provided
                if not token:
                    return InfraStatus(
                        service="Vault",
                        available=True,
                        address=addr,
                        error="No token provided",
                        suggestion="Set VAULT_TOKEN to enable secret access",
                    )
                return InfraStatus(
                    service="Vault",
                    available=True,
                    address=addr,
                )

            if response.status_code == 501:
                return InfraStatus(
                    service="Vault",
                    available=False,
                    address=addr,
                    error="Vault not initialized",
                    suggestion="Run 'vault operator init' to initialize Vault",
                )

            if response.status_code == 503:
                return InfraStatus(
                    service="Vault",
                    available=False,
                    address=addr,
                    error="Vault is sealed",
                    suggestion="Run 'vault operator unseal' to unseal Vault",
                )

            # Other unexpected status
            return InfraStatus(
                service="Vault",
                available=False,
                address=addr,
                error=f"Unexpected status: {response.status_code}",
                suggestion="Check Vault server status",
            )

    except Exception as e:
        error_msg, suggestion = _parse_connection_error(e, "vault", addr)
        logger.debug(f"Vault health check failed: {e}")
        return InfraStatus(
            service="Vault",
            available=False,
            address=addr,
            error=error_msg,
            suggestion=suggestion,
        )


def check_all_infrastructure(
    nomad_addr: str | None = None,
    nomad_token: str | None = None,
    consul_addr: str | None = None,
    consul_token: str | None = None,
    fabio_addr: str | None = None,
    vault_addr: str | None = None,
    vault_token: str | None = None,
    timeout: float = 5.0,
) -> InfraHealthReport:
    """Check health of all infrastructure services.

    Args:
        nomad_addr: Nomad server address.
        nomad_token: Nomad ACL token.
        consul_addr: Consul HTTP address.
        consul_token: Consul ACL token.
        fabio_addr: Fabio admin address.
        vault_addr: Vault server address.
        vault_token: Vault token.
        timeout: Connection timeout in seconds.

    Returns:
        InfraHealthReport with status of all services.
    """
    report = InfraHealthReport()

    # Check each service
    report.statuses.append(check_nomad_health(nomad_addr, nomad_token, timeout))
    report.statuses.append(check_consul_health(consul_addr, consul_token, timeout))
    report.statuses.append(check_fabio_health(fabio_addr, timeout))
    report.statuses.append(check_vault_health(vault_addr, vault_token, timeout))

    return report


def check_infrastructure_from_settings(settings) -> InfraHealthReport:
    """Check health of all infrastructure services using Settings object.

    Args:
        settings: Application Settings object.

    Returns:
        InfraHealthReport with status of all services.
    """
    return check_all_infrastructure(
        nomad_addr=settings.nomad_addr,
        nomad_token=settings.nomad_token,
        consul_addr=settings.consul_http_addr,
        consul_token=settings.consul_http_token,
        fabio_addr=settings.fabio_admin_addr,
        vault_addr=settings.vault_addr,
        vault_token=os.environ.get("VAULT_TOKEN"),  # Not in settings
    )
