"""Nomad version detection for feature auto-detection.

This module provides:
- NomadVersion dataclass with feature detection methods
- Auto-detection of Nomad server version
- Vault format selection based on Nomad version
"""

import logging
import os
import re
from dataclasses import dataclass

import nomad

logger = logging.getLogger(__name__)


@dataclass
class NomadVersion:
    """Nomad server version with feature detection."""

    major: int
    minor: int
    patch: int
    prerelease: str | None = None

    def __str__(self) -> str:
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            version += f"-{self.prerelease}"
        return version

    def supports_native_vault_env(self) -> bool:
        """Check if Nomad supports native vault env stanza (1.4+).

        Nomad 1.4 introduced the `vault.env` stanza for direct secret injection
        without template blocks.

        Returns:
            True if native vault env is supported.
        """
        return (self.major, self.minor) >= (1, 4)

    def supports_template_parser(self) -> bool:
        """Check if Nomad supports template parser option (1.5+).

        Nomad 1.5 added the `template.parse` option for parsing template output.

        Returns:
            True if template parser is supported.
        """
        return (self.major, self.minor) >= (1, 5)

    def supports_consul_connect_native(self) -> bool:
        """Check if Nomad supports Consul Connect native (1.3+).

        Returns:
            True if Consul Connect native is supported.
        """
        return (self.major, self.minor) >= (1, 3)

    def supports_csi_volumes(self) -> bool:
        """Check if Nomad supports CSI volumes (0.11+).

        Returns:
            True if CSI volumes are supported.
        """
        return (self.major, self.minor) >= (0, 11) or self.major >= 1

    def supports_scaling_policy(self) -> bool:
        """Check if Nomad supports scaling policies (0.11+).

        Returns:
            True if scaling policies are supported.
        """
        return (self.major, self.minor) >= (0, 11) or self.major >= 1

    @classmethod
    def parse(cls, version_string: str) -> "NomadVersion":
        """Parse a version string into NomadVersion.

        Args:
            version_string: Version string like "1.4.3" or "1.5.0-rc1".

        Returns:
            NomadVersion instance.
        """
        # Remove 'v' prefix if present
        version_string = version_string.lstrip("v")

        # Match version pattern
        match = re.match(
            r"^(\d+)\.(\d+)\.(\d+)(?:-(.+))?$",
            version_string,
        )

        if not match:
            logger.warning(f"Could not parse version: {version_string}, assuming 1.0.0")
            return cls(major=1, minor=0, patch=0)

        return cls(
            major=int(match.group(1)),
            minor=int(match.group(2)),
            patch=int(match.group(3)),
            prerelease=match.group(4),
        )


def get_nomad_version(
    addr: str | None = None,
    token: str | None = None,
) -> NomadVersion:
    """Get the Nomad server version.

    Args:
        addr: Nomad server address. Defaults to NOMAD_ADDR env var.
        token: Nomad ACL token. Defaults to NOMAD_TOKEN env var.

    Returns:
        NomadVersion instance.
    """
    addr = addr or os.environ.get("NOMAD_ADDR", "http://localhost:4646")
    token = token or os.environ.get("NOMAD_TOKEN")

    try:
        # Initialize Nomad client
        client = nomad.Nomad(
            host=addr.replace("http://", "").replace("https://", "").split(":")[0],
            port=int(addr.split(":")[-1]) if ":" in addr.split("/")[-1] else 4646,
            secure=addr.startswith("https"),
            token=token,
            timeout=10,
        )

        # Get agent info which includes version
        agent_info = client.agent.self()
        version_string = agent_info.get("config", {}).get("Version", "1.0.0")

        return NomadVersion.parse(version_string)

    except Exception as e:
        error_str = str(e).lower()

        # Provide more specific error messages based on error type
        if "connection refused" in error_str or "[errno 111]" in error_str:
            logger.warning(
                f"Nomad connection refused at {addr}. "
                f"Check if Nomad is running. Using default version 1.4.0"
            )
        elif "self does not exist" in error_str:
            logger.warning(
                f"Nomad agent API returned unexpected response at {addr}. "
                f"Check Nomad server health. Using default version 1.4.0"
            )
        elif "timeout" in error_str or "timed out" in error_str:
            logger.warning(
                f"Nomad connection timeout at {addr}. "
                f"Check network connectivity. Using default version 1.4.0"
            )
        elif "401" in str(e) or "unauthorized" in error_str:
            logger.warning(
                f"Nomad authentication failed at {addr}. "
                f"Check NOMAD_TOKEN. Using default version 1.4.0"
            )
        else:
            logger.warning(f"Failed to get Nomad version from {addr}: {e}. Using default 1.4.0")

        # Return a reasonable default that supports most features
        return NomadVersion(major=1, minor=4, patch=0)


def detect_vault_format(
    addr: str | None = None,
    token: str | None = None,
) -> str:
    """Detect the appropriate Vault format based on Nomad version.

    Args:
        addr: Nomad server address.
        token: Nomad ACL token.

    Returns:
        'env_stanza' for Nomad 1.4+, 'template' for older versions.
    """
    version = get_nomad_version(addr, token)

    if version.supports_native_vault_env():
        return "env_stanza"
    else:
        return "template"


# Cache for version to avoid repeated API calls
_cached_version: NomadVersion | None = None


def get_cached_nomad_version(
    addr: str | None = None,
    token: str | None = None,
    force_refresh: bool = False,
) -> NomadVersion:
    """Get cached Nomad version.

    Args:
        addr: Nomad server address.
        token: Nomad ACL token.
        force_refresh: Force refresh of cached version.

    Returns:
        NomadVersion instance.
    """
    global _cached_version

    if _cached_version is None or force_refresh:
        _cached_version = get_nomad_version(addr, token)

    return _cached_version


def clear_version_cache() -> None:
    """Clear the cached Nomad version."""
    global _cached_version
    _cached_version = None
