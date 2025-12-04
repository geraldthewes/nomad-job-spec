"""Vault API client and LangChain tools for secrets discovery and validation.

This module provides:
- VaultClient for querying Vault API (list secrets, validate paths, read metadata)
- LangChain tools for LLM-driven secret path suggestions
- Convention-based path mapping from environment variables
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Literal

import hvac
from hvac.exceptions import Forbidden, InvalidPath, VaultError
from langchain_core.tools import tool

from src.memory import search_env_var_config

logger = logging.getLogger(__name__)


@dataclass
class VaultSecretMetadata:
    """Metadata about a Vault secret path."""

    path: str
    keys: list[str]
    version: int | None = None
    created_time: str | None = None
    updated_time: str | None = None


@dataclass
class VaultPathSuggestion:
    """A suggested Vault path mapping for an environment variable."""

    env_var: str
    suggested_path: str
    key: str
    confidence: float  # 0.0-1.0 based on naming conventions


@dataclass
class EnvVarConfig:
    """Configuration for an environment variable with its source.

    Environment variables can come from different sources:
    - fixed: Static value set directly in the job spec (e.g., APP_HOST=0.0.0.0)
    - consul: Value from Consul KV store (non-secret configuration)
    - vault: Secret value from HashiCorp Vault
    """

    name: str  # Environment variable name
    source: Literal["fixed", "consul", "vault", "unknown"]  # Source type
    value: str  # Fixed value, Consul KV path, or Vault path
    confidence: float = 0.0  # 0.0-1.0 confidence in the suggestion

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "source": self.source,
            "value": self.value,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EnvVarConfig":
        """Create from dictionary representation."""
        return cls(
            name=data["name"],
            source=data["source"],
            value=data["value"],
            confidence=data.get("confidence", 0.0),
        )


@dataclass
class VaultConventions:
    """Conventions for Vault path patterns and environment variable mappings."""

    path_patterns: dict[str, str] = field(default_factory=dict)
    env_var_mappings: dict[str, dict[str, str]] = field(default_factory=dict)
    default_policy_pattern: str = "{app_name}-policy"

    @classmethod
    def default(cls) -> "VaultConventions":
        """Return default conventions."""
        return cls(
            path_patterns={
                "database": "secret/data/{app_name}/db",
                "redis": "secret/data/{app_name}/redis",
                "api_keys": "secret/data/{app_name}/app",
                "aws": "secret/data/aws/{app_name}",
                "s3": "secret/data/aws/{app_name}",
            },
            env_var_mappings={
                "DATABASE_URL": {"path": "db", "key": "connection_string"},
                "DB_PASSWORD": {"path": "db", "key": "password"},
                "DB_USER": {"path": "db", "key": "user"},
                "DB_HOST": {"path": "db", "key": "host"},
                "REDIS_URL": {"path": "redis", "key": "url"},
                "REDIS_PASSWORD": {"path": "redis", "key": "password"},
                "API_KEY": {"path": "app", "key": "api_key"},
                "SECRET_KEY": {"path": "app", "key": "secret_key"},
                "AWS_ACCESS_KEY_ID": {"path": "aws", "key": "access_key"},
                "AWS_SECRET_ACCESS_KEY": {"path": "aws", "key": "secret_key"},
                "AWS_REGION": {"path": "aws", "key": "region"},
            },
        )


class VaultClient:
    """Client for interacting with HashiCorp Vault.

    Supports KV v2 secret engine for reading metadata and listing secrets.
    Does NOT read actual secret values - only metadata for security.
    """

    def __init__(
        self,
        addr: str | None = None,
        token: str | None = None,
        namespace: str | None = None,
    ):
        """Initialize Vault client.

        Args:
            addr: Vault server address. Defaults to VAULT_ADDR env var.
            token: Vault token. Defaults to VAULT_TOKEN env var.
            namespace: Vault namespace (Enterprise only).
        """
        self.addr = addr or os.environ.get("VAULT_ADDR", "http://localhost:8200")
        self.token = token or os.environ.get("VAULT_TOKEN")
        self.namespace = namespace or os.environ.get("VAULT_NAMESPACE")

        # Note: No warning logged here - infrastructure health check handles
        # reporting token issues to the user in a unified way

        self._client = hvac.Client(
            url=self.addr,
            token=self.token,
            namespace=self.namespace,
        )
        self._conventions = VaultConventions.default()

    @property
    def is_authenticated(self) -> bool:
        """Check if client is authenticated."""
        try:
            return self._client.is_authenticated()
        except VaultError:
            return False

    def set_conventions(self, conventions: VaultConventions) -> None:
        """Set the conventions to use for path suggestions."""
        self._conventions = conventions

    def list_secrets(self, path: str = "secret/metadata/") -> list[str]:
        """List secret paths under the given prefix.

        Args:
            path: Path prefix to list. Should be in metadata format for KV v2.

        Returns:
            List of secret paths/keys.
        """
        try:
            # Ensure path uses metadata prefix for KV v2
            if not path.startswith("secret/metadata"):
                path = path.replace("secret/data/", "secret/metadata/")
                if not path.startswith("secret/metadata"):
                    path = f"secret/metadata/{path.lstrip('secret/')}"

            response = self._client.secrets.kv.v2.list_secrets(
                path=path.replace("secret/metadata/", ""),
                mount_point="secret",
            )
            return response.get("data", {}).get("keys", [])
        except InvalidPath:
            logger.debug(f"Vault path not found: {path}")
            return []
        except Forbidden:
            logger.warning(
                f"Vault access denied to list: {path}. "
                f"Check VAULT_TOKEN has 'list' capability on this path."
            )
            return []
        except VaultError as e:
            error_str = str(e).lower()
            if "connection refused" in error_str:
                logger.error(
                    f"Vault connection refused at {self.addr}. "
                    f"Check if Vault is running."
                )
            elif "timeout" in error_str:
                logger.error(
                    f"Vault connection timeout at {self.addr}. "
                    f"Check network connectivity."
                )
            else:
                logger.error(f"Vault error listing secrets at {self.addr}: {e}")
            return []

    def read_metadata(self, path: str) -> VaultSecretMetadata | None:
        """Read metadata about a secret (not the actual secret value).

        Args:
            path: Full path to the secret (e.g., secret/data/myapp/db).

        Returns:
            VaultSecretMetadata or None if not found.
        """
        try:
            # Convert to KV v2 path format
            secret_path = path.replace("secret/data/", "").replace("secret/", "")

            # Read metadata
            metadata = self._client.secrets.kv.v2.read_secret_metadata(
                path=secret_path,
                mount_point="secret",
            )

            # Read the secret to get keys (without exposing values)
            secret = self._client.secrets.kv.v2.read_secret_version(
                path=secret_path,
                mount_point="secret",
            )

            data = metadata.get("data", {})
            secret_data = secret.get("data", {}).get("data", {})

            return VaultSecretMetadata(
                path=path,
                keys=list(secret_data.keys()),
                version=data.get("current_version"),
                created_time=data.get("created_time"),
                updated_time=data.get("updated_time"),
            )
        except InvalidPath:
            logger.debug(f"Vault secret not found: {path}")
            return None
        except Forbidden:
            logger.warning(
                f"Vault access denied to read: {path}. "
                f"Check VAULT_TOKEN has 'read' capability on this path."
            )
            return None
        except VaultError as e:
            error_str = str(e).lower()
            if "connection refused" in error_str:
                logger.error(
                    f"Vault connection refused at {self.addr}. "
                    f"Check if Vault is running."
                )
            elif "timeout" in error_str:
                logger.error(
                    f"Vault connection timeout at {self.addr}. "
                    f"Check network connectivity."
                )
            else:
                logger.error(f"Vault error reading metadata at {self.addr}: {e}")
            return None

    def validate_path(self, path: str) -> tuple[bool, str | None]:
        """Validate that a Vault path exists and is accessible.

        Args:
            path: Full path to validate.

        Returns:
            Tuple of (is_valid, error_message).
        """
        try:
            secret_path = path.replace("secret/data/", "").replace("secret/", "")
            self._client.secrets.kv.v2.read_secret_metadata(
                path=secret_path,
                mount_point="secret",
            )
            return True, None
        except InvalidPath:
            return False, f"Path not found: {path}"
        except Forbidden:
            return False, f"Access denied to path: {path}"
        except VaultError as e:
            return False, f"Vault error: {e}"

    def suggest_mappings(
        self,
        env_vars: list[str],
        app_name: str,
    ) -> list[VaultPathSuggestion]:
        """Suggest Vault path mappings for environment variables.

        Uses conventions and validates paths exist in Vault.

        Args:
            env_vars: List of environment variable names.
            app_name: Application name for path interpolation.

        Returns:
            List of path suggestions with confidence scores.
        """
        suggestions = []

        for env_var in env_vars:
            env_upper = env_var.upper()

            # Check direct mapping in conventions
            if env_upper in self._conventions.env_var_mappings:
                mapping = self._conventions.env_var_mappings[env_upper]
                path_key = mapping["path"]
                secret_key = mapping["key"]

                # Find the full path pattern
                full_path = None
                for pattern_name, pattern in self._conventions.path_patterns.items():
                    if path_key in pattern_name or pattern_name in path_key:
                        full_path = pattern.format(app_name=app_name)
                        break

                if not full_path:
                    # Default pattern
                    full_path = f"secret/data/{app_name}/{path_key}"

                # Validate path exists
                is_valid, _ = self.validate_path(full_path)
                confidence = 0.9 if is_valid else 0.5

                suggestions.append(
                    VaultPathSuggestion(
                        env_var=env_var,
                        suggested_path=full_path,
                        key=secret_key,
                        confidence=confidence,
                    )
                )
            else:
                # Try to infer from naming patterns
                suggestion = self._infer_from_name(env_var, app_name)
                if suggestion:
                    suggestions.append(suggestion)

        return suggestions

    def _infer_from_name(
        self, env_var: str, app_name: str
    ) -> VaultPathSuggestion | None:
        """Infer a Vault path from environment variable naming patterns."""
        env_upper = env_var.upper()

        # Common patterns
        patterns = [
            (["PASSWORD", "PASS", "PWD"], "password", 0.7),
            (["SECRET", "KEY"], "key", 0.6),
            (["TOKEN"], "token", 0.6),
            (["API"], "api_key", 0.5),
            (["USER", "USERNAME"], "user", 0.5),
            (["HOST", "ADDR", "URL"], "host", 0.4),
        ]

        for keywords, default_key, confidence in patterns:
            if any(kw in env_upper for kw in keywords):
                # Try to determine the service from the variable name
                service = "app"
                for svc in ["DB", "DATABASE", "REDIS", "MONGO", "POSTGRES", "MYSQL"]:
                    if svc in env_upper:
                        service = "db" if "DB" in svc or "DATABASE" in svc else svc.lower()
                        break

                path = f"secret/data/{app_name}/{service}"
                is_valid, _ = self.validate_path(path)

                return VaultPathSuggestion(
                    env_var=env_var,
                    suggested_path=path,
                    key=default_key,
                    confidence=confidence if is_valid else confidence * 0.5,
                )

        return None


def suggest_env_configs(
    env_vars: list[str],
    app_name: str,
    vault_client: VaultClient | None = None,
) -> list[EnvVarConfig]:
    """Suggest environment variable configurations with appropriate sources.

    Analyzes variable names to infer the best source (fixed, consul, or vault)
    and provides smart defaults where applicable.

    Args:
        env_vars: List of environment variable names.
        app_name: Application name for path interpolation.
        vault_client: Optional VaultClient for Vault path validation.

    Returns:
        List of EnvVarConfig with suggested sources and values.
    """
    configs = []

    # Fixed value patterns with smart defaults
    fixed_patterns: dict[str, tuple[str, float]] = {
        # Exact matches
        "HOST": ("0.0.0.0", 0.8),
        "APP_HOST": ("0.0.0.0", 0.8),
        "BIND_HOST": ("0.0.0.0", 0.8),
        "LISTEN_HOST": ("0.0.0.0", 0.8),
        "PORT": ("8080", 0.6),
        "APP_PORT": ("8080", 0.6),
        "HTTP_PORT": ("8080", 0.6),
        "LISTEN_PORT": ("8080", 0.6),
        "CONSUL_HOST": ("consul.service.consul", 0.6),
        "CONSUL_HTTP_ADDR": ("http://consul.service.consul:8500", 0.6),
        "CONSUL_ADDR": ("consul.service.consul:8500", 0.6),
        "CONSUL_PORT": ("8500", 0.6),
        # Vault infrastructure
        "VAULT_ADDR": ("http://vault.service.consul:8200", 0.6),
        "VAULT_SKIP_VERIFY": ("false", 0.5),
        "LOG_LEVEL": ("info", 0.5),
        "DEBUG": ("false", 0.5),
        "ENVIRONMENT": ("production", 0.4),
        "ENV": ("production", 0.4),
    }

    # Secret patterns that should use Vault
    secret_keywords = [
        "PASSWORD",
        "PASS",
        "PWD",
        "SECRET",
        "KEY",
        "TOKEN",
        "CREDENTIAL",
        "PRIVATE",
        "API_KEY",
        "APIKEY",
        "AUTH",
    ]

    # AWS-specific patterns (always Vault)
    aws_mappings: dict[str, str] = {
        "AWS_ACCESS_KEY_ID": "access_key",
        "AWS_SECRET_ACCESS_KEY": "secret_key",
        "AWS_SESSION_TOKEN": "session_token",
        "AWS_REGION": "region",
    }

    for env_var in env_vars:
        env_upper = env_var.upper()

        # Check memory first for previously configured variables (global)
        memory_result = search_env_var_config(env_upper)
        if memory_result:
            configs.append(
                EnvVarConfig(
                    name=env_var,
                    source=memory_result.source,  # type: ignore[arg-type]
                    value=memory_result.value_pattern,
                    confidence=0.8,  # High confidence for remembered configs
                )
            )
            continue

        # Check for exact fixed pattern match first
        if env_upper in fixed_patterns:
            default_value, confidence = fixed_patterns[env_upper]
            configs.append(
                EnvVarConfig(
                    name=env_var,
                    source="fixed",
                    value=default_value,
                    confidence=confidence,
                )
            )
            continue

        # Check for AWS patterns (always Vault)
        if env_upper in aws_mappings:
            key = aws_mappings[env_upper]
            vault_path = f"secret/data/aws/{app_name}#{key}"
            # Validate if client available
            confidence = 0.5
            if vault_client:
                is_valid, _ = vault_client.validate_path(f"secret/data/aws/{app_name}")
                confidence = 0.9 if is_valid else 0.5
            configs.append(
                EnvVarConfig(
                    name=env_var,
                    source="vault",
                    value=vault_path,
                    confidence=confidence,
                )
            )
            continue

        # Check if it looks like a secret (Vault)
        is_secret = any(kw in env_upper for kw in secret_keywords)
        if is_secret:
            # Determine service category from variable name
            service = "app"
            for svc in ["DB", "DATABASE", "REDIS", "MONGO", "POSTGRES", "MYSQL", "RABBIT", "KAFKA"]:
                if svc in env_upper:
                    service = "db" if svc in ["DB", "DATABASE", "POSTGRES", "MYSQL"] else svc.lower()
                    break

            # Determine key name
            key = "value"
            for kw in ["PASSWORD", "PASS", "PWD"]:
                if kw in env_upper:
                    key = "password"
                    break
            for kw in ["TOKEN"]:
                if kw in env_upper:
                    key = "token"
                    break
            for kw in ["SECRET", "KEY"]:
                if kw in env_upper:
                    key = "key"
                    break

            vault_path = f"secret/data/{app_name}/{service}#{key}"
            confidence = 0.5
            if vault_client:
                is_valid, _ = vault_client.validate_path(f"secret/data/{app_name}/{service}")
                confidence = 0.7 if is_valid else 0.5
            configs.append(
                EnvVarConfig(
                    name=env_var,
                    source="vault",
                    value=vault_path,
                    confidence=confidence,
                )
            )
            continue

        # Check for URL/connection patterns (likely Consul for non-secrets)
        url_keywords = ["URL", "ADDR", "ENDPOINT", "HOST", "URI"]
        is_url_pattern = any(kw in env_upper for kw in url_keywords)
        if is_url_pattern:
            # Use Consul KV path
            # Convert variable name to a sensible path component
            path_component = env_var.lower().replace("_", "/")
            consul_path = f"{app_name}/config/{path_component}"
            configs.append(
                EnvVarConfig(
                    name=env_var,
                    source="consul",
                    value=consul_path,
                    confidence=0.4,
                )
            )
            continue

        # Default: mark as unknown - requires user input
        configs.append(
            EnvVarConfig(
                name=env_var,
                source="unknown",
                value="<requires input>",
                confidence=0.0,
            )
        )

    return configs


# Global client instance (lazy initialization)
_vault_client: VaultClient | None = None


def get_vault_client() -> VaultClient:
    """Get or create the global Vault client instance."""
    global _vault_client
    if _vault_client is None:
        _vault_client = VaultClient()
    return _vault_client


def set_vault_client(client: VaultClient) -> None:
    """Set the global Vault client instance."""
    global _vault_client
    _vault_client = client


# LangChain Tools


@tool
def list_vault_secrets(path: str = "secret/metadata/") -> str:
    """List available secrets under a Vault path.

    Use this to discover what secrets exist in Vault before suggesting
    mappings for environment variables.

    Args:
        path: Vault path prefix to list (default: secret/metadata/).

    Returns:
        JSON string with list of secret paths.
    """
    client = get_vault_client()
    secrets = client.list_secrets(path)
    return json.dumps({"path": path, "secrets": secrets}, indent=2)


@tool
def suggest_vault_mappings(env_vars: str, app_name: str) -> str:
    """Suggest Vault secret path mappings for environment variables.

    Based on naming conventions, suggests which Vault paths and keys
    should be used for each environment variable. Validates that
    suggested paths exist in Vault.

    Args:
        env_vars: Comma-separated list of environment variable names.
        app_name: Application name for path interpolation.

    Returns:
        JSON string with suggested mappings and confidence scores.
    """
    client = get_vault_client()
    env_list = [e.strip() for e in env_vars.split(",")]
    suggestions = client.suggest_mappings(env_list, app_name)

    result = {
        "app_name": app_name,
        "suggestions": [
            {
                "env_var": s.env_var,
                "suggested_path": s.suggested_path,
                "key": s.key,
                "confidence": s.confidence,
                "vault_reference": f"{s.suggested_path}#{s.key}",
            }
            for s in suggestions
        ],
    }
    return json.dumps(result, indent=2)


@tool
def validate_vault_paths(paths_json: str) -> str:
    """Validate that Vault secret paths exist and are accessible.

    Use this to verify user-provided Vault paths before generating
    the Nomad job specification.

    Args:
        paths_json: JSON object mapping env_var names to vault paths.
                   Example: {"DB_PASSWORD": "secret/data/myapp/db"}

    Returns:
        JSON string with validation results for each path.
    """
    client = get_vault_client()

    try:
        paths = json.loads(paths_json)
    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid JSON input"})

    results: dict[str, Any] = {"valid": [], "invalid": []}

    for env_var, path in paths.items():
        is_valid, error = client.validate_path(path)
        if is_valid:
            metadata = client.read_metadata(path)
            results["valid"].append(
                {
                    "env_var": env_var,
                    "path": path,
                    "available_keys": metadata.keys if metadata else [],
                }
            )
        else:
            results["invalid"].append(
                {
                    "env_var": env_var,
                    "path": path,
                    "error": error,
                }
            )

    results["all_valid"] = len(results["invalid"]) == 0
    return json.dumps(results, indent=2)


@tool
def get_vault_secret_keys(path: str) -> str:
    """Get the available keys in a Vault secret.

    Use this to discover what keys are available at a specific
    Vault path, so you can suggest the correct key for an env var.

    Args:
        path: Full Vault path (e.g., secret/data/myapp/db).

    Returns:
        JSON string with available keys at the path.
    """
    client = get_vault_client()
    metadata = client.read_metadata(path)

    if metadata:
        return json.dumps(
            {
                "path": path,
                "keys": metadata.keys,
                "version": metadata.version,
            },
            indent=2,
        )
    else:
        return json.dumps(
            {
                "path": path,
                "error": "Path not found or access denied",
                "keys": [],
            },
            indent=2,
        )
