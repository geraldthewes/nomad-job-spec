"""Parser for deploy/.env.deploy files.

This module provides parsing and validation for the .env.deploy file format
which explicitly declares environment variable sources:

    env:LOG_LEVEL=info           # Fixed environment value
    vault:DB_PASS=secret/path:key  # Vault secret reference
    nomad:APP_PORT=assigned       # Nomad-assigned dynamic port
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)


@dataclass
class EnvDeployEntry:
    """Single entry from .env.deploy file."""

    name: str  # Environment variable name
    source: Literal["env", "vault", "nomad"]  # Source type
    value: str  # Fixed value, vault_path:field, or "assigned"

    # Vault-specific parsed fields (populated for vault source)
    vault_path: str | None = None
    vault_field: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "name": self.name,
            "source": self.source,
            "value": self.value,
        }
        if self.source == "vault":
            result["vault_path"] = self.vault_path
            result["vault_field"] = self.vault_field
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EnvDeployEntry":
        """Create from dictionary representation."""
        return cls(
            name=data["name"],
            source=data["source"],
            value=data["value"],
            vault_path=data.get("vault_path"),
            vault_field=data.get("vault_field"),
        )


@dataclass
class EnvDeployConfig:
    """Parsed .env.deploy file."""

    entries: dict[str, EnvDeployEntry] = field(default_factory=dict)  # var_name -> entry
    file_path: Path | None = None

    def get_env_entries(self) -> list[EnvDeployEntry]:
        """Get all entries with source='env'."""
        return [e for e in self.entries.values() if e.source == "env"]

    def get_vault_entries(self) -> list[EnvDeployEntry]:
        """Get all entries with source='vault'."""
        return [e for e in self.entries.values() if e.source == "vault"]

    def get_nomad_entries(self) -> list[EnvDeployEntry]:
        """Get all entries with source='nomad'."""
        return [e for e in self.entries.values() if e.source == "nomad"]

    def get_vault_paths_grouped(self) -> dict[str, list[EnvDeployEntry]]:
        """Group vault entries by path for efficient HCL generation."""
        grouped: dict[str, list[EnvDeployEntry]] = {}
        for entry in self.get_vault_entries():
            if entry.vault_path:
                if entry.vault_path not in grouped:
                    grouped[entry.vault_path] = []
                grouped[entry.vault_path].append(entry)
        return grouped

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "file_path": str(self.file_path) if self.file_path else None,
            "entries": {name: entry.to_dict() for name, entry in self.entries.items()},
        }


class EnvDeployParseError(Exception):
    """Error parsing .env.deploy file."""

    def __init__(self, message: str, line_number: int | None = None, line: str | None = None):
        self.line_number = line_number
        self.line = line
        if line_number and line:
            super().__init__(f"Line {line_number}: {message}\n  -> {line}")
        elif line_number:
            super().__init__(f"Line {line_number}: {message}")
        else:
            super().__init__(message)


def parse_env_deploy(path: Path) -> EnvDeployConfig:
    """Parse a .env.deploy file.

    Args:
        path: Path to the .env.deploy file.

    Returns:
        EnvDeployConfig with parsed entries.

    Raises:
        FileNotFoundError: If file doesn't exist.
        EnvDeployParseError: If file format is invalid.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Environment deploy file not found: {path}\n\n"
            f"Expected format (deploy/.env.deploy):\n"
            f"  env:LOG_LEVEL=info\n"
            f"  vault:DB_PASSWORD=secret/data/myapp/db:password\n"
            f"  nomad:APP_PORT=assigned"
        )

    config = EnvDeployConfig(file_path=path)
    valid_sources = {"env", "vault", "nomad"}

    with open(path) as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            # Parse line format: type:VAR=value
            match = re.match(r"^(\w+):(\w+)=(.*)$", line)
            if not match:
                raise EnvDeployParseError(
                    "Invalid format. Expected: type:VAR_NAME=value",
                    line_number=line_number,
                    line=line,
                )

            source, var_name, value = match.groups()
            source = source.lower()

            # Validate source type
            if source not in valid_sources:
                raise EnvDeployParseError(
                    f"Unknown type '{source}'. Valid types: {', '.join(sorted(valid_sources))}",
                    line_number=line_number,
                    line=line,
                )

            # Create entry based on source type
            entry = _create_entry(source, var_name, value, line_number, line)
            config.entries[var_name] = entry

    return config


def _create_entry(
    source: str,
    var_name: str,
    value: str,
    line_number: int,
    line: str,
) -> EnvDeployEntry:
    """Create an EnvDeployEntry based on source type."""
    if source == "env":
        return EnvDeployEntry(
            name=var_name,
            source="env",
            value=value,
        )

    elif source == "vault":
        # Parse vault format: path:field
        if ":" not in value:
            raise EnvDeployParseError(
                "Vault entry must have format: vault:VAR=path:field",
                line_number=line_number,
                line=line,
            )
        # Split on last colon to handle paths like secret/data/app:field
        last_colon = value.rfind(":")
        vault_path = value[:last_colon]
        vault_field = value[last_colon + 1:]

        if not vault_path or not vault_field:
            raise EnvDeployParseError(
                "Vault entry must have both path and field: vault:VAR=path:field",
                line_number=line_number,
                line=line,
            )

        return EnvDeployEntry(
            name=var_name,
            source="vault",
            value=value,
            vault_path=vault_path,
            vault_field=vault_field,
        )

    elif source == "nomad":
        # Validate nomad format
        if value.lower() != "assigned":
            raise EnvDeployParseError(
                "Nomad entry must have value 'assigned': nomad:VAR=assigned",
                line_number=line_number,
                line=line,
            )
        return EnvDeployEntry(
            name=var_name,
            source="nomad",
            value="assigned",
        )

    # Should never reach here due to earlier validation
    raise EnvDeployParseError(f"Unknown source type: {source}", line_number=line_number, line=line)


@dataclass
class ValidationResult:
    """Result of validating env var coverage."""

    is_valid: bool
    missing_vars: list[str]  # In Dockerfile but not in .env.deploy
    extra_vars: list[str]  # In .env.deploy but not in Dockerfile
    errors: list[str]  # Other validation errors

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "is_valid": self.is_valid,
            "missing_vars": self.missing_vars,
            "extra_vars": self.extra_vars,
            "errors": self.errors,
        }


def convert_to_job_config_fields(
    env_var_configs: list[dict[str, Any]],
    port_label: str = "http",
) -> dict[str, Any]:
    """Convert env_var_configs to JobConfig-compatible fields.

    This function takes the env_var_configs from the analysis subgraph
    and converts them to the format expected by JobConfig:
    - env_vars: dict of fixed env vars
    - vault_secrets: dict of vault env vars (for VaultConfig)
    - nomad_port_vars: dict of nomad port env vars

    Args:
        env_var_configs: List of env var config dicts from analysis.
        port_label: Default port label for nomad port vars (default: "http").

    Returns:
        Dict with keys: env_vars, vault_secrets, nomad_port_vars
    """
    env_vars: dict[str, str] = {}
    vault_secrets: dict[str, str] = {}
    nomad_port_vars: dict[str, str] = {}

    for config in env_var_configs:
        name = config.get("name", "")
        source = config.get("source", "env")
        value = config.get("value", "")

        if source == "env":
            env_vars[name] = value
        elif source == "vault":
            # For vault, value can be either:
            # - "path:field" format from .env.deploy
            # - "path#field" format (Nomad native)
            vault_path = config.get("vault_path")
            vault_field = config.get("vault_field")
            if vault_path and vault_field:
                # Use # separator for Nomad native format
                vault_secrets[name] = f"{vault_path}#{vault_field}"
            elif ":" in value:
                # Convert path:field to path#field
                vault_secrets[name] = value.replace(":", "#", 1)
            else:
                vault_secrets[name] = value
        elif source == "nomad":
            # For nomad, the value should be "assigned" but we use port_label
            nomad_port_vars[name] = port_label

    return {
        "env_vars": env_vars,
        "vault_secrets": vault_secrets,
        "nomad_port_vars": nomad_port_vars,
    }


def validate_env_coverage(
    dockerfile_vars: list[str],
    env_deploy: EnvDeployConfig,
) -> ValidationResult:
    """Validate that .env.deploy covers all Dockerfile ENV vars.

    Args:
        dockerfile_vars: List of env var names from Dockerfile.
        env_deploy: Parsed .env.deploy configuration.

    Returns:
        ValidationResult with missing and extra vars.
    """
    dockerfile_set = set(dockerfile_vars)
    deploy_set = set(env_deploy.entries.keys())

    missing_vars = sorted(dockerfile_set - deploy_set)
    extra_vars = sorted(deploy_set - dockerfile_set)

    errors = []
    if missing_vars:
        errors.append(
            f"Missing .env.deploy entries for Dockerfile ENV vars: {', '.join(missing_vars)}"
        )

    is_valid = len(missing_vars) == 0

    if extra_vars:
        logger.warning(
            f"Extra entries in .env.deploy not in Dockerfile: {', '.join(extra_vars)}"
        )

    return ValidationResult(
        is_valid=is_valid,
        missing_vars=missing_vars,
        extra_vars=extra_vars,
        errors=errors,
    )
