"""Base classes for the extractor registry pattern.

Extractors discover and extract deployment configuration from various
source files (build.yaml, Makefile, Dockerfile, etc.).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PortConfig:
    """Port configuration extracted from source."""

    name: str
    container_port: int
    static: bool = False
    host_port: int | None = None
    # Port configurability fields (from analyze_ports node)
    is_configurable: bool | None = None  # None = unknown
    env_var: str | None = None  # e.g., "APP_PORT"
    default_value: int | None = None


@dataclass
class ResourceConfig:
    """Resource configuration extracted from source."""

    cpu: int | None = None  # MHz
    memory: int | None = None  # MB
    disk: int | None = None  # MB


@dataclass
class VaultSecret:
    """Vault secret configuration extracted from source."""

    path: str  # e.g., "secret/data/aws/transcription"
    fields: dict[str, str]  # vault_field -> env_var mapping


@dataclass
class HealthCheckConfig:
    """Health check configuration extracted from source."""

    type: str  # "http", "tcp", "script"
    path: str | None = None  # For HTTP checks
    port: str | None = None  # Port name
    interval: str = "10s"
    timeout: str = "2s"


@dataclass
class ExtractionResult:
    """Standardized result from any extractor.

    All fields except source_type and source_file are optional.
    The merge node combines results from multiple extractors based on priority.
    """

    # Required metadata
    source_type: str  # "jobforge", "makefile_docker", "dockerfile", etc.
    source_file: str  # Path to the file extracted from
    confidence: float = 0.5  # 0.0-1.0, how reliable this extraction is

    # Extracted fields (all optional)
    job_name: str | None = None
    docker_image: str | None = None  # Full image with registry/tag
    registry_url: str | None = None
    image_name: str | None = None
    image_tag: str | None = None

    # Port configuration
    ports: list[PortConfig] | None = None

    # Environment configuration
    env_vars: dict[str, str] | None = None  # Fixed env vars (key -> value)
    vault_secrets: list[VaultSecret] | None = None  # Vault paths + mappings
    vault_policies: list[str] | None = None

    # Resource configuration
    resources: ResourceConfig | None = None

    # Health check
    health_check: HealthCheckConfig | None = None

    # Constraints and requirements
    requires_gpu: bool | None = None
    requires_amd64: bool | None = None
    constraints: list[str] | None = None

    # Storage
    requires_storage: bool | None = None
    storage_path: str | None = None

    # Raw data for debugging
    raw_data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        result = {
            "source_type": self.source_type,
            "source_file": self.source_file,
            "confidence": self.confidence,
        }

        # Add non-None fields
        if self.job_name is not None:
            result["job_name"] = self.job_name
        if self.docker_image is not None:
            result["docker_image"] = self.docker_image
        if self.registry_url is not None:
            result["registry_url"] = self.registry_url
        if self.image_name is not None:
            result["image_name"] = self.image_name
        if self.image_tag is not None:
            result["image_tag"] = self.image_tag
        if self.ports is not None:
            result["ports"] = [
                {
                    "name": p.name,
                    "container_port": p.container_port,
                    "static": p.static,
                    "host_port": p.host_port,
                    "is_configurable": p.is_configurable,
                    "env_var": p.env_var,
                    "default_value": p.default_value,
                }
                for p in self.ports
            ]
        if self.env_vars is not None:
            result["env_vars"] = self.env_vars
        if self.vault_secrets is not None:
            result["vault_secrets"] = [
                {"path": s.path, "fields": s.fields} for s in self.vault_secrets
            ]
        if self.vault_policies is not None:
            result["vault_policies"] = self.vault_policies
        if self.resources is not None:
            result["resources"] = {
                "cpu": self.resources.cpu,
                "memory": self.resources.memory,
                "disk": self.resources.disk,
            }
        if self.health_check is not None:
            result["health_check"] = {
                "type": self.health_check.type,
                "path": self.health_check.path,
                "port": self.health_check.port,
                "interval": self.health_check.interval,
                "timeout": self.health_check.timeout,
            }
        if self.requires_gpu is not None:
            result["requires_gpu"] = self.requires_gpu
        if self.requires_amd64 is not None:
            result["requires_amd64"] = self.requires_amd64
        if self.constraints is not None:
            result["constraints"] = self.constraints
        if self.requires_storage is not None:
            result["requires_storage"] = self.requires_storage
        if self.storage_path is not None:
            result["storage_path"] = self.storage_path

        return result


class BaseExtractor(ABC):
    """Abstract base class for all extractors.

    Each extractor is responsible for:
    1. Detecting if it can handle a given file
    2. Extracting configuration from that file
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this extractor (e.g., 'jobforge', 'makefile_docker')."""
        ...

    @property
    @abstractmethod
    def file_patterns(self) -> list[str]:
        """Glob patterns for files this extractor can handle.

        Examples: ["build.yaml"], ["Makefile"], ["docker-compose*.yml"]
        """
        ...

    @property
    def priority(self) -> int:
        """Priority for merging (higher = preferred). Default is 50.

        Suggested values:
        - 100: Explicit config (jobforge)
        - 70: Build tools (makefile)
        - 50: Standard files (Dockerfile)
        - 30: Inference (app code)
        """
        return 50

    @abstractmethod
    def can_extract(self, file_path: str, content: str | None = None) -> bool:
        """Check if this extractor can handle the given file.

        Args:
            file_path: Path to the file.
            content: Optional file content (for deeper inspection).

        Returns:
            True if this extractor can extract from this file.
        """
        ...

    @abstractmethod
    def extract(self, file_path: str, codebase_path: str) -> ExtractionResult:
        """Extract configuration from the file.

        Args:
            file_path: Path to the specific file to extract from.
            codebase_path: Root path of the codebase (for context).

        Returns:
            ExtractionResult with extracted configuration.
        """
        ...
