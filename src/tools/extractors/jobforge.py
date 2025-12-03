"""JobForge extractor for build.yaml files.

Parses build.yaml files following the JobForge spec:
https://github.com/geraldthewes/jobforge/blob/main/docs/JobSpec.md
"""

import logging
import os
from pathlib import Path

import yaml

from src.tools.extractors.base import (
    BaseExtractor,
    ExtractionResult,
    ResourceConfig,
    VaultSecret,
)
from src.tools.extractors import register_extractor

logger = logging.getLogger(__name__)


@register_extractor
class JobforgeExtractor(BaseExtractor):
    """Extractor for JobForge build.yaml files.

    JobForge is a CI/CD tool for building and testing Docker images.
    The build.yaml file contains comprehensive deployment configuration.
    """

    @property
    def name(self) -> str:
        return "jobforge"

    @property
    def file_patterns(self) -> list[str]:
        return ["build.yaml", "build.yml"]

    @property
    def priority(self) -> int:
        # Highest priority - explicit configuration
        return 100

    def can_extract(self, file_path: str, content: str | None = None) -> bool:
        """Check if this is a valid JobForge build.yaml file."""
        file_name = os.path.basename(file_path)
        if file_name not in self.file_patterns:
            return False

        # Optionally validate content structure
        if content:
            try:
                data = yaml.safe_load(content)
                # Check for required JobForge fields
                return isinstance(data, dict) and "image_name" in data
            except yaml.YAMLError:
                return False

        return True

    def extract(self, file_path: str, codebase_path: str) -> ExtractionResult:
        """Extract configuration from a JobForge build.yaml file.

        Args:
            file_path: Path to the build.yaml file.
            codebase_path: Root path of the codebase.

        Returns:
            ExtractionResult with extracted configuration.
        """
        with open(file_path) as f:
            data = yaml.safe_load(f)

        # Extract root-level fields
        image_name = data.get("image_name")
        registry_url = data.get("registry_url")
        image_tags = data.get("image_tags", [])
        image_tag = image_tags[0] if image_tags else "latest"

        # Build full docker image reference
        docker_image = None
        if image_name:
            if registry_url:
                docker_image = f"{registry_url}/{image_name}:{image_tag}"
            else:
                docker_image = f"{image_name}:{image_tag}"

        # Extract resource_limits from root level (applies to all phases)
        resources = None
        resource_limits = data.get("resource_limits", {})
        # Check both root level and build-specific limits
        build_limits = resource_limits.get("build", resource_limits)
        if build_limits:
            resources = ResourceConfig(
                cpu=_parse_int(build_limits.get("cpu")),
                memory=_parse_int(build_limits.get("memory")),
                disk=_parse_int(build_limits.get("disk")),
            )

        # Extract from test section (per JobForge spec)
        test_section = data.get("test", {})

        # Environment variables
        env_vars = test_section.get("env", {})

        # Vault configuration
        vault_policies = test_section.get("vault_policies", [])
        vault_secrets_raw = test_section.get("vault_secrets", [])
        vault_secrets = []
        for secret in vault_secrets_raw:
            if isinstance(secret, dict) and "path" in secret:
                fields = secret.get("fields", {})
                vault_secrets.append(VaultSecret(path=secret["path"], fields=fields))

        # GPU requirement
        requires_gpu = test_section.get("gpu_required", False)

        # Constraints
        constraints = test_section.get("constraints", [])

        # Derive job name from image_name if not explicitly set
        job_name = image_name

        return ExtractionResult(
            source_type=self.name,
            source_file=file_path,
            confidence=0.95,  # High confidence - explicit config
            job_name=job_name,
            docker_image=docker_image,
            registry_url=registry_url,
            image_name=image_name,
            image_tag=image_tag,
            env_vars=env_vars if env_vars else None,
            vault_secrets=vault_secrets if vault_secrets else None,
            vault_policies=vault_policies if vault_policies else None,
            resources=resources,
            requires_gpu=requires_gpu if requires_gpu else None,
            constraints=constraints if constraints else None,
            raw_data=data,
        )


def _parse_int(value: str | int | None) -> int | None:
    """Parse a value that may be string or int to int."""
    if value is None:
        return None
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (ValueError, TypeError):
        return None
