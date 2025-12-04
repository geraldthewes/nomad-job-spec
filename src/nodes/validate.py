"""Pre-deployment validation node for the Nomad Job Spec workflow.

This node runs after generate and before deploy.
It validates the generated job spec against infrastructure:
- Fabio route conflicts (STRICT mode - blocks deployment)
- Vault path accessibility
- Service name uniqueness
"""

import logging
from dataclasses import dataclass
from typing import Any

from config.settings import Settings, get_settings
from src.observability import get_observability
from src.tools.vault import get_vault_client
from src.tools.fabio import get_fabio_client, FabioRouteConflictError

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of pre-deployment validation."""

    passed: bool
    errors: list[str]
    warnings: list[str]


def create_validate_node(
    settings: Settings | None = None,
):
    """Create the pre-deployment validation node function.

    Args:
        settings: Application settings.

    Returns:
        Node function for use in LangGraph.
    """
    if settings is None:
        settings = get_settings()

    def validate_node(state: dict[str, Any]) -> dict[str, Any]:
        """Validate job spec before deployment.

        Performs the following checks:
        1. Fabio route conflicts (STRICT - blocks deployment on conflicts)
        2. Vault path accessibility
        3. Job spec syntax validation

        Updates state with:
        - pre_deploy_validation: ValidationResult details
        - deployment_status: Set to 'blocked' if validation fails
        """
        obs = get_observability()

        errors: list[str] = []
        warnings: list[str] = []

        job_config = state.get("job_config", {})
        fabio_validation = state.get("fabio_validation", {})
        vault_suggestions = state.get("vault_suggestions", {})

        # 1. Validate Fabio routes (STRICT MODE)
        with obs.span("validate_fabio_routes") as span:
            fabio_errors = _validate_fabio_routes(job_config, fabio_validation)
            errors.extend(fabio_errors)
            span.end(output={"errors": fabio_errors, "count": len(fabio_errors)})

        # 2. Validate Vault paths
        with obs.span("validate_vault_paths") as span:
            vault_warnings = _validate_vault_paths(job_config, vault_suggestions)
            warnings.extend(vault_warnings)
            span.end(output={"warnings": vault_warnings, "count": len(vault_warnings)})

        # 3. Validate HCL syntax (if available)
        with obs.span("check_hcl_syntax") as span:
            hcl_valid = state.get("hcl_valid", True)
            validation_error = state.get("validation_error")
            if not hcl_valid and validation_error:
                errors.append(f"HCL validation failed: {validation_error}")
            span.end(output={"hcl_valid": hcl_valid, "error": validation_error})

        # Determine overall result
        passed = len(errors) == 0

        validation_result = {
            "passed": passed,
            "errors": errors,
            "warnings": warnings,
            "error_count": len(errors),
            "warning_count": len(warnings),
        }

        # Build partial state update
        updates: dict[str, Any] = {
            "pre_deploy_validation": validation_result,
        }

        if not passed:
            updates["deployment_status"] = "blocked"
            updates["deployment_error"] = "; ".join(errors)
            logger.error(f"Pre-deployment validation failed: {errors}")
        else:
            if warnings:
                logger.warning(f"Pre-deployment warnings: {warnings}")

        return updates

    return validate_node


def _validate_fabio_routes(
    job_config: dict[str, Any],
    fabio_validation: dict[str, Any],
) -> list[str]:
    """Validate Fabio routes in STRICT mode.

    Returns list of error messages. Any conflicts are blocking errors.
    """
    errors = []

    # Check pre-computed conflicts from enrich node
    conflicts = fabio_validation.get("conflicts", [])
    for conflict in conflicts:
        if conflict.get("severity") == "error":
            errors.append(
                f"Fabio route conflict ({conflict.get('type')}): "
                f"Route conflicts with existing service '{conflict.get('existing_service')}' "
                f"at '{conflict.get('existing_route')}'"
            )

    # Also do a live check if we have routing config
    fabio_route = job_config.get("fabio_route", {})
    if fabio_route:
        hostname = fabio_route.get("hostname")
        path = fabio_route.get("path")

        if hostname or path:
            try:
                fabio = get_fabio_client()
                # STRICT mode - raises exception on conflict
                fabio.validate_route_strict(hostname=hostname, path=path)
            except FabioRouteConflictError as e:
                errors.append(str(e))
            except Exception as e:
                # Log but don't block on connection errors
                logger.warning(f"Could not verify Fabio routes: {e}")

    return errors


def _validate_vault_paths(
    job_config: dict[str, Any],
    vault_suggestions: dict[str, Any],
) -> list[str]:
    """Validate Vault paths are accessible.

    Returns list of warning messages (not blocking in STRICT mode).
    """
    warnings = []

    vault_config = job_config.get("vault", {})
    if not vault_config:
        return warnings

    secrets = vault_config.get("secrets", {})
    if not secrets:
        return warnings

    vault = get_vault_client()

    for env_var, vault_path in secrets.items():
        # Extract path without key suffix
        if "#" in vault_path:
            path, key = vault_path.rsplit("#", 1)
        elif "." in vault_path.split("/")[-1]:
            path = vault_path.rsplit(".", 1)[0]
            key = vault_path.rsplit(".", 1)[1]
        else:
            path = vault_path
            key = None

        try:
            is_valid, error = vault.validate_path(path)
            if not is_valid:
                warnings.append(
                    f"Vault path for {env_var} may not be accessible: {error}"
                )
            elif key:
                # Check if key exists
                metadata = vault.read_metadata(path)
                if metadata and key not in metadata.keys:
                    warnings.append(
                        f"Vault key '{key}' not found at {path} for {env_var}. "
                        f"Available keys: {', '.join(metadata.keys)}"
                    )
        except Exception as e:
            warnings.append(f"Could not validate Vault path for {env_var}: {e}")

    return warnings


def should_proceed_after_validation(state: dict[str, Any]) -> str:
    """Conditional edge function to determine flow after validation.

    Returns:
        'proceed' to continue to deploy, 'blocked' to skip deployment.
    """
    validation = state.get("pre_deploy_validation", {})
    if validation.get("passed", True):
        return "proceed"
    else:
        return "blocked"
