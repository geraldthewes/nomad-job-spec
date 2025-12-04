"""Environment variable extraction node for the analysis subgraph.

This node extracts environment variables from Dockerfile ENV declarations
and validates them against the deploy/.env.deploy configuration file.
"""

import logging
from pathlib import Path
from typing import Any

from src.observability import get_observability
from src.tools.codebase import parse_dockerfile
from src.tools.env_deploy import (
    EnvDeployConfig,
    EnvDeployParseError,
    parse_env_deploy,
    validate_env_coverage,
)
from src.tools.vault import VaultClient, get_vault_client

logger = logging.getLogger(__name__)


def create_extract_env_vars_node():
    """Create the extract_env_vars node function.

    This node:
    1. Parses Dockerfile for ENV declarations (authoritative source of required vars)
    2. Parses deploy/.env.deploy for typed values (env:, vault:, nomad:)
    3. Validates coverage (all Dockerfile vars must be in .env.deploy)
    4. Validates Vault paths if available

    Returns:
        Node function for use in LangGraph.
    """

    def extract_env_vars_node(state: dict[str, Any]) -> dict[str, Any]:
        """Extract environment variables from Dockerfile and .env.deploy.

        Args:
            state: AnalysisState from the subgraph.

        Returns:
            Dictionary with env_deploy_config and validation results.
        """
        obs = get_observability()
        codebase_path = Path(state.get("codebase_path", ""))
        selected_dockerfile = state.get("selected_dockerfile")

        # Output fields
        env_deploy_config: dict[str, Any] = {}
        env_var_validation: dict[str, Any] = {
            "is_valid": False,
            "dockerfile_vars": [],
            "missing_vars": [],
            "extra_vars": [],
            "vault_validation": {},
            "errors": [],
        }

        # 1. Parse Dockerfile for ENV declarations
        dockerfile_vars: list[str] = []
        with obs.span("parse_dockerfile_env", input={"dockerfile": selected_dockerfile}) as span:
            if selected_dockerfile:
                dockerfile_path = codebase_path / selected_dockerfile
            else:
                dockerfile_path = codebase_path / "Dockerfile"

            if not dockerfile_path.exists():
                error_msg = f"Dockerfile not found: {dockerfile_path}"
                logger.error(error_msg)
                env_var_validation["errors"].append(error_msg)
                span.end(level="ERROR", status_message=error_msg)
                return {
                    "env_deploy_config": env_deploy_config,
                    "env_var_validation": env_var_validation,
                }

            try:
                dockerfile_info = parse_dockerfile(str(dockerfile_path))
                dockerfile_vars = dockerfile_info.env_var_names
                env_var_validation["dockerfile_vars"] = dockerfile_vars
                logger.info(f"Found {len(dockerfile_vars)} ENV declarations in Dockerfile")
                span.end(output={"env_vars": dockerfile_vars, "count": len(dockerfile_vars)})
            except Exception as e:
                error_msg = f"Failed to parse Dockerfile: {e}"
                logger.error(error_msg)
                env_var_validation["errors"].append(error_msg)
                span.end(level="ERROR", status_message=error_msg)
                return {
                    "env_deploy_config": env_deploy_config,
                    "env_var_validation": env_var_validation,
                }

        # 2. Parse deploy/.env.deploy (optional - for backward compatibility)
        env_deploy_path = codebase_path / "deploy" / ".env.deploy"
        config = None
        with obs.span("parse_env_deploy", input={"path": str(env_deploy_path)}) as span:
            try:
                config = parse_env_deploy(env_deploy_path)
                env_deploy_config = config.to_dict()
                logger.info(f"Parsed {len(config.entries)} entries from .env.deploy")
                span.end(output={
                    "entries": len(config.entries),
                    "env_count": len(config.get_env_entries()),
                    "vault_count": len(config.get_vault_entries()),
                    "nomad_count": len(config.get_nomad_entries()),
                })
            except FileNotFoundError:
                # .env.deploy is optional for backward compatibility
                # The enrich node will fall back to inference
                logger.warning(
                    f"No .env.deploy found at {env_deploy_path}. "
                    f"Using inference-based env var configuration. "
                    f"Consider creating deploy/.env.deploy for explicit configuration."
                )
                span.end(level="WARNING", status_message="File not found - using inference fallback")
                # Return early with empty config - enrich node will use inference
                return {
                    "env_deploy_config": {},
                    "env_var_validation": {
                        "is_valid": True,  # Not a failure, just using fallback
                        "dockerfile_vars": dockerfile_vars,
                        "missing_vars": [],
                        "extra_vars": [],
                        "vault_validation": {},
                        "errors": [],
                        "using_fallback": True,
                    },
                }
            except EnvDeployParseError as e:
                error_msg = f"Failed to parse .env.deploy: {e}"
                logger.error(error_msg)
                env_var_validation["errors"].append(error_msg)
                span.end(level="ERROR", status_message=str(e))
                return {
                    "env_deploy_config": env_deploy_config,
                    "env_var_validation": env_var_validation,
                }

        # 3. Validate coverage (all Dockerfile vars in .env.deploy)
        with obs.span("validate_env_coverage") as span:
            validation_result = validate_env_coverage(dockerfile_vars, config)
            env_var_validation["is_valid"] = validation_result.is_valid
            env_var_validation["missing_vars"] = validation_result.missing_vars
            env_var_validation["extra_vars"] = validation_result.extra_vars
            if validation_result.errors:
                env_var_validation["errors"].extend(validation_result.errors)

            if validation_result.is_valid:
                logger.info("Environment variable coverage validated successfully")
                span.end(output={"is_valid": True})
            else:
                logger.warning(f"Missing env vars: {validation_result.missing_vars}")
                span.end(
                    level="WARNING",
                    output={
                        "is_valid": False,
                        "missing": validation_result.missing_vars,
                    },
                )

        # 4. Validate Vault paths (if Vault available)
        vault_entries = config.get_vault_entries()
        if vault_entries:
            with obs.span("validate_vault_paths", input={"count": len(vault_entries)}) as span:
                vault_validation: dict[str, Any] = {"valid": [], "invalid": [], "skipped": False}
                try:
                    vault = get_vault_client()
                    if not vault.is_authenticated:
                        logger.warning("Vault client not authenticated, skipping path validation")
                        vault_validation["skipped"] = True
                        span.end(output={"skipped": True, "reason": "not_authenticated"})
                    else:
                        for entry in vault_entries:
                            if entry.vault_path:
                                is_valid, error = vault.validate_path(entry.vault_path)
                                if is_valid:
                                    # Check if field exists
                                    metadata = vault.read_metadata(entry.vault_path)
                                    if metadata and entry.vault_field:
                                        if entry.vault_field in metadata.keys:
                                            vault_validation["valid"].append({
                                                "env_var": entry.name,
                                                "path": entry.vault_path,
                                                "field": entry.vault_field,
                                            })
                                        else:
                                            vault_validation["invalid"].append({
                                                "env_var": entry.name,
                                                "path": entry.vault_path,
                                                "field": entry.vault_field,
                                                "error": f"Field '{entry.vault_field}' not found. Available: {metadata.keys}",
                                            })
                                    else:
                                        vault_validation["valid"].append({
                                            "env_var": entry.name,
                                            "path": entry.vault_path,
                                            "field": entry.vault_field,
                                        })
                                else:
                                    vault_validation["invalid"].append({
                                        "env_var": entry.name,
                                        "path": entry.vault_path,
                                        "field": entry.vault_field,
                                        "error": error,
                                    })

                        logger.info(
                            f"Vault path validation: {len(vault_validation['valid'])} valid, "
                            f"{len(vault_validation['invalid'])} invalid"
                        )
                        span.end(output={
                            "valid_count": len(vault_validation["valid"]),
                            "invalid_count": len(vault_validation["invalid"]),
                        })

                        # Add invalid paths to errors
                        for invalid in vault_validation["invalid"]:
                            env_var_validation["errors"].append(
                                f"Vault validation failed for {invalid['env_var']}: {invalid['error']}"
                            )

                except Exception as e:
                    logger.warning(f"Vault path validation failed: {e}")
                    vault_validation["skipped"] = True
                    vault_validation["error"] = str(e)
                    span.end(level="WARNING", status_message=str(e))

                env_var_validation["vault_validation"] = vault_validation

        return {
            "env_deploy_config": env_deploy_config,
            "env_var_validation": env_var_validation,
        }

    return extract_env_vars_node
