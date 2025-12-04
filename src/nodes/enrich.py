"""Infrastructure enrichment node for the Nomad Job Spec workflow.

This node runs after analyze and before question generation.
It queries Vault, Consul, and Fabio to enrich the analysis with:
- Vault secret path suggestions for detected environment variables
- Consul conventions and service discovery information
- Fabio route availability checks
"""

import logging
from typing import Any

from config.settings import Settings, get_settings
from src.observability import get_observability
from src.tools.vault import (
    VaultClient,
    VaultConventions,
    get_vault_client,
    set_vault_client,
    suggest_env_configs,
)
from src.tools.consul import ConsulClient, get_consul_client, set_consul_client
from src.tools.fabio import FabioClient, get_fabio_client, set_fabio_client
from src.tools.nomad_version import get_cached_nomad_version
from src.tools.infra_status import InfraStatus

logger = logging.getLogger(__name__)


def _default_conventions() -> dict[str, Any]:
    """Return default conventions structure when Consul is unavailable."""
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


def create_enrich_node(
    settings: Settings | None = None,
    vault_client: VaultClient | None = None,
    consul_client: ConsulClient | None = None,
    fabio_client: FabioClient | None = None,
):
    """Create the enrichment node function.

    Args:
        settings: Application settings.
        vault_client: Optional pre-configured Vault client.
        consul_client: Optional pre-configured Consul client.
        fabio_client: Optional pre-configured Fabio client.

    Returns:
        Node function for use in LangGraph.
    """
    if settings is None:
        settings = get_settings()

    # Initialize clients with settings if not provided
    # This ensures they use .env values via pydantic-settings
    if vault_client:
        set_vault_client(vault_client)
    else:
        set_vault_client(VaultClient(
            addr=settings.vault_addr,
            token=settings.vault_token,
            namespace=settings.vault_namespace,
        ))

    if consul_client:
        set_consul_client(consul_client)
    else:
        set_consul_client(ConsulClient(
            addr=settings.consul_http_addr,
            token=settings.consul_http_token,
        ))

    if fabio_client:
        set_fabio_client(fabio_client)
    else:
        set_fabio_client(FabioClient(
            addr=settings.fabio_admin_addr,
        ))

    def enrich_node(state: dict[str, Any]) -> dict[str, Any]:
        """Enrich codebase analysis with infrastructure context.

        Queries Vault, Consul, and Fabio to add:
        - vault_suggestions: Mapped env vars to Vault paths
        - consul_conventions: Naming conventions from Consul KV
        - consul_services: Available service dependencies
        - fabio_validation: Route availability check
        - nomad_version: Detected Nomad version info

        Note: This node is designed to be resilient - if infrastructure
        services are unavailable, it will continue with empty/default values.
        """
        obs = get_observability()

        analysis = state.get("codebase_analysis", {})
        app_name = _extract_app_name(state)

        # Try to get clients - may fail if services unavailable
        # We track initialization status to provide better feedback
        vault = None
        consul = None
        fabio = None
        infra_issues: list[dict[str, str]] = []

        with obs.span("init_vault_client") as span:
            try:
                vault = get_vault_client()
                span.end(output={"status": "connected"})
            except Exception as e:
                error_str = str(e)
                if "connection refused" in error_str.lower():
                    msg = "Vault connection refused - check if Vault is running"
                elif "timeout" in error_str.lower():
                    msg = "Vault connection timeout - check network"
                else:
                    msg = f"Vault initialization failed: {error_str[:100]}"
                logger.warning(msg)
                infra_issues.append({"service": "Vault", "error": msg})
                span.end(level="WARNING", status_message=msg)

        with obs.span("init_consul_client") as span:
            try:
                consul = get_consul_client()
                span.end(output={"status": "connected"})
            except Exception as e:
                error_str = str(e)
                if "connection refused" in error_str.lower():
                    msg = "Consul connection refused - check if Consul is running"
                elif "invalid" in error_str.lower():
                    msg = f"Consul address configuration issue: {error_str[:100]}"
                else:
                    msg = f"Consul initialization failed: {error_str[:100]}"
                logger.warning(msg)
                infra_issues.append({"service": "Consul", "error": msg})
                span.end(level="WARNING", status_message=msg)

        with obs.span("init_fabio_client") as span:
            try:
                fabio = get_fabio_client()
                span.end(output={"status": "connected"})
            except Exception as e:
                error_str = str(e)
                if "connection refused" in error_str.lower():
                    msg = "Fabio connection refused - check if Fabio is running"
                else:
                    msg = f"Fabio initialization failed: {error_str[:100]}"
                logger.warning(msg)
                infra_issues.append({"service": "Fabio", "error": msg})
                span.end(level="WARNING", status_message=msg)

        # Initialize enrichment data
        vault_suggestions: dict[str, Any] = {"suggestions": [], "error": None}
        consul_conventions: dict[str, Any] = {}
        consul_services: dict[str, Any] = {"available": [], "dependencies": []}
        fabio_validation: dict[str, Any] = {"suggested_hostname": None, "conflicts": []}
        nomad_info: dict[str, Any] = {}

        # 1. Load conventions from Consul (or use defaults)
        with obs.span("load_conventions", input={"source": "consul" if consul else "defaults"}) as span:
            if consul is None:
                consul_conventions = _default_conventions()
                logger.info("Using default conventions (Consul unavailable)")
                span.end(output={"source": "defaults"})
            else:
                try:
                    consul_conventions = consul.get_conventions()
                    logger.info("Loaded conventions from Consul KV")
                    span.end(output={"source": "consul", "keys": list(consul_conventions.keys())})
                except Exception as e:
                    logger.warning(f"Failed to load conventions from Consul: {e}")
                    consul_conventions = _default_conventions()
                    span.end(level="WARNING", status_message=str(e), output={"source": "defaults_fallback"})

        # 2. Update Vault conventions if found and vault client available
        if vault is not None:
            with obs.span("set_vault_conventions") as span:
                vault_conv = consul_conventions.get("vault", {})
                if vault_conv:
                    try:
                        vault.set_conventions(
                            VaultConventions(
                                path_patterns=vault_conv.get("path_patterns", {}),
                                env_var_mappings=vault_conv.get("env_var_mappings", {}),
                                default_policy_pattern=vault_conv.get(
                                    "default_policy_pattern", "{app_name}-policy"
                                ),
                            )
                        )
                        span.end(output={"status": "set", "patterns": list(vault_conv.get("path_patterns", {}).keys())})
                    except Exception as e:
                        logger.warning(f"Failed to set Vault conventions: {e}")
                        span.end(level="WARNING", status_message=str(e))
                else:
                    span.end(output={"status": "no_conventions"})

        # 3. Generate multi-source env var configurations
        env_vars = analysis.get("env_vars_required", [])
        env_var_configs: list[dict] = []

        with obs.span("suggest_env_configs", input={"env_vars": env_vars, "count": len(env_vars)}) as span:
            if env_vars:
                try:
                    # Use new multi-source suggestion logic
                    configs = suggest_env_configs(env_vars, app_name, vault)
                    env_var_configs = [cfg.to_dict() for cfg in configs]
                    fixed_count = sum(1 for c in configs if c.source == "fixed")
                    consul_count = sum(1 for c in configs if c.source == "consul")
                    vault_count = sum(1 for c in configs if c.source == "vault")
                    logger.info(
                        f"Generated {len(env_var_configs)} env var configs "
                        f"(fixed: {fixed_count}, consul: {consul_count}, vault: {vault_count})"
                    )
                    span.end(output={
                        "total": len(env_var_configs),
                        "fixed": fixed_count,
                        "consul": consul_count,
                        "vault": vault_count,
                    })
                except Exception as e:
                    logger.warning(f"Failed to suggest env var configs: {e}")
                    span.end(level="WARNING", status_message=str(e))
            else:
                span.end(output={"total": 0, "reason": "no_env_vars"})

        # Apply port env var mapping from port_analysis (single app listening port)
        # Only the app's listening port env var should use Nomad's dynamic port
        port_analysis = state.get("port_analysis", {})
        port_env_mapping = port_analysis.get("recommended_env_mapping", {})
        if port_env_mapping:
            with obs.span("apply_port_env_mapping", input={"mapping": list(port_env_mapping.keys())}) as span:
                for env_var, nomad_ref in port_env_mapping.items():
                    # Find existing config for this env var and update it
                    found = False
                    for cfg in env_var_configs:
                        if cfg.get("name") == env_var:
                            cfg["source"] = "nomad"
                            cfg["value"] = nomad_ref
                            cfg["confidence"] = 0.95  # High confidence - LLM identified this
                            found = True
                            break
                    # If not found, add new config
                    if not found:
                        env_var_configs.append({
                            "name": env_var,
                            "source": "nomad",
                            "value": nomad_ref,
                            "confidence": 0.95,
                        })
                logger.info(f"Applied port env mapping: {env_var} -> {nomad_ref}")
                span.end(output={"env_var": env_var, "mapping": nomad_ref})

        # Also maintain legacy vault_suggestions for backward compatibility
        with obs.span("suggest_vault_mappings", input={"env_vars_count": len(env_vars)}) as span:
            if env_vars and vault is not None:
                try:
                    suggestions = vault.suggest_mappings(env_vars, app_name)
                    vault_suggestions["suggestions"] = [
                        {
                            "env_var": s.env_var,
                            "suggested_path": s.suggested_path,
                            "key": s.key,
                            "confidence": s.confidence,
                            "vault_reference": f"{s.suggested_path}#{s.key}",
                        }
                        for s in suggestions
                    ]
                    logger.info(f"Generated {len(suggestions)} Vault path suggestions")
                    span.end(output={"suggestions_count": len(suggestions)})
                except Exception as e:
                    logger.warning(f"Failed to suggest Vault mappings: {e}")
                    vault_suggestions["error"] = str(e)
                    span.end(level="WARNING", status_message=str(e))
            else:
                span.end(output={"skipped": True, "reason": "no_env_vars_or_vault"})

        # 4. Check Consul for service dependencies
        with obs.span("query_consul_services") as span:
            if consul is not None:
                try:
                    services = consul.list_services()
                    consul_services["available"] = list(services.keys())

                    # Identify potential dependencies from analysis
                    dependencies = _identify_dependencies(analysis)
                    for dep in dependencies:
                        health = consul.get_service_health(dep)
                        healthy = [s for s in health if s.health_status == "passing"]
                        consul_services["dependencies"].append({
                            "service": dep,
                            "available": len(healthy) > 0,
                            "healthy_instances": len(healthy),
                            "consul_dns": f"{dep}.service.consul",
                        })
                    span.end(output={
                        "services_available": len(consul_services["available"]),
                        "dependencies_checked": len(dependencies),
                    })
                except Exception as e:
                    logger.warning(f"Failed to query Consul services: {e}")
                    span.end(level="WARNING", status_message=str(e))
            else:
                span.end(output={"skipped": True, "reason": "consul_unavailable"})

        # 5. Check Fabio route availability
        with obs.span("validate_fabio_routes", input={"app_name": app_name}) as span:
            if fabio is not None:
                try:
                    fabio_conventions = consul_conventions.get("fabio", {})
                    suffix = fabio_conventions.get("hostname_suffix", ".cluster")
                    reserved = fabio_conventions.get("reserved_hostnames", [])

                    # Suggest hostname
                    suggested = fabio.suggest_hostname(app_name, suffix)
                    fabio_validation["suggested_hostname"] = suggested
                    fabio_validation["fabio_tag"] = f"urlprefix-{suggested}:9999/"

                    # Check for conflicts
                    if suggested not in reserved:
                        conflict = fabio.check_route_conflict(hostname=suggested)
                        if conflict:
                            fabio_validation["conflicts"].append({
                                "type": conflict.conflict_type,
                                "severity": conflict.severity,
                                "existing_service": conflict.existing_route.service,
                                "existing_route": conflict.existing_route.src,
                            })
                    else:
                        fabio_validation["conflicts"].append({
                            "type": "reserved",
                            "severity": "error",
                            "message": f"Hostname '{suggested}' is reserved",
                        })
                    span.end(output={
                        "suggested_hostname": suggested,
                        "conflicts_count": len(fabio_validation["conflicts"]),
                    })
                except Exception as e:
                    logger.warning(f"Failed to validate Fabio routes: {e}")
                    fabio_validation["error"] = str(e)
                    span.end(level="WARNING", status_message=str(e))
            else:
                # Provide default hostname suggestion even without Fabio
                fabio_conventions = consul_conventions.get("fabio", {})
                suffix = fabio_conventions.get("hostname_suffix", ".cluster")
                fabio_validation["suggested_hostname"] = f"{app_name}{suffix}"
                fabio_validation["fabio_tag"] = f"urlprefix-{app_name}{suffix}:9999/"
                span.end(output={
                    "suggested_hostname": fabio_validation["suggested_hostname"],
                    "source": "default",
                })

        # 6. Get Nomad version info
        with obs.span("get_nomad_version") as span:
            try:
                version = get_cached_nomad_version(
                    addr=settings.nomad_addr,
                    token=settings.nomad_token,
                )
                nomad_info = {
                    "version": str(version),
                    "supports_native_vault_env": version.supports_native_vault_env(),
                    "supports_csi_volumes": version.supports_csi_volumes(),
                    "recommended_vault_format": (
                        "env_stanza" if version.supports_native_vault_env() else "template"
                    ),
                }
                span.end(output=nomad_info)
            except Exception as e:
                logger.warning(f"Failed to get Nomad version: {e}")
                nomad_info = {
                    "version": "unknown",
                    "supports_native_vault_env": True,  # Assume modern Nomad
                    "recommended_vault_format": "env_stanza",
                }
                span.end(level="WARNING", status_message=str(e), output=nomad_info)

        return {
            "app_name": app_name,  # Store extracted app name for later use
            "env_var_configs": env_var_configs,  # Multi-source env var configurations
            "vault_suggestions": vault_suggestions,  # Legacy, for backward compatibility
            "consul_conventions": consul_conventions,
            "consul_services": consul_services,
            "fabio_validation": fabio_validation,
            "nomad_info": nomad_info,
            "infra_issues": infra_issues,  # Track any infrastructure connection issues
        }

    return enrich_node


def _extract_app_name(state: dict[str, Any]) -> str:
    """Extract application name from state or analysis."""
    # Try job_name first
    if state.get("job_name"):
        return state["job_name"]

    # Try to extract from codebase path
    codebase_path = state.get("codebase_path", "")
    if codebase_path:
        import os
        return os.path.basename(codebase_path.rstrip("/"))

    # Try to extract from analysis
    analysis = state.get("codebase_analysis", {})
    if analysis.get("project_name"):
        return analysis["project_name"]

    return "app"


def _identify_dependencies(analysis: dict[str, Any]) -> list[str]:
    """Identify service dependencies from codebase analysis.

    Looks for common patterns in dependencies and environment variables.
    """
    dependencies = set()

    # Check requirements/dependencies
    deps = analysis.get("dependencies", [])
    for dep in deps:
        dep_lower = dep.lower()
        if any(db in dep_lower for db in ["psycopg", "postgres", "pg"]):
            dependencies.add("postgres")
        if any(db in dep_lower for db in ["mysql", "mysqlclient", "pymysql"]):
            dependencies.add("mysql")
        if any(db in dep_lower for db in ["redis", "aioredis"]):
            dependencies.add("redis")
        if any(db in dep_lower for db in ["mongo", "pymongo"]):
            dependencies.add("mongodb")
        if any(db in dep_lower for db in ["elasticsearch"]):
            dependencies.add("elasticsearch")
        if any(db in dep_lower for db in ["rabbitmq", "pika", "amqp"]):
            dependencies.add("rabbitmq")
        if any(db in dep_lower for db in ["kafka"]):
            dependencies.add("kafka")

    # Check environment variables for connection strings
    env_vars = analysis.get("env_vars_required", [])
    for var in env_vars:
        var_upper = var.upper()
        if "POSTGRES" in var_upper or "PG_" in var_upper:
            dependencies.add("postgres")
        if "MYSQL" in var_upper:
            dependencies.add("mysql")
        if "REDIS" in var_upper:
            dependencies.add("redis")
        if "MONGO" in var_upper:
            dependencies.add("mongodb")
        if "RABBIT" in var_upper or "AMQP" in var_upper:
            dependencies.add("rabbitmq")
        if "KAFKA" in var_upper:
            dependencies.add("kafka")

    return list(dependencies)
