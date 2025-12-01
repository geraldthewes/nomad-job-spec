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
from src.tools.vault import VaultClient, VaultConventions, get_vault_client, set_vault_client
from src.tools.consul import ConsulClient, get_consul_client, set_consul_client
from src.tools.fabio import FabioClient, get_fabio_client, set_fabio_client
from src.tools.nomad_version import get_cached_nomad_version

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

    # Initialize clients if provided
    if vault_client:
        set_vault_client(vault_client)
    if consul_client:
        set_consul_client(consul_client)
    if fabio_client:
        set_fabio_client(fabio_client)

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
        analysis = state.get("codebase_analysis", {})
        app_name = _extract_app_name(state)

        # Try to get clients - may fail if services unavailable
        vault = None
        consul = None
        fabio = None

        try:
            vault = get_vault_client()
        except Exception as e:
            logger.warning(f"Could not initialize Vault client: {e}")

        try:
            consul = get_consul_client()
        except Exception as e:
            logger.warning(f"Could not initialize Consul client: {e}")

        try:
            fabio = get_fabio_client()
        except Exception as e:
            logger.warning(f"Could not initialize Fabio client: {e}")

        # Initialize enrichment data
        vault_suggestions: dict[str, Any] = {"suggestions": [], "error": None}
        consul_conventions: dict[str, Any] = {}
        consul_services: dict[str, Any] = {"available": [], "dependencies": []}
        fabio_validation: dict[str, Any] = {"suggested_hostname": None, "conflicts": []}
        nomad_info: dict[str, Any] = {}

        # 1. Load conventions from Consul (or use defaults)
        if consul is None:
            consul_conventions = _default_conventions()
            logger.info("Using default conventions (Consul unavailable)")
        else:
            try:
                consul_conventions = consul.get_conventions()
                logger.info("Loaded conventions from Consul KV")
            except Exception as e:
                logger.warning(f"Failed to load conventions from Consul: {e}")
                consul_conventions = _default_conventions()

        # 2. Update Vault conventions if found and vault client available
        if vault is not None:
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
                except Exception as e:
                    logger.warning(f"Failed to set Vault conventions: {e}")

        # 3. Query Vault for secret path suggestions
        env_vars = analysis.get("env_vars_required", [])
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
            except Exception as e:
                logger.warning(f"Failed to suggest Vault mappings: {e}")
                vault_suggestions["error"] = str(e)

        # 4. Check Consul for service dependencies
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
            except Exception as e:
                logger.warning(f"Failed to query Consul services: {e}")

        # 5. Check Fabio route availability
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
            except Exception as e:
                logger.warning(f"Failed to validate Fabio routes: {e}")
                fabio_validation["error"] = str(e)
        else:
            # Provide default hostname suggestion even without Fabio
            fabio_conventions = consul_conventions.get("fabio", {})
            suffix = fabio_conventions.get("hostname_suffix", ".cluster")
            fabio_validation["suggested_hostname"] = f"{app_name}{suffix}"
            fabio_validation["fabio_tag"] = f"urlprefix-{app_name}{suffix}:9999/"

        # 6. Get Nomad version info
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
        except Exception as e:
            logger.warning(f"Failed to get Nomad version: {e}")
            nomad_info = {
                "version": "unknown",
                "supports_native_vault_env": True,  # Assume modern Nomad
                "recommended_vault_format": "env_stanza",
            }

        return {
            **state,
            "vault_suggestions": vault_suggestions,
            "consul_conventions": consul_conventions,
            "consul_services": consul_services,
            "fabio_validation": fabio_validation,
            "nomad_info": nomad_info,
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
