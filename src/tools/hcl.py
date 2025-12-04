"""HCL generation and validation tools for Nomad job specifications.

This module generates HCL compatible with the cluster's patterns:
- Terraform templating with ${datacenter} variable
- Proper escaping of Nomad runtime variables ($$)
- Architecture constraints for databases
- CSI volume patterns with init tasks
- Fabio routing tags
- Vault integration patterns
"""

import json
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class ServiceType(Enum):
    """Service type classification for resource allocation."""
    LIGHT = "light"  # Infrastructure: 100-200 CPU, 64-128 MB
    MEDIUM = "medium"  # Applications: 500-1000 CPU, 512-1024 MB
    HEAVY = "heavy"  # Databases: 1000-2000 CPU, 2048-4096 MB
    COMPUTE = "compute"  # ML/compute: 4000-8000 CPU, 8192-16384 MB


class NetworkMode(Enum):
    """Network mode for the job."""
    BRIDGE = "bridge"  # Default, dynamic ports
    HOST = "host"  # Required for static ports < 1024


@dataclass
class PortConfig:
    """Configuration for a single port."""
    name: str
    container_port: int
    static: bool = False  # If True, use static port allocation
    host_port: int | None = None  # Only used if static=True


@dataclass
class VolumeConfig:
    """CSI volume configuration."""
    name: str
    source: str  # Volume ID in Terraform/Nomad
    mount_path: str
    read_only: bool = False
    fs_type: str = "ext4"
    # UID:GID for init task chown (None = no init task needed)
    owner_uid: int | None = None
    owner_gid: int | None = None


@dataclass
class VaultConfig:
    """Vault integration configuration."""
    policies: list[str]
    secrets: dict[str, str] = field(default_factory=dict)  # env_var -> vault_path
    use_custom_delimiters: bool = False  # Use [[ ]] instead of {{ }}
    use_native_env: bool = False  # Use Nomad 1.4+ native vault env stanza


@dataclass
class FabioRoute:
    """Fabio routing configuration."""
    hostname: str | None = None  # e.g., "myapp.cluster" for host-based
    path: str | None = None  # e.g., "/myapp" for path-based
    strip_path: bool = False


@dataclass
class JobConfig:
    """Configuration for a Nomad job specification."""

    # Job metadata
    job_name: str
    job_type: str = "service"  # service, batch, system
    use_terraform_datacenter: bool = True  # Use ${datacenter} variable
    datacenters: list[str] = field(default_factory=lambda: ["dc1"])
    namespace: str = "default"
    region: str = "global"

    # Group/task configuration
    group_name: str | None = None
    task_name: str | None = None
    count: int = 1

    # Docker configuration
    image: str = ""
    force_pull: bool = True
    network_mode: NetworkMode = NetworkMode.BRIDGE
    ports: list[PortConfig] = field(default_factory=list)
    env_vars: dict[str, str] = field(default_factory=dict)
    consul_vars: dict[str, str] = field(default_factory=dict)  # env_var -> consul_kv_path
    nomad_port_vars: dict[str, str] = field(default_factory=dict)  # env_var -> port_label (for NOMAD_PORT_*)
    dns_servers: list[str] = field(default_factory=list)

    # Architecture constraint
    require_amd64: bool = True  # Most services need amd64 (not ARM64 servers)
    node_constraint: str | None = None  # Regexp for specific nodes

    # Resource limits
    service_type: ServiceType = ServiceType.MEDIUM
    cpu: int | None = None  # Override, otherwise based on service_type
    memory: int | None = None  # Override, otherwise based on service_type

    # Health check
    health_check_type: str = "http"  # http, tcp, none
    health_check_path: str = "/health"
    health_check_interval: str = "10s"
    health_check_timeout: str = "2s"

    # Service registration
    service_name: str | None = None
    service_tags: list[str] = field(default_factory=list)
    fabio_route: FabioRoute | None = None

    # Storage
    volumes: list[VolumeConfig] = field(default_factory=list)

    # Secrets
    vault: VaultConfig | None = None

    # Update policy
    max_parallel: int = 1
    min_healthy_time: str = "10s"
    healthy_deadline: str = "3m"
    progress_deadline: str = "5m"
    auto_revert: bool = True
    stagger: str | None = None  # Delay between instances

    # Restart policy
    restart_attempts: int = 3
    restart_interval: str = "5m"
    restart_delay: str = "15s"
    restart_mode: str = "delay"

    def __post_init__(self):
        """Set default values based on job_name if not provided."""
        if self.group_name is None:
            self.group_name = f"{self.job_name}"
        if self.task_name is None:
            self.task_name = self.job_name
        if self.service_name is None:
            self.service_name = self.job_name

        # Set resources based on service type if not overridden
        if self.cpu is None:
            self.cpu = self._default_cpu()
        if self.memory is None:
            self.memory = self._default_memory()

        # Convert legacy port dict format
        if isinstance(self.ports, dict):
            self.ports = [
                PortConfig(name=name, container_port=port)
                for name, port in self.ports.items()
            ]

    def _default_cpu(self) -> int:
        """Get default CPU based on service type."""
        return {
            ServiceType.LIGHT: 200,
            ServiceType.MEDIUM: 500,
            ServiceType.HEAVY: 1000,
            ServiceType.COMPUTE: 4000,
        }.get(self.service_type, 500)

    def _default_memory(self) -> int:
        """Get default memory based on service type."""
        return {
            ServiceType.LIGHT: 128,
            ServiceType.MEDIUM: 512,
            ServiceType.HEAVY: 2048,
            ServiceType.COMPUTE: 8192,
        }.get(self.service_type, 512)


def generate_hcl(config: JobConfig) -> str:
    """Generate a Nomad job specification in HCL format.

    Generates HCL following cluster patterns:
    - Terraform templating for datacenter
    - Proper $$ escaping for Nomad runtime vars
    - Architecture constraints
    - CSI volumes with init tasks
    - Fabio routing
    - Vault integration

    Args:
        config: Job configuration.

    Returns:
        HCL string for the Nomad job specification.
    """
    parts = []

    # Job header
    datacenter = '["${datacenter}"]' if config.use_terraform_datacenter else json.dumps(config.datacenters)
    parts.append(f'''job "{config.job_name}" {{
  datacenters = {datacenter}
  type        = "{config.job_type}"''')

    # Add namespace if not default
    if config.namespace != "default":
        parts.append(f'  namespace   = "{config.namespace}"')

    parts.append("")

    # Architecture constraint at job level (for non-system jobs)
    if config.require_amd64 and config.job_type != "system":
        parts.append('''  constraint {
    attribute = "$${attr.cpu.arch}"
    value     = "amd64"
  }
''')

    # Node constraint if specified
    if config.node_constraint:
        parts.append(f'''  constraint {{
    attribute = "${{node.unique.name}}"
    operator  = "regexp"
    value     = "{config.node_constraint}"
  }}
''')

    # Group
    parts.append(f'''  group "{config.group_name}" {{
    count = {config.count}''')

    # Constraint inside group for system jobs
    if config.require_amd64 and config.job_type == "system":
        parts.append('''
    constraint {
      attribute = "$${attr.cpu.arch}"
      value     = "amd64"
    }''')

    # Volumes
    for volume in config.volumes:
        parts.append(_build_volume_block(volume))

    # Network
    parts.append(_build_network_block(config))

    # Restart policy
    parts.append(f'''
    restart {{
      attempts = {config.restart_attempts}
      interval = "{config.restart_interval}"
      delay    = "{config.restart_delay}"
      mode     = "{config.restart_mode}"
    }}''')

    # Init task for volume permissions
    for volume in config.volumes:
        if volume.owner_uid is not None:
            parts.append(_build_init_task(volume))

    # Main task
    parts.append(_build_task_block(config))

    # Close group
    parts.append("  }")

    # Update policy
    parts.append(_build_update_block(config))

    # Close job
    parts.append("}")

    return "\n".join(parts)


def _build_volume_block(volume: VolumeConfig) -> str:
    """Build a CSI volume block."""
    return f'''
    volume "{volume.name}" {{
      type            = "csi"
      source          = "{volume.source}"
      attachment_mode = "file-system"
      access_mode     = "single-node-writer"
      per_alloc       = false
      mount_options {{
        fs_type = "{volume.fs_type}"
      }}
    }}'''


def _build_network_block(config: JobConfig) -> str:
    """Build the network block for the job spec."""
    ports = config.ports if config.ports else [PortConfig(name="http", container_port=8080)]

    # Determine if we need host mode
    needs_host_mode = config.network_mode == NetworkMode.HOST or any(
        p.static and (p.host_port or p.container_port) < 1024 for p in ports
    )

    mode_line = '\n      mode = "host"' if needs_host_mode else ""

    port_lines = []
    for port in ports:
        if port.static:
            port_num = port.host_port if port.host_port else port.container_port
            port_lines.append(f'''      port "{port.name}" {{
        static = {port_num}
      }}''')
        else:
            port_lines.append(f'''      port "{port.name}" {{
        to = {port.container_port}
      }}''')

    return f'''
    network {{{mode_line}
{chr(10).join(port_lines)}
    }}'''


def _build_init_task(volume: VolumeConfig) -> str:
    """Build init task for volume permission setup."""
    uid = volume.owner_uid
    gid = volume.owner_gid if volume.owner_gid else volume.owner_uid

    return f'''
    task "init-{volume.name}" {{
      driver = "docker"
      lifecycle {{
        hook    = "prestart"
        sidecar = false
      }}

      config {{
        image   = "busybox:latest"
        command = "/bin/sh"
        args    = ["-c", "chown -R {uid}:{gid} {volume.mount_path}"]
      }}

      resources {{
        cpu    = 100
        memory = 64
      }}

      volume_mount {{
        volume      = "{volume.name}"
        destination = "{volume.mount_path}"
        read_only   = false
      }}
    }}'''


def _build_task_block(config: JobConfig) -> str:
    """Build the task block for the job spec."""
    parts = []

    parts.append(f'''
    task "{config.task_name}" {{
      driver = "docker"''')

    # Vault block - only generate standalone if NOT using native env stanza
    # (native env stanza generates its own vault block with policies included)
    if config.vault:
        if config.vault.use_native_env and config.vault.secrets:
            pass  # Will be handled by _build_vault_env_stanza below
        else:
            policies = json.dumps(config.vault.policies)
            parts.append(f'''
      vault {{
        policies = {policies}
      }}''')

    # Docker config
    parts.append(_build_docker_config(config))

    # Volume mounts
    for volume in config.volumes:
        parts.append(f'''
      volume_mount {{
        volume      = "{volume.name}"
        destination = "{volume.mount_path}"
        read_only   = {"true" if volume.read_only else "false"}
      }}''')

    # Resources
    parts.append(f'''
      resources {{
        cpu    = {config.cpu}
        memory = {config.memory}
      }}''')

    # Vault secrets - use native env stanza if enabled, otherwise templates
    if config.vault and config.vault.secrets:
        if config.vault.use_native_env:
            parts.append(_build_vault_env_stanza(config.vault))
        else:
            parts.append(_build_vault_templates(config.vault))

    # Consul KV templates for non-secret configuration
    if config.consul_vars:
        parts.append(_build_consul_templates(config.consul_vars))

    # Environment variables (fixed values and nomad port mappings)
    if config.env_vars or config.nomad_port_vars:
        parts.append(_build_env_block(config))

    # Service registration (skip for batch jobs)
    if config.job_type != "batch":
        parts.append(_build_service_block(config))

    # Close task
    parts.append("    }")

    return "\n".join(parts)


def _build_docker_config(config: JobConfig) -> str:
    """Build the Docker driver config block."""
    port_names = [p.name for p in config.ports] if config.ports else ["http"]

    parts = [f'''
      config {{
        image      = "{config.image}"''']

    if config.force_pull:
        parts.append("        force_pull = true")

    parts.append(f"        ports      = {json.dumps(port_names)}")

    if config.dns_servers:
        parts.append(f"        dns_servers = {json.dumps(config.dns_servers)}")

    if config.network_mode == NetworkMode.HOST:
        parts.append('        network_mode = "host"')

    parts.append("      }")

    return "\n".join(parts)


def _build_vault_templates(vault: VaultConfig) -> str:
    """Build Vault secret templates."""
    parts = []

    left_delim = "[[" if vault.use_custom_delimiters else "{{"
    right_delim = "]]" if vault.use_custom_delimiters else "}}"

    for env_var, secret_path in vault.secrets.items():
        # Parse path like "secret/myapp/db.password" or "secret/myapp/db#password"
        vault_path, key = _parse_vault_path(secret_path)

        template = f'''
      template {{
        data = <<EOH
{left_delim} with secret "{vault_path}" {right_delim}
{env_var}="{left_delim} .Data.data.{key} {right_delim}"
{left_delim} end {right_delim}
EOH
        destination = "secrets/{env_var.lower()}.env"
        env         = true
        change_mode = "restart"'''

        if vault.use_custom_delimiters:
            template += '''
        left_delimiter  = "[["
        right_delimiter = "]]"'''

        template += "\n      }"
        parts.append(template)

    return "\n".join(parts)


def _build_vault_env_stanza(vault: VaultConfig) -> str:
    """Build Nomad 1.4+ native vault env stanza with policies.

    This generates the newer format with policies included:
    vault {
      policies = ["app-policy"]
      env {
        DB_PASSWORD = "secret/data/myapp/db#password"
      }
    }
    """
    policies = json.dumps(vault.policies)
    env_lines = []
    for env_var, secret_path in vault.secrets.items():
        # Normalize path to use # separator for Nomad native format
        vault_path, key = _parse_vault_path(secret_path)
        # Ensure path has /data/ for KV v2
        if not vault_path.startswith("secret/data/"):
            vault_path = vault_path.replace("secret/", "secret/data/")
        env_lines.append(f'        {env_var} = "{vault_path}#{key}"')

    return f'''
      vault {{
        policies = {policies}
        env {{
{chr(10).join(env_lines)}
        }}
      }}'''


def _parse_vault_path(secret_path: str) -> tuple[str, str]:
    """Parse a Vault secret path into path and key components.

    Supports formats:
    - "secret/data/myapp/db#password" -> ("secret/data/myapp/db", "password")
    - "secret/myapp/db.password" -> ("secret/myapp/db", "password")
    - "secret/myapp/db" -> ("secret/myapp/db", "value")

    Returns:
        Tuple of (vault_path, key).
    """
    # Check for # separator (Nomad native format)
    if "#" in secret_path:
        return secret_path.rsplit("#", 1)

    # Check for . separator in last path component
    if "." in secret_path.split("/")[-1]:
        path_parts = secret_path.rsplit(".", 1)
        return path_parts[0], path_parts[1]

    # No key specified, default to "value"
    return secret_path, "value"


def _build_consul_templates(consul_vars: dict[str, str]) -> str:
    """Build Consul KV templates for environment variables.

    Generates Nomad template blocks that read values from Consul KV store.
    Each variable gets its own template block that writes to a .env file.
    The template outputs KEY=VALUE format which is parsed when env=true.

    Args:
        consul_vars: Dict mapping env var names to Consul KV paths.
                    e.g., {"REDIS_URL": "myapp/config/redis_url"}

    Returns:
        HCL template blocks as a string.
    """
    if not consul_vars:
        return ""

    parts = []

    for env_var, consul_path in consul_vars.items():
        template = f'''
      template {{
        data = <<EOH
{env_var}={{{{ key "{consul_path}" }}}}
EOH
        destination = "local/{env_var.lower()}.env"
        env         = true
        change_mode = "restart"
      }}'''
        parts.append(template)

    return "\n".join(parts)


def _build_env_block(config: JobConfig) -> str:
    """Build the environment variables block.

    Handles three types of env vars:
    - Fixed values from config.env_vars
    - Nomad dynamic ports from config.nomad_port_vars
    """
    env_lines = []

    # Fixed env vars
    for key, value in config.env_vars.items():
        # Escape Nomad runtime variables
        if value.startswith("${NOMAD_"):
            value = value.replace("${", "$${")
        env_lines.append(f'        {key} = "{value}"')

    # Nomad port env vars (env_var -> port_label)
    for env_var, port_label in config.nomad_port_vars.items():
        # Use $$ to escape the Nomad runtime variable
        env_lines.append(f'        {env_var} = "${{NOMAD_PORT_{port_label}}}"')

    if not env_lines:
        return ""

    return f'''
      env {{
{chr(10).join(env_lines)}
      }}'''


def _build_service_block(config: JobConfig) -> str:
    """Build the service registration block."""
    port_name = config.ports[0].name if config.ports else "http"

    # Build tags
    tags = list(config.service_tags)
    if config.fabio_route:
        if config.fabio_route.hostname:
            tags.append(f"urlprefix-{config.fabio_route.hostname}:9999/")
        elif config.fabio_route.path:
            tag = f"urlprefix-{config.fabio_route.path}"
            if config.fabio_route.strip_path:
                tag += f" strip={config.fabio_route.path}"
            tags.append(tag)

    tags_str = json.dumps(tags) if tags else "[]"

    # Build health check
    if config.health_check_type == "http":
        check_block = f'''
        check {{
          type     = "http"
          path     = "{config.health_check_path}"
          interval = "{config.health_check_interval}"
          timeout  = "{config.health_check_timeout}"
        }}'''
    elif config.health_check_type == "tcp":
        check_block = f'''
        check {{
          type     = "tcp"
          interval = "{config.health_check_interval}"
          timeout  = "{config.health_check_timeout}"
        }}'''
    else:
        check_block = ""

    return f'''
      service {{
        name = "{config.service_name}"
        port = "{port_name}"
        tags = {tags_str}
{check_block}
      }}'''


def _build_update_block(config: JobConfig) -> str:
    """Build the update policy block."""
    parts = [f'''
  update {{
    max_parallel      = {config.max_parallel}
    min_healthy_time  = "{config.min_healthy_time}"
    healthy_deadline  = "{config.healthy_deadline}"
    progress_deadline = "{config.progress_deadline}"
    auto_revert       = {"true" if config.auto_revert else "false"}''']

    if config.stagger:
        parts.append(f'    stagger           = "{config.stagger}"')

    parts.append("  }")

    return "\n".join(parts)


# Keep original simple generation for backwards compatibility
def generate_simple_hcl(
    job_name: str,
    image: str,
    ports: dict[str, int] | None = None,
    env_vars: dict[str, str] | None = None,
    cpu: int = 500,
    memory: int = 256,
) -> str:
    """Generate a simple HCL job spec (backwards compatible).

    For more control, use JobConfig and generate_hcl() directly.
    """
    port_configs = []
    if ports:
        for name, port in ports.items():
            port_configs.append(PortConfig(name=name, container_port=port))

    config = JobConfig(
        job_name=job_name,
        image=image,
        ports=port_configs,
        env_vars=env_vars or {},
        cpu=cpu,
        memory=memory,
    )
    return generate_hcl(config)


def validate_hcl(hcl_content: str, nomad_addr: str | None = None) -> tuple[bool, str | None]:
    """Validate HCL content using Nomad CLI.

    Args:
        hcl_content: The HCL job specification content.
        nomad_addr: Optional Nomad server address for validation.

    Returns:
        Tuple of (is_valid, error_message).
    """
    # Write HCL to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".nomad", delete=False) as f:
        f.write(hcl_content)
        temp_path = f.name

    try:
        # Build command
        cmd = ["nomad", "job", "validate"]
        if nomad_addr:
            cmd.extend(["-address", nomad_addr])
        cmd.append(temp_path)

        # Run validation
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            return True, None
        else:
            error = result.stderr or result.stdout
            return False, error.strip()

    except subprocess.TimeoutExpired:
        return False, "Validation timed out"
    except FileNotFoundError:
        # Nomad CLI not installed - do basic syntax check
        return _basic_hcl_validation(hcl_content)
    finally:
        Path(temp_path).unlink(missing_ok=True)


def _basic_hcl_validation(hcl_content: str) -> tuple[bool, str | None]:
    """Basic HCL syntax validation without Nomad CLI.

    This is a fallback when nomad CLI is not available.
    """
    errors = []

    # Check for balanced braces
    open_braces = hcl_content.count("{")
    close_braces = hcl_content.count("}")
    if open_braces != close_braces:
        errors.append(f"Unbalanced braces: {open_braces} open, {close_braces} close")

    # Check for required blocks
    if 'job "' not in hcl_content:
        errors.append("Missing job block")
    if "group " not in hcl_content:
        errors.append("Missing group block")
    if "task " not in hcl_content:
        errors.append("Missing task block")

    # Check for driver specification
    if 'driver = "docker"' not in hcl_content and 'driver = "exec"' not in hcl_content:
        errors.append("Missing or unsupported driver specification")

    # Check for image in docker driver
    if 'driver = "docker"' in hcl_content and "image = " not in hcl_content:
        errors.append("Docker driver specified but no image configured")

    if errors:
        return False, "; ".join(errors)
    return True, None


def hcl_to_json(hcl_content: str) -> dict[str, Any] | None:
    """Convert HCL to JSON using Nomad CLI.

    Args:
        hcl_content: The HCL job specification content.

    Returns:
        Dictionary representation of the job, or None if conversion fails.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".nomad", delete=False) as f:
        f.write(hcl_content)
        temp_path = f.name

    try:
        result = subprocess.run(
            ["nomad", "job", "run", "-output", temp_path],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            return json.loads(result.stdout)
        return None

    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
        return None
    finally:
        Path(temp_path).unlink(missing_ok=True)


def extract_job_name(hcl_content: str) -> str | None:
    """Extract the job name from HCL content.

    Args:
        hcl_content: The HCL job specification content.

    Returns:
        Job name or None if not found.
    """
    match = re.search(r'job\s+"([^"]+)"', hcl_content)
    return match.group(1) if match else None


def sanitize_job_name(name: str) -> str:
    """Sanitize a string to be a valid Nomad job name.

    Args:
        name: Original name.

    Returns:
        Sanitized job name.
    """
    # Convert to lowercase
    name = name.lower()
    # Replace invalid characters with hyphens
    name = re.sub(r"[^a-z0-9-]", "-", name)
    # Remove consecutive hyphens
    name = re.sub(r"-+", "-", name)
    # Remove leading/trailing hyphens
    name = name.strip("-")
    # Ensure it doesn't start with a number
    if name and name[0].isdigit():
        name = f"job-{name}"
    # Truncate to 63 characters (Nomad limit)
    return name[:63]


def merge_hcl_configs(base_config: JobConfig, overrides: dict[str, Any]) -> JobConfig:
    """Merge override values into a base job configuration.

    Args:
        base_config: Base JobConfig instance.
        overrides: Dictionary of values to override.

    Returns:
        New JobConfig with merged values.
    """
    # Get current values as dict
    config_dict = {
        "job_name": base_config.job_name,
        "job_type": base_config.job_type,
        "use_terraform_datacenter": base_config.use_terraform_datacenter,
        "datacenters": list(base_config.datacenters),
        "namespace": base_config.namespace,
        "region": base_config.region,
        "group_name": base_config.group_name,
        "task_name": base_config.task_name,
        "count": base_config.count,
        "image": base_config.image,
        "force_pull": base_config.force_pull,
        "network_mode": base_config.network_mode,
        "ports": list(base_config.ports),
        "env_vars": dict(base_config.env_vars),
        "consul_vars": dict(base_config.consul_vars),
        "nomad_port_vars": dict(base_config.nomad_port_vars),
        "dns_servers": list(base_config.dns_servers),
        "require_amd64": base_config.require_amd64,
        "node_constraint": base_config.node_constraint,
        "service_type": base_config.service_type,
        "cpu": base_config.cpu,
        "memory": base_config.memory,
        "health_check_type": base_config.health_check_type,
        "health_check_path": base_config.health_check_path,
        "health_check_interval": base_config.health_check_interval,
        "health_check_timeout": base_config.health_check_timeout,
        "service_name": base_config.service_name,
        "service_tags": list(base_config.service_tags),
        "fabio_route": base_config.fabio_route,
        "volumes": list(base_config.volumes),
        "vault": base_config.vault,
        "max_parallel": base_config.max_parallel,
        "min_healthy_time": base_config.min_healthy_time,
        "healthy_deadline": base_config.healthy_deadline,
        "progress_deadline": base_config.progress_deadline,
        "auto_revert": base_config.auto_revert,
        "stagger": base_config.stagger,
        "restart_attempts": base_config.restart_attempts,
        "restart_interval": base_config.restart_interval,
        "restart_delay": base_config.restart_delay,
        "restart_mode": base_config.restart_mode,
    }

    # Apply overrides
    for key, value in overrides.items():
        if key in config_dict:
            if isinstance(config_dict[key], dict) and isinstance(value, dict):
                config_dict[key].update(value)
            elif isinstance(config_dict[key], list) and isinstance(value, list):
                config_dict[key] = value
            else:
                config_dict[key] = value

    return JobConfig(**config_dict)
