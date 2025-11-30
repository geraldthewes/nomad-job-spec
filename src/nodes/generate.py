"""HCL spec generation node for the LangGraph workflow."""

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

from src.tools.hcl import (
    JobConfig,
    PortConfig,
    VolumeConfig,
    VaultConfig,
    FabioRoute,
    ServiceType,
    generate_hcl,
    validate_hcl,
    sanitize_job_name,
)
from config.settings import get_settings


GENERATION_SYSTEM_PROMPT = """You are an expert DevOps engineer generating HashiCorp Nomad job specifications.

Your task is to generate a valid Nomad job configuration based on the provided codebase analysis and user requirements.

## Cluster Environment

This Nomad cluster uses:
- Terraform templating: datacenters will use ${datacenter} variable
- Architecture constraint: Most services need AMD64 nodes (gpu001-gpu007)
- CSI volumes via Ceph for persistent storage
- Fabio load balancer on port 9999 for external HTTP routing
- Vault integration for secrets

## Service Types and Resources

- LIGHT: 200 CPU, 128 MB (proxies, sidecars)
- MEDIUM: 500 CPU, 512 MB (web apps, APIs)
- HEAVY: 1000 CPU, 2048 MB (databases)
- COMPUTE: 4000 CPU, 8192 MB (ML workloads)

## Guidelines

1. Use the Docker driver for all jobs
2. Set resources based on service type classification
3. Configure HTTP health checks when the app has endpoints
4. Use dynamic ports (not static) unless port < 1024 is required
5. For Fabio routing, use hostname-based: "urlprefix-hostname:9999/"
6. For storage, specify volume name, path, and owner UID for init task
7. For secrets, list Vault policy and secret paths

You must output ONLY a valid JSON object with the following structure (no markdown, no explanation):
{
    "job_name": "sanitized-job-name",
    "service_type": "LIGHT|MEDIUM|HEAVY|COMPUTE",
    "image": "docker-image:tag",
    "ports": [{"name": "http", "container_port": 8080, "static": false}],
    "env_vars": {"VAR_NAME": "value"},
    "cpu": null,
    "memory": null,
    "health_check_type": "http|tcp|none",
    "health_check_path": "/health",
    "count": 1,
    "service_tags": ["tag1", "tag2"],
    "require_amd64": true,
    "fabio_hostname": null,
    "fabio_path": null,
    "volume": null,
    "vault_policies": [],
    "vault_secrets": {}
}

Notes on volume format (if needed):
{
    "volume": {
        "name": "data",
        "source": "myapp-data",
        "mount_path": "/var/lib/myapp/data",
        "owner_uid": 1000
    }
}

Notes on vault_secrets format (if needed):
{
    "vault_secrets": {
        "DB_PASSWORD": "secret/myapp/db.password",
        "API_KEY": "secret/myapp/api.key"
    }
}

Important:
- Job names must be lowercase, alphanumeric with hyphens only
- Set cpu/memory to null to use service_type defaults
- Most database images require AMD64 architecture
- Only use static ports if absolutely required (e.g., port 80, 443)
"""


def generate_spec_node(
    state: dict[str, Any],
    llm: BaseChatModel,
) -> dict[str, Any]:
    """Generate a Nomad job specification from analysis and user responses.

    Args:
        state: Current graph state with 'codebase_analysis' and 'user_responses'.
        llm: LLM instance for generation.

    Returns:
        Updated state with 'job_spec' (HCL string) and 'job_config' fields.
    """
    settings = get_settings()

    analysis = state.get("codebase_analysis", {})
    user_responses = state.get("user_responses", {})
    prompt = state.get("prompt", "")
    memories = state.get("relevant_memories", [])

    # Build context for LLM
    context = _build_generation_context(analysis, user_responses, prompt, memories)

    # Query LLM for configuration
    messages = [
        SystemMessage(content=GENERATION_SYSTEM_PROMPT),
        HumanMessage(content=context),
    ]

    response = llm.invoke(messages)
    response_text = response.content if hasattr(response, "content") else str(response)

    # Parse LLM response into JobConfig
    try:
        config_dict = _parse_llm_response(response_text)
        config = _build_job_config(config_dict, analysis, settings)
    except Exception as e:
        # Fallback to basic config from analysis
        config = _build_fallback_config(analysis, prompt, settings)
        config_dict = {"error": str(e), "fallback": True}

    # Generate HCL
    hcl_content = generate_hcl(config)

    # Validate HCL
    is_valid, validation_error = validate_hcl(hcl_content, settings.nomad_address)

    return {
        **state,
        "job_spec": hcl_content,
        "job_config": config_dict,
        "job_name": config.job_name,
        "hcl_valid": is_valid,
        "validation_error": validation_error,
    }


def _build_generation_context(
    analysis: dict[str, Any],
    user_responses: dict[str, str],
    prompt: str,
    memories: list[str],
) -> str:
    """Build context string for LLM generation."""
    parts = [
        "## User Request",
        prompt or "Deploy this application",
        "",
        "## Codebase Analysis",
        f"```json\n{json.dumps(analysis, indent=2)}\n```",
    ]

    if user_responses:
        parts.extend([
            "",
            "## User Responses to Questions",
            f"```json\n{json.dumps(user_responses, indent=2)}\n```",
        ])

    if memories:
        parts.extend([
            "",
            "## Relevant Past Experiences",
            *[f"- {m}" for m in memories],
        ])

    parts.extend([
        "",
        "Generate a Nomad job configuration for this application.",
        "Output ONLY a valid JSON object with the job configuration.",
    ])

    return "\n".join(parts)


def _parse_llm_response(response_text: str) -> dict[str, Any]:
    """Parse LLM response to extract job configuration."""
    # Try to extract JSON from response
    json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", response_text, re.DOTALL)

    if json_match:
        return json.loads(json_match.group(0))

    # Try markdown code block
    code_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    if code_block:
        return json.loads(code_block.group(1))

    raise ValueError(f"Could not parse JSON from response: {response_text[:500]}")


def _build_job_config(
    config_dict: dict[str, Any],
    analysis: dict[str, Any],
    settings,
) -> JobConfig:
    """Build JobConfig from LLM response and analysis."""
    # Get job name
    job_name = config_dict.get("job_name", "unnamed-job")
    job_name = sanitize_job_name(job_name)

    # Get image
    image = config_dict.get("image", "")
    if not image and analysis.get("dockerfile", {}).get("base_image"):
        # Use base image as fallback (not ideal but better than nothing)
        image = analysis["dockerfile"]["base_image"]

    # Parse service type
    service_type_str = config_dict.get("service_type", "MEDIUM").upper()
    service_type = ServiceType[service_type_str] if service_type_str in ServiceType.__members__ else ServiceType.MEDIUM

    # Get ports - handle both old dict format and new list format
    raw_ports = config_dict.get("ports", [])
    if isinstance(raw_ports, dict):
        # Legacy format: {"http": 8080}
        ports = [PortConfig(name=name, container_port=port) for name, port in raw_ports.items()]
    elif isinstance(raw_ports, list):
        # New format: [{"name": "http", "container_port": 8080, "static": false}]
        ports = []
        for p in raw_ports:
            if isinstance(p, dict):
                ports.append(PortConfig(
                    name=p.get("name", "http"),
                    container_port=p.get("container_port", 8080),
                    static=p.get("static", False),
                    host_port=p.get("host_port"),
                ))
            elif isinstance(p, PortConfig):
                ports.append(p)
    else:
        ports = []

    # Fallback to Dockerfile exposed ports
    if not ports and analysis.get("dockerfile", {}).get("exposed_ports"):
        exposed = analysis["dockerfile"]["exposed_ports"]
        if exposed:
            ports = [PortConfig(name="http", container_port=exposed[0])]

    if not ports:
        ports = [PortConfig(name="http", container_port=8080)]  # Default

    # Get environment variables
    env_vars = config_dict.get("env_vars", {})

    # Get resources (None means use service_type defaults)
    cpu = config_dict.get("cpu")
    memory = config_dict.get("memory")

    # Get health check config
    health_check_type = config_dict.get("health_check_type", "http")
    health_check_path = config_dict.get("health_check_path", "/health")
    if health_check_type == "none":
        health_check_type = "tcp"  # Fall back to TCP

    # Get architecture requirement
    require_amd64 = config_dict.get("require_amd64", True)

    # Build Fabio route if specified
    fabio_route = None
    if config_dict.get("fabio_hostname") or config_dict.get("fabio_path"):
        fabio_route = FabioRoute(
            hostname=config_dict.get("fabio_hostname"),
            path=config_dict.get("fabio_path"),
        )

    # Build volume config if specified
    volumes = []
    if config_dict.get("volume"):
        vol = config_dict["volume"]
        volumes.append(VolumeConfig(
            name=vol.get("name", "data"),
            source=vol.get("source", f"{job_name}-data"),
            mount_path=vol.get("mount_path", "/data"),
            owner_uid=vol.get("owner_uid"),
            owner_gid=vol.get("owner_gid"),
        ))

    # Build Vault config if specified
    vault = None
    if config_dict.get("vault_policies") or config_dict.get("vault_secrets"):
        vault = VaultConfig(
            policies=config_dict.get("vault_policies", []),
            secrets=config_dict.get("vault_secrets", {}),
        )

    # Build config
    return JobConfig(
        job_name=job_name,
        datacenters=[settings.nomad_datacenter],
        namespace=settings.nomad_namespace,
        region=settings.nomad_region,
        image=image,
        ports=ports,
        env_vars=env_vars,
        service_type=service_type,
        cpu=cpu,
        memory=memory,
        health_check_type=health_check_type,
        health_check_path=health_check_path,
        count=config_dict.get("count", 1),
        service_tags=config_dict.get("service_tags", []),
        require_amd64=require_amd64,
        fabio_route=fabio_route,
        volumes=volumes,
        vault=vault,
    )


def _build_fallback_config(
    analysis: dict[str, Any],
    prompt: str,
    settings,
) -> JobConfig:
    """Build a fallback JobConfig when LLM parsing fails."""
    # Extract job name from prompt or analysis
    job_name = "unnamed-job"
    if prompt:
        # Try to extract a name from the prompt
        words = prompt.lower().split()
        for i, word in enumerate(words):
            if word in ("deploy", "create", "run") and i + 1 < len(words):
                job_name = words[i + 1]
                break

    job_name = sanitize_job_name(job_name)

    # Get ports from analysis
    ports = [PortConfig(name="http", container_port=8080)]
    if analysis.get("dockerfile", {}).get("exposed_ports"):
        exposed = analysis["dockerfile"]["exposed_ports"]
        if exposed:
            ports = [PortConfig(name="http", container_port=exposed[0])]

    # Get resources
    resources = analysis.get("suggested_resources", {})

    return JobConfig(
        job_name=job_name,
        datacenters=[settings.nomad_datacenter],
        namespace=settings.nomad_namespace,
        region=settings.nomad_region,
        image=analysis.get("dockerfile", {}).get("base_image", "nginx:latest"),
        ports=ports,
        env_vars={},
        cpu=resources.get("cpu", settings.default_cpu),
        memory=resources.get("memory", settings.default_memory),
        service_type=ServiceType.MEDIUM,  # Default to medium for fallback
    )


def regenerate_spec_with_fix(
    state: dict[str, Any],
    llm: BaseChatModel,
) -> dict[str, Any]:
    """Regenerate spec with error context for fixing deployment issues.

    This is used in the fix iteration loop.

    Args:
        state: Current state with deployment error information.
        llm: LLM instance.

    Returns:
        Updated state with new job_spec.
    """
    error = state.get("deployment_error", "Unknown error")
    current_spec = state.get("job_spec", "")
    memories = state.get("relevant_memories", [])

    fix_prompt = f"""The previous job specification failed to deploy with the following error:

ERROR: {error}

CURRENT SPEC:
```hcl
{current_spec}
```

{"SIMILAR PAST ERRORS AND FIXES:" + chr(10) + chr(10).join(f"- {m}" for m in memories) if memories else ""}

Please generate a FIXED job configuration that addresses this error.
Common fixes:
- "insufficient memory" -> increase memory allocation
- "port already in use" -> use dynamic ports (remove 'static' port binding)
- "image not found" -> verify image name and tag
- "constraint not satisfied" -> adjust datacenter or remove constraints

Output ONLY a valid JSON object with the fixed job configuration.
"""

    messages = [
        SystemMessage(content=GENERATION_SYSTEM_PROMPT),
        HumanMessage(content=fix_prompt),
    ]

    response = llm.invoke(messages)
    response_text = response.content if hasattr(response, "content") else str(response)

    settings = get_settings()

    try:
        config_dict = _parse_llm_response(response_text)
        config = _build_job_config(config_dict, state.get("codebase_analysis", {}), settings)
    except Exception:
        # If fix fails, try incrementing resources
        current_config = state.get("job_config", {})
        config = _increment_resources(current_config, error, settings)

    hcl_content = generate_hcl(config)
    is_valid, validation_error = validate_hcl(hcl_content, settings.nomad_address)

    return {
        **state,
        "job_spec": hcl_content,
        "job_config": config.__dict__ if hasattr(config, "__dict__") else {},
        "job_name": config.job_name,
        "hcl_valid": is_valid,
        "validation_error": validation_error,
        "iteration_count": state.get("iteration_count", 0) + 1,
    }


def _increment_resources(
    current_config: dict[str, Any],
    error: str,
    settings,
) -> JobConfig:
    """Increment resources as a fallback fix strategy."""
    error_lower = error.lower()

    cpu = current_config.get("cpu", settings.default_cpu)
    memory = current_config.get("memory", settings.default_memory)

    if "memory" in error_lower or "oom" in error_lower:
        memory = min(memory * 2, 4096)  # Double memory, max 4GB
    elif "cpu" in error_lower:
        cpu = min(cpu * 2, 4000)  # Double CPU, max 4000MHz
    else:
        # Generic increase
        memory = min(int(memory * 1.5), 4096)
        cpu = min(int(cpu * 1.5), 4000)

    # Get ports - handle both old dict format and new list format
    raw_ports = current_config.get("ports", [])
    if isinstance(raw_ports, dict):
        ports = [PortConfig(name=name, container_port=port) for name, port in raw_ports.items()]
    elif isinstance(raw_ports, list) and raw_ports and isinstance(raw_ports[0], PortConfig):
        ports = raw_ports
    else:
        ports = [PortConfig(name="http", container_port=8080)]

    return JobConfig(
        job_name=current_config.get("job_name", "unnamed-job"),
        datacenters=current_config.get("datacenters", [settings.nomad_datacenter]),
        namespace=current_config.get("namespace", settings.nomad_namespace),
        region=current_config.get("region", settings.nomad_region),
        image=current_config.get("image", "nginx:latest"),
        ports=ports,
        env_vars=current_config.get("env_vars", {}),
        cpu=cpu,
        memory=memory,
    )


def create_generate_node(llm: BaseChatModel):
    """Create a generate node function with the given LLM.

    Args:
        llm: LLM instance for generation.

    Returns:
        Node function for use in LangGraph.
    """

    def node(state: dict[str, Any]) -> dict[str, Any]:
        return generate_spec_node(state, llm)

    return node


def create_fix_node(llm: BaseChatModel):
    """Create a fix node function with the given LLM.

    Args:
        llm: LLM instance for fixing.

    Returns:
        Node function for use in LangGraph.
    """

    def node(state: dict[str, Any]) -> dict[str, Any]:
        return regenerate_spec_with_fix(state, llm)

    return node
