"""HCL spec generation node for the LangGraph workflow."""

import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

from src.observability import get_observability
from src.tools.hcl import (
    JobConfig,
    PortConfig,
    VolumeConfig,
    VaultConfig,
    FabioRoute,
    ServiceType,
    NetworkMode,
    generate_hcl,
    validate_hcl,
    sanitize_job_name,
)
from config.settings import get_settings

logger = logging.getLogger(__name__)


# Fallback prompt used when LangFuse and local file are unavailable
_FALLBACK_GENERATION_PROMPT = """You are an expert DevOps engineer generating HashiCorp Nomad job specifications.

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
        "DB_PASSWORD": "secret/data/myapp/db#password",
        "API_KEY": "secret/data/myapp/api#key"
    }
}

Use the "#" separator for vault secrets (e.g., "secret/data/app/db#password").
If Vault path suggestions are provided, use them as-is.

Important:
- Job names must be lowercase, alphanumeric with hyphens only
- Set cpu/memory to null to use service_type defaults
- Most database images require AMD64 architecture
- Only use static ports if absolutely required (e.g., port 80, 443)
"""


# Fallback fix prompt used when LangFuse and local file are unavailable
_FALLBACK_FIX_PROMPT = """You are an expert DevOps engineer fixing a failed Nomad job deployment.

The previous job specification failed to deploy. Your task is to analyze the error and generate a corrected job configuration.

Common fixes:
- "insufficient memory" -> increase memory allocation
- "port already in use" -> use dynamic ports (remove 'static' port binding)
- "image not found" -> verify image name and tag
- "constraint not satisfied" -> adjust datacenter or remove constraints
- "permission denied" -> check volume mount permissions and owner UID
- "health check failed" -> verify health check path and increase timeout

Output ONLY a valid JSON object with the fixed job configuration using the same structure as the generation prompt.
"""


def _get_generation_prompt() -> str:
    """Get the generation system prompt.

    Tries LangFuse first, falls back to local file, then hardcoded default.

    Returns:
        The generation system prompt string.
    """
    try:
        from src.prompts import get_prompt_manager, PromptNotFoundError

        manager = get_prompt_manager()
        return manager.get_prompt_text("generation")
    except PromptNotFoundError:
        logger.warning("Generation prompt not found, using fallback")
        return _FALLBACK_GENERATION_PROMPT
    except Exception as e:
        logger.warning(f"Failed to get generation prompt: {e}, using fallback")
        return _FALLBACK_GENERATION_PROMPT


def _get_fix_prompt() -> str:
    """Get the fix system prompt.

    Tries LangFuse first, falls back to local file, then hardcoded default.

    Returns:
        The fix system prompt string.
    """
    try:
        from src.prompts import get_prompt_manager, PromptNotFoundError

        manager = get_prompt_manager()
        return manager.get_prompt_text("fix")
    except PromptNotFoundError:
        logger.warning("Fix prompt not found, using fallback")
        return _FALLBACK_FIX_PROMPT
    except Exception as e:
        logger.warning(f"Failed to get fix prompt: {e}, using fallback")
        return _FALLBACK_FIX_PROMPT


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
    obs = get_observability()
    trace = obs.create_trace(
        name="generate_node",
        input={"prompt": state.get("prompt", "")[:200]},
    )

    settings = get_settings()

    analysis = state.get("codebase_analysis", {})
    user_responses = state.get("user_responses", {})
    prompt = state.get("prompt", "")
    memories = state.get("relevant_memories", [])

    # Get enrichment data from enrich node
    vault_suggestions = state.get("vault_suggestions", {})
    fabio_validation = state.get("fabio_validation", {})
    nomad_info = state.get("nomad_info", {})
    env_var_configs = state.get("env_var_configs", [])

    # Extract confirmed env configs from user responses if available
    confirmed_env_configs = _extract_confirmed_env_configs(user_responses, env_var_configs)

    # Extract network mode from user responses (defaults to bridge)
    network_mode = _extract_network_mode(user_responses)

    # Build context for LLM with enrichment data
    with obs.span("build_generation_context", trace=trace) as span:
        context = _build_generation_context(
            analysis,
            user_responses,
            prompt,
            memories,
            vault_suggestions=vault_suggestions,
            fabio_validation=fabio_validation,
            nomad_info=nomad_info,
            env_var_configs=confirmed_env_configs,
        )
        span.end(output={"context_length": len(context)})

    # Get the generation prompt
    generation_prompt = _get_generation_prompt()

    # Query LLM for configuration
    with obs.span("llm_generate", trace=trace, input={"prompt_length": len(generation_prompt)}) as span:
        messages = [
            SystemMessage(content=generation_prompt),
            HumanMessage(content=context),
        ]

        response = llm.invoke(messages)
        response_text = response.content if hasattr(response, "content") else str(response)
        span.end(output={"response_length": len(response_text)})

    # Parse LLM response into JobConfig
    config_dict = None
    with obs.span("parse_llm_response", trace=trace) as span:
        try:
            config_dict = _parse_llm_response(response_text)
            # Pass nomad_info, env_var_configs, and network_mode for proper configuration
            config = _build_job_config(
                config_dict,
                analysis,
                settings,
                nomad_info,
                env_var_configs=confirmed_env_configs,
                network_mode=network_mode,
            )
            span.end(output={"job_name": config.job_name, "service_type": str(config.service_type)})
        except Exception as e:
            # Fallback to basic config from analysis
            config = _build_fallback_config(analysis, prompt, settings)
            config_dict = {"error": str(e), "fallback": True}
            span.end(level="WARNING", status_message=f"Using fallback config: {str(e)}")

    # Generate HCL
    with obs.span("generate_hcl", trace=trace, input={"job_name": config.job_name}) as span:
        hcl_content = generate_hcl(config)
        span.end(output={"hcl_length": len(hcl_content)})

    # Validate HCL
    with obs.span("validate_hcl", trace=trace) as span:
        is_valid, validation_error = validate_hcl(hcl_content, settings.nomad_addr)
        span.end(output={"is_valid": is_valid, "error": validation_error})

    if trace:
        trace.end(output={
            "job_name": config.job_name,
            "hcl_valid": is_valid,
            "hcl_length": len(hcl_content),
        })

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
    vault_suggestions: dict[str, Any] | None = None,
    fabio_validation: dict[str, Any] | None = None,
    nomad_info: dict[str, Any] | None = None,
    env_var_configs: list[dict] | None = None,
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

    # Add confirmed env var configurations (multi-source)
    # These take precedence over vault_suggestions
    if env_var_configs:
        fixed_vars = [c for c in env_var_configs if c.get("source") == "fixed"]
        consul_vars = [c for c in env_var_configs if c.get("source") == "consul"]
        vault_vars = [c for c in env_var_configs if c.get("source") == "vault"]

        parts.extend([
            "",
            "## Environment Variable Configuration (User Confirmed)",
            "Use exactly these configurations for environment variables:",
        ])

        if fixed_vars:
            parts.append("\nFixed values (set directly in env block):")
            for c in fixed_vars:
                parts.append(f"  - {c['name']}: \"{c['value']}\"")

        if consul_vars:
            parts.append("\nConsul KV paths (use template to read from Consul):")
            for c in consul_vars:
                parts.append(f"  - {c['name']}: {c['value']}")

        if vault_vars:
            parts.append("\nVault secrets (use Vault integration):")
            for c in vault_vars:
                parts.append(f"  - {c['name']}: {c['value']}")

    # Fallback: Add Vault suggestions if no confirmed configs
    elif vault_suggestions and vault_suggestions.get("suggestions"):
        parts.extend([
            "",
            "## Vault Secret Path Suggestions",
            "Use these validated Vault paths for secrets:",
        ])
        for s in vault_suggestions["suggestions"]:
            parts.append(f"- {s['env_var']}: {s['vault_reference']}")

    # Add Fabio routing info if available
    if fabio_validation:
        hostname = fabio_validation.get("suggested_hostname")
        if hostname:
            parts.extend([
                "",
                "## Fabio Routing",
                f"Suggested hostname: {hostname}",
                f"Fabio tag: {fabio_validation.get('fabio_tag', f'urlprefix-{hostname}:9999/')}",
            ])

    # Add Nomad version info
    if nomad_info:
        vault_format = nomad_info.get("recommended_vault_format", "template")
        parts.extend([
            "",
            f"## Nomad Environment",
            f"Nomad version: {nomad_info.get('version', 'unknown')}",
            f"Vault format: {vault_format} ({'use native vault env stanza' if vault_format == 'env_stanza' else 'use template blocks'})",
        ])

    # Add port env var mappings (from port analysis)
    # These are Nomad runtime variables that MUST be used
    nomad_vars = [c for c in (env_var_configs or []) if c.get("source") == "nomad"]
    if nomad_vars:
        parts.extend([
            "",
            "## Port Environment Variables (Nomad Runtime)",
            "IMPORTANT: These env vars MUST use Nomad dynamic port variables:",
        ])
        for c in nomad_vars:
            parts.append(f"  - {c['name']} = \"{c['value']}\"")
        parts.append("Do NOT use fixed values for these - use the ${NOMAD_PORT_*} references.")

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


def _extract_confirmed_env_configs(
    user_responses: dict[str, Any],
    default_configs: list[dict],
) -> list[dict]:
    """Extract confirmed env var configs from user responses.

    User responses may contain structured env_configs responses from
    the interactive collection flow.

    Args:
        user_responses: Dict of user responses to questions.
        default_configs: Default configs from enrich node to use if no user input.

    Returns:
        List of confirmed env var config dicts.
    """
    # Look for env_configs response in user_responses
    for key, value in user_responses.items():
        if isinstance(value, dict) and value.get("type") == "env_configs":
            return value.get("configs", [])

    # Fallback to default configs from enrich node
    return default_configs


def _extract_network_mode(user_responses: dict[str, Any]) -> NetworkMode:
    """Extract network mode choice from user responses.

    Looks for a network_mode response and returns the appropriate NetworkMode.
    Defaults to BRIDGE if not specified.

    Args:
        user_responses: Dict of user responses to questions.

    Returns:
        NetworkMode enum value.
    """
    for key, value in user_responses.items():
        if isinstance(value, dict) and value.get("type") == "network_mode":
            choice = value.get("value", "bridge")
            if choice == "host":
                return NetworkMode.HOST
            return NetworkMode.BRIDGE
        # Also check for simple string response
        if key == "network_mode" or (isinstance(value, str) and value in ("host", "bridge")):
            if value == "host":
                return NetworkMode.HOST
            return NetworkMode.BRIDGE

    return NetworkMode.BRIDGE  # Default


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
    nomad_info: dict[str, Any] | None = None,
    env_var_configs: list[dict] | None = None,
    network_mode: NetworkMode = NetworkMode.BRIDGE,
) -> JobConfig:
    """Build JobConfig from LLM response and analysis.

    Args:
        config_dict: Parsed LLM response.
        analysis: Codebase analysis.
        settings: App settings.
        nomad_info: Nomad version info.
        env_var_configs: Confirmed multi-source env var configurations.
        network_mode: Network mode (bridge or host) from user response.
    """
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

    # Build environment variables from confirmed configs (multi-source)
    env_vars = {}
    consul_vars = {}
    vault_secrets = {}

    if env_var_configs:
        for cfg in env_var_configs:
            name = cfg.get("name")
            source = cfg.get("source")
            value = cfg.get("value", "")

            if source == "fixed":
                env_vars[name] = value
            elif source == "consul":
                consul_vars[name] = value
            elif source == "vault":
                vault_secrets[name] = value
            elif source == "nomad":
                # Nomad runtime variables (e.g., ${NOMAD_PORT_http})
                # These go directly into env_vars
                env_vars[name] = value
    else:
        # Fallback to LLM-provided values
        env_vars = config_dict.get("env_vars", {})
        vault_secrets = config_dict.get("vault_secrets", {})

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

    # Build Vault config if we have vault secrets or policies
    vault = None
    all_vault_secrets = {**vault_secrets, **config_dict.get("vault_secrets", {})}
    if config_dict.get("vault_policies") or all_vault_secrets:
        # Check if Nomad supports native vault env stanza
        use_native_env = False
        if nomad_info:
            use_native_env = nomad_info.get("supports_native_vault_env", False)

        vault = VaultConfig(
            policies=config_dict.get("vault_policies", []),
            secrets=all_vault_secrets,
            use_native_env=use_native_env,
        )

    # Build config with multi-source env vars
    return JobConfig(
        job_name=job_name,
        datacenters=[settings.nomad_datacenter],
        namespace=settings.nomad_namespace,
        region=settings.nomad_region,
        image=image,
        ports=ports,
        env_vars=env_vars,
        consul_vars=consul_vars,
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
        network_mode=network_mode,
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
    obs = get_observability()
    iteration = state.get("iteration_count", 0) + 1
    trace = obs.create_trace(
        name="fix_node",
        input={
            "iteration": iteration,
            "error": state.get("deployment_error", "Unknown error")[:200],
        },
    )

    error = state.get("deployment_error", "Unknown error")
    current_spec = state.get("job_spec", "")
    memories = state.get("relevant_memories", [])

    with obs.span("build_fix_prompt", trace=trace) as span:
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
        span.end(output={"prompt_length": len(fix_prompt), "memories_count": len(memories)})

    # Get the fix system prompt
    fix_system_prompt = _get_fix_prompt()

    with obs.span("llm_fix", trace=trace) as span:
        messages = [
            SystemMessage(content=fix_system_prompt),
            HumanMessage(content=fix_prompt),
        ]

        response = llm.invoke(messages)
        response_text = response.content if hasattr(response, "content") else str(response)
        span.end(output={"response_length": len(response_text)})

    settings = get_settings()

    with obs.span("parse_and_build_config", trace=trace) as span:
        try:
            config_dict = _parse_llm_response(response_text)
            config = _build_job_config(config_dict, state.get("codebase_analysis", {}), settings)
            span.end(output={"job_name": config.job_name, "fix_method": "llm"})
        except Exception as e:
            # If fix fails, try incrementing resources
            current_config = state.get("job_config", {})
            config = _increment_resources(current_config, error, settings)
            span.end(level="WARNING", status_message=f"Using resource increment: {str(e)}", output={"fix_method": "increment"})

    with obs.span("generate_and_validate_hcl", trace=trace) as span:
        hcl_content = generate_hcl(config)
        is_valid, validation_error = validate_hcl(hcl_content, settings.nomad_addr)
        span.end(output={"is_valid": is_valid, "hcl_length": len(hcl_content)})

    if trace:
        trace.end(output={
            "job_name": config.job_name,
            "iteration": iteration,
            "hcl_valid": is_valid,
        })

    return {
        **state,
        "job_spec": hcl_content,
        "job_config": config.__dict__ if hasattr(config, "__dict__") else {},
        "job_name": config.job_name,
        "hcl_valid": is_valid,
        "validation_error": validation_error,
        "iteration_count": iteration,
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
