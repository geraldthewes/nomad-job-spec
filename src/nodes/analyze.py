"""Codebase analysis node for the LangGraph workflow."""

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

from src.observability import get_observability
from src.tools.codebase import (
    CodebaseAnalysis,
    analyze_codebase as analyze_codebase_tool,
    get_relevant_files_content,
)

logger = logging.getLogger(__name__)


# Fallback prompt used when LangFuse and local file are unavailable
_FALLBACK_ANALYSIS_PROMPT = """You are an expert DevOps engineer analyzing a codebase for deployment to HashiCorp Nomad.

Your task is to analyze the provided codebase information and extract deployment-relevant details.

## Cluster Environment

This Nomad cluster has the following characteristics:
- Mixed architecture: AMD64 (gpu001-gpu007) and ARM64 (cluster00-01) nodes
- CSI volumes via Ceph for persistent storage
- Fabio load balancer on port 9999 for external routing
- Vault integration for secrets management
- All jobs use Terraform templating with ${datacenter} variable

## Service Categories

Classify the application into one of these categories:
- LIGHT: Infrastructure services (100-200 CPU, 64-128 MB) - proxies, sidecars
- MEDIUM: Standard applications (500-1000 CPU, 512-1024 MB) - web apps, APIs
- HEAVY: Databases and stateful services (1000-2000 CPU, 2048-4096 MB) - PostgreSQL, Redis
- COMPUTE: ML/compute workloads (4000-8000 CPU, 8192-16384 MB) - model inference

## Focus Areas

1. Docker image requirements (base image, build requirements)
2. Port mappings needed (static vs dynamic, host mode requirements)
3. Environment variables required (especially secrets needing Vault)
4. Resource requirements based on service category
5. Health check endpoints
6. Storage requirements (does it need persistent CSI volumes?)
7. Architecture requirements (does it need AMD64 for compatibility?)
8. External routing (does it need Fabio routing?)

Output your analysis as a JSON object with the following structure:
{
    "summary": "Brief description of the application",
    "service_type": "LIGHT|MEDIUM|HEAVY|COMPUTE",
    "docker_image": "Recommended Docker image or build instructions",
    "ports": [{"name": "port_name", "container_port": port, "static": false}],
    "env_vars": {"VAR_NAME": "description or default value"},
    "secrets": ["List of env vars that should come from Vault"],
    "resources": {"cpu": cpu_mhz, "memory": memory_mb},
    "health_check": {"type": "http|tcp", "path": "/health", "port": "port_name"},
    "requires_amd64": true,
    "requires_storage": false,
    "storage_path": "/data",
    "storage_owner_uid": null,
    "external_routing": {"hostname": "app.example.com", "path": null},
    "dependencies": ["list of external dependencies"],
    "warnings": ["Any potential issues or concerns"]
}

Important cluster-specific notes:
- Most database images (PostgreSQL, MySQL, MongoDB) require AMD64
- Services needing ports < 1024 require host network mode
- Stateful services need CSI volumes with init tasks for permissions
- Use dynamic ports (not static) unless absolutely necessary
"""


def _get_analysis_prompt() -> str:
    """Get the analysis system prompt.

    Tries LangFuse first, falls back to local file, then hardcoded default.

    Returns:
        The analysis system prompt string.
    """
    try:
        from src.prompts import get_prompt_manager, PromptNotFoundError

        manager = get_prompt_manager()
        return manager.get_prompt_text("analysis")
    except PromptNotFoundError:
        logger.warning("Analysis prompt not found, using fallback")
        return _FALLBACK_ANALYSIS_PROMPT
    except Exception as e:
        logger.warning(f"Failed to get analysis prompt: {e}, using fallback")
        return _FALLBACK_ANALYSIS_PROMPT


def analyze_codebase_node(
    state: dict[str, Any],
    llm: BaseChatModel | None = None,
) -> dict[str, Any]:
    """Analyze a codebase and extract deployment information.

    This node performs two types of analysis:
    1. Static analysis using file parsing (Dockerfile, package.json, etc.)
    2. LLM-based analysis for deeper understanding

    Args:
        state: Current graph state containing 'codebase_path'.
        llm: Optional LLM instance. If not provided, only static analysis is performed.

    Returns:
        Updated state with 'codebase_analysis' field.
    """
    obs = get_observability()
    trace = obs.create_trace(
        name="analyze_node",
        input={"codebase_path": state.get("codebase_path")},
    )

    codebase_path = state.get("codebase_path")
    if not codebase_path:
        if trace:
            trace.update(level="ERROR", status_message="No codebase path provided")
        return {
            **state,
            "codebase_analysis": {"error": "No codebase path provided"},
        }

    # Step 1: Static analysis - pass span so tool creates child spans under it
    selected_dockerfile = state.get("selected_dockerfile")
    with obs.span("static_analysis", trace=trace, input={"path": codebase_path, "selected_dockerfile": selected_dockerfile}) as span:
        try:
            static_analysis: CodebaseAnalysis = analyze_codebase_tool(
                codebase_path,
                selected_dockerfile=selected_dockerfile,
                parent_span=span,
            )
            span.end(output={
                "dockerfiles_found": len(static_analysis.dockerfiles_found),
                "selected_dockerfile": selected_dockerfile,
                "language": static_analysis.dependencies.language if static_analysis.dependencies else None,
                "env_vars_count": len(static_analysis.env_vars_required),
            })
        except Exception as e:
            span.end(level="ERROR", status_message=str(e))
            if trace:
                trace.end(level="ERROR", status_message=f"Static analysis failed: {str(e)}")
            return {
                **state,
                "codebase_analysis": {"error": f"Static analysis failed: {str(e)}"},
            }

    # Step 2: LLM-enhanced analysis (if LLM provided)
    llm_analysis = None
    if llm:
        with obs.span("llm_analysis", trace=trace) as span:
            try:
                llm_analysis = _perform_llm_analysis(codebase_path, static_analysis, llm)
                span.end(output={
                    "has_summary": "summary" in llm_analysis if llm_analysis else False,
                    "has_resources": "resources" in llm_analysis if llm_analysis else False,
                })
            except Exception as e:
                # LLM analysis is optional - continue with static analysis only
                static_analysis.errors.append(f"LLM analysis failed: {str(e)}")
                span.end(level="WARNING", status_message=str(e))

    # Merge analyses
    with obs.span("merge_analyses", trace=trace) as span:
        final_analysis = _merge_analyses(static_analysis, llm_analysis)
        span.end(output={"has_llm_analysis": llm_analysis is not None})

    if trace:
        trace.end(output={
            "dockerfiles": final_analysis.get("dockerfiles_found", []),
            "language": final_analysis.get("dependencies", {}).get("language") if final_analysis.get("dependencies") else None,
            "env_vars_count": len(final_analysis.get("env_vars_required", [])),
            "errors_count": len(final_analysis.get("errors", [])),
        })

    return {
        **state,
        "codebase_analysis": final_analysis,
    }


def _perform_llm_analysis(
    codebase_path: str,
    static_analysis: CodebaseAnalysis,
    llm: BaseChatModel,
) -> dict[str, Any]:
    """Perform LLM-based analysis of the codebase.

    Args:
        codebase_path: Path to the codebase.
        static_analysis: Results from static analysis.
        llm: LLM instance for analysis.

    Returns:
        Dictionary with LLM analysis results.
    """
    # Get relevant file contents
    files_content = get_relevant_files_content(codebase_path)

    # Build context for LLM
    context_parts = [
        "## Static Analysis Results",
        f"```json\n{static_analysis.to_json()}\n```",
        "",
        "## Relevant Files",
    ]

    for filename, content in files_content.items():
        context_parts.append(f"### {filename}")
        context_parts.append(f"```\n{content[:5000]}\n```")  # Limit each file
        context_parts.append("")

    context = "\n".join(context_parts)

    # Get the analysis prompt
    analysis_prompt = _get_analysis_prompt()

    # Query LLM
    messages = [
        SystemMessage(content=analysis_prompt),
        HumanMessage(content=f"Analyze this codebase for Nomad deployment:\n\n{context}"),
    ]

    response = llm.invoke(messages)
    response_text = response.content if hasattr(response, "content") else str(response)

    # Parse JSON response
    try:
        # Try to extract JSON from response
        json_match = _extract_json(response_text)
        if json_match:
            return json.loads(json_match)
    except json.JSONDecodeError:
        pass

    # Return raw response if JSON parsing fails
    return {"raw_analysis": response_text}


def _extract_json(text: str) -> str | None:
    """Extract JSON object from text that may contain other content."""
    import re

    # Try to find JSON block in markdown
    json_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_block:
        return json_block.group(1)

    # Try to find raw JSON object
    json_obj = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if json_obj:
        return json_obj.group(0)

    return None


def _merge_analyses(
    static: CodebaseAnalysis,
    llm: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge static and LLM analyses into a unified result.

    Args:
        static: Static analysis results.
        llm: LLM analysis results (may be None).

    Returns:
        Merged analysis dictionary.
    """
    result = static.to_dict()

    if llm:
        # Add LLM insights
        result["llm_analysis"] = llm

        # Override resources if LLM provides better estimates
        if "resources" in llm:
            llm_resources = llm["resources"]
            if "cpu" in llm_resources:
                result["suggested_resources"]["cpu"] = max(
                    result["suggested_resources"].get("cpu", 500),
                    llm_resources["cpu"],
                )
            if "memory" in llm_resources:
                result["suggested_resources"]["memory"] = max(
                    result["suggested_resources"].get("memory", 256),
                    llm_resources["memory"],
                )

        # Add LLM-detected env vars
        if "env_vars" in llm:
            llm_env_vars = list(llm["env_vars"].keys())
            existing_env_vars = set(result.get("env_vars_required", []))
            result["env_vars_required"] = sorted(existing_env_vars | set(llm_env_vars))

        # Add health check info
        if "health_check" in llm:
            result["health_check"] = llm["health_check"]

        # Add warnings
        if "warnings" in llm:
            result["warnings"] = llm["warnings"]

        # Add summary
        if "summary" in llm:
            result["summary"] = llm["summary"]

    return result


def create_analyze_node(llm: BaseChatModel | None = None):
    """Create an analyze node function with the given LLM.

    This is a factory function that creates a node function with the LLM bound.

    Args:
        llm: LLM instance for analysis.

    Returns:
        Node function for use in LangGraph.
    """

    def node(state: dict[str, Any]) -> dict[str, Any]:
        return analyze_codebase_node(state, llm)

    return node
