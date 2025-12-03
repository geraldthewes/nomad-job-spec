"""Port configurability analysis node for the LangGraph workflow.

This node analyzes Dockerfile and application code to determine whether
application ports are configurable via environment variables. This is critical
for Nomad job specs where configurable ports should use ${NOMAD_PORT_<name>}.

Port-configurable apps need their port env var mapped to Nomad's dynamic port.
Hardcoded ports require user warnings as they may need refactoring.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

from src.observability import get_observability

logger = logging.getLogger(__name__)


# Fallback prompt when LangFuse/local file unavailable
_FALLBACK_PROMPT = """You are analyzing application code to determine port configurability for Nomad deployment.

## Your Task

Analyze the provided Dockerfile and application code to determine:
1. Whether each exposed port is configurable via an environment variable
2. Which environment variable controls each port
3. What the default port value is

## Port Configurability Patterns

### Configurable Ports (GOOD)
The application reads the port from an environment variable:
- Python: `port = int(os.getenv('APP_PORT', 8000))`
- Node.js: `const PORT = process.env.PORT || 3000`
- Go: `port := os.Getenv("PORT")`
- Java: `System.getenv("SERVER_PORT")`
- CMD with env: `CMD ["--port", "$APP_PORT"]`
- Shell: `uvicorn app:app --port $PORT`

### Hardcoded Ports (PROBLEMATIC)
The port is fixed in the code:
- `app.listen(8080)`
- `server.bind('0.0.0.0:8080')`
- `http.ListenAndServe(":8080", nil)`

## Common Port Environment Variables
- PORT, APP_PORT, SERVER_PORT, HTTP_PORT
- API_PORT, WEB_PORT, SERVICE_PORT
- LISTEN_PORT, BIND_PORT

## Output Format

Respond with a JSON object:
```json
{
    "ports": [
        {
            "name": "http",
            "container_port": 8000,
            "is_configurable": true,
            "env_var": "APP_PORT",
            "default_value": 8000,
            "confidence": 0.95,
            "evidence": "config.py line 5: APP_PORT = int(os.getenv('APP_PORT', 8000))"
        }
    ],
    "hardcoded_warnings": [
        {
            "port": 9090,
            "location": "metrics.py:12",
            "suggestion": "Consider using METRICS_PORT environment variable"
        }
    ],
    "reasoning": "Brief explanation of the analysis"
}
```

## Important Notes

- If EXPOSE is in Dockerfile but no code reference found, mark is_configurable as null (unknown)
- Check both Dockerfile ENV declarations AND application code
- Trace env var usage from Dockerfile CMD/ENTRYPOINT through to application
- Multiple ports may have different configurability status
"""


def _get_prompt() -> str:
    """Get the analyze_ports prompt.

    Tries LangFuse first, falls back to local file, then hardcoded default.
    """
    try:
        from src.prompts import get_prompt_manager

        manager = get_prompt_manager()
        return manager.get_prompt_text("analyze_ports")
    except Exception:
        return _FALLBACK_PROMPT


def _read_file_safely(file_path: str, max_chars: int = 10000) -> str | None:
    """Read a file safely, returning None if it fails."""
    try:
        with open(file_path) as f:
            content = f.read()
        if len(content) > max_chars:
            content = content[:max_chars] + "\n... (truncated)"
        return content
    except Exception as e:
        logger.warning(f"Could not read {file_path}: {e}")
        return None


def _extract_json(text: str) -> str | None:
    """Extract JSON object from text that may contain other content."""
    # Try to find JSON block in markdown
    json_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_block:
        return json_block.group(1)

    # Try to find raw JSON object
    json_obj = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if json_obj:
        return json_obj.group(0)

    return None


def _static_port_env_detection(codebase_path: str) -> list[dict[str, Any]]:
    """Perform static regex-based detection of port environment variables.

    Scans common patterns in source files:
    - Python: os.getenv("PORT", 8000), os.environ.get("PORT")
    - Node.js: process.env.PORT || 3000
    - Go: os.Getenv("PORT")
    - Shell: $PORT, ${PORT:-8000}
    """
    results = []
    path = Path(codebase_path)

    # Patterns to match port env var usage
    patterns = [
        # Python: os.getenv("APP_PORT", 8000) or os.environ.get("PORT")
        (
            r'os\.(?:getenv|environ\.get)\s*\(\s*["\'](\w*PORT\w*)["\'](?:\s*,\s*["\']?(\d+))?',
            "python",
        ),
        # Python with int(): int(os.getenv("PORT", 8000))
        (
            r'int\s*\(\s*os\.(?:getenv|environ\.get)\s*\(\s*["\'](\w*PORT\w*)["\'](?:[^)]*,\s*["\']?(\d+))?',
            "python",
        ),
        # Node.js: process.env.PORT || 3000
        (r"process\.env\.(\w*PORT\w*)\s*\|\|\s*(\d+)", "nodejs"),
        # Node.js: process.env.PORT (without default)
        (r"process\.env\.(\w*PORT\w*)", "nodejs"),
        # Go: os.Getenv("PORT")
        (r'os\.Getenv\s*\(\s*["\'](\w*PORT\w*)["\']', "go"),
        # Shell/Dockerfile: $PORT or ${PORT:-8000}
        (r"\$\{?(\w*PORT\w*)(?::-(\d+))?\}?", "shell"),
    ]

    # Files to scan
    file_patterns = [
        "**/*.py",
        "**/*.js",
        "**/*.ts",
        "**/*.go",
        "Dockerfile*",
        "**/Dockerfile*",
        "docker-compose*.yml",
        "docker-compose*.yaml",
    ]

    found_vars: dict[str, dict[str, Any]] = {}

    for file_pattern in file_patterns:
        for file_path in path.glob(file_pattern):
            if not file_path.is_file():
                continue

            content = _read_file_safely(str(file_path), max_chars=50000)
            if not content:
                continue

            rel_path = str(file_path.relative_to(path))

            for pattern, lang in patterns:
                for match in re.finditer(pattern, content, re.IGNORECASE):
                    env_var = match.group(1)
                    default_val = match.group(2) if len(match.groups()) > 1 else None

                    # Find line number
                    line_start = content.rfind("\n", 0, match.start()) + 1
                    line_num = content[:match.start()].count("\n") + 1
                    line_content = content[line_start:content.find("\n", match.start())]

                    if env_var not in found_vars:
                        found_vars[env_var] = {
                            "env_var": env_var,
                            "default_value": int(default_val) if default_val else None,
                            "references": [],
                            "language": lang,
                        }
                    elif default_val and not found_vars[env_var]["default_value"]:
                        found_vars[env_var]["default_value"] = int(default_val)

                    found_vars[env_var]["references"].append({
                        "file": rel_path,
                        "line": line_num,
                        "snippet": line_content.strip()[:100],
                    })

    results = list(found_vars.values())
    return results


def _get_port_relevant_code(codebase_path: str, max_files: int = 10) -> dict[str, str]:
    """Get source files that likely contain port configuration.

    Prioritizes:
    - config.py, settings.py, app.py, main.py, server.py (Python)
    - server.js, app.js, index.js, config.js (Node.js)
    - main.go, server.go, config.go (Go)
    """
    path = Path(codebase_path)
    priority_files = [
        # Python
        "config.py",
        "settings.py",
        "app.py",
        "main.py",
        "server.py",
        "src/config.py",
        "src/settings.py",
        "src/main.py",
        "src/app.py",
        "app/config.py",
        "app/main.py",
        # Node.js
        "server.js",
        "app.js",
        "index.js",
        "config.js",
        "src/server.js",
        "src/index.js",
        "src/config.js",
        # Go
        "main.go",
        "server.go",
        "config.go",
        "cmd/main.go",
        "cmd/server/main.go",
    ]

    files = {}
    for filename in priority_files:
        file_path = path / filename
        if file_path.is_file() and len(files) < max_files:
            content = _read_file_safely(str(file_path), max_chars=5000)
            if content:
                files[filename] = content

    return files


def _generate_env_mappings(ports: list[dict[str, Any]]) -> dict[str, str]:
    """Generate recommended env var to NOMAD_PORT mappings.

    For configurable ports, maps their env var to ${NOMAD_PORT_<name>}.
    """
    mappings = {}
    for port in ports:
        if port.get("is_configurable") and port.get("env_var"):
            port_name = port.get("name", "http")
            mappings[port["env_var"]] = f"${{NOMAD_PORT_{port_name}}}"
    return mappings


def analyze_ports_node(
    state: dict[str, Any],
    llm: BaseChatModel,
) -> dict[str, Any]:
    """Analyze port configurability in the codebase.

    This node examines:
    1. Ports discovered by extractors (from merged_extraction)
    2. Dockerfile ENV and CMD/ENTRYPOINT
    3. Application source code for port configuration patterns

    It determines whether each port is configurable via environment variable,
    enabling proper ${NOMAD_PORT_<name>} mapping in the generated HCL.

    Args:
        state: Current graph state containing:
            - codebase_path: Path to the codebase
            - merged_extraction: Contains 'ports' from extractors
            - discovered_sources: Contains 'dockerfile' path
        llm: LLM instance for analysis (required).

    Returns:
        Updated state with 'port_analysis' field.

    Raises:
        ValueError: If no LLM is provided.
    """
    if llm is None:
        raise ValueError(
            "LLM is required for port analysis. "
            "Please configure an LLM provider before running the workflow."
        )

    obs = get_observability()
    trace = obs.create_trace(
        name="analyze_ports_node",
        input={
            "codebase_path": state.get("codebase_path"),
        },
    )

    codebase_path = state.get("codebase_path", "")
    merged_extraction = state.get("merged_extraction", {})
    discovered_sources = state.get("discovered_sources", {})

    # Get ports from extraction
    extracted_ports = merged_extraction.get("ports", [])

    if not codebase_path:
        raise ValueError(
            "No codebase_path provided. Cannot analyze ports without a codebase."
        )

    # Step 1: Static analysis (provides hints for LLM)
    static_results = []
    with obs.span("static_port_analysis", trace=trace) as span:
        try:
            static_results = _static_port_env_detection(codebase_path)
            span.end(output={"static_findings": len(static_results)})
        except Exception as e:
            logger.warning(f"Static port analysis failed: {e}")
            span.end(level="WARNING", status_message=str(e))

    # Step 2: LLM analysis (required)
    llm_result = None
    with obs.span("llm_port_analysis", trace=trace) as span:
        try:
            llm_result = _llm_port_analysis(
                codebase_path,
                extracted_ports,
                discovered_sources,
                llm,
            )
            span.end(output={"llm_ports": len(llm_result.get("ports", []))})
        except Exception as e:
            logger.error(f"LLM port analysis failed: {e}")
            span.end(level="ERROR", status_message=str(e))
            raise RuntimeError(f"Port analysis failed: {e}") from e

    # Step 3: Merge results (LLM takes precedence)
    with obs.span("merge_port_analysis", trace=trace) as span:
        final_analysis = _merge_port_analysis(
            extracted_ports,
            static_results,
            llm_result,
        )
        span.end(output={
            "configurable_ports": sum(
                1 for p in final_analysis["ports"] if p.get("is_configurable")
            ),
            "hardcoded_warnings": len(final_analysis.get("hardcoded_ports", [])),
        })

    # Step 4: Generate recommended env mappings
    recommended_mappings = _generate_env_mappings(final_analysis["ports"])
    final_analysis["recommended_env_mappings"] = recommended_mappings

    if trace:
        trace.end(output={
            "ports_analyzed": len(final_analysis["ports"]),
            "configurable": sum(
                1 for p in final_analysis["ports"] if p.get("is_configurable")
            ),
            "hardcoded_warnings": len(final_analysis.get("hardcoded_ports", [])),
            "env_mappings": list(recommended_mappings.keys()),
        })

    logger.info(
        f"Port analysis: {len(final_analysis['ports'])} ports, "
        f"{len(recommended_mappings)} env mappings"
    )

    return {
        **state,
        "port_analysis": final_analysis,
    }


def _llm_port_analysis(
    codebase_path: str,
    extracted_ports: list[dict[str, Any]],
    discovered_sources: dict[str, str],
    llm: BaseChatModel,
) -> dict[str, Any]:
    """Perform LLM-assisted port analysis."""
    # Read Dockerfile
    dockerfile_content = ""
    dockerfile_path = discovered_sources.get("dockerfile")
    if dockerfile_path:
        dockerfile_content = _read_file_safely(dockerfile_path, max_chars=5000) or ""

    # Read application code files
    app_code = _get_port_relevant_code(codebase_path)

    # Build context
    context_parts = [
        "## Dockerfile",
        f"```dockerfile\n{dockerfile_content or 'Not found'}\n```",
        "",
        "## Extracted Ports from Build Config",
        f"```json\n{json.dumps(extracted_ports, indent=2)}\n```",
        "",
        "## Application Code (port-relevant files)",
    ]

    for file_path, content in app_code.items():
        context_parts.append(f"### {file_path}")
        context_parts.append(f"```\n{content}\n```")

    context = "\n".join(context_parts)

    # Get prompt and invoke LLM
    prompt = _get_prompt()
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=f"Analyze port configurability:\n\n{context}"),
    ]

    response = llm.invoke(messages)
    response_text = response.content if hasattr(response, "content") else str(response)

    # Parse JSON response
    json_str = _extract_json(response_text)
    if json_str:
        try:
            result = json.loads(json_str)
            result["analysis_method"] = "llm"
            return result
        except json.JSONDecodeError:
            logger.warning("Could not parse JSON from LLM response")

    return {"ports": [], "hardcoded_warnings": [], "analysis_method": "llm_failed"}


def _merge_port_analysis(
    extracted_ports: list[dict[str, Any]],
    static_results: list[dict[str, Any]],
    llm_result: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge static and LLM analysis results.

    Priority: LLM > Static > Extracted defaults
    """
    # Start with extracted ports as base
    ports_by_number: dict[int, dict[str, Any]] = {}

    # Add extracted ports first (lowest priority)
    for port in extracted_ports:
        container_port = port.get("container_port")
        if container_port:
            ports_by_number[container_port] = {
                "name": port.get("name", "http"),
                "container_port": container_port,
                "is_configurable": None,  # Unknown from extraction
                "env_var": None,
                "default_value": container_port,
                "confidence": 0.3,
                "evidence": "From Dockerfile/build config",
            }

    # Add/update from static analysis
    for static in static_results:
        env_var = static.get("env_var", "")
        default_val = static.get("default_value")

        if default_val:
            if default_val in ports_by_number:
                # Update existing port
                ports_by_number[default_val]["is_configurable"] = True
                ports_by_number[default_val]["env_var"] = env_var
                ports_by_number[default_val]["confidence"] = 0.7
                if static.get("references"):
                    ref = static["references"][0]
                    ports_by_number[default_val]["evidence"] = (
                        f"{ref['file']}:{ref['line']}: {ref['snippet']}"
                    )
            else:
                # New port from static analysis
                ports_by_number[default_val] = {
                    "name": _infer_port_name(env_var, default_val),
                    "container_port": default_val,
                    "is_configurable": True,
                    "env_var": env_var,
                    "default_value": default_val,
                    "confidence": 0.7,
                    "evidence": (
                        f"{static['references'][0]['file']}:{static['references'][0]['line']}"
                        if static.get("references")
                        else "Static analysis"
                    ),
                }

    # LLM results take precedence
    hardcoded_ports = []
    if llm_result and llm_result.get("ports"):
        for llm_port in llm_result["ports"]:
            container_port = llm_port.get("container_port")
            if container_port:
                ports_by_number[container_port] = {
                    "name": llm_port.get("name", _infer_port_name("", container_port)),
                    "container_port": container_port,
                    "is_configurable": llm_port.get("is_configurable"),
                    "env_var": llm_port.get("env_var"),
                    "default_value": llm_port.get("default_value", container_port),
                    "confidence": llm_port.get("confidence", 0.9),
                    "evidence": llm_port.get("evidence", "LLM analysis"),
                }

        hardcoded_ports = llm_result.get("hardcoded_warnings", [])

    return {
        "ports": list(ports_by_number.values()),
        "hardcoded_ports": hardcoded_ports,
        "analysis_method": "llm" if llm_result else "static",
    }


def _infer_port_name(env_var: str, port: int) -> str:
    """Infer a port name from env var or port number."""
    env_lower = env_var.lower()

    if "http" in env_lower or "web" in env_lower or "app" in env_lower:
        return "http"
    if "grpc" in env_lower:
        return "grpc"
    if "metrics" in env_lower or "prometheus" in env_lower:
        return "metrics"
    if "health" in env_lower:
        return "health"

    # Infer from common ports
    common_ports = {
        80: "http",
        443: "https",
        8080: "http",
        8000: "http",
        3000: "http",
        5000: "http",
        9090: "metrics",
        9091: "metrics",
        50051: "grpc",
    }
    return common_ports.get(port, "http")


def create_analyze_ports_node(llm: BaseChatModel):
    """Create an analyze_ports node function with the given LLM.

    Args:
        llm: LLM instance for analysis (required).

    Returns:
        Node function for use in LangGraph.

    Raises:
        ValueError: If no LLM is provided.
    """
    if llm is None:
        raise ValueError(
            "LLM is required for analyze_ports node. "
            "Please provide an LLM instance."
        )

    def node(state: dict[str, Any]) -> dict[str, Any]:
        return analyze_ports_node(state, llm)

    return node
