"""Port detection node for the LangGraph workflow.

This node analyzes Dockerfile and application code to determine the single
port/env var the application listens on. This is critical for Nomad job specs
where configurable ports should use ${NOMAD_PORT_<name>}.

Unlike the previous broad approach, this uses a focused LLM query to ask
specifically: "What port/env var is this app listening on?"
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
_FALLBACK_PROMPT = """You are analyzing a Dockerfile and application code to determine what port the service listens on.

## Your Task

In the service defined in the Dockerfile, determine what port or environment variable holding the port this app is listening to.

**Important:**
- Do NOT return port numbers used only in tests
- If the app is configured to take its port from an environment variable, only report that environment variable
- Focus on the main service entrypoint, not auxiliary services

## What to Look For

1. **Environment variable port** (configurable - best for Nomad):
   - Python: `port = int(os.getenv('APP_PORT', 8000))`
   - Node.js: `const PORT = process.env.PORT || 3000`
   - Go: `port := os.Getenv("PORT")`
   - Shell/CMD: `uvicorn app:app --port $PORT`

2. **Hardcoded port** (fixed in code):
   - `app.listen(8080)`
   - `EXPOSE 8000` in Dockerfile with no env var override

## Output Format

Respond with ONLY a JSON object (no markdown, no explanation):

```json
{
    "type": "env_var",
    "value": "APP_PORT",
    "default_port": 8000,
    "evidence": "src/main.py:15 - port = int(os.getenv('APP_PORT', 8000))"
}
```

Where:
- `type`: One of "env_var", "hardcoded", or "unknown"
- `value`: The environment variable name (if type=env_var) or port number (if type=hardcoded)
- `default_port`: The default port value (required for env_var, same as value for hardcoded)
- `evidence`: File and line reference showing where this was determined
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


def _get_entrypoint_code(codebase_path: str, dockerfile_content: str) -> dict[str, str]:
    """Get the main entrypoint code file based on Dockerfile CMD/ENTRYPOINT.

    Focuses on the single file that contains the app's entry point,
    not a broad scan of all source files.
    """
    path = Path(codebase_path)
    files = {}

    # Extract entrypoint from Dockerfile
    entrypoint_file = None

    # Look for CMD or ENTRYPOINT in Dockerfile
    cmd_match = re.search(
        r'(?:CMD|ENTRYPOINT)\s*\[?\s*["\']?(?:python|python3|node|go run)?\s*["\']?\s*,?\s*["\']?([^"\';\]\s]+)',
        dockerfile_content,
        re.IGNORECASE,
    )
    if cmd_match:
        entrypoint_file = cmd_match.group(1)
        # Clean up common patterns
        entrypoint_file = entrypoint_file.replace("-m", "").strip()
        if entrypoint_file.startswith("src."):
            entrypoint_file = entrypoint_file.replace(".", "/") + ".py"

    # Common entrypoint files to check
    candidates = [
        entrypoint_file,
        "main.py",
        "app.py",
        "server.py",
        "src/main.py",
        "src/app.py",
        "app/main.py",
        "index.js",
        "server.js",
        "src/index.js",
        "main.go",
        "cmd/main.go",
    ]

    for candidate in candidates:
        if candidate:
            file_path = path / candidate
            if file_path.is_file():
                content = _read_file_safely(str(file_path), max_chars=8000)
                if content:
                    files[candidate] = content
                    # Only need one main file
                    if len(files) >= 2:
                        break

    return files


def analyze_ports_node(
    state: dict[str, Any],
    llm: BaseChatModel,
) -> dict[str, Any]:
    """Analyze what port/env var the application listens on.

    Uses a focused LLM query to determine THE listening port, not all port vars.

    Args:
        state: Current graph state containing:
            - codebase_path: Path to the codebase
            - discovered_sources: Contains 'dockerfile' path
        llm: LLM instance for analysis (required).

    Returns:
        Updated state with 'port_analysis' field containing:
        - app_listening_port: {type, value, default_port, evidence}
        - supports_dynamic_port: True if type == "env_var"
        - recommended_env_mapping: {ENV_VAR: "${NOMAD_PORT_http}"} if applicable

    Raises:
        ValueError: If no LLM is provided.
    """
    if llm is None:
        raise ValueError(
            "LLM is required for port analysis. "
            "Please configure an LLM provider before running the workflow."
        )

    obs = get_observability()

    codebase_path = state.get("codebase_path", "")
    discovered_sources = state.get("discovered_sources", {})

    if not codebase_path:
        raise ValueError(
            "No codebase_path provided. Cannot analyze ports without a codebase."
        )

    # Read Dockerfile - get from selected_dockerfile (not discovered_sources)
    dockerfile_content = ""
    dockerfile_path = state.get("selected_dockerfile")

    # Convert relative path to absolute if needed
    if dockerfile_path and not dockerfile_path.startswith("/"):
        dockerfile_path = str(Path(codebase_path) / dockerfile_path)

    if dockerfile_path:
        dockerfile_content = _read_file_safely(dockerfile_path, max_chars=5000) or ""

    # Get entrypoint code (focused, not broad scan)
    entrypoint_code = _get_entrypoint_code(codebase_path, dockerfile_content)

    # Build context for LLM
    context_parts = [
        "## Dockerfile",
        f"```dockerfile\n{dockerfile_content or 'Not found'}\n```",
        "",
        "## Application Entrypoint Code",
    ]

    for file_path, content in entrypoint_code.items():
        context_parts.append(f"### {file_path}")
        context_parts.append(f"```\n{content}\n```")

    context = "\n".join(context_parts)

    # Query LLM with focused prompt (LLM call auto-traced by callback handler)
    with obs.span("llm_detect_port") as span:
        prompt = _get_prompt()
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=f"Analyze the listening port for this application:\n\n{context}"),
        ]

        response = llm.invoke(messages)
        response_text = response.content if hasattr(response, "content") else str(response)
        span.end(output={"response_length": len(response_text)})

    # Parse LLM response
    port_info = {
        "type": "unknown",
        "value": None,
        "default_port": None,
        "evidence": None,
    }

    json_str = _extract_json(response_text)
    if json_str:
        try:
            parsed = json.loads(json_str)
            port_info = {
                "type": parsed.get("type", "unknown"),
                "value": parsed.get("value"),
                "default_port": parsed.get("default_port"),
                "evidence": parsed.get("evidence"),
            }
        except json.JSONDecodeError:
            logger.warning("Could not parse JSON from LLM response")

    # Determine if app supports dynamic port allocation
    supports_dynamic_port = port_info["type"] == "env_var"

    # Generate env mapping only for the detected port env var
    recommended_env_mapping = {}
    if supports_dynamic_port and port_info["value"]:
        recommended_env_mapping[port_info["value"]] = "${NOMAD_PORT_http}"

    # Build final analysis
    final_analysis = {
        "app_listening_port": port_info,
        "supports_dynamic_port": supports_dynamic_port,
        "recommended_env_mapping": recommended_env_mapping,
    }

    logger.info(
        f"Port analysis: type={port_info['type']}, "
        f"value={port_info['value']}, "
        f"supports_dynamic={supports_dynamic_port}"
    )

    return {
        "port_analysis": final_analysis,
    }


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
