"""Analyze build system node for the LangGraph workflow.

This node uses an LLM to understand HOW the Docker image is built by analyzing
discovered files (Makefile, docker-compose.yml, jobforge spec, etc.).

This is a focused node with one job: determine the build mechanism and
where the configuration lives.
"""

import json
import logging
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

from src.observability import get_observability

logger = logging.getLogger(__name__)


# Fallback prompt used when LangFuse and local file are unavailable
_FALLBACK_PROMPT = """You are analyzing a codebase to understand how Docker images are built.

Your task is to examine the provided files and determine:
1. What build mechanism/tool is used (jobforge, docker build, docker-compose, or other)
2. Where the build configuration file is located (the path to the actual config)
3. Which Dockerfile is being built by the build system

## Build Mechanisms

- **jobforge**: A CI/CD tool that uses `build.yaml` files. Look for `jobforge submit-job` commands in Makefiles.
- **docker**: Direct Docker builds using `docker build` commands. The Dockerfile is the config.
- **docker-compose**: Uses docker-compose.yml to define builds. Look for `docker-compose build` commands.
- **other**: Any other build mechanism not listed above.

## Analysis Strategy

1. If a Makefile exists, analyze it first - it often orchestrates the build
2. Look for commands that trigger builds (make build, docker build, jobforge, etc.)
3. Trace any file references to find the actual configuration
4. **Identify which Dockerfile is used**: Look for `-f` flags in docker build commands, or `dockerfile:` keys in build.yaml/docker-compose.yml
5. If no Makefile, check if build files exist directly (build.yaml, docker-compose.yml)

## Output Format

Respond with a JSON object:
```json
{
    "mechanism": "jobforge|docker|docker-compose|other",
    "config_path": "path/to/config/file",
    "dockerfile_used": "path/to/Dockerfile",
    "reasoning": "Brief explanation of how you determined this"
}
```

**dockerfile_used**: The Dockerfile path referenced by the build system. Examples:
- If Makefile has `docker build -f docker/Dockerfile.prod .` → `"dockerfile_used": "docker/Dockerfile.prod"`
- If build.yaml has `dockerfile: Dockerfile` → `"dockerfile_used": "Dockerfile"`
- If no explicit Dockerfile specified, assume `"dockerfile_used": "Dockerfile"` (Docker default)

If you cannot determine the build mechanism, use:
```json
{
    "mechanism": "unknown",
    "config_path": null,
    "dockerfile_used": null,
    "reasoning": "Explanation of why it couldn't be determined"
}
```
"""


def _get_prompt() -> str:
    """Get the analyze_build_system prompt.

    Tries LangFuse first, falls back to local file, then hardcoded default.

    Returns:
        The prompt string.
    """
    try:
        from src.prompts import get_prompt_manager, PromptNotFoundError

        manager = get_prompt_manager()
        return manager.get_prompt_text("analyze_build_system")
    except Exception:
        # Fall back to hardcoded prompt
        return _FALLBACK_PROMPT


def _read_file_safely(file_path: str, max_chars: int = 10000) -> str | None:
    """Read a file safely, returning None if it fails."""
    try:
        with open(file_path) as f:
            content = f.read()
        # Truncate if too long
        if len(content) > max_chars:
            content = content[:max_chars] + "\n... (truncated)"
        return content
    except Exception as e:
        logger.warning(f"Could not read {file_path}: {e}")
        return None


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


def analyze_build_system_node(
    state: dict[str, Any],
    llm: BaseChatModel | None = None,
) -> dict[str, Any]:
    """Analyze discovered files to understand the build mechanism.

    This node reads all discovered source files and uses an LLM to determine:
    - What tool is used to build the Docker image
    - Where the build configuration is located

    Args:
        state: Current graph state containing 'discovered_sources' and 'codebase_path'.
        llm: LLM instance for analysis.

    Returns:
        Updated state with 'build_system_analysis' field.
    """
    obs = get_observability()

    discovered_sources = state.get("discovered_sources", {})
    codebase_path = state.get("codebase_path", "")

    # Default result for when analysis can't be performed
    default_result = {
        "mechanism": "unknown",
        "config_path": None,
        "dockerfile_used": None,
        "reasoning": "Could not analyze build system",
        "fallback_used": True,
    }

    if not discovered_sources:
        logger.info("No sources discovered, skipping build system analysis")
        return {
            "build_system_analysis": {
                **default_result,
                "reasoning": "No source files discovered",
            },
        }

    if not llm:
        logger.warning("No LLM provided, skipping build system analysis")
        return {
            "build_system_analysis": {
                **default_result,
                "reasoning": "No LLM available for analysis",
            },
        }

    # Step 1: Read content of all discovered files
    files_content = {}
    with obs.span("read_discovered_files") as span:
        for source_type, file_path in discovered_sources.items():
            content = _read_file_safely(file_path)
            if content:
                # Use relative path for display
                try:
                    rel_path = str(Path(file_path).relative_to(codebase_path))
                except ValueError:
                    rel_path = file_path
                files_content[rel_path] = {
                    "type": source_type,
                    "content": content,
                }
        span.end(output={"files_read": list(files_content.keys())})

    if not files_content:
        return {
            "build_system_analysis": {
                **default_result,
                "reasoning": "Could not read any discovered files",
            },
        }

    # Step 2: Build context for LLM
    context_parts = ["## Files found in the repository:\n"]
    for file_path, info in files_content.items():
        context_parts.append(f"### {file_path} (type: {info['type']})")
        context_parts.append(f"```\n{info['content']}\n```\n")

    context = "\n".join(context_parts)

    # Step 3: Call LLM (auto-traced by callback handler)
    with obs.span("llm_analysis") as span:
        try:
            prompt = _get_prompt()
            messages = [
                SystemMessage(content=prompt),
                HumanMessage(content=f"Analyze how Docker images are built in this repository:\n\n{context}"),
            ]

            response = llm.invoke(messages)
            response_text = response.content if hasattr(response, "content") else str(response)

            # Parse JSON response
            json_str = _extract_json(response_text)
            if json_str:
                result = json.loads(json_str)
                # Ensure required fields
                result.setdefault("mechanism", "unknown")
                result.setdefault("config_path", None)
                result.setdefault("dockerfile_used", None)
                result.setdefault("reasoning", "")
                result["fallback_used"] = False

                # Resolve relative config_path to absolute
                if result["config_path"] and not Path(result["config_path"]).is_absolute():
                    result["config_path"] = str(Path(codebase_path) / result["config_path"])

                # Resolve relative dockerfile_used to absolute
                if result["dockerfile_used"] and not Path(result["dockerfile_used"]).is_absolute():
                    result["dockerfile_used"] = str(Path(codebase_path) / result["dockerfile_used"])

                span.end(output={
                    "mechanism": result["mechanism"],
                    "config_path": result["config_path"],
                    "dockerfile_used": result["dockerfile_used"],
                })
            else:
                logger.warning("Could not parse JSON from LLM response")
                result = {
                    **default_result,
                    "reasoning": f"Could not parse LLM response: {response_text[:200]}",
                }
                span.end(level="WARNING", status_message="JSON parse failed")

        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            result = {
                **default_result,
                "reasoning": f"LLM analysis failed: {str(e)}",
            }
            span.end(level="ERROR", status_message=str(e))

    logger.info(
        f"Build system analysis: mechanism={result.get('mechanism')}, "
        f"config_path={result.get('config_path')}, "
        f"dockerfile_used={result.get('dockerfile_used')}"
    )

    return {
        "build_system_analysis": result,
    }


def create_analyze_build_system_node(llm: BaseChatModel | None = None):
    """Create an analyze_build_system node function with the given LLM.

    Args:
        llm: LLM instance for analysis.

    Returns:
        Node function for use in LangGraph.
    """

    def node(state: dict[str, Any]) -> dict[str, Any]:
        return analyze_build_system_node(state, llm)

    return node
