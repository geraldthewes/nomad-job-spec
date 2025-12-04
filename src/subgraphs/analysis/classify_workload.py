"""Workload classification node for the analysis subgraph.

This node analyzes the Dockerfile CMD/ENTRYPOINT to determine whether
the application is a long-running service or a batch job. This affects
downstream spec generation (job type, service registration, health checks).
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
_FALLBACK_PROMPT = """You are analyzing a Dockerfile to determine if the application is a long-running service or a batch job.

## Nomad Job Types

**service**: Long-running processes that should be restarted if they exit.
Examples:
- Web servers: uvicorn, gunicorn, nginx, apache
- API servers: flask run, node server.js, go run main.go (with http.ListenAndServe)
- Daemons: redis-server, postgres, mongod
- Message consumers: celery worker, kafka consumer (continuous polling)

**batch**: Run-to-completion tasks that exit when done.
Examples:
- Scripts: python script.py, node migrate.js
- Data migrations: alembic upgrade, django migrate
- One-time tasks: backup scripts, report generators
- ETL jobs: data processing pipelines

## Analysis Focus

Look at the CMD or ENTRYPOINT in the Dockerfile:
1. Does it start a server/daemon that listens continuously? -> service
2. Does it run a script that processes data and exits? -> batch
3. Does it have a loop or wait indefinitely? -> service
4. Does it perform a task and terminate? -> batch

## Output Format

Respond with ONLY a JSON object (no markdown, no explanation):

```json
{
    "workload_type": "service",
    "confidence": "high",
    "evidence": "CMD uses uvicorn to run a FastAPI web server that listens on port 8000"
}
```

Where:
- `workload_type`: One of "service" or "batch"
- `confidence`: One of "high", "medium", or "low"
  - high: Clear indicators (uvicorn, gunicorn, redis-server, or explicit script execution)
  - medium: Some indicators but not definitive
  - low: Ambiguous or cannot determine
- `evidence`: Brief explanation of why you classified it this way
"""


def _get_prompt() -> str:
    """Get the classify_workload prompt.

    Tries LangFuse first, falls back to local file, then hardcoded default.
    """
    try:
        from src.prompts import get_prompt_manager

        manager = get_prompt_manager()
        return manager.get_prompt_text("classify_workload")
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


def classify_workload_node(
    state: dict[str, Any],
    llm: BaseChatModel,
) -> dict[str, Any]:
    """Classify the workload type based on Dockerfile analysis.

    Determines whether the application is a long-running service or a batch job
    by analyzing the CMD/ENTRYPOINT in the Dockerfile.

    Args:
        state: Current graph state containing:
            - codebase_path: Path to the codebase
            - selected_dockerfile: Path to the selected Dockerfile
        llm: LLM instance for analysis (required).

    Returns:
        Updated state with 'workload_classification' field containing:
        - workload_type: "service" or "batch"
        - confidence: "high", "medium", or "low"
        - evidence: Explanation of the classification

    Raises:
        ValueError: If no LLM is provided.
    """
    if llm is None:
        raise ValueError(
            "LLM is required for workload classification. "
            "Please configure an LLM provider before running the workflow."
        )

    obs = get_observability()

    codebase_path = state.get("codebase_path", "")
    dockerfile_path = state.get("selected_dockerfile")

    # Default classification (conservative fallback)
    default_classification = {
        "workload_type": "service",
        "confidence": "low",
        "evidence": "No Dockerfile available for analysis; defaulting to service",
    }

    # Convert relative path to absolute if needed
    if dockerfile_path and not dockerfile_path.startswith("/"):
        dockerfile_path = str(Path(codebase_path) / dockerfile_path)

    # Read Dockerfile
    dockerfile_content = None
    if dockerfile_path:
        dockerfile_content = _read_file_safely(dockerfile_path, max_chars=5000)

    if not dockerfile_content:
        logger.warning("No Dockerfile content available for workload classification")
        return {"workload_classification": default_classification}

    # Query LLM for classification
    with obs.span("llm_classify_workload") as span:
        prompt = _get_prompt()
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(
                content=f"Analyze this Dockerfile and classify the workload type:\n\n```dockerfile\n{dockerfile_content}\n```"
            ),
        ]

        response = llm.invoke(messages)
        response_text = response.content if hasattr(response, "content") else str(response)
        span.end(output={"response_length": len(response_text)})

    # Parse LLM response
    classification = default_classification.copy()

    json_str = _extract_json(response_text)
    if json_str:
        try:
            parsed = json.loads(json_str)
            workload_type = parsed.get("workload_type", "service")
            confidence = parsed.get("confidence", "low")
            evidence = parsed.get("evidence", "")

            # Validate workload_type
            if workload_type not in ("service", "batch"):
                logger.warning(f"Invalid workload_type '{workload_type}', defaulting to service")
                workload_type = "service"

            # Validate confidence
            if confidence not in ("high", "medium", "low"):
                logger.warning(f"Invalid confidence '{confidence}', defaulting to low")
                confidence = "low"

            classification = {
                "workload_type": workload_type,
                "confidence": confidence,
                "evidence": evidence,
            }
        except json.JSONDecodeError:
            logger.warning("Could not parse JSON from LLM response for workload classification")

    logger.info(
        f"Workload classification: type={classification['workload_type']}, "
        f"confidence={classification['confidence']}"
    )

    return {"workload_classification": classification}


def create_classify_workload_node(llm: BaseChatModel):
    """Create a classify_workload node function with the given LLM.

    Args:
        llm: LLM instance for analysis (required).

    Returns:
        Node function for use in LangGraph.

    Raises:
        ValueError: If no LLM is provided.
    """
    if llm is None:
        raise ValueError(
            "LLM is required for classify_workload node. "
            "Please provide an LLM instance."
        )

    def node(state: dict[str, Any]) -> dict[str, Any]:
        return classify_workload_node(state, llm)

    return node
