"""GPU detection node for the analysis subgraph.

This node analyzes the Dockerfile and build configuration to determine whether
the application requires or benefits from GPU resources. It merges information
from both sources, with explicit config taking precedence for the final decision.
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
_FALLBACK_PROMPT = """You are analyzing a Dockerfile to determine if the application requires or benefits from GPU resources.

## GPU Indicators

**High confidence GPU signals:**
- Base images: `nvidia/cuda:*`, `pytorch/pytorch:*-cuda*`, `tensorflow/tensorflow:*-gpu*`, `nvcr.io/nvidia/*`
- Package installs: `nvidia-cuda-toolkit`, `cuda-*`, `cudnn*`, `libnccl*`
- Environment variables: `NVIDIA_VISIBLE_DEVICES`, `CUDA_VISIBLE_DEVICES`, `NVIDIA_DRIVER_CAPABILITIES`
- Libraries: `torch` with CUDA, `tensorflow-gpu`, `cupy`, `rapids`

**Medium confidence GPU signals:**
- ML frameworks (`torch`, `tensorflow`, `jax`) without explicit CPU-only markers
- Base images like `pytorch/pytorch:*` or `tensorflow/tensorflow:*` without explicit GPU/CPU suffix
- References to GPU in comments or documentation within Dockerfile

**Non-GPU indicators:**
- Explicit CPU-only images: `*-cpu`, `*-nocuda`
- Standard base images: `python:*`, `node:*`, `alpine:*`, `ubuntu:*` without GPU packages
- No ML frameworks or CUDA-related packages

## Multi-stage Build Handling

For multi-stage builds, focus primarily on the FINAL stage (the one that runs in production). Build-time GPU dependencies in earlier stages don't necessarily mean runtime GPU is required.

## CUDA Version Extraction

If you detect GPU usage, try to extract the CUDA version from:
- Base image tag: `nvidia/cuda:12.1-runtime` -> "12.1"
- Package versions: `cuda-toolkit-12-1` -> "12.1"
- Environment variables: `CUDA_VERSION=11.8` -> "11.8"

## Output Format

Respond with ONLY a JSON object (no markdown, no explanation):

```json
{
    "requires_gpu": true,
    "confidence": "high",
    "evidence": "Base image nvidia/cuda:12.1-runtime indicates CUDA GPU runtime dependency",
    "cuda_version": "12.1"
}
```

Where:
- `requires_gpu`: Boolean indicating if GPU is required/beneficial
- `confidence`: One of "high", "medium", or "low"
  - high: Clear GPU indicators (nvidia/cuda base, CUDA packages, GPU env vars)
  - medium: ML frameworks present but GPU usage unclear
  - low: Ambiguous or cannot determine
- `evidence`: Brief explanation of what indicators you found
- `cuda_version`: Extracted CUDA version string if detectable, or null
"""


def _get_prompt() -> str:
    """Get the detect_gpu prompt.

    Tries LangFuse first, falls back to local file, then hardcoded default.
    """
    try:
        from src.prompts import get_prompt_manager

        manager = get_prompt_manager()
        return manager.get_prompt_text("detect_gpu")
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


def _analyze_dockerfile_with_llm(
    dockerfile_content: str,
    llm: BaseChatModel,
) -> dict[str, Any]:
    """Analyze Dockerfile content with LLM for GPU indicators.

    Args:
        dockerfile_content: The content of the Dockerfile.
        llm: LLM instance for analysis.

    Returns:
        Dictionary with dockerfile analysis results:
        - detected: bool (whether GPU was detected)
        - confidence: str ("high", "medium", "low")
        - evidence: str (explanation)
        - cuda_version: str | None
    """
    obs = get_observability()

    default_result = {
        "detected": False,
        "confidence": "low",
        "evidence": "Could not analyze Dockerfile",
        "cuda_version": None,
    }

    with obs.span("llm_detect_gpu") as span:
        prompt = _get_prompt()
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(
                content=f"Analyze this Dockerfile for GPU requirements:\n\n```dockerfile\n{dockerfile_content}\n```"
            ),
        ]

        response = llm.invoke(messages)
        response_text = response.content if hasattr(response, "content") else str(response)
        span.end(output={"response_length": len(response_text)})

    # Parse LLM response
    json_str = _extract_json(response_text)
    if json_str:
        try:
            parsed = json.loads(json_str)
            requires_gpu = parsed.get("requires_gpu", False)
            confidence = parsed.get("confidence", "low")
            evidence = parsed.get("evidence", "")
            cuda_version = parsed.get("cuda_version")

            # Validate confidence
            if confidence not in ("high", "medium", "low"):
                logger.warning(f"Invalid confidence '{confidence}', defaulting to low")
                confidence = "low"

            return {
                "detected": bool(requires_gpu),
                "confidence": confidence,
                "evidence": evidence,
                "cuda_version": cuda_version,
            }
        except json.JSONDecodeError:
            logger.warning("Could not parse JSON from LLM response for GPU detection")

    return default_result


def detect_gpu_node(
    state: dict[str, Any],
    llm: BaseChatModel,
) -> dict[str, Any]:
    """Detect GPU requirements from config and Dockerfile analysis.

    This node analyzes BOTH sources and merges results:
    1. Check merged_extraction.requires_gpu for explicit config
    2. Analyze Dockerfile for GPU indicators and cuda_version
    3. Merge: config is authoritative for boolean, Dockerfile provides supplementary data

    Args:
        state: Current graph state containing:
            - codebase_path: Path to the codebase
            - selected_dockerfile: Path to the selected Dockerfile
            - merged_extraction: Merged extraction results (may contain requires_gpu)
        llm: LLM instance for analysis (required).

    Returns:
        Updated state with 'gpu_detection' field containing:
        - requires_gpu: bool (final merged decision)
        - confidence: "high", "medium", or "low"
        - evidence: Explanation of the decision
        - config_value: bool | None (what config said, if set)
        - dockerfile_detected: bool (whether Dockerfile indicated GPU)
        - cuda_version: str | None (extracted from Dockerfile)

    Raises:
        ValueError: If no LLM is provided.
    """
    if llm is None:
        raise ValueError(
            "LLM is required for GPU detection. "
            "Please configure an LLM provider before running the workflow."
        )

    codebase_path = state.get("codebase_path", "")
    dockerfile_path = state.get("selected_dockerfile")
    merged_extraction = state.get("merged_extraction", {})

    # 1. Check config value (may be None, True, or False)
    config_value = merged_extraction.get("requires_gpu")

    # 2. Analyze Dockerfile (always, when available)
    dockerfile_result = {
        "detected": False,
        "confidence": "low",
        "evidence": "No Dockerfile available for analysis",
        "cuda_version": None,
    }

    # Convert relative path to absolute if needed
    if dockerfile_path and not dockerfile_path.startswith("/"):
        dockerfile_path = str(Path(codebase_path) / dockerfile_path)

    dockerfile_content = None
    if dockerfile_path:
        dockerfile_content = _read_file_safely(dockerfile_path, max_chars=10000)

    if dockerfile_content:
        dockerfile_result = _analyze_dockerfile_with_llm(dockerfile_content, llm)

    # 3. Merge results: config is authoritative for boolean when set
    if config_value is not None:
        requires_gpu = config_value
        confidence = "high"  # Explicit config is always high confidence
        if dockerfile_result["detected"] and not config_value:
            evidence = (
                f"Config explicitly sets requires_gpu=false, overriding Dockerfile detection. "
                f"Dockerfile analysis: {dockerfile_result['evidence']}"
            )
        elif dockerfile_result["detected"] and config_value:
            evidence = (
                f"Config confirms GPU requirement. "
                f"Dockerfile analysis: {dockerfile_result['evidence']}"
            )
        elif config_value:
            evidence = "Config explicitly sets requires_gpu=true"
        else:
            evidence = "Config explicitly sets requires_gpu=false"
    elif dockerfile_result["detected"]:
        requires_gpu = True
        confidence = dockerfile_result["confidence"]
        evidence = dockerfile_result["evidence"]
    else:
        requires_gpu = False
        confidence = dockerfile_result["confidence"] if dockerfile_content else "low"
        evidence = dockerfile_result["evidence"]

    gpu_detection = {
        "requires_gpu": requires_gpu,
        "confidence": confidence,
        "evidence": evidence,
        "config_value": config_value,
        "dockerfile_detected": dockerfile_result["detected"],
        "cuda_version": dockerfile_result["cuda_version"],
    }

    logger.info(
        f"GPU detection: requires_gpu={requires_gpu}, "
        f"confidence={confidence}, "
        f"config_value={config_value}, "
        f"dockerfile_detected={dockerfile_result['detected']}, "
        f"cuda_version={dockerfile_result['cuda_version']}"
    )

    return {"gpu_detection": gpu_detection}


def create_detect_gpu_node(llm: BaseChatModel):
    """Create a detect_gpu node function with the given LLM.

    Args:
        llm: LLM instance for analysis (required).

    Returns:
        Node function for use in LangGraph.

    Raises:
        ValueError: If no LLM is provided.
    """
    if llm is None:
        raise ValueError(
            "LLM is required for detect_gpu node. "
            "Please provide an LLM instance."
        )

    def node(state: dict[str, Any]) -> dict[str, Any]:
        return detect_gpu_node(state, llm)

    return node
