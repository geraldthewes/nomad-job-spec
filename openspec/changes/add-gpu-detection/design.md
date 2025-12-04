# Design: GPU Detection Node

## Overview

This document captures architectural decisions for the GPU detection capability in the analysis subgraph.

## Architecture Decision: LLM-Based vs Rule-Based Detection

### Options Considered

1. **Rule-based detection**: Regex patterns matching known GPU indicators
2. **LLM-based detection**: Prompt-based analysis similar to `classify_workload`
3. **Hybrid**: Rules for high-confidence cases, LLM for ambiguous cases

### Decision: LLM-Based Detection

Rationale:
- Consistent with existing `classify_workload` pattern
- Handles edge cases (custom base images, multi-stage builds, conditional installs)
- Easier to improve via prompt tuning without code changes
- LangFuse prompt management already in place

Trade-offs:
- Slightly slower than pure regex (acceptable given other LLM calls in pipeline)
- Requires LLM availability (same constraint as other analysis nodes)

## State Design

### New Field: `gpu_detection`

```python
gpu_detection: dict[str, Any]
# Contains:
# - requires_gpu: bool           # Final merged decision
# - confidence: "high" | "medium" | "low"
# - evidence: str                # Explanation of decision
# - config_value: bool | None    # What config said (if set)
# - dockerfile_detected: bool    # What Dockerfile analysis found
# - cuda_version: str | None     # Detected CUDA version from Dockerfile
```

### Input Sources

The node reads from two sources available in state:

1. **`merged_extraction.requires_gpu`**: Explicit config from jobforge.yaml/build.yaml
2. **`selected_dockerfile`**: Dockerfile path for content analysis

### Merge Strategy

Both sources are analyzed, then merged:

```python
# 1. Check config (may be None, True, or False)
config_value = merged_extraction.get("requires_gpu")

# 2. Always analyze Dockerfile (when available)
dockerfile_result = analyze_dockerfile(...)  # {detected: bool, cuda_version: str|None, ...}

# 3. Merge: config is authoritative for final boolean when set
if config_value is not None:
    requires_gpu = config_value
    confidence = "high"  # Explicit config
elif dockerfile_result:
    requires_gpu = dockerfile_result["detected"]
    confidence = dockerfile_result["confidence"]
else:
    requires_gpu = False
    confidence = "low"

# 4. Return merged result with both perspectives
return {
    "requires_gpu": requires_gpu,
    "confidence": confidence,
    "config_value": config_value,
    "dockerfile_detected": dockerfile_result.get("detected", False),
    "cuda_version": dockerfile_result.get("cuda_version"),
    ...
}
```

This ensures Dockerfile analysis always runs to extract `cuda_version` and evidence, even when config sets the boolean.

## Node Placement

```
classify_workload -> detect_gpu -> analyze_ports
```

Rationale for this position:
1. After `classify_workload`: Both analyze Dockerfile, could share context in future
2. Before `analyze_ports`: GPU detection doesn't depend on port analysis
3. Has access to `merged_extraction` from parent state for config-based detection
4. Parallel execution not needed: Sequential is fine given shared Dockerfile reading

## Prompt Design Principles

The LangFuse prompt (`detect_gpu`) should:
1. Focus on Dockerfile content analysis
2. Provide clear examples of GPU vs non-GPU indicators
3. Handle multi-stage builds (check final stage primarily)
4. Return structured JSON output matching `classify_workload` pattern
5. Include CUDA version extraction when detectable

## Error Handling

- Missing Dockerfile: Return `requires_gpu: False, confidence: "low"`
- LLM parsing failure: Log warning, default to `requires_gpu: False`
- Network issues: Same fallback behavior as other LLM nodes

## Testing Strategy

1. **Unit tests** with mock LLM responses
2. **Fixture Dockerfiles**: nvidia/cuda, pytorch GPU, tensorflow GPU, plain python, node.js
3. **Edge cases**: Multi-stage builds, buildx syntax, ARG-based image selection
