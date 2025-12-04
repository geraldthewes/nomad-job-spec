# Proposal: Add GPU Detection to Analysis Subgraph

## Summary

Add a `detect_gpu` node to the analysis subgraph that analyzes Dockerfiles to determine if a workload requires or benefits from GPU resources. This enables automatic GPU constraint configuration in generated Nomad job specs.

## Motivation

Currently, GPU requirements are only detected when explicitly configured in jobforge.yaml (`gpu_required: true`). Many containerized ML/AI workloads use NVIDIA CUDA base images or install GPU libraries without explicit configuration. Detecting these patterns automatically would:

1. Reduce user burden - no need to manually specify GPU requirements
2. Prevent deployment failures - GPU workloads scheduled on non-GPU nodes fail
3. Enable smarter resource allocation - route GPU workloads to gpu001 (RTX 3080) or gpu007 (GTX 1080 Ti)

## Scope

This change adds:
- A new `detect_gpu` node in the analysis subgraph
- A LangFuse prompt for GPU detection analysis
- Consolidation of GPU info from build config AND Dockerfile
- Unit tests for the detection node
- State field `gpu_detection` in AnalysisState

This change does NOT include:
- Modifications to spec generation (will use existing `requires_gpu` field)
- CUDA version matching against cluster capabilities
- Multi-GPU detection or GPU memory requirements

## Detection Strategy

The node analyzes BOTH sources and merges results:

### Source 1: Explicit Build Configuration
- `jobforge.yaml` / `build.yaml` with `gpu_required: true/false`
- Already extracted by jobforge extractor and available in `merged_extraction`
- Provides explicit user intent

### Source 2: Dockerfile Analysis
Analyze Dockerfile content via LLM for GPU indicators:

**High confidence signals:**
- Base images: `nvidia/cuda:*`, `pytorch/pytorch:*-cuda*`, `tensorflow/tensorflow:*-gpu*`
- Package installs: `nvidia-cuda-toolkit`, `cuda-*`, `cudnn*`
- Environment variables: `NVIDIA_VISIBLE_DEVICES`, `CUDA_VISIBLE_DEVICES`

**Medium confidence signals:**
- ML framework imports detected alongside GPU-compatible base images
- `torch`, `tensorflow`, `jax` in requirements with no explicit CPU-only markers

**Low confidence signals:**
- Generic Python ML image with unclear GPU usage

### Merge Logic
Both sources are always analyzed (when available), then merged:

1. **Final `requires_gpu` decision:**
   - If config is set, config value is authoritative (explicit intent)
   - Else use Dockerfile detection result
   - Else default to `false`

2. **Supplementary data from Dockerfile (always extracted):**
   - `cuda_version`: Extracted even if config sets the boolean
   - `dockerfile_indicators`: List of detected GPU signals for evidence

3. **Output includes both perspectives:**
   ```python
   {
       "requires_gpu": bool,        # Final merged decision
       "confidence": str,           # Overall confidence
       "evidence": str,             # Explanation of decision
       "config_value": bool | None, # What config said (if set)
       "dockerfile_detected": bool, # What Dockerfile analysis found
       "cuda_version": str | None,  # From Dockerfile (e.g., "12.1")
   }
   ```

This ensures Dockerfile analysis provides `cuda_version` and context even when config explicitly sets the GPU requirement.

## Integration Point

The node fits into the existing analysis subgraph flow:

```
START -> classify_workload -> detect_gpu -> analyze_ports -> analyze -> enrich -> END
```

This placement allows GPU detection results to inform downstream analysis and question generation.

## Related Specifications

- `analysis-subgraph`: Adds new node and state field
- `spec-generation`: No changes needed (already supports `requires_gpu`)
