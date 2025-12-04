# Tasks: Add GPU Detection

## Implementation Order

Tasks are ordered for incremental, testable progress.

### 1. Create detect_gpu prompt
- [x] Create `prompts/detect_gpu.json` with LangFuse prompt structure
- [x] Include GPU indicator examples (nvidia/cuda, pytorch-cuda, tensorflow-gpu)
- [x] Define JSON output schema: `{requires_gpu, confidence, evidence, cuda_version}`
- [x] Push to LangFuse with development label

**Validation**: `nomad-spec-prompt push --label development` succeeds ✓

### 2. Implement detect_gpu node
- [x] Create `src/subgraphs/analysis/detect_gpu.py`
- [x] Follow `classify_workload.py` pattern exactly
- [x] Read `merged_extraction.requires_gpu` for config value
- [x] Always analyze Dockerfile via LLM (when available) for cuda_version and indicators
- [x] Merge results: config authoritative for boolean, Dockerfile provides supplementary data
- [x] Implement `detect_gpu_node()` function with merge logic
- [x] Implement `create_detect_gpu_node()` factory function
- [x] Add fallback prompt for LangFuse unavailability
- [x] Output includes: `config_value`, `dockerfile_detected`, `cuda_version`

**Validation**: Module imports without errors ✓

### 3. Add unit tests for detect_gpu
- [x] Create `tests/test_subgraphs/test_detect_gpu.py`
- [x] Test config=True with GPU Dockerfile: `requires_gpu=True`, `dockerfile_detected=True`, `cuda_version` extracted
- [x] Test config=True with non-GPU Dockerfile: `requires_gpu=True`, `dockerfile_detected=False`
- [x] Test config=False with GPU Dockerfile: `requires_gpu=False`, `dockerfile_detected=True`, `cuda_version` still extracted
- [x] Test config=None with GPU Dockerfile: `requires_gpu=True` (from Dockerfile), `config_value=None`
- [x] Test config=None with non-GPU Dockerfile: `requires_gpu=False`
- [x] Test nvidia/cuda base image detection with cuda_version extraction
- [x] Test pytorch GPU image detection (high confidence)
- [x] Test tensorflow GPU image detection (high confidence)
- [x] Test missing Dockerfile: `dockerfile_detected=False`, `cuda_version=None`
- [x] Test malformed LLM response handling

**Validation**: `pytest tests/test_subgraphs/test_detect_gpu.py -v` passes (16 tests) ✓

### 4. Integrate into analysis subgraph
- [x] Update `AnalysisState` TypedDict with `gpu_detection` field
- [x] Import `create_detect_gpu_node` in `graph.py`
- [x] Add `detect_gpu` node to subgraph
- [x] Update edge: `classify_workload -> detect_gpu -> analyze_ports`
- [x] Update `create_analysis_subgraph_node()` to return `gpu_detection`

**Validation**: Analysis subgraph test passes with new node ✓

### 5. Update analysis subgraph tests
- [x] Update `tests/test_subgraphs/test_analysis.py` with gpu_detection expectations
- [x] Add integration test verifying full subgraph flow with GPU detection

**Validation**: `pytest tests/test_subgraphs/test_analysis.py -v` passes (8 tests) ✓

### 6. Wire gpu_detection to requires_gpu
- [x] Add `gpu_detection` field to main `AgentState` TypedDict
- [x] Analysis subgraph returns gpu_detection to parent state
- [x] Log detection result at INFO level

**Validation**: GPU detection result flows through state ✓

### 7. Update Configuration Summary display
- [x] Update `src/main.py` to display GPU detection confidence/evidence
- [x] Show detection source: "config", "dockerfile", or "detection"
- [x] Display CUDA version when detected
- [x] Handle config override case (config=False with GPU Dockerfile)

**Validation**: Display logic updated with rich formatting ✓

## Dependencies

- Task 2 depends on Task 1 (prompt must exist)
- Task 3 depends on Task 2 (node must exist to test)
- Task 4 depends on Tasks 2, 3 (node tested before integration)
- Tasks 5, 6, 7 can proceed in parallel after Task 4

## Out of Scope (Future Work)

- CUDA version compatibility checking against cluster
- GPU memory requirement detection
- Multi-GPU detection
- HitL confirmation for low-confidence detections
