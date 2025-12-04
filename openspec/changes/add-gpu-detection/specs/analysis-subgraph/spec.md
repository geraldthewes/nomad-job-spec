# analysis-subgraph Spec Delta

## ADDED Requirements

### Requirement: GPU Detection Node
The system SHALL include a `detect_gpu` node that analyzes both build configuration and Dockerfile, merging results with config as authoritative for the final boolean.

#### Scenario: Config true with GPU Dockerfile
- **GIVEN** `merged_extraction.requires_gpu` is `True` from jobforge.yaml/build.yaml
- **AND** a Dockerfile with `nvidia/cuda:12.1` base image
- **WHEN** the `detect_gpu` node executes
- **THEN** the node outputs `requires_gpu: true`
- **AND** `config_value: true`
- **AND** `dockerfile_detected: true`
- **AND** `cuda_version: "12.1"` is extracted from Dockerfile
- **AND** confidence is "high"

#### Scenario: Config true with non-GPU Dockerfile
- **GIVEN** `merged_extraction.requires_gpu` is `True` from jobforge.yaml/build.yaml
- **AND** a Dockerfile with `python:3.11-slim` base image
- **WHEN** the `detect_gpu` node executes
- **THEN** the node outputs `requires_gpu: true`
- **AND** `config_value: true`
- **AND** `dockerfile_detected: false`
- **AND** `cuda_version: null`
- **AND** confidence is "high"

#### Scenario: Config false with GPU Dockerfile (cuda_version still extracted)
- **GIVEN** `merged_extraction.requires_gpu` is `False` from jobforge.yaml/build.yaml
- **AND** a Dockerfile with `nvidia/cuda:11.8` base image
- **WHEN** the `detect_gpu` node executes
- **THEN** the node outputs `requires_gpu: false` (config is authoritative)
- **AND** `config_value: false`
- **AND** `dockerfile_detected: true`
- **AND** `cuda_version: "11.8"` is still extracted for informational purposes
- **AND** evidence notes config overrides Dockerfile detection

#### Scenario: No config with GPU Dockerfile
- **GIVEN** `merged_extraction.requires_gpu` is not set (None)
- **AND** a Dockerfile with `nvidia/cuda:*` base image
- **WHEN** the `detect_gpu` node executes
- **THEN** the node outputs `requires_gpu: true` (from Dockerfile)
- **AND** `config_value: null`
- **AND** `dockerfile_detected: true`
- **AND** confidence reflects Dockerfile analysis confidence

#### Scenario: NVIDIA CUDA base image detection
- **GIVEN** a Dockerfile with base image `nvidia/cuda:*`
- **WHEN** the `detect_gpu` node analyzes the Dockerfile
- **THEN** `dockerfile_detected` is `true`
- **AND** confidence is "high"
- **AND** evidence references the CUDA base image
- **AND** `cuda_version` is extracted from image tag

#### Scenario: PyTorch GPU image detection
- **GIVEN** a Dockerfile with base image `pytorch/pytorch:*-cuda*` or similar GPU variant
- **WHEN** the `detect_gpu` node analyzes the Dockerfile
- **THEN** `dockerfile_detected` is `true`
- **AND** confidence is "high"
- **AND** `cuda_version` is extracted if present in the image tag

#### Scenario: TensorFlow GPU image detection
- **GIVEN** a Dockerfile with base image `tensorflow/tensorflow:*-gpu*`
- **WHEN** the `detect_gpu` node analyzes the Dockerfile
- **THEN** `dockerfile_detected` is `true`
- **AND** confidence is "high"

#### Scenario: CUDA toolkit installation detection
- **GIVEN** a Dockerfile that installs `nvidia-cuda-toolkit`, `cuda-*`, or `cudnn*` packages
- **WHEN** the `detect_gpu` node analyzes the Dockerfile
- **THEN** `dockerfile_detected` is `true`
- **AND** confidence is "high" or "medium" based on context

#### Scenario: GPU environment variables detection
- **GIVEN** a Dockerfile that sets `NVIDIA_VISIBLE_DEVICES` or `CUDA_VISIBLE_DEVICES`
- **WHEN** the `detect_gpu` node analyzes the Dockerfile
- **THEN** `dockerfile_detected` is `true`
- **AND** evidence references the environment variable

#### Scenario: Plain application image (no GPU)
- **GIVEN** a Dockerfile with base image `python:3.11-slim` or `node:20-alpine`
- **AND** no GPU-related packages or environment variables
- **WHEN** the `detect_gpu` node analyzes the Dockerfile
- **THEN** `dockerfile_detected` is `false`
- **AND** confidence is "high"

#### Scenario: Missing Dockerfile
- **GIVEN** no Dockerfile is available for analysis
- **WHEN** the `detect_gpu` node executes
- **THEN** `dockerfile_detected` is `false`
- **AND** `cuda_version` is `null`
- **AND** if `config_value` is also `null`, `requires_gpu` defaults to `false` with "low" confidence

#### Scenario: Multi-stage build analysis
- **GIVEN** a multi-stage Dockerfile where only the final stage uses a GPU image
- **WHEN** the `detect_gpu` node analyzes the Dockerfile
- **THEN** the node primarily considers the final stage for GPU detection
- **AND** evidence notes the multi-stage build context

## MODIFIED Requirements

### Requirement: AnalysisState Definition
The system SHALL define an `AnalysisState` TypedDict that contains only fields relevant to the analysis phase, separate from the full `AgentState`.

#### Scenario: Output fields (MODIFIED)
- **WHEN** analysis subgraph completes
- **THEN** AnalysisState contains: workload_classification, **gpu_detection**, port_analysis, codebase_analysis, app_name, env_var_configs, vault_suggestions, consul_conventions, consul_services, fabio_validation, nomad_info, infra_issues
- **AND** gpu_detection contains:
  - requires_gpu (bool): Final merged decision
  - confidence (str): "high", "medium", or "low"
  - evidence (str): Explanation of decision
  - config_value (bool | None): What config said, if set
  - dockerfile_detected (bool): Whether Dockerfile indicated GPU usage
  - cuda_version (str | None): Extracted CUDA version from Dockerfile

### Requirement: Analysis Subgraph Structure
The system SHALL implement an analysis subgraph as a LangGraph StateGraph that orchestrates workload classification, GPU detection, port analysis, codebase analysis, and infrastructure enrichment nodes in sequence.

#### Scenario: Subgraph node sequence (MODIFIED)
- **WHEN** the analysis subgraph is invoked
- **THEN** nodes execute in order: classify_workload -> **detect_gpu** -> analyze_ports -> analyze -> enrich
- **AND** each node receives the accumulated state from previous nodes
