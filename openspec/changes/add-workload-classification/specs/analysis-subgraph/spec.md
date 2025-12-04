## ADDED Requirements

### Requirement: Workload Classification Node
The system SHALL include a `classify_workload` node that analyzes the Dockerfile to determine whether the application is a long-running service or a batch job.

#### Scenario: Service classification
- **WHEN** the Dockerfile CMD/ENTRYPOINT runs a web server (e.g., uvicorn, gunicorn, nginx, node server.js)
- **THEN** the node outputs `workload_type: "service"`
- **AND** includes evidence explaining the classification

#### Scenario: Batch classification
- **WHEN** the Dockerfile CMD/ENTRYPOINT runs a script that exits (e.g., python script.py, data migration)
- **THEN** the node outputs `workload_type: "batch"`
- **AND** includes evidence explaining the classification

#### Scenario: Missing Dockerfile
- **WHEN** no Dockerfile is available for analysis
- **THEN** the node defaults to `workload_type: "service"`
- **AND** sets confidence to "low"

#### Scenario: Low confidence classification
- **WHEN** the LLM cannot confidently determine workload type
- **THEN** the node returns confidence: "low" or "medium"
- **AND** downstream nodes may prompt user for confirmation

## MODIFIED Requirements

### Requirement: Analysis Subgraph Structure
The system SHALL implement an analysis subgraph as a LangGraph StateGraph that orchestrates workload classification, port analysis, codebase analysis, and infrastructure enrichment nodes in sequence.

#### Scenario: Subgraph node sequence
- **WHEN** the analysis subgraph is invoked
- **THEN** nodes execute in order: classify_workload -> analyze_ports -> analyze -> enrich
- **AND** each node receives the accumulated state from previous nodes

#### Scenario: Subgraph compilation
- **WHEN** the analysis subgraph is created via `create_analysis_subgraph()`
- **THEN** it returns a compiled graph that can be invoked with `AnalysisState`
- **AND** the compiled graph can be embedded in the main workflow

### Requirement: AnalysisState Definition
The system SHALL define an `AnalysisState` TypedDict that contains only fields relevant to the analysis phase, separate from the full `AgentState`.

#### Scenario: Input fields
- **WHEN** AnalysisState is initialized
- **THEN** it accepts: codebase_path (str), selected_dockerfile (str | None), discovered_sources (dict), build_system_analysis (dict), merged_extraction (dict)
- **AND** build_system_analysis contains the jobforge/build config info (mechanism, config_path, dockerfile_used)

#### Scenario: Output fields
- **WHEN** analysis subgraph completes
- **THEN** AnalysisState contains: workload_classification, port_analysis, codebase_analysis, app_name, env_var_configs, vault_suggestions, consul_conventions, consul_services, fabio_validation, nomad_info, infra_issues
- **AND** workload_classification contains workload_type, confidence, and evidence fields
