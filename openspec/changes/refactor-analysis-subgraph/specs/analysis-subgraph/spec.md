## ADDED Requirements

### Requirement: Analysis Subgraph Structure
The system SHALL implement an analysis subgraph as a LangGraph StateGraph that orchestrates port analysis, codebase analysis, and infrastructure enrichment nodes in sequence.

#### Scenario: Subgraph node sequence
- **WHEN** the analysis subgraph is invoked
- **THEN** nodes execute in order: analyze_ports -> analyze -> enrich
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
- **THEN** AnalysisState contains: port_analysis, codebase_analysis, app_name, env_var_configs, vault_suggestions, consul_conventions, consul_services, fabio_validation, nomad_info, infra_issues

### Requirement: Node Factory Compatibility
The system SHALL reuse existing node factory functions (`create_analyze_ports_node`, `create_analyze_node`, `create_enrich_node`) without modification.

#### Scenario: Existing node reuse
- **WHEN** the analysis subgraph is constructed
- **THEN** it uses the same node factory functions as the previous flat workflow
- **AND** node behavior remains identical

### Requirement: Observability Preservation
The system SHALL ensure all nodes within the analysis subgraph are traced via LangFuse, maintaining the same observability as the flat workflow.

#### Scenario: Subgraph tracing
- **WHEN** the analysis subgraph executes
- **THEN** each internal node appears as a child span under the subgraph span in LangFuse
- **AND** LLM calls within nodes are traced with their prompts and responses

### Requirement: Independent Testability
The system SHALL allow the analysis subgraph to be tested in isolation without requiring the full workflow.

#### Scenario: Isolated subgraph test
- **WHEN** a test invokes the compiled analysis subgraph directly
- **THEN** it can provide mock AnalysisState input
- **AND** verify output fields without running deploy or question nodes

### Requirement: Extension Point
The system SHALL allow new analysis nodes to be added to the subgraph without modifying the main workflow.

#### Scenario: Adding a new analysis node
- **WHEN** a new analysis concern (e.g., GPU detection) needs to be added
- **THEN** a new node can be inserted into the subgraph's node sequence
- **AND** the main workflow requires no changes
