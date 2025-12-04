## MODIFIED Requirements

### Requirement: Workflow Node Sequence
The system SHALL execute nodes in the following sequence: discover_sources, analyze_build_system, [confirm], extract, merge, analysis_subgraph, generate_questions, collect_responses, generate_spec, deploy_job, verify_deployment, with conditional routing to fix_iteration on failure.

The analysis_subgraph encapsulates port analysis, codebase analysis, and infrastructure enrichment as a nested graph with its own `AnalysisState`.

#### Scenario: Successful workflow path
- **WHEN** all nodes complete without errors and deployment succeeds
- **THEN** the workflow follows: discover_sources -> analyze_build_system -> [confirm] -> extract -> merge -> analysis_subgraph -> generate_questions -> collect_responses -> generate_spec -> deploy_job -> verify_deployment -> complete
- **AND** the analysis_subgraph internally executes: analyze_ports -> analyze -> enrich

#### Scenario: Failed deployment triggers fix iteration
- **WHEN** verify_deployment detects a failed allocation
- **THEN** the workflow routes to fix_iteration node
- **AND** fix_iteration routes back to generate_spec with error context

## ADDED Requirements

### Requirement: Analysis Subgraph Integration
The system SHALL integrate the analysis subgraph as a single node in the main workflow, with state mapping between `AgentState` and `AnalysisState` at subgraph boundaries.

#### Scenario: Subgraph state input mapping
- **WHEN** the main workflow reaches the analysis_subgraph node
- **THEN** the system extracts relevant fields from `AgentState` (codebase_path, selected_dockerfile, discovered_sources, build_system_analysis, merged_extraction)
- **AND** constructs an `AnalysisState` for the subgraph
- **AND** build_system_analysis provides jobforge/build config info from earlier nodes

#### Scenario: Subgraph state output mapping
- **WHEN** the analysis subgraph completes
- **THEN** the system merges output fields back into `AgentState` (port_analysis, codebase_analysis, app_name, env_var_configs, vault_suggestions, consul_conventions, consul_services, fabio_validation, nomad_info, infra_issues)
- **AND** the main workflow continues to generate_questions
