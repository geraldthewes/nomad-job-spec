# Change: Refactor Analysis Pipeline into a Subgraph

## Why

The current analysis pipeline (`analyze_ports` -> `analyze` -> `enrich`) handles multiple concerns in separate nodes but lacks a cohesive architecture for extensibility. As we need to support more deployment questions (GPU requirements, network mode, vault secrets, service names), these should be modular nodes within a dedicated subgraph rather than scattered across the main workflow.

A subgraph provides:
- Clear separation of concerns with its own `AnalysisState`
- Ability to add new analysis nodes without modifying the main workflow
- Potential for HitL interrupts within the analysis phase
- Better testability of the analysis pipeline in isolation

## What Changes

- **NEW**: Create an analysis subgraph that encapsulates `analyze_ports`, `analyze`, and `enrich` nodes
- **NEW**: Define `AnalysisState` TypedDict for subgraph-internal state
- **MODIFIED**: Main workflow replaces three nodes with a single `analysis_subgraph` node
- **MODIFIED**: State mapping between `AgentState` and `AnalysisState` at subgraph boundaries

This is a **migration-first approach**: existing node logic remains unchanged, only the orchestration is restructured.

## Impact

- Affected specs: `core-workflow`, new `analysis-subgraph` capability
- Affected code:
  - `src/graph.py` - main workflow definition
  - `src/nodes/analyze.py` - no code changes, just moved into subgraph
  - `src/nodes/analyze_ports.py` - no code changes, just moved into subgraph
  - `src/nodes/enrich.py` - no code changes, just moved into subgraph
  - New file: `src/subgraphs/analysis.py` - subgraph definition

## Success Criteria

1. All existing tests pass without modification
2. Workflow produces identical output for same inputs
3. Subgraph can be tested independently
4. Clear extension point for future analysis nodes
