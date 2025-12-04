# Design: Analysis Subgraph Architecture

## Context

The current workflow has three consecutive analysis nodes:
```
merge -> analyze_ports -> analyze -> enrich -> question
```

These nodes share a common purpose: gathering deployment configuration data. As requirements grow (GPU detection, network mode selection, vault secrets mapping, service naming), we need a modular architecture.

### Stakeholders
- Developers extending the analysis pipeline
- The main workflow graph (consumer of analysis results)
- Future HitL interactions within analysis

### Constraints
- Must not break existing behavior
- Must maintain observability via LangFuse
- Must support future HitL interrupts for user confirmation

## Goals / Non-Goals

### Goals
- Encapsulate analysis nodes in a reusable subgraph
- Define clear `AnalysisState` contract separate from `AgentState`
- Enable independent testing of the analysis pipeline
- Provide extension points for new analysis nodes

### Non-Goals
- Implement new analysis capabilities (GPU, network mode, etc.) - future work
- Change the logic within existing nodes
- Modify the HitL patterns in the main workflow

## Decisions

### Decision 1: Use LangGraph Subgraph Pattern

**What**: Implement analysis as a compiled subgraph invoked as a single node in the main workflow.

**Why**: LangGraph supports subgraphs as first-class citizens. A subgraph:
- Has its own state schema
- Can be compiled and tested independently
- Appears as a single node to the parent graph
- Supports internal conditional edges and interrupts

**How**:
```python
# src/subgraphs/analysis.py
from langgraph.graph import StateGraph

class AnalysisState(TypedDict):
    # Inputs (from parent)
    codebase_path: str
    selected_dockerfile: str | None
    discovered_sources: dict[str, str]
    build_system_analysis: dict[str, Any]  # Jobforge/build config info
    merged_extraction: dict[str, Any]

    # Internal state
    port_analysis: dict[str, Any]
    codebase_analysis: dict[str, Any]

    # Outputs (to parent)
    app_name: str
    env_var_configs: list[dict[str, Any]]
    vault_suggestions: dict[str, Any]
    consul_conventions: dict[str, Any]
    consul_services: dict[str, Any]
    fabio_validation: dict[str, Any]
    nomad_info: dict[str, Any]
    infra_issues: list[dict[str, str]]

def create_analysis_subgraph(llm, settings) -> CompiledGraph:
    workflow = StateGraph(AnalysisState)

    workflow.add_node("analyze_ports", create_analyze_ports_node(llm))
    workflow.add_node("analyze", create_analyze_node(llm))
    workflow.add_node("enrich", create_enrich_node(settings))

    workflow.add_edge(START, "analyze_ports")
    workflow.add_edge("analyze_ports", "analyze")
    workflow.add_edge("analyze", "enrich")
    workflow.add_edge("enrich", END)

    return workflow.compile()
```

**Alternatives considered**:
1. **Function composition**: Wrap nodes in a single function. Rejected because it loses LangGraph benefits (tracing, state management, conditional routing).
2. **Keep as separate nodes**: Status quo. Rejected because it doesn't scale for additional analysis concerns.

### Decision 2: State Mapping at Boundaries

**What**: Map between `AgentState` and `AnalysisState` when entering/exiting the subgraph.

**Why**: Clean separation of concerns. The subgraph doesn't need to know about deployment status, questions, or iteration counts.

**How**:
```python
def analysis_subgraph_node(state: AgentState) -> dict[str, Any]:
    """Wrapper that maps AgentState to AnalysisState and back."""
    # Extract inputs for subgraph
    analysis_input = {
        "codebase_path": state["codebase_path"],
        "selected_dockerfile": state.get("selected_dockerfile"),
        "discovered_sources": state.get("discovered_sources", {}),
        "build_system_analysis": state.get("build_system_analysis", {}),
        "merged_extraction": state.get("merged_extraction", {}),
    }

    # Run subgraph
    result = compiled_analysis_subgraph.invoke(analysis_input)

    # Return outputs to merge into AgentState
    return {
        "port_analysis": result["port_analysis"],
        "codebase_analysis": result["codebase_analysis"],
        "app_name": result["app_name"],
        "env_var_configs": result["env_var_configs"],
        # ... other fields
    }
```

### Decision 3: Preserve Existing Node Logic

**What**: Move existing node files unchanged into subgraph orchestration.

**Why**:
- Lower risk - no logic changes
- Easier to verify correctness
- Allows incremental refactoring later

**How**: The `create_*_node` factory functions remain unchanged. Only `src/graph.py` changes to use the subgraph.

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| State mapping bugs | Unit test the mapping functions explicitly |
| Observability gaps | Ensure subgraph nodes are traced via existing LangFuse callback |
| Increased complexity | Offset by better modularity; document the pattern |
| Breaking existing tests | Run full test suite; no logic changes in nodes |

## Migration Plan

1. **Create subgraph module** (`src/subgraphs/analysis.py`)
2. **Add state mapping wrapper**
3. **Update `src/graph.py`** to use subgraph as single node
4. **Verify all tests pass**
5. **No rollback needed** - this is purely structural refactoring

## Open Questions

1. **HitL within subgraph**: Should future analysis questions (e.g., "Confirm GPU requirement?") use `interrupt()` inside the subgraph? Initial answer: Yes, but defer implementation.
2. **Subgraph location**: `Propose `src/subgraphs/` for clarity.
