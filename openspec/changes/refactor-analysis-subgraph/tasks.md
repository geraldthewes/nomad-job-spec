# Tasks: Refactor Analysis Subgraph

## 1. Subgraph Infrastructure

- [x] 1.1 Create `src/subgraphs/` directory
- [x] 1.2 Create `src/subgraphs/__init__.py` with exports
- [x] 1.3 Define `AnalysisState` TypedDict in `src/subgraphs/analysis.py`
- [x] 1.4 Implement `create_analysis_subgraph()` function
- [x] 1.5 Implement state mapping wrapper `create_analysis_subgraph_node()`

## 2. Main Workflow Integration

- [x] 2.1 Update `src/graph.py` imports to include analysis subgraph
- [x] 2.2 Replace `analyze_ports`, `analyze`, `enrich` nodes with single `analysis` subgraph node
- [x] 2.3 Update edge definitions: `merge -> analysis -> question`
- [x] 2.4 Remove old node factory calls from `create_workflow()`

## 3. Testing

- [x] 3.1 Add unit tests for `AnalysisState` mapping functions
- [x] 3.2 Add integration test for analysis subgraph in isolation
- [x] 3.3 Verify existing workflow tests pass unchanged
- [x] 3.4 Test with sample codebase to confirm identical output

## 4. Documentation

- [x] 4.1 Update `CLAUDE.md` architecture section to reflect subgraph pattern
- [x] 4.2 Add docstrings to new subgraph module

## Dependencies

- Task 2.x depends on Task 1.x completion
- Task 3.1-3.2 can run in parallel with Task 2.x
- Task 3.3-3.4 requires Task 2.x completion
- Task 4.x can run in parallel after Task 1.x

## Verification

After completion:
1. `pytest tests/` - 146 tests pass (2 pre-existing failures unrelated to this change)
2. Manual test with real codebase produces same job spec
3. LangFuse traces show subgraph nodes properly nested

## Implementation Notes

- Created `src/subgraphs/analysis.py` with `AnalysisState`, `create_analysis_subgraph()`, and `create_analysis_subgraph_node()`
- Updated `src/graph.py` to use the analysis subgraph as a single node
- Added tests in `tests/test_subgraphs/test_analysis.py`
- Updated `tests/conftest.py` mock_llm to handle port analysis prompts
- Updated `tests/test_graph.py` to check for `analysis` node instead of `analyze`
