## 1. Prompt Creation
- [x] 1.1 Create `classify_workload` prompt in LangFuse with development label
- [x] 1.2 Add local fallback prompt in `classify_workload.py`

## 2. Classification Node Implementation
- [x] 2.1 Create `src/subgraphs/analysis/classify_workload.py` with `create_classify_workload_node` factory
- [x] 2.2 Implement Dockerfile reading and LLM invocation
- [x] 2.3 Parse JSON response with type, confidence, evidence fields
- [x] 2.4 Handle missing Dockerfile gracefully (default to service)

## 3. State Updates
- [x] 3.1 Add `workload_classification` field to `AnalysisState` in `graph.py`
- [x] 3.2 Add `workload_classification` field to `AgentState` in `src/graph.py`
- [x] 3.3 Update state mapping in `create_analysis_subgraph_node`

## 4. Subgraph Integration
- [x] 4.1 Import `create_classify_workload_node` in `src/subgraphs/analysis/graph.py`
- [x] 4.2 Add `classify_workload` node to subgraph
- [x] 4.3 Update edge sequence: START → classify_workload → analyze_ports → analyze → enrich

## 5. Spec Generation Integration
- [x] 5.1 Update `_build_job_config` in `src/nodes/generate.py` to use `workload_classification.workload_type`
- [x] 5.2 Pass workload_type to `JobConfig.job_type`
- [x] 5.3 Skip service registration block for batch jobs in HCL generator

## 6. Question Generation Integration
- [x] 6.1 Update question node to check `workload_classification.confidence`
- [x] 6.2 Add workload type confirmation question when confidence is "low"

## 7. Testing
- [x] 7.1 Unit test for classify_workload node with mock LLM
- [x] 7.2 Test case: Dockerfile with uvicorn → service
- [x] 7.3 Test case: Dockerfile with `python script.py` → batch
- [x] 7.4 Test case: Missing Dockerfile → defaults to service
- [x] 7.5 Integration test: full subgraph with classification

## 8. Validation
- [x] 8.1 Run `openspec validate add-workload-classification --strict`
- [ ] 8.2 Manual test with sample Dockerfiles
