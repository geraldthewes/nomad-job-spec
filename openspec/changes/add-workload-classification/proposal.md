# Change: Add Workload Classification Node

## Why
The system currently defaults all jobs to `type = "service"`, but Nomad distinguishes between `service` (long-running) and `batch` (one-time) jobs. Batch jobs have different lifecycle semantics—they run once and exit rather than being restarted continuously. Misclassifying a batch job as a service leads to unnecessary restarts and incorrect health check expectations.

By analyzing the Dockerfile CMD/ENTRYPOINT, we can infer whether the application is a long-running service (e.g., uvicorn, nginx, node server) or a batch process (e.g., python script.py, data migration).

## What Changes
- Add a `classify_workload` node to the analysis subgraph (runs first in sequence)
- New `workload_type` field in `AnalysisState` and `AgentState` (values: `service`, `batch`)
- LangFuse prompt `classify_workload` for LLM-based classification
- Spec generation uses detected `workload_type` instead of hardcoded "service"
- Skip service registration and health check blocks for batch jobs

## Impact
- Affected specs: `analysis-subgraph`, `spec-generation`
- Affected code:
  - `src/subgraphs/analysis/classify_workload.py` (new file)
  - `src/subgraphs/analysis/graph.py` (add node to sequence)
  - `src/graph.py` (add `workload_type` to AgentState)
  - `src/nodes/generate.py` (use workload_type for job_type)
  - LangFuse: new `classify_workload` prompt
