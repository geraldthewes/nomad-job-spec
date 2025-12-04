## Context

The agent generates Nomad job specifications but currently hardcodes `job_type = "service"`. Nomad has three job types:
- `service`: Long-running tasks, continuously restarted, needs health checks
- `batch`: Run-to-completion tasks, not restarted after success
- `system`: One per node (rarely auto-detected, usually explicit)

Detecting `service` vs `batch` is critical because:
1. Batch jobs should NOT have service registration (no Consul service, no Fabio routes)
2. Batch jobs may skip health checks or use different restart policies
3. Questions about scaling/load balancing don't apply to batch jobs

## Goals / Non-Goals

**Goals:**
- Accurately classify Dockerfile-based applications as `service` or `batch`
- Integrate classification early in analysis so downstream nodes can use it
- Store prompt in LangFuse for iteration and tuning

**Non-Goals:**
- Auto-detect `system` jobs (these are infrastructure-specific, require explicit user choice)
- Handle non-Docker workloads (out of scope)
- Change question generation logic in this proposal (future enhancement)

## Decisions

### Decision: Position in Analysis Pipeline
Place `classify_workload` as the **first** node in the analysis subgraph, before `analyze_ports`.

**Rationale:** Workload type could theoretically affect how we interpret ports (batch jobs often don't need exposed ports), though in the current design we run all nodes regardless. Having it first establishes context for later nodes.

**Alternative considered:** Run in parallel with `analyze_ports`. Rejected because LangGraph subgraph edges are sequential by default and the overhead is minimal.

### Decision: LLM-Based Classification
Use the LLM (via LangFuse-managed prompt) rather than heuristics.

**Rationale:** As demonstrated, even a simple prompt correctly identifies patterns like `uvicorn` → service. Heuristics would need constant updating for new frameworks. The LLM generalizes better.

**Alternative considered:** Regex-based detection (e.g., if CMD contains "uvicorn|gunicorn|nginx" → service). Rejected because it's brittle and won't handle unfamiliar entrypoints.

### Decision: Classification Output
Output a structured JSON with:
```json
{
  "workload_type": "service" | "batch",
  "confidence": "high" | "medium" | "low",
  "evidence": "CMD uses uvicorn to run a web server on port 8000"
}
```

**Rationale:** Confidence level allows the question node to ask for confirmation on low-confidence classifications.

### Decision: Fallback Behavior
If classification fails or confidence is low, default to `"service"` (current behavior) and let the user correct via questions.

**Rationale:** Services are more common and the default produces a working (if suboptimal) spec. Better to be conservative.

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| LLM misclassifies workload | Include confidence score; question node can ask for confirmation on low confidence |
| No Dockerfile available | Skip classification; default to service; log warning |
| Batch job incorrectly gets service blocks | Spec generation checks workload_type before adding service registration |

## Resolved Questions

**Q: Should the question generation node ask "Is this a batch job or a service?" when confidence is medium/low?**
A: Yes. When confidence is "low", the question node should include a confirmation question about workload type.

**Q: Should batch jobs skip the entire enrich node (no Consul/Fabio needed)?**
A: No, not in this iteration. Batch jobs will still run through the full enrich flow. We're focusing on the service flow for now; optimizing the batch path is a future enhancement.
