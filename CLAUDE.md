<!-- OPENSPEC:START -->
# OpenSpec Instructions

These instructions are for AI assistants working in this project.

Always open `@/openspec/AGENTS.md` when the request:
- Mentions planning or proposals (words like proposal, spec, change, plan)
- Introduces new capabilities, breaking changes, architecture shifts, or big performance/security work
- Sounds ambiguous and you need the authoritative spec before coding

Use `@/openspec/AGENTS.md` to learn:
- How to create and apply change proposals
- Spec format and conventions
- Project structure and guidelines

Keep this managed block so 'openspec update' can refresh the instructions.

<!-- OPENSPEC:END -->

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project is a LangGraph-based AI agent system designed to automate Nomad job specification creation and deployment. The agent accepts high-level prompts, analyzes codebases, generates clarifying questions, creates Nomad job specs (HCL/JSON), deploys them, verifies status, and iterates on fixes with long-term memory capabilities.

You can assume you have an LLM at your disposal

## Technology Stack

- **Agent Framework**: LangGraph for stateful workflow orchestration with iterative cycles
- **LLM**: Qwen Coder (~30B variant, e.g., Qwen3-32B) served via vLLM for efficient inference
- **Memory Layer**: Mem0 with Qdrant vector store backend for cluster-specific long-term memory
- **Observability**: LangFuse for tracing, monitoring, and centralized prompt management
- **Prompt Optimization**: DSPy for algorithmic prompt tuning based on performance metrics
- **Infrastructure**: Self-maintained Nomad cluster for job deployment

## Architecture

### Core Workflow Loop
The agent follows a stateful graph-based workflow:
1. Analyze codebase (via Git repo or local files)
2. Generate clarifying questions based on best practices
3. Collect user responses (Human-in-the-Loop)
4. Generate Nomad job spec (HCL/JSON)
5. Deploy to Nomad cluster via API
6. Verify deployment status
7. If failed, iterate with fixes (conditional loop back)

### Key Components

- **LangGraph Graph**: Orchestrates nodes (analysis, deployment, etc.) with conditional edges for error handling loops
- **vLLM + Qwen Coder**: Hosted LLM wrapped in LangChain's VLLM class (endpoint: `http://localhost:8000`)
- **Mem0**: Persistent memory storing learnings like "Past error: Insufficient memory—recommend 4GB"
- **Qdrant**: Vector database backend for semantic search on past interactions
- **LangFuse**: Traces all LLM calls and tool invocations; manages versioned prompts with RBAC
- **DSPy**: Optimizes prompts algorithmically using metrics; refined prompts stored back in LangFuse
- **Custom Tools**: Codebase analysis (GitPython + LLM), Nomad API calls (nomad-python), deployment verification

### Subgraphs

The workflow uses LangGraph subgraphs to encapsulate related functionality:

- **Analysis Subgraph** (`src/subgraphs/analysis.py`): Encapsulates port analysis, codebase analysis, and infrastructure enrichment. Has its own `AnalysisState` TypedDict separate from `AgentState`.

Subgraph pattern:
```python
# Subgraph is compiled and invoked as a single node in the main workflow
analysis_node = create_analysis_subgraph_node(llm, settings)
workflow.add_node("analysis", analysis_node)
workflow.add_edge("merge", "analysis")
workflow.add_edge("analysis", "question")
```

Benefits:
- Clean separation of concerns with focused state
- Independent testing of analysis pipeline
- Extension point for new analysis nodes without modifying main workflow

### State Management

Agent state is persisted using Mem0 across sessions:
```python
class AgentState(TypedDict):
    prompt: str
    codebase_path: str
    questions: List[str]
    user_responses: Dict
    job_spec: str
    deployment_status: str
    memories: List[str]
```

## Project Structure

Expected repository layout:
```
main.py           # Graph runner and entry point
src/
  graph.py        # Main LangGraph workflow definition
  subgraphs/      # Nested subgraphs (e.g., analysis.py)
  nodes/          # Individual node implementations
  tools/          # Custom tools (codebase analyzer, Nomad deployer, verifier)
  memory/         # Mem0 setup and configuration
  observability/  # LangFuse integration
  prompts/        # Prompt management and DSPy optimization
config/           # Settings and configuration
requirements.txt  # Python dependencies
```

## Dependencies

Python 3.10+ required with:
- langgraph
- langchain
- langchain-community
- vllm
- mem0ai
- qdrant-client
- gitpython
- nomad-python
- langfuse
- dspy-ai

## Development Workflow

### Initial Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Launch vLLM server with Qwen model: `python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-32B --port 8000`
3. Initialize Qdrant instance (local or hosted)
4. Configure Mem0 with Qdrant backend
5. Set up LangFuse for observability

### LangGraph Nodes Implementation
Each node should:
- Fetch prompts from LangFuse (versioned, with placeholders)
- Query Mem0 before actions to inject relevant memories into context
- Trace execution via LangFuse wrapper
- Use DSPy-optimized prompts where applicable

**Key Nodes**:
- `analyze_codebase`: Extract dependencies, runtime, Nomad requirements
- `generate_questions`: Create 3-5 questions from best practices
- `collect_responses`: HitL pause for user input
- `generate_spec`: Create HCL/JSON spec informed by memories
- `deploy_job`: Register job via Nomad API
- `verify_deployment`: Poll for success (5 min timeout)
- `fix_iteration`: Rerun generation with error context from Mem0

### Conditional Edges
- Success path: analysis → questions → responses → spec → deploy → verify → complete
- Failure path: verify → fix_iteration → generate_spec (loop back)

### Testing Strategy
1. Mock Nomad with local setup or simulator
2. Provide sample repository
3. Simulate user questions
4. Induce errors to test iteration loops
5. Monitor via LangFuse dashboards
6. Optimize prompts using DSPy based on traced failures

## Prompt Management

Langfuse holds the truth, always make sure you download the latest prompt from Langfuse when making changes

See [docs/prompts.md](docs/prompts.md) for detailed documentation.

ALWAYS EDIT PROMPTS IN JOBFORGE

**Quick Reference:**

```bash
# List local prompts
nomad-spec-prompt list

# Push prompts to LangFuse
nomad-spec-prompt push --label development

# Pull prompts from LangFuse
nomad-spec-prompt pull --label production
```

**Key Files:**
- `prompts/*.json` - Local prompt files (version controlled)
- `src/prompts/__init__.py` - PromptManager with LangFuse integration
- `src/prompt_cli.py` - CLI tool for prompt management

**Architecture:**
- LangFuse is the runtime source (with labels: development, staging, production)
- Local JSON files are the version-controlled baseline and fallback
- Changes flow: Local → LangFuse → Runtime

## Memory Integration

Mem0 stores cluster-specific learnings:
- Add memory: `mem0.add("Error: Resource exhaustion—resolution: Increase CPU to 2", user_id="cluster_admin")`
- Search: `mem0.search("similar deployment errors")`
- Configure Qdrant: `mem0.config.vector_store = {"provider": "qdrant", "config": {"host": "your-qdrant-host"}}`

Memory is retrieved before spec generation and error fixing to provide contextual awareness.

## Custom Tools

Implement using LangChain's `@tool` decorator:

1. **Codebase Analyzer**: Clone/load repo with GitPython, extract deps/entrypoints/resources via LLM
2. **Nomad Deployer**: Use `nomad-python` or subprocess for CLI calls to register jobs
3. **Verifier**: Poll Nomad API for allocation status, return success/error after 5 minutes

All tool calls traced via LangFuse.

## Entry Point

CLI command structure:
```bash
python main.py --prompt "Create spec for <repo.git>" --codebase-path /path/to/repo
```

## Security Considerations

- Nomad API credentials via environment variables only
- Restrict agent to read-only codebase access
- Validate generated HCL before deployment using JSON schemas
- Use LangFuse RBAC for prompt management

## Performance Optimization

- vLLM's PagedAttention for memory efficiency
- Dynamic batching for multiple inference requests
- Monitor latency via LangFuse dashboards
- Fallback to smaller Qwen variants if 30B is too slow
- Qdrant handles scaling; limit embeddings to key facts (errors only)

## Expected Improvements

Research benchmarks indicate:
- Mem0 integration: 26% accuracy improvement, 91% faster response times in persistent scenarios
- Mem0 reduces repeat errors by 40-50% in iterative agents
- DSPy reduces prompt engineering effort via automated optimization

## Common Patterns

### Error Handling
- Log all failures via LangFuse
- Store error + resolution in Mem0 for future reference
- Use conditional edges to route back to fix nodes
- Escalate impossible deploys (invalid code) to user

### Iteration Logic
If `verify_deployment` fails:
1. Extract error from Nomad API response
2. Query Mem0 for similar past errors
3. Inject error context + memories into fix prompt
4. Route back to `generate_spec` node
5. Limit iterations to prevent infinite loops (max 3 attempts)

### LangGraph Human-in-the-Loop Pattern

**IMPORTANT**: Use the proper `interrupt()` pattern, not `interrupt_before`.

The correct way to implement HitL in LangGraph (see https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/wait-user-input/):

```python
from langgraph.types import interrupt, Command

# CORRECT: Call interrupt() INSIDE the node
def collect_responses_node(state):
    questions = state.get("questions", [])
    user_responses = interrupt(questions)  # Pauses here, returns user input when resumed
    return {**state, "user_responses": user_responses}

# Resume with Command(resume=value)
for event in graph.stream(Command(resume=responses), config):
    pass
```

**DO NOT** use the older pattern of `interrupt_before=["node_name"]` with a pass-through node and manual state updates via `graph.update_state()`. This pattern is harder to reason about and can lead to state synchronization issues.

Key points:
- `interrupt(value)` pauses the graph and passes `value` to the caller
- Resume with `Command(resume=response)` - the response becomes the return value of `interrupt()`
- Get interrupt value from state: `state.tasks[0].interrupts[0].value`
- The node handles both the pause and the response in one place

## Containerization

Prepare for Docker deployment:
- Expose as API or CLI
- Scale vLLM on GPUs if needed
- Mount configuration for Nomad endpoints
- Persistent volume for Qdrant data
