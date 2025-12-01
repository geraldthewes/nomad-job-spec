 User approved Claude's plan
  ⎿  Plan saved to: ~/.claude/plans/idempotent-watching-rainbow.md · /plan to edit
     Nomad Job Spec Agent - Implementation Plan

     Executive Summary

     Build a LangGraph-based AI agent that automates Nomad Docker job specification creation through an iterative workflow: analyze
      codebase → ask clarifying questions → generate HCL spec → deploy → verify → fix if needed.

     Scope: Full stack (LangGraph + vLLM/Qwen + Mem0/Qdrant + LangFuse + DSPy) for Docker-only jobs.

     Important: All external services are already running and configured:
     - vLLM + Qwen 30B (serving at configured endpoint)
     - Nomad cluster (production-ready)
     - Qdrant (vector database for Mem0)
     - LangFuse (observability platform)

     We are building only the agent application - no infrastructure setup required.

     ---
     Critical PRD Issues Addressed

     | Issue                     | Resolution                                                             |
     |---------------------------|------------------------------------------------------------------------|
     | Over-engineering risk     | Implement in 4 phases with working software at each phase              |
     | Missing HCL validation    | Use python-nomad to validate specs before deployment                   |
     | Codebase analysis unclear | Focus on Dockerfile/docker-compose detection + dependency file parsing |
     | Testing strategy weak     | Use Nomad's -check-index for dry-runs + integration test suite         |
     | No error categorization   | Classify Nomad errors into fixable vs unfixable categories             |

     ---
     Architecture Overview

     ┌─────────────────────────────────────────────────────────────────┐
     │                         CLI Entry Point                          │
     │                    python main.py --prompt "..."                 │
     └─────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
     ┌─────────────────────────────────────────────────────────────────┐
     │                      LangGraph Orchestrator                      │
     │  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐     │
     │  │ Analyze  │──▶│ Question │──▶│ Generate │──▶│  Deploy  │     │
     │  │ Codebase │   │   HitL   │   │   Spec   │   │   Job    │     │
     │  └──────────┘   └──────────┘   └──────────┘   └──────────┘     │
     │        │                            ▲              │            │
     │        │                            │              ▼            │
     │        │                       ┌──────────┐   ┌──────────┐     │
     │        │                       │   Fix    │◀──│  Verify  │     │
     │        │                       │   Spec   │   │  Status  │     │
     │        │                       └──────────┘   └──────────┘     │
     │        │                            │              │            │
     │        ▼                            ▼              ▼            │
     │  ┌─────────────────────────────────────────────────────────┐   │
     │  │                    Mem0 Memory Layer                     │   │
     │  │              (Qdrant Vector Store Backend)               │   │
     │  └─────────────────────────────────────────────────────────┘   │
     └─────────────────────────────────────────────────────────────────┘
              │                    │                    │
              ▼                    ▼                    ▼
         ┌─────────┐         ┌─────────┐          ┌─────────┐
         │  vLLM   │         │LangFuse │          │  Nomad  │
         │  Qwen   │         │  Trace  │          │   API   │
         └─────────┘         └─────────┘          └─────────┘

     ---
     Project Structure

     nomad-job-spec/
     ├── src/
     │   ├── __init__.py
     │   ├── main.py              # CLI entry point + graph runner
     │   ├── graph.py             # LangGraph definition (nodes, edges, state)
     │   ├── nodes/
     │   │   ├── __init__.py
     │   │   ├── analyze.py       # Codebase analysis node
     │   │   ├── question.py      # HitL questioning node
     │   │   ├── generate.py      # HCL spec generation node
     │   │   ├── deploy.py        # Nomad deployment node
     │   │   ├── verify.py        # Status verification node
     │   │   └── fix.py           # Error fixing node
     │   ├── tools/
     │   │   ├── __init__.py
     │   │   ├── codebase.py      # GitPython + file analysis
     │   │   ├── nomad.py         # Nomad API wrapper
     │   │   └── hcl.py           # HCL generation/validation
     │   ├── memory/
     │   │   ├── __init__.py
     │   │   └── mem0_client.py   # Mem0 + Qdrant setup
     │   ├── llm/
     │   │   ├── __init__.py
     │   │   └── provider.py      # LLM abstraction (vLLM/OpenAI/etc)
     │   ├── observability/
     │   │   ├── __init__.py
     │   │   └── langfuse.py      # Tracing + prompt management
     │   └── prompts/
     │       ├── __init__.py
     │       ├── templates.py     # Base prompt templates
     │       └── optimizer.py     # DSPy optimization
     ├── tests/
     │   ├── __init__.py
     │   ├── conftest.py          # Pytest fixtures
     │   ├── test_graph.py        # Integration tests
     │   ├── test_nodes/          # Unit tests per node
     │   ├── test_tools/          # Tool unit tests
     │   └── fixtures/
     │       └── sample_repos/    # Test codebases
     ├── config/
     │   ├── settings.py          # Pydantic settings
     │   └── prompts/             # Prompt YAML files (LangFuse managed)
     ├── pyproject.toml           # Dependencies + project config
     └── README.md

     ---
     Phase 1: Core Graph + LLM Integration (Foundation)

     Goals

     - Working LangGraph with basic flow
     - LLM integration via abstraction layer
     - Simple CLI entry point

     Implementation Steps

     1.1 Project Setup

     # pyproject.toml with dependencies:
     # langgraph, langchain, langchain-community, langchain-openai
     # python-nomad, gitpython, pydantic, pydantic-settings
     # typer (CLI), rich (output formatting)

     1.2 LLM Abstraction Layer (src/llm/provider.py)

     from abc import ABC, abstractmethod
     from langchain_community.llms import VLLM
     from langchain_openai import ChatOpenAI

     class LLMProvider(ABC):
         @abstractmethod
         def get_llm(self): ...

     class VLLMProvider(LLMProvider):
         def __init__(self, base_url: str, model: str):
             self.base_url = base_url
             self.model = model

         def get_llm(self):
             return VLLM(
                 openai_api_base=f"{self.base_url}/v1",
                 model_name=self.model,
                 openai_api_key="EMPTY"
             )

     1.3 Agent State Definition (src/graph.py)

     from typing import TypedDict, List, Dict, Optional, Annotated
     from langgraph.graph.message import add_messages

     class AgentState(TypedDict):
         # Input
         prompt: str
         codebase_path: str

         # Analysis results
         codebase_analysis: Dict  # deps, dockerfile info, entrypoint

         # Conversation
         messages: Annotated[list, add_messages]
         questions: List[str]
         user_responses: Dict[str, str]

         # Generation
         job_spec: str  # HCL content
         job_spec_json: Dict  # Parsed for validation

         # Deployment
         job_id: Optional[str]
         deployment_status: str  # pending, running, failed, success
         deployment_error: Optional[str]

         # Iteration control
         iteration_count: int
         max_iterations: int  # default 3

         # Memory context
         relevant_memories: List[str]

     1.4 Basic Nodes (Skeleton)

     Analyze Node (src/nodes/analyze.py):
     - Clone/read repo
     - Detect Dockerfile (parse FROM, EXPOSE, CMD)
     - Parse package.json / requirements.txt / go.mod
     - Extract environment variable patterns
     - Return structured analysis

     Generate Node (src/nodes/generate.py):
     - Use LLM with analysis + user responses
     - Generate HCL for Docker job
     - Validate HCL syntax

     1.5 Graph Assembly

     from langgraph.graph import StateGraph, START, END

     workflow = StateGraph(AgentState)
     workflow.add_node("analyze", analyze_codebase)
     workflow.add_node("question", generate_questions)
     workflow.add_node("collect", collect_responses)  # HitL
     workflow.add_node("generate", generate_spec)
     workflow.add_node("deploy", deploy_job)
     workflow.add_node("verify", verify_deployment)
     workflow.add_node("fix", fix_spec)

     # Edges
     workflow.add_edge(START, "analyze")
     workflow.add_edge("analyze", "question")
     workflow.add_edge("question", "collect")
     workflow.add_edge("collect", "generate")
     workflow.add_edge("generate", "deploy")
     workflow.add_edge("deploy", "verify")

     # Conditional: verify → fix → generate OR verify → END
     workflow.add_conditional_edges(
         "verify",
         should_retry,
         {"retry": "fix", "success": END, "give_up": END}
     )
     workflow.add_edge("fix", "generate")

     Deliverables Phase 1

     - Working graph that can analyze a repo and generate basic HCL
     - CLI: python -m src.main --prompt "Deploy X" --path /repo
     - Unit tests for each node
     - Integration test with mock LLM responses

     ---
     Phase 2: Nomad Integration + HitL

     Goals

     - Real Nomad API integration
     - Human-in-the-loop questioning
     - Deployment verification loop

     Implementation Steps

     2.1 Nomad Tools (src/tools/nomad.py)

     import nomad

     class NomadClient:
         def __init__(self, address: str, token: str = None):
             self.client = nomad.Nomad(host=address, token=token)

         def validate_job(self, hcl: str) -> tuple[bool, str]:
             """Dry-run validation via plan endpoint"""
             try:
                 # Convert HCL to JSON, submit for planning
                 result = self.client.job.plan(job_id, job_dict, diff=True)
                 return True, None
             except nomad.api.exceptions.BadRequestNomadException as e:
                 return False, str(e)

         def register_job(self, hcl: str) -> str:
             """Deploy and return job ID"""
             ...

         def get_job_status(self, job_id: str) -> dict:
             """Get deployment status + allocation health"""
             ...

         def get_allocation_logs(self, job_id: str) -> str:
             """Fetch recent logs for debugging failures"""
             ...

     2.2 HitL Node with LangGraph Interrupt

     from langgraph.checkpoint.memory import MemorySaver

     # In graph setup:
     checkpointer = MemorySaver()
     app = workflow.compile(checkpointer=checkpointer, interrupt_before=["collect"])

     # Running with HitL:
     config = {"configurable": {"thread_id": "user-session-1"}}
     for event in app.stream(initial_state, config):
         if "__interrupt__" in event:
             # Display questions to user
             questions = event["__interrupt__"]["questions"]
             responses = get_user_input(questions)  # CLI prompt
             app.update_state(config, {"user_responses": responses})

     2.3 Verification Logic

     def verify_deployment(state: AgentState) -> AgentState:
         """
         Poll Nomad for job status.
         Categories:
         - SUCCESS: All allocations healthy for 60s
         - RETRY: Allocation failed with fixable error (resource, port, image)
         - GIVE_UP: Unfixable error (invalid config, auth, cluster issue)
         """
         client = get_nomad_client()

         for attempt in range(12):  # 5 min max (12 * 25s)
             status = client.get_job_status(state["job_id"])

             if status["healthy"]:
                 return {**state, "deployment_status": "success"}

             if status["failed"]:
                 error = categorize_error(status["error"])
                 if error.is_fixable and state["iteration_count"] < state["max_iterations"]:
                     return {**state,
                             "deployment_status": "failed",
                             "deployment_error": error.message}
                 else:
                     return {**state, "deployment_status": "give_up"}

             time.sleep(25)

         return {**state, "deployment_status": "timeout"}

     2.4 Error Categorization

     FIXABLE_ERRORS = [
         ("insufficient memory", "increase memory allocation"),
         ("port already in use", "change port or use dynamic ports"),
         ("image not found", "verify image name and registry"),
         ("constraint not satisfied", "adjust constraints or datacenters"),
     ]

     UNFIXABLE_ERRORS = [
         "permission denied",
         "authentication failed",
         "cluster unreachable",
     ]

     Deliverables Phase 2

     - Real Nomad deployment + verification
     - Interactive CLI for answering questions
     - Max 3 retry iterations with intelligent error analysis
     - Integration tests against real/dev Nomad cluster

     ---
     Phase 3: Memory Layer (Mem0 + Qdrant)

     Goals

     - Persistent memory across sessions
     - Learn from past errors
     - Personalization per cluster/user

     Implementation Steps

     3.1 Mem0 Setup (src/memory/mem0_client.py)

     from mem0 import MemoryClient  # or Memory for self-hosted

     class AgentMemory:
         def __init__(self, qdrant_host: str, collection: str = "nomad_agent"):
             self.mem = Memory(
                 config={
                     "vector_store": {
                         "provider": "qdrant",
                         "config": {
                             "host": qdrant_host,
                             "port": 6333,
                             "collection_name": collection
                         }
                     }
                 }
             )

         def add_deployment_result(self, job_spec: str, error: str, fix: str, user_id: str):
             """Store error → fix mapping for future retrieval"""
             memory_text = f"Job error: {error}\nResolution: {fix}\nSpec context: {job_spec[:500]}"
             self.mem.add(memory_text, user_id=user_id, metadata={"type": "deployment_fix"})

         def search_similar_errors(self, error: str, user_id: str, limit: int = 3) -> List[str]:
             """Find past fixes for similar errors"""
             results = self.mem.search(error, user_id=user_id, limit=limit)
             return [r["memory"] for r in results]

         def add_cluster_preference(self, preference: str, user_id: str):
             """Store cluster-specific settings"""
             self.mem.add(preference, user_id=user_id, metadata={"type": "preference"})

     3.2 Memory-Enhanced Fix Node

     def fix_spec(state: AgentState) -> AgentState:
         memory = get_agent_memory()

         # Search for similar past errors
         similar_fixes = memory.search_similar_errors(
             state["deployment_error"],
             user_id=state.get("cluster_id", "default")
         )

         # Inject memories into fix prompt
         prompt = f"""
         The job deployment failed with error: {state["deployment_error"]}

         Similar past errors and their fixes:
         {chr(10).join(similar_fixes)}

         Current job spec:
         {state["job_spec"]}

         Generate a fixed job spec that addresses this error.
         """

         fixed_spec = llm.invoke(prompt)

         return {**state,
                 "job_spec": fixed_spec,
                 "iteration_count": state["iteration_count"] + 1}

     3.3 Memory Persistence After Success

     def post_success_hook(state: AgentState):
         """Store successful patterns for future use"""
         memory = get_agent_memory()

         if state["iteration_count"] > 0:
             # We had to fix something - store the learning
             memory.add_deployment_result(
                 job_spec=state["job_spec"],
                 error=state["deployment_error"],
                 fix="[successful after fix]",
                 user_id=state.get("cluster_id", "default")
             )

     Deliverables Phase 3

     - Memory client connecting to existing Qdrant
     - Memory search integrated into fix node
     - Learning from successful fixes
     - Memory retrieval in analysis node (cluster preferences)

     ---
     Phase 4: Observability + Prompt Optimization

     Goals

     - LangFuse tracing for all LLM calls
     - Centralized prompt management
     - DSPy optimization pipeline

     Implementation Steps

     4.1 LangFuse Integration (src/observability/langfuse.py)

     from langfuse import Langfuse
     from langfuse.callback import CallbackHandler

     langfuse = Langfuse()

     def get_traced_llm(llm, trace_name: str):
         """Wrap LLM with LangFuse tracing"""
         handler = CallbackHandler(trace_name=trace_name)
         return llm.with_config(callbacks=[handler])

     def get_prompt(name: str, version: int = None) -> str:
         """Fetch prompt from LangFuse"""
         prompt = langfuse.get_prompt(name, version=version)
         return prompt.compile()  # Returns template with placeholders

     4.2 Prompt Templates in LangFuse

     Create these prompts in LangFuse dashboard:

     analyze_codebase (v1):
     Analyze the following codebase files for Docker deployment.

     Files:
     {files}

     Extract:
     1. Base image (from Dockerfile or infer from language)
     2. Exposed ports
     3. Entry command
     4. Environment variables needed
     5. Resource estimates (memory, CPU)
     6. External dependencies (databases, caches, etc.)

     Output as JSON.

     generate_nomad_spec (v1):
     Generate a Nomad job specification in HCL format for a Docker job.

     Codebase Analysis:
     {analysis}

     User Requirements:
     {user_responses}

     Cluster Context:
     {memories}

     Requirements:
     - Use job type "service"
     - Include health checks
     - Set appropriate resource limits
     - Use dynamic ports unless specific port requested
     - Include restart policy

     Output only valid HCL, no explanation.

     4.3 DSPy Optimization (src/prompts/optimizer.py)

     import dspy
     from dspy.teleprompt import BootstrapFewShot

     class SpecGenerator(dspy.Signature):
         """Generate Nomad job specification from analysis."""
         analysis = dspy.InputField(desc="Codebase analysis JSON")
         requirements = dspy.InputField(desc="User requirements")
         spec = dspy.OutputField(desc="Valid Nomad HCL specification")

     class NomadSpecModule(dspy.Module):
         def __init__(self):
             self.generate = dspy.ChainOfThought(SpecGenerator)

         def forward(self, analysis, requirements):
             return self.generate(analysis=analysis, requirements=requirements)

     def optimize_prompts(training_data: List[dict]):
         """
         training_data: [{"analysis": ..., "requirements": ..., "spec": ...}, ...]
         Collected from LangFuse traces of successful deployments
         """
         def metric(example, pred, trace=None):
             # Validate generated spec is syntactically correct
             # and deploys successfully (or matches expected pattern)
             return validate_hcl(pred.spec)

         optimizer = BootstrapFewShot(metric=metric, max_bootstrapped_demos=4)
         optimized = optimizer.compile(NomadSpecModule(), trainset=training_data)
         return optimized

     4.4 Feedback Loop

     def collect_training_data():
         """Export successful generations from LangFuse for DSPy optimization"""
         traces = langfuse.get_traces(
             name="generate_spec",
             filter={"metadata.deployment_status": "success"}
         )

         return [
             {
                 "analysis": t.input["analysis"],
                 "requirements": t.input["requirements"],
                 "spec": t.output
             }
             for t in traces
         ]

     Deliverables Phase 4

     - All LLM calls traced in LangFuse
     - Prompts managed in LangFuse with versioning
     - DSPy optimization script
     - Dashboard for monitoring success rates

     ---
     HCL Generation Strategy

     Since we're Docker-only, the HCL generation is constrained:

     job "{{job_name}}" {
       datacenters = ["{{datacenter}}"]
       type = "service"

       group "{{group_name}}" {
         count = {{count}}

         network {
           port "{{port_name}}" {
             to = {{container_port}}
           }
         }

         task "{{task_name}}" {
           driver = "docker"

           config {
             image = "{{image}}"
             ports = ["{{port_name}}"]
           }

           env {
             {{#each env_vars}}
             {{key}} = "{{value}}"
             {{/each}}
           }

           resources {
             cpu    = {{cpu}}
             memory = {{memory}}
           }

           service {
             name = "{{service_name}}"
             port = "{{port_name}}"

             check {
               type     = "{{check_type}}"
               path     = "{{health_path}}"
               interval = "10s"
               timeout  = "2s"
             }
           }
         }
       }
     }

     The LLM fills in the template values based on analysis + user responses.

     ---
     Testing Strategy

     Unit Tests

     - Each node tested in isolation with mock state
     - Tool functions tested with mock Nomad API
     - HCL validation tested with known good/bad specs

     Integration Tests (Real Cluster)

     import pytest
     from src.tools.nomad import NomadClient

     @pytest.fixture
     def nomad_client():
         """Connect to real Nomad cluster (configured via env vars)"""
         return NomadClient(
             address=os.environ["NOMAD_ADDR"],
             token=os.environ.get("NOMAD_TOKEN")
         )

     @pytest.fixture
     def llm_client():
         """Connect to real vLLM instance"""
         from src.llm.provider import VLLMProvider
         return VLLMProvider(
             base_url=os.environ["VLLM_BASE_URL"],
             model=os.environ["VLLM_MODEL"]
         ).get_llm()

     class TestRealDeployment:
         """Integration tests against real Nomad cluster"""

         def test_simple_nginx_deployment(self, nomad_client, llm_client):
             """Deploy a simple nginx container"""
             result = run_graph(
                 prompt="Deploy nginx web server on port 80",
                 codebase_path="tests/fixtures/sample_repos/nginx-simple",
                 llm=llm_client,
                 nomad=nomad_client,
             )
             assert result["deployment_status"] == "success"
             assert result["job_id"] is not None

             # Cleanup
             nomad_client.stop_job(result["job_id"])

         def test_express_app_with_env_vars(self, nomad_client, llm_client):
             """Deploy Express app, verify env var handling"""
             result = run_graph(
                 prompt="Deploy this Node.js API",
                 codebase_path="tests/fixtures/sample_repos/express-app",
                 llm=llm_client,
                 nomad=nomad_client,
                 user_responses={"NODE_ENV": "production", "PORT": "3000"}
             )
             assert result["deployment_status"] == "success"

             # Cleanup
             nomad_client.stop_job(result["job_id"])

         def test_retry_on_resource_error(self, nomad_client, llm_client):
             """Verify agent retries when initial resources are insufficient"""
             # This test uses a repo that initially requests too little memory
             result = run_graph(
                 prompt="Deploy this memory-hungry service",
                 codebase_path="tests/fixtures/sample_repos/memory-test",
                 llm=llm_client,
                 nomad=nomad_client,
             )
             assert result["deployment_status"] == "success"
             assert result["iteration_count"] >= 1  # Had to retry

             nomad_client.stop_job(result["job_id"])

     @pytest.fixture(autouse=True)
     def cleanup_test_jobs(nomad_client):
         """Cleanup any jobs created during tests"""
         yield
         # After each test, remove jobs with test- prefix
         jobs = nomad_client.list_jobs(prefix="test-")
         for job in jobs:
             nomad_client.stop_job(job["ID"], purge=True)

     Test Fixtures

     Create sample repos in tests/fixtures/sample_repos/:
     - nginx-simple/ - Basic nginx with Dockerfile
     - express-app/ - Node.js with Dockerfile and env vars
     - flask-app/ - Python with requirements.txt and Dockerfile
     - memory-test/ - App that needs specific memory allocation

     CLI Interface Tests

     from typer.testing import CliRunner
     from src.main import app

     runner = CliRunner()

     def test_cli_help():
         result = runner.invoke(app, ["--help"])
         assert result.exit_code == 0
         assert "Deploy" in result.output

     def test_cli_analyze_only():
         """Test analyze mode without deployment"""
         result = runner.invoke(app, [
             "--prompt", "Analyze this repo",
             "--path", "tests/fixtures/sample_repos/express-app",
             "--dry-run"
         ])
         assert result.exit_code == 0
         assert "Analysis complete" in result.output

     ---
     Configuration

     # config/settings.py
     from pydantic_settings import BaseSettings

     class Settings(BaseSettings):
         # LLM
         llm_provider: str = "vllm"  # vllm, openai, anthropic
         vllm_base_url: str = "http://localhost:8000"
         vllm_model: str = "Qwen/Qwen3-32B"

         # Nomad
         nomad_addr: str = "http://localhost:4646"
         nomad_token: str | None = None
         nomad_namespace: str = "default"

         # Memory
         qdrant_host: str = "localhost"
         qdrant_port: int = 6333

         # Observability (both keys required when enabled)
         langfuse_enabled: bool = False
         langfuse_public_key: str | None = None
         langfuse_secret_key: str | None = None
         langfuse_base_url: str = "https://cloud.langfuse.com"

         # Agent
         max_iterations: int = 3
         verification_timeout: int = 300  # seconds

         class Config:
             env_file = ".env"

     ---
     Risk Mitigation

     | Risk                       | Mitigation                                         |
     |----------------------------|----------------------------------------------------|
     | LLM generates invalid HCL  | Validate with nomad job validate before deployment |
     | Infinite retry loops       | Hard cap at 3 iterations + error classification    |
     | Qdrant unavailable         | Graceful degradation - agent works without memory  |
     | vLLM latency               | Timeout handling + async generation                |
     | Secrets in generated specs | Warning system + scan for common secret patterns   |

     ---
     Implementation Order

     1. Week 1: Phase 1 - Core graph + LLM integration
     2. Week 2: Phase 2 - Nomad integration + HitL
     3. Week 3: Phase 3 - Memory layer
     4. Week 4: Phase 4 - Observability + optimization
     5. Week 5: Testing, documentation, refinement

     ---
     Files to Create

     | File                          | Priority | Phase |
     |-------------------------------|----------|-------|
     | pyproject.toml                | P0       | 1     |
     | src/main.py                   | P0       | 1     |
     | src/graph.py                  | P0       | 1     |
     | src/llm/provider.py           | P0       | 1     |
     | src/nodes/analyze.py          | P0       | 1     |
     | src/nodes/generate.py         | P0       | 1     |
     | src/tools/codebase.py         | P0       | 1     |
     | src/tools/hcl.py              | P0       | 1     |
     | config/settings.py            | P0       | 1     |
     | src/nodes/question.py         | P1       | 2     |
     | src/nodes/deploy.py           | P1       | 2     |
     | src/nodes/verify.py           | P1       | 2     |
     | src/nodes/fix.py              | P1       | 2     |
     | src/tools/nomad.py            | P1       | 2     |
     | src/memory/mem0_client.py     | P2       | 3     |
     | src/observability/langfuse.py | P3       | 4     |
     | src/prompts/templates.py      | P3       | 4     |
     | src/prompts/optimizer.py      | P3       | 4     |

     ---
     Success Criteria

     1. Functional: Given a repo with Dockerfile, generate valid Nomad HCL and deploy successfully
     2. Iterative: Automatically retry failed deployments up to 3 times with fixes
     3. Learning: Past errors inform future generations via Mem0
     4. Observable: All LLM calls visible in LangFuse
     5. Maintainable: Clean separation of concerns, comprehensive tests



 Phase 1 Complete!

  I've successfully implemented the foundation of the Nomad Job Spec Agent. Here's a summary:

  Files Created (Phase 1)

  | File                  | Purpose                                        |
  |-----------------------|------------------------------------------------|
  | pyproject.toml        | Project dependencies and configuration         |
  | config/settings.py    | Pydantic settings for all configuration        |
  | src/llm/provider.py   | LLM abstraction (vLLM, OpenAI, Anthropic)      |
  | src/tools/codebase.py | Codebase analysis (Dockerfile, deps, env vars) |
  | src/tools/hcl.py      | HCL generation and validation                  |
  | src/nodes/analyze.py  | Analysis LangGraph node                        |
  | src/nodes/generate.py | Generation LangGraph node                      |
  | src/graph.py          | LangGraph workflow definition                  |
  | src/main.py           | CLI entry point (Typer)                        |
  | tests/                | 89 passing unit tests                          |

  What Works Now

  # Analyze a codebase
  nomad-spec analyze --path ./my-app

  # Generate a job specification
  nomad-spec generate -p "Deploy this API" --path ./my-app -o job.nomad

  # Validate HCL
  nomad-spec validate job.nomad

  Test Results

  89 passed in 1.33s

  Next Steps (Phase 2-4)

  The plan includes:
  - Phase 2: Real Nomad deployment + HitL verification loop (src/tools/nomad.py, src/nodes/deploy.py, src/nodes/verify.py)
  - Phase 3: Mem0 memory layer (src/memory/mem0_client.py)
  - Phase 4: LangFuse observability + DSPy optimization (src/observability/langfuse.py, src/prompts/optimizer.py)

  Would you like me to continue with Phase 2 (Nomad integration and deployment)?


 I've completed the integration of cluster-specific knowledge from data/nomad-job-specification.md. Here's a summary of what was
  accomplished:

  Cluster Integration Complete

  1. HCL Generation (src/tools/hcl.py)

  - Added new dataclasses: ServiceType, NetworkMode, PortConfig, VolumeConfig, VaultConfig, FabioRoute
  - Updated JobConfig with cluster-specific fields:
    - use_terraform_datacenter: bool - Uses ["${datacenter}"] variable
    - require_amd64: bool - Architecture constraint for AMD64 nodes
    - network_mode: NetworkMode - HOST mode for static ports < 1024
    - service_type: ServiceType - LIGHT/MEDIUM/HEAVY/COMPUTE classification
    - CSI volume support with init tasks for permissions
    - Fabio routing tags (urlprefix-hostname:9999/)
    - Vault integration with custom delimiters
  - Generates HCL with proper $${} escaping for Nomad runtime variables

  2. Analysis Prompts (src/nodes/analyze.py)

  - Updated to include cluster environment details (mixed ARM64/AMD64 architecture)
  - Classifies apps into service categories with appropriate resource defaults
  - Asks about storage requirements, architecture needs, and external routing

  3. Generation Prompts (src/nodes/generate.py)

  - Updated to use cluster-aware JSON schema with service types
  - Handles CSI volumes, Fabio routes, and Vault secrets
  - Supports both legacy dict port format and new PortConfig list format

  4. New Test Fixtures

  - postgres-db/ - Database with volume requirements (PostgreSQL)
  - flask-api/ - Python API with env vars and secrets

  5. Test Coverage

  - 99 tests passing
  - New tests for CSI volumes, Fabio routing, Vault integration, static ports
  - Tests for new sample repos (postgres-db, flask-api)

  The HCL generator now produces cluster-compatible job specs matching the patterns in your documentation, including Terraform
  templating, architecture constraints, and proper handling of stateful services.

● I've completed both tasks. Here's a summary:

  Documentation and Build System Complete

  README.md Updates

  The README now includes:
  - Comprehensive feature list with cluster-specific capabilities
  - Installation instructions (from source and production)
  - Quick start guide with examples for analyze, generate, and validate commands
  - Generated HCL examples showing:
    - Terraform templating
    - Architecture constraints
    - CSI volumes with init tasks
    - Vault integration
    - Fabio routing
  - Service type classification table
  - Full configuration reference
  - Development setup and project structure
  - API usage examples for programmatic access
  - Troubleshooting section

  Makefile

  The Makefile provides these targets:

  | Target                      | Description                   |
  |-----------------------------|-------------------------------|
  | make install                | Install the package           |
  | make install-dev            | Install with dev dependencies |
  | make test                   | Run tests                     |
  | make test-cov               | Run tests with coverage       |
  | make lint                   | Run ruff linting              |
  | make format                 | Format code                   |
  | make typecheck              | Run mypy type checking        |
  | make clean                  | Remove build artifacts        |
  | make build                  | Build wheel and sdist         |
  | make publish                | Publish to PyPI               |
  | make publish-test           | Publish to TestPyPI           |
  | make version                | Show current version          |
  | make bump-patch/minor/major | Bump version                  |

  To Publish

  # Install build/publish dependencies
  make publish-deps

  # Build the packages
  make build

  # Publish to TestPyPI first (recommended)
  make publish-test

  # Or publish directly to PyPI
  make publish

  The package builds successfully and produces:
  - nomad_job_spec-0.1.0-py3-none-any.whl (wheel)
  - nomad_job_spec-0.1.0.tar.gz (source distribution)

All changes have been committed. The untracked directories are:
  - dist/ - build artifacts (should be gitignored)
  - docs/ - your original PRD documentation
  - notes/ - any notes you have

  Would you like me to:
  1. Add a .gitignore file to exclude dist/, __pycache__/, etc.?
  2. Add the docs/ folder to the repository?

  Here's the commit summary:

  [master (root-commit) 28ea46f] Initial commit: Nomad Job Spec Agent v0.1.0
   41 files changed, 7607 insertions(+)


● Yes, the configuration will fetch values from both the .env file and environment variables. Here's how it works:

  The project uses pydantic_settings.BaseSettings (see config/settings.py:10-18), which has the following resolution order:

  1. Environment variables (highest priority)
  2. .env file values
  3. Default values defined in the Field() declarations

  This is standard Pydantic Settings behavior. The SettingsConfigDict at lines 13-18 specifies:
  - env_file=".env" - load from .env file if present
  - case_sensitive=False - environment variable names are case-insensitive

  So if you set NOMAD_ADDR=http://production:4646 in your shell environment, it will override any value in the .env file. This allows you to:

  - Use .env for local development defaults
  - Override with actual environment variables in production (e.g., in Docker, systemd, or Kubernetes)


Here is Claude's plan:
╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌
 Plan: Interactive Chat Flow for Nomad Job Spec Generator

 Goal

 Change the CLI to ask for the deployment prompt interactively after analyzing the codebase, rather than requiring it as a CLI argument.

 Desired user experience:
 $ nomad-spec generate --path . -o job.nomad
 # CLI analyzes codebase
 # Shows analysis summary
 # Asks: "What would you like to deploy?"
 # Asks follow-up questions about env vars, ports, resources, etc.

 Files to Modify

 | File         | Changes                                                              |
 |--------------|----------------------------------------------------------------------|
 | src/main.py  | Make -p optional, add _collect_deployment_prompt(), restructure flow |
 | src/graph.py | Make prompt optional in create_initial_state() and run_graph()       |

 Implementation

 1. src/main.py - CLI Changes

 1.1 Make --prompt/-p optional (line 29-33)

 # Change from:
 prompt: str = typer.Option(
     ...,  # Required
     "--prompt", "-p",
     help="Deployment request (e.g., 'Deploy a Node.js API')",
 )

 # Change to:
 prompt: Optional[str] = typer.Option(
     None,  # Optional
     "--prompt", "-p",
     help="Deployment request. If not provided, will be asked interactively after analysis.",
 )

 1.2 Add _collect_deployment_prompt() function (after _collect_user_responses)

 def _collect_deployment_prompt(analysis: dict) -> str:
     """Display analysis summary and collect deployment prompt from user."""
     console.print("\n[bold]Codebase Analysis Complete[/bold]\n")

     # Build summary table
     table = Table(title="Analysis Summary")
     table.add_column("Property", style="cyan")
     table.add_column("Value", style="green")

     dockerfile = analysis.get("dockerfile", {})
     if dockerfile:
         table.add_row("Base Image", dockerfile.get("base_image") or "Not found")
         ports = dockerfile.get("exposed_ports", [])
         if ports:
             table.add_row("Exposed Ports", ", ".join(map(str, ports)))

     deps = analysis.get("dependencies", {})
     if deps:
         table.add_row("Language", deps.get("language") or "Unknown")

     env_vars = analysis.get("env_vars_required", [])[:5]
     if env_vars:
         table.add_row("Env Vars Detected", ", ".join(env_vars))

     resources = analysis.get("suggested_resources", {})
     table.add_row("Suggested CPU", f"{resources.get('cpu', 500)} MHz")
     table.add_row("Suggested Memory", f"{resources.get('memory', 256)} MB")

     console.print(table)
     console.print()

     return Prompt.ask(
         "[bold cyan]What would you like to deploy?[/bold cyan]",
         default="Deploy this application"
     )

 1.3 Restructure generate() function flow (lines 85-145)

 The key changes:
 1. Move the configuration panel to after we have the prompt
 2. If no prompt provided, collect it interactively after analysis completes
 3. Update state with the prompt before continuing

 # After validating path and initializing LLM (around line 108):

 # Create initial state (prompt may be empty)
 state = create_initial_state(
     prompt=prompt or "",  # Empty if not provided
     codebase_path=str(codebase_path.absolute()) if not is_git_url else path,
     cluster_id=cluster_id,
     max_iterations=settings.max_iterations,
 )

 # Run graph with HitL for questions
 config = {"configurable": {"thread_id": f"session-{cluster_id}"}}

 try:
     # First run - analyze and generate questions
     with console.status("[bold green]Analyzing codebase..."):
         for event in graph.stream(state, config):
             if verbose:
                 _print_event(event)
             if "__interrupt__" in event:
                 break

     # Get current state with analysis
     current_state = graph.get_state(config)
     analysis = current_state.values.get("codebase_analysis", {})

     # If no prompt provided, collect it interactively
     if not prompt and not no_questions:
         prompt = _collect_deployment_prompt(analysis)
         graph.update_state(config, {"prompt": prompt})

     # Now display configuration panel (prompt is known)
     console.print(Panel(
         f"[bold]Nomad Job Spec Generator[/bold]\n\n"
         f"Prompt: {prompt or 'Deploy this application'}\n"
         f"Codebase: {path}\n"
         f"Cluster: {cluster_id}",
         title="Configuration",
     ))

     # Continue with existing question collection flow...

 2. src/graph.py - State Changes

 2.1 Update create_initial_state() signature (line 57-62)

 def create_initial_state(
     codebase_path: str,
     prompt: str = "",  # Now optional, moved after required param
     cluster_id: str = "default",
     max_iterations: int = 3,
 ) -> dict[str, Any]:

 2.2 Update run_graph() signature (line 306-314)

 def run_graph(
     codebase_path: str,
     prompt: str = "",  # Now optional
     llm: BaseChatModel,
     # ... rest unchanged
 )

 Note: Also update the call to create_initial_state() inside run_graph() at line 335.

 3. Handle --no-questions Edge Case

 When --no-questions is used without -p, use a default prompt:

 # In generate(), after checking no_questions:
 if no_questions and not prompt:
     prompt = "Deploy this application"

 Implementation Order

 1. src/graph.py: Make prompt optional in create_initial_state() and run_graph()
 2. src/main.py: Add _collect_deployment_prompt() function
 3. src/main.py: Make --prompt/-p optional in CLI
 4. src/main.py: Restructure generate() flow to handle interactive prompt
 5. Test manually with: nomad-spec generate --path .

 Testing Scenarios

 1. Interactive mode: nomad-spec generate --path . → Should analyze, show summary, ask for prompt
 2. CLI prompt: nomad-spec generate -p "Deploy API" --path . → Should work as before
 3. No questions: nomad-spec generate --path . --no-questions → Should use default prompt
 4. With output: nomad-spec generate --path . -o job.nomad → Should work with interactive prompt
╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌

 Would you like to proceed?

 Summary

  I've implemented the changes to make the CLI ask for the deployment prompt interactively after analyzing the codebase:

  Changes made:

  1. src/graph.py:
    - Made prompt optional in create_initial_state() (default: "")
    - Made prompt optional in run_graph() (default: "")
    - Reordered parameters so codebase_path (required) comes first
  2. src/main.py:
    - Changed --prompt/-p from required to optional
    - Added _collect_deployment_prompt() function that displays an analysis summary table and asks "What would you like to deploy?"
    - Restructured generate() flow:
        - If prompt provided on CLI: shows config panel immediately, runs as before
      - If no prompt: analyzes codebase first, shows analysis summary, asks for prompt interactively, then shows config panel and continues with follow-up
  questions
    - Added handling for --no-questions without prompt (uses default "Deploy this application")

  New user experience:
  $ nomad-spec generate --path .
  [Analyzing codebase...]

  Codebase Analysis Complete

  ┌─ Analysis Summary ─────────────────────────────┐
  │ Property         │ Value                       │
  ├──────────────────┼─────────────────────────────┤
  │ Base Image       │ python:3.11-slim            │
  │ Exposed Ports    │ 8080                        │
  │ Language         │ python                      │
  │ Env Vars Detected│ AWS_KEY, DATABASE_URL       │
  │ Suggested CPU    │ 500 MHz                     │
  │ Suggested Memory │ 256 MB                      │
  └──────────────────┴─────────────────────────────┘

  What would you like to deploy? [Deploy this application]: _

  ╭─── Configuration ───╮
  │ Prompt: ...         │
  │ Codebase: .         │
  │ Cluster: default    │
  ╰─────────────────────╯

  Please answer the following questions:
  1. Environment variables...

Plan: Vault, Consul, and Fabio Tool Integration

 Summary

 Add full Vault, Consul, and Fabio integration to the Nomad Job Spec Agent, enabling:
 - Automatic Vault secret path suggestions and validation
 - Consul KV-based conventions storage and service discovery
 - Fabio route conflict detection (strict mode - blocks deployment)
 - Support for both Nomad template blocks and 1.4+ native env stanza

 User Requirements

 - Conventions: Stored in Consul KV at config/nomad-agent/conventions
 - Vault Auth: Token-based only (VAULT_TOKEN env var)
 - Fabio Conflicts: Strict mode - block deployment on conflicts
 - Secret Format: Support both template blocks and Nomad 1.4+ native vault env stanza

 ---
 New Files to Create

 1. src/tools/vault.py - Vault API Client

 @dataclass
 class VaultSecretMetadata:
     path: str
     keys: list[str]
     version: int | None

 @dataclass
 class VaultPathSuggestion:
     env_var: str
     suggested_path: str
     key: str
     confidence: float

 class VaultClient:
     def __init__(self, addr: str, token: str, namespace: str | None = None)
     def list_secrets(self, path: str) -> list[str]
     def read_metadata(self, path: str) -> VaultSecretMetadata | None
     def validate_path(self, path: str) -> tuple[bool, str | None]

 # LangChain tools
 @tool def list_vault_secrets(path: str) -> str
 @tool def suggest_vault_mappings(env_vars: list[str], app_name: str) -> str
 @tool def validate_vault_paths(paths: dict[str, str]) -> str

 Dependencies: hvac>=2.1.0

 2. src/tools/consul.py - Consul Client

 @dataclass
 class ConsulService:
     name: str
     address: str
     port: int
     tags: list[str]
     health_status: str

 class ConsulClient:
     def __init__(self, addr: str, token: str | None = None)
     def get_kv(self, key: str) -> dict | None
     def list_kv(self, prefix: str) -> list[dict]
     def get_conventions(self) -> dict  # From config/nomad-agent/conventions
     def list_services() -> dict[str, list[str]]
     def get_service_health(name: str) -> list[ConsulService]

 # LangChain tools
 @tool def query_consul_kv(key: str) -> str
 @tool def list_consul_services() -> str
 @tool def get_service_endpoints(service_name: str) -> str

 Dependencies: python-consul>=1.1.0

 3. src/tools/fabio.py - Fabio Route Validation

 @dataclass
 class FabioRouteEntry:
     service: str
     src: str
     dst: str
     weight: float

 @dataclass
 class RouteConflict:
     existing_route: FabioRouteEntry
     proposed_route: str
     conflict_type: str  # exact_match, path_overlap, hostname_collision

 class FabioRouteConflictError(Exception):
     """Raised when strict mode detects a route conflict."""
     pass

 class FabioClient:
     def __init__(self, addr: str = "http://localhost:9998")
     def get_routes(self) -> list[FabioRouteEntry]
     def check_route_conflict(self, hostname: str | None, path: str | None) -> RouteConflict | None

 # LangChain tools
 @tool def list_fabio_routes() -> str
 @tool def validate_fabio_route(hostname: str, path: str) -> str

 Dependencies: httpx>=0.27.0

 4. src/tools/nomad_version.py - Version Detection

 @dataclass
 class NomadVersion:
     major: int
     minor: int
     patch: int

     def supports_native_vault_env(self) -> bool:
         return (self.major, self.minor) >= (1, 4)

 def get_nomad_version(addr: str, token: str | None) -> NomadVersion
 def detect_vault_format(addr: str, token: str | None) -> str  # 'env_stanza' or 'template'

 5. src/nodes/enrich.py - Enrichment Node

 New workflow node that runs after analyze and before question:
 - Queries Vault for available secret paths matching detected env vars
 - Loads conventions from Consul KV
 - Checks Fabio for existing routes on proposed hostname
 - Adds suggestions to state for use in question generation

 6. src/nodes/validate.py - Pre-Deploy Validation

 New workflow node that runs after generate and before deploy:
 - Validates Fabio routes don't conflict (raises FabioRouteConflictError in strict mode)
 - Validates Vault paths are accessible
 - Returns validation errors to block deployment

 ---
 Files to Modify

 1. config/settings.py

 Add new settings:
 # Vault
 vault_addr: str = Field(default="http://localhost:8200")
 vault_namespace: str | None = Field(default=None)

 # Consul
 consul_http_addr: str = Field(default="http://localhost:8500")
 consul_http_token: str | None = Field(default=None)
 consul_conventions_path: str = Field(default="config/nomad-agent/conventions")

 # Fabio
 fabio_admin_addr: str = Field(default="http://localhost:9998")

 2. src/tools/hcl.py

 Enhance VaultConfig:
 @dataclass
 class VaultConfig:
     policies: list[str]
     secrets: dict[str, str]  # env_var -> vault_path
     use_custom_delimiters: bool = False
     use_native_env: bool = False  # NEW: Nomad 1.4+ env stanza

 Add _build_vault_env_stanza() function for native format:
 vault {
   env {
     DB_PASSWORD = "secret/data/myapp/db#password"
   }
 }

 3. src/graph.py

 Update workflow:
 START -> analyze -> enrich -> question -> collect -> generate -> validate -> deploy -> verify

 Add to AgentState:
 vault_suggestions: dict[str, Any]
 consul_services: dict[str, Any]
 fabio_validation: dict[str, Any]

 4. src/nodes/analyze.py

 Update ANALYSIS_SYSTEM_PROMPT to identify:
 - Env vars that likely need secrets (password, key, token, secret patterns)
 - Output vault_suggestions with confidence scores

 5. src/nodes/generate.py

 Add Nomad version detection to auto-select Vault format:
 nomad_version = get_nomad_version(settings.nomad_addr)
 use_native_vault = nomad_version.supports_native_vault_env()

 6. pyproject.toml

 Add dependencies:
 "hvac>=2.1.0",
 "python-consul>=1.1.0",
 "httpx>=0.27.0",

 ---
 Consul KV Convention Structure

 Store at config/nomad-agent/conventions:

 {
   "vault": {
     "path_patterns": {
       "database": "secret/data/{app_name}/db",
       "redis": "secret/data/{app_name}/redis",
       "api_keys": "secret/data/{app_name}/app",
       "aws": "secret/data/aws/{app_name}"
     },
     "env_var_mappings": {
       "DB_PASSWORD": {"path": "db", "key": "password"},
       "AWS_ACCESS_KEY_ID": {"path": "aws", "key": "access_key"},
       "AWS_SECRET_ACCESS_KEY": {"path": "aws", "key": "secret_key"}
     }
   },
   "fabio": {
     "default_port": 9999,
     "hostname_suffix": ".cluster"
   }
 }

 ---
 Updated Workflow

 START
   │
   ▼
 ┌─────────┐
 │ analyze │  Detect env vars, Dockerfile, dependencies
 └────┬────┘
      │
      ▼
 ┌─────────┐
 │ enrich  │  Query Vault paths, load Consul conventions,
 └────┬────┘  check Fabio routes - add suggestions to state
      │
      ▼
 ┌──────────┐
 │ question │  Include Vault path suggestions in questions
 └────┬─────┘  "AWS keys detected - use secret/data/aws/transcription?"
      │
      ▼
 ┌─────────┐
 │ collect │  Human-in-the-loop - user confirms/modifies paths
 └────┬────┘
      │
      ▼
 ┌──────────┐
 │ generate │  Auto-detect Nomad version, use appropriate
 └────┬─────┘  Vault format (template vs native env stanza)
      │
      ▼
 ┌──────────┐
 │ validate │  Check Fabio conflicts (STRICT - block on conflict)
 └────┬─────┘  Validate Vault paths accessible
      │
      ├── FAIL ──▶ Return validation errors
      │
      ▼
 ┌────────┐
 │ deploy │
 └────┬───┘
      │
      ▼
    END

 ---
 Implementation Order

 Phase 1: Core Clients

 1. src/tools/vault.py - VaultClient with hvac
 2. src/tools/consul.py - ConsulClient with python-consul
 3. src/tools/fabio.py - FabioClient with httpx
 4. src/tools/nomad_version.py - Version detection
 5. Update config/settings.py
 6. Update pyproject.toml

 Phase 2: LangChain Tools

 1. Add @tool functions to each client module
 2. Update src/tools/__init__.py to export tools
 3. Write unit tests

 Phase 3: Workflow Nodes

 1. src/nodes/enrich.py - Enrichment node
 2. src/nodes/validate.py - Validation node
 3. Update src/graph.py with new nodes
 4. Update AgentState

 Phase 4: HCL Enhancement

 1. Update src/tools/hcl.py for native vault env stanza
 2. Update src/nodes/generate.py for auto-detection
 3. Update analysis prompt for vault suggestions

 Phase 5: Testing

 1. Unit tests for all new clients
 2. Integration tests with mock services
 3. Update .env.example

 ---
 Critical Files Summary

 | File                       | Action | Purpose                                  |
 |----------------------------|--------|------------------------------------------|
 | src/tools/vault.py         | Create | Vault API client, secret path validation |
 | src/tools/consul.py        | Create | Consul KV, service discovery             |
 | src/tools/fabio.py         | Create | Route conflict detection                 |
 | src/tools/nomad_version.py | Create | Auto-detect Nomad features               |
 | src/nodes/enrich.py        | Create | Infrastructure enrichment node           |
 | src/nodes/validate.py      | Create | Pre-deploy validation                    |
 | config/settings.py         | Modify | Add Vault/Consul/Fabio settings          |
 | src/tools/hcl.py           | Modify | Native vault env stanza support          |
 | src/graph.py               | Modify | Add enrich/validate nodes                |
 | src/nodes/generate.py      | Modify | Auto-select vault format                 |
 | pyproject.toml             | Modify | Add hvac, python-consul, httpx           |
╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌

 Would you like to proceed?

  Summary

  I've successfully implemented full Vault, Consul, and Fabio integration for the Nomad Job Spec agent. Here's what was added:

  New Files Created

  1. src/tools/vault.py - Vault API client with:
    - VaultClient class for querying secret metadata and validating paths
    - VaultConventions for convention-based path suggestions
    - LangChain tools: list_vault_secrets, suggest_vault_mappings, validate_vault_paths, get_vault_secret_keys
  2. src/tools/consul.py - Consul client with:
    - ConsulClient class for KV store and service discovery
    - Conventions storage/retrieval from Consul KV at config/nomad-agent/conventions
    - LangChain tools: query_consul_kv, list_consul_services, get_service_endpoints, get_agent_conventions, check_service_dependency
  3. src/tools/fabio.py - Fabio route validation with:
    - FabioClient class for querying routing table
    - FabioRouteConflictError for strict mode validation
    - LangChain tools: list_fabio_routes, validate_fabio_route, get_existing_hostnames, suggest_fabio_hostname, get_service_routes
  4. src/tools/nomad_version.py - Nomad version detection with:
    - NomadVersion dataclass with feature detection methods
    - Auto-detection of Nomad 1.4+ for native vault env stanza support
  5. src/nodes/enrich.py - New workflow node that:
    - Queries Vault for secret path suggestions based on detected env vars
    - Loads conventions from Consul KV
    - Checks Fabio for route availability
    - Detects Nomad version for feature selection
    - Gracefully handles unavailable services
  6. src/nodes/validate.py - Pre-deployment validation node that:
    - Validates Fabio routes in STRICT mode (blocks deployment on conflicts)
    - Validates Vault path accessibility
    - Returns validation errors to prevent bad deployments

  Files Modified

  1. config/settings.py - Added settings for Vault, Consul, and Fabio addresses
  2. pyproject.toml - Added hvac, python-consul, httpx dependencies
  3. src/graph.py - Updated workflow with enrich and validate nodes, added new state fields
  4. src/tools/hcl.py - Added support for Nomad 1.4+ native vault { env { } } stanza
  5. src/nodes/generate.py - Enhanced context with Vault suggestions and Nomad version info

  Updated Workflow

  START -> analyze -> enrich -> question -> collect -> generate -> validate -> deploy -> verify

  Key Features

  - Vault Secrets: When you see env vars like AWS_ACCESS_KEY_ID, you can now answer "use secret/data/aws/transcription" and the agent will:
    - Validate the path exists in Vault
    - Generate proper Nomad job specs with either template blocks OR native env stanza (auto-detected based on Nomad version)
  - Conventions: Stored in Consul KV at config/nomad-agent/conventions for centralized management
  - Strict Mode: Fabio route conflicts block deployment
  - Resilient Design: All infrastructure queries are wrapped in try/except so tests and development work even without Vault/Consul/Fabio running



