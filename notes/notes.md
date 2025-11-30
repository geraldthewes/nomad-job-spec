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
         nomad_address: str = "http://localhost:4646"
         nomad_token: str | None = None
         nomad_namespace: str = "default"

         # Memory
         qdrant_host: str = "localhost"
         qdrant_port: int = 6333

         # Observability
         langfuse_public_key: str
         langfuse_secret_key: str
         langfuse_host: str = "https://cloud.langfuse.com"

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

