"""LangGraph workflow definition for the Nomad Job Spec Agent."""

from typing import Any, Annotated, Literal, NotRequired, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

from src.nodes.analyze_build_system import create_analyze_build_system_node
from src.nodes.confirm_dockerfile import (
    create_confirm_node,
    should_confirm_dockerfile,
)
from src.nodes.discover_sources import create_discover_sources_node
from src.nodes.extract import create_extract_node
from src.nodes.merge import create_merge_node
from src.nodes.generate import create_generate_node, create_fix_node
from src.nodes.validate import create_validate_node, should_proceed_after_validation
from src.nodes.question import create_question_node, create_collect_node
from src.nodes.deploy import create_deploy_node, create_verify_node
from src.subgraphs.analysis import create_analysis_subgraph_node
from config.settings import Settings, get_settings


class AgentState(TypedDict, total=False):
    """State for the Nomad Job Spec Agent.

    Uses TypedDict for proper type checking and LangGraph integration.
    Fields marked with NotRequired are optional and may not be present.
    """

    # Input (required)
    prompt: str
    codebase_path: str

    # Dockerfile selection (pre-analysis)
    dockerfiles_found: list[str]
    selected_dockerfile: str | None

    # Analysis
    codebase_analysis: dict[str, Any]

    # Conversation - messages uses add_messages reducer for proper merging
    messages: Annotated[list[BaseMessage], add_messages]
    questions: list[str]
    user_responses: dict[str, str]

    # Generation
    job_spec: str  # HCL content
    job_config: dict[str, Any]
    job_name: str
    hcl_valid: bool
    validation_error: str | None

    # Deployment
    job_id: str | None
    deployment_status: str  # pending, deploying, success, failed, give_up
    deployment_error: str | None

    # Iteration
    iteration_count: int
    max_iterations: int

    # Memory
    relevant_memories: list[str]
    cluster_id: str

    # Infrastructure enrichment (from enrich node)
    app_name: str  # Extracted app name (set by enrich node)
    env_var_configs: list[dict[str, Any]]  # Multi-source env var configurations
    vault_suggestions: dict[str, Any]  # Legacy, for backward compatibility
    consul_conventions: dict[str, Any]
    consul_services: dict[str, Any]
    fabio_validation: dict[str, Any]
    nomad_info: dict[str, Any]
    infra_issues: list[dict[str, str]]  # Infrastructure connection issues

    # Pre-deployment validation
    pre_deploy_validation: dict[str, Any]

    # Source discovery and extraction (from discover_sources, extract, merge nodes)
    discovered_sources: dict[str, str]  # source_type -> file_path
    build_system_analysis: dict[str, Any]  # Analysis of how images are built
    extractions: list[dict[str, Any]]  # List of extraction results
    merged_extraction: dict[str, Any]  # Combined extraction with priority
    extraction_sources: dict[str, dict[str, Any]]  # Field -> source attribution

    # Port configurability analysis (from analyze_ports node)
    port_analysis: dict[str, Any]  # Port env var mappings and warnings

    # Workload classification (from classify_workload node)
    workload_classification: dict[str, Any]  # workload_type, confidence, evidence


def create_initial_state(
    codebase_path: str,
    prompt: str = "",
    cluster_id: str = "default",
    max_iterations: int = 3,
) -> dict[str, Any]:
    """Create the initial state for the agent.

    Args:
        codebase_path: Path to the codebase to analyze.
        prompt: User's deployment request. If not provided, can be set later interactively.
        cluster_id: Identifier for the Nomad cluster (for memory).
        max_iterations: Maximum fix iterations allowed.

    Returns:
        Initial state dictionary.
    """
    return {
        "prompt": prompt,
        "codebase_path": codebase_path,
        # Dockerfile selection (pre-analysis)
        "dockerfiles_found": [],
        "selected_dockerfile": None,
        # Analysis
        "codebase_analysis": {},
        "messages": [],
        "questions": [],
        "user_responses": {},
        "job_spec": "",
        "job_config": {},
        "job_name": "",
        "hcl_valid": False,
        "validation_error": None,
        "job_id": None,
        "deployment_status": "pending",
        "deployment_error": None,
        "iteration_count": 0,
        "max_iterations": max_iterations,
        "relevant_memories": [],
        "cluster_id": cluster_id,
        # Infrastructure enrichment
        "env_var_configs": [],
        "vault_suggestions": {},
        "consul_conventions": {},
        "consul_services": {},
        "fabio_validation": {},
        "nomad_info": {},
        # Validation
        "pre_deploy_validation": {},
        # Source discovery and extraction
        "discovered_sources": {},
        "build_system_analysis": {},
        "extractions": [],
        "merged_extraction": {},
        "extraction_sources": {},
        # Port configurability analysis
        "port_analysis": {},
        # Workload classification
        "workload_classification": {},
    }


def should_retry(state: dict[str, Any]) -> Literal["retry", "success", "give_up"]:
    """Determine if we should retry after verification.

    Args:
        state: Current graph state.

    Returns:
        "retry" to fix and retry, "success" if deployment succeeded, "give_up" if we should stop.
    """
    status = state.get("deployment_status", "pending")

    if status == "success":
        return "success"

    if status in ("give_up", "timeout"):
        return "give_up"

    # Check iteration count
    iteration = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 3)

    if iteration >= max_iterations:
        return "give_up"

    # Check if error is fixable
    error = state.get("deployment_error", "")
    if _is_unfixable_error(error):
        return "give_up"

    return "retry"


def _is_unfixable_error(error: str) -> bool:
    """Check if an error is unfixable and we should give up."""
    unfixable_patterns = [
        "permission denied",
        "authentication failed",
        "unauthorized",
        "cluster unreachable",
        "no route to host",
        "connection refused",  # Cluster not reachable
        "invalid acl token",
    ]

    error_lower = error.lower()
    return any(pattern in error_lower for pattern in unfixable_patterns)


def create_workflow(
    llm: BaseChatModel,
    settings: Settings | None = None,
    include_deployment: bool = True,
) -> StateGraph:
    """Create the LangGraph workflow for job spec generation.

    Workflow with source discovery and extraction:
    START -> discover_sources -> analyze_build_system -> [confirm] -> extract -> merge -> analyze_ports -> analyze -> enrich -> question -> collect -> generate -> validate -> deploy -> verify

    The analyze_build_system node uses LLM to understand how images are built AND
    identifies which Dockerfile is used. The confirm node then lets the user confirm
    or override this finding.
    The extract/merge nodes run extractors on discovered sources (build.yaml, Makefile, etc.).
    The analyze_ports node determines port configurability for Nomad dynamic port allocation.

    Args:
        llm: LLM instance for analysis and generation.
        settings: Application settings.
        include_deployment: Whether to include deployment nodes (Phase 2).

    Returns:
        Compiled LangGraph workflow.
    """
    if settings is None:
        settings = get_settings()

    # Create node functions
    discover_sources_node = create_discover_sources_node()
    confirm_node = create_confirm_node()
    analyze_build_system_node = create_analyze_build_system_node(llm)
    extract_node = create_extract_node()
    merge_node = create_merge_node()
    # Analysis subgraph replaces: analyze_ports, analyze, enrich
    analysis_node = create_analysis_subgraph_node(llm, settings)
    question_node = create_question_node()
    collect_node = create_collect_node()
    generate_node = create_generate_node(llm)
    validate_node = create_validate_node(settings)
    fix_node = create_fix_node(llm)

    # Build graph with typed state
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("discover_sources", discover_sources_node)
    workflow.add_node("analyze_build_system", analyze_build_system_node)
    workflow.add_node("confirm", confirm_node)
    workflow.add_node("extract", extract_node)
    workflow.add_node("merge", merge_node)
    # Analysis subgraph (encapsulates: analyze_ports -> analyze -> enrich)
    workflow.add_node("analysis", analysis_node)
    workflow.add_node("question", question_node)
    workflow.add_node("collect", collect_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("validate", validate_node)

    # Flow: discover_sources -> analyze_build_system -> [confirm] -> extract -> merge -> analysis -> question -> collect -> generate -> validate
    # The build system analysis runs BEFORE Dockerfile confirmation, so the LLM
    # identifies which Dockerfile is actually used, and the user confirms that finding.
    # The analysis subgraph encapsulates: analyze_ports -> analyze -> enrich
    workflow.add_edge(START, "discover_sources")
    workflow.add_edge("discover_sources", "analyze_build_system")

    # Conditional: confirm Dockerfile identified by build system analysis
    workflow.add_conditional_edges(
        "analyze_build_system",
        should_confirm_dockerfile,
        {
            "confirm": "confirm",
            "skip": "extract",
        },
    )
    workflow.add_edge("confirm", "extract")

    workflow.add_edge("extract", "merge")
    workflow.add_edge("merge", "analysis")  # Analysis subgraph
    workflow.add_edge("analysis", "question")
    workflow.add_edge("question", "collect")
    workflow.add_edge("collect", "generate")
    workflow.add_edge("generate", "validate")

    if include_deployment:
        # Add deployment nodes (Phase 2 - stubs for now)
        deploy_node = create_deploy_node(settings)
        verify_node = create_verify_node(settings)
        workflow.add_node("deploy", deploy_node)
        workflow.add_node("verify", verify_node)
        workflow.add_node("fix", fix_node)

        # Conditional routing after validation (STRICT mode)
        workflow.add_conditional_edges(
            "validate",
            should_proceed_after_validation,
            {
                "proceed": "deploy",
                "blocked": END,  # Validation failed, stop here
            },
        )

        workflow.add_edge("deploy", "verify")

        # Conditional routing after verification
        workflow.add_conditional_edges(
            "verify",
            should_retry,
            {
                "retry": "fix",
                "success": END,
                "give_up": END,
            },
        )
        workflow.add_edge("fix", "generate")
    else:
        # Without deployment, end after validation
        workflow.add_edge("validate", END)

    return workflow


def compile_graph(
    llm: BaseChatModel,
    settings: Settings | None = None,
    include_deployment: bool = True,
    enable_checkpointing: bool = True,
    session_id: str | None = None,
):
    """Compile the workflow graph for execution.

    Args:
        llm: LLM instance.
        settings: Application settings.
        include_deployment: Whether to include deployment nodes.
        enable_checkpointing: Whether to enable state checkpointing for HitL.
        session_id: Deprecated - no longer used. Tracing is now handled via
            callback handler passed to graph.stream() in main.py.

    Returns:
        Compiled graph ready for execution.
    """
    if settings is None:
        settings = get_settings()

    # NOTE: LangFuse tracing is now handled at graph.stream() time via config callbacks,
    # not at compile time. This allows proper nesting of LLM calls under LangGraph nodes.
    # See main.py for the callback handler setup.

    workflow = create_workflow(llm, settings, include_deployment)

    if enable_checkpointing:
        checkpointer = MemorySaver()
        # Interrupt before confirm (Dockerfile) for HitL
        # The collect node uses interrupt() internally for questions
        return workflow.compile(
            checkpointer=checkpointer,
            interrupt_before=["confirm"],
        )
    else:
        return workflow.compile()


def run_graph(
    codebase_path: str,
    llm: BaseChatModel,
    prompt: str = "",
    settings: Settings | None = None,
    user_responses: dict[str, str] | None = None,
    cluster_id: str = "default",
    include_deployment: bool = False,
) -> dict[str, Any]:
    """Run the graph to completion (non-interactive mode).

    This is useful for testing or when responses are pre-provided.

    Args:
        codebase_path: Path to the codebase.
        llm: LLM instance.
        prompt: User's deployment request. Defaults to empty string.
        settings: Application settings.
        user_responses: Pre-provided user responses.
        cluster_id: Cluster identifier for memory.
        include_deployment: Whether to include deployment.

    Returns:
        Final state after graph execution.
    """
    if settings is None:
        settings = get_settings()

    # Create initial state
    state = create_initial_state(
        codebase_path=codebase_path,
        prompt=prompt,
        cluster_id=cluster_id,
        max_iterations=settings.max_iterations,
    )

    # Add pre-provided responses
    if user_responses:
        state["user_responses"] = user_responses

    # Compile without checkpointing for non-interactive mode
    graph = compile_graph(
        llm=llm,
        settings=settings,
        include_deployment=include_deployment,
        enable_checkpointing=False,
    )

    # Run to completion
    result = graph.invoke(state)
    return result
