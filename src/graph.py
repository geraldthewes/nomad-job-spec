"""LangGraph workflow definition for the Nomad Job Spec Agent."""

from typing import Any, Annotated, Literal
from dataclasses import dataclass, field

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

from src.nodes.analyze import create_analyze_node
from src.nodes.generate import create_generate_node, create_fix_node
from src.nodes.enrich import create_enrich_node
from src.nodes.validate import create_validate_node, should_proceed_after_validation
from config.settings import Settings, get_settings


# Type for the state with message history
class AgentState(dict):
    """State for the Nomad Job Spec Agent.

    This is a TypedDict-like class for use with LangGraph.
    """

    # Input
    prompt: str
    codebase_path: str

    # Analysis
    codebase_analysis: dict[str, Any]

    # Conversation
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
    vault_suggestions: dict[str, Any]
    consul_conventions: dict[str, Any]
    consul_services: dict[str, Any]
    fabio_validation: dict[str, Any]
    nomad_info: dict[str, Any]

    # Pre-deployment validation
    pre_deploy_validation: dict[str, Any]


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
        "vault_suggestions": {},
        "consul_conventions": {},
        "consul_services": {},
        "fabio_validation": {},
        "nomad_info": {},
        # Validation
        "pre_deploy_validation": {},
    }


def generate_questions_node(state: dict[str, Any]) -> dict[str, Any]:
    """Generate clarifying questions based on analysis.

    This node creates questions to ask the user before generating the spec.
    Now enhanced with Vault path suggestions from enrichment.
    """
    analysis = state.get("codebase_analysis", {})
    vault_suggestions = state.get("vault_suggestions", {})
    fabio_validation = state.get("fabio_validation", {})
    questions = []

    # Check if image is missing
    dockerfile = analysis.get("dockerfile")
    if not dockerfile or not dockerfile.get("base_image"):
        questions.append("What Docker image should be used for this deployment?")

    # Enhanced environment variable question with Vault suggestions
    # Uses structured dict for interactive step-by-step confirmation
    env_vars = analysis.get("env_vars_required", [])
    if env_vars:
        suggestions = vault_suggestions.get("suggestions", [])
        if suggestions:
            # Structured question for interactive Vault path confirmation
            questions.append({
                "type": "vault_paths",
                "suggestions": suggestions[:10],  # Limit to first 10
                "prompt": "Environment variable Vault paths",
            })
        else:
            questions.append(
                f"The following environment variables were detected: {', '.join(env_vars[:5])}. "
                "What Vault paths should be used for these secrets?"
            )

    # Check for ports
    if not (dockerfile and dockerfile.get("exposed_ports")):
        questions.append("What port does this application listen on?")

    # Fabio routing question with suggestion
    suggested_hostname = fabio_validation.get("suggested_hostname")
    conflicts = fabio_validation.get("conflicts", [])
    if suggested_hostname:
        if conflicts:
            conflict_info = "; ".join(
                f"{c['type']} with {c.get('existing_service', 'unknown')}"
                for c in conflicts
            )
            questions.append(
                f"Suggested hostname '{suggested_hostname}' has conflicts: {conflict_info}. "
                "Please provide an alternative hostname."
            )
        else:
            questions.append(
                f"Suggested Fabio hostname: {suggested_hostname} (tag: urlprefix-{suggested_hostname}:9999/). "
                "Confirm or provide alternative."
            )

    # Resource questions
    resources = analysis.get("suggested_resources", {})
    questions.append(
        f"Suggested resources: {resources.get('cpu', 500)}MHz CPU, "
        f"{resources.get('memory', 256)}MB memory. Is this appropriate?"
    )

    # Scaling question
    questions.append("How many instances should be deployed initially?")

    return {
        **state,
        "questions": questions,
    }


def collect_responses_node(state: dict[str, Any]) -> dict[str, Any]:
    """Node for collecting user responses.

    In the actual flow, this is an interrupt point where the CLI collects responses.
    This node just passes through - the actual collection happens via graph interrupt.
    """
    return state


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

    Updated workflow with infrastructure enrichment and validation:
    START -> analyze -> enrich -> question -> collect -> generate -> validate -> deploy -> verify

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
    analyze_node = create_analyze_node(llm)
    enrich_node = create_enrich_node(settings)
    generate_node = create_generate_node(llm)
    validate_node = create_validate_node(settings)
    fix_node = create_fix_node(llm)

    # Build graph
    workflow = StateGraph(dict)

    # Add nodes
    workflow.add_node("analyze", analyze_node)
    workflow.add_node("enrich", enrich_node)  # NEW: Infrastructure enrichment
    workflow.add_node("question", generate_questions_node)
    workflow.add_node("collect", collect_responses_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("validate", validate_node)  # NEW: Pre-deploy validation

    # Updated flow: analyze -> enrich -> question -> collect -> generate -> validate
    workflow.add_edge(START, "analyze")
    workflow.add_edge("analyze", "enrich")  # NEW
    workflow.add_edge("enrich", "question")  # MODIFIED
    workflow.add_edge("question", "collect")
    workflow.add_edge("collect", "generate")
    workflow.add_edge("generate", "validate")  # NEW

    if include_deployment:
        # Add deployment nodes (Phase 2 - stubs for now)
        workflow.add_node("deploy", _deploy_stub)
        workflow.add_node("verify", _verify_stub)
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


def _deploy_stub(state: dict[str, Any]) -> dict[str, Any]:
    """Stub for deployment node (implemented in Phase 2)."""
    return {
        **state,
        "deployment_status": "success",  # Stub returns success
        "job_id": f"stub-{state.get('job_name', 'job')}",
    }


def _verify_stub(state: dict[str, Any]) -> dict[str, Any]:
    """Stub for verification node (implemented in Phase 2)."""
    return {
        **state,
        "deployment_status": "success",
    }


def compile_graph(
    llm: BaseChatModel,
    settings: Settings | None = None,
    include_deployment: bool = True,
    enable_checkpointing: bool = True,
):
    """Compile the workflow graph for execution.

    Args:
        llm: LLM instance.
        settings: Application settings.
        include_deployment: Whether to include deployment nodes.
        enable_checkpointing: Whether to enable state checkpointing for HitL.

    Returns:
        Compiled graph ready for execution.
    """
    workflow = create_workflow(llm, settings, include_deployment)

    if enable_checkpointing:
        checkpointer = MemorySaver()
        # Interrupt before collect for HitL
        return workflow.compile(
            checkpointer=checkpointer,
            interrupt_before=["collect"],
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
