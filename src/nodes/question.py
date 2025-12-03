"""Question generation and response collection nodes for the Nomad Job Spec Agent."""

import logging
from typing import Any

from langgraph.types import interrupt

logger = logging.getLogger(__name__)


def generate_questions_node(state: dict[str, Any]) -> dict[str, Any]:
    """Generate clarifying questions based on analysis.

    This node creates questions to ask the user before generating the spec.
    Now enhanced with multi-source env var configurations from enrichment.

    Args:
        state: Current graph state containing codebase_analysis and enrichment data.

    Returns:
        Updated state with questions list.
    """
    analysis = state.get("codebase_analysis", {})
    env_var_configs = state.get("env_var_configs", [])
    vault_suggestions = state.get("vault_suggestions", {})  # Legacy fallback
    fabio_validation = state.get("fabio_validation", {})
    questions = []

    # Check if image is missing
    dockerfile = analysis.get("dockerfile")
    if not dockerfile or not dockerfile.get("base_image"):
        questions.append("What Docker image should be used for this deployment?")

    # Port-related questions FIRST (before env_configs, since network mode affects APP_PORT)
    port_analysis = state.get("port_analysis", {})
    app_port = port_analysis.get("app_listening_port", {})
    port_type = app_port.get("type", "unknown")

    if port_type == "unknown":
        # No port detected - ask the user
        questions.append("What port does this application listen on?")
    elif port_analysis.get("supports_dynamic_port"):
        # App supports dynamic port allocation (env var based)
        # Ask which network mode to use - this must come BEFORE env_configs
        # because the answer determines whether APP_PORT uses ${NOMAD_PORT_http} or fixed value
        env_var = app_port.get("value", "PORT")
        default_port = app_port.get("default_port", 8000)
        network_q = {
            "type": "network_mode",
            "prompt": (
                f"This app listens on port configured via {env_var} "
                f"(default: {default_port}). Which network mode?"
            ),
            "env_var": env_var,
            "default_port": default_port,
            "options": [
                {
                    "value": "bridge",
                    "label": "Bridge (recommended)",
                    "description": f"Container uses fixed port {default_port}, Nomad handles external mapping",
                },
                {
                    "value": "host",
                    "label": "Host",
                    "description": f"App uses ${{NOMAD_PORT_http}} to avoid port conflicts on host",
                },
            ],
            "default": "bridge",
        }
        questions.append(network_q)
    # else: port is hardcoded, no question needed

    # Environment variable configuration AFTER network mode (so we can apply the choice)
    env_vars = analysis.get("env_vars_required", [])
    if env_vars:
        if env_var_configs:
            # New format: multi-source configuration
            questions.append({
                "type": "env_configs",
                "configs": env_var_configs[:15],  # Limit to first 15
                "prompt": "Environment variable configuration",
            })
        elif vault_suggestions.get("suggestions"):
            # Legacy fallback: Vault-only suggestions
            questions.append({
                "type": "vault_paths",
                "suggestions": vault_suggestions["suggestions"][:10],
                "prompt": "Environment variable Vault paths",
            })
        else:
            questions.append(
                f"The following environment variables were detected: {', '.join(env_vars[:5])}. "
                "How should these be configured? (fixed value, Consul KV, or Vault secret)"
            )

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
    """Node for collecting user responses using LangGraph interrupt.

    This node uses interrupt() to pause execution and wait for user input.
    The questions are passed to interrupt(), and when the graph is resumed
    with Command(resume=responses), those responses are returned here.

    Args:
        state: Current graph state containing questions.

    Returns:
        Updated state with user_responses from the interrupt.
    """
    questions = state.get("questions", [])
    if not questions:
        return state

    # Interrupt and wait for user responses
    # The interrupt() call pauses here and returns whatever value
    # is passed to Command(resume=value) when execution resumes
    user_responses = interrupt(questions)

    return {
        **state,
        "user_responses": user_responses,
    }


def create_question_node():
    """Create the question generation node for LangGraph integration.

    Returns:
        Node function that generates clarifying questions.
    """
    def node(state: dict[str, Any]) -> dict[str, Any]:
        return generate_questions_node(state)

    return node


def create_collect_node():
    """Create the response collection node for LangGraph integration.

    Returns:
        Node function that serves as HitL interrupt point.
    """
    def node(state: dict[str, Any]) -> dict[str, Any]:
        return collect_responses_node(state)

    return node
