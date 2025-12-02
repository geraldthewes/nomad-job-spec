"""Question generation and response collection nodes for the Nomad Job Spec Agent."""

import logging
from typing import Any

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

    # Environment variable configuration with multi-source support
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

    Args:
        state: Current graph state.

    Returns:
        Unchanged state (pass-through node).
    """
    return state


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
