"""Dockerfile confirmation node for the Nomad Job Spec workflow.

This node runs AFTER build system analysis to:
1. Present the LLM-identified Dockerfile to the user for confirmation
2. Allow user to override if the identified Dockerfile is incorrect

This replaces the old "select" flow where users chose Dockerfiles before
build analysis. Now the build system tells us which Dockerfile it uses,
and the user confirms (or overrides).
"""

import logging
from typing import Any, Literal

from src.observability import get_observability

logger = logging.getLogger(__name__)


def confirm_dockerfile_node(state: dict[str, Any]) -> dict[str, Any]:
    """Interrupt point for Dockerfile confirmation.

    This is a pass-through node - actual confirmation happens via CLI interrupt.
    The CLI will check build_system_analysis.dockerfile_used and prompt the user.

    Args:
        state: Current graph state.

    Returns:
        Unchanged state (pass-through node).
    """
    obs = get_observability()
    trace = obs.create_trace(
        name="confirm_dockerfile_node",
        input={
            "dockerfile_used": state.get("build_system_analysis", {}).get("dockerfile_used"),
            "dockerfiles_found": state.get("dockerfiles_found", []),
        },
    )

    build_analysis = state.get("build_system_analysis", {})
    dockerfile_used = build_analysis.get("dockerfile_used")
    dockerfiles_found = state.get("dockerfiles_found", [])
    selected = state.get("selected_dockerfile")

    # Log current state for debugging
    if dockerfile_used:
        logger.info(f"Build system identified Dockerfile: {dockerfile_used}")
    else:
        logger.info("Build system did not identify a specific Dockerfile")

    if selected:
        logger.info(f"Dockerfile already selected: {selected}")

    if trace:
        trace.end(output={
            "dockerfile_used": dockerfile_used,
            "selected_dockerfile": selected,
            "awaiting_confirmation": not selected and dockerfile_used is not None,
        })

    return state


def should_confirm_dockerfile(state: dict[str, Any]) -> Literal["confirm", "skip"]:
    """Conditional edge: determine if user confirmation is needed.

    Confirmation is needed when:
    - Build system identified a Dockerfile (dockerfile_used is set)
    - User hasn't already selected one (via CLI flag or previous run)

    Args:
        state: Current graph state.

    Returns:
        "confirm" if user should confirm/override, "skip" otherwise.
    """
    selected = state.get("selected_dockerfile")

    # Already selected - skip confirmation
    if selected:
        logger.debug("Skipping confirmation - Dockerfile already selected")
        return "skip"

    # Check if build system identified a Dockerfile
    build_analysis = state.get("build_system_analysis", {})
    dockerfile_used = build_analysis.get("dockerfile_used")
    dockerfiles_found = state.get("dockerfiles_found", [])

    # If build system identified something, we need confirmation
    if dockerfile_used:
        return "confirm"

    # If no dockerfile_used but we have discovered Dockerfiles, also need selection
    if dockerfiles_found:
        return "confirm"

    # No Dockerfiles at all - skip (will error elsewhere)
    logger.debug("Skipping confirmation - no Dockerfiles found or identified")
    return "skip"


def create_confirm_node():
    """Create the confirmation node for LangGraph integration."""

    def node(state: dict[str, Any]) -> dict[str, Any]:
        return confirm_dockerfile_node(state)

    return node
