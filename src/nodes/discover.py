"""Dockerfile discovery node for the Nomad Job Spec workflow.

This node runs BEFORE analysis to:
1. Find all Dockerfiles in the codebase
2. Present options to user for selection (via interrupt)
"""

import logging
from pathlib import Path
from typing import Any, Literal

from src.observability import get_observability

logger = logging.getLogger(__name__)


def discover_dockerfiles_node(state: dict[str, Any]) -> dict[str, Any]:
    """Discover all Dockerfiles in the codebase.

    This is a lightweight discovery pass - no parsing, just finding paths.

    Args:
        state: Current graph state containing 'codebase_path'.

    Returns:
        Updated state with 'dockerfiles_found' and optionally 'selected_dockerfile'.
    """
    obs = get_observability()
    trace = obs.create_trace(
        name="discover_dockerfiles_node",
        input={"codebase_path": state.get("codebase_path")},
    )

    codebase_path = state.get("codebase_path")
    if not codebase_path:
        if trace:
            trace.end(level="ERROR", status_message="No codebase path provided")
        return {
            **state,
            "dockerfiles_found": [],
            "selected_dockerfile": None,
        }

    path = Path(codebase_path)

    # Handle git URLs - clone first if needed
    if codebase_path.startswith(("http://", "https://", "git@")):
        from src.tools.codebase import clone_repository

        with obs.span("clone_repository", trace=trace, input={"url": codebase_path}) as span:
            codebase_path = clone_repository(codebase_path)
            path = Path(codebase_path)
            span.end(output={"cloned_path": codebase_path})
            # Update state with cloned path
            state = {**state, "codebase_path": codebase_path}

    if not path.exists():
        if trace:
            trace.end(
                level="ERROR", status_message=f"Path does not exist: {codebase_path}"
            )
        return {
            **state,
            "dockerfiles_found": [],
            "selected_dockerfile": None,
        }

    # Find all Dockerfiles
    with obs.span("find_dockerfiles", trace=trace, input={"path": str(path)}) as span:
        dockerfiles_found = []
        for dockerfile_path in path.glob("**/Dockerfile*"):
            # Skip directories, backup files, and documentation
            if dockerfile_path.is_dir():
                continue
            name = dockerfile_path.name.lower()
            if name.endswith((".md", ".txt", ".bak", ".orig", ".swp", "~")):
                continue
            # Skip files in common non-source directories
            rel_path = str(dockerfile_path.relative_to(path))
            skip_dirs = ["node_modules", ".git", "vendor", "__pycache__"]
            if any(part in rel_path.split("/") for part in skip_dirs):
                continue
            dockerfiles_found.append(rel_path)

        # Sort: prefer root Dockerfile first, then alphabetically
        def dockerfile_sort_key(p: str) -> tuple:
            depth = p.count("/")
            is_plain_dockerfile = p.lower() in ("dockerfile", "dockerfile")
            return (depth, not is_plain_dockerfile, p.lower())

        dockerfiles_found.sort(key=dockerfile_sort_key)
        span.end(
            output={"dockerfiles_found": dockerfiles_found, "count": len(dockerfiles_found)}
        )

    # Log discovery results - selection happens via interrupt
    selected_dockerfile = None
    if len(dockerfiles_found) == 0:
        if trace:
            trace.end(level="ERROR", status_message="No Dockerfiles found in codebase")
        raise FileNotFoundError(
            f"No Dockerfiles found in {codebase_path}. "
            "Please ensure the codebase contains a Dockerfile."
        )
    elif len(dockerfiles_found) == 1:
        logger.info(f"Found 1 Dockerfile: {dockerfiles_found[0]} - awaiting confirmation")
    else:
        logger.info(
            f"Found {len(dockerfiles_found)} Dockerfiles - user selection required"
        )

    if trace:
        trace.end(
            output={
                "dockerfiles_found": dockerfiles_found,
                "count": len(dockerfiles_found),
                "auto_selected": selected_dockerfile,
            }
        )

    return {
        **state,
        "dockerfiles_found": dockerfiles_found,
        "selected_dockerfile": selected_dockerfile,
    }


def select_dockerfile_node(state: dict[str, Any]) -> dict[str, Any]:
    """Interrupt point for Dockerfile selection.

    This is a pass-through node - actual selection happens via CLI interrupt.
    Similar to collect_responses_node in question.py.

    Args:
        state: Current graph state.

    Returns:
        Unchanged state (pass-through node).
    """
    # Validation: ensure selection was made if multiple Dockerfiles exist
    dockerfiles = state.get("dockerfiles_found", [])
    selected = state.get("selected_dockerfile")

    if len(dockerfiles) > 1 and not selected:
        logger.warning("No Dockerfile selected despite multiple options")

    return state


def should_select_dockerfile(state: dict[str, Any]) -> Literal["select", "skip"]:
    """Conditional edge: determine if user selection/confirmation is needed.

    Args:
        state: Current graph state.

    Returns:
        "select" if user must choose/confirm, "skip" if no Dockerfiles at all.
    """
    dockerfiles = state.get("dockerfiles_found", [])
    selected = state.get("selected_dockerfile")

    # Skip only if no Dockerfiles found or already selected (e.g., via CLI flag)
    if selected or len(dockerfiles) == 0:
        return "skip"

    # Always require confirmation, even for single Dockerfile
    return "select"


def create_discover_node():
    """Create the discovery node for LangGraph integration."""

    def node(state: dict[str, Any]) -> dict[str, Any]:
        return discover_dockerfiles_node(state)

    return node


def create_select_node():
    """Create the selection node for LangGraph integration."""

    def node(state: dict[str, Any]) -> dict[str, Any]:
        return select_dockerfile_node(state)

    return node
