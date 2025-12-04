"""Deployment and verification nodes for the Nomad Job Spec Agent.

Note: These are currently stubs for Phase 2 implementation.
The actual deployment logic will register jobs with the Nomad cluster
and poll for deployment status.
"""

import logging
from typing import Any

from config.settings import Settings, get_settings

logger = logging.getLogger(__name__)


def deploy_node(state: dict[str, Any], settings: Settings | None = None) -> dict[str, Any]:
    """Deploy the generated job spec to Nomad cluster.

    This is currently a stub that returns success. Phase 2 will implement:
    - Register job via Nomad API
    - Handle deployment errors
    - Track job_id for verification

    Args:
        state: Current graph state containing job_spec and job_name.
        settings: Application settings.

    Returns:
        Updated state with deployment_status and job_id.
    """
    if settings is None:
        settings = get_settings()

    job_name = state.get("job_name", "job")

    logger.info(f"[STUB] Deploying job: {job_name}")

    return {
        "deployment_status": "success",  # Stub returns success
        "job_id": f"stub-{job_name}",
    }


def verify_node(state: dict[str, Any], settings: Settings | None = None) -> dict[str, Any]:
    """Verify the deployment status of a Nomad job.

    This is currently a stub that returns success. Phase 2 will implement:
    - Poll Nomad API for allocation status
    - Check task health
    - Handle deployment timeouts (5 minute max)
    - Extract error messages on failure

    Args:
        state: Current graph state containing job_id.
        settings: Application settings.

    Returns:
        Updated state with deployment_status (and deployment_error on failure).
    """
    if settings is None:
        settings = get_settings()

    job_id = state.get("job_id", "unknown")

    logger.info(f"[STUB] Verifying deployment for job: {job_id}")

    return {
        "deployment_status": "success",
    }


def create_deploy_node(settings: Settings | None = None):
    """Create the deploy node for LangGraph integration.

    Args:
        settings: Application settings.

    Returns:
        Node function that deploys the job spec.
    """
    if settings is None:
        settings = get_settings()

    def node(state: dict[str, Any]) -> dict[str, Any]:
        return deploy_node(state, settings)

    return node


def create_verify_node(settings: Settings | None = None):
    """Create the verify node for LangGraph integration.

    Args:
        settings: Application settings.

    Returns:
        Node function that verifies deployment status.
    """
    if settings is None:
        settings = get_settings()

    def node(state: dict[str, Any]) -> dict[str, Any]:
        return verify_node(state, settings)

    return node
