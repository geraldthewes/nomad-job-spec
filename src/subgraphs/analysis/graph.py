"""Analysis subgraph for the Nomad Job Spec Agent.

This subgraph encapsulates the analysis pipeline:
- classify_workload: Determine if app is service or batch job
- detect_gpu: Detect GPU requirements from config and Dockerfile
- analyze_ports: Determine port configuration
- analyze: Codebase analysis with LLM
- enrich: Infrastructure enrichment (Vault, Consul, Fabio)

The subgraph has its own AnalysisState, separate from the main AgentState,
allowing it to be tested independently and extended with new analysis nodes.
"""

import logging
from typing import Any, TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_core.language_models import BaseChatModel

from config.settings import Settings, get_settings
from .analyze import create_analyze_node
from .analyze_ports import create_analyze_ports_node
from .classify_workload import create_classify_workload_node
from .detect_gpu import create_detect_gpu_node
from .enrich import create_enrich_node
from .extract_env_vars import create_extract_env_vars_node

logger = logging.getLogger(__name__)


class AnalysisState(TypedDict, total=False):
    """State for the analysis subgraph.

    This is a focused state containing only fields relevant to the analysis phase.
    It is separate from AgentState to maintain clean boundaries.

    Input fields (from parent workflow):
        codebase_path: Path to the codebase being analyzed
        selected_dockerfile: Path to the selected Dockerfile (may be None)
        discovered_sources: Mapping of source type to file path
        build_system_analysis: Jobforge/build config info from earlier nodes
        merged_extraction: Combined extraction results from extract/merge nodes

    Internal/Output fields (produced by subgraph nodes):
        workload_classification: Classification result (workload_type, confidence, evidence)
        gpu_detection: GPU detection result (requires_gpu, confidence, evidence, config_value, dockerfile_detected, cuda_version)
        port_analysis: Port configuration analysis from analyze_ports
        codebase_analysis: Full codebase analysis from analyze node
        app_name: Extracted application name
        env_var_configs: Multi-source environment variable configurations
        vault_suggestions: Vault secret path suggestions
        consul_conventions: Naming conventions from Consul
        consul_services: Available Consul services
        fabio_validation: Fabio route validation results
        nomad_info: Nomad version and capability info
        infra_issues: Infrastructure connection issues encountered
    """

    # Inputs (from parent workflow)
    codebase_path: str
    selected_dockerfile: str | None
    discovered_sources: dict[str, str]
    build_system_analysis: dict[str, Any]
    merged_extraction: dict[str, Any]

    # Internal state (passed between nodes)
    workload_classification: dict[str, Any]
    gpu_detection: dict[str, Any]
    port_analysis: dict[str, Any]
    codebase_analysis: dict[str, Any]

    # Outputs (returned to parent workflow)
    app_name: str
    env_deploy_config: dict[str, Any]  # Parsed deploy/.env.deploy configuration
    env_var_validation: dict[str, Any]  # Validation results for env vars
    env_var_configs: list[dict[str, Any]]  # Legacy: multi-source env configs (for backward compat)
    vault_suggestions: dict[str, Any]
    consul_conventions: dict[str, Any]
    consul_services: dict[str, Any]
    fabio_validation: dict[str, Any]
    nomad_info: dict[str, Any]
    infra_issues: list[dict[str, str]]


def create_analysis_subgraph(
    llm: BaseChatModel,
    settings: Settings | None = None,
) -> StateGraph:
    """Create the analysis subgraph.

    This subgraph orchestrates the analysis pipeline:
    classify_workload -> detect_gpu -> analyze_ports -> analyze -> enrich

    Args:
        llm: LLM instance for analysis nodes.
        settings: Application settings for enrich node.

    Returns:
        Compiled StateGraph ready for execution.
    """
    if settings is None:
        settings = get_settings()

    # Create node functions using existing factories
    classify_workload_node = create_classify_workload_node(llm)
    detect_gpu_node = create_detect_gpu_node(llm)
    analyze_ports_node = create_analyze_ports_node(llm)
    analyze_node = create_analyze_node(llm)
    extract_env_vars_node = create_extract_env_vars_node()
    enrich_node = create_enrich_node(settings)

    # Build subgraph
    workflow = StateGraph(AnalysisState)

    workflow.add_node("classify_workload", classify_workload_node)
    workflow.add_node("detect_gpu", detect_gpu_node)
    workflow.add_node("analyze_ports", analyze_ports_node)
    workflow.add_node("analyze", analyze_node)
    workflow.add_node("extract_env_vars", extract_env_vars_node)
    workflow.add_node("enrich", enrich_node)

    # Linear flow: classify_workload -> detect_gpu -> analyze_ports -> analyze -> extract_env_vars -> enrich
    workflow.add_edge(START, "classify_workload")
    workflow.add_edge("classify_workload", "detect_gpu")
    workflow.add_edge("detect_gpu", "analyze_ports")
    workflow.add_edge("analyze_ports", "analyze")
    workflow.add_edge("analyze", "extract_env_vars")
    workflow.add_edge("extract_env_vars", "enrich")
    workflow.add_edge("enrich", END)

    return workflow


def create_analysis_subgraph_node(
    llm: BaseChatModel,
    settings: Settings | None = None,
):
    """Create a node function that wraps the analysis subgraph.

    This wrapper handles state mapping between AgentState and AnalysisState:
    - Extracts relevant fields from AgentState as input
    - Runs the compiled subgraph
    - Returns output fields to merge back into AgentState

    Args:
        llm: LLM instance for analysis.
        settings: Application settings.

    Returns:
        Node function for use in the main LangGraph workflow.
    """
    if settings is None:
        settings = get_settings()

    # Compile the subgraph once
    subgraph = create_analysis_subgraph(llm, settings)
    compiled_subgraph = subgraph.compile()

    def analysis_node(state: dict[str, Any]) -> dict[str, Any]:
        """Execute the analysis subgraph with state mapping.

        Args:
            state: AgentState from the parent workflow.

        Returns:
            Dictionary of fields to merge back into AgentState.
        """
        # Map AgentState -> AnalysisState (extract inputs)
        analysis_input: AnalysisState = {
            "codebase_path": state.get("codebase_path", ""),
            "selected_dockerfile": state.get("selected_dockerfile"),
            "discovered_sources": state.get("discovered_sources", {}),
            "build_system_analysis": state.get("build_system_analysis", {}),
            "merged_extraction": state.get("merged_extraction", {}),
        }

        logger.info("Entering analysis subgraph")

        # Run the compiled subgraph
        result = compiled_subgraph.invoke(analysis_input)

        logger.info("Exiting analysis subgraph")

        # Map AnalysisState -> AgentState (return outputs)
        return {
            "workload_classification": result.get("workload_classification", {}),
            "gpu_detection": result.get("gpu_detection", {}),
            "port_analysis": result.get("port_analysis", {}),
            "codebase_analysis": result.get("codebase_analysis", {}),
            "app_name": result.get("app_name", ""),
            "env_deploy_config": result.get("env_deploy_config", {}),
            "env_var_validation": result.get("env_var_validation", {}),
            "env_var_configs": result.get("env_var_configs", []),
            "vault_suggestions": result.get("vault_suggestions", {}),
            "consul_conventions": result.get("consul_conventions", {}),
            "consul_services": result.get("consul_services", {}),
            "fabio_validation": result.get("fabio_validation", {}),
            "nomad_info": result.get("nomad_info", {}),
            "infra_issues": result.get("infra_issues", []),
        }

    return analysis_node
