"""Analysis subgraph package.

This package contains the analysis subgraph and its component nodes:
- classify_workload: Workload type classification (service vs batch)
- detect_gpu: GPU requirement detection from config and Dockerfile
- analyze_ports: Port configuration analysis
- analyze: Codebase analysis with LLM
- enrich: Infrastructure enrichment (Vault, Consul, Fabio)
"""

from src.subgraphs.analysis.graph import (
    AnalysisState,
    create_analysis_subgraph,
    create_analysis_subgraph_node,
)

__all__ = [
    "AnalysisState",
    "create_analysis_subgraph",
    "create_analysis_subgraph_node",
]
