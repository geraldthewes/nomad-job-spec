"""Tests for the analysis subgraph."""

import pytest
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage

from src.subgraphs.analysis import (
    AnalysisState,
    create_analysis_subgraph,
    create_analysis_subgraph_node,
)


class TestAnalysisState:
    """Tests for AnalysisState type definition."""

    def test_state_accepts_input_fields(self):
        """Test that AnalysisState accepts all input fields."""
        state: AnalysisState = {
            "codebase_path": "/path/to/code",
            "selected_dockerfile": "Dockerfile",
            "discovered_sources": {"dockerfile": "/path/Dockerfile"},
            "build_system_analysis": {"mechanism": "docker"},
            "merged_extraction": {"image": "nginx"},
        }

        assert state["codebase_path"] == "/path/to/code"
        assert state["selected_dockerfile"] == "Dockerfile"
        assert state["discovered_sources"]["dockerfile"] == "/path/Dockerfile"
        assert state["build_system_analysis"]["mechanism"] == "docker"

    def test_state_accepts_none_dockerfile(self):
        """Test that selected_dockerfile can be None."""
        state: AnalysisState = {
            "codebase_path": "/path/to/code",
            "selected_dockerfile": None,
            "discovered_sources": {},
            "build_system_analysis": {},
            "merged_extraction": {},
        }

        assert state["selected_dockerfile"] is None


class TestCreateAnalysisSubgraph:
    """Tests for subgraph creation."""

    def test_creates_subgraph(self, mock_llm, mock_settings):
        """Test that subgraph is created successfully."""
        subgraph = create_analysis_subgraph(mock_llm, mock_settings)

        assert subgraph is not None

    def test_subgraph_has_required_nodes(self, mock_llm, mock_settings):
        """Test that subgraph has the expected nodes."""
        subgraph = create_analysis_subgraph(mock_llm, mock_settings)

        nodes = subgraph.nodes
        assert "classify_workload" in nodes
        assert "analyze_ports" in nodes
        assert "analyze" in nodes
        assert "enrich" in nodes

    def test_subgraph_compiles(self, mock_llm, mock_settings):
        """Test that subgraph compiles without errors."""
        subgraph = create_analysis_subgraph(mock_llm, mock_settings)
        compiled = subgraph.compile()

        assert compiled is not None


class TestCreateAnalysisSubgraphNode:
    """Tests for the subgraph wrapper node."""

    def test_creates_node_function(self, mock_llm, mock_settings):
        """Test that wrapper creates a callable node function."""
        node = create_analysis_subgraph_node(mock_llm, mock_settings)

        assert callable(node)

    def test_node_maps_input_state(self, mock_llm_for_analysis, mock_settings):
        """Test that node correctly extracts input fields from AgentState."""
        node = create_analysis_subgraph_node(mock_llm_for_analysis, mock_settings)

        # Simulate AgentState input
        agent_state = {
            "prompt": "Deploy my app",  # Should be ignored
            "codebase_path": "/test/path",
            "selected_dockerfile": "Dockerfile.prod",
            "discovered_sources": {"dockerfile": "/test/path/Dockerfile.prod"},
            "build_system_analysis": {"mechanism": "docker", "dockerfile_used": "Dockerfile.prod"},
            "merged_extraction": {"image": "nginx:latest"},
            "deployment_status": "pending",  # Should be ignored
        }

        with patch("src.subgraphs.analysis.graph.get_settings", return_value=mock_settings):
            result = node(agent_state)

        # Verify output fields are returned
        assert "port_analysis" in result
        assert "codebase_analysis" in result
        assert "env_var_configs" in result

    def test_node_returns_output_fields(self, mock_llm_for_analysis, mock_settings):
        """Test that node returns all expected output fields."""
        node = create_analysis_subgraph_node(mock_llm_for_analysis, mock_settings)

        agent_state = {
            "codebase_path": "/test/path",
            "selected_dockerfile": None,
            "discovered_sources": {},
            "build_system_analysis": {},
            "merged_extraction": {},
        }

        with patch("src.subgraphs.analysis.graph.get_settings", return_value=mock_settings):
            result = node(agent_state)

        # Check all expected output fields
        expected_fields = [
            "workload_classification",
            "port_analysis",
            "codebase_analysis",
            "app_name",
            "env_var_configs",
            "vault_suggestions",
            "consul_conventions",
            "consul_services",
            "fabio_validation",
            "nomad_info",
            "infra_issues",
        ]
        for field in expected_fields:
            assert field in result, f"Missing output field: {field}"


@pytest.fixture
def mock_llm_for_analysis():
    """Create a mock LLM that handles analysis-specific prompts."""
    llm = MagicMock()

    def mock_invoke(messages):
        """Return mock responses based on message content."""
        content = str(messages[-1].content) if messages else ""

        if "workload" in content.lower() or "batch" in content.lower():
            # Workload classification response
            return AIMessage(content="""{
                "workload_type": "service",
                "confidence": "high",
                "evidence": "CMD uses uvicorn to run a web server"
            }""")

        elif "port" in content.lower() or "listening" in content.lower():
            # Port analysis response
            return AIMessage(content="""{
                "type": "env_var",
                "value": "PORT",
                "default_port": 3000,
                "evidence": "app.py:10 - port = int(os.getenv('PORT', 3000))"
            }""")

        elif "analyze" in content.lower() or "codebase" in content.lower():
            # Codebase analysis response
            return AIMessage(content="""{
                "summary": "A simple web application",
                "service_type": "MEDIUM",
                "docker_image": "nginx:latest",
                "ports": [{"name": "http", "container_port": 80, "static": false}],
                "env_vars": {},
                "secrets": [],
                "resources": {"cpu": 500, "memory": 512},
                "health_check": {"type": "http", "path": "/health", "port": "http"},
                "requires_amd64": true,
                "requires_storage": false,
                "dependencies": [],
                "warnings": []
            }""")

        else:
            return AIMessage(content="Mock response")

    llm.invoke = mock_invoke
    return llm
