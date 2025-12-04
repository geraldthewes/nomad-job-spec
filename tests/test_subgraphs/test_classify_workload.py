"""Tests for the classify_workload node."""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import tempfile

from langchain_core.messages import AIMessage

from src.subgraphs.analysis.classify_workload import (
    classify_workload_node,
    create_classify_workload_node,
)


class TestClassifyWorkloadNode:
    """Tests for the classify_workload_node function."""

    def test_classifies_service_from_uvicorn(self, mock_llm_service):
        """Test that Dockerfile with uvicorn is classified as service."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dockerfile = Path(tmpdir) / "Dockerfile"
            dockerfile.write_text("FROM python:3.11\nCMD uvicorn app:app --host 0.0.0.0 --port 8000")

            state = {
                "codebase_path": tmpdir,
                "selected_dockerfile": "Dockerfile",
            }

            result = classify_workload_node(state, mock_llm_service)

            assert "workload_classification" in result
            assert result["workload_classification"]["workload_type"] == "service"
            assert result["workload_classification"]["confidence"] == "high"

    def test_classifies_batch_from_script(self, mock_llm_batch):
        """Test that Dockerfile running a script is classified as batch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dockerfile = Path(tmpdir) / "Dockerfile"
            dockerfile.write_text("FROM python:3.11\nCMD python migrate.py")

            state = {
                "codebase_path": tmpdir,
                "selected_dockerfile": "Dockerfile",
            }

            result = classify_workload_node(state, mock_llm_batch)

            assert "workload_classification" in result
            assert result["workload_classification"]["workload_type"] == "batch"
            assert result["workload_classification"]["confidence"] == "high"

    def test_defaults_to_service_when_no_dockerfile(self, mock_llm_service):
        """Test that missing Dockerfile defaults to service with low confidence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = {
                "codebase_path": tmpdir,
                "selected_dockerfile": None,
            }

            result = classify_workload_node(state, mock_llm_service)

            assert "workload_classification" in result
            assert result["workload_classification"]["workload_type"] == "service"
            assert result["workload_classification"]["confidence"] == "low"

    def test_handles_relative_dockerfile_path(self, mock_llm_service):
        """Test that relative Dockerfile paths are resolved correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dockerfile = Path(tmpdir) / "Dockerfile"
            dockerfile.write_text("FROM nginx\nCMD nginx -g 'daemon off;'")

            state = {
                "codebase_path": tmpdir,
                "selected_dockerfile": "Dockerfile",  # Relative path
            }

            result = classify_workload_node(state, mock_llm_service)

            assert "workload_classification" in result
            # LLM was called (not defaulting due to missing file)
            assert result["workload_classification"]["confidence"] == "high"

    def test_raises_without_llm(self):
        """Test that ValueError is raised when no LLM provided."""
        state = {"codebase_path": "/test"}

        with pytest.raises(ValueError, match="LLM is required"):
            classify_workload_node(state, None)


class TestCreateClassifyWorkloadNode:
    """Tests for the node factory function."""

    def test_creates_callable_node(self, mock_llm_service):
        """Test that factory creates a callable node function."""
        node = create_classify_workload_node(mock_llm_service)
        assert callable(node)

    def test_raises_without_llm(self):
        """Test that factory raises ValueError without LLM."""
        with pytest.raises(ValueError, match="LLM is required"):
            create_classify_workload_node(None)

    def test_node_invokes_correctly(self, mock_llm_service):
        """Test that created node can be invoked with state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dockerfile = Path(tmpdir) / "Dockerfile"
            dockerfile.write_text("FROM python:3.11\nCMD uvicorn app:app")

            node = create_classify_workload_node(mock_llm_service)
            state = {
                "codebase_path": tmpdir,
                "selected_dockerfile": "Dockerfile",
            }

            result = node(state)

            assert "workload_classification" in result


@pytest.fixture
def mock_llm_service():
    """Mock LLM that returns service classification."""
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content="""{
        "workload_type": "service",
        "confidence": "high",
        "evidence": "CMD uses uvicorn to run a web server"
    }""")
    return llm


@pytest.fixture
def mock_llm_batch():
    """Mock LLM that returns batch classification."""
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content="""{
        "workload_type": "batch",
        "confidence": "high",
        "evidence": "CMD runs python script that exits after completion"
    }""")
    return llm
