"""Tests for the detect_gpu node."""

import pytest
from unittest.mock import MagicMock
from pathlib import Path
import tempfile

from langchain_core.messages import AIMessage

from src.subgraphs.analysis.detect_gpu import (
    detect_gpu_node,
    create_detect_gpu_node,
)


class TestDetectGpuNode:
    """Tests for the detect_gpu_node function."""

    # ==================== Config + Dockerfile Merge Tests ====================

    def test_config_true_with_gpu_dockerfile(self, mock_llm_gpu_cuda12):
        """Test config=True with GPU Dockerfile extracts cuda_version."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dockerfile = Path(tmpdir) / "Dockerfile"
            dockerfile.write_text("FROM nvidia/cuda:12.1-runtime\nCMD python app.py")

            state = {
                "codebase_path": tmpdir,
                "selected_dockerfile": "Dockerfile",
                "merged_extraction": {"requires_gpu": True},
            }

            result = detect_gpu_node(state, mock_llm_gpu_cuda12)

            assert "gpu_detection" in result
            detection = result["gpu_detection"]
            assert detection["requires_gpu"] is True
            assert detection["config_value"] is True
            assert detection["dockerfile_detected"] is True
            assert detection["cuda_version"] == "12.1"
            assert detection["confidence"] == "high"

    def test_config_true_with_non_gpu_dockerfile(self, mock_llm_no_gpu):
        """Test config=True with non-GPU Dockerfile still returns requires_gpu=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dockerfile = Path(tmpdir) / "Dockerfile"
            dockerfile.write_text("FROM python:3.11-slim\nCMD python app.py")

            state = {
                "codebase_path": tmpdir,
                "selected_dockerfile": "Dockerfile",
                "merged_extraction": {"requires_gpu": True},
            }

            result = detect_gpu_node(state, mock_llm_no_gpu)

            detection = result["gpu_detection"]
            assert detection["requires_gpu"] is True
            assert detection["config_value"] is True
            assert detection["dockerfile_detected"] is False
            assert detection["cuda_version"] is None
            assert detection["confidence"] == "high"

    def test_config_false_with_gpu_dockerfile_still_extracts_cuda(self, mock_llm_gpu_cuda11):
        """Test config=False with GPU Dockerfile: requires_gpu=False but cuda_version extracted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dockerfile = Path(tmpdir) / "Dockerfile"
            dockerfile.write_text("FROM nvidia/cuda:11.8-devel\nCMD python build.py")

            state = {
                "codebase_path": tmpdir,
                "selected_dockerfile": "Dockerfile",
                "merged_extraction": {"requires_gpu": False},
            }

            result = detect_gpu_node(state, mock_llm_gpu_cuda11)

            detection = result["gpu_detection"]
            assert detection["requires_gpu"] is False  # Config is authoritative
            assert detection["config_value"] is False
            assert detection["dockerfile_detected"] is True
            assert detection["cuda_version"] == "11.8"  # Still extracted
            assert detection["confidence"] == "high"
            assert "overriding" in detection["evidence"].lower()

    def test_config_none_with_gpu_dockerfile(self, mock_llm_gpu_cuda12):
        """Test config=None with GPU Dockerfile uses Dockerfile detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dockerfile = Path(tmpdir) / "Dockerfile"
            dockerfile.write_text("FROM nvidia/cuda:12.1-runtime\nCMD python app.py")

            state = {
                "codebase_path": tmpdir,
                "selected_dockerfile": "Dockerfile",
                "merged_extraction": {},  # No requires_gpu key
            }

            result = detect_gpu_node(state, mock_llm_gpu_cuda12)

            detection = result["gpu_detection"]
            assert detection["requires_gpu"] is True  # From Dockerfile
            assert detection["config_value"] is None
            assert detection["dockerfile_detected"] is True
            assert detection["cuda_version"] == "12.1"

    def test_config_none_with_non_gpu_dockerfile(self, mock_llm_no_gpu):
        """Test config=None with non-GPU Dockerfile returns requires_gpu=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dockerfile = Path(tmpdir) / "Dockerfile"
            dockerfile.write_text("FROM python:3.11-slim\nCMD python app.py")

            state = {
                "codebase_path": tmpdir,
                "selected_dockerfile": "Dockerfile",
                "merged_extraction": {},
            }

            result = detect_gpu_node(state, mock_llm_no_gpu)

            detection = result["gpu_detection"]
            assert detection["requires_gpu"] is False
            assert detection["config_value"] is None
            assert detection["dockerfile_detected"] is False
            assert detection["cuda_version"] is None

    # ==================== Dockerfile Detection Tests ====================

    def test_detects_nvidia_cuda_base_image(self, mock_llm_gpu_cuda12):
        """Test detection of nvidia/cuda base image."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dockerfile = Path(tmpdir) / "Dockerfile"
            dockerfile.write_text("FROM nvidia/cuda:12.1-runtime-ubuntu22.04\nCMD python train.py")

            state = {
                "codebase_path": tmpdir,
                "selected_dockerfile": "Dockerfile",
                "merged_extraction": {},
            }

            result = detect_gpu_node(state, mock_llm_gpu_cuda12)

            detection = result["gpu_detection"]
            assert detection["requires_gpu"] is True
            assert detection["dockerfile_detected"] is True
            assert detection["confidence"] == "high"
            assert detection["cuda_version"] == "12.1"

    def test_detects_pytorch_gpu_image(self, mock_llm_pytorch_gpu):
        """Test detection of PyTorch GPU image."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dockerfile = Path(tmpdir) / "Dockerfile"
            dockerfile.write_text("FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime\nCMD python train.py")

            state = {
                "codebase_path": tmpdir,
                "selected_dockerfile": "Dockerfile",
                "merged_extraction": {},
            }

            result = detect_gpu_node(state, mock_llm_pytorch_gpu)

            detection = result["gpu_detection"]
            assert detection["requires_gpu"] is True
            assert detection["dockerfile_detected"] is True
            assert detection["confidence"] == "high"

    def test_detects_tensorflow_gpu_image(self, mock_llm_tensorflow_gpu):
        """Test detection of TensorFlow GPU image."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dockerfile = Path(tmpdir) / "Dockerfile"
            dockerfile.write_text("FROM tensorflow/tensorflow:2.15.0-gpu\nCMD python train.py")

            state = {
                "codebase_path": tmpdir,
                "selected_dockerfile": "Dockerfile",
                "merged_extraction": {},
            }

            result = detect_gpu_node(state, mock_llm_tensorflow_gpu)

            detection = result["gpu_detection"]
            assert detection["requires_gpu"] is True
            assert detection["dockerfile_detected"] is True
            assert detection["confidence"] == "high"

    def test_plain_python_image_no_gpu(self, mock_llm_no_gpu):
        """Test that plain Python image is detected as non-GPU."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dockerfile = Path(tmpdir) / "Dockerfile"
            dockerfile.write_text("FROM python:3.11-slim\nRUN pip install flask\nCMD flask run")

            state = {
                "codebase_path": tmpdir,
                "selected_dockerfile": "Dockerfile",
                "merged_extraction": {},
            }

            result = detect_gpu_node(state, mock_llm_no_gpu)

            detection = result["gpu_detection"]
            assert detection["requires_gpu"] is False
            assert detection["dockerfile_detected"] is False
            assert detection["confidence"] == "high"

    # ==================== Edge Cases ====================

    def test_missing_dockerfile_returns_false(self, mock_llm_no_gpu):
        """Test that missing Dockerfile returns requires_gpu=False with low confidence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = {
                "codebase_path": tmpdir,
                "selected_dockerfile": None,
                "merged_extraction": {},
            }

            result = detect_gpu_node(state, mock_llm_no_gpu)

            detection = result["gpu_detection"]
            assert detection["requires_gpu"] is False
            assert detection["dockerfile_detected"] is False
            assert detection["cuda_version"] is None
            assert detection["confidence"] == "low"

    def test_handles_malformed_llm_response(self, mock_llm_malformed):
        """Test handling of malformed LLM response."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dockerfile = Path(tmpdir) / "Dockerfile"
            dockerfile.write_text("FROM nvidia/cuda:12.1-runtime\nCMD python app.py")

            state = {
                "codebase_path": tmpdir,
                "selected_dockerfile": "Dockerfile",
                "merged_extraction": {},
            }

            result = detect_gpu_node(state, mock_llm_malformed)

            detection = result["gpu_detection"]
            # Should default to safe values
            assert detection["requires_gpu"] is False
            assert detection["dockerfile_detected"] is False
            assert detection["confidence"] == "low"

    def test_raises_without_llm(self):
        """Test that ValueError is raised when no LLM provided."""
        state = {"codebase_path": "/test", "merged_extraction": {}}

        with pytest.raises(ValueError, match="LLM is required"):
            detect_gpu_node(state, None)

    def test_handles_relative_dockerfile_path(self, mock_llm_gpu_cuda12):
        """Test that relative Dockerfile paths are resolved correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dockerfile = Path(tmpdir) / "Dockerfile"
            dockerfile.write_text("FROM nvidia/cuda:12.1-runtime\nCMD python app.py")

            state = {
                "codebase_path": tmpdir,
                "selected_dockerfile": "Dockerfile",  # Relative path
                "merged_extraction": {},
            }

            result = detect_gpu_node(state, mock_llm_gpu_cuda12)

            detection = result["gpu_detection"]
            assert detection["dockerfile_detected"] is True


class TestCreateDetectGpuNode:
    """Tests for the node factory function."""

    def test_creates_callable_node(self, mock_llm_no_gpu):
        """Test that factory creates a callable node function."""
        node = create_detect_gpu_node(mock_llm_no_gpu)
        assert callable(node)

    def test_raises_without_llm(self):
        """Test that factory raises ValueError without LLM."""
        with pytest.raises(ValueError, match="LLM is required"):
            create_detect_gpu_node(None)

    def test_node_invokes_correctly(self, mock_llm_gpu_cuda12):
        """Test that created node can be invoked with state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dockerfile = Path(tmpdir) / "Dockerfile"
            dockerfile.write_text("FROM nvidia/cuda:12.1-runtime\nCMD python app.py")

            node = create_detect_gpu_node(mock_llm_gpu_cuda12)
            state = {
                "codebase_path": tmpdir,
                "selected_dockerfile": "Dockerfile",
                "merged_extraction": {},
            }

            result = node(state)

            assert "gpu_detection" in result


# ==================== Fixtures ====================


@pytest.fixture
def mock_llm_gpu_cuda12():
    """Mock LLM that returns GPU detection with CUDA 12.1."""
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content="""{
        "requires_gpu": true,
        "confidence": "high",
        "evidence": "Base image nvidia/cuda:12.1-runtime indicates CUDA GPU runtime dependency",
        "cuda_version": "12.1"
    }""")
    return llm


@pytest.fixture
def mock_llm_gpu_cuda11():
    """Mock LLM that returns GPU detection with CUDA 11.8."""
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content="""{
        "requires_gpu": true,
        "confidence": "high",
        "evidence": "Base image nvidia/cuda:11.8-devel indicates CUDA GPU build dependency",
        "cuda_version": "11.8"
    }""")
    return llm


@pytest.fixture
def mock_llm_pytorch_gpu():
    """Mock LLM that returns PyTorch GPU detection."""
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content="""{
        "requires_gpu": true,
        "confidence": "high",
        "evidence": "Base image pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime is a GPU-enabled PyTorch image",
        "cuda_version": "12.1"
    }""")
    return llm


@pytest.fixture
def mock_llm_tensorflow_gpu():
    """Mock LLM that returns TensorFlow GPU detection."""
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content="""{
        "requires_gpu": true,
        "confidence": "high",
        "evidence": "Base image tensorflow/tensorflow:2.15.0-gpu is a GPU-enabled TensorFlow image",
        "cuda_version": null
    }""")
    return llm


@pytest.fixture
def mock_llm_no_gpu():
    """Mock LLM that returns no GPU detection."""
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content="""{
        "requires_gpu": false,
        "confidence": "high",
        "evidence": "Standard Python base image with no GPU-related packages or dependencies",
        "cuda_version": null
    }""")
    return llm


@pytest.fixture
def mock_llm_malformed():
    """Mock LLM that returns malformed response."""
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content="This is not valid JSON at all!")
    return llm
