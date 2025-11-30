"""Tests for the analyze node."""

import pytest

from src.nodes.analyze import (
    analyze_codebase_node,
    create_analyze_node,
    _merge_analyses,
    _extract_json,
)
from src.tools.codebase import CodebaseAnalysis, DockerfileInfo, DependencyInfo


class TestAnalyzeCodebaseNode:
    """Tests for the analyze_codebase_node function."""

    def test_analyze_without_llm(self, express_app_repo):
        """Test analysis without LLM (static only)."""
        if not express_app_repo.exists():
            pytest.skip("Fixture not available")

        state = {"codebase_path": str(express_app_repo)}

        result = analyze_codebase_node(state, llm=None)

        assert "codebase_analysis" in result
        analysis = result["codebase_analysis"]
        assert "dockerfile" in analysis
        assert "dependencies" in analysis
        assert analysis["dependencies"]["language"] == "nodejs"

    def test_analyze_with_mock_llm(self, express_app_repo, mock_llm):
        """Test analysis with mock LLM."""
        if not express_app_repo.exists():
            pytest.skip("Fixture not available")

        state = {"codebase_path": str(express_app_repo)}

        result = analyze_codebase_node(state, llm=mock_llm)

        assert "codebase_analysis" in result
        analysis = result["codebase_analysis"]
        # Should have both static and LLM analysis
        assert "dockerfile" in analysis
        assert "llm_analysis" in analysis or "summary" in analysis

    def test_analyze_missing_path(self):
        """Test error handling for missing path."""
        state = {}

        result = analyze_codebase_node(state, llm=None)

        assert "codebase_analysis" in result
        assert "error" in result["codebase_analysis"]

    def test_analyze_nonexistent_path(self):
        """Test error handling for nonexistent path."""
        state = {"codebase_path": "/nonexistent/path"}

        result = analyze_codebase_node(state, llm=None)

        assert "codebase_analysis" in result
        assert "error" in result["codebase_analysis"]


class TestCreateAnalyzeNode:
    """Tests for the create_analyze_node factory."""

    def test_creates_callable(self, mock_llm):
        """Test that factory returns a callable."""
        node = create_analyze_node(mock_llm)

        assert callable(node)

    def test_node_works_without_llm(self, express_app_repo):
        """Test that node works when created without LLM."""
        if not express_app_repo.exists():
            pytest.skip("Fixture not available")

        node = create_analyze_node(None)
        state = {"codebase_path": str(express_app_repo)}

        result = node(state)

        assert "codebase_analysis" in result


class TestMergeAnalyses:
    """Tests for _merge_analyses function."""

    def test_merge_with_llm_analysis(self):
        """Test merging static and LLM analyses."""
        static = CodebaseAnalysis(
            path="/test",
            dockerfile=DockerfileInfo(base_image="node:18", exposed_ports=[3000]),
            dependencies=DependencyInfo(language="nodejs"),
            suggested_resources={"cpu": 500, "memory": 256},
        )

        llm = {
            "summary": "A Node.js API service",
            "resources": {"cpu": 750, "memory": 512},
            "health_check": {"type": "http", "path": "/health"},
            "warnings": ["Consider using Alpine image"],
        }

        result = _merge_analyses(static, llm)

        # LLM analysis should be included
        assert result["llm_analysis"] == llm
        assert result["summary"] == "A Node.js API service"
        assert result["warnings"] == ["Consider using Alpine image"]
        assert result["health_check"] == {"type": "http", "path": "/health"}

        # Resources should be max of both
        assert result["suggested_resources"]["cpu"] == 750
        assert result["suggested_resources"]["memory"] == 512

    def test_merge_without_llm(self):
        """Test merging when LLM analysis is None."""
        static = CodebaseAnalysis(
            path="/test",
            dockerfile=DockerfileInfo(base_image="python:3.11"),
            suggested_resources={"cpu": 500, "memory": 256},
        )

        result = _merge_analyses(static, None)

        assert "llm_analysis" not in result
        assert result["suggested_resources"]["cpu"] == 500


class TestExtractJson:
    """Tests for JSON extraction from LLM responses."""

    def test_extract_plain_json(self):
        """Test extracting plain JSON."""
        text = '{"key": "value", "number": 42}'

        result = _extract_json(text)

        assert result == '{"key": "value", "number": 42}'

    def test_extract_json_from_markdown(self):
        """Test extracting JSON from markdown code block."""
        text = """Here's the analysis:

```json
{"summary": "test", "ports": {"http": 80}}
```

Let me know if you need more info."""

        result = _extract_json(text)

        assert result is not None
        assert "summary" in result

    def test_extract_json_from_plain_code_block(self):
        """Test extracting JSON from plain code block."""
        text = """```
{"key": "value"}
```"""

        result = _extract_json(text)

        assert result is not None

    def test_extract_nested_json(self):
        """Test extracting nested JSON objects."""
        text = '{"outer": {"inner": "value"}, "array": [1, 2]}'

        result = _extract_json(text)

        assert result is not None
        assert "outer" in result

    def test_extract_no_json(self):
        """Test handling text with no JSON."""
        text = "This is just plain text without any JSON."

        result = _extract_json(text)

        assert result is None
