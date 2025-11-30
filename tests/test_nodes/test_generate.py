"""Tests for the generate node."""

import pytest

from src.nodes.generate import (
    generate_spec_node,
    regenerate_spec_with_fix,
    create_generate_node,
    create_fix_node,
    _build_generation_context,
    _parse_llm_response,
    _increment_resources,
)


class TestGenerateSpecNode:
    """Tests for the generate_spec_node function."""

    def test_generate_basic_spec(self, mock_llm, mock_settings):
        """Test generating a basic job spec."""
        state = {
            "codebase_path": "/test/app",
            "prompt": "Deploy a web app",
            "codebase_analysis": {
                "dockerfile": {
                    "base_image": "node:18",
                    "exposed_ports": [3000],
                },
                "dependencies": {
                    "language": "nodejs",
                },
                "suggested_resources": {"cpu": 500, "memory": 256},
            },
            "user_responses": {},
            "relevant_memories": [],
        }

        result = generate_spec_node(state, mock_llm)

        assert "job_spec" in result
        assert "job_name" in result
        assert result["job_spec"]  # Not empty
        assert 'job "' in result["job_spec"]

    def test_generate_with_user_responses(self, mock_llm, mock_settings):
        """Test generating spec with user responses."""
        state = {
            "codebase_path": "/test/app",
            "prompt": "Deploy API",
            "codebase_analysis": {"suggested_resources": {"cpu": 500, "memory": 256}},
            "user_responses": {
                "port": "8080",
                "instances": "3",
            },
            "relevant_memories": [],
        }

        result = generate_spec_node(state, mock_llm)

        assert "job_spec" in result
        assert result["job_spec"]


class TestRegenerateSpecWithFix:
    """Tests for the regenerate_spec_with_fix function."""

    def test_fix_memory_error(self, mock_llm, mock_settings):
        """Test fixing a memory-related error."""
        state = {
            "codebase_path": "/test/app",
            "prompt": "Deploy app",
            "codebase_analysis": {},
            "job_spec": "job {}",
            "job_config": {"cpu": 500, "memory": 256, "job_name": "test"},
            "deployment_error": "OOM: insufficient memory",
            "relevant_memories": [],
            "iteration_count": 0,
        }

        result = regenerate_spec_with_fix(state, mock_llm)

        assert result["iteration_count"] == 1
        assert "job_spec" in result

    def test_fix_with_memories(self, mock_llm, mock_settings):
        """Test fixing with relevant memories."""
        state = {
            "codebase_path": "/test/app",
            "prompt": "Deploy app",
            "codebase_analysis": {},
            "job_spec": "job {}",
            "job_config": {"cpu": 500, "memory": 256, "job_name": "test"},
            "deployment_error": "Port conflict",
            "relevant_memories": [
                "Past error: Port 8080 was in use, switched to dynamic ports",
            ],
            "iteration_count": 0,
        }

        result = regenerate_spec_with_fix(state, mock_llm)

        assert result["iteration_count"] == 1


class TestCreateGenerateNode:
    """Tests for create_generate_node factory."""

    def test_creates_callable(self, mock_llm):
        """Test that factory returns callable."""
        node = create_generate_node(mock_llm)
        assert callable(node)

    def test_node_returns_valid_state(self, mock_llm):
        """Test that created node returns valid state."""
        node = create_generate_node(mock_llm)
        state = {
            "codebase_path": "/test",
            "prompt": "Deploy",
            "codebase_analysis": {"suggested_resources": {"cpu": 500, "memory": 256}},
            "user_responses": {},
            "relevant_memories": [],
        }

        result = node(state)

        assert "job_spec" in result
        assert "job_name" in result


class TestCreateFixNode:
    """Tests for create_fix_node factory."""

    def test_creates_callable(self, mock_llm):
        """Test that factory returns callable."""
        node = create_fix_node(mock_llm)
        assert callable(node)


class TestBuildGenerationContext:
    """Tests for _build_generation_context function."""

    def test_basic_context(self):
        """Test building basic context."""
        analysis = {"language": "python", "ports": [8000]}
        responses = {"count": "3"}
        prompt = "Deploy my app"
        memories = []

        result = _build_generation_context(analysis, responses, prompt, memories)

        assert "Deploy my app" in result
        assert "python" in result
        assert "count" in result

    def test_context_with_memories(self):
        """Test context includes memories."""
        analysis = {}
        responses = {}
        prompt = "Deploy"
        memories = ["Previous error: used wrong port", "Cluster prefers Alpine images"]

        result = _build_generation_context(analysis, responses, prompt, memories)

        assert "Previous error" in result
        assert "Alpine" in result


class TestParseLLMResponse:
    """Tests for _parse_llm_response function."""

    def test_parse_plain_json(self):
        """Test parsing plain JSON response."""
        response = '{"job_name": "test", "image": "nginx", "cpu": 500}'

        result = _parse_llm_response(response)

        assert result["job_name"] == "test"
        assert result["image"] == "nginx"
        assert result["cpu"] == 500

    def test_parse_json_in_markdown(self):
        """Test parsing JSON from markdown code block."""
        response = """Here's the config:

```json
{"job_name": "test-app", "image": "node:18", "ports": {"http": 3000}}
```

This should work well."""

        result = _parse_llm_response(response)

        assert result["job_name"] == "test-app"
        assert result["image"] == "node:18"
        assert result["ports"] == {"http": 3000}

    def test_parse_invalid_json(self):
        """Test that invalid JSON raises error."""
        response = "This is not JSON at all"

        with pytest.raises(ValueError):
            _parse_llm_response(response)


class TestIncrementResources:
    """Tests for _increment_resources function."""

    def test_increment_on_memory_error(self, mock_settings):
        """Test resource increment for memory error."""
        config = {"cpu": 500, "memory": 256, "job_name": "test", "image": "nginx"}
        error = "OOM killed: insufficient memory"

        result = _increment_resources(config, error, mock_settings)

        assert result.memory > 256  # Should increase
        assert result.memory == 512  # Doubled

    def test_increment_on_cpu_error(self, mock_settings):
        """Test resource increment for CPU error."""
        config = {"cpu": 500, "memory": 256, "job_name": "test", "image": "nginx"}
        error = "CPU quota exceeded"

        result = _increment_resources(config, error, mock_settings)

        assert result.cpu > 500  # Should increase

    def test_increment_generic_error(self, mock_settings):
        """Test resource increment for generic error."""
        config = {"cpu": 500, "memory": 256, "job_name": "test", "image": "nginx"}
        error = "Allocation failed"

        result = _increment_resources(config, error, mock_settings)

        # Both should increase for generic errors
        assert result.cpu > 500
        assert result.memory > 256

    def test_increment_caps_at_max(self, mock_settings):
        """Test that resources are capped at maximum."""
        config = {"cpu": 3000, "memory": 3000, "job_name": "test", "image": "nginx"}
        error = "Need more memory"

        result = _increment_resources(config, error, mock_settings)

        assert result.memory <= 4096  # Max memory
        assert result.cpu <= 4000  # Max CPU
