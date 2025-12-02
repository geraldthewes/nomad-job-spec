"""Integration tests for the LangGraph workflow."""

import pytest

from src.graph import (
    create_initial_state,
    create_workflow,
    compile_graph,
    run_graph,
    should_retry,
)
from src.nodes.question import generate_questions_node


class TestCreateInitialState:
    """Tests for initial state creation."""

    def test_creates_valid_state(self):
        """Test that initial state has all required fields."""
        state = create_initial_state(
            prompt="Deploy my app",
            codebase_path="/path/to/app",
        )

        assert state["prompt"] == "Deploy my app"
        assert state["codebase_path"] == "/path/to/app"
        assert state["deployment_status"] == "pending"
        assert state["iteration_count"] == 0
        assert state["max_iterations"] == 3  # default
        assert state["user_responses"] == {}
        assert state["questions"] == []

    def test_custom_max_iterations(self):
        """Test custom max iterations."""
        state = create_initial_state(
            prompt="Deploy",
            codebase_path="/app",
            max_iterations=5,
        )

        assert state["max_iterations"] == 5

    def test_custom_cluster_id(self):
        """Test custom cluster ID."""
        state = create_initial_state(
            prompt="Deploy",
            codebase_path="/app",
            cluster_id="production-cluster",
        )

        assert state["cluster_id"] == "production-cluster"


class TestGenerateQuestionsNode:
    """Tests for question generation."""

    def test_generates_questions_for_missing_image(self):
        """Test questions when no Docker image detected."""
        state = {
            "codebase_analysis": {
                "dockerfile": None,
                "env_vars_required": [],
                "suggested_resources": {"cpu": 500, "memory": 256},
            }
        }

        result = generate_questions_node(state)

        assert "questions" in result
        questions = result["questions"]
        assert any("image" in q.lower() for q in questions)

    def test_generates_questions_for_env_vars(self):
        """Test questions when env vars detected."""
        state = {
            "codebase_analysis": {
                "dockerfile": {"base_image": "node:18", "exposed_ports": [3000]},
                "env_vars_required": ["DATABASE_URL", "API_KEY", "SECRET"],
                "suggested_resources": {"cpu": 500, "memory": 256},
            }
        }

        result = generate_questions_node(state)

        questions = result["questions"]
        assert any("environment" in q.lower() or "DATABASE_URL" in q for q in questions)

    def test_always_asks_resource_question(self):
        """Test that resource question is always included."""
        state = {
            "codebase_analysis": {
                "dockerfile": {"base_image": "nginx", "exposed_ports": [80]},
                "env_vars_required": [],
                "suggested_resources": {"cpu": 500, "memory": 256},
            }
        }

        result = generate_questions_node(state)

        questions = result["questions"]
        assert any("resource" in q.lower() or "cpu" in q.lower() for q in questions)


class TestShouldRetry:
    """Tests for retry decision logic."""

    def test_success_returns_success(self):
        """Test that success status returns 'success'."""
        state = {"deployment_status": "success", "iteration_count": 0, "max_iterations": 3}

        result = should_retry(state)

        assert result == "success"

    def test_give_up_returns_give_up(self):
        """Test that give_up status returns 'give_up'."""
        state = {"deployment_status": "give_up", "iteration_count": 0, "max_iterations": 3}

        result = should_retry(state)

        assert result == "give_up"

    def test_max_iterations_exceeded(self):
        """Test that exceeding max iterations returns 'give_up'."""
        state = {
            "deployment_status": "failed",
            "iteration_count": 3,
            "max_iterations": 3,
            "deployment_error": "Resource error",
        }

        result = should_retry(state)

        assert result == "give_up"

    def test_fixable_error_returns_retry(self):
        """Test that fixable error returns 'retry'."""
        state = {
            "deployment_status": "failed",
            "iteration_count": 1,
            "max_iterations": 3,
            "deployment_error": "Insufficient memory for allocation",
        }

        result = should_retry(state)

        assert result == "retry"

    def test_unfixable_error_returns_give_up(self):
        """Test that unfixable error returns 'give_up'."""
        state = {
            "deployment_status": "failed",
            "iteration_count": 1,
            "max_iterations": 3,
            "deployment_error": "Permission denied: invalid ACL token",
        }

        result = should_retry(state)

        assert result == "give_up"


class TestCreateWorkflow:
    """Tests for workflow creation."""

    def test_creates_workflow(self, mock_llm, mock_settings):
        """Test that workflow is created successfully."""
        workflow = create_workflow(mock_llm, mock_settings, include_deployment=False)

        assert workflow is not None

    def test_workflow_has_required_nodes(self, mock_llm, mock_settings):
        """Test that workflow has required nodes."""
        workflow = create_workflow(mock_llm, mock_settings, include_deployment=False)

        # Get node names from the workflow
        nodes = workflow.nodes
        assert "analyze" in nodes
        assert "question" in nodes
        assert "collect" in nodes
        assert "generate" in nodes


class TestCompileGraph:
    """Tests for graph compilation."""

    def test_compile_without_checkpointing(self, mock_llm, mock_settings):
        """Test compiling graph without checkpointing."""
        graph = compile_graph(
            llm=mock_llm,
            settings=mock_settings,
            include_deployment=False,
            enable_checkpointing=False,
        )

        assert graph is not None

    def test_compile_with_checkpointing(self, mock_llm, mock_settings):
        """Test compiling graph with checkpointing."""
        graph = compile_graph(
            llm=mock_llm,
            settings=mock_settings,
            include_deployment=False,
            enable_checkpointing=True,
        )

        assert graph is not None


class TestRunGraph:
    """Integration tests for running the full graph."""

    def test_run_simple_workflow(self, mock_llm, mock_settings, express_app_repo):
        """Test running workflow end-to-end."""
        if not express_app_repo.exists():
            pytest.skip("Fixture not available")

        result = run_graph(
            prompt="Deploy this Node.js app",
            codebase_path=str(express_app_repo),
            llm=mock_llm,
            settings=mock_settings,
            include_deployment=False,
        )

        assert "job_spec" in result
        assert result["job_spec"]  # Not empty
        assert "job_name" in result

    def test_run_with_user_responses(self, mock_llm, mock_settings, express_app_repo):
        """Test running workflow with pre-provided responses."""
        if not express_app_repo.exists():
            pytest.skip("Fixture not available")

        result = run_graph(
            prompt="Deploy API",
            codebase_path=str(express_app_repo),
            llm=mock_llm,
            settings=mock_settings,
            user_responses={"port": "3000", "instances": "2"},
            include_deployment=False,
        )

        assert "job_spec" in result
        assert result["user_responses"]["port"] == "3000"

    def test_run_generates_valid_hcl(self, mock_llm, mock_settings, nginx_simple_repo):
        """Test that generated HCL is valid."""
        if not nginx_simple_repo.exists():
            pytest.skip("Fixture not available")

        result = run_graph(
            prompt="Deploy nginx",
            codebase_path=str(nginx_simple_repo),
            llm=mock_llm,
            settings=mock_settings,
            include_deployment=False,
        )

        job_spec = result.get("job_spec", "")
        assert 'job "' in job_spec
        assert "driver" in job_spec
        assert "docker" in job_spec
