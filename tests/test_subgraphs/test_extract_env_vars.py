"""Tests for the extract_env_vars node."""

import pytest
from pathlib import Path

from src.subgraphs.analysis.extract_env_vars import create_extract_env_vars_node


class TestExtractEnvVarsNode:
    """Tests for the extract_env_vars node."""

    def test_creates_node_function(self):
        """Test that create_extract_env_vars_node returns a callable."""
        node = create_extract_env_vars_node()
        assert callable(node)

    def test_extracts_dockerfile_env_vars(self, tmp_path):
        """Test extraction of ENV vars from Dockerfile."""
        # Create Dockerfile with ENV declarations
        dockerfile = tmp_path / "Dockerfile"
        dockerfile.write_text("""FROM python:3.11
ENV LOG_LEVEL=info
ENV DEBUG=false
ENV DB_PASSWORD
ENV APP_PORT=8080
""")

        # Create deploy/.env.deploy
        deploy_dir = tmp_path / "deploy"
        deploy_dir.mkdir()
        env_deploy = deploy_dir / ".env.deploy"
        env_deploy.write_text("""env:LOG_LEVEL=info
env:DEBUG=false
vault:DB_PASSWORD=secret/data/app/db:password
nomad:APP_PORT=assigned
""")

        node = create_extract_env_vars_node()
        state = {
            "codebase_path": str(tmp_path),
            "selected_dockerfile": "Dockerfile",
        }

        result = node(state)

        assert "env_deploy_config" in result
        assert "env_var_validation" in result
        assert result["env_var_validation"]["is_valid"]
        assert set(result["env_var_validation"]["dockerfile_vars"]) == {
            "LOG_LEVEL", "DEBUG", "DB_PASSWORD", "APP_PORT"
        }

    def test_validates_missing_env_deploy_entries(self, tmp_path):
        """Test that missing .env.deploy entries are detected."""
        dockerfile = tmp_path / "Dockerfile"
        dockerfile.write_text("""FROM python:3.11
ENV LOG_LEVEL=info
ENV DB_PASSWORD
ENV MISSING_VAR
""")

        deploy_dir = tmp_path / "deploy"
        deploy_dir.mkdir()
        env_deploy = deploy_dir / ".env.deploy"
        env_deploy.write_text("""env:LOG_LEVEL=info
vault:DB_PASSWORD=secret/data/app/db:password
""")  # Missing MISSING_VAR

        node = create_extract_env_vars_node()
        state = {
            "codebase_path": str(tmp_path),
            "selected_dockerfile": "Dockerfile",
        }

        result = node(state)

        assert not result["env_var_validation"]["is_valid"]
        assert "MISSING_VAR" in result["env_var_validation"]["missing_vars"]
        assert len(result["env_var_validation"]["errors"]) > 0

    def test_parses_env_deploy_correctly(self, tmp_path):
        """Test correct parsing of .env.deploy entries."""
        dockerfile = tmp_path / "Dockerfile"
        dockerfile.write_text("""FROM python:3.11
ENV LOG_LEVEL
ENV DB_PASS
ENV APP_PORT
""")

        deploy_dir = tmp_path / "deploy"
        deploy_dir.mkdir()
        env_deploy = deploy_dir / ".env.deploy"
        env_deploy.write_text("""env:LOG_LEVEL=debug
vault:DB_PASS=secret/data/myapp/db:password
nomad:APP_PORT=assigned
""")

        node = create_extract_env_vars_node()
        state = {
            "codebase_path": str(tmp_path),
            "selected_dockerfile": "Dockerfile",
        }

        result = node(state)

        entries = result["env_deploy_config"]["entries"]
        assert entries["LOG_LEVEL"]["source"] == "env"
        assert entries["LOG_LEVEL"]["value"] == "debug"
        assert entries["DB_PASS"]["source"] == "vault"
        assert entries["DB_PASS"]["vault_path"] == "secret/data/myapp/db"
        assert entries["DB_PASS"]["vault_field"] == "password"
        assert entries["APP_PORT"]["source"] == "nomad"
        assert entries["APP_PORT"]["value"] == "assigned"

    def test_handles_missing_env_deploy_file(self, tmp_path):
        """Test fallback when .env.deploy file is missing."""
        dockerfile = tmp_path / "Dockerfile"
        dockerfile.write_text("""FROM python:3.11
ENV LOG_LEVEL=info
""")
        # No deploy/.env.deploy created

        node = create_extract_env_vars_node()
        state = {
            "codebase_path": str(tmp_path),
            "selected_dockerfile": "Dockerfile",
        }

        result = node(state)

        # Fallback mode: is_valid is True (not a failure, just using inference)
        assert result["env_var_validation"]["is_valid"]
        assert result["env_var_validation"].get("using_fallback") is True
        assert result["env_deploy_config"] == {}  # Empty config triggers inference in enrich

    def test_handles_missing_dockerfile(self, tmp_path):
        """Test error when Dockerfile is missing."""
        # No Dockerfile created

        node = create_extract_env_vars_node()
        state = {
            "codebase_path": str(tmp_path),
            "selected_dockerfile": "Dockerfile",
        }

        result = node(state)

        assert not result["env_var_validation"]["is_valid"]
        assert len(result["env_var_validation"]["errors"]) > 0
        assert "dockerfile" in result["env_var_validation"]["errors"][0].lower()

    def test_extra_env_deploy_entries_allowed(self, tmp_path):
        """Test that extra entries in .env.deploy are allowed."""
        dockerfile = tmp_path / "Dockerfile"
        dockerfile.write_text("""FROM python:3.11
ENV LOG_LEVEL=info
""")

        deploy_dir = tmp_path / "deploy"
        deploy_dir.mkdir()
        env_deploy = deploy_dir / ".env.deploy"
        env_deploy.write_text("""env:LOG_LEVEL=info
env:EXTRA_VAR=extra_value
""")

        node = create_extract_env_vars_node()
        state = {
            "codebase_path": str(tmp_path),
            "selected_dockerfile": "Dockerfile",
        }

        result = node(state)

        assert result["env_var_validation"]["is_valid"]  # Still valid
        assert "EXTRA_VAR" in result["env_var_validation"]["extra_vars"]

    def test_handles_invalid_env_deploy_format(self, tmp_path):
        """Test error handling for malformed .env.deploy."""
        dockerfile = tmp_path / "Dockerfile"
        dockerfile.write_text("""FROM python:3.11
ENV LOG_LEVEL=info
""")

        deploy_dir = tmp_path / "deploy"
        deploy_dir.mkdir()
        env_deploy = deploy_dir / ".env.deploy"
        env_deploy.write_text("""LOG_LEVEL=info
""")  # Missing type prefix

        node = create_extract_env_vars_node()
        state = {
            "codebase_path": str(tmp_path),
            "selected_dockerfile": "Dockerfile",
        }

        result = node(state)

        assert not result["env_var_validation"]["is_valid"]
        assert len(result["env_var_validation"]["errors"]) > 0
