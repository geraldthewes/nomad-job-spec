"""Tests for codebase analysis tools."""

import pytest
from pathlib import Path

from src.tools.codebase import (
    parse_dockerfile,
    detect_language_and_deps,
    detect_env_vars,
    suggest_resources,
    analyze_codebase,
    DockerfileInfo,
    DependencyInfo,
    _parse_env_line,
)


class TestParseDockerfile:
    """Tests for Dockerfile parsing."""

    def test_parse_basic_dockerfile(self, tmp_path):
        """Test parsing a basic Dockerfile."""
        dockerfile = tmp_path / "Dockerfile"
        dockerfile.write_text("""FROM python:3.11-slim
WORKDIR /app
COPY . .
EXPOSE 8000
ENV DEBUG=false
CMD ["python", "app.py"]
""")

        result = parse_dockerfile(str(dockerfile))

        assert result.base_image == "python:3.11-slim"
        assert 8000 in result.exposed_ports
        assert result.cmd == '["python", "app.py"]'
        assert result.env_vars.get("DEBUG") == "false"
        assert result.workdir == "/app"

    def test_parse_node_dockerfile(self, express_app_repo):
        """Test parsing Node.js Dockerfile from fixture."""
        dockerfile_path = express_app_repo / "Dockerfile"
        if not dockerfile_path.exists():
            pytest.skip("Fixture not available")

        result = parse_dockerfile(str(dockerfile_path))

        assert "node" in result.base_image.lower()
        assert 3000 in result.exposed_ports
        assert result.env_vars.get("NODE_ENV") == "production"

    def test_parse_multiple_ports(self, tmp_path):
        """Test parsing Dockerfile with multiple EXPOSE statements."""
        dockerfile = tmp_path / "Dockerfile"
        dockerfile.write_text("""FROM nginx
EXPOSE 80
EXPOSE 443
""")

        result = parse_dockerfile(str(dockerfile))

        assert 80 in result.exposed_ports
        assert 443 in result.exposed_ports

    def test_parse_entrypoint(self, tmp_path):
        """Test parsing ENTRYPOINT instruction."""
        dockerfile = tmp_path / "Dockerfile"
        dockerfile.write_text("""FROM alpine
ENTRYPOINT ["./start.sh"]
CMD ["--config", "/etc/app.conf"]
""")

        result = parse_dockerfile(str(dockerfile))

        assert result.entrypoint == '["./start.sh"]'
        assert result.cmd == '["--config", "/etc/app.conf"]'

    def test_parse_multi_env_single_line(self, tmp_path):
        """Test parsing multiple ENV vars on single line."""
        dockerfile = tmp_path / "Dockerfile"
        dockerfile.write_text("""FROM python:3.11
ENV VAR1=value1 VAR2=value2 VAR3=value3
""")

        result = parse_dockerfile(str(dockerfile))

        assert "VAR1" in result.env_var_names
        assert "VAR2" in result.env_var_names
        assert "VAR3" in result.env_var_names
        assert result.env_vars.get("VAR1") == "value1"
        assert result.env_vars.get("VAR2") == "value2"
        assert result.env_vars.get("VAR3") == "value3"

    def test_parse_env_without_value(self, tmp_path):
        """Test parsing ENV without default value."""
        dockerfile = tmp_path / "Dockerfile"
        dockerfile.write_text("""FROM python:3.11
ENV MY_VAR
ENV ANOTHER_VAR
""")

        result = parse_dockerfile(str(dockerfile))

        assert "MY_VAR" in result.env_var_names
        assert "ANOTHER_VAR" in result.env_var_names
        # Variables without values should not be in env_vars dict
        assert "MY_VAR" not in result.env_vars
        assert "ANOTHER_VAR" not in result.env_vars

    def test_parse_env_legacy_format(self, tmp_path):
        """Test parsing legacy ENV format (space separated)."""
        dockerfile = tmp_path / "Dockerfile"
        dockerfile.write_text("""FROM python:3.11
ENV LOG_LEVEL info
ENV DEBUG false
""")

        result = parse_dockerfile(str(dockerfile))

        assert "LOG_LEVEL" in result.env_var_names
        assert "DEBUG" in result.env_var_names
        assert result.env_vars.get("LOG_LEVEL") == "info"
        assert result.env_vars.get("DEBUG") == "false"

    def test_parse_env_quoted_values(self, tmp_path):
        """Test parsing ENV with quoted values."""
        dockerfile = tmp_path / "Dockerfile"
        dockerfile.write_text('''FROM python:3.11
ENV MSG="Hello World"
ENV PATH_VAR='some/path'
''')

        result = parse_dockerfile(str(dockerfile))

        assert result.env_vars.get("MSG") == "Hello World"
        assert result.env_vars.get("PATH_VAR") == "some/path"

    def test_env_var_names_list(self, tmp_path):
        """Test that env_var_names contains all declared vars."""
        dockerfile = tmp_path / "Dockerfile"
        dockerfile.write_text("""FROM python:3.11
ENV LOG_LEVEL=info
ENV DEBUG=false
ENV SECRET
ENV API_KEY=abc123
""")

        result = parse_dockerfile(str(dockerfile))

        assert len(result.env_var_names) == 4
        assert "LOG_LEVEL" in result.env_var_names
        assert "DEBUG" in result.env_var_names
        assert "SECRET" in result.env_var_names
        assert "API_KEY" in result.env_var_names


class TestParseEnvLine:
    """Tests for _parse_env_line helper function."""

    def test_single_var_with_value(self):
        """Test parsing single var with value."""
        result = _parse_env_line("VAR=value")
        assert result == [("VAR", "value")]

    def test_single_var_no_value(self):
        """Test parsing single var without value."""
        result = _parse_env_line("VAR")
        assert result == [("VAR", None)]

    def test_multiple_vars(self):
        """Test parsing multiple vars on one line."""
        result = _parse_env_line("VAR1=val1 VAR2=val2 VAR3=val3")
        assert ("VAR1", "val1") in result
        assert ("VAR2", "val2") in result
        assert ("VAR3", "val3") in result

    def test_legacy_format(self):
        """Test parsing legacy space-separated format."""
        result = _parse_env_line("LOG_LEVEL info")
        assert result == [("LOG_LEVEL", "info")]

    def test_quoted_value_double(self):
        """Test parsing double-quoted value."""
        result = _parse_env_line('MSG="Hello World"')
        assert result == [("MSG", "Hello World")]

    def test_quoted_value_single(self):
        """Test parsing single-quoted value."""
        result = _parse_env_line("MSG='Hello World'")
        assert result == [("MSG", "Hello World")]

    def test_empty_value(self):
        """Test parsing empty value."""
        result = _parse_env_line("VAR=")
        assert result == [("VAR", "")]


class TestDetectLanguageAndDeps:
    """Tests for language and dependency detection."""

    def test_detect_nodejs_deps(self, express_app_repo):
        """Test detection of Node.js project."""
        if not express_app_repo.exists():
            pytest.skip("Fixture not available")

        result = detect_language_and_deps(str(express_app_repo))

        assert result.language == "nodejs"
        assert result.package_manager == "npm"
        assert "express" in result.dependencies

    def test_detect_python_requirements(self, tmp_path):
        """Test detection of Python project with requirements.txt."""
        req_file = tmp_path / "requirements.txt"
        req_file.write_text("""flask>=2.0.0
requests
sqlalchemy~=2.0
# Comment line
pytest>=7.0
""")

        result = detect_language_and_deps(str(tmp_path))

        assert result.language == "python"
        assert result.package_manager == "pip"
        assert "flask" in result.dependencies
        assert "requests" in result.dependencies
        assert "sqlalchemy" in result.dependencies

    def test_detect_go_mod(self, tmp_path):
        """Test detection of Go project."""
        go_mod = tmp_path / "go.mod"
        go_mod.write_text("""module github.com/example/myapp

go 1.21

require (
    github.com/gin-gonic/gin v1.9.0
    github.com/go-redis/redis/v9 v9.0.0
)
""")

        result = detect_language_and_deps(str(tmp_path))

        assert result.language == "go"
        assert result.package_manager == "go modules"


class TestDetectEnvVars:
    """Tests for environment variable detection."""

    def test_detect_python_env_vars(self, tmp_path):
        """Test detection of env vars in Python code."""
        py_file = tmp_path / "app.py"
        py_file.write_text("""
import os

DATABASE_URL = os.getenv('DATABASE_URL')
SECRET_KEY = os.environ.get('SECRET_KEY', 'default')
DEBUG = os.getenv('DEBUG', 'false')
""")

        result = detect_env_vars(str(tmp_path))

        assert "DATABASE_URL" in result
        assert "SECRET_KEY" in result
        assert "DEBUG" in result

    def test_detect_nodejs_env_vars(self, express_app_repo):
        """Test detection of env vars in Node.js code."""
        if not express_app_repo.exists():
            pytest.skip("Fixture not available")

        result = detect_env_vars(str(express_app_repo))

        assert "PORT" in result
        assert "API_KEY" in result

    def test_detect_env_example(self, tmp_path):
        """Test detection from .env.example file."""
        env_example = tmp_path / ".env.example"
        env_example.write_text("""
DATABASE_URL=postgres://localhost/db
REDIS_URL=redis://localhost:6379
LOG_LEVEL=info
""")

        result = detect_env_vars(str(tmp_path))

        assert "DATABASE_URL" in result
        assert "REDIS_URL" in result
        assert "LOG_LEVEL" in result


class TestSuggestResources:
    """Tests for resource suggestion."""

    def test_suggest_nodejs_resources(self):
        """Test resource suggestions for Node.js."""
        deps = DependencyInfo(language="nodejs", package_manager="npm")

        result = suggest_resources(None, deps)

        # Node.js should get higher memory
        assert result["memory"] >= 512

    def test_suggest_go_resources(self):
        """Test resource suggestions for Go."""
        deps = DependencyInfo(language="go", package_manager="go modules")

        result = suggest_resources(None, deps)

        # Go is efficient
        assert result["cpu"] <= 500
        assert result["memory"] <= 256

    def test_suggest_alpine_reduces_memory(self):
        """Test that Alpine base image reduces memory estimate."""
        dockerfile_info = DockerfileInfo(base_image="python:3.11-alpine")
        deps = DependencyInfo(language="python")

        result = suggest_resources(dockerfile_info, deps)

        # Alpine should reduce memory requirement
        assert result["memory"] < 384  # Less than non-alpine Python


class TestAnalyzeCodebase:
    """Integration tests for full codebase analysis."""

    def test_analyze_express_app(self, express_app_repo):
        """Test analyzing Express.js application."""
        if not express_app_repo.exists():
            pytest.skip("Fixture not available")

        result = analyze_codebase(str(express_app_repo))

        assert result.dockerfile is not None
        assert "node" in result.dockerfile.base_image.lower()
        assert result.dependencies is not None
        assert result.dependencies.language == "nodejs"
        assert len(result.files_analyzed) > 0
        assert len(result.errors) == 0

    def test_analyze_nginx_simple(self, nginx_simple_repo):
        """Test analyzing simple nginx setup."""
        if not nginx_simple_repo.exists():
            pytest.skip("Fixture not available")

        result = analyze_codebase(str(nginx_simple_repo))

        assert result.dockerfile is not None
        assert "nginx" in result.dockerfile.base_image.lower()
        assert 80 in result.dockerfile.exposed_ports

    def test_analyze_nonexistent_path(self):
        """Test that nonexistent path raises error."""
        with pytest.raises(FileNotFoundError):
            analyze_codebase("/nonexistent/path/to/repo")

    def test_to_json_serialization(self, express_app_repo):
        """Test JSON serialization of analysis."""
        if not express_app_repo.exists():
            pytest.skip("Fixture not available")

        result = analyze_codebase(str(express_app_repo))
        json_str = result.to_json()

        import json
        data = json.loads(json_str)

        assert "path" in data
        assert "dockerfile" in data
        assert "dependencies" in data

    def test_analyze_postgres_db(self, postgres_db_repo):
        """Test analyzing a PostgreSQL database repo."""
        if not postgres_db_repo.exists():
            pytest.skip("Fixture not available")

        result = analyze_codebase(str(postgres_db_repo))

        # Should detect postgres from Dockerfile
        assert result.dockerfile is not None
        assert "postgres" in result.dockerfile.base_image.lower()
        assert 5432 in result.dockerfile.exposed_ports
        # Should detect env vars from Dockerfile
        assert "POSTGRES_USER" in result.dockerfile.env_vars or len(result.env_vars_required) > 0

    def test_analyze_flask_api(self, flask_api_repo):
        """Test analyzing a Flask API repo."""
        if not flask_api_repo.exists():
            pytest.skip("Fixture not available")

        result = analyze_codebase(str(flask_api_repo))

        # Should detect Python from dependencies
        assert result.dependencies is not None
        assert result.dependencies.language == "python"
        # Should detect flask from requirements
        assert any("flask" in dep.lower() for dep in result.dependencies.dependencies)
        # Should detect env vars from .env.example
        assert len(result.env_vars_required) > 0
        # Should detect DATABASE_URL
        assert any("DATABASE" in var for var in result.env_vars_required)
