"""Pytest configuration and fixtures."""

import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage


# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_repos_dir():
    """Return path to sample repos fixture directory."""
    return FIXTURES_DIR / "sample_repos"


@pytest.fixture
def nginx_simple_repo(sample_repos_dir):
    """Return path to simple nginx sample repo."""
    return sample_repos_dir / "nginx-simple"


@pytest.fixture
def express_app_repo(sample_repos_dir):
    """Return path to Express.js sample repo."""
    return sample_repos_dir / "express-app"


@pytest.fixture
def postgres_db_repo(sample_repos_dir):
    """Return path to PostgreSQL database sample repo."""
    return sample_repos_dir / "postgres-db"


@pytest.fixture
def flask_api_repo(sample_repos_dir):
    """Return path to Flask API sample repo."""
    return sample_repos_dir / "flask-api"


@pytest.fixture
def mock_llm():
    """Create a mock LLM that returns predictable cluster-aware responses."""
    llm = MagicMock()

    def mock_invoke(messages):
        """Return mock responses based on message content."""
        content = str(messages[-1].content) if messages else ""

        if "analyze" in content.lower() or "codebase" in content.lower():
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

        elif "generate" in content.lower() or "nomad" in content.lower():
            return AIMessage(content="""{
                "job_name": "test-app",
                "service_type": "MEDIUM",
                "image": "nginx:latest",
                "ports": [{"name": "http", "container_port": 80, "static": false}],
                "env_vars": {},
                "cpu": null,
                "memory": null,
                "health_check_type": "http",
                "health_check_path": "/health",
                "count": 1,
                "service_tags": ["web"],
                "require_amd64": true,
                "fabio_hostname": null,
                "fabio_path": null,
                "volume": null,
                "vault_policies": [],
                "vault_secrets": {}
            }""")

        else:
            return AIMessage(content="Mock response")

    llm.invoke = mock_invoke
    return llm


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    from config.settings import Settings

    return Settings(
        llm_provider="vllm",
        vllm_base_url="http://localhost:8000/v1",
        vllm_model="test-model",
        nomad_addr="http://localhost:4646",
        nomad_datacenter="dc1",
        nomad_namespace="default",
        nomad_region="global",
        qdrant_host="localhost",
        qdrant_port=6333,
        langfuse_enabled=False,
        memory_enabled=False,
        max_iterations=3,
        default_cpu=500,
        default_memory=256,
    )


@pytest.fixture
def sample_dockerfile_content():
    """Return sample Dockerfile content for testing."""
    return """FROM node:18-alpine

WORKDIR /app

COPY package*.json ./
RUN npm ci --only=production

COPY . .

ENV NODE_ENV=production
ENV PORT=3000

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=3s CMD wget -q --spider http://localhost:3000/health || exit 1

CMD ["node", "server.js"]
"""


@pytest.fixture
def sample_package_json():
    """Return sample package.json content for testing."""
    return """{
  "name": "sample-express-app",
  "version": "1.0.0",
  "main": "server.js",
  "scripts": {
    "start": "node server.js"
  },
  "dependencies": {
    "express": "^4.18.0",
    "cors": "^2.8.5"
  },
  "devDependencies": {
    "jest": "^29.0.0"
  }
}
"""


@pytest.fixture
def sample_hcl():
    """Return sample valid HCL content for testing."""
    return '''job "test-app" {
  region      = "global"
  datacenters = ["dc1"]
  type        = "service"
  namespace   = "default"

  group "test-app-group" {
    count = 1

    network {
      port "http" {
        to = 8080
      }
    }

    restart {
      attempts = 3
      interval = "30m"
      delay    = "15s"
      mode     = "fail"
    }

    task "test-app-task" {
      driver = "docker"

      config {
        image = "nginx:latest"
        ports = ["http"]
      }

      resources {
        cpu    = 500
        memory = 256
      }

      service {
        name = "test-app"
        port = "http"
        tags = []

        check {
          type     = "http"
          path     = "/health"
          interval = "10s"
          timeout  = "2s"
        }
      }
    }
  }
}
'''
