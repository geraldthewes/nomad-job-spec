# Nomad Job Spec Agent

AI-powered Nomad job specification generator using LangGraph. Analyzes codebases and generates cluster-compatible HCL job specifications with support for CSI volumes, Vault secrets, Fabio routing, and more.

## Features

- **Codebase Analysis**: Automatically detect Docker images, ports, environment variables, and resource requirements
- **Cluster-Aware Generation**: Generate HCL compatible with your cluster's patterns (Terraform templating, architecture constraints)
- **Service Classification**: Automatically categorize services (LIGHT/MEDIUM/HEAVY/COMPUTE) for appropriate resource allocation
- **CSI Volume Support**: Generate volume blocks with init tasks for proper permissions
- **Vault Integration**: Template secrets from Vault with custom delimiters
- **Fabio Routing**: Generate proper routing tags for the Fabio load balancer
- **Interactive CLI**: Answer clarifying questions for customized specs
- **Memory Layer**: Learn from past deployments via Mem0/Qdrant
- **Observability**: LangFuse tracing for debugging and optimization

## Installation

### From Source (Development)

```bash
# Clone the repository
git clone https://github.com/gerald/nomad-job-spec.git
cd nomad-job-spec

# Install in development mode
pip install -e ".[dev]"

# Or use make
make install-dev
```

### Production Installation

```bash
pip install nomad-job-spec
```

## Quick Start

### 1. Configure Environment

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Key configuration options:

| Variable | Description | Default |
|----------|-------------|---------|
| `VLLM_BASE_URL` | vLLM server endpoint | `http://localhost:8000` |
| `VLLM_MODEL` | Model name | `Qwen/Qwen3-32B` |
| `NOMAD_ADDRESS` | Nomad cluster address | `http://localhost:4646` |
| `NOMAD_DATACENTER` | Default datacenter | `dc1` |
| `QDRANT_HOST` | Qdrant server for memory | `localhost` |
| `MEMORY_ENABLED` | Enable learning from past deployments | `true` |
| `LANGFUSE_ENABLED` | Enable observability tracing | `true` |

### 2. Analyze a Codebase

```bash
# Analyze a local codebase
nomad-spec analyze --path ./my-app

# Analyze with verbose output
nomad-spec analyze --path ./my-app --verbose
```

### 3. Generate a Job Specification

```bash
# Basic generation
nomad-spec generate --prompt "Deploy this Node.js API" --path ./my-app

# Save to file
nomad-spec generate -p "Deploy with 2 replicas" --path ./my-app -o job.nomad

# With specific options
nomad-spec generate \
  --prompt "Deploy PostgreSQL with persistent storage" \
  --path ./postgres-app \
  --output postgres.nomad \
  --datacenter dc1
```

### 4. Validate an Existing Spec

```bash
# Validate HCL syntax
nomad-spec validate job.nomad

# Validate against Nomad cluster
nomad-spec validate job.nomad --address http://nomad.example.com:4646
```

## Generated HCL Features

The generator produces HCL compatible with the cluster's patterns:

### Terraform Templating

```hcl
job "my-app" {
  datacenters = ["${datacenter}"]  # Terraform variable
  # ...
}
```

### Architecture Constraints

```hcl
constraint {
  attribute = "$${attr.cpu.arch}"  # Escaped for Nomad
  value     = "amd64"
}
```

### CSI Volumes with Init Tasks

```hcl
volume "data" {
  type   = "csi"
  source = "postgres-data"
  # ...
}

task "init-data" {
  lifecycle {
    hook    = "prestart"
    sidecar = false
  }
  config {
    image   = "busybox:latest"
    command = "/bin/sh"
    args    = ["-c", "chown -R 999:999 /var/lib/postgresql/data"]
  }
  # ...
}
```

### Vault Integration

```hcl
vault {
  policies = ["myapp-policy"]
}

template {
  data = <<EOH
{{ with secret "secret/myapp/db" }}
DB_PASSWORD="{{ .Data.data.password }}"
{{ end }}
EOH
  destination = "secrets/db.env"
  env         = true
}
```

### Fabio Routing

```hcl
service {
  name = "my-app"
  tags = ["urlprefix-myapp.example.com:9999/"]
  # ...
}
```

## Service Types

The generator classifies services into resource categories:

| Type | CPU (MHz) | Memory (MB) | Use Case |
|------|-----------|-------------|----------|
| LIGHT | 200 | 128 | Proxies, sidecars, infrastructure |
| MEDIUM | 500 | 512 | Web apps, APIs, standard services |
| HEAVY | 1000 | 2048 | Databases, stateful services |
| COMPUTE | 4000 | 8192 | ML workloads, compute-intensive |

## Configuration Reference

### Environment Variables

```bash
# LLM Configuration
LLM_PROVIDER=vllm              # vllm, openai, or anthropic
VLLM_BASE_URL=http://localhost:8000
VLLM_MODEL=Qwen/Qwen3-32B
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=4096

# Nomad Configuration
NOMAD_ADDRESS=http://localhost:4646
NOMAD_NAMESPACE=default
NOMAD_REGION=global
NOMAD_DATACENTER=dc1

# Memory Layer (Mem0 + Qdrant)
QDRANT_HOST=localhost
QDRANT_PORT=6333
MEMORY_ENABLED=true

# Observability (LangFuse)
LANGFUSE_ENABLED=true
LANGFUSE_HOST=https://cloud.langfuse.com

# Agent Configuration
MAX_ITERATIONS=3
DEFAULT_CPU=500
DEFAULT_MEMORY=256
```

## Development

### Setup

```bash
# Install dev dependencies
make install-dev

# Run linting
make lint

# Run tests
make test

# Run tests with coverage
make test-cov

# Format code
make format
```

### Project Structure

```
nomad-job-spec/
├── src/
│   ├── main.py           # CLI entry point
│   ├── graph.py          # LangGraph workflow
│   ├── llm/
│   │   └── provider.py   # LLM abstraction
│   ├── nodes/
│   │   ├── analyze.py    # Codebase analysis node
│   │   └── generate.py   # HCL generation node
│   └── tools/
│       ├── codebase.py   # Codebase analysis tools
│       └── hcl.py        # HCL generation/validation
├── config/
│   └── settings.py       # Pydantic settings
├── tests/
│   └── fixtures/
│       └── sample_repos/ # Test fixtures
├── data/
│   └── nomad-job-specification.md  # Cluster documentation
├── Makefile
└── pyproject.toml
```

### Running Tests

```bash
# Run all tests
make test

# Run specific test file
pytest tests/test_tools/test_hcl.py -v

# Run with coverage report
make test-cov
```

## API Usage

You can also use the library programmatically:

```python
from src.tools.hcl import JobConfig, PortConfig, VolumeConfig, generate_hcl
from src.tools.codebase import analyze_codebase

# Analyze a codebase
analysis = analyze_codebase("/path/to/app")
print(f"Detected: {analysis.dependencies.language}")
print(f"Ports: {analysis.dockerfile.exposed_ports}")

# Generate HCL manually
config = JobConfig(
    job_name="my-app",
    image="myapp:latest",
    ports=[PortConfig(name="http", container_port=8080)],
    service_type=ServiceType.MEDIUM,
    fabio_route=FabioRoute(hostname="myapp.example.com"),
)

hcl = generate_hcl(config)
print(hcl)
```

## Troubleshooting

### Common Issues

**LLM Connection Error**
```
Error: Could not connect to vLLM server
```
Ensure vLLM is running: `curl http://localhost:8000/health`

**Nomad Validation Failed**
```
Error: job validation failed
```
Check that the Nomad CLI is installed and can reach your cluster.

**Memory Layer Disabled**
```
Warning: Memory layer not available
```
Ensure Qdrant is running and `MEMORY_ENABLED=true`.

### Debug Mode

```bash
# Enable debug logging
LOG_LEVEL=DEBUG nomad-spec generate -p "Deploy app" --path ./app
```

## License

MIT

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `make test`
5. Submit a pull request
