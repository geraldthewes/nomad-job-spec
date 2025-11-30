# Nomad Job Specification Guide for AI Agents

This guide documents cluster-specific knowledge for creating Nomad job specifications. It assumes familiarity with standard Nomad HCL syntax and focuses on patterns, constraints, and best practices specific to this cluster.

---

## Table of Contents

1. [Cluster Architecture](#1-cluster-architecture)
2. [Job Structure Patterns](#2-job-structure-patterns)
3. [Network Configuration](#3-network-configuration)
4. [Constraint Patterns](#4-constraint-patterns)
5. [Storage Patterns](#5-storage-patterns)
6. [Service Discovery & Consul](#6-service-discovery--consul)
7. [Fabio Load Balancer](#7-fabio-load-balancer)
8. [Secrets Management (Vault)](#8-secrets-management-vault)
9. [Configuration Templates](#9-configuration-templates)
10. [Lifecycle Hooks](#10-lifecycle-hooks)
11. [Update & Restart Policies](#11-update--restart-policies)
12. [Resource Allocation](#12-resource-allocation)
13. [Service-Type Patterns](#13-service-type-patterns)
14. [Observability Integration](#14-observability-integration)
15. [Anti-Patterns & Common Mistakes](#15-anti-patterns--common-mistakes)
16. [Complete Examples](#16-complete-examples)
17. [Decision Trees](#17-decision-trees)
18. [Pre-Deployment Checklist](#18-pre-deployment-checklist)
19. [Quick Reference Tables](#19-quick-reference-tables)

---

## 1. Cluster Architecture

### Node Inventory

| Node | IP Address | Architecture | Nomad Role | Special Capabilities |
|------|------------|--------------|------------|---------------------|
| cluster00 | 10.0.1.50 | arm64 | server | Nomad server, ARM64 Ceph |
| cluster01 | 10.0.1.51 | arm64 | server | Nomad server, ARM64 Ceph |
| gpu001 | 10.0.1.11 | amd64 | both | RTX 3080 GPU |
| gpu002 | 10.0.1.12 | amd64 | both | Consul/Nomad server |
| gpu003 | 10.0.1.13 | amd64 | both | Consul/Nomad server |
| gpu004 | 10.0.1.14 | amd64 | - | vLLM dedicated (outside Nomad) |
| gpu005 | 10.0.1.15 | amd64 | both | Ceph OSD node |
| gpu006 | 10.0.1.16 | amd64 | both | Ceph OSD node |
| gpu007 | 10.0.1.17 | amd64 | both | GTX 1080 Ti GPU, Ceph OSD |

### Ceph Storage Clusters

| Cluster | ID | Nodes | Usage |
|---------|-----|-------|-------|
| x86-64 | `70464857-9ed6-11f0-8df5-d45d64d7d4f0` | gpu005, gpu006, gpu007 | CSI volumes for databases |
| ARM64 | `978a0282-0b1b-11f0-91c0-2ccf67943910` | cluster00-03 | Reserved for ARM64 workloads |

### Key Architecture Rules

- **Databases and stateful services**: Constrain to `amd64` (gpu00x nodes)
- **ARM64 servers (cluster00-01)**: Nomad servers only, not for workloads
- **GPU nodes**: gpu001 (RTX 3080), gpu007 (GTX 1080 Ti)
- **Ceph OSD nodes (gpu005, gpu006)**: Never drain both simultaneously

---

## 2. Job Structure Patterns

### Basic Job Header

All jobs use Terraform variable interpolation for the datacenter:

```hcl
job "my-service" {
  datacenters = ["${datacenter}"]
  type        = "service"

  group "main" {
    count = 1
    # ... tasks
  }
}
```

### Job Types

| Type | Use Case | Example Services |
|------|----------|------------------|
| `service` | Long-running services | Databases, web apps, APIs |
| `system` | One instance per node | Fabio, CSI plugin nodes |
| `batch` | One-time or scheduled | Data migrations, cleanup jobs |

### Namespaces

| Namespace | Purpose |
|-----------|---------|
| `default` | Most services |
| `vault-cluster` | Vault only (CRITICAL - never destroy) |

---

## 3. Network Configuration

### Static Ports

Use for databases, critical infrastructure, and direct-access services:

```hcl
network {
  mode = "host"  # Required for static ports
  port "http" {
    static = 6333
    to     = 6333  # Container port (optional if same)
  }
}
```

### Dynamic Ports

Use for web apps behind Fabio and non-critical services:

```hcl
network {
  port "http" {
    to = 8080  # Nomad assigns random host port
  }
}

# Access via Nomad runtime variable
env {
  PORT = "${NOMAD_PORT_http}"
}
```

### Network Modes

| Mode | When to Use |
|------|-------------|
| `host` | Static ports, privileged ports (<1024), DNS binding |
| (default/bridge) | Dynamic ports, most services |

### Reserved Port Allocations

**Core Infrastructure:**
| Service | Port |
|---------|------|
| DNS (dnsmasq) | 53 |
| Nomad HTTP | 4646 |
| Docker Registry | 5000 |
| Ceph Monitor | 6789 |
| Vault API | 8200 |
| Consul HTTP | 8500 |
| Fabio Admin | 9998 |
| Fabio LB | 9999 |

**Databases:**
| Service | Port |
|---------|------|
| PostgreSQL | 5432 |
| Valkey/Redis | 6379 |
| Qdrant HTTP | 6333 |
| Qdrant gRPC | 6334 |
| Neo4j HTTP | 7474 |
| Neo4j Bolt | 7687 |
| ClickHouse HTTP | 8123 |
| ClickHouse Native | 9000 |

**Observability:**
| Service | Port |
|---------|------|
| Prometheus | 9090 |
| Jaeger OTLP HTTP | 4318 |
| Jaeger UI | 16686 |

---

## 4. Constraint Patterns

### Architecture Constraint

**Required for databases and most data-intensive services:**

```hcl
constraint {
  attribute = "$${attr.cpu.arch}"
  value     = "amd64"
}
```

**Note:** Use `$${}` for Nomad runtime variables when the job file is processed by Terraform's `templatefile()`. The double `$` escapes Terraform interpolation.

### Node Targeting

Pin to specific nodes using regexp:

```hcl
constraint {
  attribute = "$${node.unique.name}"
  operator  = "regexp"
  value     = "^(gpu002|gpu003|gpu005)$"
}
```

### Distinct Hosts

For HA services requiring instances on separate nodes:

```hcl
constraint {
  operator = "distinct_hosts"
  value    = "true"
}
```

### Spread Directive

Distribute instances across nodes:

```hcl
spread {
  attribute = "$${node.unique.name}"
  weight    = 100
}
```

---

## 5. Storage Patterns

### CSI Volume Declaration

For persistent storage backed by Ceph:

```hcl
volume "data" {
  type            = "csi"
  source          = "my-service-data"  # Must match volume ID in Terraform
  attachment_mode = "file-system"
  access_mode     = "single-node-writer"
  per_alloc       = false
  mount_options {
    fs_type = "ext4"
  }
}
```

### Volume Mount

```hcl
task "main" {
  volume_mount {
    volume      = "data"
    destination = "/data"
    read_only   = false
  }
}
```

### Init Task for Permissions

**Required when container runs as non-root user:**

```hcl
task "init" {
  driver = "docker"
  lifecycle {
    hook    = "prestart"
    sidecar = false
  }

  config {
    image   = "busybox:latest"
    command = "/bin/sh"
    args    = ["-c", "chown -R 1000:1000 /data"]
  }

  resources {
    cpu    = 100
    memory = 64
  }

  volume_mount {
    volume      = "data"
    destination = "/data"
    read_only   = false
  }
}
```

### Common UID/GID Requirements

| Service | UID:GID | chown Command |
|---------|---------|---------------|
| Prometheus | 65534:65534 | `chown -R 65534:65534 /prometheus` |
| ClickHouse | 101:101 | `chown -R 101:101 /var/lib/clickhouse` |
| Neo4j | 7474:7474 | `chown -R 7474:7474 /data` |
| Qdrant | 1000:1000 | `chown -R 1000:1000 /data` |

### Existing Volume Naming Conventions

| Pattern | Example |
|---------|---------|
| Single instance | `prometheus-data`, `clickhouse-data` |
| Multi-instance | `qdrant-data-0`, `qdrant-data-1`, `qdrant-data-2` |
| Per-node | `jaeger-data-0`, `dgraph-alpha-0` |

---

## 6. Service Discovery & Consul

### Basic Service Registration

```hcl
service {
  name = "my-service"
  port = "http"
  tags = [
    "api",
    "http",
  ]

  check {
    type     = "http"
    path     = "/health"
    interval = "10s"
    timeout  = "2s"
  }
}
```

### Multiple Services per Task

For services with multiple interfaces:

```hcl
# HTTP API
service {
  name = "qdrant-api"
  port = "http"
  tags = ["api", "http"]
  check {
    type     = "http"
    path     = "/healthz"
    interval = "10s"
    timeout  = "2s"
  }
}

# gRPC endpoint
service {
  name = "qdrant-grpc"
  port = "grpc"
  tags = ["grpc"]
  check {
    type     = "http"
    port     = "http"  # Reuse HTTP health check
    path     = "/healthz"
    interval = "10s"
    timeout  = "2s"
  }
}

# Metrics endpoint (for Prometheus)
service {
  name = "qdrant-metrics"
  port = "http"
  tags = ["metrics", "prometheus"]
  check {
    type     = "http"
    path     = "/healthz"
    interval = "10s"
    timeout  = "2s"
  }
}
```

### Consul DNS Resolution

Services are accessible via DNS:
- Pattern: `service-name.service.consul`
- Example: `qdrant-api.service.consul`, `clickhouse.service.consul`

### Dynamic Service Discovery in Templates

```hcl
template {
  data = <<EOH
{{ with service "clickhouse" }}{{ with index . 0 }}
CLICKHOUSE_HOST="{{ .Address }}"
CLICKHOUSE_PORT="{{ .Port }}"
{{ end }}{{ end }}
EOH
  destination = "secrets/service.env"
  env         = true
}
```

---

## 7. Fabio Load Balancer

Fabio runs on all nodes (port 9999) and routes traffic based on service tags.

### Host-Based Routing (Preferred)

```hcl
service {
  tags = ["urlprefix-myapp.cluster:9999/"]
}
```

**Requirements:**
1. Add DNS entry to `terraform/dns-config/hosts.json`:
   ```json
   {
     "myapp.cluster": "10.0.1.12"
   }
   ```
2. Run `./update-dns-config.sh` to apply

**Access:** `http://myapp.cluster:9999/`

### Path-Based Routing

```hcl
service {
  tags = ["urlprefix-/myapp strip=/myapp"]
}
```

**Access:** `http://any-node:9999/myapp`

**Note:** Only use if your app handles the stripped prefix correctly.

### Existing DNS Entries

| Hostname | IP | Service |
|----------|-----|---------|
| langfuse.cluster | 10.0.1.12 | Langfuse |
| openwebui.cluster | 10.0.1.17 | Open-WebUI |
| pyexec.cluster | 10.0.1.12 | Python Executor |
| dgraph.cluster | 10.0.1.12 | DGraph |
| registry.cluster | 10.0.1.13 | Docker Registry |

---

## 8. Secrets Management (Vault)

### Vault Block

```hcl
task "main" {
  vault {
    policies = ["my-service-policy"]
  }
  # ... config and templates
}
```

### Template with Vault Secrets

**Standard delimiters (when no Consul templates):**

```hcl
template {
  data = <<EOH
{{ with secret "secret/myapp/config" }}
DATABASE_URL="{{ .Data.data.connection_string }}"
API_KEY="{{ .Data.data.api_key }}"
{{ end }}
EOH
  destination = "secrets/app.env"
  env         = true
  change_mode = "restart"
}
```

**Custom delimiters (when mixing Vault and Consul):**

```hcl
template {
  data = <<EOH
[[ with secret "secret/myapp/postgres" ]]
DATABASE_URL="postgresql://[[ .Data.data.user ]]:[[ .Data.data.password ]]@postgres.cluster:5432/mydb"
[[ end ]]
EOH
  destination   = "secrets/postgres.env"
  env           = true
  change_mode   = "restart"
  left_delimiter  = "[["
  right_delimiter = "]]"
}
```

### Common Vault Secret Paths

| Path Pattern | Content |
|--------------|---------|
| `secret/[service]/postgres` | PostgreSQL credentials |
| `secret/[service]/clickhouse` | ClickHouse credentials |
| `secret/[service]/redis` | Redis password |
| `secret/[service]/s3` | S3/RGW credentials |
| `secret/[service]/app` | Application secrets (API keys, etc.) |

### Combining Vault and Consul Service Discovery

```hcl
template {
  data = <<EOH
{{ with secret "secret/langfuse/clickhouse" }}
{{- $user := .Data.data.user -}}
{{- $password := .Data.data.password -}}
{{- $database := .Data.data.database -}}
CLICKHOUSE_USER="{{ $user }}"
CLICKHOUSE_PASSWORD="{{ $password }}"
{{with service "clickhouse"}}{{with index . 0}}
CLICKHOUSE_HOST="{{ .Address }}"
CLICKHOUSE_PORT="{{ .Port }}"
CLICKHOUSE_URL="http://{{ .Address }}:{{ .Port }}"
{{end}}{{end}}
{{ end }}
EOH
  destination = "secrets/clickhouse.env"
  env         = true
  change_mode = "restart"
}
```

---

## 9. Configuration Templates

### Environment Files

```hcl
template {
  data = <<EOH
NODE_ENV=production
LOG_LEVEL=info
PORT={{ env "NOMAD_PORT_http" }}
EOH
  destination = "local/app.env"
  env         = true
  change_mode = "restart"
}
```

### Configuration Files

```hcl
template {
  data = <<EOH
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'nomad'
    consul_sd_configs:
      - server: '{{ env "NOMAD_IP_http" }}:8500'
EOH
  destination = "local/prometheus.yml"
  change_mode = "signal"
  change_signal = "SIGHUP"
}
```

### Executable Scripts

```hcl
template {
  data = <<EOH
#!/bin/sh
set -e
echo "Starting service..."
exec /app/server --config /local/config.yml
EOH
  destination = "local/start.sh"
  perms       = "755"
  change_mode = "noop"  # Don't restart on script changes
}

config {
  command = "/bin/sh"
  args    = ["/local/start.sh"]
}
```

---

## 10. Lifecycle Hooks

### Prestart Tasks

Run before the main task starts:

```hcl
task "init" {
  driver = "docker"
  lifecycle {
    hook    = "prestart"
    sidecar = false  # Runs once, exits
  }

  config {
    image   = "busybox:latest"
    command = "/bin/sh"
    args    = ["-c", "mkdir -p /data/storage && chown -R 1000:1000 /data"]
  }

  resources {
    cpu    = 100
    memory = 64
  }

  volume_mount {
    volume      = "data"
    destination = "/data"
  }
}
```

### Poststop Tasks

Run after the main task stops (cleanup):

```hcl
task "cleanup" {
  driver = "docker"
  lifecycle {
    hook    = "poststop"
    sidecar = false
  }

  config {
    image   = "curlimages/curl:latest"
    command = "/bin/sh"
    args    = ["-c", <<EOF
echo "Removing peer from cluster..."
curl -X DELETE "http://service.consul:8080/cluster/peer/$${NOMAD_ALLOC_ID}"
EOF
    ]
  }

  resources {
    cpu    = 100
    memory = 128
  }
}
```

### Sidecar Containers

Long-running helper containers:

```hcl
task "log-shipper" {
  driver = "docker"
  lifecycle {
    hook    = "prestart"
    sidecar = true  # Keeps running alongside main task
  }
  # ... config
}
```

---

## 11. Update & Restart Policies

### Update Strategy

```hcl
update {
  max_parallel      = 1           # Update one instance at a time
  min_healthy_time  = "30s"       # Time instance must be healthy
  healthy_deadline  = "5m"        # Max time to become healthy
  progress_deadline = "10m"       # Max time for entire update
  auto_revert       = true        # Rollback on failure
  stagger           = "1m"        # Delay between instances
}
```

### Health Check Options

| Option | Use Case |
|--------|----------|
| `health_check = "checks"` | Use Consul health checks (default) |
| `health_check = "task_states"` | Use Nomad task state only |

### Restart Policy

```hcl
restart {
  attempts = 3        # Retry up to 3 times
  interval = "5m"     # Within 5-minute window
  delay    = "15s"    # Wait before retry
  mode     = "delay"  # Fixed delay between retries
}
```

### Recommended Settings by Service Type

| Service Type | max_parallel | stagger | min_healthy_time |
|-------------|--------------|---------|------------------|
| Databases | 1 | 1m-2m | 30s-60s |
| Web apps | 1 | 30s | 10s-30s |
| Infrastructure | 1 | 30s | 10s |
| System jobs | 1 | 10s | 10s |

---

## 12. Resource Allocation

### Light Services (Infrastructure)

```hcl
resources {
  cpu    = 100-200
  memory = 64-128
}
```

Examples: dnsmasq, init tasks, Fabio, utility containers

### Medium Services (Applications)

```hcl
resources {
  cpu    = 500-1000
  memory = 512-1024
}
```

Examples: web apps, APIs, langfuse-web, searxng

### Heavy Services (Databases)

```hcl
resources {
  cpu    = 1000-2000
  memory = 2048-4096
}
```

Examples: ClickHouse, Neo4j, Qdrant instances

### Compute-Intensive Services

```hcl
resources {
  cpu    = 4000-8000
  memory = 8192-16384
}
```

Examples: python-executor, ML inference services

---

## 13. Service-Type Patterns

### Stateless Web Application

```hcl
job "web-app" {
  datacenters = ["${datacenter}"]
  type        = "service"

  constraint {
    attribute = "$${attr.cpu.arch}"
    value     = "amd64"
  }

  group "web" {
    count = 1

    network {
      port "http" {
        to = 8080
      }
    }

    service {
      name = "web-app"
      port = "http"
      tags = ["urlprefix-webapp.cluster:9999/"]

      check {
        type     = "http"
        path     = "/health"
        interval = "10s"
        timeout  = "2s"
      }
    }

    task "server" {
      driver = "docker"

      config {
        image = "my-app:latest"
        ports = ["http"]
      }

      env {
        PORT = "$${NOMAD_PORT_http}"
      }

      resources {
        cpu    = 500
        memory = 512
      }
    }
  }

  update {
    max_parallel     = 1
    min_healthy_time = "10s"
    healthy_deadline = "3m"
    auto_revert      = true
  }
}
```

### Stateful Database with CSI

```hcl
job "database" {
  datacenters = ["${datacenter}"]
  type        = "service"

  group "db" {
    count = 1

    constraint {
      attribute = "$${attr.cpu.arch}"
      value     = "amd64"
    }

    volume "data" {
      type            = "csi"
      source          = "database-data"
      attachment_mode = "file-system"
      access_mode     = "single-node-writer"
      mount_options {
        fs_type = "ext4"
      }
    }

    network {
      port "db" {
        static = 5432
      }
    }

    task "init" {
      driver = "docker"
      lifecycle {
        hook    = "prestart"
        sidecar = false
      }

      config {
        image   = "busybox:latest"
        command = "/bin/sh"
        args    = ["-c", "chown -R 999:999 /data"]
      }

      resources {
        cpu    = 100
        memory = 64
      }

      volume_mount {
        volume      = "data"
        destination = "/data"
      }
    }

    task "database" {
      driver = "docker"

      config {
        image = "postgres:15"
        ports = ["db"]
      }

      volume_mount {
        volume      = "data"
        destination = "/var/lib/postgresql/data"
      }

      resources {
        cpu    = 1000
        memory = 2048
      }

      service {
        name = "postgres"
        port = "db"

        check {
          type     = "tcp"
          interval = "10s"
          timeout  = "2s"
        }
      }
    }
  }
}
```

### Clustered Database (Multi-Group)

For databases requiring separate groups per instance:

```hcl
job "clustered-db" {
  datacenters = ["${datacenter}"]
  type        = "service"

  # Instance 0 - Bootstrap
  group "node-0" {
    count = 1

    constraint {
      attribute = "$${attr.cpu.arch}"
      value     = "amd64"
    }

    spread {
      attribute = "$${node.unique.name}"
      weight    = 100
    }

    volume "data" {
      type   = "csi"
      source = "clustered-db-data-0"
      # ... mount options
    }

    network {
      port "http" { static = 7000 }
      port "cluster" { static = 7001 }
    }

    task "init" {
      # ... permission setup
    }

    task "db" {
      driver = "docker"

      template {
        data = <<EOH
#!/bin/sh
# Bootstrap node starts without peers
exec /db/server --node-id=0 --bootstrap
EOH
        destination = "local/start.sh"
        perms       = "755"
      }

      config {
        image   = "clustered-db:latest"
        command = "/bin/sh"
        args    = ["/local/start.sh"]
      }
      # ... resources, services
    }
  }

  # Instance 1 - Joins cluster
  group "node-1" {
    count = 1

    constraint {
      attribute = "$${attr.cpu.arch}"
      value     = "amd64"
    }

    spread {
      attribute = "$${node.unique.name}"
      weight    = 100
    }

    volume "data" {
      type   = "csi"
      source = "clustered-db-data-1"
      # ...
    }

    network {
      port "http" { static = 7000 }
      port "cluster" { static = 7001 }
    }

    task "db" {
      template {
        data = <<EOH
#!/bin/sh
# Wait for bootstrap node
BOOTSTRAP=""
for i in 1 2 3 4 5 6 7 8 9 10; do
  PEER=$(getent hosts clustered-db-cluster.service.consul | head -1 | awk '{print $1}')
  if [ -n "$PEER" ]; then
    BOOTSTRAP="$PEER:7001"
    break
  fi
  sleep 5
done
exec /db/server --node-id=1 --join="$BOOTSTRAP"
EOH
        destination = "local/start.sh"
        perms       = "755"
      }
      # ... config, resources
    }
  }

  update {
    max_parallel      = 1
    stagger           = "1m"
    min_healthy_time  = "30s"
    healthy_deadline  = "5m"
    auto_revert       = true
  }
}
```

### System Job

Runs one instance per node:

```hcl
job "system-service" {
  datacenters = ["${datacenter}"]
  type = "system"

  group "main" {
    network {
      mode = "host"
      port "http" {
        static = 9999
      }
    }

    task "service" {
      driver = "docker"

      config {
        image        = "my-system-service:latest"
        network_mode = "host"
        ports        = ["http"]
      }

      resources {
        cpu    = 200
        memory = 128
      }

      service {
        name         = "system-service"
        port         = "http"
        address_mode = "auto"
        address      = "$${attr.unique.network.ip-address}"

        check {
          type     = "http"
          path     = "/health"
          interval = "10s"
          timeout  = "3s"
        }
      }
    }
  }

  update {
    max_parallel     = 1
    min_healthy_time = "10s"
    healthy_deadline = "2m"
    auto_revert      = true
  }
}
```

---

## 14. Observability Integration

### Prometheus Metrics

Add `metrics` tag for Prometheus scraping:

```hcl
service {
  name = "my-service-metrics"
  port = "http"
  tags = [
    "metrics",
    "prometheus",
  ]

  check {
    type     = "http"
    path     = "/metrics"
    interval = "30s"
    timeout  = "5s"
  }
}
```

### Jaeger Tracing

Configure OTLP endpoint via environment:

```hcl
env {
  OTEL_EXPORTER_OTLP_ENDPOINT = "http://jaeger-collector.service.consul:4318"
  OTEL_SERVICE_NAME           = "my-service"
  OTEL_TRACES_EXPORTER        = "otlp"
}
```

Or via Consul service discovery:

```hcl
template {
  data = <<EOH
{{with service "jaeger-collector"}}{{with index . 0}}
OTEL_EXPORTER_OTLP_ENDPOINT="http://{{ .Address }}:4318"
{{end}}{{end}}
OTEL_SERVICE_NAME="my-service"
EOH
  destination = "secrets/tracing.env"
  env         = true
}
```

---

## 15. Anti-Patterns & Common Mistakes

### Missing Architecture Constraint

**Wrong:**
```hcl
job "postgres" {
  # No constraint - may schedule on ARM64 node
  group "db" { ... }
}
```

**Correct:**
```hcl
job "postgres" {
  constraint {
    attribute = "$${attr.cpu.arch}"
    value     = "amd64"
  }
  group "db" { ... }
}
```

### Forgetting Init Task for Permissions

**Wrong:**
```hcl
task "prometheus" {
  volume_mount {
    volume      = "data"
    destination = "/prometheus"
  }
  # Container runs as nobody (65534) but volume is root-owned
}
```

**Correct:**
```hcl
task "init" {
  lifecycle { hook = "prestart"; sidecar = false }
  config {
    image   = "busybox:latest"
    command = "/bin/sh"
    args    = ["-c", "chown -R 65534:65534 /prometheus"]
  }
  volume_mount { ... }
}
```

### Wrong Variable Escaping

**Wrong (Terraform consumes the variable):**
```hcl
env {
  PORT = "${NOMAD_PORT_http}"  # Terraform error: NOMAD_PORT_http undefined
}
```

**Correct (escaped for Nomad runtime):**
```hcl
env {
  PORT = "$${NOMAD_PORT_http}"  # Nomad resolves at runtime
}
```

### Static Port Without Network Mode

**Wrong:**
```hcl
network {
  port "dns" {
    static = 53  # Privileged port requires host mode
  }
}
```

**Correct:**
```hcl
network {
  mode = "host"
  port "dns" {
    static = 53
  }
}
```

### Missing DNS Entry for Host-Based Routing

**Wrong:**
```hcl
service {
  tags = ["urlprefix-myapp.cluster:9999/"]  # No DNS entry exists
}
# Access to http://myapp.cluster:9999/ fails with DNS error
```

**Correct:**
1. Add to `terraform/dns-config/hosts.json`:
   ```json
   { "myapp.cluster": "10.0.1.12" }
   ```
2. Run `./update-dns-config.sh`

### Missing Health Check

**Wrong:**
```hcl
service {
  name = "my-service"
  port = "http"
  # No health check - Fabio may route to unhealthy instance
}
```

**Correct:**
```hcl
service {
  name = "my-service"
  port = "http"
  check {
    type     = "http"
    path     = "/health"
    interval = "10s"
    timeout  = "2s"
  }
}
```

### Using Dynamic Port When Static Required

**Wrong (for database):**
```hcl
network {
  port "db" {
    to = 5432  # Port changes on reschedule, breaks clients
  }
}
```

**Correct:**
```hcl
network {
  port "db" {
    static = 5432
  }
}
```

---

## 16. Complete Examples

### Example 1: Simple Stateless Web App

Based on python-executor pattern:

```hcl
job "api-service" {
  datacenters = ["${datacenter}"]
  type = "service"

  constraint {
    attribute = "$${attr.cpu.arch}"
    value     = "amd64"
  }

  group "api" {
    count = 1

    network {
      mode = "host"
      port "http" {
        to = 8080
      }
    }

    service {
      name = "api-service"
      port = "http"
      tags = [
        "api",
        "urlprefix-api.cluster:9999/"
      ]

      check {
        type     = "http"
        path     = "/health"
        interval = "10s"
        timeout  = "3s"
      }
    }

    task "server" {
      driver = "docker"

      config {
        image      = "registry.cluster:5000/api-service:latest"
        force_pull = true
        ports      = ["http"]
      }

      env {
        PORT      = "$${NOMAD_PORT_http}"
        HOST      = "0.0.0.0"
        LOG_LEVEL = "info"
      }

      resources {
        cpu    = 1000
        memory = 1024
      }

      restart {
        attempts = 3
        interval = "5m"
        delay    = "15s"
        mode     = "delay"
      }
    }
  }

  update {
    max_parallel      = 1
    health_check      = "checks"
    min_healthy_time  = "10s"
    healthy_deadline  = "3m"
    progress_deadline = "5m"
    auto_revert       = true
  }
}
```

### Example 2: Web App with Vault Secrets

Based on langfuse-web pattern:

```hcl
job "secure-app" {
  datacenters = ["${datacenter}"]
  type        = "service"

  group "web" {
    count = 1

    constraint {
      attribute = "$${attr.cpu.arch}"
      value     = "amd64"
    }

    network {
      port "http" {
        static = 3000
      }
    }

    task "app" {
      driver = "docker"

      vault {
        policies = ["secure-app-policy"]
      }

      config {
        image       = "secure-app:latest"
        ports       = ["http"]
        dns_servers = ["10.0.1.12", "10.0.1.13"]
      }

      # PostgreSQL from Vault
      template {
        data = <<EOH
[[ with secret "secret/secure-app/postgres" ]]
DATABASE_URL="postgresql://[[ .Data.data.user ]]:[[ .Data.data.password ]]@postgres.cluster:5432/[[ .Data.data.database ]]"
[[ end ]]
EOH
        destination     = "secrets/postgres.env"
        env             = true
        change_mode     = "restart"
        left_delimiter  = "[["
        right_delimiter = "]]"
      }

      # Redis via Consul service discovery
      template {
        data = <<EOH
{{with service "redis"}}{{with index . 0}}
REDIS_URL="redis://{{ .Address }}:{{ .Port }}"
{{end}}{{end}}
EOH
        destination = "secrets/redis.env"
        env         = true
        change_mode = "restart"
      }

      # App secrets from Vault
      template {
        data = <<EOH
[[ with secret "secret/secure-app/app" ]]
SECRET_KEY="[[ .Data.data.secret_key ]]"
API_KEY="[[ .Data.data.api_key ]]"
[[ end ]]
EOH
        destination     = "secrets/app.env"
        env             = true
        change_mode     = "restart"
        left_delimiter  = "[["
        right_delimiter = "]]"
      }

      env {
        NODE_ENV = "production"
        PORT     = "3000"
      }

      resources {
        cpu    = 1000
        memory = 1024
      }

      service {
        name = "secure-app"
        port = "http"
        tags = [
          "web",
          "urlprefix-secureapp.cluster:9999/",
        ]

        check {
          type     = "http"
          path     = "/api/health"
          interval = "30s"
          timeout  = "5s"
        }
      }

      restart {
        attempts = 3
        interval = "5m"
        delay    = "30s"
        mode     = "delay"
      }
    }
  }

  update {
    max_parallel      = 1
    health_check      = "task_states"
    min_healthy_time  = "30s"
    healthy_deadline  = "5m"
    progress_deadline = "10m"
    auto_revert       = true
  }
}
```

### Example 3: Stateful Database with CSI Volume

Based on prometheus pattern:

```hcl
job "timeseries-db" {
  datacenters = ["${datacenter}"]
  type        = "service"

  group "db" {
    count = 1

    constraint {
      attribute = "$${attr.cpu.arch}"
      value     = "amd64"
    }

    volume "data" {
      type            = "csi"
      source          = "timeseries-db-data"
      attachment_mode = "file-system"
      access_mode     = "single-node-writer"
      per_alloc       = false
      mount_options {
        fs_type = "ext4"
      }
    }

    network {
      port "http" {
        static = 9090
      }
    }

    task "init" {
      driver = "docker"
      lifecycle {
        hook    = "prestart"
        sidecar = false
      }

      config {
        image   = "busybox:latest"
        command = "/bin/sh"
        args    = ["-c", "chown -R 65534:65534 /data"]
      }

      resources {
        cpu    = 100
        memory = 64
      }

      volume_mount {
        volume      = "data"
        destination = "/data"
        read_only   = false
      }
    }

    task "db" {
      driver = "docker"

      config {
        image = "prom/prometheus:latest"
        args  = [
          "--config.file=/etc/prometheus/prometheus.yml",
          "--storage.tsdb.path=/data",
          "--storage.tsdb.retention.size=16GB"
        ]
        ports = ["http"]
      }

      resources {
        cpu    = 1000
        memory = 2048
      }

      volume_mount {
        volume      = "data"
        destination = "/data"
        read_only   = false
      }

      template {
        data = <<EOH
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'self'
    static_configs:
      - targets: ['localhost:9090']
EOH
        destination = "local/prometheus.yml"
      }

      service {
        name = "timeseries-db"
        port = "http"
        tags = ["database", "metrics"]

        check {
          type     = "http"
          path     = "/-/healthy"
          interval = "10s"
          timeout  = "2s"
        }
      }
    }
  }

  update {
    max_parallel     = 1
    min_healthy_time = "30s"
    healthy_deadline = "5m"
    auto_revert      = true
  }
}
```

---

## 17. Decision Trees

### Port Allocation

```
Is this a critical infrastructure service (Vault, Consul, DNS)?
  └─ YES → Use static port (check port allocation table)
  └─ NO → Is this a database that clients connect to directly?
          └─ YES → Use static port
          └─ NO → Is this accessed via Fabio load balancer?
                  └─ YES → Use dynamic port (just `to = PORT`)
                  └─ NO → Consider dynamic unless specific reason for static
```

### Storage Selection

```
Does the service need persistent data?
  └─ NO → No volume needed
  └─ YES → Is it configuration only (small, rarely changes)?
          └─ YES → Use template to generate config file
          └─ NO → Use CSI volume
                  └─ Does the container run as non-root?
                      └─ YES → Add init task with chown
                      └─ NO → Mount directly
```

### Routing Method

```
Does the service need external HTTP access?
  └─ NO → Use Consul DNS only (service.service.consul)
  └─ YES → Does the app work correctly with URL path prefixes?
          └─ YES → Use path-based: urlprefix-/myapp strip=/myapp
          └─ NO → Use host-based: urlprefix-myapp.cluster:9999/
                  └─ Add DNS entry to terraform/dns-config/hosts.json
```

### Architecture Constraint

```
Is this a database or data-intensive service?
  └─ YES → Constrain to amd64
  └─ NO → Is the Docker image multi-arch (supports amd64 + arm64)?
          └─ YES → No constraint needed (Nomad scheduler picks)
          └─ NO → Constrain to the architecture the image supports
```

### Vault Integration

```
Does the service need secrets (passwords, API keys, certificates)?
  └─ NO → Use env block for static configuration
  └─ YES → Are secrets also using Consul service discovery?
          └─ NO → Use standard {{ }} delimiters
          └─ YES → Use [[ ]] delimiters to avoid conflicts
```

---

## 18. Pre-Deployment Checklist

### Job Structure

- [ ] Job has `datacenters = ["${datacenter}"]` for Terraform templating
- [ ] Job type is appropriate: `service`, `system`, or `batch`
- [ ] Group count matches HA requirements (1 for single, 3+ for HA)

### Constraints

- [ ] Architecture constraint added for database/data-intensive images
- [ ] Node targeting uses regexp for flexibility
- [ ] `distinct_hosts` constraint added for HA services
- [ ] Spread directive used if instances should distribute across nodes

### Networking

- [ ] Port allocation checked against reserved ports table
- [ ] Static ports use `mode = "host"` if port < 1024
- [ ] Dynamic ports use just `to` parameter (no `static`)
- [ ] Runtime variables escaped: `$${NOMAD_PORT_xxx}` not `${NOMAD_PORT_xxx}`

### Storage

- [ ] CSI volume `source` matches volume ID in Terraform
- [ ] Init task added for containers running as non-root
- [ ] Volume mount `destination` matches container's expected path
- [ ] `access_mode = "single-node-writer"` for single-instance services

### Service Discovery

- [ ] Service name is unique and descriptive
- [ ] Health check path exists and returns 200 when healthy
- [ ] Health check interval/timeout are appropriate (10s/2s typical)
- [ ] Tags include routing configuration if external access needed
- [ ] Tags include `metrics` if Prometheus should scrape

### Fabio Routing (if external access needed)

- [ ] Host-based: DNS entry added to `terraform/dns-config/hosts.json`
- [ ] Host-based: Run `./update-dns-config.sh` after adding entry
- [ ] Path-based: App handles stripped prefix correctly
- [ ] Port 9999 used in urlprefix tag

### Secrets (if Vault integration needed)

- [ ] `vault { policies = ["policy-name"] }` block present in task
- [ ] Vault policy exists with required secret paths
- [ ] Template uses `[[` `]]` delimiters when mixed with Consul
- [ ] `change_mode = "restart"` set on secret templates
- [ ] Secret paths match existing Vault structure

### Updates & Restarts

- [ ] Update block includes `auto_revert = true`
- [ ] `min_healthy_time` gives service time to fully initialize
- [ ] `stagger` set for multi-instance rolling updates
- [ ] Restart policy has reasonable attempts (typically 3)

### Resources

- [ ] CPU/memory match service type guidelines
- [ ] Not over-provisioned (wastes cluster resources)
- [ ] Not under-provisioned (causes OOM kills)

---

## 19. Quick Reference Tables

### Node Architecture Quick Reference

| Pattern | Nodes |
|---------|-------|
| All amd64 clients | gpu001, gpu002, gpu003, gpu005, gpu006, gpu007 |
| ARM64 servers | cluster00, cluster01 |
| GPU nodes | gpu001 (RTX 3080), gpu007 (GTX 1080 Ti) |
| Ceph OSD nodes | gpu005, gpu006, gpu007 |

### Common Docker Images

| Purpose | Image |
|---------|-------|
| Init/setup tasks | `busybox:latest` |
| HTTP operations | `curlimages/curl:latest` |
| Custom services | `registry.cluster:5000/[service]:latest` |

### Template Delimiter Reference

| Use Case | Left | Right |
|----------|------|-------|
| Consul templates only | `{{` | `}}` |
| Vault secrets only | `{{` | `}}` |
| Mixed Vault + Consul | `[[` | `]]` (for Vault) |

### Environment Variable Sources

| Source | Syntax | Available |
|--------|--------|-----------|
| Static value | `VAR = "value"` | Always |
| Nomad runtime | `VAR = "$${NOMAD_PORT_http}"` | At task start |
| Template (Consul) | `{{ env "NOMAD_IP_http" }}` | Template render |
| Template (Vault) | `{{ .Data.data.key }}` | Template render |

### CSI Volume Checklist

| Item | Requirement |
|------|-------------|
| plugin_id | `ceph-csi` |
| attachment_mode | `file-system` |
| access_mode | `single-node-writer` |
| fs_type | `ext4` |
| Capacity | Match Terraform volume definition |

---

## Summary

This guide covers cluster-specific patterns for creating Nomad job specifications. Key takeaways:

1. **Always constrain databases to amd64** - ARM64 nodes are servers only
2. **Use CSI volumes for persistent data** - With init tasks for permission setup
3. **Prefer host-based Fabio routing** - Add DNS entries for new services
4. **Use Vault for secrets** - With appropriate delimiter escaping
5. **Follow resource allocation guidelines** - Light/medium/heavy/compute-intensive
6. **Include health checks** - Required for proper load balancing
7. **Use the validation checklist** - Before deploying new services

Reference the existing job files in `terraform/jobs/` for proven patterns.
