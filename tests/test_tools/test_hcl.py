"""Tests for HCL generation and validation tools."""

import pytest

from src.tools.hcl import (
    JobConfig,
    PortConfig,
    VolumeConfig,
    VaultConfig,
    FabioRoute,
    ServiceType,
    NetworkMode,
    generate_hcl,
    validate_hcl,
    extract_job_name,
    sanitize_job_name,
    merge_hcl_configs,
    _basic_hcl_validation,
)


class TestJobConfig:
    """Tests for JobConfig dataclass."""

    def test_default_values(self):
        """Test JobConfig with minimal required fields."""
        config = JobConfig(job_name="test-job", image="nginx:latest")

        assert config.job_name == "test-job"
        assert config.job_type == "service"
        assert config.datacenters == ["dc1"]
        assert config.count == 1
        # Default service type is MEDIUM: 500 CPU, 512 memory
        assert config.cpu == 500
        assert config.memory == 512
        assert config.service_type == ServiceType.MEDIUM

    def test_auto_generated_names(self):
        """Test that group_name and task_name are auto-generated from job_name."""
        config = JobConfig(job_name="my-app", image="nginx")

        # New cluster pattern: group/task names match job name
        assert config.group_name == "my-app"
        assert config.task_name == "my-app"
        assert config.service_name == "my-app"

    def test_custom_values(self):
        """Test JobConfig with custom values."""
        config = JobConfig(
            job_name="custom-job",
            image="custom:latest",
            datacenters=["dc1", "dc2"],
            ports=[
                PortConfig(name="http", container_port=8080),
                PortConfig(name="grpc", container_port=9090),
            ],
            cpu=1000,
            memory=512,
            count=3,
        )

        assert config.datacenters == ["dc1", "dc2"]
        assert len(config.ports) == 2
        assert config.ports[0].name == "http"
        assert config.ports[0].container_port == 8080
        assert config.ports[1].name == "grpc"
        assert config.ports[1].container_port == 9090
        assert config.cpu == 1000
        assert config.memory == 512
        assert config.count == 3

    def test_legacy_port_dict_conversion(self):
        """Test that legacy dict port format is converted to PortConfig list."""
        config = JobConfig(
            job_name="test",
            image="nginx",
            ports={"http": 8080, "grpc": 9090},  # Legacy dict format
        )

        assert len(config.ports) == 2
        assert all(isinstance(p, PortConfig) for p in config.ports)
        port_names = {p.name for p in config.ports}
        assert "http" in port_names
        assert "grpc" in port_names

    def test_service_type_resources(self):
        """Test resource defaults based on service type."""
        light = JobConfig(job_name="light", image="nginx", service_type=ServiceType.LIGHT)
        assert light.cpu == 200
        assert light.memory == 128

        heavy = JobConfig(job_name="heavy", image="postgres", service_type=ServiceType.HEAVY)
        assert heavy.cpu == 1000
        assert heavy.memory == 2048

        compute = JobConfig(job_name="compute", image="ml", service_type=ServiceType.COMPUTE)
        assert compute.cpu == 4000
        assert compute.memory == 8192


class TestGenerateHCL:
    """Tests for HCL generation."""

    def test_generate_basic_hcl(self):
        """Test generating a basic job spec with cluster patterns."""
        config = JobConfig(
            job_name="basic-job",
            image="nginx:latest",
        )

        hcl = generate_hcl(config)

        assert 'job "basic-job"' in hcl
        assert 'driver = "docker"' in hcl
        assert 'image      = "nginx:latest"' in hcl  # Note: aligned indentation
        assert 'type' in hcl and '"service"' in hcl
        # Cluster patterns: Terraform datacenter and amd64 constraint
        assert '["${datacenter}"]' in hcl
        assert '$${attr.cpu.arch}' in hcl
        assert 'value     = "amd64"' in hcl

    def test_generate_with_ports(self):
        """Test generating HCL with port mappings."""
        config = JobConfig(
            job_name="port-job",
            image="app:latest",
            ports=[
                PortConfig(name="http", container_port=8080),
                PortConfig(name="metrics", container_port=9090),
            ],
        )

        hcl = generate_hcl(config)

        assert 'port "http"' in hcl
        assert "to = 8080" in hcl
        assert 'port "metrics"' in hcl
        assert "to = 9090" in hcl

    def test_generate_with_env_vars(self):
        """Test generating HCL with environment variables."""
        config = JobConfig(
            job_name="env-job",
            image="app:latest",
            env_vars={"DATABASE_URL": "postgres://localhost/db", "DEBUG": "true"},
        )

        hcl = generate_hcl(config)

        assert 'DATABASE_URL = "postgres://localhost/db"' in hcl
        assert 'DEBUG = "true"' in hcl

    def test_generate_with_resources(self):
        """Test generating HCL with resource limits."""
        config = JobConfig(
            job_name="resource-job",
            image="app:latest",
            cpu=2000,
            memory=1024,
        )

        hcl = generate_hcl(config)

        assert "cpu    = 2000" in hcl
        assert "memory = 1024" in hcl

    def test_generate_with_health_check(self):
        """Test generating HCL with HTTP health check."""
        config = JobConfig(
            job_name="health-job",
            image="app:latest",
            health_check_type="http",
            health_check_path="/api/health",
            health_check_interval="15s",
            health_check_timeout="5s",
        )

        hcl = generate_hcl(config)

        assert 'type     = "http"' in hcl
        assert 'path     = "/api/health"' in hcl
        assert 'interval = "15s"' in hcl
        assert 'timeout  = "5s"' in hcl

    def test_generate_with_service_tags(self):
        """Test generating HCL with service tags."""
        config = JobConfig(
            job_name="tagged-job",
            image="app:latest",
            service_tags=["web", "api", "v2"],
        )

        hcl = generate_hcl(config)

        assert '["web", "api", "v2"]' in hcl

    def test_generate_with_csi_volume(self):
        """Test generating HCL with CSI volume and init task."""
        config = JobConfig(
            job_name="db-job",
            image="postgres:15",
            service_type=ServiceType.HEAVY,
            volumes=[
                VolumeConfig(
                    name="data",
                    source="postgres-data",
                    mount_path="/var/lib/postgresql/data",
                    owner_uid=999,  # postgres user
                    owner_gid=999,
                )
            ],
        )

        hcl = generate_hcl(config)

        # Volume block
        assert 'volume "data"' in hcl
        assert 'type            = "csi"' in hcl
        assert 'source          = "postgres-data"' in hcl
        # Init task for chown
        assert 'task "init-data"' in hcl
        assert 'lifecycle {' in hcl
        assert 'hook    = "prestart"' in hcl
        assert 'chown -R 999:999' in hcl
        # Volume mount in task
        assert 'volume_mount {' in hcl

    def test_generate_with_fabio_route(self):
        """Test generating HCL with Fabio routing."""
        config = JobConfig(
            job_name="web-app",
            image="myapp:latest",
            fabio_route=FabioRoute(hostname="myapp.cluster"),
        )

        hcl = generate_hcl(config)

        assert "urlprefix-myapp.cluster:9999/" in hcl

    def test_generate_with_vault(self):
        """Test generating HCL with Vault integration."""
        config = JobConfig(
            job_name="secure-app",
            image="app:latest",
            vault=VaultConfig(
                policies=["myapp-policy"],
                secrets={"DB_PASSWORD": "secret/myapp/db.password"},
            ),
        )

        hcl = generate_hcl(config)

        assert 'vault {' in hcl
        assert '["myapp-policy"]' in hcl
        assert 'template {' in hcl
        assert 'secret/myapp/db' in hcl

    def test_generate_without_terraform_datacenter(self):
        """Test generating HCL without Terraform templating."""
        config = JobConfig(
            job_name="static-dc",
            image="nginx:latest",
            use_terraform_datacenter=False,
            datacenters=["dc1", "dc2"],
        )

        hcl = generate_hcl(config)

        assert '["dc1", "dc2"]' in hcl
        assert '${datacenter}' not in hcl

    def test_generate_static_port(self):
        """Test generating HCL with static port allocation."""
        config = JobConfig(
            job_name="static-port-app",
            image="nginx:latest",
            network_mode=NetworkMode.HOST,
            ports=[PortConfig(name="http", container_port=80, static=True, host_port=80)],
        )

        hcl = generate_hcl(config)

        assert 'mode = "host"' in hcl
        assert "static = 80" in hcl


class TestValidateHCL:
    """Tests for HCL validation."""

    def test_basic_validation_valid(self, sample_hcl):
        """Test that valid HCL passes basic validation."""
        is_valid, error = _basic_hcl_validation(sample_hcl)

        assert is_valid is True
        assert error is None

    def test_basic_validation_unbalanced_braces(self):
        """Test that unbalanced braces are detected."""
        invalid_hcl = 'job "test" { group "g" {'

        is_valid, error = _basic_hcl_validation(invalid_hcl)

        assert is_valid is False
        assert "braces" in error.lower()

    def test_basic_validation_missing_job(self):
        """Test that missing job block is detected."""
        invalid_hcl = 'group "test" { task "t" { driver = "docker" } }'

        is_valid, error = _basic_hcl_validation(invalid_hcl)

        assert is_valid is False
        assert "job" in error.lower()

    def test_basic_validation_missing_driver(self):
        """Test that missing driver is detected."""
        invalid_hcl = '''job "test" {
          group "g" {
            task "t" {
              config {
                image = "nginx"
              }
            }
          }
        }'''

        is_valid, error = _basic_hcl_validation(invalid_hcl)

        assert is_valid is False
        assert "driver" in error.lower()


class TestExtractJobName:
    """Tests for job name extraction."""

    def test_extract_simple_name(self):
        """Test extracting job name from HCL."""
        hcl = 'job "my-service" { type = "service" }'

        result = extract_job_name(hcl)

        assert result == "my-service"

    def test_extract_with_whitespace(self):
        """Test extraction handles whitespace."""
        hcl = 'job   "spaced-job"   {'

        result = extract_job_name(hcl)

        assert result == "spaced-job"

    def test_extract_no_job(self):
        """Test extraction returns None when no job block."""
        hcl = 'group "test" {}'

        result = extract_job_name(hcl)

        assert result is None


class TestSanitizeJobName:
    """Tests for job name sanitization."""

    def test_sanitize_valid_name(self):
        """Test that valid name passes through."""
        result = sanitize_job_name("my-valid-job")
        assert result == "my-valid-job"

    def test_sanitize_uppercase(self):
        """Test uppercase is converted to lowercase."""
        result = sanitize_job_name("My-Job-Name")
        assert result == "my-job-name"

    def test_sanitize_special_chars(self):
        """Test special characters are replaced."""
        result = sanitize_job_name("my_job.name@test")
        assert result == "my-job-name-test"

    def test_sanitize_consecutive_hyphens(self):
        """Test consecutive hyphens are collapsed."""
        result = sanitize_job_name("my---job")
        assert result == "my-job"

    def test_sanitize_leading_number(self):
        """Test leading number is handled."""
        result = sanitize_job_name("123job")
        assert result == "job-123job"

    def test_sanitize_truncates_long_name(self):
        """Test that long names are truncated to 63 chars."""
        long_name = "a" * 100
        result = sanitize_job_name(long_name)
        assert len(result) == 63


class TestMergeConfigs:
    """Tests for config merging."""

    def test_merge_basic_override(self):
        """Test basic value override."""
        base = JobConfig(job_name="base", image="nginx:1.0", cpu=500)
        overrides = {"cpu": 1000, "memory": 1024}

        result = merge_hcl_configs(base, overrides)

        assert result.job_name == "base"
        assert result.image == "nginx:1.0"
        assert result.cpu == 1000
        assert result.memory == 1024

    def test_merge_list_values(self):
        """Test merging list values like ports."""
        base = JobConfig(
            job_name="base",
            image="nginx",
            ports=[PortConfig(name="http", container_port=80)],
        )
        new_ports = [
            PortConfig(name="http", container_port=8080),
            PortConfig(name="https", container_port=443),
        ]
        overrides = {"ports": new_ports}

        result = merge_hcl_configs(base, overrides)

        assert len(result.ports) == 2
        assert result.ports[0].container_port == 8080

    def test_merge_preserves_unspecified(self):
        """Test that unspecified values are preserved."""
        base = JobConfig(
            job_name="base",
            image="nginx",
            cpu=500,
            memory=1024,
            health_check_path="/custom",
        )
        overrides = {"cpu": 2000}

        result = merge_hcl_configs(base, overrides)

        assert result.memory == 1024
        assert result.health_check_path == "/custom"

    def test_merge_dict_env_vars(self):
        """Test merging dict values like env_vars."""
        base = JobConfig(
            job_name="base",
            image="nginx",
            env_vars={"KEY1": "value1"},
        )
        overrides = {"env_vars": {"KEY1": "updated", "KEY2": "value2"}}

        result = merge_hcl_configs(base, overrides)

        # Dict merge extends, doesn't replace
        assert result.env_vars["KEY1"] == "updated"
        assert result.env_vars["KEY2"] == "value2"
