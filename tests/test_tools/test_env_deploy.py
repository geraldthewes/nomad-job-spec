"""Tests for the env_deploy parser."""

import pytest
from pathlib import Path
from src.tools.env_deploy import (
    EnvDeployEntry,
    EnvDeployConfig,
    EnvDeployParseError,
    parse_env_deploy,
    validate_env_coverage,
    ValidationResult,
    convert_to_job_config_fields,
)


class TestEnvDeployEntry:
    """Tests for EnvDeployEntry dataclass."""

    def test_env_entry_to_dict(self):
        """Test env entry serialization."""
        entry = EnvDeployEntry(name="LOG_LEVEL", source="env", value="info")
        result = entry.to_dict()
        assert result == {
            "name": "LOG_LEVEL",
            "source": "env",
            "value": "info",
        }

    def test_vault_entry_to_dict(self):
        """Test vault entry serialization includes path and field."""
        entry = EnvDeployEntry(
            name="DB_PASSWORD",
            source="vault",
            value="secret/data/myapp/db:password",
            vault_path="secret/data/myapp/db",
            vault_field="password",
        )
        result = entry.to_dict()
        assert result == {
            "name": "DB_PASSWORD",
            "source": "vault",
            "value": "secret/data/myapp/db:password",
            "vault_path": "secret/data/myapp/db",
            "vault_field": "password",
        }

    def test_nomad_entry_to_dict(self):
        """Test nomad entry serialization."""
        entry = EnvDeployEntry(name="APP_PORT", source="nomad", value="assigned")
        result = entry.to_dict()
        assert result == {
            "name": "APP_PORT",
            "source": "nomad",
            "value": "assigned",
        }

    def test_from_dict_env(self):
        """Test creating env entry from dict."""
        data = {"name": "DEBUG", "source": "env", "value": "false"}
        entry = EnvDeployEntry.from_dict(data)
        assert entry.name == "DEBUG"
        assert entry.source == "env"
        assert entry.value == "false"

    def test_from_dict_vault(self):
        """Test creating vault entry from dict."""
        data = {
            "name": "API_KEY",
            "source": "vault",
            "value": "secret/data/app:key",
            "vault_path": "secret/data/app",
            "vault_field": "key",
        }
        entry = EnvDeployEntry.from_dict(data)
        assert entry.name == "API_KEY"
        assert entry.source == "vault"
        assert entry.vault_path == "secret/data/app"
        assert entry.vault_field == "key"


class TestEnvDeployConfig:
    """Tests for EnvDeployConfig dataclass."""

    def test_get_env_entries(self):
        """Test filtering env entries."""
        config = EnvDeployConfig(
            entries={
                "LOG_LEVEL": EnvDeployEntry(name="LOG_LEVEL", source="env", value="info"),
                "DB_PASS": EnvDeployEntry(
                    name="DB_PASS",
                    source="vault",
                    value="path:key",
                    vault_path="path",
                    vault_field="key",
                ),
                "PORT": EnvDeployEntry(name="PORT", source="nomad", value="assigned"),
            }
        )
        env_entries = config.get_env_entries()
        assert len(env_entries) == 1
        assert env_entries[0].name == "LOG_LEVEL"

    def test_get_vault_entries(self):
        """Test filtering vault entries."""
        config = EnvDeployConfig(
            entries={
                "LOG_LEVEL": EnvDeployEntry(name="LOG_LEVEL", source="env", value="info"),
                "DB_PASS": EnvDeployEntry(
                    name="DB_PASS",
                    source="vault",
                    value="path:key",
                    vault_path="path",
                    vault_field="key",
                ),
            }
        )
        vault_entries = config.get_vault_entries()
        assert len(vault_entries) == 1
        assert vault_entries[0].name == "DB_PASS"

    def test_get_nomad_entries(self):
        """Test filtering nomad entries."""
        config = EnvDeployConfig(
            entries={
                "LOG_LEVEL": EnvDeployEntry(name="LOG_LEVEL", source="env", value="info"),
                "APP_PORT": EnvDeployEntry(name="APP_PORT", source="nomad", value="assigned"),
            }
        )
        nomad_entries = config.get_nomad_entries()
        assert len(nomad_entries) == 1
        assert nomad_entries[0].name == "APP_PORT"

    def test_get_vault_paths_grouped(self):
        """Test grouping vault entries by path."""
        config = EnvDeployConfig(
            entries={
                "AWS_KEY": EnvDeployEntry(
                    name="AWS_KEY",
                    source="vault",
                    value="secret/data/aws:access_key",
                    vault_path="secret/data/aws",
                    vault_field="access_key",
                ),
                "AWS_SECRET": EnvDeployEntry(
                    name="AWS_SECRET",
                    source="vault",
                    value="secret/data/aws:secret_key",
                    vault_path="secret/data/aws",
                    vault_field="secret_key",
                ),
                "DB_PASS": EnvDeployEntry(
                    name="DB_PASS",
                    source="vault",
                    value="secret/data/db:password",
                    vault_path="secret/data/db",
                    vault_field="password",
                ),
            }
        )
        grouped = config.get_vault_paths_grouped()
        assert len(grouped) == 2
        assert len(grouped["secret/data/aws"]) == 2
        assert len(grouped["secret/data/db"]) == 1


class TestParseEnvDeploy:
    """Tests for parse_env_deploy function."""

    def test_parse_env_entry(self, tmp_path):
        """Test parsing env type entry."""
        env_file = tmp_path / ".env.deploy"
        env_file.write_text("env:LOG_LEVEL=info\n")

        config = parse_env_deploy(env_file)
        assert "LOG_LEVEL" in config.entries
        entry = config.entries["LOG_LEVEL"]
        assert entry.source == "env"
        assert entry.value == "info"

    def test_parse_vault_entry(self, tmp_path):
        """Test parsing vault type entry."""
        env_file = tmp_path / ".env.deploy"
        env_file.write_text("vault:DB_PASSWORD=secret/data/myapp/db:password\n")

        config = parse_env_deploy(env_file)
        assert "DB_PASSWORD" in config.entries
        entry = config.entries["DB_PASSWORD"]
        assert entry.source == "vault"
        assert entry.vault_path == "secret/data/myapp/db"
        assert entry.vault_field == "password"

    def test_parse_nomad_entry(self, tmp_path):
        """Test parsing nomad type entry."""
        env_file = tmp_path / ".env.deploy"
        env_file.write_text("nomad:APP_PORT=assigned\n")

        config = parse_env_deploy(env_file)
        assert "APP_PORT" in config.entries
        entry = config.entries["APP_PORT"]
        assert entry.source == "nomad"
        assert entry.value == "assigned"

    def test_parse_multiple_entries(self, tmp_path):
        """Test parsing file with multiple entries."""
        env_file = tmp_path / ".env.deploy"
        env_file.write_text(
            "env:LOG_LEVEL=info\n"
            "env:DEBUG=false\n"
            "vault:DB_PASSWORD=secret/data/app/db:password\n"
            "vault:AWS_KEY=secret/data/aws:access_key\n"
            "nomad:APP_PORT=assigned\n"
        )

        config = parse_env_deploy(env_file)
        assert len(config.entries) == 5
        assert config.entries["LOG_LEVEL"].source == "env"
        assert config.entries["DEBUG"].source == "env"
        assert config.entries["DB_PASSWORD"].source == "vault"
        assert config.entries["AWS_KEY"].source == "vault"
        assert config.entries["APP_PORT"].source == "nomad"

    def test_parse_with_comments(self, tmp_path):
        """Test parsing file with comments."""
        env_file = tmp_path / ".env.deploy"
        env_file.write_text(
            "# This is a comment\n"
            "env:LOG_LEVEL=info\n"
            "# Another comment\n"
            "vault:DB_PASS=path:key\n"
        )

        config = parse_env_deploy(env_file)
        assert len(config.entries) == 2

    def test_parse_with_empty_lines(self, tmp_path):
        """Test parsing file with empty lines."""
        env_file = tmp_path / ".env.deploy"
        env_file.write_text(
            "env:LOG_LEVEL=info\n"
            "\n"
            "   \n"
            "vault:DB_PASS=path:key\n"
        )

        config = parse_env_deploy(env_file)
        assert len(config.entries) == 2

    def test_parse_empty_value(self, tmp_path):
        """Test parsing entry with empty value."""
        env_file = tmp_path / ".env.deploy"
        env_file.write_text("env:EMPTY_VAR=\n")

        config = parse_env_deploy(env_file)
        assert config.entries["EMPTY_VAR"].value == ""

    def test_parse_value_with_equals(self, tmp_path):
        """Test parsing value containing equals sign."""
        env_file = tmp_path / ".env.deploy"
        env_file.write_text("env:CONNECTION_STRING=postgres://user:pass@host/db?sslmode=require\n")

        config = parse_env_deploy(env_file)
        assert config.entries["CONNECTION_STRING"].value == "postgres://user:pass@host/db?sslmode=require"

    def test_file_not_found(self, tmp_path):
        """Test error when file doesn't exist."""
        env_file = tmp_path / "nonexistent.env.deploy"

        with pytest.raises(FileNotFoundError) as exc_info:
            parse_env_deploy(env_file)
        assert "Environment deploy file not found" in str(exc_info.value)
        assert "Expected format" in str(exc_info.value)

    def test_invalid_format_no_colon(self, tmp_path):
        """Test error for line without type prefix."""
        env_file = tmp_path / ".env.deploy"
        env_file.write_text("LOG_LEVEL=info\n")

        with pytest.raises(EnvDeployParseError) as exc_info:
            parse_env_deploy(env_file)
        assert "Invalid format" in str(exc_info.value)
        assert "Line 1" in str(exc_info.value)

    def test_unknown_type_prefix(self, tmp_path):
        """Test error for unknown type prefix."""
        env_file = tmp_path / ".env.deploy"
        env_file.write_text("unknown:VAR=value\n")

        with pytest.raises(EnvDeployParseError) as exc_info:
            parse_env_deploy(env_file)
        assert "Unknown type 'unknown'" in str(exc_info.value)
        assert "env, nomad, vault" in str(exc_info.value)

    def test_vault_missing_field(self, tmp_path):
        """Test error when vault entry is missing field."""
        env_file = tmp_path / ".env.deploy"
        env_file.write_text("vault:DB_PASS=secret/data/app\n")

        with pytest.raises(EnvDeployParseError) as exc_info:
            parse_env_deploy(env_file)
        assert "must have format" in str(exc_info.value)
        assert "path:field" in str(exc_info.value)

    def test_vault_empty_field(self, tmp_path):
        """Test error when vault field is empty."""
        env_file = tmp_path / ".env.deploy"
        env_file.write_text("vault:DB_PASS=secret/data/app:\n")

        with pytest.raises(EnvDeployParseError) as exc_info:
            parse_env_deploy(env_file)
        assert "both path and field" in str(exc_info.value)

    def test_nomad_wrong_value(self, tmp_path):
        """Test error when nomad entry has wrong value."""
        env_file = tmp_path / ".env.deploy"
        env_file.write_text("nomad:APP_PORT=8080\n")

        with pytest.raises(EnvDeployParseError) as exc_info:
            parse_env_deploy(env_file)
        assert "must have value 'assigned'" in str(exc_info.value)

    def test_nomad_case_insensitive_assigned(self, tmp_path):
        """Test nomad entry accepts 'ASSIGNED' (case insensitive)."""
        env_file = tmp_path / ".env.deploy"
        env_file.write_text("nomad:APP_PORT=ASSIGNED\n")

        config = parse_env_deploy(env_file)
        assert config.entries["APP_PORT"].value == "assigned"

    def test_type_prefix_case_insensitive(self, tmp_path):
        """Test type prefix is case insensitive."""
        env_file = tmp_path / ".env.deploy"
        env_file.write_text(
            "ENV:VAR1=value1\n"
            "Vault:VAR2=path:field\n"
            "NOMAD:VAR3=assigned\n"
        )

        config = parse_env_deploy(env_file)
        assert config.entries["VAR1"].source == "env"
        assert config.entries["VAR2"].source == "vault"
        assert config.entries["VAR3"].source == "nomad"


class TestValidateEnvCoverage:
    """Tests for validate_env_coverage function."""

    def test_valid_complete_coverage(self):
        """Test validation passes with complete coverage."""
        dockerfile_vars = ["LOG_LEVEL", "DB_PASSWORD", "APP_PORT"]
        config = EnvDeployConfig(
            entries={
                "LOG_LEVEL": EnvDeployEntry(name="LOG_LEVEL", source="env", value="info"),
                "DB_PASSWORD": EnvDeployEntry(
                    name="DB_PASSWORD",
                    source="vault",
                    value="path:key",
                    vault_path="path",
                    vault_field="key",
                ),
                "APP_PORT": EnvDeployEntry(name="APP_PORT", source="nomad", value="assigned"),
            }
        )

        result = validate_env_coverage(dockerfile_vars, config)
        assert result.is_valid
        assert result.missing_vars == []
        assert result.extra_vars == []
        assert result.errors == []

    def test_missing_vars(self):
        """Test validation fails when vars are missing."""
        dockerfile_vars = ["LOG_LEVEL", "DB_PASSWORD", "APP_PORT"]
        config = EnvDeployConfig(
            entries={
                "LOG_LEVEL": EnvDeployEntry(name="LOG_LEVEL", source="env", value="info"),
            }
        )

        result = validate_env_coverage(dockerfile_vars, config)
        assert not result.is_valid
        assert sorted(result.missing_vars) == ["APP_PORT", "DB_PASSWORD"]
        assert len(result.errors) == 1
        assert "Missing .env.deploy entries" in result.errors[0]

    def test_extra_vars_allowed(self):
        """Test extra vars in .env.deploy are allowed but flagged."""
        dockerfile_vars = ["LOG_LEVEL"]
        config = EnvDeployConfig(
            entries={
                "LOG_LEVEL": EnvDeployEntry(name="LOG_LEVEL", source="env", value="info"),
                "EXTRA_VAR": EnvDeployEntry(name="EXTRA_VAR", source="env", value="extra"),
            }
        )

        result = validate_env_coverage(dockerfile_vars, config)
        assert result.is_valid  # Still valid
        assert result.missing_vars == []
        assert result.extra_vars == ["EXTRA_VAR"]

    def test_empty_dockerfile_vars(self):
        """Test validation with no Dockerfile vars."""
        dockerfile_vars: list[str] = []
        config = EnvDeployConfig(
            entries={
                "SOME_VAR": EnvDeployEntry(name="SOME_VAR", source="env", value="value"),
            }
        )

        result = validate_env_coverage(dockerfile_vars, config)
        assert result.is_valid
        assert result.extra_vars == ["SOME_VAR"]

    def test_empty_env_deploy(self):
        """Test validation with empty .env.deploy."""
        dockerfile_vars = ["LOG_LEVEL", "DB_PASSWORD"]
        config = EnvDeployConfig(entries={})

        result = validate_env_coverage(dockerfile_vars, config)
        assert not result.is_valid
        assert sorted(result.missing_vars) == ["DB_PASSWORD", "LOG_LEVEL"]

    def test_result_to_dict(self):
        """Test ValidationResult serialization."""
        result = ValidationResult(
            is_valid=False,
            missing_vars=["VAR1", "VAR2"],
            extra_vars=["VAR3"],
            errors=["Some error"],
        )
        data = result.to_dict()
        assert data == {
            "is_valid": False,
            "missing_vars": ["VAR1", "VAR2"],
            "extra_vars": ["VAR3"],
            "errors": ["Some error"],
        }


class TestConvertToJobConfigFields:
    """Tests for convert_to_job_config_fields function."""

    def test_converts_env_entries(self):
        """Test conversion of env type entries."""
        configs = [
            {"name": "LOG_LEVEL", "source": "env", "value": "info"},
            {"name": "DEBUG", "source": "env", "value": "false"},
        ]

        result = convert_to_job_config_fields(configs)

        assert result["env_vars"] == {"LOG_LEVEL": "info", "DEBUG": "false"}
        assert result["vault_secrets"] == {}
        assert result["nomad_port_vars"] == {}

    def test_converts_vault_entries(self):
        """Test conversion of vault type entries."""
        configs = [
            {
                "name": "DB_PASS",
                "source": "vault",
                "value": "secret/data/app/db:password",
                "vault_path": "secret/data/app/db",
                "vault_field": "password",
            },
        ]

        result = convert_to_job_config_fields(configs)

        assert result["env_vars"] == {}
        assert result["vault_secrets"] == {"DB_PASS": "secret/data/app/db#password"}
        assert result["nomad_port_vars"] == {}

    def test_converts_nomad_entries(self):
        """Test conversion of nomad type entries."""
        configs = [
            {"name": "APP_PORT", "source": "nomad", "value": "assigned"},
        ]

        result = convert_to_job_config_fields(configs)

        assert result["env_vars"] == {}
        assert result["vault_secrets"] == {}
        assert result["nomad_port_vars"] == {"APP_PORT": "http"}

    def test_converts_nomad_with_custom_port_label(self):
        """Test conversion with custom port label."""
        configs = [
            {"name": "APP_PORT", "source": "nomad", "value": "assigned"},
        ]

        result = convert_to_job_config_fields(configs, port_label="web")

        assert result["nomad_port_vars"] == {"APP_PORT": "web"}

    def test_converts_mixed_entries(self):
        """Test conversion of mixed source types."""
        configs = [
            {"name": "LOG_LEVEL", "source": "env", "value": "debug"},
            {
                "name": "DB_PASS",
                "source": "vault",
                "value": "secret/data/app:password",
                "vault_path": "secret/data/app",
                "vault_field": "password",
            },
            {"name": "APP_PORT", "source": "nomad", "value": "assigned"},
        ]

        result = convert_to_job_config_fields(configs)

        assert result["env_vars"] == {"LOG_LEVEL": "debug"}
        assert result["vault_secrets"] == {"DB_PASS": "secret/data/app#password"}
        assert result["nomad_port_vars"] == {"APP_PORT": "http"}

    def test_handles_vault_colon_format(self):
        """Test vault path with colon separator is converted to hash."""
        configs = [
            {
                "name": "API_KEY",
                "source": "vault",
                "value": "secret/data/api:key",
            },
        ]

        result = convert_to_job_config_fields(configs)

        assert result["vault_secrets"] == {"API_KEY": "secret/data/api#key"}

    def test_handles_empty_configs(self):
        """Test with empty config list."""
        result = convert_to_job_config_fields([])

        assert result["env_vars"] == {}
        assert result["vault_secrets"] == {}
        assert result["nomad_port_vars"] == {}
