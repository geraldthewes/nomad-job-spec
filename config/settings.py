"""Application settings using Pydantic Settings."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM Configuration
    llm_provider: Literal["vllm", "openai", "anthropic"] = Field(
        default="vllm",
        description="LLM provider to use",
    )
    vllm_base_url: str = Field(
        default="http://localhost:8000",
        description="vLLM server base URL",
    )
    vllm_model: str = Field(
        default="Qwen/Qwen3-32B",
        description="Model name for vLLM",
    )
    openai_api_key: str | None = Field(
        default=None,
        description="OpenAI API key (if using OpenAI provider)",
    )
    anthropic_api_key: str | None = Field(
        default=None,
        description="Anthropic API key (if using Anthropic provider)",
    )
    llm_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="LLM temperature for generation",
    )
    llm_max_tokens: int = Field(
        default=4096,
        description="Maximum tokens for LLM response",
    )

    # Nomad Configuration
    nomad_addr: str = Field(
        default="http://localhost:4646",
        description="Nomad cluster address",
    )
    nomad_token: str | None = Field(
        default=None,
        description="Nomad ACL token",
    )
    nomad_namespace: str = Field(
        default="default",
        description="Nomad namespace for job deployment",
    )
    nomad_region: str = Field(
        default="global",
        description="Nomad region",
    )
    nomad_datacenter: str = Field(
        default="dc1",
        description="Default datacenter for jobs",
    )

    # Memory Layer Configuration (Mem0 + Qdrant)
    qdrant_host: str = Field(
        default="localhost",
        description="Qdrant server host",
    )
    qdrant_port: int = Field(
        default=6333,
        description="Qdrant server port",
    )
    qdrant_collection: str = Field(
        default="nomad_agent",
        description="Qdrant collection name for agent memory",
    )
    memory_enabled: bool = Field(
        default=True,
        description="Enable memory layer (Mem0)",
    )

    # Observability Configuration (LangFuse)
    # Both public and secret keys are required when langfuse_enabled=True
    # See: https://langfuse.com/docs/sdk/python/low-level-sdk
    langfuse_enabled: bool = Field(
        default=False,
        description="Enable LangFuse tracing (requires public and secret keys)",
    )
    langfuse_public_key: str | None = Field(
        default=None,
        description="LangFuse public key (required when enabled)",
    )
    langfuse_secret_key: str | None = Field(
        default=None,
        description="LangFuse secret key (required when enabled)",
    )
    langfuse_base_url: str = Field(
        default="https://cloud.langfuse.com",
        description="LangFuse base URL",
    )

    @model_validator(mode="after")
    def validate_langfuse_keys(self) -> "Settings":
        """Validate that both LangFuse keys are provided when enabled."""
        if self.langfuse_enabled:
            missing = []
            if not self.langfuse_public_key:
                missing.append("LANGFUSE_PUBLIC_KEY")
            if not self.langfuse_secret_key:
                missing.append("LANGFUSE_SECRET_KEY")
            if missing:
                raise ValueError(
                    f"LangFuse is enabled but missing required keys: {', '.join(missing)}. "
                    "Set these environment variables or disable LangFuse with LANGFUSE_ENABLED=false"
                )
        return self

    # Agent Configuration
    max_iterations: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum retry iterations for failed deployments",
    )
    verification_timeout: int = Field(
        default=300,
        description="Timeout in seconds for deployment verification",
    )
    verification_poll_interval: int = Field(
        default=10,
        description="Poll interval in seconds for deployment status",
    )

    # Resource Defaults
    default_cpu: int = Field(
        default=500,
        description="Default CPU allocation in MHz",
    )
    default_memory: int = Field(
        default=256,
        description="Default memory allocation in MB",
    )

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level",
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()
