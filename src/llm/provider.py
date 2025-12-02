"""LLM provider abstraction layer supporting vLLM, OpenAI, and Anthropic."""

from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from config.settings import Settings, get_settings


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def get_llm(self) -> BaseChatModel:
        """Get a LangChain-compatible LLM instance."""
        ...

    @abstractmethod
    def get_model_name(self) -> str:
        """Get the model name being used."""
        ...


class VLLMProvider(LLMProvider):
    """Provider for vLLM-served models using OpenAI-compatible API."""

    def __init__(
        self,
        base_url: str,
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def get_llm(self) -> BaseChatModel:
        """Get a ChatOpenAI instance configured for vLLM."""
        return ChatOpenAI(
            openai_api_base=self.base_url,
            openai_api_key="EMPTY",  # vLLM doesn't require a key
            model_name=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    def get_model_name(self) -> str:
        return self.model


class OpenAIProvider(LLMProvider):
    """Provider for OpenAI models."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4-turbo-preview",
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def get_llm(self) -> BaseChatModel:
        """Get a ChatOpenAI instance."""
        return ChatOpenAI(
            openai_api_key=self.api_key,
            model_name=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    def get_model_name(self) -> str:
        return self.model


class AnthropicProvider(LLMProvider):
    """Provider for Anthropic Claude models."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-sonnet-20240229",
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def get_llm(self) -> BaseChatModel:
        """Get a ChatAnthropic instance."""
        # Import here to avoid requiring anthropic if not used
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            anthropic_api_key=self.api_key,
            model_name=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    def get_model_name(self) -> str:
        return self.model


def create_provider(settings: Settings | None = None) -> LLMProvider:
    """Create an LLM provider based on settings.

    Args:
        settings: Application settings. If None, uses default settings.

    Returns:
        Configured LLMProvider instance.

    Raises:
        ValueError: If provider configuration is invalid.
    """
    if settings is None:
        settings = get_settings()

    provider_type = settings.llm_provider.lower()

    if provider_type == "vllm":
        return VLLMProvider(
            base_url=settings.vllm_base_url,
            model=settings.vllm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )

    elif provider_type == "openai":
        if not settings.openai_api_key:
            raise ValueError("OpenAI API key is required for OpenAI provider")
        return OpenAIProvider(
            api_key=settings.openai_api_key,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )

    elif provider_type == "anthropic":
        if not settings.anthropic_api_key:
            raise ValueError("Anthropic API key is required for Anthropic provider")
        return AnthropicProvider(
            api_key=settings.anthropic_api_key,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )

    else:
        raise ValueError(f"Unknown LLM provider: {provider_type}")


@lru_cache
def get_llm(settings: Settings | None = None) -> BaseChatModel:
    """Get the configured LLM instance.

    This function is cached to reuse the same LLM instance.

    Args:
        settings: Application settings. If None, uses default settings.

    Returns:
        LangChain-compatible LLM instance.
    """
    provider = create_provider(settings)
    return provider.get_llm()


def get_llm_with_callbacks(
    callbacks: list[Any] | None = None,
    settings: Settings | None = None,
    trace_name: str | None = None,
    session_id: str | None = None,
) -> BaseChatModel:
    """Get the configured LLM instance with callbacks.

    Use this when you need tracing or other callbacks. If no callbacks
    are provided and LangFuse is enabled, automatically adds the
    LangFuse callback handler.

    Args:
        callbacks: List of LangChain callbacks (e.g., LangFuse handler).
        settings: Application settings. If None, uses default settings.
        trace_name: Optional trace name for LangFuse.
        session_id: Optional session ID for LangFuse trace grouping.

    Returns:
        LangChain-compatible LLM instance with callbacks configured.
    """
    if settings is None:
        settings = get_settings()

    llm = get_llm(settings)

    all_callbacks = list(callbacks) if callbacks else []

    # Add LangFuse handler if enabled and no explicit callbacks provided
    if not callbacks:
        from src.observability import get_observability

        obs = get_observability(settings)
        handler = obs.get_handler(trace_name=trace_name, session_id=session_id)
        if handler:
            all_callbacks.append(handler)

    if all_callbacks:
        return llm.with_config(callbacks=all_callbacks)
    return llm
