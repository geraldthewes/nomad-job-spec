"""Observability with LangFuse tracing and prompt management."""

import logging
from typing import TYPE_CHECKING, Any

from config.settings import Settings, get_settings

if TYPE_CHECKING:
    from langfuse import Langfuse
    from langfuse.langchain import CallbackHandler

logger = logging.getLogger(__name__)


class ObservabilityManager:
    """Manages LangFuse integration for tracing and observability.

    Features:
    - Lazy initialization of LangFuse client
    - Callback handler creation for LangChain integration
    - Graceful degradation when LangFuse is disabled or unavailable
    - Connection health checking via auth_check()
    """

    def __init__(self, settings: Settings):
        """Initialize the observability manager.

        Args:
            settings: Application settings with LangFuse configuration.
        """
        self._settings = settings
        self._client: "Langfuse | None" = None
        self._initialized = False
        self._available = False

    def _initialize(self) -> None:
        """Lazy initialization of LangFuse client.

        Attempts to create and verify the LangFuse connection.
        On failure, logs a warning and marks as unavailable.
        """
        if self._initialized:
            return

        self._initialized = True

        if not self._settings.langfuse_enabled:
            logger.debug("LangFuse is disabled in settings")
            return

        try:
            from langfuse import Langfuse

            self._client = Langfuse(
                public_key=self._settings.langfuse_public_key,
                secret_key=self._settings.langfuse_secret_key,
                host=self._settings.langfuse_base_url,
            )

            # Verify connection
            if self._client.auth_check():
                self._available = True
                logger.info("LangFuse connection verified successfully")
            else:
                logger.warning("LangFuse auth_check failed - tracing disabled")
                self._client = None

        except ImportError:
            logger.warning("langfuse package not installed - tracing disabled")
        except Exception as e:
            logger.warning(f"Failed to initialize LangFuse: {e} - tracing disabled")
            self._client = None

    def is_enabled(self) -> bool:
        """Check if LangFuse is enabled and available.

        Returns:
            True if LangFuse is configured, connected, and ready for use.
        """
        self._initialize()
        return self._available

    def get_client(self) -> "Langfuse | None":
        """Get the LangFuse client, or None if unavailable.

        Returns:
            The LangFuse client instance, or None if disabled/unavailable.
        """
        self._initialize()
        return self._client

    def get_handler(
        self,
        trace_name: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "CallbackHandler | None":
        """Get a LangChain callback handler for tracing.

        Creates a new CallbackHandler instance that will trace LLM calls
        to LangFuse. Each handler creates a new trace.

        Args:
            trace_name: Name for the trace (e.g., "nomad-job-spec").
            session_id: Session identifier for grouping related traces.
            user_id: User identifier for the trace.
            metadata: Additional metadata to attach to the trace.

        Returns:
            A CallbackHandler instance, or None if LangFuse is unavailable.
        """
        if not self.is_enabled():
            return None

        try:
            from langfuse.langchain import CallbackHandler

            return CallbackHandler(
                public_key=self._settings.langfuse_public_key,
                secret_key=self._settings.langfuse_secret_key,
                host=self._settings.langfuse_base_url,
                trace_name=trace_name,
                session_id=session_id,
                user_id=user_id,
                metadata=metadata,
            )
        except ImportError:
            logger.warning("langfuse.callback not available")
            return None
        except Exception as e:
            logger.warning(f"Failed to create LangFuse handler: {e}")
            return None

    def create_trace(self, name: str, **kwargs: Any) -> Any:
        """Create a manual trace for custom instrumentation.

        Args:
            name: Name for the trace.
            **kwargs: Additional trace parameters (session_id, user_id, etc.)

        Returns:
            A Trace object, or None if LangFuse is unavailable.
        """
        client = self.get_client()
        if client is None:
            return None

        try:
            return client.trace(name=name, **kwargs)
        except Exception as e:
            logger.warning(f"Failed to create trace: {e}")
            return None

    def flush(self) -> None:
        """Flush pending traces to LangFuse.

        Call this before application exit to ensure all traces are sent.
        """
        if self._client is not None:
            try:
                self._client.flush()
            except Exception as e:
                logger.warning(f"Failed to flush LangFuse: {e}")

    def shutdown(self) -> None:
        """Shutdown LangFuse client cleanly.

        Call this on application shutdown for clean resource cleanup.
        """
        if self._client is not None:
            try:
                self._client.flush()
                self._client.shutdown()
            except Exception as e:
                logger.warning(f"Error during LangFuse shutdown: {e}")
            finally:
                self._client = None
                self._available = False


# Module-level singleton
_observability: ObservabilityManager | None = None


def get_observability(settings: Settings | None = None) -> ObservabilityManager:
    """Get the global ObservabilityManager instance.

    Creates a singleton instance on first call. If custom settings are
    provided, creates a new instance with those settings.

    Args:
        settings: Optional custom settings. If None, uses default settings.

    Returns:
        The ObservabilityManager instance.
    """
    global _observability

    if settings is not None:
        # Custom settings always creates a new instance
        return ObservabilityManager(settings)

    if _observability is None:
        _observability = ObservabilityManager(get_settings())

    return _observability


def reset_observability() -> None:
    """Reset the global observability manager.

    Useful for testing to ensure clean state between tests.
    """
    global _observability

    if _observability is not None:
        _observability.shutdown()
        _observability = None
