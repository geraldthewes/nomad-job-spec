"""Observability with LangFuse tracing and prompt management."""

import functools
import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, TypeVar

from config.settings import Settings, get_settings

if TYPE_CHECKING:
    from langfuse import Langfuse
    from langfuse.langchain import CallbackHandler

logger = logging.getLogger(__name__)

# Type variable for generic function decoration
F = TypeVar("F", bound=Callable[..., Any])


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

        Note: In LangFuse v3+, the CallbackHandler reads credentials from
        environment variables (LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY,
        LANGFUSE_HOST). Trace metadata like session_id and user_id should
        be passed via LangChain's config metadata.

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

            # LangFuse v3+ CallbackHandler reads credentials from env vars
            # Trace attributes are passed via LangChain config metadata
            handler = CallbackHandler()

            # Store trace context for use in LangChain config
            handler._trace_name = trace_name
            handler._session_id = session_id
            handler._user_id = user_id
            handler._metadata = metadata

            return handler
        except ImportError:
            logger.warning("langfuse.langchain not available")
            return None
        except Exception as e:
            logger.warning(f"Failed to create LangFuse handler: {e}")
            return None

    def get_langchain_config(
        self,
        handler: "CallbackHandler | None",
    ) -> dict[str, Any]:
        """Get LangChain config dict with LangFuse metadata.

        Use this to pass trace metadata when invoking LangChain components.

        Args:
            handler: The CallbackHandler from get_handler().

        Returns:
            Config dict with callbacks and metadata for LangChain invoke().
        """
        if handler is None:
            return {}

        config: dict[str, Any] = {"callbacks": [handler]}

        # Build metadata from stored trace context
        metadata: dict[str, Any] = {}
        if hasattr(handler, "_session_id") and handler._session_id:
            metadata["langfuse_session_id"] = handler._session_id
        if hasattr(handler, "_user_id") and handler._user_id:
            metadata["langfuse_user_id"] = handler._user_id
        if hasattr(handler, "_trace_name") and handler._trace_name:
            metadata["langfuse_trace_name"] = handler._trace_name
        if hasattr(handler, "_metadata") and handler._metadata:
            metadata.update(handler._metadata)

        if metadata:
            config["metadata"] = metadata

        return config

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

    @contextmanager
    def span(
        self,
        name: str,
        trace: Any = None,
        parent: Any = None,
        input: Any = None,
        metadata: dict[str, Any] | None = None,
    ):
        """Context manager for creating a span within a trace.

        Usage:
            with obs.span("parse_dockerfile", input={"path": dockerfile_path}) as span:
                result = parse_dockerfile(dockerfile_path)
                span.end(output=result)

        Args:
            name: Name for the span (e.g., "parse_dockerfile", "query_vault").
            trace: Parent trace object. If None, creates a new trace.
            parent: Parent span for nesting.
            input: Input data to record with the span.
            metadata: Additional metadata for the span.

        Yields:
            Span object (or a no-op wrapper if LangFuse is unavailable).
        """
        client = self.get_client()

        if client is None:
            # Return a no-op span wrapper
            yield _NoOpSpan()
            return

        try:
            # Create trace if not provided
            if trace is None and parent is None:
                trace = client.trace(name=f"{name}-trace")

            # Create the span
            span_kwargs: dict[str, Any] = {"name": name}
            if input is not None:
                span_kwargs["input"] = input
            if metadata:
                span_kwargs["metadata"] = metadata

            if parent is not None:
                span_obj = parent.span(**span_kwargs)
            elif trace is not None:
                span_obj = trace.span(**span_kwargs)
            else:
                # Fallback: create as a trace
                span_obj = client.trace(**span_kwargs)

            try:
                yield span_obj
            except Exception as e:
                # Record error in span
                span_obj.end(
                    level="ERROR",
                    status_message=str(e),
                )
                raise
            else:
                # Span ended successfully (output should be set by caller via span.end())
                pass

        except Exception as e:
            logger.warning(f"Failed to create span '{name}': {e}")
            yield _NoOpSpan()

    def traced(
        self,
        name: str | None = None,
        capture_input: bool = True,
        capture_output: bool = True,
    ) -> Callable[[F], F]:
        """Decorator for tracing function calls.

        Usage:
            @obs.traced("parse_dockerfile")
            def parse_dockerfile(path: str) -> DockerfileInfo:
                ...

        Args:
            name: Span name. Defaults to function name.
            capture_input: Whether to capture function arguments.
            capture_output: Whether to capture return value.

        Returns:
            Decorated function.
        """
        def decorator(func: F) -> F:
            span_name = name or func.__name__

            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                # Build input dict if capturing
                input_data = None
                if capture_input:
                    input_data = {"args": args, "kwargs": kwargs}

                with self.span(span_name, input=input_data) as span:
                    result = func(*args, **kwargs)
                    if capture_output and hasattr(span, "end"):
                        span.end(output=result)
                    return result

            return wrapper  # type: ignore

        return decorator

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


class _NoOpSpan:
    """No-op span for when LangFuse is unavailable."""

    def end(self, **kwargs: Any) -> None:
        """No-op end method."""
        pass

    def span(self, **kwargs: Any) -> "_NoOpSpan":
        """Return another no-op span for nesting."""
        return _NoOpSpan()

    def update(self, **kwargs: Any) -> None:
        """No-op update method."""
        pass


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
