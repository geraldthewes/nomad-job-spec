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

    def create_trace(self, name: str, **kwargs: Any) -> "_SpanWrapper | _NoOpSpan":
        """Create a manual trace for custom instrumentation.

        In LangFuse v3+, traces are created implicitly via start_span().
        This method creates a root span that acts as the trace container.

        Args:
            name: Name for the trace.
            **kwargs: Additional trace parameters (input, metadata, etc.)

        Returns:
            A wrapped span object acting as the trace root, or _NoOpSpan if unavailable.
        """
        client = self.get_client()
        if client is None:
            return _NoOpSpan()

        try:
            # In v3+, start_span creates a span (and implicitly a trace)
            span_kwargs: dict[str, Any] = {"name": name}
            if "input" in kwargs:
                span_kwargs["input"] = kwargs["input"]
            if "metadata" in kwargs:
                span_kwargs["metadata"] = kwargs["metadata"]

            raw_span = client.start_span(**span_kwargs)
            return _SpanWrapper(raw_span)
        except Exception as e:
            logger.warning(f"Failed to create trace: {e}")
            return _NoOpSpan()

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
            trace: Parent span/trace object. If None, creates a new root span.
            parent: Parent span for nesting (alias for trace).
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

        # Use parent if provided, otherwise use trace
        parent_span = parent or trace
        span_obj = None

        try:
            # Build span kwargs
            span_kwargs: dict[str, Any] = {"name": name}
            if input is not None:
                span_kwargs["input"] = input
            if metadata:
                span_kwargs["metadata"] = metadata

            # Create span - either as child of parent or as new root
            if parent_span is not None and hasattr(parent_span, "start_span"):
                # parent_span might be a _SpanWrapper, which returns a wrapped span
                raw_or_wrapped = parent_span.start_span(**span_kwargs)
                if isinstance(raw_or_wrapped, _SpanWrapper):
                    wrapped_span = raw_or_wrapped
                else:
                    wrapped_span = _SpanWrapper(raw_or_wrapped)
            else:
                # Create new root span (implicitly creates trace)
                raw_span = client.start_span(**span_kwargs)
                wrapped_span = _SpanWrapper(raw_span)

        except Exception as e:
            logger.warning(f"Failed to create span '{name}': {e}")
            yield _NoOpSpan()
            return

        try:
            yield wrapped_span
        except Exception as e:
            # Record error in span
            wrapped_span.end(
                level="ERROR",
                status_message=str(e),
            )
            raise

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


class _SpanWrapper:
    """Wrapper for LangfuseSpan that provides a compatible end() method.

    In LangFuse v3+, end() only accepts end_time. This wrapper allows
    passing output, level, status_message etc. to end() for convenience.
    """

    def __init__(self, span: Any):
        self._span = span

    def end(
        self,
        output: Any = None,
        level: str | None = None,
        status_message: str | None = None,
        **kwargs: Any,
    ) -> None:
        """End the span with optional output, level, and status_message.

        In v3+, we call update() first to set these values, then end().
        """
        if self._span is None:
            return

        # Build update kwargs for any non-None values
        update_kwargs: dict[str, Any] = {}
        if output is not None:
            update_kwargs["output"] = output
        if level is not None:
            update_kwargs["level"] = level
        if status_message is not None:
            update_kwargs["status_message"] = status_message

        # Update the span with any provided values
        if update_kwargs and hasattr(self._span, "update"):
            self._span.update(**update_kwargs)

        # End the span
        if hasattr(self._span, "end"):
            self._span.end()

    def start_span(self, **kwargs: Any) -> "_SpanWrapper":
        """Create a child span."""
        if self._span is not None and hasattr(self._span, "start_span"):
            return _SpanWrapper(self._span.start_span(**kwargs))
        return _NoOpSpan()

    def update(self, **kwargs: Any) -> "_SpanWrapper":
        """Update span attributes."""
        if self._span is not None and hasattr(self._span, "update"):
            self._span.update(**kwargs)
        return self

    def update_trace(self, **kwargs: Any) -> "_SpanWrapper":
        """Update the parent trace."""
        if self._span is not None and hasattr(self._span, "update_trace"):
            self._span.update_trace(**kwargs)
        return self


class _NoOpSpan:
    """No-op span for when LangFuse is unavailable."""

    def end(self, **kwargs: Any) -> None:
        """No-op end method."""
        pass

    def start_span(self, **kwargs: Any) -> "_NoOpSpan":
        """Return another no-op span for nesting (v3+ API)."""
        return _NoOpSpan()

    def span(self, **kwargs: Any) -> "_NoOpSpan":
        """Return another no-op span for nesting (legacy API)."""
        return _NoOpSpan()

    def update(self, **kwargs: Any) -> "_NoOpSpan":
        """No-op update method."""
        return self

    def update_trace(self, **kwargs: Any) -> "_NoOpSpan":
        """No-op update_trace method."""
        return self


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
