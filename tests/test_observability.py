"""Tests for the observability module."""

import pytest
from unittest.mock import MagicMock, patch

from config.settings import Settings
from src.observability import (
    ObservabilityManager,
    get_observability,
    reset_observability,
)


@pytest.fixture
def disabled_settings():
    """Create settings with LangFuse disabled."""
    return Settings(
        llm_provider="vllm",
        vllm_base_url="http://localhost:8000/v1",
        vllm_model="test-model",
        nomad_addr="http://localhost:4646",
        nomad_datacenter="dc1",
        langfuse_enabled=False,
        memory_enabled=False,
    )


@pytest.fixture
def enabled_settings():
    """Create settings with LangFuse enabled."""
    return Settings(
        llm_provider="vllm",
        vllm_base_url="http://localhost:8000/v1",
        vllm_model="test-model",
        nomad_addr="http://localhost:4646",
        nomad_datacenter="dc1",
        langfuse_enabled=True,
        langfuse_public_key="test-public-key",
        langfuse_secret_key="test-secret-key",
        langfuse_base_url="https://cloud.langfuse.com",
        memory_enabled=False,
    )


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the global singleton before and after each test."""
    reset_observability()
    yield
    reset_observability()


class TestObservabilityManager:
    """Tests for ObservabilityManager class."""

    def test_disabled_by_default(self, disabled_settings):
        """Verify observability is disabled when langfuse_enabled=False."""
        manager = ObservabilityManager(disabled_settings)

        assert manager.is_enabled() is False
        assert manager.get_client() is None
        assert manager.get_handler() is None

    def test_is_enabled_checks_settings(self, disabled_settings):
        """Verify is_enabled returns False when disabled in settings."""
        manager = ObservabilityManager(disabled_settings)

        # Should be false because langfuse_enabled=False
        assert manager.is_enabled() is False

    @patch("langfuse.Langfuse")
    def test_enabled_with_valid_keys(self, mock_langfuse_class, enabled_settings):
        """Verify initialization with valid credentials."""
        # Mock the LangFuse client
        mock_client = MagicMock()
        mock_client.auth_check.return_value = True
        mock_langfuse_class.return_value = mock_client

        manager = ObservabilityManager(enabled_settings)

        assert manager.is_enabled() is True
        assert manager.get_client() is mock_client

        # Verify LangFuse was initialized with correct parameters
        mock_langfuse_class.assert_called_once_with(
            public_key="test-public-key",
            secret_key="test-secret-key",
            host="https://cloud.langfuse.com",
        )

    @patch("langfuse.Langfuse")
    def test_handles_auth_failure(self, mock_langfuse_class, enabled_settings):
        """Verify graceful degradation when auth_check fails."""
        mock_client = MagicMock()
        mock_client.auth_check.return_value = False
        mock_langfuse_class.return_value = mock_client

        manager = ObservabilityManager(enabled_settings)

        assert manager.is_enabled() is False
        assert manager.get_client() is None

    @patch("langfuse.Langfuse")
    def test_handles_connection_failure(self, mock_langfuse_class, enabled_settings):
        """Verify graceful degradation on connection failure."""
        mock_langfuse_class.side_effect = Exception("Connection failed")

        manager = ObservabilityManager(enabled_settings)

        # Should not raise, should just be disabled
        assert manager.is_enabled() is False
        assert manager.get_client() is None

    def test_handles_import_error(self, enabled_settings):
        """Verify graceful handling when langfuse package not installed."""
        manager = ObservabilityManager(enabled_settings)

        # Mock the import to fail
        with patch.dict("sys.modules", {"langfuse": None}):
            with patch("builtins.__import__", side_effect=ImportError("No module")):
                # Force re-initialization
                manager._initialized = False
                manager._initialize()

                assert manager.is_enabled() is False

    def test_get_handler_returns_handler(self, enabled_settings):
        """Verify get_handler returns a CallbackHandler when enabled."""
        # Import the langchain module first to ensure it's loaded
        import langfuse.langchain

        with patch("langfuse.Langfuse") as mock_langfuse_class:
            mock_client = MagicMock()
            mock_client.auth_check.return_value = True
            mock_langfuse_class.return_value = mock_client

            with patch.object(langfuse.langchain, "CallbackHandler") as mock_handler_class:
                mock_handler = MagicMock()
                mock_handler_class.return_value = mock_handler

                manager = ObservabilityManager(enabled_settings)
                handler = manager.get_handler(trace_name="test", session_id="session-123")

                assert handler is mock_handler
                mock_handler_class.assert_called_once_with(
                    public_key="test-public-key",
                    secret_key="test-secret-key",
                    host="https://cloud.langfuse.com",
                    trace_name="test",
                    session_id="session-123",
                    user_id=None,
                    metadata=None,
                )

    def test_get_handler_returns_none_when_disabled(self, disabled_settings):
        """Verify get_handler returns None when disabled."""
        manager = ObservabilityManager(disabled_settings)

        assert manager.get_handler() is None

    @patch("langfuse.Langfuse")
    def test_flush_calls_client_flush(self, mock_langfuse_class, enabled_settings):
        """Verify flush() calls the client's flush method."""
        mock_client = MagicMock()
        mock_client.auth_check.return_value = True
        mock_langfuse_class.return_value = mock_client

        manager = ObservabilityManager(enabled_settings)
        manager.is_enabled()  # Initialize
        manager.flush()

        mock_client.flush.assert_called_once()

    def test_flush_handles_no_client(self, disabled_settings):
        """Verify flush() is safe when there's no client."""
        manager = ObservabilityManager(disabled_settings)

        # Should not raise
        manager.flush()

    @patch("langfuse.Langfuse")
    def test_shutdown_cleans_up(self, mock_langfuse_class, enabled_settings):
        """Verify shutdown() properly cleans up resources."""
        mock_client = MagicMock()
        mock_client.auth_check.return_value = True
        mock_langfuse_class.return_value = mock_client

        manager = ObservabilityManager(enabled_settings)
        manager.is_enabled()  # Initialize
        manager.shutdown()

        mock_client.flush.assert_called_once()
        mock_client.shutdown.assert_called_once()
        assert manager._client is None
        assert manager._available is False


class TestGetObservability:
    """Tests for get_observability function."""

    def test_returns_singleton(self, disabled_settings):
        """Verify get_observability returns the same instance."""
        with patch("src.observability.get_settings", return_value=disabled_settings):
            manager1 = get_observability()
            manager2 = get_observability()

            assert manager1 is manager2

    def test_custom_settings_creates_new(self, disabled_settings):
        """Verify custom settings creates a new instance."""
        manager1 = get_observability(disabled_settings)
        manager2 = get_observability(disabled_settings)

        # Custom settings should create new instances each time
        assert manager1 is not manager2

    def test_reset_clears_singleton(self, disabled_settings):
        """Verify reset_observability clears the singleton."""
        with patch("src.observability.get_settings", return_value=disabled_settings):
            manager1 = get_observability()
            reset_observability()
            manager2 = get_observability()

            assert manager1 is not manager2
