"""Tests for the prompt management module."""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from config.settings import Settings
from src.prompts import (
    PromptManager,
    PromptNotFoundError,
    get_prompt_manager,
    reset_prompt_manager,
)


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
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
def temp_prompts_dir(tmp_path):
    """Create a temporary prompts directory with test files."""
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()

    # Create a test prompt file
    test_prompt = {
        "name": "test",
        "type": "chat",
        "prompt": [
            {"role": "system", "content": "You are a helpful assistant."}
        ],
        "config": {"model": "auto", "temperature": 0.5},
        "version": 1,
        "labels": ["production"],
        "tags": ["test"],
    }
    (prompts_dir / "test.json").write_text(json.dumps(test_prompt))

    # Create a text-type prompt
    text_prompt = {
        "name": "simple",
        "type": "text",
        "prompt": "This is a simple text prompt.",
        "version": 1,
    }
    (prompts_dir / "simple.json").write_text(json.dumps(text_prompt))

    return prompts_dir


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the global singleton before and after each test."""
    reset_prompt_manager()
    yield
    reset_prompt_manager()


class TestPromptManager:
    """Tests for PromptManager class."""

    def test_loads_from_file(self, mock_settings, temp_prompts_dir):
        """Verify loading prompt from local JSON file."""
        manager = PromptManager(mock_settings, prompts_dir=temp_prompts_dir)

        template = manager.get_prompt("test")

        # Verify it's a ChatPromptTemplate
        assert template is not None
        # Check that we can format it (this validates the structure)
        messages = template.format_messages()
        assert len(messages) == 1
        assert messages[0].content == "You are a helpful assistant."

    def test_get_prompt_text(self, mock_settings, temp_prompts_dir):
        """Verify get_prompt_text extracts the system message."""
        manager = PromptManager(mock_settings, prompts_dir=temp_prompts_dir)

        text = manager.get_prompt_text("test")

        assert text == "You are a helpful assistant."

    def test_text_prompt_type(self, mock_settings, temp_prompts_dir):
        """Verify text-type prompts are handled correctly."""
        manager = PromptManager(mock_settings, prompts_dir=temp_prompts_dir)

        text = manager.get_prompt_text("simple")

        assert text == "This is a simple text prompt."

    def test_prompt_not_found_error(self, mock_settings, temp_prompts_dir):
        """Verify PromptNotFoundError for missing prompt."""
        manager = PromptManager(mock_settings, prompts_dir=temp_prompts_dir)

        with pytest.raises(PromptNotFoundError) as exc_info:
            manager.get_prompt("nonexistent")

        assert "nonexistent" in str(exc_info.value)

    def test_caching(self, mock_settings, temp_prompts_dir):
        """Verify prompt caching works."""
        manager = PromptManager(mock_settings, prompts_dir=temp_prompts_dir)

        # First call should load from file
        template1 = manager.get_prompt("test")

        # Second call should use cache
        template2 = manager.get_prompt("test")

        # Both should work and return valid templates
        assert template1 is not None
        assert template2 is not None

        # Cache should have the entry
        assert "test:None:None" in manager._cache

    def test_clear_cache(self, mock_settings, temp_prompts_dir):
        """Verify cache clearing works."""
        manager = PromptManager(mock_settings, prompts_dir=temp_prompts_dir)

        manager.get_prompt("test")
        assert len(manager._cache) > 0

        manager.clear_cache()
        assert len(manager._cache) == 0

    def test_list_prompts(self, mock_settings, temp_prompts_dir):
        """Verify listing available prompts."""
        manager = PromptManager(mock_settings, prompts_dir=temp_prompts_dir)

        prompts = manager.list_prompts()

        assert "test" in prompts
        assert "simple" in prompts
        assert len(prompts) == 2

    def test_list_prompts_empty_dir(self, mock_settings, tmp_path):
        """Verify listing prompts from non-existent directory."""
        manager = PromptManager(mock_settings, prompts_dir=tmp_path / "nonexistent")

        prompts = manager.list_prompts()

        assert prompts == []

    def test_invalid_json_file(self, mock_settings, temp_prompts_dir):
        """Verify handling of invalid JSON files."""
        # Create an invalid JSON file
        (temp_prompts_dir / "invalid.json").write_text("not valid json {")

        manager = PromptManager(mock_settings, prompts_dir=temp_prompts_dir)

        with pytest.raises(PromptNotFoundError):
            manager.get_prompt("invalid")

    def test_missing_prompt_field(self, mock_settings, temp_prompts_dir):
        """Verify handling of JSON without 'prompt' field."""
        # Create a JSON file without the required 'prompt' field
        bad_prompt = {"name": "bad", "version": 1}
        (temp_prompts_dir / "bad.json").write_text(json.dumps(bad_prompt))

        manager = PromptManager(mock_settings, prompts_dir=temp_prompts_dir)

        with pytest.raises(PromptNotFoundError):
            manager.get_prompt("bad")

    def test_langfuse_fallback_to_file(self, mock_settings, temp_prompts_dir):
        """Verify fallback to file when LangFuse unavailable."""
        with patch("src.observability.get_observability") as mock_get_obs:
            # Mock observability to return None client
            mock_obs = MagicMock()
            mock_obs.get_client.return_value = None
            mock_get_obs.return_value = mock_obs

            manager = PromptManager(mock_settings, prompts_dir=temp_prompts_dir)

            # Should fall back to file
            text = manager.get_prompt_text("test")

            assert text == "You are a helpful assistant."

    def test_langfuse_prompt_used_when_available(
        self, mock_settings, temp_prompts_dir
    ):
        """Verify LangFuse prompt is used when available."""
        with patch("src.observability.get_observability") as mock_get_obs:
            # Mock LangFuse client with a prompt
            mock_prompt = MagicMock()
            mock_prompt.prompt = [
                {"role": "system", "content": "From LangFuse!"}
            ]
            mock_prompt.config = {}
            mock_prompt.version = 2

            mock_client = MagicMock()
            mock_client.get_prompt.return_value = mock_prompt

            mock_obs = MagicMock()
            mock_obs.get_client.return_value = mock_client
            mock_get_obs.return_value = mock_obs

            manager = PromptManager(mock_settings, prompts_dir=temp_prompts_dir)

            text = manager.get_prompt_text("test")

            assert text == "From LangFuse!"
            mock_client.get_prompt.assert_called_once()

    def test_langfuse_error_falls_back_to_file(
        self, mock_settings, temp_prompts_dir
    ):
        """Verify fallback to file when LangFuse raises exception."""
        with patch("src.observability.get_observability") as mock_get_obs:
            mock_client = MagicMock()
            mock_client.get_prompt.side_effect = Exception("LangFuse error")

            mock_obs = MagicMock()
            mock_obs.get_client.return_value = mock_client
            mock_get_obs.return_value = mock_obs

            manager = PromptManager(mock_settings, prompts_dir=temp_prompts_dir)

            # Should fall back to file
            text = manager.get_prompt_text("test")

            assert text == "You are a helpful assistant."


class TestGetPromptManager:
    """Tests for get_prompt_manager function."""

    def test_returns_singleton(self, mock_settings):
        """Verify get_prompt_manager returns the same instance."""
        with patch("src.prompts.get_settings", return_value=mock_settings):
            manager1 = get_prompt_manager()
            manager2 = get_prompt_manager()

            assert manager1 is manager2

    def test_custom_settings_creates_new(self, mock_settings):
        """Verify custom settings creates a new instance."""
        manager1 = get_prompt_manager(mock_settings)
        manager2 = get_prompt_manager(mock_settings)

        # Custom settings should create new instances each time
        assert manager1 is not manager2

    def test_reset_clears_singleton(self, mock_settings):
        """Verify reset_prompt_manager clears the singleton."""
        with patch("src.prompts.get_settings", return_value=mock_settings):
            manager1 = get_prompt_manager()
            reset_prompt_manager()
            manager2 = get_prompt_manager()

            assert manager1 is not manager2


class TestRealPromptFiles:
    """Tests using the real prompt files in the prompts/ directory."""

    def test_analysis_prompt_loads(self, mock_settings):
        """Verify the analysis.json prompt file loads correctly."""
        from src.prompts import PROMPTS_DIR

        if not (PROMPTS_DIR / "analysis.json").exists():
            pytest.skip("analysis.json not found in prompts directory")

        manager = PromptManager(mock_settings)
        text = manager.get_prompt_text("analysis")

        assert "DevOps" in text
        assert "Nomad" in text

    def test_generation_prompt_loads(self, mock_settings):
        """Verify the generation.json prompt file loads correctly."""
        from src.prompts import PROMPTS_DIR

        if not (PROMPTS_DIR / "generation.json").exists():
            pytest.skip("generation.json not found in prompts directory")

        manager = PromptManager(mock_settings)
        text = manager.get_prompt_text("generation")

        assert "Nomad" in text
        assert "job" in text.lower()

    def test_fix_prompt_loads(self, mock_settings):
        """Verify the fix.json prompt file loads correctly."""
        from src.prompts import PROMPTS_DIR

        if not (PROMPTS_DIR / "fix.json").exists():
            pytest.skip("fix.json not found in prompts directory")

        manager = PromptManager(mock_settings)
        text = manager.get_prompt_text("fix")

        assert "fix" in text.lower() or "error" in text.lower()
