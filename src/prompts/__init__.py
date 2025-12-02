"""Prompt management with LangFuse integration and local fallback."""

import json
import logging
from pathlib import Path
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from config.settings import Settings, get_settings

logger = logging.getLogger(__name__)

# Default prompts directory (relative to project root)
PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"


class PromptNotFoundError(Exception):
    """Raised when a prompt cannot be found in LangFuse or fallback."""

    pass


class PromptManager:
    """Manages prompt fetching with LangFuse primary and local fallback.

    Features:
    - Fetching prompts from LangFuse with version/label support
    - Automatic fallback to local JSON files when LangFuse unavailable
    - Caching for performance
    - Conversion to LangChain ChatPromptTemplate
    """

    def __init__(self, settings: Settings, prompts_dir: Path | None = None):
        """Initialize the prompt manager.

        Args:
            settings: Application settings.
            prompts_dir: Optional custom prompts directory.
        """
        self._settings = settings
        self._prompts_dir = prompts_dir or PROMPTS_DIR
        self._cache: dict[str, dict[str, Any]] = {}

    def _get_langfuse_client(self) -> Any:
        """Get LangFuse client from observability manager.

        Returns:
            The LangFuse client, or None if unavailable.
        """
        # Import here to avoid circular imports
        from src.observability import get_observability

        obs = get_observability(self._settings)
        return obs.get_client()

    def _fetch_from_langfuse(
        self,
        name: str,
        version: int | None = None,
        label: str | None = None,
    ) -> dict[str, Any] | None:
        """Fetch prompt from LangFuse.

        Args:
            name: Prompt name.
            version: Optional specific version number.
            label: Optional label (e.g., "production").

        Returns:
            Prompt data dict, or None if not found/unavailable.
        """
        client = self._get_langfuse_client()
        if client is None:
            return None

        try:
            # Fetch prompt from LangFuse
            # If version is specified, use it; otherwise use label (default: production)
            if version is not None:
                prompt = client.get_prompt(name, version=version)
            elif label:
                prompt = client.get_prompt(name, label=label)
            else:
                prompt = client.get_prompt(name)

            # Convert LangFuse prompt to our internal format
            if prompt is None:
                return None

            # LangFuse returns a Prompt object with get_langchain_prompt() method
            # or we can access the raw data
            if hasattr(prompt, "prompt"):
                # Chat prompt - prompt.prompt is a list of messages
                content = prompt.prompt
            else:
                # Text prompt
                content = str(prompt)

            return {
                "name": name,
                "type": "chat" if isinstance(content, list) else "text",
                "prompt": content,
                "config": getattr(prompt, "config", {}),
                "version": getattr(prompt, "version", 1),
            }

        except Exception as e:
            logger.debug(f"Failed to fetch prompt '{name}' from LangFuse: {e}")
            return None

    def _load_from_file(self, name: str) -> dict[str, Any] | None:
        """Load prompt from local JSON file.

        Args:
            name: Prompt name (used as filename without extension).

        Returns:
            Prompt data dict, or None if file not found/invalid.
        """
        file_path = self._prompts_dir / f"{name}.json"

        if not file_path.exists():
            logger.debug(f"Prompt file not found: {file_path}")
            return None

        try:
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)

            # Validate required fields
            if "prompt" not in data:
                logger.warning(f"Invalid prompt file {file_path}: missing 'prompt' field")
                return None

            return data

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse prompt file {file_path}: {e}")
            return None
        except Exception as e:
            logger.warning(f"Error loading prompt file {file_path}: {e}")
            return None

    def _convert_to_template(self, data: dict[str, Any]) -> ChatPromptTemplate:
        """Convert prompt data to LangChain ChatPromptTemplate.

        Args:
            data: Prompt data dict with 'prompt' field.

        Returns:
            LangChain ChatPromptTemplate.
        """
        prompt_content = data.get("prompt")
        prompt_type = data.get("type", "text")

        if prompt_type == "chat" and isinstance(prompt_content, list):
            # Chat prompt - list of message dicts
            messages = []
            for msg in prompt_content:
                role = msg.get("role", "system")
                content = msg.get("content", "")
                messages.append((role, content))
            return ChatPromptTemplate.from_messages(messages)
        else:
            # Text prompt - single string
            content = str(prompt_content)
            return ChatPromptTemplate.from_messages([("system", content)])

    def get_prompt(
        self,
        name: str,
        version: int | None = None,
        label: str | None = None,
        use_cache: bool = True,
    ) -> ChatPromptTemplate:
        """Get a prompt as a LangChain ChatPromptTemplate.

        Tries LangFuse first, falls back to local file if unavailable.

        Args:
            name: Prompt name.
            version: Optional specific version number.
            label: Optional label (e.g., "production").
            use_cache: Whether to use cached prompts.

        Returns:
            LangChain ChatPromptTemplate.

        Raises:
            PromptNotFoundError: If prompt not found in LangFuse or local file.
        """
        # Check cache first
        cache_key = f"{name}:{version}:{label}"
        if use_cache and cache_key in self._cache:
            return self._convert_to_template(self._cache[cache_key])

        # Try LangFuse first
        data = self._fetch_from_langfuse(name, version, label)

        if data is not None:
            logger.debug(f"Loaded prompt '{name}' from LangFuse")
            if use_cache:
                self._cache[cache_key] = data
            return self._convert_to_template(data)

        # Fall back to local file
        data = self._load_from_file(name)

        if data is not None:
            logger.debug(f"Loaded prompt '{name}' from local file")
            if use_cache:
                self._cache[cache_key] = data
            return self._convert_to_template(data)

        raise PromptNotFoundError(
            f"Prompt '{name}' not found in LangFuse or local prompts directory"
        )

    def get_prompt_text(
        self,
        name: str,
        version: int | None = None,
        label: str | None = None,
    ) -> str:
        """Get raw prompt text (system message content).

        Convenience method that extracts the system message content
        from a chat prompt.

        Args:
            name: Prompt name.
            version: Optional specific version number.
            label: Optional label (e.g., "production").

        Returns:
            The system message content as a string.

        Raises:
            PromptNotFoundError: If prompt not found.
        """
        # Check cache first
        cache_key = f"{name}:{version}:{label}"
        if cache_key in self._cache:
            data = self._cache[cache_key]
        else:
            # Try LangFuse
            data = self._fetch_from_langfuse(name, version, label)

            if data is None:
                # Fall back to file
                data = self._load_from_file(name)

            if data is None:
                raise PromptNotFoundError(
                    f"Prompt '{name}' not found in LangFuse or local prompts directory"
                )

            self._cache[cache_key] = data

        # Extract text from prompt data
        prompt_content = data.get("prompt")
        prompt_type = data.get("type", "text")

        if prompt_type == "chat" and isinstance(prompt_content, list):
            # Chat prompt - find system message
            for msg in prompt_content:
                if msg.get("role") == "system":
                    return msg.get("content", "")
            # No system message, return first message content
            if prompt_content:
                return prompt_content[0].get("content", "")
            return ""
        else:
            # Text prompt
            return str(prompt_content)

    def list_prompts(self) -> list[str]:
        """List available prompt names from local files.

        Returns:
            List of prompt names (without .json extension).
        """
        if not self._prompts_dir.exists():
            return []

        return [
            f.stem
            for f in self._prompts_dir.glob("*.json")
            if f.is_file()
        ]

    def clear_cache(self) -> None:
        """Clear the prompt cache."""
        self._cache.clear()


# Module-level singleton
_prompt_manager: PromptManager | None = None


def get_prompt_manager(settings: Settings | None = None) -> PromptManager:
    """Get the global PromptManager instance.

    Args:
        settings: Optional custom settings. If None, uses default settings.

    Returns:
        The PromptManager instance.
    """
    global _prompt_manager

    if settings is not None:
        # Custom settings always creates a new instance
        return PromptManager(settings)

    if _prompt_manager is None:
        _prompt_manager = PromptManager(get_settings())

    return _prompt_manager


def reset_prompt_manager() -> None:
    """Reset the global prompt manager.

    Useful for testing to ensure clean state between tests.
    """
    global _prompt_manager
    _prompt_manager = None
