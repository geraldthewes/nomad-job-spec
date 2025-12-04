"""Memory layer using Mem0 and Qdrant."""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from mem0 import Memory

from config.settings import get_settings

logger = logging.getLogger(__name__)

# Module-level state for "warn once" pattern
_connection_warned: bool = False
_initialization_attempted: bool = False
_memory_client: Memory | None = None


class SaveResult(Enum):
    """Result of a memory save operation."""

    SUCCESS = "success"
    DISABLED = "disabled"  # Memory feature disabled in settings
    CONNECTION_ERROR = "connection_error"
    NO_CLIENT = "no_client"


@dataclass
class BatchSaveResult:
    """Result of a batch save operation."""

    saved: int = 0
    skipped: int = 0
    failed: int = 0
    disabled: bool = False

    @property
    def attempted(self) -> int:
        """Total configs that were attempted (saved + failed)."""
        return self.saved + self.failed


@dataclass
class EnvVarMemory:
    """Remembered environment variable configuration."""

    name: str
    source: str  # fixed, consul, vault
    value_pattern: str  # The value or pattern template


def _log_connection_error(error: Exception, context: str) -> None:
    """Log a connection error once, then suppress repeated warnings.

    Args:
        error: The exception that occurred.
        context: Description of what operation was attempted.
    """
    global _connection_warned

    if _connection_warned:
        logger.debug(f"Qdrant {context} failed (suppressed): {error}")
        return

    _connection_warned = True

    # Parse error for better messaging
    error_str = str(error).lower()
    settings = get_settings()
    addr = f"{settings.qdrant_host}:{settings.qdrant_port}"

    if "connection refused" in error_str or "[errno 111]" in error_str:
        logger.warning(
            f"Qdrant connection refused at {addr}. "
            f"Memory features disabled. Check if Qdrant is running."
        )
    elif "timeout" in error_str or "timed out" in error_str:
        logger.warning(
            f"Qdrant connection timeout at {addr}. "
            f"Memory features disabled. Check network connectivity."
        )
    elif "name or service not known" in error_str or "nodename nor servname" in error_str:
        logger.warning(
            f"DNS resolution failed for Qdrant host '{settings.qdrant_host}'. "
            f"Memory features disabled. Check DNS or use IP address."
        )
    else:
        logger.warning(
            f"Qdrant connection error during {context}: {error}. "
            f"Memory features disabled."
        )


def check_qdrant_connectivity() -> bool:
    """Check if Qdrant is reachable and log status.

    This is an eager check that can be called at startup to provide
    immediate feedback about memory layer availability.

    Returns:
        True if Qdrant is reachable, False otherwise.
    """
    settings = get_settings()

    if not settings.memory_enabled:
        logger.debug("Memory layer disabled via settings")
        return False

    addr = f"{settings.qdrant_host}:{settings.qdrant_port}"

    try:
        import httpx

        health_url = f"http://{addr}/healthz"

        with httpx.Client(timeout=5.0) as client:
            response = client.get(health_url)
            response.raise_for_status()

        logger.info(f"Qdrant connectivity verified at {addr}")
        return True
    except Exception as e:
        _log_connection_error(e, "startup check")
        return False


def get_memory_client() -> Memory | None:
    """Get or create the global Mem0 client instance.

    Returns None if memory is disabled in settings or unavailable.
    Logs once on first connection failure.
    """
    global _memory_client, _initialization_attempted
    settings = get_settings()

    if not settings.memory_enabled:
        return None

    if _memory_client is None and not _initialization_attempted:
        _initialization_attempted = True
        try:
            from qdrant_client import QdrantClient

            # Create a custom Qdrant client with longer timeout
            # Default 5s timeout is too short for collection creation
            qdrant_client = QdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port,
                timeout=30,  # 30 second timeout for collection operations
            )

            config = {
                "vector_store": {
                    "provider": "qdrant",
                    "config": {
                        "client": qdrant_client,
                        "collection_name": settings.qdrant_collection,
                    },
                },
            }
            logger.debug(
                f"Initializing Mem0 with Qdrant at "
                f"{settings.qdrant_host}:{settings.qdrant_port}"
            )
            _memory_client = Memory.from_config(config)
            logger.info(
                f"Memory layer initialized (Qdrant: {settings.qdrant_host}:{settings.qdrant_port})"
            )
        except Exception as e:
            _log_connection_error(e, "initialization")
            return None

    return _memory_client


def search_env_var_config(var_name: str) -> EnvVarMemory | None:
    """Search memory for a previously configured environment variable.

    Args:
        var_name: The environment variable name to search for.

    Returns:
        EnvVarMemory if found, None otherwise.
    """
    try:
        client = get_memory_client()
        if client is None:
            return None

        # Search for exact env var configuration
        query = f"ENV_VAR_CONFIG: {var_name}"
        results = client.search(query, user_id="global", limit=1)

        if not results or not results.get("results"):
            return None

        # Parse the memory content
        for result in results["results"]:
            memory_text = result.get("memory", "")
            if f"ENV_VAR_CONFIG: {var_name}" in memory_text:
                # Parse: "ENV_VAR_CONFIG: VAR_NAME -> source=X, value_pattern=Y"
                parts = memory_text.split(" -> ", 1)
                if len(parts) == 2:
                    config_part = parts[1]
                    source = None
                    value_pattern = None

                    for item in config_part.split(", "):
                        if item.startswith("source="):
                            source = item.split("=", 1)[1]
                        elif item.startswith("value_pattern="):
                            value_pattern = item.split("=", 1)[1]

                    if source and value_pattern:
                        return EnvVarMemory(
                            name=var_name,
                            source=source,
                            value_pattern=value_pattern,
                        )
    except Exception as e:
        _log_connection_error(e, f"search for {var_name}")

    return None


def save_env_var_config(var_name: str, source: str, value: str) -> SaveResult:
    """Save an environment variable configuration to memory.

    Args:
        var_name: The environment variable name.
        source: The source type (fixed, consul, vault).
        value: The value or path pattern.

    Returns:
        SaveResult indicating the outcome.
    """
    settings = get_settings()
    if not settings.memory_enabled:
        return SaveResult.DISABLED

    try:
        client = get_memory_client()
        if client is None:
            return SaveResult.NO_CLIENT

        # For value patterns, generalize app-specific paths
        # e.g., "myapp/config/foo" -> "{app_name}/config/foo"
        value_pattern = value

        memory_text = f"ENV_VAR_CONFIG: {var_name} -> source={source}, value_pattern={value_pattern}"
        client.add(memory_text, user_id="global")
        logger.debug(f"Saved env var config to memory: {var_name}")
        return SaveResult.SUCCESS
    except Exception as e:
        _log_connection_error(e, f"save {var_name}")
        return SaveResult.CONNECTION_ERROR


def save_env_var_configs_batch(
    configs: list[dict[str, Any]], original_configs: list[dict[str, Any]]
) -> BatchSaveResult:
    """Save multiple environment variable configurations that were changed by user.

    Only saves configs where the user changed the source or value from the original.

    Args:
        configs: List of confirmed env config dicts with name, source, value.
        original_configs: List of original suggested configs before user edits.

    Returns:
        BatchSaveResult with counts of saved, skipped, and failed configs.
    """
    settings = get_settings()
    if not settings.memory_enabled:
        return BatchSaveResult(disabled=True, skipped=len(configs))

    # Build lookup of original configs
    original_lookup = {c["name"]: c for c in original_configs}

    result = BatchSaveResult()

    for cfg in configs:
        name = cfg["name"]
        source = cfg["source"]
        value = cfg["value"]

        # Skip if source is still unknown (user didn't configure it)
        if source == "unknown":
            result.skipped += 1
            continue

        # Check if this was changed from original
        original = original_lookup.get(name)
        if original:
            # Save if source or value was changed, or if originally unknown
            if (
                original.get("source") == "unknown"
                or original.get("source") != source
                or original.get("value") != value
            ):
                save_result = save_env_var_config(name, source, value)
                if save_result == SaveResult.SUCCESS:
                    result.saved += 1
                elif save_result == SaveResult.CONNECTION_ERROR:
                    result.failed += 1
                else:
                    result.skipped += 1
            else:
                result.skipped += 1
        else:
            result.skipped += 1

    return result
