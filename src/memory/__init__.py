"""Memory layer using Mem0 and Qdrant."""

from dataclasses import dataclass
from typing import Any

from mem0 import Memory

from config.settings import get_settings


@dataclass
class EnvVarMemory:
    """Remembered environment variable configuration."""

    name: str
    source: str  # fixed, consul, vault
    value_pattern: str  # The value or pattern template


_memory_client: Memory | None = None


def get_memory_client() -> Memory | None:
    """Get or create the global Mem0 client instance.

    Returns None if memory is disabled in settings or unavailable.
    """
    global _memory_client
    settings = get_settings()

    if not settings.memory_enabled:
        return None

    if _memory_client is None:
        try:
            config = {
                "vector_store": {
                    "provider": "qdrant",
                    "config": {
                        "host": settings.qdrant_host,
                        "port": settings.qdrant_port,
                        "collection_name": settings.qdrant_collection,
                    },
                },
            }
            _memory_client = Memory.from_config(config)
        except Exception:
            # Qdrant not available - memory features disabled
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
    except Exception:
        # Memory unavailable - gracefully continue without memory
        pass

    return None


def save_env_var_config(var_name: str, source: str, value: str) -> bool:
    """Save an environment variable configuration to memory.

    Args:
        var_name: The environment variable name.
        source: The source type (fixed, consul, vault).
        value: The value or path pattern.

    Returns:
        True if saved successfully, False otherwise.
    """
    try:
        client = get_memory_client()
        if client is None:
            return False

        # For value patterns, generalize app-specific paths
        # e.g., "myapp/config/foo" -> "{app_name}/config/foo"
        value_pattern = value

        memory_text = f"ENV_VAR_CONFIG: {var_name} -> source={source}, value_pattern={value_pattern}"
        client.add(memory_text, user_id="global")
        return True
    except Exception:
        return False


def save_env_var_configs_batch(
    configs: list[dict[str, Any]], original_configs: list[dict[str, Any]]
) -> int:
    """Save multiple environment variable configurations that were changed by user.

    Only saves configs where the user changed the source or value from the original.

    Args:
        configs: List of confirmed env config dicts with name, source, value.
        original_configs: List of original suggested configs before user edits.

    Returns:
        Number of configs saved to memory.
    """
    # Build lookup of original configs
    original_lookup = {c["name"]: c for c in original_configs}

    saved_count = 0
    for cfg in configs:
        name = cfg["name"]
        source = cfg["source"]
        value = cfg["value"]

        # Skip if source is still unknown (user didn't configure it)
        if source == "unknown":
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
                if save_env_var_config(name, source, value):
                    saved_count += 1

    return saved_count
