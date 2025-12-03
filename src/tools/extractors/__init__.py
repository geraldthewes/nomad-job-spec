"""Extractor registry for discovering and extracting deployment configuration.

The extractor pattern allows easy addition of new source types:
1. Create a new extractor class inheriting from BaseExtractor
2. Register it in EXTRACTORS dict
3. The extract node will automatically use it when appropriate files are found
"""

from src.tools.extractors.base import (
    BaseExtractor,
    ExtractionResult,
    HealthCheckConfig,
    PortConfig,
    ResourceConfig,
    VaultSecret,
)

# Registry of all available extractors
# Import extractors here as they are implemented
_EXTRACTORS: dict[str, type[BaseExtractor]] = {}


def register_extractor(extractor_class: type[BaseExtractor]) -> type[BaseExtractor]:
    """Decorator to register an extractor in the registry.

    Usage:
        @register_extractor
        class MyExtractor(BaseExtractor):
            ...
    """
    instance = extractor_class()
    _EXTRACTORS[instance.name] = extractor_class
    return extractor_class


def get_extractor(name: str) -> BaseExtractor | None:
    """Get an extractor instance by name."""
    extractor_class = _EXTRACTORS.get(name)
    if extractor_class:
        return extractor_class()
    return None


def get_all_extractors() -> list[BaseExtractor]:
    """Get instances of all registered extractors, sorted by priority (highest first)."""
    extractors = [cls() for cls in _EXTRACTORS.values()]
    return sorted(extractors, key=lambda e: e.priority, reverse=True)


def get_extractor_for_file(file_path: str, content: str | None = None) -> BaseExtractor | None:
    """Find the best extractor for a given file.

    Checks all registered extractors in priority order and returns
    the first one that can handle the file.

    Args:
        file_path: Path to the file.
        content: Optional file content for deeper inspection.

    Returns:
        The best matching extractor, or None if no extractor can handle the file.
    """
    for extractor in get_all_extractors():
        if extractor.can_extract(file_path, content):
            return extractor
    return None


# Import extractors to trigger registration
# These imports are at the bottom to avoid circular imports
try:
    from src.tools.extractors.jobforge import JobforgeExtractor  # noqa: F401
except ImportError:
    pass  # Extractor not yet implemented

try:
    from src.tools.extractors.makefile import (  # noqa: F401
        MakefileComposeExtractor,
        MakefileDockerExtractor,
    )
except ImportError:
    pass  # Extractors not yet implemented


__all__ = [
    "BaseExtractor",
    "ExtractionResult",
    "HealthCheckConfig",
    "PortConfig",
    "ResourceConfig",
    "VaultSecret",
    "register_extractor",
    "get_extractor",
    "get_all_extractors",
    "get_extractor_for_file",
]
