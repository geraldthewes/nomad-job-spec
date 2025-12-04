"""Merge node for the LangGraph workflow.

This node combines extraction results from multiple sources into a
unified analysis, using priority to resolve conflicts.
"""

import logging
from typing import Any

from src.observability import get_observability
from src.tools.extractors import get_extractor

logger = logging.getLogger(__name__)

# Priority order for source types (higher = preferred)
# Used when extractor priority isn't available
DEFAULT_PRIORITIES = {
    "jobforge": 100,
    "makefile_docker": 70,
    "makefile_compose": 70,
    "dockerfile": 50,
    "app_code": 30,
}


def merge_extractions_node(state: dict[str, Any]) -> dict[str, Any]:
    """Merge extraction results into a unified analysis.

    This node combines results from multiple extractors, using priority
    to resolve conflicts. Higher priority sources override lower priority
    sources, but gaps are filled from lower priority sources.

    Args:
        state: Current graph state containing 'extractions'.

    Returns:
        Updated state with 'merged_extraction' field.
    """
    obs = get_observability()

    extractions = state.get("extractions", [])

    if not extractions:
        logger.info("No extractions to merge")
        return {
            "merged_extraction": {},
            "extraction_sources": {},
        }

    # Sort extractions by priority (highest first)
    def get_priority(extraction: dict) -> int:
        source_type = extraction.get("source_type", "")
        # Try to get priority from extractor
        extractor = get_extractor(source_type)
        if extractor:
            return extractor.priority
        # Fallback to default priorities
        return DEFAULT_PRIORITIES.get(source_type, 0)

    sorted_extractions = sorted(extractions, key=get_priority, reverse=True)

    with obs.span("merge") as span:
        merged = {}
        sources = {}  # Track which source provided each field

        # Fields to merge
        fields_to_merge = [
            "job_name",
            "docker_image",
            "registry_url",
            "image_name",
            "image_tag",
            "ports",
            "env_vars",
            "vault_secrets",
            "vault_policies",
            "resources",
            "health_check",
            "requires_gpu",
            "requires_amd64",
            "constraints",
            "requires_storage",
            "storage_path",
        ]

        for extraction in sorted_extractions:
            source_type = extraction.get("source_type", "unknown")
            confidence = extraction.get("confidence", 0.5)

            for field in fields_to_merge:
                value = extraction.get(field)

                # Skip None/empty values
                if value is None:
                    continue
                if isinstance(value, (list, dict)) and not value:
                    continue

                # Only set if not already set (higher priority sources are processed first)
                if field not in merged:
                    merged[field] = value
                    sources[field] = {
                        "source_type": source_type,
                        "confidence": confidence,
                        "source_file": extraction.get("source_file", ""),
                    }
                    logger.debug(
                        f"Merged {field} from {source_type} "
                        f"(confidence={confidence})"
                    )
                elif field in ("env_vars",) and isinstance(value, dict):
                    # For env_vars, merge dicts (add missing keys)
                    existing = merged.get(field, {})
                    for k, v in value.items():
                        if k not in existing:
                            existing[k] = v
                            if field not in sources:
                                sources[field] = {}
                            # Track source for new keys
                    merged[field] = existing
                elif field in ("vault_secrets", "vault_policies", "constraints", "ports"):
                    # For lists, we could merge but for now prefer higher priority
                    pass

        span.end(
            output={
                "fields_merged": list(merged.keys()),
                "sources": list(sources.keys()),
            }
        )

    return {
        "merged_extraction": merged,
        "extraction_sources": sources,
    }


def create_merge_node():
    """Create the merge node function.

    Returns:
        Node function for use in LangGraph.
    """
    return merge_extractions_node
