"""Extract node for the LangGraph workflow.

This node runs extractors on discovered source files to extract
deployment configuration.
"""

import logging
from typing import Any

from src.observability import get_observability
from src.tools.extractors import get_extractor, get_extractor_for_file, ExtractionResult

logger = logging.getLogger(__name__)


def extract_node(state: dict[str, Any]) -> dict[str, Any]:
    """Run extractors on discovered source files.

    This node iterates through the discovered sources and runs the
    appropriate extractor for each, collecting all extraction results.

    Args:
        state: Current graph state containing 'discovered_sources' and 'codebase_path'.

    Returns:
        Updated state with 'extractions' field containing list of ExtractionResult dicts.
    """
    obs = get_observability()
    trace = obs.create_trace(
        name="extract_node",
        input={
            "codebase_path": state.get("codebase_path"),
            "discovered_sources": state.get("discovered_sources"),
        },
    )

    discovered_sources = state.get("discovered_sources", {})
    codebase_path = state.get("codebase_path", "")
    extractions: list[dict[str, Any]] = []

    if not discovered_sources:
        logger.info("No sources discovered, skipping extraction")
        if trace:
            trace.end(output={"extractions_count": 0})
        return {
            **state,
            "extractions": [],
        }

    with obs.span("run_extractors", trace=trace) as span:
        for source_type, file_path in discovered_sources.items():
            # Get the appropriate extractor
            extractor = get_extractor(source_type)

            if not extractor:
                # Try to find an extractor by file pattern
                try:
                    with open(file_path) as f:
                        content = f.read()
                    extractor = get_extractor_for_file(file_path, content)
                except Exception as e:
                    logger.warning(f"Could not read {file_path}: {e}")
                    continue

            if not extractor:
                logger.warning(f"No extractor found for {source_type}: {file_path}")
                continue

            # Run the extraction
            with obs.span(f"extract_{source_type}", trace=trace) as extract_span:
                try:
                    result = extractor.extract(file_path, codebase_path)
                    extractions.append(result.to_dict())
                    logger.info(
                        f"Extracted from {source_type} ({file_path}): "
                        f"confidence={result.confidence}"
                    )
                    extract_span.end(
                        output={
                            "source_type": result.source_type,
                            "confidence": result.confidence,
                            "job_name": result.job_name,
                            "docker_image": result.docker_image,
                        }
                    )
                except Exception as e:
                    logger.error(f"Extraction failed for {source_type} ({file_path}): {e}")
                    extract_span.end(level="ERROR", status_message=str(e))

        span.end(output={"extractions_count": len(extractions)})

    if trace:
        trace.end(
            output={
                "extractions_count": len(extractions),
                "source_types": [e.get("source_type") for e in extractions],
            }
        )

    return {
        **state,
        "extractions": extractions,
    }


def create_extract_node():
    """Create the extract node function.

    Returns:
        Node function for use in LangGraph.
    """
    return extract_node
