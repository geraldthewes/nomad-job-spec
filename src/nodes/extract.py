"""Extract node for the LangGraph workflow.

This node runs extractors on discovered source files to extract
deployment configuration.

It uses build_system_analysis (from the analyze_build_system node) to
determine which extractor to use and on which file.
"""

import logging
import os
from typing import Any

from src.observability import get_observability
from src.tools.extractors import get_extractor, get_extractor_for_file, ExtractionResult

logger = logging.getLogger(__name__)

# Map build mechanism to extractor name
MECHANISM_TO_EXTRACTOR = {
    "jobforge": "jobforge",
    "docker": "dockerfile",
    "docker-compose": "docker_compose",
}


def extract_node(state: dict[str, Any]) -> dict[str, Any]:
    """Run extractors on discovered source files.

    This node uses build_system_analysis to determine the primary extraction,
    then supplements with any other discovered sources.

    Args:
        state: Current graph state containing 'discovered_sources', 'codebase_path',
               and 'build_system_analysis'.

    Returns:
        Updated state with 'extractions' field containing list of ExtractionResult dicts.
    """
    obs = get_observability()
    trace = obs.create_trace(
        name="extract_node",
        input={
            "codebase_path": state.get("codebase_path"),
            "discovered_sources": state.get("discovered_sources"),
            "build_system_analysis": state.get("build_system_analysis"),
        },
    )

    discovered_sources = state.get("discovered_sources", {})
    build_analysis = state.get("build_system_analysis", {})
    codebase_path = state.get("codebase_path", "")
    extractions: list[dict[str, Any]] = []
    extracted_files: set[str] = set()  # Track files we've already extracted

    # Step 1: Use build_system_analysis if available
    with obs.span("extract_from_analysis", trace=trace) as span:
        mechanism = build_analysis.get("mechanism")
        config_path = build_analysis.get("config_path")

        if mechanism and mechanism != "unknown" and config_path:
            extractor_name = MECHANISM_TO_EXTRACTOR.get(mechanism)
            if extractor_name:
                extractor = get_extractor(extractor_name)
                if extractor and os.path.exists(config_path):
                    try:
                        # Verify extractor can handle this file
                        with open(config_path) as f:
                            content = f.read()
                        if extractor.can_extract(config_path, content):
                            result = extractor.extract(config_path, codebase_path)
                            extractions.append(result.to_dict())
                            extracted_files.add(config_path)
                            logger.info(
                                f"Extracted from build analysis: mechanism={mechanism}, "
                                f"config={config_path}, confidence={result.confidence}"
                            )
                            span.end(output={
                                "mechanism": mechanism,
                                "config_path": config_path,
                                "success": True,
                            })
                        else:
                            logger.warning(
                                f"Extractor {extractor_name} cannot handle {config_path}"
                            )
                            span.end(output={"success": False, "reason": "extractor_mismatch"})
                    except Exception as e:
                        logger.error(f"Extraction from analysis failed: {e}")
                        span.end(level="ERROR", status_message=str(e))
                else:
                    logger.warning(
                        f"No extractor for mechanism={mechanism} or config not found: {config_path}"
                    )
                    span.end(output={"success": False, "reason": "no_extractor_or_file"})
            else:
                span.end(output={"success": False, "reason": "unknown_mechanism"})
        else:
            span.end(output={"success": False, "reason": "no_analysis"})

    # Step 2: Process discovered sources (skip files already extracted)
    with obs.span("run_extractors", trace=trace) as span:
        for source_type, file_path in discovered_sources.items():
            # Skip if we already extracted this file via build analysis
            if file_path in extracted_files:
                logger.debug(f"Skipping {file_path} - already extracted via build analysis")
                continue

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
                    extracted_files.add(file_path)
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
