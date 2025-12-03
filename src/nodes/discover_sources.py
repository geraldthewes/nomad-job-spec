"""Discover sources node for the LangGraph workflow.

This node scans the codebase for known source files that can provide
deployment configuration (build.yaml, Makefile, docker-compose.yml, etc.).

This replaces and extends the original discover.py node by:
1. Handling git URL cloning (moved from discover.py)
2. Finding all source files (build.yaml, Makefile, etc.)
3. Finding all Dockerfiles (original discover.py behavior)
"""

import logging
import os
from pathlib import Path
from typing import Any

from src.observability import get_observability
from src.tools.extractors import get_all_extractors

logger = logging.getLogger(__name__)

# Known source files to look for, in priority order
# The first file found for each category is preferred
SOURCE_FILE_PATTERNS = {
    "jobforge": ["build.yaml", "build.yml"],
    "makefile": ["Makefile", "makefile", "GNUmakefile"],
    "docker_compose": [
        "docker-compose.yml",
        "docker-compose.yaml",
        "compose.yml",
        "compose.yaml",
    ],
}


def discover_sources_node(state: dict[str, Any]) -> dict[str, Any]:
    """Discover all source files that can provide deployment configuration.

    This node:
    1. Clones git repos if URL provided (moved from discover.py)
    2. Scans for known source patterns (build.yaml, Makefile, etc.)
    3. Finds all Dockerfiles (original discover.py behavior)

    Args:
        state: Current graph state containing 'codebase_path'.

    Returns:
        Updated state with 'discovered_sources' and 'dockerfiles_found' fields.
    """
    obs = get_observability()
    trace = obs.create_trace(
        name="discover_sources_node",
        input={"codebase_path": state.get("codebase_path")},
    )

    codebase_path = state.get("codebase_path")
    if not codebase_path:
        if trace:
            trace.update(level="ERROR", status_message="No codebase path provided")
        return {
            **state,
            "discovered_sources": {},
            "dockerfiles_found": [],
            "selected_dockerfile": None,
        }

    # Handle git URLs - clone first if needed
    if codebase_path.startswith(("http://", "https://", "git@")):
        from src.tools.codebase import clone_repository

        with obs.span("clone_repository", trace=trace, input={"url": codebase_path}) as span:
            codebase_path = clone_repository(codebase_path)
            span.end(output={"cloned_path": codebase_path})
            # Update state with cloned path
            state = {**state, "codebase_path": codebase_path}

    codebase = Path(codebase_path)

    if not codebase.exists():
        if trace:
            trace.end(level="ERROR", status_message=f"Path does not exist: {codebase_path}")
        return {
            **state,
            "discovered_sources": {},
            "dockerfiles_found": [],
            "selected_dockerfile": None,
        }

    discovered = {}

    with obs.span("scan_sources", trace=trace) as span:
        # Scan for each category of source file
        for source_type, patterns in SOURCE_FILE_PATTERNS.items():
            for pattern in patterns:
                # Check in root directory first
                file_path = codebase / pattern
                if file_path.exists() and file_path.is_file():
                    discovered[source_type] = str(file_path)
                    logger.info(f"Discovered {source_type}: {file_path}")
                    break  # Use first match for this category

        # Also discover additional files using extractor patterns
        for extractor in get_all_extractors():
            if extractor.name in discovered:
                continue  # Already found this type

            for pattern in extractor.file_patterns:
                file_path = codebase / pattern
                if file_path.exists() and file_path.is_file():
                    # Verify extractor can handle it
                    try:
                        with open(file_path) as f:
                            content = f.read()
                        if extractor.can_extract(str(file_path), content):
                            discovered[extractor.name] = str(file_path)
                            logger.info(
                                f"Discovered {extractor.name}: {file_path} "
                                f"(via extractor pattern)"
                            )
                            break
                    except Exception as e:
                        logger.warning(f"Could not read {file_path}: {e}")

        span.end(output={"discovered_count": len(discovered), "sources": list(discovered.keys())})

    # Find all Dockerfiles (original discover.py behavior)
    with obs.span("find_dockerfiles", trace=trace, input={"path": str(codebase)}) as span:
        dockerfiles_found = []
        for dockerfile_path in codebase.glob("**/Dockerfile*"):
            # Skip directories, backup files, and documentation
            if dockerfile_path.is_dir():
                continue
            name = dockerfile_path.name.lower()
            if name.endswith((".md", ".txt", ".bak", ".orig", ".swp", "~")):
                continue
            # Skip files in common non-source directories
            rel_path = str(dockerfile_path.relative_to(codebase))
            skip_dirs = ["node_modules", ".git", "vendor", "__pycache__"]
            if any(part in rel_path.split("/") for part in skip_dirs):
                continue
            dockerfiles_found.append(rel_path)

        # Sort: prefer root Dockerfile first, then alphabetically
        def dockerfile_sort_key(p: str) -> tuple:
            depth = p.count("/")
            is_plain_dockerfile = p.lower() in ("dockerfile", "dockerfile")
            return (depth, not is_plain_dockerfile, p.lower())

        dockerfiles_found.sort(key=dockerfile_sort_key)
        span.end(output={"dockerfiles_found": dockerfiles_found, "count": len(dockerfiles_found)})

    # Log discovery results
    if len(dockerfiles_found) == 0:
        logger.warning("No Dockerfiles found in codebase")
    elif len(dockerfiles_found) == 1:
        logger.info(f"Found 1 Dockerfile: {dockerfiles_found[0]}")
    else:
        logger.info(f"Found {len(dockerfiles_found)} Dockerfiles - user selection required")

    if discovered:
        logger.info(f"Discovered sources: {list(discovered.keys())}")

    if trace:
        trace.end(
            output={
                "discovered_sources": discovered,
                "dockerfiles_found": dockerfiles_found,
            }
        )

    return {
        **state,
        "discovered_sources": discovered,
        "dockerfiles_found": dockerfiles_found,
        "selected_dockerfile": None,
    }


def create_discover_sources_node():
    """Create the discover sources node function.

    Returns:
        Node function for use in LangGraph.
    """
    return discover_sources_node
