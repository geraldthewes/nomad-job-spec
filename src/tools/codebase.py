"""Codebase analysis tools for extracting deployment-relevant information."""

import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import git

if TYPE_CHECKING:
    from src.observability import _SpanWrapper, _NoOpSpan

logger = logging.getLogger(__name__)


@dataclass
class DockerfileInfo:
    """Information extracted from a Dockerfile."""

    base_image: str | None = None
    exposed_ports: list[int] = field(default_factory=list)
    cmd: str | None = None
    entrypoint: str | None = None
    env_vars: dict[str, str] = field(default_factory=dict)
    workdir: str | None = None
    healthcheck: str | None = None


@dataclass
class DependencyInfo:
    """Information about project dependencies."""

    language: str | None = None
    package_manager: str | None = None
    dependencies: list[str] = field(default_factory=list)
    dev_dependencies: list[str] = field(default_factory=list)


@dataclass
class CodebaseAnalysis:
    """Complete analysis of a codebase for Nomad deployment."""

    path: str
    dockerfile: DockerfileInfo | None = None
    dockerfiles_found: list[str] = field(default_factory=list)
    dependencies: DependencyInfo | None = None
    env_vars_required: list[str] = field(default_factory=list)
    suggested_resources: dict[str, int] = field(default_factory=dict)
    files_analyzed: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "path": self.path,
            "dockerfile": {
                "base_image": self.dockerfile.base_image,
                "exposed_ports": self.dockerfile.exposed_ports,
                "cmd": self.dockerfile.cmd,
                "entrypoint": self.dockerfile.entrypoint,
                "env_vars": self.dockerfile.env_vars,
                "workdir": self.dockerfile.workdir,
                "healthcheck": self.dockerfile.healthcheck,
            }
            if self.dockerfile
            else None,
            "dependencies": {
                "language": self.dependencies.language,
                "package_manager": self.dependencies.package_manager,
                "dependencies": self.dependencies.dependencies,
                "dev_dependencies": self.dependencies.dev_dependencies,
            }
            if self.dependencies
            else None,
            "env_vars_required": self.env_vars_required,
            "suggested_resources": self.suggested_resources,
            "files_analyzed": self.files_analyzed,
            "dockerfiles_found": self.dockerfiles_found,
            "errors": self.errors,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


def clone_repository(repo_url: str, target_dir: str | None = None) -> str:
    """Clone a git repository to a local directory.

    Args:
        repo_url: Git repository URL (HTTPS or SSH).
        target_dir: Target directory. If None, creates a temp directory.

    Returns:
        Path to the cloned repository.

    Raises:
        git.GitCommandError: If clone fails.
    """
    if target_dir is None:
        target_dir = tempfile.mkdtemp(prefix="nomad-agent-")

    git.Repo.clone_from(repo_url, target_dir)
    return target_dir


def parse_dockerfile(dockerfile_path: str) -> DockerfileInfo:
    """Parse a Dockerfile to extract deployment-relevant information.

    Args:
        dockerfile_path: Path to the Dockerfile.

    Returns:
        DockerfileInfo with extracted information.
    """
    info = DockerfileInfo()

    with open(dockerfile_path) as f:
        content = f.read()

    # Parse FROM instruction (base image)
    from_match = re.search(r"^FROM\s+(\S+)", content, re.MULTILINE | re.IGNORECASE)
    if from_match:
        info.base_image = from_match.group(1)

    # Parse EXPOSE instructions
    expose_matches = re.findall(r"^EXPOSE\s+(\d+)", content, re.MULTILINE | re.IGNORECASE)
    info.exposed_ports = [int(port) for port in expose_matches]

    # Parse CMD instruction
    cmd_match = re.search(
        r'^CMD\s+(\[.*?\]|".*?"|\'.*?\'|\S+.*?)$', content, re.MULTILINE | re.IGNORECASE
    )
    if cmd_match:
        info.cmd = cmd_match.group(1).strip()

    # Parse ENTRYPOINT instruction
    entrypoint_match = re.search(
        r'^ENTRYPOINT\s+(\[.*?\]|".*?"|\'.*?\'|\S+.*?)$',
        content,
        re.MULTILINE | re.IGNORECASE,
    )
    if entrypoint_match:
        info.entrypoint = entrypoint_match.group(1).strip()

    # Parse ENV instructions
    env_matches = re.findall(
        r"^ENV\s+(\w+)(?:=|\s+)(.+?)$", content, re.MULTILINE | re.IGNORECASE
    )
    for key, value in env_matches:
        info.env_vars[key] = value.strip().strip('"').strip("'")

    # Parse WORKDIR instruction
    workdir_match = re.search(r"^WORKDIR\s+(\S+)", content, re.MULTILINE | re.IGNORECASE)
    if workdir_match:
        info.workdir = workdir_match.group(1)

    # Parse HEALTHCHECK instruction
    healthcheck_match = re.search(
        r"^HEALTHCHECK\s+(.+?)$", content, re.MULTILINE | re.IGNORECASE
    )
    if healthcheck_match:
        info.healthcheck = healthcheck_match.group(1).strip()

    return info


def detect_language_and_deps(codebase_path: str) -> DependencyInfo:
    """Detect programming language and parse dependencies.

    Args:
        codebase_path: Path to the codebase root.

    Returns:
        DependencyInfo with detected language and dependencies.
    """
    info = DependencyInfo()
    path = Path(codebase_path)

    # Python - requirements.txt or pyproject.toml
    requirements_txt = path / "requirements.txt"
    pyproject_toml = path / "pyproject.toml"

    if requirements_txt.exists():
        info.language = "python"
        info.package_manager = "pip"
        with open(requirements_txt) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    # Extract package name without version specifier
                    pkg = re.split(r"[<>=~!]", line)[0].strip()
                    if pkg:
                        info.dependencies.append(pkg)

    elif pyproject_toml.exists():
        info.language = "python"
        info.package_manager = "poetry/pip"
        # Basic parsing - could use tomllib for full parsing
        with open(pyproject_toml) as f:
            content = f.read()
            # Look for dependencies section
            deps_match = re.search(
                r'\[project\.dependencies\](.*?)(?=\[|\Z)',
                content,
                re.DOTALL
            )
            if deps_match:
                for line in deps_match.group(1).split("\n"):
                    line = line.strip().strip('"').strip("'")
                    if line and not line.startswith("#"):
                        pkg = re.split(r"[<>=~!]", line)[0].strip()
                        if pkg:
                            info.dependencies.append(pkg)

    # Node.js - package.json
    package_json = path / "package.json"
    if package_json.exists():
        info.language = "nodejs"
        info.package_manager = "npm"
        with open(package_json) as f:
            data = json.load(f)
            if "dependencies" in data:
                info.dependencies = list(data["dependencies"].keys())
            if "devDependencies" in data:
                info.dev_dependencies = list(data["devDependencies"].keys())

    # Go - go.mod
    go_mod = path / "go.mod"
    if go_mod.exists():
        info.language = "go"
        info.package_manager = "go modules"
        with open(go_mod) as f:
            for line in f:
                if line.strip().startswith("require"):
                    continue
                match = re.match(r'\s*(\S+)\s+v', line)
                if match:
                    info.dependencies.append(match.group(1))

    # Rust - Cargo.toml
    cargo_toml = path / "Cargo.toml"
    if cargo_toml.exists():
        info.language = "rust"
        info.package_manager = "cargo"
        with open(cargo_toml) as f:
            in_deps = False
            for line in f:
                if "[dependencies]" in line:
                    in_deps = True
                    continue
                if line.startswith("[") and in_deps:
                    in_deps = False
                if in_deps:
                    match = re.match(r'(\w+)\s*=', line)
                    if match:
                        info.dependencies.append(match.group(1))

    return info


def detect_env_vars(codebase_path: str) -> list[str]:
    """Detect environment variables used in the codebase.

    Scans for common patterns like os.getenv(), process.env, etc.

    Args:
        codebase_path: Path to the codebase root.

    Returns:
        List of environment variable names detected.
    """
    env_vars: set[str] = set()
    path = Path(codebase_path)

    # File patterns to scan
    patterns = ["**/*.py", "**/*.js", "**/*.ts", "**/*.go", "**/*.rs", "**/*.env.example"]

    # Regex patterns for env var detection
    env_patterns = [
        r'os\.(?:getenv|environ\.get)\s*\(\s*["\'](\w+)["\']',  # Python
        r'process\.env\.(\w+)',  # Node.js
        r'os\.Getenv\s*\(\s*["\'](\w+)["\']',  # Go
        r'std::env::var\s*\(\s*["\'](\w+)["\']',  # Rust
        r'^(\w+)=',  # .env files
    ]

    for pattern in patterns:
        for file_path in path.glob(pattern):
            if file_path.is_file():
                try:
                    with open(file_path, errors="ignore") as f:
                        content = f.read()
                        for env_pattern in env_patterns:
                            matches = re.findall(env_pattern, content, re.MULTILINE)
                            env_vars.update(matches)
                except (OSError, UnicodeDecodeError):
                    continue

    # Filter out common non-essential vars
    exclude = {"PATH", "HOME", "USER", "PWD", "SHELL", "TERM", "LANG"}
    return sorted(env_vars - exclude)


def suggest_resources(dockerfile_info: DockerfileInfo | None, deps: DependencyInfo | None) -> dict[str, int]:
    """Suggest resource allocations based on analysis.

    Args:
        dockerfile_info: Parsed Dockerfile information.
        deps: Dependency information.

    Returns:
        Dictionary with cpu (MHz) and memory (MB) suggestions.
    """
    # Default conservative values
    cpu = 500  # MHz
    memory = 256  # MB

    if deps:
        # Language-based adjustments
        if deps.language == "nodejs":
            memory = 512  # Node.js tends to use more memory
        elif deps.language == "go":
            cpu = 300  # Go is efficient
            memory = 128
        elif deps.language == "python":
            memory = 384
        elif deps.language == "rust":
            cpu = 300
            memory = 128

        # Dependency count adjustments
        dep_count = len(deps.dependencies)
        if dep_count > 20:
            memory = max(memory, 512)
        if dep_count > 50:
            memory = max(memory, 768)

    if dockerfile_info:
        # Base image adjustments
        if dockerfile_info.base_image:
            base = dockerfile_info.base_image.lower()
            if "alpine" in base:
                memory = max(128, memory - 128)  # Alpine is smaller
            elif "node" in base:
                memory = max(memory, 512)
            elif "python" in base:
                memory = max(memory, 384)

    return {"cpu": cpu, "memory": memory}


def analyze_codebase(
    codebase_path: str,
    selected_dockerfile: str | None = None,
    parent_span: "(_SpanWrapper | _NoOpSpan | None)" = None,
) -> CodebaseAnalysis:
    """Perform complete analysis of a codebase for Nomad deployment.

    Args:
        codebase_path: Path to the codebase (local or will be cloned if URL).
        selected_dockerfile: Specific Dockerfile to parse. If None, uses first found.
        parent_span: Optional parent span from the calling node. When provided,
            tool operations are recorded as child spans under it.

    Returns:
        CodebaseAnalysis with all extracted information.
    """
    from src.observability import get_observability, _NoOpSpan

    obs = get_observability()

    # Helper to create spans under parent if available
    def create_span(name: str, **kwargs):
        if parent_span is not None:
            return obs.span(name, parent=parent_span, **kwargs)
        else:
            # No parent - return a no-op context manager
            from contextlib import contextmanager

            @contextmanager
            def noop():
                yield _NoOpSpan()

            return noop()

    # Handle git URLs
    if codebase_path.startswith(("http://", "https://", "git@")):
        with create_span("clone_repository", input={"url": codebase_path}) as span:
            codebase_path = clone_repository(codebase_path)
            span.end(output={"cloned_path": codebase_path})

    path = Path(codebase_path)
    if not path.exists():
        raise FileNotFoundError(f"Codebase path does not exist: {codebase_path}")

    analysis = CodebaseAnalysis(path=str(path.absolute()))
    files_analyzed = []

    # Determine which Dockerfile to parse
    # If selected_dockerfile is provided (from discover node), use it directly
    # Otherwise, search for Dockerfiles
    dockerfile_to_parse = None

    if selected_dockerfile:
        # Trust the selection from discover node - don't re-search
        dockerfile_to_parse = selected_dockerfile
        analysis.dockerfiles_found = [selected_dockerfile]
    else:
        # No pre-selection, find all Dockerfiles
        with create_span("find_dockerfiles", input={"path": str(path)}) as span:
            dockerfiles_found = []
            for dockerfile_path in path.glob("**/Dockerfile*"):
                # Skip directories, backup files, and documentation
                if dockerfile_path.is_dir():
                    continue
                name = dockerfile_path.name.lower()
                if name.endswith((".md", ".txt", ".bak", ".orig", ".swp", "~")):
                    continue
                # Skip files in common non-source directories
                rel_path = str(dockerfile_path.relative_to(path))
                if any(part in rel_path.split("/") for part in ["node_modules", ".git", "vendor", "__pycache__"]):
                    continue
                dockerfiles_found.append(rel_path)

            # Sort: prefer root Dockerfile first, then alphabetically
            def dockerfile_sort_key(p: str) -> tuple:
                depth = p.count("/")
                is_plain_dockerfile = p.lower() in ("dockerfile", "dockerfile")
                return (depth, not is_plain_dockerfile, p.lower())

            dockerfiles_found.sort(key=dockerfile_sort_key)
            analysis.dockerfiles_found = dockerfiles_found
            span.end(output={"dockerfiles_found": dockerfiles_found, "count": len(dockerfiles_found)})

            if dockerfiles_found:
                dockerfile_to_parse = dockerfiles_found[0]

    # Parse the Dockerfile
    if dockerfile_to_parse:
        primary_dockerfile = path / dockerfile_to_parse
        with create_span("parse_dockerfile", input={"dockerfile": dockerfile_to_parse}) as span:
            try:
                analysis.dockerfile = parse_dockerfile(str(primary_dockerfile))
                files_analyzed.append(dockerfile_to_parse)
                span.end(output={
                    "base_image": analysis.dockerfile.base_image,
                    "exposed_ports": analysis.dockerfile.exposed_ports,
                    "cmd": analysis.dockerfile.cmd,
                })
            except Exception as e:
                analysis.errors.append(f"Error parsing Dockerfile: {e}")
                span.end(level="ERROR", status_message=str(e))

    # Detect language and dependencies
    with create_span("detect_language_and_deps", input={"path": str(path)}) as span:
        try:
            analysis.dependencies = detect_language_and_deps(str(path))
            # Track which files were analyzed
            for dep_file in ["requirements.txt", "pyproject.toml", "package.json", "go.mod", "Cargo.toml"]:
                if (path / dep_file).exists():
                    files_analyzed.append(dep_file)
            span.end(output={
                "language": analysis.dependencies.language if analysis.dependencies else None,
                "package_manager": analysis.dependencies.package_manager if analysis.dependencies else None,
                "dep_count": len(analysis.dependencies.dependencies) if analysis.dependencies else 0,
            })
        except Exception as e:
            analysis.errors.append(f"Error detecting dependencies: {e}")
            span.end(level="ERROR", status_message=str(e))

    # Detect environment variables
    with create_span("detect_env_vars", input={"path": str(path)}) as span:
        try:
            analysis.env_vars_required = detect_env_vars(str(path))
            span.end(output={"env_vars": analysis.env_vars_required, "count": len(analysis.env_vars_required)})
        except Exception as e:
            analysis.errors.append(f"Error detecting env vars: {e}")
            span.end(level="ERROR", status_message=str(e))

    # Suggest resources
    with create_span("suggest_resources") as span:
        analysis.suggested_resources = suggest_resources(analysis.dockerfile, analysis.dependencies)
        span.end(output=analysis.suggested_resources)

    analysis.files_analyzed = files_analyzed
    return analysis


def get_relevant_files_content(codebase_path: str, max_size: int = 50000) -> dict[str, str]:
    """Get content of relevant files for LLM analysis.

    Args:
        codebase_path: Path to the codebase.
        max_size: Maximum total size of content to return.

    Returns:
        Dictionary mapping file paths to their content.
    """
    path = Path(codebase_path)
    relevant_files = [
        "Dockerfile",
        "docker-compose.yml",
        "docker-compose.yaml",
        "package.json",
        "requirements.txt",
        "pyproject.toml",
        "go.mod",
        "Cargo.toml",
        ".env.example",
        "README.md",
    ]

    contents: dict[str, str] = {}
    total_size = 0

    for filename in relevant_files:
        file_path = path / filename
        if file_path.exists() and file_path.is_file():
            try:
                content = file_path.read_text(errors="ignore")
                if total_size + len(content) <= max_size:
                    contents[filename] = content
                    total_size += len(content)
            except Exception:
                continue

    return contents
