"""CLI entry point for the Nomad Job Spec Agent."""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.syntax import Syntax
from rich.table import Table

from config.settings import get_settings
from src.llm.provider import get_llm
from src.graph import compile_graph, create_initial_state
from src.observability import get_observability
from src.tools.infra_status import InfraHealthReport, check_infrastructure_from_settings

app = typer.Typer(
    name="nomad-spec",
    help="AI-powered Nomad job specification generator",
    add_completion=False,
)
console = Console()


@app.command()
def generate(
    prompt: Optional[str] = typer.Option(
        None,
        "--prompt", "-p",
        help="Deployment request. If not provided, will be asked interactively after analysis.",
    ),
    path: str = typer.Option(
        ...,
        "--path",
        help="Path to codebase (local directory or git URL)",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output", "-o",
        help="Output file for generated HCL (default: stdout)",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Analyze only, don't deploy",
    ),
    no_questions: bool = typer.Option(
        False,
        "--no-questions",
        help="Skip interactive questions, use defaults",
    ),
    cluster_id: str = typer.Option(
        "default",
        "--cluster",
        help="Cluster identifier for memory context",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Show detailed output",
    ),
    skip_infra_check: bool = typer.Option(
        False,
        "--skip-infra-check",
        help="Skip infrastructure connectivity checks and use defaults",
    ),
):
    """Generate a Nomad job specification from a codebase.

    Examples:

        nomad-spec generate --path ./my-app

        nomad-spec generate -p "Deploy nginx" --path . -o job.nomad

        nomad-spec generate --path https://github.com/user/repo
    """
    settings = get_settings()

    # Initialize observability
    obs = get_observability(settings)

    # Generate session ID for trace grouping (UUID for easy Langfuse matching)
    import uuid
    session_id = str(uuid.uuid4())

    if obs.is_enabled():
        console.print(f"[dim]LangFuse tracing enabled | Session: {session_id}[/dim]")

    # Validate path
    codebase_path = Path(path)
    is_git_url = path.startswith(("http://", "https://", "git@"))

    if not is_git_url and not codebase_path.exists():
        console.print(f"[red]Error:[/red] Path does not exist: {path}")
        raise typer.Exit(code=1)

    # Handle --no-questions without prompt: use default
    if no_questions and not prompt:
        prompt = "Deploy this application"

    # Check infrastructure health (unless skipped)
    if not skip_infra_check:
        with console.status("[bold green]Checking infrastructure..."):
            health_report = check_infrastructure_from_settings(settings)

        # Always display infrastructure status
        _display_infra_status(health_report)

        if not health_report.all_healthy:
            failures = health_report.get_failures()

            if not no_questions:
                if not Confirm.ask(
                    f"\n[yellow]{len(failures)} service(s) unavailable.[/yellow] Continue with available services?"
                ):
                    console.print("[dim]Aborted by user[/dim]")
                    raise typer.Exit(code=1)
                console.print()

    # Show initial configuration if prompt was provided
    if prompt:
        console.print(Panel(
            f"[bold]Nomad Job Spec Generator[/bold]\n\n"
            f"Prompt: {prompt}\n"
            f"Codebase: {path}\n"
            f"Datacenter: {settings.nomad_datacenter}  Namespace: {cluster_id}",
            title="Configuration",
        ))

    # Initialize LLM
    with console.status("[bold green]Initializing LLM..."):
        try:
            llm = get_llm()
        except Exception as e:
            console.print(f"[red]Error initializing LLM:[/red] {e}")
            raise typer.Exit(code=1)

    # Compile graph
    graph = compile_graph(
        llm=llm,
        settings=settings,
        include_deployment=not dry_run,
        enable_checkpointing=not no_questions,
        session_id=session_id,
    )

    # Create initial state (prompt may be empty, will be collected interactively)
    state = create_initial_state(
        codebase_path=str(codebase_path.absolute()) if not is_git_url else path,
        prompt=prompt or "",
        cluster_id=cluster_id,
        max_iterations=settings.max_iterations,
    )

    # Run graph with HitL for questions
    config = {"configurable": {"thread_id": f"session-{cluster_id}"}}

    # Use ExitStack to manage Langfuse tracing context
    from contextlib import ExitStack

    try:
        with ExitStack() as stack:
            # Set up Langfuse attribute propagation if enabled
            # This propagates session_id and tags to all traces created within this context
            if obs.is_enabled():
                try:
                    from langfuse import propagate_attributes

                    # Propagate session_id and tags to all traces (each node gets its own trace)
                    stack.enter_context(propagate_attributes(
                        session_id=session_id,
                        tags=["nomad-job-spec"],
                    ))
                except Exception as e:
                    logger.warning(f"Failed to set up Langfuse attribute propagation: {e}")
            # First run - discover sources and analyze build system
            with console.status("[bold green]Analyzing build system..."):
                for event in graph.stream(state, config):
                    if verbose:
                        _print_event(event)

                    # Check for interrupt (Dockerfile confirmation or questions ready)
                    if "__interrupt__" in event:
                        break

            # Get current state to check if Dockerfile confirmation is needed
            current_state = graph.get_state(config)
            dockerfiles = current_state.values.get("dockerfiles_found", [])
            selected = current_state.values.get("selected_dockerfile")
            build_analysis = current_state.values.get("build_system_analysis", {})

            # Handle Dockerfile confirmation/selection if needed
            if not selected and not no_questions:
                selected = _collect_dockerfile_confirmation(current_state.values)
                if selected:
                    # Merge with full current state to preserve all values
                    updated_state = {**current_state.values, "selected_dockerfile": selected}
                    graph.update_state(config, updated_state)
            elif not selected and no_questions:
                # Auto-accept in no-questions mode
                selected = _auto_select_dockerfile(current_state.values)
                if selected:
                    updated_state = {**current_state.values, "selected_dockerfile": selected}
                    graph.update_state(config, updated_state)

            # Continue to analysis (whether selection was made or skipped)
            with console.status("[bold green]Analyzing codebase..."):
                for event in graph.stream(None, config):
                    if verbose:
                        _print_event(event)

                    # Check for interrupt (questions ready)
                    if "__interrupt__" in event:
                        break

            # Refresh current state after analysis
            current_state = graph.get_state(config)

            analysis = current_state.values.get("codebase_analysis", {})
            selected_dockerfile = current_state.values.get("selected_dockerfile")

            # If no prompt was provided, auto-generate based on selected Dockerfile
            if not prompt:
                prompt = f"Deploy using {selected_dockerfile}" if selected_dockerfile else "Deploy this application"
                graph.update_state(config, {"prompt": prompt})

            # Display configuration summary after enrichment
            _display_configuration_summary(current_state.values)

            if not no_questions and current_state.values.get("questions"):
                # Display questions and collect responses
                responses = _collect_user_responses(current_state.values)

                # Update state with responses and any confirmed vault paths
                state_updates = {"user_responses": responses}

                # Check for structured vault path responses and update vault_suggestions
                vault_updates = _extract_vault_updates(responses)
                if vault_updates:
                    state_updates["vault_suggestions"] = {"suggestions": vault_updates}

                graph.update_state(config, state_updates)

                # Continue execution
                with console.status("[bold green]Generating job specification..."):
                    for event in graph.stream(None, config):
                        if verbose:
                            _print_event(event)

            # Get final state
            final_state = graph.get_state(config)
            result = final_state.values

    except Exception as e:
        console.print(f"[red]Error during execution:[/red] {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(code=1)
    finally:
        # Flush any pending traces
        obs.flush()

    # Display results
    _display_results(result, output, verbose)


@app.command()
def analyze(
    path: str = typer.Option(
        ...,
        "--path",
        help="Path to codebase",
    ),
    output_format: str = typer.Option(
        "table",
        "--format", "-f",
        help="Output format: table, json",
    ),
):
    """Analyze a codebase without generating a job spec.

    Examples:

        nomad-spec analyze --path ./my-app

        nomad-spec analyze --path . --format json
    """
    from src.tools.codebase import analyze_codebase

    codebase_path = Path(path)
    if not codebase_path.exists():
        console.print(f"[red]Error:[/red] Path does not exist: {path}")
        raise typer.Exit(code=1)

    with console.status("[bold green]Analyzing codebase..."):
        try:
            analysis = analyze_codebase(str(codebase_path.absolute()))
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(code=1)

    if output_format == "json":
        console.print(analysis.to_json())
    else:
        _display_analysis_table(analysis)


@app.command()
def validate(
    file: str = typer.Argument(
        ...,
        help="HCL file to validate",
    ),
):
    """Validate a Nomad job specification file.

    Examples:

        nomad-spec validate job.nomad
    """
    from src.tools.hcl import validate_hcl

    file_path = Path(file)
    if not file_path.exists():
        console.print(f"[red]Error:[/red] File does not exist: {file}")
        raise typer.Exit(code=1)

    hcl_content = file_path.read_text()
    settings = get_settings()

    with console.status("[bold green]Validating..."):
        is_valid, error = validate_hcl(hcl_content, settings.nomad_addr)

    if is_valid:
        console.print("[green]Valid[/green] Job specification is valid")
    else:
        console.print(f"[red]Invalid[/red] {error}")
        raise typer.Exit(code=1)


def _collect_user_responses(state: dict) -> dict[str, Any]:
    """Collect user responses to generated questions.

    Handles both plain string questions and structured questions
    (like env_configs or vault_paths which use interactive step-by-step flow).
    """
    questions = state.get("questions", [])
    responses = {}

    # Get app_name for context in prompts
    app_name = state.get("app_name", "app")

    console.print("\n[bold]Please answer the following questions:[/bold]\n")

    for i, question in enumerate(questions, 1):
        if isinstance(question, dict):
            q_type = question.get("type")
            if q_type == "env_configs":
                # Route to interactive env config handler (new format)
                env_configs = _collect_env_config_responses(
                    question["configs"],
                    app_name,
                )
                responses[f"q{i}"] = {"type": "env_configs", "configs": env_configs}
            elif q_type == "vault_paths":
                # Route to legacy Vault path handler for backward compatibility
                vault_responses = _collect_vault_path_responses(question["suggestions"])
                responses[f"q{i}"] = vault_responses
            else:
                # Unknown structured question type, ask as string
                response = Prompt.ask(f"[cyan]{i}.[/cyan] {question}")
                responses[f"q{i}"] = response
        else:
            # Original behavior for plain string questions
            response = Prompt.ask(f"[cyan]{i}.[/cyan] {question}")
            responses[f"q{i}"] = response

    return responses


def _collect_env_config_responses(
    configs: list[dict],
    app_name: str,
) -> list[dict]:
    """Interactive step-by-step environment variable configuration.

    Flow:
    1. Show summary table of all suggestions
    2. Ask confirm/edit
    3. If edit: step through each variable one by one, asking for source and value
    4. Show final summary and ask confirm/edit again
    5. Return final configurations

    Args:
        configs: List of env config dicts with keys:
            - name: Environment variable name
            - source: Source type ("fixed", "consul", "vault")
            - value: Fixed value, Consul path, or Vault path
            - confidence: Confidence score (0.0-1.0)
        app_name: Application name for display hints.

    Returns:
        List of confirmed env config dicts.
    """
    # Work with a copy to avoid mutating the original
    current_configs = [dict(c) for c in configs]

    while True:
        # Display summary table
        _display_env_config_table(current_configs)

        # Ask confirm/edit
        choice = Prompt.ask(
            "\nAccept all suggestions?",
            choices=["confirm", "edit"],
            default="confirm"
        )

        if choice == "confirm":
            return current_configs

        # Edit mode: step through each variable
        console.print("\n[dim]For each variable, select a source and provide the value/path.[/dim]\n")

        for idx, cfg in enumerate(current_configs, 1):
            var_name = cfg["name"]
            console.print(f"[cyan][{idx}/{len(current_configs)}][/cyan] [bold]{var_name}[/bold]")

            # Ask for source type
            new_source = Prompt.ask(
                "  Source",
                choices=["fixed", "consul", "vault"],
                default=cfg["source"]
            )

            # Provide context-appropriate hint for value
            if new_source == "fixed":
                hint = "Value"
            elif new_source == "consul":
                hint = f"Consul KV path (e.g., {app_name}/config/...)"
            else:
                hint = "Vault path (e.g., secret/data/.../key)"

            new_value = Prompt.ask(
                f"  {hint}",
                default=cfg["value"]
            )

            # Update the config
            cfg["source"] = new_source
            cfg["value"] = new_value
            cfg["confidence"] = 1.0  # User confirmed

            console.print()  # Blank line between variables

        # After editing all, show final summary (loop back to top)


def _display_env_config_table(configs: list[dict]):
    """Display environment variable configurations in a formatted table.

    Args:
        configs: List of env config dicts with name, source, value, confidence.
    """
    table = Table(title="Environment Variable Configuration")
    table.add_column("Variable", style="cyan")
    table.add_column("Source", style="magenta", justify="center")
    table.add_column("Value/Path", style="green")
    table.add_column("Confidence", justify="right")

    for cfg in configs:
        var_name = cfg["name"]
        source = cfg["source"]
        value = cfg["value"]
        confidence = cfg.get("confidence", 0)

        # Source styling
        if source == "fixed":
            source_str = "[blue]fixed[/blue]"
        elif source == "consul":
            source_str = "[yellow]consul[/yellow]"
        else:
            source_str = "[magenta]vault[/magenta]"

        # Confidence styling
        if confidence >= 0.9:
            conf_str = "[green]validated[/green]"
        elif confidence >= 0.5:
            conf_str = f"[yellow]{int(confidence * 100)}%[/yellow]"
        else:
            conf_str = f"[red]{int(confidence * 100)}%[/red]"

        table.add_row(var_name, source_str, value, conf_str)

    console.print()
    console.print(table)


def _display_configuration_summary(state: dict):
    """Display comprehensive configuration summary before user confirmation.

    Shows all tracked values including job metadata, Docker config, resources,
    network settings, service configuration, and Vault integration.

    Values are pulled from merged_extraction first (from extractors like jobforge),
    then fall back to codebase_analysis for LLM-derived values.

    Args:
        state: Current graph state with merged_extraction, codebase_analysis, and enrichment data.
    """
    # Primary source: merged extraction from build tools (jobforge, makefile, etc.)
    merged = state.get("merged_extraction", {})
    sources = state.get("extraction_sources", {})

    # Fallback: LLM analysis
    analysis = state.get("codebase_analysis", {})

    # Enrichment data
    fabio = state.get("fabio_validation", {})
    vault_suggestions = state.get("vault_suggestions", {})
    env_var_configs = state.get("env_var_configs", [])

    def get_source(field: str, fallback: str = "analysis") -> str:
        """Get source attribution for a field."""
        if field in sources:
            return sources[field].get("source_type", fallback)
        return fallback

    def get_value(field: str, default=None):
        """Get value from merged extraction or analysis."""
        if field in merged:
            return merged[field]
        return analysis.get(field, default)

    # Main configuration table
    table = Table(title="Configuration Summary")
    table.add_column("Property", style="cyan", width=20)
    table.add_column("Value", style="green")
    table.add_column("Source", style="dim", justify="right")

    # Job metadata
    job_name = get_value("job_name") or state.get("job_name") or analysis.get("project_name") or "unknown"
    job_source = get_source("job_name", "derived")
    table.add_row("Job Name", job_name, job_source)

    # Docker image
    docker_image = get_value("docker_image", "")
    if docker_image:
        # Truncate long image names for display
        display_image = docker_image if len(docker_image) <= 50 else docker_image[:47] + "..."
        table.add_row("Docker Image", display_image, get_source("docker_image", "analysis"))
    else:
        table.add_row("Docker Image", "[dim]not specified[/dim]", "-")

    # Service type and resources
    service_type = get_value("service_type", "MEDIUM")

    # Resources from extraction or analysis
    extracted_resources = merged.get("resources", {})
    analysis_resources = analysis.get("resources", {})
    resources = extracted_resources if extracted_resources else analysis_resources

    if isinstance(resources, dict):
        cpu = resources.get("cpu", "-")
        memory = resources.get("memory", "-")
    else:
        cpu = "-"
        memory = "-"

    resource_source = get_source("resources", service_type)
    table.add_row("Service Type", service_type, get_source("service_type", "analysis"))
    table.add_row("CPU", f"{cpu} MHz" if cpu != "-" else "-", resource_source)
    table.add_row("Memory", f"{memory} MB" if memory != "-" else "-", resource_source)

    # Ports
    ports = get_value("ports", [])
    if ports:
        port_strs = []
        for p in ports:
            if isinstance(p, dict):
                port_strs.append(f"{p.get('name', 'http')}:{p.get('container_port', '?')}")
            else:
                port_strs.append(str(p))
        table.add_row("Ports", ", ".join(port_strs), get_source("ports", "analysis"))
    else:
        table.add_row("Ports", "[dim]none detected[/dim]", "-")

    # Health check
    health_check = get_value("health_check", {})
    if health_check:
        hc_type = health_check.get("type", "http")
        hc_path = health_check.get("path", "/health")
        table.add_row("Health Check", f"{hc_type.upper()} {hc_path}", get_source("health_check", "analysis"))
    else:
        table.add_row("Health Check", "[dim]none[/dim]", "-")

    # Fabio routing
    if fabio:
        hostname = fabio.get("hostname", "")
        path = fabio.get("path", "")
        strip = fabio.get("strip_path", False)
        if hostname:
            route_str = hostname
            if path:
                route_str += path
            if strip:
                route_str += " [dim](strip)[/dim]"
            table.add_row("Fabio Route", route_str, "enrichment")
        elif path:
            route_str = path
            if strip:
                route_str += " [dim](strip)[/dim]"
            table.add_row("Fabio Route", route_str, "enrichment")
    else:
        table.add_row("Fabio Route", "[dim]none[/dim]", "-")

    # Vault integration - check extraction first
    vault_secrets = get_value("vault_secrets", [])
    vault_policies = get_value("vault_policies", [])
    secrets = analysis.get("secrets", [])

    if vault_secrets:
        table.add_row(
            "Vault Secrets",
            f"{len(vault_secrets)} configured",
            get_source("vault_secrets", "extraction")
        )
    elif secrets:
        table.add_row("Vault Secrets", f"{len(secrets)} variables", "analysis")
    elif vault_suggestions:
        table.add_row("Vault Secrets", f"{len(vault_suggestions)} suggested", "enrichment")
    else:
        table.add_row("Vault Secrets", "[dim]none[/dim]", "-")

    # Vault policies (if extracted)
    if vault_policies:
        table.add_row(
            "Vault Policies",
            ", ".join(vault_policies),
            get_source("vault_policies", "extraction")
        )

    # Environment variables summary - combine extraction and enrichment
    env_vars = get_value("env_vars", {})
    if env_var_configs:
        fixed_count = sum(1 for c in env_var_configs if c.get("source") == "fixed")
        consul_count = sum(1 for c in env_var_configs if c.get("source") == "consul")
        vault_count = sum(1 for c in env_var_configs if c.get("source") == "vault")
        parts = []
        if fixed_count:
            parts.append(f"{fixed_count} fixed")
        if consul_count:
            parts.append(f"{consul_count} consul")
        if vault_count:
            parts.append(f"{vault_count} vault")
        table.add_row("Env Variables", ", ".join(parts) if parts else "0", "enrichment")
    elif env_vars:
        source = get_source("env_vars", "analysis")
        table.add_row("Env Variables", f"{len(env_vars)} detected", source)
    else:
        analysis_env = analysis.get("env_vars", {})
        if analysis_env:
            table.add_row("Env Variables", f"{len(analysis_env)} detected", "analysis")
        else:
            table.add_row("Env Variables", "[dim]none[/dim]", "-")

    # GPU requirement (from extraction)
    requires_gpu = get_value("requires_gpu", False)
    if requires_gpu:
        table.add_row("GPU Required", "Yes", get_source("requires_gpu", "extraction"))

    # Architecture requirements
    requires_amd64 = get_value("requires_amd64", False)
    if requires_amd64:
        table.add_row("Architecture", "AMD64 required", get_source("requires_amd64", "analysis"))

    # Storage requirements
    requires_storage = get_value("requires_storage", False)
    if requires_storage:
        storage_path = get_value("storage_path", "/data")
        table.add_row("Storage", f"CSI volume at {storage_path}", get_source("requires_storage", "analysis"))

    console.print()
    console.print(table)


# Keep backward compatibility aliases
def _collect_vault_path_responses(suggestions: list[dict]) -> dict[str, str]:
    """Legacy function for backward compatibility.

    Converts vault-only suggestions to env configs and back.
    """
    # Convert vault suggestions to env config format
    configs = []
    for s in suggestions:
        vault_ref = s.get("vault_reference") or f"{s['suggested_path']}#{s['key']}"
        configs.append({
            "name": s["env_var"],
            "source": "vault",
            "value": vault_ref,
            "confidence": s.get("confidence", 0.5),
        })

    # Use the new function
    app_name = "app"  # Default for legacy calls
    result_configs = _collect_env_config_responses(configs, app_name)

    # Convert back to vault path dict format
    return {c["name"]: c["value"] for c in result_configs}


def _display_vault_suggestions_table(suggestions: list[dict], paths: dict[str, str]):
    """Legacy function for backward compatibility."""
    # Convert to env config format for display
    configs = []
    for s in suggestions:
        env_var = s["env_var"]
        configs.append({
            "name": env_var,
            "source": "vault",
            "value": paths.get(env_var, ""),
            "confidence": s.get("confidence", 0.5),
        })
    _display_env_config_table(configs)


def _extract_vault_updates(responses: dict[str, Any]) -> list[dict] | None:
    """Extract vault path updates from user responses.

    Looks for structured vault path responses (dicts) and converts them
    to the vault_suggestions format for state update.

    Args:
        responses: User responses dict, where some values may be dicts
                   from the interactive vault path flow.

    Returns:
        List of suggestion dicts ready for vault_suggestions state, or None
        if no vault path responses found.
    """
    vault_updates = []

    for key, value in responses.items():
        if isinstance(value, dict):
            # This is a structured vault path response
            for env_var, path in value.items():
                # Parse path#key format
                if "#" in path:
                    path_part, key_part = path.rsplit("#", 1)
                else:
                    path_part = path
                    key_part = ""

                vault_updates.append({
                    "env_var": env_var,
                    "suggested_path": path_part,
                    "key": key_part,
                    "vault_reference": path,
                    "confidence": 1.0,  # User confirmed
                })

    return vault_updates if vault_updates else None


def _collect_dockerfile_confirmation(state: dict) -> str | None:
    """Collect user's Dockerfile confirmation based on build system analysis.

    This is the new flow where the build system analysis identifies which
    Dockerfile is used, and the user confirms (or overrides) that finding.

    Args:
        state: Current graph state with build_system_analysis and dockerfiles_found.

    Returns:
        Confirmed Dockerfile path, or None if no Dockerfiles.
    """
    build_analysis = state.get("build_system_analysis", {})
    dockerfile_identified = build_analysis.get("dockerfile_used")
    dockerfiles_found = state.get("dockerfiles_found", [])
    codebase_path = state.get("codebase_path", "")

    if not dockerfiles_found:
        return None

    # Convert absolute dockerfile_identified to relative for comparison
    dockerfile_rel = None
    if dockerfile_identified:
        try:
            from pathlib import Path
            if Path(dockerfile_identified).is_absolute() and codebase_path:
                dockerfile_rel = str(Path(dockerfile_identified).relative_to(codebase_path))
            else:
                dockerfile_rel = dockerfile_identified
        except ValueError:
            dockerfile_rel = dockerfile_identified

    if dockerfile_rel:
        # Check if identified Dockerfile is in the discovered list
        if dockerfile_rel in dockerfiles_found:
            # Happy path: LLM identified a known Dockerfile
            mechanism = build_analysis.get("mechanism", "unknown")
            console.print(f"\n[bold]Build system analysis:[/bold]")
            console.print(f"  Mechanism: [cyan]{mechanism}[/cyan]")
            console.print(f"  Dockerfile: [cyan]{dockerfile_rel}[/cyan]")

            confirm = Prompt.ask(
                "\n[bold cyan]Use this Dockerfile?[/bold cyan]",
                choices=["y", "n"],
                default="y"
            )
            if confirm.lower() == "y":
                console.print(f"[green]Confirmed:[/green] {dockerfile_rel}\n")
                return dockerfile_rel
            else:
                # User rejected - fall through to manual selection
                console.print("[yellow]Selecting manually...[/yellow]")
        else:
            # Mismatch: LLM identified something not in discovery
            console.print(f"\n[yellow]Warning:[/yellow] Build system references "
                         f"[cyan]{dockerfile_rel}[/cyan] but it wasn't found in the codebase.")
            console.print("Falling back to manual selection.\n")

    # Fall back to manual selection from discovered Dockerfiles
    return _manual_dockerfile_selection(dockerfiles_found)


def _manual_dockerfile_selection(dockerfiles: list[str]) -> str | None:
    """Manually select a Dockerfile from the discovered list.

    Args:
        dockerfiles: List of discovered Dockerfile paths.

    Returns:
        Selected Dockerfile path.
    """
    if len(dockerfiles) == 0:
        return None

    if len(dockerfiles) == 1:
        dockerfile = dockerfiles[0]
        console.print(f"\n[bold]Found Dockerfile:[/bold] [cyan]{dockerfile}[/cyan]")
        confirm = Prompt.ask(
            "[bold cyan]Use this Dockerfile?[/bold cyan]",
            choices=["y", "n"],
            default="y"
        )
        if confirm.lower() == "y":
            console.print(f"[green]Using:[/green] {dockerfile}\n")
            return dockerfile
        else:
            console.print("[yellow]No Dockerfile selected[/yellow]\n")
            return None

    # Multiple Dockerfiles - show selection menu
    console.print(f"\n[bold]Found {len(dockerfiles)} Dockerfiles in the repository:[/bold]")
    for i, df in enumerate(dockerfiles, 1):
        console.print(f"  [cyan][{i}][/cyan] {df}")
    console.print()

    while True:
        choice = Prompt.ask(
            "[bold cyan]Select Dockerfile to deploy[/bold cyan]",
            default="1"
        )
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(dockerfiles):
                selected = dockerfiles[idx]
                console.print(f"[green]Selected:[/green] {selected}\n")
                return selected
            console.print(f"[red]Please enter a number between 1 and {len(dockerfiles)}[/red]")
        except ValueError:
            console.print("[red]Please enter a valid number[/red]")


def _auto_select_dockerfile(state: dict) -> str | None:
    """Auto-select Dockerfile for --no-questions mode.

    Prefers LLM-identified Dockerfile if valid, otherwise uses first discovered.

    Args:
        state: Current graph state.

    Returns:
        Selected Dockerfile path.
    """
    build_analysis = state.get("build_system_analysis", {})
    dockerfile_identified = build_analysis.get("dockerfile_used")
    dockerfiles_found = state.get("dockerfiles_found", [])
    codebase_path = state.get("codebase_path", "")

    if not dockerfiles_found:
        return None

    # Convert absolute dockerfile_identified to relative for comparison
    if dockerfile_identified:
        try:
            from pathlib import Path
            if Path(dockerfile_identified).is_absolute() and codebase_path:
                dockerfile_rel = str(Path(dockerfile_identified).relative_to(codebase_path))
            else:
                dockerfile_rel = dockerfile_identified

            if dockerfile_rel in dockerfiles_found:
                logger.info(f"Auto-selected LLM-identified Dockerfile: {dockerfile_rel}")
                return dockerfile_rel
        except ValueError:
            pass

    # Fall back to first discovered Dockerfile
    selected = dockerfiles_found[0]
    logger.info(f"Auto-selected first discovered Dockerfile: {selected}")
    return selected


def _collect_dockerfile_selection(state: dict) -> str | None:
    """Legacy function - redirects to new confirmation flow.

    Args:
        state: Current graph state with dockerfiles_found.

    Returns:
        Selected Dockerfile path, or None if no Dockerfiles.
    """
    return _collect_dockerfile_confirmation(state)


def _display_analysis_table(analysis):
    """Display analysis results in a table format."""
    table = Table(title="Codebase Analysis")

    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    # Dockerfile info
    if analysis.dockerfile:
        table.add_row("Base Image", analysis.dockerfile.base_image or "Not found")
        table.add_row("Exposed Ports", ", ".join(map(str, analysis.dockerfile.exposed_ports)) or "None")
        table.add_row("CMD", analysis.dockerfile.cmd or "Not set")

    # Dependencies
    if analysis.dependencies:
        table.add_row("Language", analysis.dependencies.language or "Unknown")
        table.add_row("Package Manager", analysis.dependencies.package_manager or "Unknown")
        dep_count = len(analysis.dependencies.dependencies)
        table.add_row("Dependencies", f"{dep_count} packages")

    # Resources
    table.add_row("Suggested CPU", f"{analysis.suggested_resources.get('cpu', 500)} MHz")
    table.add_row("Suggested Memory", f"{analysis.suggested_resources.get('memory', 256)} MB")

    # Env vars
    env_vars = analysis.env_vars_required[:5]
    if env_vars:
        table.add_row("Env Vars Detected", ", ".join(env_vars))

    # Files analyzed
    table.add_row("Files Analyzed", ", ".join(analysis.files_analyzed))

    console.print(table)

    if analysis.errors:
        console.print("\n[yellow]Warnings:[/yellow]")
        for error in analysis.errors:
            console.print(f"  - {error}")


def _display_results(result: dict, output: Optional[str], verbose: bool):
    """Display generation results."""
    job_spec = result.get("job_spec", "")

    if not job_spec:
        console.print("[red]Error:[/red] No job specification generated")
        if result.get("codebase_analysis", {}).get("error"):
            console.print(f"Analysis error: {result['codebase_analysis']['error']}")
        return

    # Display validation status
    if result.get("hcl_valid"):
        console.print("[green]HCL Valid[/green]")
    else:
        console.print(f"[yellow]HCL Validation:[/yellow] {result.get('validation_error', 'Unknown')}")

    # Display job info
    console.print(f"\n[bold]Job Name:[/bold] {result.get('job_name', 'unknown')}")

    # Display HCL
    if output:
        # Write to file
        output_path = Path(output)
        output_path.write_text(job_spec)
        console.print(f"\n[green]Saved to:[/green] {output}")
    else:
        # Display to console
        console.print("\n[bold]Generated Job Specification:[/bold]\n")
        syntax = Syntax(job_spec, "hcl", theme="monokai", line_numbers=True)
        console.print(syntax)

    # Verbose output
    if verbose:
        console.print("\n[bold]Full Analysis:[/bold]")
        console.print(json.dumps(result.get("codebase_analysis", {}), indent=2))


def _print_event(event: dict):
    """Print a graph event for verbose mode."""
    for node, data in event.items():
        if node.startswith("__"):
            continue
        console.print(f"[dim]Node: {node}[/dim]")


def _display_infra_status(health: InfraHealthReport):
    """Display infrastructure health status in a rich table.

    Args:
        health: InfraHealthReport with status of all services.
    """
    table = Table(title="Infrastructure Status")
    table.add_column("Service", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Address")
    table.add_column("Issue / Action")

    for status in health.statuses:
        if status.available:
            status_str = "[green]OK[/green]"
            action = "-"
        else:
            status_str = "[red]FAILED[/red]"
            # Combine error and suggestion for clarity
            parts = []
            if status.error:
                parts.append(status.error)
            if status.suggestion:
                parts.append(f"[dim]{status.suggestion}[/dim]")
            action = "\n".join(parts) if parts else "Unknown error"

        table.add_row(
            status.service,
            status_str,
            status.address,
            action,
        )

    console.print()
    console.print(table)


if __name__ == "__main__":
    app()
