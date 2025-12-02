"""CLI entry point for the Nomad Job Spec Agent."""

import json
import sys
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.syntax import Syntax
from rich.table import Table

from config.settings import get_settings
from src.llm.provider import get_llm
from src.graph import compile_graph, create_initial_state
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

    try:
        # First run - analyze and generate questions
        with console.status("[bold green]Analyzing codebase..."):
            for event in graph.stream(state, config):
                if verbose:
                    _print_event(event)

                # Check for interrupt (questions ready)
                if "__interrupt__" in event:
                    break

        # Get current state with analysis results
        current_state = graph.get_state(config)
        analysis = current_state.values.get("codebase_analysis", {})

        # If no prompt was provided, collect it interactively after analysis
        if not prompt and not no_questions:
            prompt = _collect_deployment_prompt(analysis)
            graph.update_state(config, {"prompt": prompt})

            # Show configuration now that we have the prompt
            console.print(Panel(
                f"[bold]Nomad Job Spec Generator[/bold]\n\n"
                f"Prompt: {prompt}\n"
                f"Codebase: {path}\n"
                f"Datacenter: {settings.nomad_datacenter}  Namespace: {cluster_id}",
                title="Configuration",
            ))

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
    (like vault_paths which use interactive step-by-step flow).
    """
    questions = state.get("questions", [])
    responses = {}

    console.print("\n[bold]Please answer the following questions:[/bold]\n")

    for i, question in enumerate(questions, 1):
        if isinstance(question, dict) and question.get("type") == "vault_paths":
            # Route to interactive Vault path handler
            vault_responses = _collect_vault_path_responses(question["suggestions"])
            responses[f"q{i}"] = vault_responses
        else:
            # Original behavior for plain string questions
            response = Prompt.ask(f"[cyan]{i}.[/cyan] {question}")
            responses[f"q{i}"] = response

    return responses


def _collect_vault_path_responses(suggestions: list[dict]) -> dict[str, str]:
    """Interactive step-by-step Vault path confirmation.

    Flow:
    1. Show summary table of all suggestions
    2. Ask confirm/edit
    3. If edit: step through each variable one by one with pre-filled defaults
    4. Show final summary and ask confirm/edit again
    5. Return final mappings as {env_var: vault_path}

    Args:
        suggestions: List of suggestion dicts with keys:
            - env_var: Environment variable name
            - suggested_path: Vault path (without key)
            - key: Key within the Vault path
            - confidence: Confidence score (0.0-1.0)
            - vault_reference: Full path#key reference

    Returns:
        Dict mapping env_var names to confirmed vault_reference paths.
    """
    # Build initial paths from suggestions
    paths = {}
    for s in suggestions:
        paths[s["env_var"]] = s.get("vault_reference") or f"{s['suggested_path']}#{s['key']}"

    while True:
        # Display summary table
        _display_vault_suggestions_table(suggestions, paths)

        # Ask confirm/edit
        choice = Prompt.ask(
            "\nAccept all suggestions?",
            choices=["confirm", "edit"],
            default="confirm"
        )

        if choice == "confirm":
            return paths

        # Edit mode: step through each variable
        console.print("\n[dim]Press Enter to accept the suggested path, or type a new path.[/dim]\n")

        for idx, s in enumerate(suggestions, 1):
            env_var = s["env_var"]
            current_path = paths[env_var]

            new_path = Prompt.ask(
                f"[cyan][{idx}/{len(suggestions)}][/cyan] {env_var}",
                default=current_path
            )
            paths[env_var] = new_path

        # After editing all, show final summary (loop back to top)
        console.print()  # Blank line before showing table again


def _display_vault_suggestions_table(suggestions: list[dict], paths: dict[str, str]):
    """Display Vault path suggestions in a formatted table.

    Args:
        suggestions: Original suggestions with confidence info.
        paths: Current path mappings (may differ from original suggestions).
    """
    table = Table(title="Vault Path Configuration")
    table.add_column("Variable", style="cyan")
    table.add_column("Path", style="green")
    table.add_column("Confidence", justify="right")

    for s in suggestions:
        env_var = s["env_var"]
        current_path = paths.get(env_var, "")
        original_path = s.get("vault_reference") or f"{s['suggested_path']}#{s['key']}"

        # Show confidence for original suggestion, or "custom" if user changed it
        if current_path == original_path:
            confidence = s.get("confidence", 0)
            if confidence >= 0.9:
                conf_str = "[green]validated[/green]"
            elif confidence >= 0.5:
                conf_str = f"[yellow]{int(confidence * 100)}%[/yellow]"
            else:
                conf_str = f"[red]{int(confidence * 100)}%[/red]"
        else:
            conf_str = "[blue]custom[/blue]"

        table.add_row(env_var, current_path, conf_str)

    console.print()
    console.print(table)


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


def _collect_deployment_prompt(analysis: dict) -> str:
    """Display analysis summary and collect deployment prompt from user.

    Args:
        analysis: The codebase analysis results.

    Returns:
        User's deployment prompt.
    """
    console.print("\n[bold]Codebase Analysis Complete[/bold]\n")

    # Build summary table
    table = Table(title="Analysis Summary")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    dockerfile = analysis.get("dockerfile", {})
    if dockerfile:
        table.add_row("Base Image", dockerfile.get("base_image") or "Not found")
        ports = dockerfile.get("exposed_ports", [])
        if ports:
            table.add_row("Exposed Ports", ", ".join(map(str, ports)))

    deps = analysis.get("dependencies", {})
    if deps:
        table.add_row("Language", deps.get("language") or "Unknown")

    env_vars = analysis.get("env_vars_required", [])[:5]
    if env_vars:
        table.add_row("Env Vars Detected", ", ".join(env_vars))

    resources = analysis.get("suggested_resources", {})
    table.add_row("Suggested CPU", f"{resources.get('cpu', 500)} MHz")
    table.add_row("Suggested Memory", f"{resources.get('memory', 256)} MB")

    console.print(table)
    console.print()

    return Prompt.ask(
        "[bold cyan]What would you like to deploy?[/bold cyan]",
        default="Deploy this application"
    )


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
