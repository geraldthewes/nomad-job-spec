"""CLI for managing LangFuse prompts.

This is a separate CLI tool from nomad-spec, used to manage the prompts
for the nomad-spec tool itself. It provides commands to push local prompts
to LangFuse and pull prompts from LangFuse to local files.

Usage:
    nomad-spec-prompt push [--name NAME] [--label LABEL]
    nomad-spec-prompt pull [--name NAME] [--label LABEL] [--version VERSION]
    nomad-spec-prompt list
"""

import typer
from rich.console import Console
from rich.table import Table

from config.settings import get_settings
from src.prompts import PromptNotFoundError, get_prompt_manager

app = typer.Typer(
    name="nomad-spec-prompt",
    help="Manage LangFuse prompts for the nomad-spec tool",
    add_completion=False,
)
console = Console()


def _check_langfuse_enabled() -> None:
    """Check that LangFuse is enabled and raise if not."""
    settings = get_settings()
    if not settings.langfuse_enabled:
        console.print(
            "[red]Error:[/red] LangFuse is not enabled.\n"
            "Set LANGFUSE_ENABLED=true and provide LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY."
        )
        raise typer.Exit(1)


@app.command("push")
def push(
    name: str | None = typer.Option(
        None,
        "--name", "-n",
        help="Specific prompt name to push. If not provided, pushes all prompts.",
    ),
    label: str | None = typer.Option(
        None,
        "--label", "-l",
        help="Label to apply (e.g., 'production', 'development'). "
             "Defaults to LANGFUSE_PROMPT_LABEL setting.",
    ),
) -> None:
    """Push local prompts to LangFuse.

    Reads prompts from the prompts/ directory and uploads them to LangFuse.
    If a prompt already exists, a new version will be created.

    Examples:

        nomad-spec-prompt push

        nomad-spec-prompt push --name analysis

        nomad-spec-prompt push --label production
    """
    _check_langfuse_enabled()

    settings = get_settings()
    manager = get_prompt_manager(settings)

    try:
        if name:
            # Push single prompt
            result = manager.push_prompt(name, label)
            console.print(
                f"[green]\u2713[/green] {result['name']} \u2192 "
                f"v{result['version']} ({result['label']})"
            )
        else:
            # Push all prompts
            prompts = manager.list_prompts()
            if not prompts:
                console.print("[yellow]No local prompts found in prompts/ directory[/yellow]")
                return

            results = manager.push_all(label)
            for r in results:
                console.print(
                    f"[green]\u2713[/green] {r['name']} \u2192 "
                    f"v{r['version']} ({r['label']})"
                )
            console.print(f"\n[green]Pushed {len(results)} prompt(s) to LangFuse[/green]")

    except PromptNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except RuntimeError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command("pull")
def pull(
    name: str | None = typer.Option(
        None,
        "--name", "-n",
        help="Specific prompt name to pull. If not provided, pulls all known prompts.",
    ),
    label: str | None = typer.Option(
        None,
        "--label", "-l",
        help="Label to pull (e.g., 'production'). Defaults to LANGFUSE_PROMPT_LABEL setting.",
    ),
    version: int | None = typer.Option(
        None,
        "--version", "-v",
        help="Specific version number to pull.",
    ),
) -> None:
    """Pull prompts from LangFuse to local files.

    Downloads prompts from LangFuse and saves them to the prompts/ directory.
    This will overwrite existing local files.

    Examples:

        nomad-spec-prompt pull

        nomad-spec-prompt pull --name analysis

        nomad-spec-prompt pull --label production

        nomad-spec-prompt pull --name analysis --version 2
    """
    _check_langfuse_enabled()

    settings = get_settings()
    manager = get_prompt_manager(settings)

    try:
        if name:
            # Pull single prompt
            path = manager.pull_prompt(name, version, label)
            console.print(f"[green]\u2713[/green] {name} \u2192 {path}")
        else:
            # Pull all known prompts (based on local files)
            prompts = manager.list_prompts()
            if not prompts:
                console.print(
                    "[yellow]No local prompts found. "
                    "Use --name to specify a prompt to pull.[/yellow]"
                )
                return

            success_count = 0
            for prompt_name in prompts:
                try:
                    path = manager.pull_prompt(prompt_name, version, label)
                    console.print(f"[green]\u2713[/green] {prompt_name} \u2192 {path}")
                    success_count += 1
                except PromptNotFoundError:
                    console.print(
                        f"[yellow]![/yellow] {prompt_name} not found in LangFuse"
                    )

            if success_count > 0:
                console.print(f"\n[green]Pulled {success_count} prompt(s) from LangFuse[/green]")

    except PromptNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except RuntimeError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command("list")
def list_prompts() -> None:
    """List available prompts.

    Shows all prompts available in the local prompts/ directory.

    Examples:

        nomad-spec-prompt list
    """
    manager = get_prompt_manager()
    prompts = manager.list_prompts()

    if not prompts:
        console.print("[yellow]No prompts found in prompts/ directory[/yellow]")
        return

    table = Table(title="Local Prompts")
    table.add_column("Name", style="cyan")
    table.add_column("File", style="dim")

    for name in sorted(prompts):
        table.add_row(name, f"prompts/{name}.json")

    console.print(table)


if __name__ == "__main__":
    app()
