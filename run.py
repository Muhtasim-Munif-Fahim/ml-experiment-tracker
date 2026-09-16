#!/usr/bin/env python3
"""Main entry point for ML Experiment Tracker."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import click
import uvicorn


@click.group(invoke_without_command=True)
@click.pass_context
def main(ctx: click.Context) -> None:
    """ML Experiment Tracker server and local tooling."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(serve)


@main.command()
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", default=8000, show_default=True, type=int)
@click.option("--reload/--no-reload", default=True, show_default=True)
def serve(host: str, port: int, reload: bool) -> None:
    """Start the FastAPI server."""
    uvicorn.run("src.api:app", host=host, port=port, reload=reload)


@main.command("compare")
@click.option(
    "--run-id",
    "run_ids",
    multiple=True,
    required=True,
    help="Run id to include. Repeat once per run (at least two).",
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["markdown", "html"], case_sensitive=False),
    default="markdown",
    show_default=True,
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Destination file. Prints the report to stdout when omitted.",
)
@click.option(
    "--storage",
    "storage_path",
    type=click.Path(path_type=Path),
    default=Path("./mlruns"),
    show_default=True,
    help="Local storage directory. Never requires S3.",
)
def compare_runs_command(
    run_ids: Sequence[str],
    fmt: str,
    output: Optional[Path],
    storage_path: Path,
) -> None:
    """Export a Markdown or HTML comparison of two or more runs."""
    from src.models import render_run_comparison
    from src.storage import LocalStorageBackend

    if len(run_ids) < 2:
        raise click.UsageError("at least two --run-id values are required")

    backend = LocalStorageBackend(storage_path)
    try:
        comparison = backend.compare_runs(list(run_ids))
    except (KeyError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    if output is None:
        text = render_run_comparison(comparison, fmt)
        click.echo(text, nl=not text.endswith("\n"))
        return

    written = backend.export_run_comparison(list(run_ids), str(output), fmt=fmt)
    click.echo(written)


if __name__ == "__main__":
    main()
