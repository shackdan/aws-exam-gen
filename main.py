"""
main.py
───────
CLI Entrypoint — AWS Certification Exam Generator & Validator

Supported commands:
  ingest    Ingest PDF documentation into ChromaDB for a certification.
  generate  Generate and validate MCQ questions for a certification.
  status    Show ChromaDB collection info for an ingested certification.
  reset     Delete a certification's ChromaDB collection (re-ingest).
  list      List all supported certifications from the registry.

Usage examples:
  python main.py ingest   --cert SAA-C03 --path ./source_docs/saa-c03/
  python main.py generate --cert SAA-C03 --count 10 --output json
  python main.py generate --cert SAA-C03 --count 20 --output csv --domain "Design Secure Architectures"
  python main.py generate --cert SAA-C03 --count 5  --output moodle_xml --out-file ./my_questions.xml
  python main.py status   --cert SAA-C03
  python main.py reset    --cert SAA-C03
  python main.py list
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# ─────────────────────────────────────────────
# Bootstrap — ensure project root is on sys.path
# so sibling modules resolve correctly when
# main.py is invoked from any working directory.
# ─────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).parent.resolve()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import (
    CHROMA_DIR,
    DATA_DIR,
    OLLAMA_CONFIG,
    OUTPUT_DIR,
    REGISTRY_PATH,
)
from schemas import ExportFormat, GenerationRequest
from utils import (
    build_output_filename,
    get_cert_metadata,
    load_registry,
    logger,
    setup_logging,
)

console = Console()


# ─────────────────────────────────────────────
# CLI root group
# ─────────────────────────────────────────────

@click.group()
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(
        ["DEBUG", "INFO", "WARNING", "ERROR"],
        case_sensitive=False,
    ),
    show_default=True,
    help="Set logging verbosity.",
)
@click.pass_context
def cli(ctx: click.Context, log_level: str) -> None:
    """
    \b
    ╔══════════════════════════════════════════════════════╗
    ║   AWS Certification Exam Generator & Validator       ║
    ║   Local AI Pipeline  ·  Powered by Ollama + RAG      ║
    ╚══════════════════════════════════════════════════════╝

    Generate realistic AWS certification practice questions
    using a local LLM (Ollama) and RAG over your own PDF docs.
    """
    # Re-configure logging with the user-supplied level
    setup_logging(log_level)
    # Ensure the click context object is always a dict
    ctx.ensure_object(dict)
    ctx.obj["log_level"] = log_level


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _abort(message: str, exit_code: int = 1) -> None:
    """Print a styled error panel and exit with the given code."""
    console.print(
        Panel(
            f"[bold red]{message}[/bold red]",
            title        = "[red]Error[/red]",
            border_style = "red",
            expand       = False,
        )
    )
    sys.exit(exit_code)


def _validate_cert_code(cert_code: str) -> dict:
    """
    Load the registry and return metadata for cert_code.
    Calls _abort() on any error so callers get a clean exit.
    """
    try:
        registry = load_registry(REGISTRY_PATH)
        return get_cert_metadata(cert_code, registry)
    except FileNotFoundError as fnf:
        _abort(str(fnf))
    except ValueError as ve:
        _abort(str(ve))


def _validate_export_format(fmt: str) -> ExportFormat:
    """
    Map a CLI format string to an ExportFormat enum value.
    Calls _abort() with a helpful message on invalid input.
    """
    format_map = {
        "json"      : ExportFormat.JSON,
        "csv"       : ExportFormat.CSV,
        "moodle_xml": ExportFormat.MOODLE,
        "moodle"    : ExportFormat.MOODLE,
        "xml"       : ExportFormat.MOODLE,
    }
    result = format_map.get(fmt.lower())
    if result is None:
        _abort(
            f"Unsupported output format: '{fmt}'. "
            f"Choose from: {', '.join(format_map.keys())}"
        )
    return result


def _confirm_reset(cert_code: str) -> bool:
    """
    Prompt the user to confirm a destructive reset operation.
    Returns True if confirmed, False if declined.
    """
    console.print(
        Panel(
            f"[bold yellow]This will permanently delete the ChromaDB "
            f"collection for [white]{cert_code}[/white].\n"
            f"All ingested document embeddings will be lost.\n"
            f"You will need to re-run 'ingest' before generating questions.[/bold yellow]",
            title        = "[yellow]⚠ Confirm Reset[/yellow]",
            border_style = "yellow",
            expand       = False,
        )
    )
    return click.confirm(
        f"  Are you sure you want to reset '{cert_code}'?",
        default=False,
    )


# ─────────────────────────────────────────────
# ingest command
# ─────────────────────────────────────────────

@cli.command("ingest")
@click.option(
    "--cert",
    required = True,
    metavar  = "CERT_CODE",
    help     = "AWS certification code to ingest documents for (e.g. SAA-C03).",
)
@click.option(
    "--path",
    required = False,
    default  = None,
    metavar  = "DIR_PATH",
    help     = (
        "Directory containing PDF files to ingest. "
        "Defaults to ./data/{CERT_CODE}/ if not specified."
    ),
)
@click.option(
    "--reset-first",
    is_flag  = True,
    default  = False,
    help     = (
        "Delete the existing collection before ingesting. "
        "Use this to do a clean re-ingest of updated documents."
    ),
)
@click.pass_context
def cmd_ingest(
    ctx:         click.Context,
    cert:        str,
    path:        Optional[str],
    reset_first: bool,
) -> None:
    """
    Ingest PDF documentation into ChromaDB for a certification.

    \b
    Scans the specified directory for PDF files (exam guides,
    whitepapers, FAQs), extracts and cleans text, splits into
    overlapping chunks, embeds via sentence-transformers (CPU),
    and upserts into a local ChromaDB collection.

    \b
    Examples:
      python main.py ingest --cert SAA-C03 --path ./docs/saa-c03/
      python main.py ingest --cert DVA-C02
      python main.py ingest --cert SAA-C03 --reset-first
    """
    cert_code = cert.upper().strip()

    # ── Validate cert code ─────────────────────
    cert_metadata = _validate_cert_code(cert_code)

    # ── Resolve source path ────────────────────
    if path:
        source_path = Path(path).resolve()
    else:
        source_path = DATA_DIR / cert_code
        console.print(
            f"[dim]No --path specified. "
            f"Using default: '{source_path}'[/dim]"
        )

    console.print(
        Panel(
            f"[bold]Certification :[/bold] {cert_metadata['name']} ({cert_code})\n"
            f"[bold]Tier          :[/bold] {cert_metadata['tier']}\n"
            f"[bold]Source path   :[/bold] {source_path}\n"
            f"[bold]ChromaDB dir  :[/bold] {CHROMA_DIR}",
            title        = "[bold blue]Ingestion Configuration[/bold blue]",
            border_style = "blue",
            expand       = False,
        )
    )

    # ── Optional reset ─────────────────────────
    if reset_first:
        if _confirm_reset(cert_code):
            from ingest import delete_collection
            try:
                deleted = delete_collection(cert_code)
                if deleted:
                    console.print(
                        f"[green]✔ Collection for '{cert_code}' deleted.[/green]"
                    )
            except RuntimeError as del_err:
                _abort(f"Reset failed: {del_err}")
        else:
            console.print("[yellow]Reset cancelled. Proceeding with existing collection.[/yellow]")

    # ── Run ingestion ──────────────────────────
    from ingest import run_ingestion_pipeline

    try:
        report = run_ingestion_pipeline(
            cert_code   = cert_code,
            source_path = source_path,
        )
    except FileNotFoundError as fnf:
        _abort(
            f"Source directory not found: {fnf}\n\n"
            f"Create the directory and place your PDF files inside:\n"
            f"  mkdir -p {source_path}"
        )
    except RuntimeError as rte:
        _abort(f"Ingestion pipeline error: {rte}")
    except Exception as exc:
        logger.debug(traceback.format_exc())
        _abort(f"Unexpected error during ingestion: {exc}")

    # ── Success summary ────────────────────────
    if report.pdfs_processed == 0:
        console.print(
            Panel(
                f"[yellow]No PDFs were successfully processed from "
                f"'{source_path}'.\n"
                "Place your AWS exam guide and whitepaper PDFs in that "
                "directory and re-run.[/yellow]",
                title        = "[yellow]Nothing Ingested[/yellow]",
                border_style = "yellow",
                expand       = False,
            )
        )
        sys.exit(0)

    console.print(
        f"\n[bold green]✔ Ingestion complete.[/bold green] "
        f"{report.chunks_upserted} new chunk(s) added to "
        f"[cyan]'{report.collection_name}'[/cyan].\n"
        f"Run [bold]python main.py generate --cert {cert_code}[/bold] "
        f"to start generating questions."
    )


# ─────────────────────────────────────────────
# generate command
# ─────────────────────────────────────────────

@cli.command("generate")
@click.option(
    "--cert",
    required = True,
    metavar  = "CERT_CODE",
    help     = "AWS certification code to generate questions for (e.g. SAA-C03).",
)
@click.option(
    "--count",
    required = True,
    type     = click.IntRange(1, 200),
    metavar  = "N",
    help     = "Number of questions to generate (1–200).",
)
@click.option(
    "--output",
    "output_format",
    default  = "json",
    type     = click.Choice(
        ["json", "csv", "moodle_xml"],
        case_sensitive=False,
    ),
    show_default = True,
    help     = "Export format for approved questions.",
)
@click.option(
    "--out-file",
    "output_file",
    default  = None,
    metavar  = "FILE_PATH",
    help     = (
        "Explicit output file path. "
        "If omitted, a timestamped file is created in ./output/."
    ),
)
@click.option(
    "--domain",
    "domain_filter",
    default  = None,
    metavar  = "DOMAIN_NAME",
    help     = (
        "Restrict generation to a specific exam domain "
        "(partial match supported, e.g. 'Secure')."
    ),
)
@click.option(
    "--difficulty",
    default  = None,
    type     = click.Choice(
        ["Foundational", "Associate", "Professional", "Specialty"],
        case_sensitive=False,
    ),
    help     = (
        "Override the difficulty tier label embedded in each question. "
        "Defaults to the certification's own tier."
    ),
)
@click.option(
    "--show-questions",
    is_flag  = True,
    default  = False,
    help     = "Print the full question table to the console after generation.",
)
@click.pass_context
def cmd_generate(
    ctx:           click.Context,
    cert:          str,
    count:         int,
    output_format: str,
    output_file:   Optional[str],
    domain_filter: Optional[str],
    difficulty:    Optional[str],
    show_questions:bool,
) -> None:
    """
    Generate and validate MCQ questions for an AWS certification.

    \b
    Queries the ChromaDB vector store for relevant context, prompts
    the local Ollama LLM to draft questions, then runs each question
    through the AI Reviewer loop (up to 2 revision attempts).
    Only Approved questions are written to the output file.

    \b
    Examples:
      python main.py generate --cert SAA-C03 --count 10
      python main.py generate --cert DVA-C02 --count 20 --output csv
      python main.py generate --cert SCS-C03 --count 5  --domain "Identity"
      python main.py generate --cert SAA-C03 --count 10 --output moodle_xml
    """
    cert_code = cert.upper().strip()

    # ── Validate cert code ─────────────────────
    cert_metadata = _validate_cert_code(cert_code)

    # ── Validate and map export format ────────
    export_fmt = _validate_export_format(output_format)

    # ── Display generation configuration ──────
    console.print(
        Panel(
            f"[bold]Certification :[/bold] {cert_metadata['name']} ({cert_code})\n"
            f"[bold]Tier          :[/bold] {cert_metadata['tier']}\n"
            f"[bold]Count         :[/bold] {count} question(s)\n"
            f"[bold]Output format :[/bold] {output_format}\n"
            f"[bold]Domain filter :[/bold] {domain_filter or '(all domains)'}\n"
            f"[bold]Difficulty    :[/bold] {difficulty or cert_metadata['tier']}\n"
            f"[bold]Model         :[/bold] {OLLAMA_CONFIG.model}\n"
            f"[bold]Context window:[/bold] {OLLAMA_CONFIG.num_ctx} tokens\n"
            f"[bold]Output dir    :[/bold] {OUTPUT_DIR}",
            title        = "[bold blue]Generation Configuration[/bold blue]",
            border_style = "blue",
            expand       = False,
        )
    )

    # ── Build GenerationRequest ────────────────
    request = GenerationRequest(
        cert_code     = cert_code,
        count         = count,
        output_format = export_fmt,
        output_path   = output_file,
        domain_filter = domain_filter,
        difficulty    = difficulty,
    )

    # ── Run pipeline ───────────────────────────
    from generator import display_results, run_generation_pipeline

    start_time = time.monotonic()

    try:
        result = run_generation_pipeline(request)
    except RuntimeError as rte:
        _abort(str(rte))
    except KeyboardInterrupt:
        console.print(
            "\n[yellow]Generation interrupted by user.[/yellow]"
        )
        sys.exit(130)
    except Exception as exc:
        logger.debug(traceback.format_exc())
        _abort(f"Unexpected error during generation: {exc}")

    # ── Display results ────────────────────────
    if show_questions and result.questions:
        from utils import print_question_table
        print_question_table(result.questions)

    display_results(result)

    # ── Actionable next steps ──────────────────
    if result.output_file:
        console.print(
            f"\n[bold green]✔ Output written to:[/bold green] "
            f"[cyan]{result.output_file}[/cyan]"
        )

    if result.approved == 0:
        console.print(
            Panel(
                "[yellow]No questions were approved by the AI Reviewer.\n"
                "Suggestions:\n"
                "  • Ensure your ChromaDB collection is populated "
                "(run 'ingest' first).\n"
                "  • Verify Ollama is running with the correct model:\n"
                f"    ollama run {OLLAMA_CONFIG.model}\n"
                "  • Try reducing --count and re-running.\n"
                "  • Check logs with --log-level DEBUG for details.[/yellow]",
                title        = "[yellow]No Questions Approved[/yellow]",
                border_style = "yellow",
                expand       = False,
            )
        )
        sys.exit(1)

    # ── Exit code reflects approval rate ──────
    # Exit 0 = all approved, Exit 2 = partial approval
    if result.rejected > 0:
        console.print(
            f"[yellow]Note: {result.rejected} question(s) were rejected "
            "after max revision attempts. "
            "These are excluded from the output file.[/yellow]"
        )
        sys.exit(0)

    sys.exit(0)


# ─────────────────────────────────────────────
# status command
# ─────────────────────────────────────────────

@cli.command("status")
@click.option(
    "--cert",
    required = True,
    metavar  = "CERT_CODE",
    help     = "AWS certification code to inspect (e.g. SAA-C03).",
)
@click.pass_context
def cmd_status(ctx: click.Context, cert: str) -> None:
    """
    Show ChromaDB collection info for an ingested certification.

    \b
    Displays: collection name, total chunk count, source PDF
    filenames, and inferred domain hints stored in the collection.

    \b
    Example:
      python main.py status --cert SAA-C03
    """
    cert_code = cert.upper().strip()

    # Validate cert code exists in registry
    _validate_cert_code(cert_code)

    from ingest import get_collection_info

    try:
        info = get_collection_info(cert_code)
    except RuntimeError as rte:
        _abort(str(rte))
    except Exception as exc:
        logger.debug(traceback.format_exc())
        _abort(f"Could not retrieve collection status: {exc}")

    # ── Source files table ─────────────────────
    files_table = Table(
        title        = f"Ingested Source Files — {cert_code}",
        show_header  = True,
        header_style = "bold magenta",
        show_lines   = True,
    )
    files_table.add_column("PDF Filename", style="cyan",  width=50)

    if info["source_files"]:
        for fname in info["source_files"]:
            files_table.add_row(fname)
    else:
        files_table.add_row("[dim](none found in sample)[/dim]")

    console.print(files_table)

    # ── Domain hints table ─────────────────────
    domains_table = Table(
        title        = f"Detected Domain Hints — {cert_code}",
        show_header  = True,
        header_style = "bold magenta",
        show_lines   = True,
    )
    domains_table.add_column("Domain Hint", style="green", width=50)

    if info["domain_hints"]:
        for hint in info["domain_hints"]:
            domains_table.add_row(hint)
    else:
        domains_table.add_row("[dim](no domain hints detected)[/dim]")

    console.print(domains_table)

    # ── Summary panel ──────────────────────────
    console.print(
        Panel(
            f"[bold]Certification  :[/bold]  {cert_code}\n"
            f"[bold]Collection name:[/bold]  {info['collection_name']}\n"
            f"[bold]Total chunks   :[/bold]  [green]{info['total_chunks']}[/green]\n"
            f"[bold]Source PDFs    :[/bold]  {len(info['source_files'])}\n"
            f"[bold]Domain hints   :[/bold]  {len(info['domain_hints'])}\n"
            f"[bold]ChromaDB path  :[/bold]  {CHROMA_DIR}",
            title        = "[bold blue]Collection Status[/bold blue]",
            border_style = "blue",
            expand       = False,
        )
    )

    if info["total_chunks"] == 0:
        console.print(
            "[yellow]Collection is empty. "
            f"Run 'python main.py ingest --cert {cert_code}' "
            "to populate it.[/yellow]"
        )


# ─────────────────────────────────────────────
# reset command
# ─────────────────────────────────────────────

@cli.command("reset")
@click.option(
    "--cert",
    required = True,
    metavar  = "CERT_CODE",
    help     = "AWS certification code whose collection should be deleted.",
)
@click.option(
    "--yes",
    is_flag  = True,
    default  = False,
    help     = "Skip the confirmation prompt (for scripting).",
)
@click.pass_context
def cmd_reset(ctx: click.Context, cert: str, yes: bool) -> None:
    """
    Delete a certification's ChromaDB collection.

    \b
    Permanently removes all stored document embeddings for the
    specified certification. Use this before re-ingesting updated
    documentation to ensure a clean collection state.

    \b
    Example:
      python main.py reset --cert SAA-C03
      python main.py reset --cert SAA-C03 --yes
    """
    cert_code = cert.upper().strip()

    # Validate cert code
    _validate_cert_code(cert_code)

    # Confirm unless --yes flag supplied
    if not yes:
        if not _confirm_reset(cert_code):
            console.print("[yellow]Reset cancelled.[/yellow]")
            sys.exit(0)

    from ingest import delete_collection

    try:
        deleted = delete_collection(cert_code)
    except RuntimeError as rte:
        _abort(f"Reset failed: {rte}")
    except Exception as exc:
        logger.debug(traceback.format_exc())
        _abort(f"Unexpected error during reset: {exc}")

    if deleted:
        console.print(
            Panel(
                f"[green]ChromaDB collection for [bold]{cert_code}[/bold] "
                f"has been deleted.\n\n"
                f"To re-ingest documents, run:\n"
                f"  [bold]python main.py ingest --cert {cert_code} "
                f"--path ./source_docs/{cert_code.lower()}/[/bold][/green]",
                title        = "[green]✔ Reset Complete[/green]",
                border_style = "green",
                expand       = False,
            )
        )
    else:
        console.print(
            f"[yellow]No collection found for '{cert_code}'. "
            "Nothing was deleted.[/yellow]"
        )


# ─────────────────────────────────────────────
# list command
# ─────────────────────────────────────────────

@cli.command("list")
@click.option(
    "--tier",
    default  = None,
    type     = click.Choice(
        ["Foundational", "Associate", "Professional", "Specialty"],
        case_sensitive=False,
    ),
    help     = "Filter certifications by tier.",
)
@click.pass_context
def cmd_list(ctx: click.Context, tier: Optional[str]) -> None:
    """
    List all supported AWS certifications from the registry.

    \b
    Shows certification codes, names, tiers, domain counts,
    and total exam question counts. Optionally filter by tier.

    \b
    Examples:
      python main.py list
      python main.py list --tier Associate
    """
    try:
        registry = load_registry(REGISTRY_PATH)
    except FileNotFoundError as fnf:
        _abort(str(fnf))

    certs = registry.get("certifications", {})

    if not certs:
        _abort("No certifications found in registry.json.")

    # ── Build display table ────────────────────
    table = Table(
        title        = "Supported AWS Certifications",
        show_header  = True,
        header_style = "bold magenta",
        show_lines   = True,
    )

    table.add_column("Code",       style="bold cyan",  width=12)
    table.add_column("Name",       style="white",      width=48)
    table.add_column("Tier",       style="green",      width=14)
    table.add_column("Domains",    style="yellow",     width=9,  justify="right")
    table.add_column("Exam Qs",    style="blue",       width=9,  justify="right")
    table.add_column("Pass Score", style="magenta",    width=11, justify="right")

    tier_colours = {
        "Foundational": "[white]Foundational[/white]",
        "Associate"   : "[green]Associate[/green]",
        "Professional": "[cyan]Professional[/cyan]",
        "Specialty"   : "[magenta]Specialty[/magenta]",
    }

    shown = 0
    for code, meta in sorted(certs.items()):
        cert_tier = meta.get("tier", "")

        # Apply tier filter if specified
        if tier and cert_tier.lower() != tier.lower():
            continue

        table.add_row(
            code,
            meta.get("name", ""),
            tier_colours.get(cert_tier, cert_tier),
            str(len(meta.get("domains", []))),
            str(meta.get("total_questions", "N/A")),
            str(meta.get("passing_score",   "N/A")),
        )
        shown += 1

    console.print(table)

    footer = (
        f"[dim]Showing {shown} certification(s)"
        + (f" filtered by tier: {tier}" if tier else "")
        + "[/dim]"
    )
    console.print(footer)

    console.print(
        "\n[dim]To ingest documents for a certification:[/dim]\n"
        "  [bold]python main.py ingest --cert <CODE> --path <DIR>[/bold]\n\n"
        "[dim]To generate questions:[/dim]\n"
        "  [bold]python main.py generate --cert <CODE> --count <N>[/bold]"
    )


# ─────────────────────────────────────────────
# Startup checks
# ─────────────────────────────────────────────

def _run_startup_checks() -> None:
    """
    Perform lightweight environment checks before any command runs.

    Checks:
      1. Python version >= 3.10
      2. registry.json is present and readable
      3. Required directories exist (created if missing)
      4. Ollama reachability (non-fatal warning only)

    Prints warnings to console but does not abort — individual
    commands handle their own dependency validation.
    """
    # Python version
    if sys.version_info < (3, 10):
        console.print(
            "[yellow]⚠ Python 3.10+ is recommended. "
            f"Current version: {sys.version}[/yellow]"
        )

    # Registry
    if not REGISTRY_PATH.exists():
        console.print(
            f"[yellow]⚠ registry.json not found at {REGISTRY_PATH}.[/yellow]"
        )

    # Directories
    for directory in (DATA_DIR, CHROMA_DIR, OUTPUT_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    # Ollama ping (non-fatal)
    try:
        import ollama
        client = ollama.Client(
            host    = OLLAMA_CONFIG.base_url,
            timeout = 5,
        )
        client.list()
    except Exception:
        console.print(
            f"[yellow]⚠ Ollama not reachable at "
            f"{OLLAMA_CONFIG.base_url}. "
            "Start Ollama before running 'generate'.\n"
            f"  Command: ollama serve[/yellow]"
        )


# ─────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────

if __name__ == "__main__":
    _run_startup_checks()
    cli(obj={})
