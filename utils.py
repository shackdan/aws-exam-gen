"""
utils.py
────────
Shared helpers: logging, JSON repair, export formatting,
Moodle XML serialization, and pretty console output.
"""

from __future__ import annotations

import ast
import csv
import json
import logging
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from xml.dom import minidom

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from schemas import ExamQuestion, GenerationResult, ValidationStatus

console = Console()


# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────

def setup_logging(level: str = "INFO") -> logging.Logger:
    """
    Configure root logger with Rich handler for pretty console output.
    Returns the package-level logger.
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )
    logger = logging.getLogger("aws_exam_gen")
    return logger


logger = setup_logging()


# ─────────────────────────────────────────────
# JSON repair / extraction
# ─────────────────────────────────────────────

def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Attempt to extract a JSON object from an LLM response that may
    contain surrounding prose or markdown code fences.

    Strategy (in order):
    1. Direct json.loads on stripped text.
    2. Extract content between first { and last }.
    3. Strip markdown fences then try again.
    4. Attempt lightweight comma / trailing-comma repair.
    5. Return None on total failure (triggers retry in caller).
    """
    # ── Strategy 1: direct parse ──────────────
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # ── Strategy 2: brace extraction ──────────
    start = text.find("{")
    end   = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # ── Strategy 3: strip markdown fences ─────
    fence_pattern = r"```(?:json)?\s*([\s\S]*?)```"
    matches = re.findall(fence_pattern, text, re.IGNORECASE)
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue

    # ── Strategy 4: lightweight JSON repair ───
    # Remove trailing commas before ] or }
    if start != -1 and end != -1:
        candidate = text[start : end + 1]
        repaired  = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass

    # ── Strategy 5: Python-style dict repair
    # Accept single quotes, trailing commas, and unquoted keys.
    if start != -1 and end != -1:
        candidate = text[start : end + 1]
        try:
            python_like = candidate.replace("true", "True").replace("false", "False").replace("null", "None")
            return ast.literal_eval(python_like)
        except (ValueError, SyntaxError):
            pass

    # ── Strategy 6: truncated-response repair ─
    # If the model ran out of context mid-response the JSON will have no
    # closing brace.  Walk the candidate character-by-character to count
    # unclosed structures, then append the required closing characters.
    if start != -1:
        candidate = text[start:]
        depth    = 0
        in_str   = False
        i        = 0
        while i < len(candidate):
            ch = candidate[i]
            if in_str:
                if ch == "\\" and i + 1 < len(candidate):
                    i += 2          # skip escaped character
                    continue
                if ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch in "{[":
                    depth += 1
                elif ch in "}]":
                    depth -= 1
            i += 1

        if depth > 0:
            suffix = ""
            if in_str:
                suffix += '"'       # close the open string
            suffix += "}" * depth   # close unclosed objects/arrays
            patched  = candidate.rstrip().rstrip(",") + suffix
            patched  = re.sub(r",\s*([}\]])", r"\1", patched)
            try:
                return json.loads(patched)
            except json.JSONDecodeError:
                pass

    logger.warning("Could not extract valid JSON from LLM response.")
    return None


def safe_json_field(data: Dict[str, Any], key: str, default: Any = None) -> Any:
    """Return data[key] if present and truthy, else default."""
    return data.get(key) or default


# ─────────────────────────────────────────────
# Progress bar factory
# ─────────────────────────────────────────────

def make_progress_bar() -> Progress:
    """Return a pre-configured Rich Progress bar for generation runs."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )


# ─────────────────────────────────────────────
# Export: JSON
# ─────────────────────────────────────────────

def export_to_json(
    questions: List[ExamQuestion],
    output_path: Path,
    cert_code: str,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Serialize questions to a structured JSON file with a metadata wrapper.

    The rag_source_chunks field is excluded from export as it is
    internal bookkeeping only.

    extra_metadata (e.g. domain_stats comparing actual vs. target content
    outline weights) is merged into the metadata block when provided, so
    content-outline alignment can be audited from the exported file itself
    instead of only from console output.
    """
    payload = {
        "metadata": {
            "cert_code":        cert_code,
            "generated_at":     datetime.now(timezone.utc).isoformat(),
            "total_questions":  len(questions),
            "approved":         sum(
                1 for q in questions
                if q.validation_status == ValidationStatus.APPROVED
            ),
            "rejected":         sum(
                1 for q in questions
                if q.validation_status == ValidationStatus.REJECTED
            ),
            **(extra_metadata or {}),
        },
        "questions": [
            json.loads(q.model_dump_json(exclude={"rag_source_chunks"}))
            for q in questions
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)

    logger.info(f"JSON export written → {output_path}")
    return output_path


# ─────────────────────────────────────────────
# Export: CSV
# ─────────────────────────────────────────────

def export_to_csv(
    questions: List[ExamQuestion],
    output_path: Path,
) -> Path:
    """
    Flatten questions to a CSV file.

    Multi-value fields (options, correct_answers, aws_services) are
    pipe-delimited so they remain readable in spreadsheet tools.
    """
    fieldnames = [
        "question_id",
        "cert_code",
        "domain",
        "question_type",
        "question_text",
        "options",
        "correct_answers",
        "explanation",
        "source_url",
        "validation_status",
        "revision_count",
        "aws_services",
        "difficulty",
        "created_at",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for q in questions:
            writer.writerow({
                "question_id":       q.question_id,
                "cert_code":         q.cert_code,
                "domain":            q.domain,
                "question_type":     q.question_type.value,
                "question_text":     q.question_text,
                "options":           " | ".join(q.options),
                "correct_answers":   ", ".join(q.correct_answers),
                "explanation":       q.explanation,
                "source_url":        q.source_url or "",
                "validation_status": q.validation_status.value,
                "revision_count":    q.revision_count,
                "aws_services":      ", ".join(q.aws_services),
                "difficulty":        q.difficulty or "",
                "created_at":        q.created_at.isoformat(),
            })

    logger.info(f"CSV export written → {output_path}")
    return output_path


# ─────────────────────────────────────────────
# Export: Moodle XML
# ─────────────────────────────────────────────

def _prettify_xml(element: ET.Element) -> str:
    """Return a well-indented XML string from an ElementTree element."""
    raw      = ET.tostring(element, encoding="unicode")
    reparsed = minidom.parseString(raw)
    return reparsed.toprettyxml(indent="  ")


def _fraction_for_option(
    letter: str,
    correct_answers: List[str],
    question_type_value: str,
) -> str:
    """
    Calculate the Moodle answer fraction string for one option.

    Single-answer  : correct → "100", wrong → "0"
    Multiple-select: correct → equal share of 100, wrong → "0"
    """
    if letter not in correct_answers:
        return "0"

    if question_type_value == "single":
        return "100"

    # Distribute credit equally across all correct answers
    share = round(100.0 / len(correct_answers), 4)
    return str(share)


def export_to_moodle_xml(
    questions: List[ExamQuestion],
    output_path: Path,
) -> Path:
    """
    Serialize questions to Moodle GIFT-compatible XML (multichoice format).

    Single-answer      → <single>true</single>
    Multiple-selection → <single>false</single>

    Explanation text is placed in <generalfeedback>.
    Each distractor receives zero credit; correct answers share 100 %.
    """
    quiz = ET.Element("quiz")

    for q in questions:
        # ── Question element ───────────────────
        question_el = ET.SubElement(quiz, "question")
        question_el.set("type", "multichoice")

        # ── Name (truncated for Moodle display) ─
        name_el       = ET.SubElement(question_el, "name")
        name_text_el  = ET.SubElement(name_el, "text")
        name_text_el.text = q.question_text[:100].replace("\n", " ")

        # ── Question body ──────────────────────
        qtext_el = ET.SubElement(question_el, "questiontext")
        qtext_el.set("format", "html")
        qtext_inner      = ET.SubElement(qtext_el, "text")
        qtext_inner.text = q.question_text

        # ── General feedback (explanation) ─────
        gfb_el      = ET.SubElement(question_el, "generalfeedback")
        gfb_el.set("format", "html")
        gfb_text_el = ET.SubElement(gfb_el, "text")
        gfb_text_el.text = q.explanation

        # ── Default grade ──────────────────────
        grade_el      = ET.SubElement(question_el, "defaultgrade")
        grade_el.text = "1.0000000"

        # ── Penalty for wrong answer ───────────
        penalty_el      = ET.SubElement(question_el, "penalty")
        penalty_el.text = "0.3333333"

        # ── Hidden flag ───────────────────────
        hidden_el      = ET.SubElement(question_el, "hidden")
        hidden_el.text = "0"

        # ── Single vs multiple ─────────────────
        single_el      = ET.SubElement(question_el, "single")
        single_el.text = (
            "true" if q.question_type.value == "single" else "false"
        )

        # ── Shuffle answers ────────────────────
        shuffle_el      = ET.SubElement(question_el, "shuffleanswers")
        shuffle_el.text = "true"

        # ── Answer numbering ───────────────────
        numbering_el      = ET.SubElement(question_el, "answernumbering")
        numbering_el.text = "ABCDE"

        # ── Answer options ─────────────────────
        for opt in q.options:
            # Expected format: "A. <text>" or "A) <text>"
            parts  = re.split(r"[.)]\s*", opt, maxsplit=1)
            letter = parts[0].strip().upper() if parts else "?"
            text   = parts[1].strip() if len(parts) > 1 else opt

            fraction = _fraction_for_option(
                letter,
                q.correct_answers,
                q.question_type.value,
            )

            answer_el = ET.SubElement(question_el, "answer")
            answer_el.set("fraction", fraction)
            answer_el.set("format", "html")

            answer_text_el      = ET.SubElement(answer_el, "text")
            answer_text_el.text = text

            # Per-answer feedback
            feedback_el      = ET.SubElement(answer_el, "feedback")
            feedback_el.set("format", "html")
            feedback_text_el = ET.SubElement(feedback_el, "text")
            feedback_text_el.text = (
                "Correct." if letter in q.correct_answers else "Incorrect."
            )

    # ── Write file ─────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    xml_string = _prettify_xml(quiz)

    with output_path.open("w", encoding="utf-8") as fh:
        fh.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        # toprettyxml already adds an xml declaration; strip the duplicate
        lines = xml_string.splitlines()
        if lines and lines[0].startswith("<?xml"):
            lines = lines[1:]
        fh.write("\n".join(lines))

    logger.info(f"Moodle XML export written → {output_path}")
    return output_path


# ─────────────────────────────────────────────
# Console display helpers
# ─────────────────────────────────────────────

def print_question_table(questions: List[ExamQuestion]) -> None:
    """
    Render a Rich table summarising a batch of questions.
    Useful for quick CLI review without opening the export file.
    """
    table = Table(
        title="Generated Questions Summary",
        show_header=True,
        header_style="bold magenta",
        show_lines=True,
    )

    table.add_column("#",              style="dim",    width=4)
    table.add_column("Type",           style="cyan",   width=10)
    table.add_column("Domain",         style="green",  width=36)
    table.add_column("Status",         style="yellow", width=10)
    table.add_column("Revisions",      style="red",    width=10)
    table.add_column("Question (truncated)", width=50)

    for idx, q in enumerate(questions, start=1):
        status_color = {
            ValidationStatus.APPROVED: "[green]Approved[/green]",
            ValidationStatus.REJECTED: "[red]Rejected[/red]",
            ValidationStatus.PENDING:  "[yellow]Pending[/yellow]",
        }.get(q.validation_status, q.validation_status.value)

        table.add_row(
            str(idx),
            q.question_type.value,
            q.domain[:34],
            status_color,
            str(q.revision_count),
            q.question_text[:48] + "…",
        )

    console.print(table)


def print_generation_summary(result: GenerationResult) -> None:
    """
    Print a Rich panel summarising the full generation run.
    Called at the end of `main.py generate`.
    """
    approved_pct = (
        round(result.approved / result.requested * 100, 1)
        if result.requested > 0
        else 0.0
    )

    summary = (
        f"[bold]Certification:[/bold]  {result.cert_code}\n"
        f"[bold]Requested:[/bold]      {result.requested}\n"
        f"[bold]Approved:[/bold]       [green]{result.approved}[/green] "
        f"({approved_pct} %)\n"
        f"[bold]Rejected:[/bold]       [red]{result.rejected}[/red]\n"
        f"[bold]Duration:[/bold]       {result.duration_seconds:.1f}s\n"
        f"[bold]Output file:[/bold]    {result.output_file or 'N/A'}"
    )

    console.print(
        Panel(
            summary,
            title="[bold blue]Generation Complete[/bold blue]",
            border_style="blue",
            expand=False,
        )
    )


def print_domain_alignment_table(domain_stats: Dict[str, Dict[str, Any]]) -> None:
    """
    Render a Rich table comparing each domain's actual share of
    *approved* questions in this run against its target weight from
    registry.json (the exam guide's content outline percentages).

    domain_stats entries are expected to carry "approved", "target_pct",
    "actual_pct", and "delta_pct" — see generator.run_generation_pipeline.
    """
    if not domain_stats:
        return

    table = Table(
        title="Content Outline Alignment (Approved Questions)",
        show_header=True,
        header_style="bold magenta",
        show_lines=True,
    )
    table.add_column("Domain",     style="green", width=40)
    table.add_column("Approved",   style="cyan",  width=10, justify="right")
    table.add_column("Target %",   style="blue",  width=10, justify="right")
    table.add_column("Actual %",   style="yellow",width=10, justify="right")
    table.add_column("Δ",          width=8,       justify="right")

    for domain, stats in domain_stats.items():
        delta = stats.get("delta_pct", 0.0)
        delta_style = "red" if abs(delta) >= 5.0 else "dim"
        table.add_row(
            domain,
            str(stats.get("approved", 0)),
            f"{stats.get('target_pct', 0.0):.1f}",
            f"{stats.get('actual_pct', 0.0):.1f}",
            f"[{delta_style}]{delta:+.1f}[/{delta_style}]",
        )

    console.print(table)


def print_validation_result(
    question_id: str,
    passed: bool,
    score: float,
    flags: List[Dict[str, Any]],
) -> None:
    """
    Print a one-line Rich log entry for each validation outcome.
    Called inside the reviewer loop for real-time feedback.
    """
    icon   = "[green]✔[/green]" if passed else "[red]✘[/red]"
    status = "[green]PASSED[/green]" if passed else "[red]FAILED[/red]"

    flag_summary = (
        ", ".join(f["field"] for f in flags) if flags else "none"
    )

    console.print(
        f"  {icon} {status} | score={score:.1f}/10 | "
        f"id={question_id[:8]}… | flags=[{flag_summary}]"
    )


# ─────────────────────────────────────────────
# Miscellaneous utilities
# ─────────────────────────────────────────────

def build_output_filename(
    cert_code: str,
    count: int,
    fmt: str,
    output_dir: Path,
) -> Path:
    """
    Build a timestamped output filename in output_dir.

    Example: output/SAA-C03_10q_20240315T143022.json
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    stem      = f"{cert_code}_{count}q_{timestamp}"
    extension = {
        "json":       ".json",
        "csv":        ".csv",
        "moodle_xml": ".xml",
    }.get(fmt, ".json")

    return output_dir / f"{stem}{extension}"


def load_registry(registry_path: Path) -> Dict[str, Any]:
    """
    Load and parse the certification registry JSON.
    Raises FileNotFoundError with a helpful message if missing.
    """
    if not registry_path.exists():
        raise FileNotFoundError(
            f"Registry file not found at {registry_path}. "
            "Ensure registry.json is present in the project root."
        )
    with registry_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def get_cert_metadata(
    cert_code: str,
    registry: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Retrieve metadata for a specific certification code.
    Raises ValueError if the cert_code is not found in the registry.
    """
    certs = registry.get("certifications", {})
    if cert_code not in certs:
        available = ", ".join(sorted(certs.keys()))
        raise ValueError(
            f"Certification '{cert_code}' not found in registry. "
            f"Available codes: {available}"
        )
    return certs[cert_code]


def select_domain_by_quota(
    domain_weights: Dict[str, float],
    domain_counts: Dict[str, int],
    total: int,
    domain_filter: Optional[str] = None,
) -> str:
    """
    Select the domain that is currently most "behind" its proportional
    target, so the domain mix converges on registry.json's domain_weights
    (i.e. the exam guide's content outline percentages) even in small
    batches — mirroring the deficit-tracking allocation already used for
    the single/multiple answer-type split (see generator._select_question_type).

    If domain_filter is provided and matches a known domain, that domain
    is returned directly (ignoring weights/quotas).

    Parameters
    ──────────
    domain_weights
        Target proportion per domain (must sum to ~1.0).
    domain_counts
        Running count of questions already generated per domain in this
        request (across the current round and any prior top-up rounds).
    total
        The overall requested question count the quotas are computed
        against (not just the slots remaining in this round).
    """
    import random

    if domain_filter:
        known = list(domain_weights.keys())
        for d in known:
            if domain_filter.lower() in d.lower():
                return d
        logger.warning(
            f"Domain filter '{domain_filter}' did not match any known domain. "
            "Falling back to quota-based selection."
        )

    deficits = {
        domain: (weight * total) - domain_counts.get(domain, 0)
        for domain, weight in domain_weights.items()
    }
    max_deficit = max(deficits.values())

    # Tie-break (common early on, e.g. all counts at 0) via weighted
    # random among the domains sharing the largest deficit.
    candidates = [d for d, dv in deficits.items() if dv >= max_deficit - 1e-9]
    if len(candidates) == 1:
        return candidates[0]

    weights = [domain_weights[d] for d in candidates]
    return random.choices(candidates, weights=weights, k=1)[0]


def chunk_list(lst: List[Any], size: int) -> List[List[Any]]:
    """Split a list into sub-lists of at most `size` items."""
    return [lst[i : i + size] for i in range(0, len(lst), size)]


def truncate_text(text: str, max_chars: int = 200) -> str:
    """Truncate text to max_chars, appending '…' if truncated."""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"


def count_tokens_approx(text: str) -> int:
    """
    Rough token count estimate: ~4 characters per token.
    Used to guard against prompts that exceed num_ctx.
    Not a replacement for a proper tokenizer.
    """
    return max(1, len(text) // 4)
