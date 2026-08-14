"""
reviewer.py
───────────
AI Reviewer & Validator Loop

Implements an independent "Principal AWS Solutions Architect Exam Reviewer"
persona that evaluates each generated ExamQuestion against strict criteria
before it is approved for export.

Validation pipeline per question:
  1.  Structural checks  (fast, local, no LLM call)
  2.  LLM deep-review    (Ollama prompt → ValidationCritique JSON)
  3.  Feedback loop      (up to max_revision_attempts if failed)
  4.  Final status stamp (Approved / Rejected)

The reviewer is intentionally kept stateless between questions so it can
be run sequentially without accumulating VRAM pressure on the GTX 1070.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_fixed,
)

from config import GENERATION_CONFIG, OLLAMA_CONFIG
from schemas import (
    CritiqueFlag,
    ExamQuestion,
    QuestionType,
    ValidationCritique,
    ValidationStatus,
)
from utils import (
    console,
    extract_json_from_text,
    logger,
    print_validation_result,
    truncate_text,
)

# ─────────────────────────────────────────────
# Module-level logger
# ─────────────────────────────────────────────
log = logging.getLogger("aws_exam_gen.reviewer")


# ─────────────────────────────────────────────
# Validation thresholds
# ─────────────────────────────────────────────

# Minimum overall score (out of 10) required for automatic approval
APPROVAL_SCORE_THRESHOLD: float = 7.0

# Any flag with this severity causes immediate rejection regardless of score
CRITICAL_FLAG_SEVERITY: str = "critical"

# Structural checks are run without an LLM call to save VRAM cycles
STRUCTURAL_CHECKS_ENABLED: bool = True


# ─────────────────────────────────────────────
# Reviewer prompt templates
# ─────────────────────────────────────────────

REVIEWER_SYSTEM_PROMPT = """\
You are a Principal AWS Solutions Architect with 15+ years of experience \
and a certified AWS exam item writer. Your role is to critically evaluate \
multiple-choice exam questions intended for official AWS certification practice \
exams.

You review each question against the following criteria:

1. TECHNICAL_ACCURACY
   - Are all AWS service names, API calls, limits, and behaviours factually \
correct as of the current AWS documentation?
   - Are architectural patterns consistent with AWS Well-Architected Framework \
best practices?
   - Do the correct answers actually solve the stated problem optimally?

2. FORMAT_COMPLIANCE
   - Single-answer questions must have EXACTLY 4 options and EXACTLY 1 \
correct answer.
   - Multiple-selection questions must have EXACTLY 5 options, 2 or more \
correct answers, and the phrase "Select TWO" / "Select THREE" (etc.) must \
appear explicitly in the question text.
   - Option labels must follow the pattern: "A. <text>", "B. <text>", etc.

3. DISTRACTOR_QUALITY
   - Are incorrect options plausible enough to challenge a candidate who \
has partial knowledge?
   - Are there any "gimme" distractors (obviously wrong, absurd, or unrelated \
to the domain)?
   - Do distractors avoid being trick questions or relying on obscure trivia \
not covered by the exam blueprint?

4. SCENARIO_QUALITY
   - Is the question scenario realistic and grounded in a genuine AWS \
workload situation?
   - Does the question test understanding rather than pure memorisation?
   - Is the question free from ambiguous wording that could make multiple \
answers defensible?

5. EXPLANATION_QUALITY
   - Does the explanation clearly justify WHY each correct answer is right?
   - Does the explanation clearly justify WHY each distractor is wrong?
   - Does the explanation cite specific AWS services, features, or \
documentation references?

Respond ONLY with a valid JSON object. Do NOT include any prose, markdown \
fences, or commentary outside the JSON structure.
"""

REVIEWER_USER_PROMPT_TEMPLATE = """\
Review the following AWS certification exam question and return your \
evaluation as a JSON object matching the schema below.

─── QUESTION TO REVIEW ───────────────────────────────────────────────────────
Certification : {cert_code}
Domain        : {domain}
Type          : {question_type}
Question Text : {question_text}
Options       :
{options_formatted}
Correct Answer(s): {correct_answers}
Explanation   : {explanation}
──────────────────────────────────────────────────────────────────────────────

─── REQUIRED JSON RESPONSE SCHEMA ────────────────────────────────────────────
{{
  "question_id": "{question_id}",
  "passed": <true | false>,
  "overall_score": <float 0.0–10.0>,
  "flags": [
    {{
      "field": "<which aspect failed: e.g. TECHNICAL_ACCURACY>",
      "severity": "<critical | major | minor>",
      "description": "<what is wrong>",
      "suggestion": "<specific fix the generator should apply>"
    }}
  ],
  "reviewer_comments": "<overall free-text summary of the review>"
}}

Scoring guide:
  9.0–10.0 → Excellent, approve immediately.
  7.0–8.9  → Good, minor issues acceptable, approve.
  5.0–6.9  → Significant issues, requires revision.
  0.0–4.9  → Fundamental flaws, likely reject after revisions.

A question PASSES (passed=true) only if:
  - overall_score >= {approval_threshold}
  - There are zero flags with severity == "critical"

Return ONLY the JSON object.
"""

REVISION_INSTRUCTION_TEMPLATE = """\
The following AWS exam question FAILED validation review.
Apply ALL of the requested corrections listed in the critique flags and \
regenerate the question in the same JSON format as before.

─── ORIGINAL QUESTION ────────────────────────────────────────────────────────
Certification : {cert_code}
Domain        : {domain}
Type          : {question_type}
Question Text : {question_text}
Options       :
{options_formatted}
Correct Answer(s): {correct_answers}
Explanation   : {explanation}

─── REVIEWER CRITIQUE ────────────────────────────────────────────────────────
Overall Score : {overall_score}/10
Comments      : {reviewer_comments}

Flags to fix:
{flags_formatted}
──────────────────────────────────────────────────────────────────────────────

Return ONLY a valid JSON object for the corrected question matching this \
schema exactly:
{{
  "cert_code": "{cert_code}",
  "domain": "{domain}",
  "question_type": "{question_type}",
  "question_text": "<revised scenario text>",
  "options": ["A. ...", "B. ...", "C. ...", "D. ...<E. ... if multiple>"],
  "correct_answers": ["<letter>", ...],
  "explanation": "<revised detailed explanation>",
  "aws_services": ["<service1>", ...]
}}
"""


# ─────────────────────────────────────────────
# Structural (non-LLM) validation helpers
# ─────────────────────────────────────────────

def _check_option_labels(options: List[str]) -> List[CritiqueFlag]:
    """
    Verify every option starts with a valid letter label.
    Expected format: "A. text", "B. text", etc.
    Returns a list of CritiqueFlag objects for any violations.
    """
    flags: List[CritiqueFlag] = []
    expected_letters = ["A", "B", "C", "D", "E"][: len(options)]

    for idx, opt in enumerate(options):
        if not opt or "." not in opt:
            flags.append(CritiqueFlag(
                field="FORMAT_COMPLIANCE",
                severity="critical",
                description=(
                    f"Option at index {idx} ('{truncate_text(opt, 40)}') "
                    "does not follow the required 'X. <text>' label format."
                ),
                suggestion=(
                    f"Reformat as '{expected_letters[idx]}. <option text>'."
                ),
            ))
            continue

        label = opt.split(".")[0].strip().upper()
        if idx < len(expected_letters) and label != expected_letters[idx]:
            flags.append(CritiqueFlag(
                field="FORMAT_COMPLIANCE",
                severity="major",
                description=(
                    f"Option label '{label}' at position {idx} "
                    f"should be '{expected_letters[idx]}'."
                ),
                suggestion=(
                    f"Relabel this option as "
                    f"'{expected_letters[idx]}. <text>'."
                ),
            ))

    return flags


def _check_correct_answers_in_options(
    options: List[str],
    correct_answers: List[str],
) -> List[CritiqueFlag]:
    """
    Ensure every correct answer letter maps to an actual option.
    """
    flags: List[CritiqueFlag] = []
    option_letters = {
        opt.split(".")[0].strip().upper()
        for opt in options
        if opt and "." in opt
    }
    for ans in correct_answers:
        if ans.upper() not in option_letters:
            flags.append(CritiqueFlag(
                field="FORMAT_COMPLIANCE",
                severity="critical",
                description=(
                    f"Correct answer '{ans}' does not correspond "
                    f"to any option letter in {sorted(option_letters)}."
                ),
                suggestion=(
                    "Ensure every correct_answer letter exists as an "
                    "option label (A–E)."
                ),
            ))
    return flags


def _check_multiple_selection_phrase(
    question_text: str,
    correct_answers: List[str],
) -> List[CritiqueFlag]:
    """
    Multiple-selection questions MUST explicitly state how many answers
    the candidate should select (e.g. 'Select TWO').
    """
    flags: List[CritiqueFlag] = []
    count_words = {2: "TWO", 3: "THREE", 4: "FOUR"}
    expected = count_words.get(len(correct_answers), str(len(correct_answers)))

    if f"select {expected}".lower() not in question_text.lower():
        flags.append(CritiqueFlag(
            field="FORMAT_COMPLIANCE",
            severity="critical",
            description=(
                f"Multiple-selection question with {len(correct_answers)} "
                f"correct answers must contain the phrase "
                f"'Select {expected}' in the question text."
            ),
            suggestion=(
                f"Add '(Select {expected})' at the end of the question stem."
            ),
        ))
    return flags


def _check_explanation_length(explanation: str) -> List[CritiqueFlag]:
    """Flag explanations that are too short to be meaningful."""
    flags: List[CritiqueFlag] = []
    if len(explanation.strip()) < 80:
        flags.append(CritiqueFlag(
            field="EXPLANATION_QUALITY",
            severity="major",
            description=(
                f"Explanation is too short ({len(explanation)} chars). "
                "It must justify correct answers AND dismiss distractors."
            ),
            suggestion=(
                "Expand the explanation to cover why each option is correct "
                "or incorrect, citing specific AWS service behaviour."
            ),
        ))
    return flags


def _check_question_text_length(question_text: str) -> List[CritiqueFlag]:
    """Flag question stems that are too brief to be scenario-based."""
    flags: List[CritiqueFlag] = []
    if len(question_text.strip()) < 40:
        flags.append(CritiqueFlag(
            field="SCENARIO_QUALITY",
            severity="major",
            description=(
                f"Question text is too short ({len(question_text)} chars) "
                "to represent a realistic AWS workload scenario."
            ),
            suggestion=(
                "Expand the scenario with context: company size, workload "
                "type, requirements (cost, HA, security), and constraints."
            ),
        ))
    return flags


def run_structural_checks(question: ExamQuestion) -> List[CritiqueFlag]:
    """
    Execute all local (non-LLM) structural checks on a question.
    Returns a combined list of CritiqueFlag objects.
    These checks are fast and free — they run before any LLM call.
    """
    flags: List[CritiqueFlag] = []

    flags.extend(_check_option_labels(question.options))
    flags.extend(
        _check_correct_answers_in_options(
            question.options, question.correct_answers
        )
    )
    flags.extend(_check_explanation_length(question.explanation))
    flags.extend(_check_question_text_length(question.question_text))

    if question.question_type == QuestionType.MULTIPLE_SELECTION:
        flags.extend(
            _check_multiple_selection_phrase(
                question.question_text, question.correct_answers
            )
        )

    return flags


# ─────────────────────────────────────────────
# Prompt construction helpers
# ─────────────────────────────────────────────

def _format_options(options: List[str]) -> str:
    """Return options as a numbered block for prompt embedding."""
    return "\n".join(f"  {opt}" for opt in options)


def _format_flags_for_revision(flags: List[CritiqueFlag]) -> str:
    """
    Render critique flags as a numbered instruction list
    for the revision prompt so the generator knows exactly what to fix.
    """
    if not flags:
        return "  (No specific flags — improve overall quality.)"

    lines = []
    for i, flag in enumerate(flags, start=1):
        lines.append(
            f"  [{i}] Field    : {flag.field}\n"
            f"      Severity : {flag.severity.upper()}\n"
            f"      Problem  : {flag.description}\n"
            f"      Fix      : {flag.suggestion}"
        )
    return "\n\n".join(lines)


def build_review_prompt(question: ExamQuestion) -> str:
    """Assemble the full user-turn prompt for the LLM review call."""
    return REVIEWER_USER_PROMPT_TEMPLATE.format(
        cert_code         = question.cert_code,
        domain            = question.domain,
        question_type     = question.question_type.value,
        question_text     = question.question_text,
        options_formatted = _format_options(question.options),
        correct_answers   = ", ".join(question.correct_answers),
        explanation       = question.explanation,
        question_id       = question.question_id,
        approval_threshold= APPROVAL_SCORE_THRESHOLD,
    )


def build_revision_prompt(
    question: ExamQuestion,
    critique: ValidationCritique,
) -> str:
    """
    Assemble the revision instruction prompt that is fed back to the
    generator when a question fails review.
    """
    return REVISION_INSTRUCTION_TEMPLATE.format(
        cert_code         = question.cert_code,
        domain            = question.domain,
        question_type     = question.question_type.value,
        question_text     = question.question_text,
        options_formatted = _format_options(question.options),
        correct_answers   = ", ".join(question.correct_answers),
        explanation       = question.explanation,
        overall_score     = critique.overall_score,
        reviewer_comments = critique.reviewer_comments,
        flags_formatted   = _format_flags_for_revision(critique.flags),
    )


# ─────────────────────────────────────────────
# LLM call with retry
# ─────────────────────────────────────────────

@retry(
    retry     = retry_if_exception_type((ConnectionError, TimeoutError)),
    stop      = stop_after_attempt(3),
    wait      = wait_fixed(2),
    reraise   = True,
)
def _call_ollama_reviewer(
    user_prompt: str,
    ollama_client: Any,
) -> str:
    """
    Send the review prompt to Ollama and return the raw response string.
    Uses the same model and context settings as the generator to avoid
    loading a second model into VRAM on the GTX 1070.
    Retries up to 3 times on transient connection / timeout errors.
    """
    response = ollama_client.chat(
        model   = OLLAMA_CONFIG.model,
        messages= [
            {"role": "system", "content": REVIEWER_SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        options = {
            "temperature" : 0.1,      # Low temp → deterministic critique
            "top_p"       : 0.9,
            "num_ctx"     : OLLAMA_CONFIG.num_ctx,
            "num_predict" : 1024,     # Critique JSON is shorter than a question
        },
        keep_alive= OLLAMA_CONFIG.keep_alive,
    )
    return response["message"]["content"]


# ─────────────────────────────────────────────
# Critique parsing
# ─────────────────────────────────────────────

def _parse_critique_response(
    raw_text: str,
    question_id: str,
) -> Optional[ValidationCritique]:
    """
    Parse the LLM's raw text response into a ValidationCritique object.
    Returns None if parsing fails completely (triggers a retry upstream).
    """
    data = extract_json_from_text(raw_text)
    if data is None:
        log.warning(
            f"[{question_id[:8]}] Reviewer returned unparseable response."
        )
        return None

    try:
        # Normalise the question_id in case the LLM echoed it differently
        data["question_id"] = question_id

        # Coerce flags into CritiqueFlag objects
        raw_flags = data.get("flags", [])
        parsed_flags: List[CritiqueFlag] = []
        for rf in raw_flags:
            try:
                parsed_flags.append(CritiqueFlag(**rf))
            except Exception as flag_err:
                log.debug(
                    f"Skipping malformed flag entry: {rf} — {flag_err}"
                )

        critique = ValidationCritique(
            question_id      = question_id,
            passed           = bool(data.get("passed", False)),
            overall_score    = float(data.get("overall_score", 0.0)),
            flags            = parsed_flags,
            reviewer_comments= str(data.get("reviewer_comments", "")),
        )
        return critique

    except Exception as parse_err:
        log.warning(
            f"[{question_id[:8]}] Could not construct ValidationCritique: "
            f"{parse_err}"
        )
        return None


def _has_critical_flag(critique: ValidationCritique) -> bool:
    """Return True if any flag is severity == 'critical'."""
    return any(
        f.severity.lower() == CRITICAL_FLAG_SEVERITY
        for f in critique.flags
    )


def _critique_passes(critique: ValidationCritique) -> bool:
    """
    A critique passes if:
      - overall_score >= APPROVAL_SCORE_THRESHOLD
      - No flags have severity == 'critical'
    This mirrors the logic described in the reviewer system prompt.
    """
    if critique.overall_score < APPROVAL_SCORE_THRESHOLD:
        return False
    if _has_critical_flag(critique):
        return False
    return True


# ─────────────────────────────────────────────
# Question patching from revision JSON
# ─────────────────────────────────────────────

def _apply_revision_to_question(
    original: ExamQuestion,
    revision_data: Dict[str, Any],
) -> ExamQuestion:
    """
    Merge the generator's revision JSON back into an ExamQuestion.

    Preserves the original question_id, cert_code, domain, and
    revision_count (incremented by 1) so the pipeline can track
    how many revision cycles were consumed.

    Raises ValueError if the revision data cannot produce a valid
    ExamQuestion (caught upstream to trigger rejection).
    """
    merged = original.model_dump()

    # Fields the generator is allowed to revise
    updatable_fields = [
        "question_text",
        "options",
        "correct_answers",
        "explanation",
        "aws_services",
        "question_type",
    ]

    for field in updatable_fields:
        if field in revision_data and revision_data[field]:
            merged[field] = revision_data[field]

    # Normalise correct_answers to uppercase strings
    merged["correct_answers"] = [
        a.upper() for a in merged.get("correct_answers", [])
    ]

    # Carry forward identity fields – never allow LLM to change these
    merged["question_id"]    = original.question_id
    merged["cert_code"]      = original.cert_code
    merged["domain"]         = original.domain
    merged["revision_count"] = original.revision_count + 1

    # Reset validation status for re-evaluation
    merged["validation_status"] = ValidationStatus.PENDING.value

    return ExamQuestion(**merged)


# ─────────────────────────────────────────────
# Core review function
# ─────────────────────────────────────────────

def review_question(
    question: ExamQuestion,
    ollama_client: Any,
    revision_callback: Optional[Callable[[ExamQuestion, ValidationCritique], ExamQuestion]] = None,
) -> Tuple[ExamQuestion, ValidationCritique]:
    """
    Run the full review cycle for a single ExamQuestion.

    Steps:
      1. Run fast structural checks (no LLM).
      2. If structural checks contain critical flags → build a synthetic
         ValidationCritique and invoke the revision callback immediately.
      3. Call the LLM reviewer for deep semantic validation.
      4. If the question fails and revision_callback is provided,
         invoke it to get a corrected question (up to max_revision_attempts).
      5. Stamp the question with Approved / Rejected status.

    Parameters
    ──────────
    question
        The ExamQuestion to review.
    ollama_client
        An initialised ollama.Client instance (injected for testability).
    revision_callback
        Optional callable that accepts (question, critique) and returns
        a revised ExamQuestion.  When None, no revisions are attempted.

    Returns
    ───────
    (final_question, final_critique)
        The question with updated validation_status and the last critique.
    """
    max_attempts = GENERATION_CONFIG.max_revision_attempts
    current_q    = question

    for attempt in range(max_attempts + 1):
        attempt_label = (
            "Initial review"
            if attempt == 0
            else f"Revision {attempt}/{max_attempts}"
        )
        log.debug(
            f"[{current_q.question_id[:8]}] {attempt_label} starting…"
        )

        # ── Step 1: Structural checks ──────────────
        if STRUCTURAL_CHECKS_ENABLED:
            structural_flags = run_structural_checks(current_q)
        else:
            structural_flags = []

        # If structural issues are critical and we haven't exhausted attempts,
        # synthesise a critique and jump straight to revision
        if structural_flags:
            has_critical = any(
                f.severity.lower() == CRITICAL_FLAG_SEVERITY
                for f in structural_flags
            )
            structural_score = 4.0 if has_critical else 6.0

            structural_critique = ValidationCritique(
                question_id      = current_q.question_id,
                passed           = False,
                overall_score    = structural_score,
                flags            = structural_flags,
                reviewer_comments= (
                    "Question failed automated structural checks before "
                    "reaching the LLM reviewer."
                ),
            )

            print_validation_result(
                question_id = current_q.question_id,
                passed      = False,
                score       = structural_score,
                flags       = [f.model_dump() for f in structural_flags],
            )

            if attempt < max_attempts and revision_callback:
                log.info(
                    f"[{current_q.question_id[:8]}] Structural failure — "
                    f"requesting revision {attempt + 1}."
                )
                try:
                    current_q = revision_callback(current_q, structural_critique)
                except Exception as rev_err:
                    log.error(
                        f"Revision callback failed on structural critique: "
                        f"{rev_err}"
                    )
                    # Cannot revise — reject immediately
                    current_q.validation_status = ValidationStatus.REJECTED
                    return current_q, structural_critique
                continue  # Re-enter loop with revised question
            else:
                # No revision possible or attempts exhausted
                current_q.validation_status = ValidationStatus.REJECTED
                return current_q, structural_critique

        # ── Step 2: LLM deep review ─────────────
        critique: Optional[ValidationCritique] = None
        raw_response: str = ""

        try:
            review_prompt = build_review_prompt(current_q)
            raw_response  = _call_ollama_reviewer(review_prompt, ollama_client)
            critique      = _parse_critique_response(
                raw_response, current_q.question_id
            )
        except Exception as llm_err:
            log.error(
                f"[{current_q.question_id[:8]}] LLM reviewer call failed: "
                f"{llm_err}"
            )

        # If LLM call or parse failed, build a fallback "uncertain" critique
        if critique is None:
            critique = ValidationCritique(
                question_id      = current_q.question_id,
                passed           = False,
                overall_score    = 5.0,
                flags            = [CritiqueFlag(
                    field       = "REVIEWER_ERROR",
                    severity    = "major",
                    description = (
                        "The AI reviewer could not parse its own response. "
                        f"Raw response snippet: "
                        f"'{truncate_text(raw_response, 120)}'"
                    ),
                    suggestion  = (
                        "Retry generation — this is likely a transient "
                        "model formatting issue."
                    ),
                )],
                reviewer_comments= "LLM reviewer response was unparseable.",
            )

        # ── Step 3: Evaluate critique outcome ──────
        passed = _critique_passes(critique)

        print_validation_result(
            question_id = current_q.question_id,
            passed      = passed,
            score       = critique.overall_score,
            flags       = [f.model_dump() for f in critique.flags],
        )

        if passed:
            current_q.validation_status = ValidationStatus.APPROVED
            log.info(
                f"[{current_q.question_id[:8]}] ✔ APPROVED "
                f"(score={critique.overall_score:.1f})"
            )
            return current_q, critique

        # ── Step 4: Request revision if attempts remain ──
        if attempt < max_attempts and revision_callback:
            log.info(
                f"[{current_q.question_id[:8]}] ✘ FAILED "
                f"(score={critique.overall_score:.1f}) — "
                f"requesting revision {attempt + 1}/{max_attempts}."
            )
            try:
                current_q = revision_callback(current_q, critique)
            except Exception as rev_err:
                log.error(
                    f"[{current_q.question_id[:8]}] Revision callback raised "
                    f"an exception: {rev_err}"
                )
                # Cannot produce a valid revision — reject
                current_q.validation_status = ValidationStatus.REJECTED
                return current_q, critique
        else:
            # Attempts exhausted or no callback — reject
            log.warning(
                f"[{current_q.question_id[:8]}] ✘ REJECTED after "
                f"{attempt} attempt(s)."
            )
            current_q.validation_status = ValidationStatus.REJECTED
            return current_q, critique

    # Safety net — should not reach here, but reject if loop exits cleanly
    current_q.validation_status = ValidationStatus.REJECTED
    final_critique = ValidationCritique(
        question_id      = current_q.question_id,
        passed           = False,
        overall_score    = 0.0,
        flags            = [],
        reviewer_comments= "Max revision attempts exhausted with no approval.",
    )
    return current_q, final_critique


# ─────────────────────────────────────────────
# Batch review orchestrator
# ─────────────────────────────────────────────

def review_batch(
    questions: List[ExamQuestion],
    ollama_client: Any,
    revision_callback: Optional[Callable[[ExamQuestion, ValidationCritique], ExamQuestion]] = None,
    on_progress: Optional[Callable[[int, int, ExamQuestion], None]] = None,
) -> Tuple[List[ExamQuestion], List[ValidationCritique]]:
    """
    Review a batch of questions sequentially.

    Sequential execution is intentional: the GTX 1070 (8 GB VRAM) cannot
    sustain concurrent Ollama inference without OOM risk.

    Parameters
    ──────────
    questions
        List of ExamQuestion objects to review.
    ollama_client
        Initialised Ollama client (shared with the generator).
    revision_callback
        Callable used to request a revised question from the generator
        when review fails.  Signature: (question, critique) → ExamQuestion.
    on_progress
        Optional callback invoked after each question is reviewed.
        Signature: (current_index, total, reviewed_question) → None.
        Useful for updating a Rich progress bar in main.py.

    Returns
    ───────
    (reviewed_questions, critiques)
        Parallel lists: reviewed_questions[i] corresponds to critiques[i].
    """
    reviewed:  List[ExamQuestion]       = []
    critiques: List[ValidationCritique] = []
    total      = len(questions)

    log.info(f"Starting batch review of {total} question(s)…")

    for idx, question in enumerate(questions):
        log.debug(
            f"Reviewing question {idx + 1}/{total} "
            f"[{question.question_id[:8]}]…"
        )

        t_start = time.monotonic()

        reviewed_q, critique = review_question(
            question          = question,
            ollama_client     = ollama_client,
            revision_callback = revision_callback,
        )

        elapsed = time.monotonic() - t_start
        log.debug(
            f"[{reviewed_q.question_id[:8]}] Review completed in "
            f"{elapsed:.1f}s — status: {reviewed_q.validation_status.value}"
        )

        reviewed.append(reviewed_q)
        critiques.append(critique)

        if on_progress:
            try:
                on_progress(idx + 1, total, reviewed_q)
            except Exception as cb_err:
                log.debug(f"on_progress callback error (non-fatal): {cb_err}")

    approved_count = sum(
        1 for q in reviewed
        if q.validation_status == ValidationStatus.APPROVED
    )
    rejected_count = total - approved_count

    log.info(
        f"Batch review complete — "
        f"Approved: {approved_count}/{total}, "
        f"Rejected: {rejected_count}/{total}."
    )

    return reviewed, critiques


# ─────────────────────────────────────────
