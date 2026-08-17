"""
schemas.py
──────────
Pydantic v2 data models for the entire pipeline.

ExamQuestion  – canonical representation of one MCQ
ValidationCritique – structured output from the AI Reviewer
GenerationRequest  – parameters passed to the generator
GenerationResult   – batch result returned to the CLI
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from text_utils import fix_shouting_caps


# ─────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────

class QuestionType(str, Enum):
    SINGLE_ANSWER      = "single"
    MULTIPLE_SELECTION = "multiple"


class ValidationStatus(str, Enum):
    PENDING  = "Pending"
    APPROVED = "Approved"
    REJECTED = "Rejected"


class ExportFormat(str, Enum):
    JSON      = "json"
    CSV       = "csv"
    MOODLE    = "moodle_xml"


# ─────────────────────────────────────────────
# Core question model
# ─────────────────────────────────────────────

class ExamQuestion(BaseModel):
    """
    Canonical representation of one AWS certification MCQ.
    Supports both single-answer (1 correct) and
    multiple-selection (≥2 correct) formats.
    """

    # ── Identity ──────────────────────────────
    question_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this question."
    )
    cert_code: str = Field(
        ...,
        description="AWS certification code, e.g. 'SAA-C03'.",
        examples=["SAA-C03", "DVA-C02", "SCS-C03"]
    )
    domain: str = Field(
        ...,
        description="Exam guide domain objective this question targets."
    )

    # ── Content ───────────────────────────────
    question_type: QuestionType = Field(
        ...,
        description="'single' for one correct answer, 'multiple' for two or more."
    )
    question_text: str = Field(
        ...,
        description=(
            "Scenario-based problem description. "
            "Must clearly state 'Select TWO' / 'Select THREE' for multiple-selection."
        ),
        min_length=20
    )
    options: List[str] = Field(
        ...,
        description=(
            "Lettered answer choices, e.g. ['A. Use Amazon S3 ...', 'B. Use EFS ...']."
            " Single-answer: exactly 4. Multiple-selection: exactly 5."
        )
    )
    correct_answers: List[str] = Field(
        ...,
        description="Correct option letters, e.g. ['A'] or ['B', 'D']."
    )
    explanation: str = Field(
        ...,
        description=(
            "Detailed breakdown: why each correct answer is right, "
            "why each distractor is wrong. Must cite AWS documentation or FAQs."
        ),
        min_length=50
    )
    source_context: Optional[str] = Field(
        default=None,
        description="Optional snippet of RAG context used to generate this question.",
    )
    source_url: Optional[str] = Field(
        default=None,
        description=(
            "Optional URL to the source article or page used to generate "
            "this question."
        ),
    )

    # ── Metadata ─────────────────────────────
    difficulty: Optional[str] = Field(
        None,
        description="Estimated difficulty: 'Foundational' | 'Associate' | 'Professional'."
    )
    aws_services: List[str] = Field(
        default_factory=list,
        description="AWS services referenced in this question, e.g. ['S3', 'CloudFront']."
    )
    rag_source_chunks: List[str] = Field(
        default_factory=list,
        description="Source chunk IDs from ChromaDB used during generation.",
        exclude=True          # omit from export – internal bookkeeping only
    )

    # ── Validation lifecycle ──────────────────
    validation_status: ValidationStatus = Field(
        ValidationStatus.PENDING,
        description="Lifecycle status after AI Reviewer loop."
    )
    revision_count: int = Field(
        0,
        description="Number of revision attempts consumed."
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of question creation."
    )

    # ─── Validators ──────────────────────────

    @field_validator("question_text")
    @classmethod
    def fix_question_text_shouting(cls, v: str) -> str:
        return fix_shouting_caps(v)

    @field_validator("explanation")
    @classmethod
    def fix_explanation_shouting(cls, v: str) -> str:
        return fix_shouting_caps(v)

    @field_validator("options")
    @classmethod
    def validate_option_count(cls, v: List[str]) -> List[str]:
        v = [fix_shouting_caps(opt) for opt in v]
        if len(v) not in (4, 5):
            raise ValueError(
                f"Options list must contain exactly 4 or 5 items, got {len(v)}."
            )
        return v

    @field_validator("correct_answers")
    @classmethod
    def validate_correct_answers_format(cls, v: List[str]) -> List[str]:
        allowed = {"A", "B", "C", "D", "E"}
        for letter in v:
            if letter.upper() not in allowed:
                raise ValueError(
                    f"Correct answer letter '{letter}' must be one of {allowed}."
                )
        return [letter.upper() for letter in v]

    @model_validator(mode="after")
    def validate_type_consistency(self) -> "ExamQuestion":
        """Enforce structural rules across fields."""

        # Single-answer → exactly 1 correct, exactly 4 options
        if self.question_type == QuestionType.SINGLE_ANSWER:
            if len(self.correct_answers) != 1:
                raise ValueError(
                    "Single-answer questions must have exactly 1 correct answer, "
                    f"got {len(self.correct_answers)}."
                )
            if len(self.options) != 4:
                raise ValueError(
                    "Single-answer questions must have exactly 4 options, "
                    f"got {len(self.options)}."
                )

        # Multiple-selection → 2+ correct, exactly 5 options
        elif self.question_type == QuestionType.MULTIPLE_SELECTION:
            if len(self.correct_answers) < 2:
                raise ValueError(
                    "Multiple-selection questions must have 2 or more correct answers."
                )
            if len(self.options) != 5:
                raise ValueError(
                    "Multiple-selection questions must have exactly 5 options, "
                    f"got {len(self.options)}."
                )
            # Question text must explicitly state the selection count
            count_word = {2: "two", 3: "three", 4: "four"}
            expected   = count_word.get(len(self.correct_answers), "")
            if expected and f"select {expected}" not in self.question_text.lower():
                raise ValueError(
                    f"Multiple-selection question text must contain "
                    f"'Select {expected.capitalize()}' (got: {self.question_text[:80]}…)."
                )

        # Correct answer letters must reference actual options
        option_letters = {
            opt.split(".")[0].strip().upper()
            for opt in self.options
            if opt and "." in opt
        }
        for ans in self.correct_answers:
            if option_letters and ans not in option_letters:
                raise ValueError(
                    f"Correct answer '{ans}' does not match any option letter "
                    f"in {option_letters}."
                )

        return self


# ─────────────────────────────────────────────
# AI Reviewer output model
# ─────────────────────────────────────────────

class CritiqueFlag(BaseModel):
    """A single identified flaw from the AI Reviewer."""
    field:       str  = Field(..., description="The field or aspect that has the issue.")
    severity:    str  = Field(..., description="'critical' | 'major' | 'minor'.")
    description: str  = Field(..., description="Human-readable explanation of the flaw.")
    suggestion:  str  = Field(..., description="Concrete fix the generator should apply.")


class ValidationCritique(BaseModel):
    """
    Structured output from the AI Reviewer agent.
    Consumed by the generator's feedback loop.
    """
    question_id:       str               = Field(...)
    passed:            bool              = Field(...)
    overall_score:     float             = Field(..., ge=0.0, le=10.0)
    flags:             List[CritiqueFlag] = Field(default_factory=list)
    reviewer_comments: str               = Field(
        default="",
        description="Free-text summary from the reviewer persona."
    )


# ─────────────────────────────────────────────
# Pipeline I/O models
# ─────────────────────────────────────────────

class GenerationRequest(BaseModel):
    """Parameters supplied by the CLI for a generation run."""
    cert_code:     str           = Field(...)
    count:         int           = Field(..., ge=1, le=200)
    output_format: ExportFormat  = Field(ExportFormat.JSON)
    output_path:   Optional[str] = Field(None)
    domain_filter: Optional[str] = Field(
        None,
        description="If set, generate questions only from this domain."
    )
    difficulty:    Optional[str] = Field(None)


class GenerationResult(BaseModel):
    """Returned to the CLI after a generation + validation run."""
    cert_code:          str                = Field(...)
    requested:          int                = Field(...)
    approved:           int                = Field(...)
    rejected:           int                = Field(...)
    questions:          List[ExamQuestion] = Field(default_factory=list)
    output_file:        Optional[str]      = Field(None)
    duration_seconds:   float              = Field(0.0)
    metadata:           Dict[str, Any]     = Field(default_factory=dict)
