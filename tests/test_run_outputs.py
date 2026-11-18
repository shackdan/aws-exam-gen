"""Tests for what a Generation Run writes: its Export and optional Rendering."""

import json

from schemas import ExamQuestion, ExportFormat, QuestionType, ValidationStatus
from utils import hash_question_text, load_prior_question_hashes, write_run_outputs


def approved_question(text: str = "A company needs durable object storage for logs. Which service fits best?") -> ExamQuestion:
    return ExamQuestion(
        cert_code="SAA-C03",
        domain="Design Resilient Architectures",
        question_type=QuestionType.SINGLE_ANSWER,
        question_text=text,
        options=["A. Amazon S3", "B. Amazon EBS", "C. Instance store", "D. Amazon EFS"],
        correct_answers=["A"],
        explanation=(
            "Amazon S3 is designed for 11 nines of durability and suits log archives. "
            "EBS, instance store and EFS are block or file storage with different trade-offs."
        ),
        validation_status=ValidationStatus.APPROVED,
    )


def test_csv_run_still_writes_json_export_to_output_dir(tmp_path):
    output_dir = tmp_path / "output"

    outputs = write_run_outputs(
        questions=[approved_question()],
        exam_code="SAA-C03",
        fmt=ExportFormat.CSV,
        out_file=None,
        output_dir=output_dir,
    )

    assert outputs.export_path.parent == output_dir
    assert outputs.export_path.name.startswith("SAA-C03_1q_")
    assert outputs.export_path.suffix == ".json"
    exported = json.loads(outputs.export_path.read_text(encoding="utf-8"))
    assert [q["question_text"] for q in exported["questions"]] == [
        "A company needs durable object storage for logs. Which service fits best?"
    ]


def test_csv_run_writes_csv_rendering_next_to_export_by_default(tmp_path):
    output_dir = tmp_path / "output"

    outputs = write_run_outputs(
        questions=[approved_question()],
        exam_code="SAA-C03",
        fmt=ExportFormat.CSV,
        out_file=None,
        output_dir=output_dir,
    )

    assert outputs.rendering_path == outputs.export_path.with_suffix(".csv")
    rendering = outputs.rendering_path.read_text(encoding="utf-8")
    assert "A company needs durable object storage for logs." in rendering


def test_out_file_moves_only_the_rendering(tmp_path):
    output_dir = tmp_path / "output"
    out_file = tmp_path / "elsewhere" / "for-my-lms.csv"

    outputs = write_run_outputs(
        questions=[approved_question()],
        exam_code="SAA-C03",
        fmt=ExportFormat.CSV,
        out_file=out_file,
        output_dir=output_dir,
    )

    assert outputs.rendering_path == out_file
    assert out_file.is_file()
    assert outputs.export_path.parent == output_dir
    assert outputs.export_path.is_file()


def test_moodle_run_writes_xml_rendering_alongside_export(tmp_path):
    output_dir = tmp_path / "output"

    outputs = write_run_outputs(
        questions=[approved_question()],
        exam_code="SAA-C03",
        fmt=ExportFormat.MOODLE,
        out_file=None,
        output_dir=output_dir,
    )

    assert outputs.export_path.suffix == ".json"
    assert outputs.rendering_path == outputs.export_path.with_suffix(".xml")
    assert "<quiz>" in outputs.rendering_path.read_text(encoding="utf-8")


def test_json_run_has_no_rendering_unless_out_file_given(tmp_path):
    output_dir = tmp_path / "output"

    plain = write_run_outputs(
        questions=[approved_question()],
        exam_code="SAA-C03",
        fmt=ExportFormat.JSON,
        out_file=None,
        output_dir=output_dir,
    )
    assert plain.rendering_path is None

    out_file = tmp_path / "copy-for-me.json"
    with_copy = write_run_outputs(
        questions=[approved_question()],
        exam_code="SAA-C03",
        fmt=ExportFormat.JSON,
        out_file=out_file,
        output_dir=output_dir,
    )
    assert with_copy.rendering_path == out_file
    assert with_copy.export_path.parent == output_dir
    assert json.loads(out_file.read_text(encoding="utf-8")) == json.loads(
        with_copy.export_path.read_text(encoding="utf-8")
    )


def test_later_run_sees_csv_runs_questions_as_duplicates(tmp_path):
    output_dir = tmp_path / "output"
    text = "A company needs durable object storage for logs. Which service fits best?"

    write_run_outputs(
        questions=[approved_question(text)],
        exam_code="SAA-C03",
        fmt=ExportFormat.CSV,
        out_file=tmp_path / "lms" / "upload.csv",
        output_dir=output_dir,
    )

    question_bank = load_prior_question_hashes(cert_code="SAA-C03", output_dir=output_dir)
    assert hash_question_text(text) in question_bank
