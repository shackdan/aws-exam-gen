"""
Pipeline-level tests for what a Generation Run exports (issue #2).

The LLM client, embedding model, ChromaDB collection, generation and
Review are replaced with fakes; everything from Review's verdicts to the
files on disk is the real pipeline.
"""

import json
from pathlib import Path

import config
import generator
import reviewer
from schemas import ExportFormat, GenerationRequest, ValidationStatus
from tests.test_run_outputs import approved_question


def run_pipeline(monkeypatch, tmp_path, verdicts):
    """Run the pipeline where Review gives each drafted Question the next verdict."""
    output_dir = tmp_path / "output"
    monkeypatch.setattr(config, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(generator, "init_llm_client", lambda model=None, base_url=None: object())
    monkeypatch.setattr(generator, "_load_embedding_model", lambda: object())

    class FakeCollection:
        def count(self):
            return 1

    monkeypatch.setattr(generator, "_get_chroma_collection", lambda exam_code: FakeCollection())

    drafted = iter(range(1000))

    def fake_generate_batch(*args, slots=None, **kwargs):
        n = slots or kwargs["request"].count
        return [
            approved_question(f"Draft question number {next(drafted)} about storing logs durably?")
            .model_copy(update={"validation_status": ValidationStatus.PENDING})
            for _ in range(n)
        ]

    verdict_iter = iter(verdicts)

    def fake_review_batch(questions, llm_client, revision_callback):
        reviewed = [
            q.model_copy(update={"validation_status": next(verdict_iter, ValidationStatus.REJECTED)})
            for q in questions
        ]
        return reviewed, []

    monkeypatch.setattr(generator, "generate_batch", fake_generate_batch)
    monkeypatch.setattr(reviewer, "review_batch", fake_review_batch)

    request = GenerationRequest(cert_code="SAA-C03", count=2, output_format=ExportFormat.CSV)
    return generator.run_generation_pipeline(request), output_dir


def test_run_with_zero_approved_questions_exports_nothing(monkeypatch, tmp_path):
    result, output_dir = run_pipeline(monkeypatch, tmp_path, verdicts=[])

    assert result.approved == 0
    assert result.rejected > 0
    assert result.output_file is None
    assert result.rendering_file is None
    assert not output_dir.exists() or list(output_dir.iterdir()) == []


def test_mixed_run_exports_only_its_approved_questions(monkeypatch, tmp_path):
    result, output_dir = run_pipeline(
        monkeypatch,
        tmp_path,
        verdicts=[ValidationStatus.APPROVED, ValidationStatus.REJECTED, ValidationStatus.APPROVED],
    )

    assert result.approved == 2
    exported = json.loads(Path(result.output_file).read_text(encoding="utf-8"))
    assert all(q["validation_status"] == "Approved" for q in exported["questions"])
    assert len(exported["questions"]) == 2
    assert Path(result.output_file).name.startswith("SAA-C03_2q_")
