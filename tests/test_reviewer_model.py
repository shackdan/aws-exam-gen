"""
Issue #4: the AI Reviewer can use a different model from the Generator.

One seam: the whole Generation Run pipeline with a fake LLM backend.
Client initialisation hands out one recording client per model name;
each client answers by role (draft, Revision, Critique). Retrieval and
the embedding model are faked; drafting, Structural Checks, Review,
Revision, Backfill and the Export on disk are real.
"""

import itertools
import json
import re
from pathlib import Path

import pytest

import config
import generator
from generator import GENERATOR_SYSTEM_PROMPT
from reviewer import REVIEWER_SYSTEM_PROMPT
from schemas import ExportFormat, GenerationRequest

GENERATOR_MODEL = "generator-model:7b"
REVIEWER_MODEL = "reviewer-model:14b"
COUNT_WORDS = {2: "TWO", 3: "THREE"}


class FakeModelClient:
    """Answers chat requests the way the named model would, recording each one."""

    def __init__(self, model, calls, critique_scores):
        self.model = model
        self._calls = calls
        self._critique_scores = critique_scores
        self._serial = itertools.count(1)

    def chat(self, messages, *, temperature, top_p, max_tokens, json_mode=False):
        system, user = messages[0]["content"], messages[-1]["content"]
        if system == REVIEWER_SYSTEM_PROMPT:
            role = "critique"
        elif "FAILED validation review" in user:
            role = "revision"
        else:
            assert system == GENERATOR_SYSTEM_PROMPT
            role = "draft"
        self._calls.append((self.model, role))
        if role == "critique":
            return json.dumps(self._critique(next(self._critique_scores, 9.0)))
        return json.dumps(self._question(user))

    @staticmethod
    def _critique(score):
        flags = [] if score >= 7 else [{
            "field": "explanation", "severity": "major",
            "description": "Explanation is thin.", "suggestion": "Expand it.",
        }]
        return {"passed": score >= 7, "overall_score": score, "flags": flags,
                "reviewer_comments": "fake critique"}

    def _question(self, prompt):
        n = next(self._serial)
        multi = ("multiple-selection question" in prompt
                 or re.search(r"Type\s*:\s*multiple", prompt) is not None)
        if multi:
            match = re.search(r"Correct answers: (\d)", prompt) or re.search(
                r"Correct Answer\(s\): \[?([^\]\n]*)", prompt)
            k = int(match.group(1)) if match and match.group(1).isdigit() else (
                len(match.group(1).split(",")) if match else 2)
            letters = "ABCDE"
            return {
                "question_text": (
                    f"Scenario {self.model} {n}: a retailer must keep order logs "
                    f"durable and cheap for seven years. Which actions meet this? "
                    f"(Select {COUNT_WORDS[k]})"),
                "options": [f"{c}. Option {c} for scenario {n}" for c in letters],
                "correct_answers": list(letters[:k]),
                "explanation": ("The correct options store logs durably at low cost; "
                                "the distractors are either volatile or expensive. Amazon S3 "
                                "lifecycle rules move ageing objects to cheaper classes."),
                "aws_services": ["S3"],
            }
        return {
            "question_text": (
                f"Scenario {self.model} {n}: a retailer must keep order logs durable "
                f"and cheap for seven years. Which service fits best?"),
            "options": ["A. Amazon S3 Glacier", "B. Instance store",
                        "C. Amazon EBS gp3", "D. Amazon ElastiCache"],
            "correct_answers": ["A"],
            "explanation": ("S3 Glacier is designed for durable, low-cost archives; "
                            "instance store is volatile, EBS and ElastiCache cost more and "
                            "are not designed for multi-year retention of logs."),
            "aws_services": ["S3"],
        }


class FakeCollection:
    def count(self):
        return 3

    def query(self, query_embeddings, n_results, include):
        docs = [f"Amazon S3 stores objects durably (chunk {i})." for i in range(3)]
        return {
            "documents": [docs],
            "metadatas": [[{"source_file": f"faq{i}.html", "page_number": 1} for i in range(3)]],
            "ids": [[f"chunk-{i}" for i in range(3)]],
        }


class FakeEmbeddingModel:
    def encode(self, text):
        return [0.1, 0.2, 0.3]


@pytest.fixture
def fake_backend(monkeypatch, tmp_path):
    """Patch the pipeline's heavy dependencies; returns a recorder of LLM traffic."""
    calls = []
    clients = {}
    state = {"critique_scores": iter([]), "unreachable": set()}

    def fake_init_llm_client(model=None, base_url=None):
        if model in state["unreachable"]:
            raise RuntimeError(f"Cannot connect to model {model}.")
        clients[model] = FakeModelClient(model, calls, state["critique_scores"])
        return clients[model]

    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(generator, "init_llm_client", fake_init_llm_client)
    monkeypatch.setattr(generator, "_load_embedding_model", lambda: FakeEmbeddingModel())
    monkeypatch.setattr(generator, "_get_chroma_collection", lambda exam_code: FakeCollection())
    monkeypatch.setattr(config.LLM_CONFIG, "model", GENERATOR_MODEL)
    monkeypatch.setattr(config.LLM_CONFIG, "reviewer_model", None, raising=False)

    class Backend:
        pass

    backend = Backend()
    backend.calls = calls
    backend.clients = clients
    backend.state = state

    def set_reviewer_model(name):
        monkeypatch.setattr(config.LLM_CONFIG, "reviewer_model", name, raising=False)

    def critique_scores(*scores):
        state["critique_scores"] = iter(scores)

    backend.set_reviewer_model = set_reviewer_model
    backend.critique_scores = critique_scores
    return backend


def run(count=2):
    request = GenerationRequest(cert_code="SAA-C03", count=count, output_format=ExportFormat.JSON)
    return generator.run_generation_pipeline(request)


def test_critiques_come_from_reviewer_model_and_drafts_and_revisions_from_generator(fake_backend):
    fake_backend.set_reviewer_model(REVIEWER_MODEL)
    fake_backend.critique_scores(4.0)  # first Critique fails, forcing one Revision

    result = run()

    roles_by_model = {}
    for model, role in fake_backend.calls:
        roles_by_model.setdefault(model, set()).add(role)
    assert roles_by_model == {
        GENERATOR_MODEL: {"draft", "revision"},
        REVIEWER_MODEL: {"critique"},
    }
    assert result.approved == 2


def export_metadata(result):
    return json.loads(Path(result.output_file).read_text(encoding="utf-8"))["metadata"]


def test_without_reviewer_model_one_model_plays_both_roles(fake_backend):
    result = run()

    assert list(fake_backend.clients) == [GENERATOR_MODEL]
    assert {model for model, _ in fake_backend.calls} == {GENERATOR_MODEL}
    assert {role for _, role in fake_backend.calls} >= {"draft", "critique"}
    metadata = export_metadata(result)
    assert metadata["generator_model"] == GENERATOR_MODEL
    assert metadata["reviewer_model"] == GENERATOR_MODEL


def test_export_records_both_models(fake_backend):
    fake_backend.set_reviewer_model(REVIEWER_MODEL)

    result = run()

    metadata = export_metadata(result)
    assert metadata["generator_model"] == GENERATOR_MODEL
    assert metadata["reviewer_model"] == REVIEWER_MODEL
    assert result.metadata["reviewer_model"] == REVIEWER_MODEL


def test_unreachable_reviewer_model_fails_at_startup_before_drafting(fake_backend):
    fake_backend.set_reviewer_model(REVIEWER_MODEL)
    fake_backend.state["unreachable"].add(REVIEWER_MODEL)

    with pytest.raises(RuntimeError, match=f"AI Reviewer model '{REVIEWER_MODEL}'"):
        run()

    assert fake_backend.calls == []
    assert not (config.OUTPUT_DIR).exists() or list(config.OUTPUT_DIR.iterdir()) == []
