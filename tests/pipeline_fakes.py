"""
Shared fakes for tests that drive the whole Generation Run pipeline.

Retrieval and the embedding model are faked; the LLM backend is a
recording client whose answers come from per-role handler functions.
Drafting, Structural Checks, Review, Revision, Backfill and the Export
on disk stay real.
"""

import json
import re
from pathlib import Path

import config
import generator
from generator import GENERATOR_SYSTEM_PROMPT
from reviewer import REVIEWER_SYSTEM_PROMPT
from schemas import ExportFormat, GenerationRequest

COUNT_WORDS = {2: "TWO", 3: "THREE"}
LONG_EXPLANATION = (
    "The correct option meets every stated requirement at the lowest cost; "
    "each distractor is either volatile, more expensive, or not designed for "
    "this kind of workload, as the AWS documentation describes."
)


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


def role_of(messages):
    system, user = messages[0]["content"], messages[-1]["content"]
    if system == REVIEWER_SYSTEM_PROMPT:
        return "critique"
    if "FAILED validation review" in user:
        return "revision"
    assert system == GENERATOR_SYSTEM_PROMPT
    return "draft"


def is_multiple(prompt):
    return ("multiple-selection question" in prompt
            or re.search(r"Type\s*:\s*multiple", prompt) is not None)


def correct_count(prompt):
    match = re.search(r"Correct answers: (\d)", prompt)
    if match:
        return int(match.group(1))
    match = re.search(r"Correct Answer\(s\): ([^\n]*)", prompt)
    return len(match.group(1).split(",")) if match else 2


def passing_critique(score=9.0):
    return {"passed": True, "overall_score": score, "flags": [],
            "reviewer_comments": "fake critique"}


class ScriptedClient:
    """
    Records every chat request as (role, user_prompt) and answers it with
    the handler for that role. A handler takes (user_prompt, n) where n
    counts requests of that role from 1, and returns a dict sent as JSON.
    """

    def __init__(self, draft, critique=None, revision=None):
        self.requests = []
        self._handlers = {
            "draft": draft,
            "critique": critique or (lambda prompt, n: passing_critique()),
            "revision": revision or draft,
        }
        self._counts = {"draft": 0, "critique": 0, "revision": 0}

    def chat(self, messages, *, temperature, top_p, max_tokens, json_mode=False):
        role = role_of(messages)
        prompt = messages[-1]["content"]
        self.requests.append((role, prompt))
        self._counts[role] += 1
        return json.dumps(self._handlers[role](prompt, self._counts[role]))

    def prompts(self, role):
        return [p for r, p in self.requests if r == role]


def install(monkeypatch, tmp_path, client):
    """Route the pipeline's LLM, retrieval and output folder to fakes."""
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(generator, "init_llm_client", lambda model=None, base_url=None: client)
    monkeypatch.setattr(generator, "_load_embedding_model", lambda: FakeEmbeddingModel())
    monkeypatch.setattr(generator, "_get_chroma_collection", lambda exam_code: FakeCollection())
    monkeypatch.setattr(config.LLM_CONFIG, "reviewer_model", None, raising=False)


def run(count):
    request = GenerationRequest(cert_code="SAA-C03", count=count, output_format=ExportFormat.JSON)
    return generator.run_generation_pipeline(request)


def exported_questions(result):
    return json.loads(Path(result.output_file).read_text(encoding="utf-8"))["questions"]
