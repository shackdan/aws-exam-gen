"""
Issue #12: a Question's AWS Services lists only what it tests, not its
Distractors.

Seam: the whole Generation Run pipeline with a fake LLM backend
(see tests/pipeline_fakes.py). A count of 4 drafts 3 Multiple-choice
and 1 Multiple-response Question.
"""

import itertools
import re

import pytest

from schemas import ExportFormat, GenerationRequest
import generator
from tests.pipeline_fakes import (
    COUNT_WORDS, LONG_EXPLANATION, ScriptedClient, correct_count,
    exported_questions, install, is_multiple, run,
)

DOMAIN = "Design Secure Architectures"


@pytest.fixture
def drafts():
    serial = itertools.count(1)

    def draft(prompt, n):
        s = next(serial)
        if is_multiple(prompt):
            k = correct_count(prompt)
            # A–C are S3 features, so the Correct Answers (A–B or A–C) name
            # only Amazon S3 and S3 Glacier; D–E are Distractor services.
            return {
                "question_text": (f"Scenario {s}: a retailer must keep order logs durable and "
                                  f"cheap for seven years. Which actions meet this? "
                                  f"(Select {COUNT_WORDS[k]})"),
                "options": ["A. Amazon S3 Lifecycle rules", "B. Amazon S3 Glacier Deep Archive",
                            "C. Amazon S3 Object Lock", "D. AWS CloudHSM", "E. AWS WAF"],
                "correct_answers": list("ABCDE"[:k]),
                "explanation": LONG_EXPLANATION,
                "aws_services": ["Amazon S3", "S3 Glacier", "CloudHSM", "AWS WAF"],
            }
        return {
            "question_text": (f"Scenario {s}: a retailer must keep order logs durable and cheap "
                              f"for seven years. Which service fits best?"),
            "options": ["A. Amazon S3 Glacier Flexible Retrieval", "B. AWS CloudHSM",
                        "C. AWS WAF", "D. Amazon Route 53"],
            "correct_answers": ["A"],
            "explanation": LONG_EXPLANATION,
            "aws_services": ["S3 Glacier", "CloudHSM", "WAF", "Route 53"],
        }

    return draft


def test_a_question_keeps_only_its_correct_answers_services(monkeypatch, tmp_path, drafts):
    install(monkeypatch, tmp_path, ScriptedClient(draft=drafts))

    result = run(count=4)

    singles = [q for q in exported_questions(result) if q["question_type"] == "single"]
    assert len(singles) == 3
    for q in singles:
        assert q["aws_services"] == ["S3 Glacier"]


def test_a_multiple_response_question_keeps_every_correct_answers_services(
        monkeypatch, tmp_path, drafts):
    install(monkeypatch, tmp_path, ScriptedClient(draft=drafts))

    result = run(count=4)

    multis = [q for q in exported_questions(result) if q["question_type"] == "multiple"]
    assert len(multis) == 1
    assert multis[0]["aws_services"] == ["Amazon S3", "S3 Glacier"]


def test_steering_counts_only_the_services_a_question_tests(monkeypatch, tmp_path, drafts):
    client = ScriptedClient(draft=drafts)
    install(monkeypatch, tmp_path, client)

    request = GenerationRequest(cert_code="SAA-C03", count=6, domain_filter=DOMAIN,
                                output_format=ExportFormat.JSON)
    generator.run_generation_pipeline(request)

    notes = [m.group(1) for p in client.prompts("draft")
             if (m := re.search(r"on these unless clearly the best fit: ([^\n]*)", p))]
    assert notes, "steering should kick in once the domain has enough Questions"
    for note in notes:
        assert "S3 Glacier" in note
        for distractor in ("CloudHSM", "WAF", "Route 53"):
            assert distractor not in note


def test_the_drafting_prompts_ask_for_the_correct_answers_services_only(
        monkeypatch, tmp_path, drafts):
    client = ScriptedClient(draft=drafts)
    install(monkeypatch, tmp_path, client)

    run(count=4)

    for prompt in client.prompts("draft"):
        assert "not the distractors" in prompt


def test_a_plural_service_name_matches_its_singular_in_the_correct_answer(monkeypatch, tmp_path):
    serial = itertools.count(1)

    def draft(prompt, n):
        s = next(serial)
        if is_multiple(prompt):
            return drafts_fixture_multi(prompt, s)
        return {
            "question_text": (f"Scenario {s}: a retailer expects traffic spikes during sales "
                              f"and wants to cut compute cost. Which solution fits best?"),
            "options": ["A. Use a Spot Instance fleet behind Auto Scaling",
                        "B. Use AWS Lambda for every workload",
                        "C. Use an Amazon CloudFront distribution",
                        "D. Use a fixed-size group of On-Demand instances"],
            "correct_answers": ["A"],
            "explanation": LONG_EXPLANATION,
            "aws_services": ["Spot Instances", "Auto Scaling", "Lambda", "CloudFront"],
        }

    install(monkeypatch, tmp_path, ScriptedClient(draft=draft))

    result = run(count=4)

    singles = [q for q in exported_questions(result) if q["question_type"] == "single"]
    assert len(singles) == 3
    for q in singles:
        assert q["aws_services"] == ["Spot Instances", "Auto Scaling"]


def drafts_fixture_multi(prompt, s):
    k = correct_count(prompt)
    return {
        "question_text": (f"Scenario {s}: a retailer must keep order logs durable and cheap "
                          f"for seven years. Which actions meet this? (Select {COUNT_WORDS[k]})"),
        "options": ["A. Amazon S3 Lifecycle rules", "B. Amazon S3 Glacier Deep Archive",
                    "C. Amazon S3 Object Lock", "D. AWS CloudHSM", "E. AWS WAF"],
        "correct_answers": list("ABCDE"[:k]),
        "explanation": LONG_EXPLANATION,
        "aws_services": ["Amazon S3", "S3 Glacier"],
    }
