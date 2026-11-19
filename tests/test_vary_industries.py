"""
Issue #13: vary scenario industries within a Generation Run.

Seam: the whole Generation Run pipeline with a fake LLM backend
(see tests/pipeline_fakes.py), asserting on the drafting prompts sent
to the Generator.
"""

import itertools
import re

from tests.pipeline_fakes import (
    COUNT_WORDS, LONG_EXPLANATION, ScriptedClient, correct_count,
    install, is_multiple, run,
)


def drafts_opening_with(*openings):
    """Draft handler whose n-th draft's scenario starts with openings[n-1]
    (the last opening repeats)."""
    serial = itertools.count(1)

    def draft(prompt, n):
        s = next(serial)
        opening = openings[min(n, len(openings)) - 1]
        if is_multiple(prompt):
            k = correct_count(prompt)
            return {
                "question_text": (f"{opening} must keep logs durable and cheap for seven "
                                  f"years (case {s}). Which actions meet this? "
                                  f"(Select {COUNT_WORDS[k]})"),
                "options": [f"{c}. Amazon S3 option {c} {s}" for c in "ABCDE"],
                "correct_answers": list("ABCDE"[:k]),
                "explanation": LONG_EXPLANATION,
                "aws_services": ["S3"],
            }
        return {
            "question_text": (f"{opening} must keep logs durable and cheap for seven years "
                              f"(case {s}). Which service fits best?"),
            "options": ["A. Amazon S3 Glacier", "B. Instance store",
                        "C. Amazon EBS gp3", "D. Amazon ElastiCache"],
            "correct_answers": ["A"],
            "explanation": LONG_EXPLANATION,
            "aws_services": ["S3 Glacier"],
        }

    return draft


def industries_to_avoid(prompt):
    match = re.search(r"different industry from: ([^\n]*)", prompt)
    return match.group(1).split(", ") if match else []


def test_after_a_healthcare_scenario_the_next_draft_avoids_healthcare(monkeypatch, tmp_path):
    client = ScriptedClient(draft=drafts_opening_with(
        "A healthcare company", "A company", "A company", "A company"))
    install(monkeypatch, tmp_path, client)

    run(count=4)

    prompts = client.prompts("draft")
    assert industries_to_avoid(prompts[0]) == []
    for later in prompts[1:]:
        assert industries_to_avoid(later) == ["healthcare"]


def test_industries_accumulate_across_the_run(monkeypatch, tmp_path):
    client = ScriptedClient(draft=drafts_opening_with(
        "A hospital network", "A bank", "An online retailer", "A company"))
    install(monkeypatch, tmp_path, client)

    run(count=4)

    prompts = client.prompts("draft")
    assert industries_to_avoid(prompts[3]) == ["healthcare", "financial services", "retail"]


def test_a_scenario_with_no_industry_adds_nothing(monkeypatch, tmp_path):
    client = ScriptedClient(draft=drafts_opening_with("A company"))
    install(monkeypatch, tmp_path, client)

    run(count=4)

    assert all(industries_to_avoid(p) == [] for p in client.prompts("draft"))


def test_a_new_generation_run_starts_with_no_industries_to_avoid(monkeypatch, tmp_path):
    first = ScriptedClient(draft=drafts_opening_with("A healthcare company"))
    install(monkeypatch, tmp_path, first)
    run(count=2)

    second = ScriptedClient(draft=drafts_opening_with("A company"))
    install(monkeypatch, tmp_path, second)
    run(count=2)

    assert industries_to_avoid(second.prompts("draft")[0]) == []
