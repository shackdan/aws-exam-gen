"""
Issue #11: every Question must actually ask something.

Seam: the whole Generation Run pipeline with a fake LLM backend
(see tests/pipeline_fakes.py). A count of 4 drafts 3 Multiple-choice
and 1 Multiple-response Question.
"""

import itertools

from tests.pipeline_fakes import (
    COUNT_WORDS, LONG_EXPLANATION, ScriptedClient, correct_count,
    exported_questions, install, is_multiple, run,
)

_serial = itertools.count(1)


def draft_without_question_mark(prompt, n):
    """A stem that sets up a scenario and never asks anything."""
    s = next(_serial)
    if is_multiple(prompt):
        k = correct_count(prompt)
        return {
            "question_text": (f"Scenario {s}: a retailer must keep order logs durable and "
                              f"cheap for seven years and wants managed storage. "
                              f"(Select {COUNT_WORDS[k]})"),
            "options": [f"{c}. Option {c} for scenario {s}" for c in "ABCDE"],
            "correct_answers": list("ABCDE"[:k]),
            "explanation": LONG_EXPLANATION,
            "aws_services": ["S3"],
        }
    return {
        "question_text": (f"Scenario {s}: a healthcare company must encrypt patient records "
                          f"at rest and keep control of the keys."),
        "options": ["A. AWS KMS customer managed keys", "B. Instance store",
                    "C. Amazon ElastiCache", "D. Amazon EBS gp3"],
        "correct_answers": ["A"],
        "explanation": LONG_EXPLANATION,
        "aws_services": ["KMS"],
    }


def revision_adds_question(prompt, n):
    s = next(_serial)
    return {
        "question_text": (f"Scenario {s}: a healthcare company must encrypt patient records "
                          f"at rest and keep control of the keys. Which solution meets "
                          f"these requirements?"),
        "options": ["A. AWS KMS customer managed keys", "B. Instance store",
                    "C. Amazon ElastiCache", "D. Amazon EBS gp3"],
        "correct_answers": ["A"],
        "explanation": LONG_EXPLANATION,
        "aws_services": ["KMS"],
    }


def test_multiple_choice_stem_without_a_question_is_revised_then_approved(monkeypatch, tmp_path):
    client = ScriptedClient(draft=draft_without_question_mark, revision=revision_adds_question)
    install(monkeypatch, tmp_path, client)

    result = run(count=4)

    revisions = client.prompts("revision")
    assert len(revisions) == 3  # one per Multiple-choice Question, none for Multiple-response
    for prompt in revisions:
        assert "Severity : CRITICAL" in prompt
        assert "?" in prompt.split("Flags to fix:")[1]
    assert result.approved == 4
    singles = [q for q in exported_questions(result) if q["question_type"] == "single"]
    assert len(singles) == 3
    assert all("?" in q["question_text"] for q in singles)


def test_multiple_response_select_phrase_counts_as_the_ask(monkeypatch, tmp_path):
    client = ScriptedClient(draft=draft_without_question_mark, revision=revision_adds_question)
    install(monkeypatch, tmp_path, client)

    result = run(count=4)

    multis = [q for q in exported_questions(result) if q["question_type"] == "multiple"]
    assert len(multis) == 1
    assert "?" not in multis[0]["question_text"]
    assert multis[0]["revision_count"] == 0


def test_drafting_and_revision_prompts_say_to_end_with_the_question(monkeypatch, tmp_path):
    client = ScriptedClient(draft=draft_without_question_mark, revision=revision_adds_question)
    install(monkeypatch, tmp_path, client)

    run(count=4)

    drafts = client.prompts("draft")
    assert any(is_multiple(p) for p in drafts) and any(not is_multiple(p) for p in drafts)
    for prompt in drafts + client.prompts("revision"):
        assert "must end with the question being asked" in prompt


def test_multiple_choice_revision_prompt_never_mentions_a_select_phrase(monkeypatch, tmp_path):
    client = ScriptedClient(draft=draft_without_question_mark, revision=revision_adds_question)
    install(monkeypatch, tmp_path, client)

    run(count=4)

    revisions = client.prompts("revision")
    assert revisions
    for prompt in revisions:
        assert "(Select" not in prompt


def test_multiple_response_revision_prompt_names_the_real_select_phrase(monkeypatch, tmp_path):
    critiques = iter([4.0])  # the Multiple-response Question's first Critique fails

    def critique(prompt, n):
        score = next(critiques, 9.0) if "(Select" in prompt else 9.0
        flags = [] if score >= 7 else [{"field": "explanation", "severity": "major",
                                        "description": "Thin.", "suggestion": "Expand."}]
        return {"passed": score >= 7, "overall_score": score, "flags": flags,
                "reviewer_comments": "fake critique"}

    client = ScriptedClient(draft=draft_without_question_mark, revision=revision_adds_question,
                            critique=critique)
    install(monkeypatch, tmp_path, client)

    run(count=4)

    multi_revisions = [p for p in client.prompts("revision") if is_multiple(p)]
    assert len(multi_revisions) == 1
    k = correct_count(multi_revisions[0])
    assert f'"(Select {COUNT_WORDS[k]})"' in multi_revisions[0]
    assert "(Select N)" not in multi_revisions[0]


def test_multiple_choice_stem_with_a_select_phrase_is_revised(monkeypatch, tmp_path):
    def draft(prompt, n):
        q = revision_adds_question(prompt, n)
        if not is_multiple(prompt):
            q["question_text"] += " (Select N)"
            return q
        return draft_without_question_mark(prompt, n)

    client = ScriptedClient(draft=draft, revision=revision_adds_question)
    install(monkeypatch, tmp_path, client)

    result = run(count=4)

    revisions = client.prompts("revision")
    assert len(revisions) == 3
    for prompt in revisions:
        assert "Severity : CRITICAL" in prompt
    assert result.approved == 4
    singles = [q for q in exported_questions(result) if q["question_type"] == "single"]
    assert all("(Select" not in q["question_text"] for q in singles)
