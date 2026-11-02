"""AI-authored toy strings and invented judge responses, never real semantic measurements."""

import json

from korean_humanizer.automated_quality import screen

SOURCE = "이것은 파서를 시험하는 가짜 글이다. 인공지능이 작성했다."
DRAFT = "인공지능이 작성한 이 글은 파서 시험을 위한 가짜 글이다."


def response(**changes):
    value = dict(
        genre="explanation_information",
        standalone="pass",
        forward="pass",
        backward="pass",
        logic="pass",
        stance="pass",
        register_match="pass",
        source_quote="가짜 글이다",
        draft_quote="가짜 글이다",
        genre_quote="파서를 시험",
    )
    value.update(changes)
    return json.dumps(value, ensure_ascii=False)


def test_ai_pass_is_never_human_review_or_acceptance():
    result = screen(SOURCE, DRAFT, response())
    assert result["automatic_pass"] is True
    assert result["rater_kind"] == "ai"
    assert result["human_review_completed"] is False
    assert result["pair_review_status"] == "pending"


def test_claimed_pass_with_fabricated_evidence_unknown_genre_or_uncertainty_is_excluded():
    for output in [
        response(source_quote="있지도 않은 주장"),
        response(genre="unclassified"),
        response(logic="unclear"),
        response(stance="fail"),
        "```json\n{}\n```",
    ]:
        assert screen(SOURCE, DRAFT, output)["automatic_pass"] is False


def test_mechanical_change_and_noop_cannot_be_rescued_by_an_ai_claim():
    assert (
        screen(SOURCE + " 2개", DRAFT + " 3개", response())["reason"]
        == "mechanical_protection_failed"
    )
    assert screen(SOURCE, SOURCE, response())["reason"] == "unchanged_not_a_reverse_example"
