import pytest
from pydantic import ValidationError

from korean_humanizer.protection import check_protection
from korean_humanizer.semantic_review import SemanticRating, summarize_semantics


def rating(**updates):
    data = dict(
        task_id="r1",
        rater_id="fixture-rater",
        rater_kind="ai",
        review_ref="test-fixture",
        propositions=[{"id": "p1", "verdict": "preserved"}],
        unsupported_addition="absent",
        register_drift="absent",
        naturalness_vs_input="same",
    )
    return SemanticRating.model_validate(data | updates)


def key():
    return {
        "mapping": {
            "r1": {
                "candidate": "model-p0",
                "expected_edit": "minimal",
                "group_id": "pair-1",
                "proposition_ids": ["p1"],
                "finish_reason": "stop",
            },
            "r2": {
                "candidate": "model-p1",
                "expected_edit": "rewrite",
                "group_id": "pair-1",
                "proposition_ids": ["p1"],
                "finish_reason": "stop",
            },
        }
    }


def test_omission_can_fail_review_even_when_mechanical_protection_passes():
    before = "초안을 읽었습니다. 일정만 확인해 주세요."
    after = "일정만 확인해 주세요."
    assert check_protection(before, after)["mechanical_pass"]
    record = rating(
        propositions=[
            {"id": "p1", "verdict": "omitted", "evidence": "초안을 읽었다는 사실이 출력에 없음"}
        ]
    )
    summary = summarize_semantics([record], key())
    cell = summary["cells"][0]
    assert cell["counts"]["with_omission"] == 1
    assert cell["counts"]["preserved_with_appropriate_edit"] == 0
    assert summary["cells"][1]["unrated"] == 1


def test_blank_and_partial_claim_reviews_cannot_pass():
    for patch in [
        {"naturalness_vs_input": None},
        {"rater_id": ""},
        {"propositions": [{"id": "p1", "verdict": "omitted"}]},
    ]:
        with pytest.raises(ValidationError):
            rating(**patch)
    with pytest.raises(ValueError, match="every proposition"):
        summarize_semantics([rating(propositions=[{"id": "wrong", "verdict": "preserved"}])], key())
    with pytest.raises(ValueError, match="duplicate"):
        summarize_semantics([rating(), rating()], key())


def test_unchanged_is_acceptable_for_minimal_but_not_rewrite_target():
    report = summarize_semantics([rating(), rating(task_id="r2")], key())
    assert report["cells"][0]["counts"]["preserved_with_appropriate_edit"] == 1
    assert report["cells"][1]["counts"]["preserved_with_appropriate_edit"] == 0


def test_unassessable_incomplete_and_ai_reviews_are_separate():
    incomplete = key()
    incomplete["mapping"]["r1"]["finish_reason"] = "length"
    report = summarize_semantics([rating()], incomplete)
    assert report["cells"][0]["counts"]["preserved_with_appropriate_edit"] == 0
    record = rating(naturalness_vs_input="unassessable", evidence="fixture cannot be rated")
    report = summarize_semantics(
        [record, rating(task_id="r2", rater_id="fixture-human", rater_kind="human")], key()
    )
    assert {c["rater_kind"] for c in report["cells"]} == {"ai", "human"}
    assert report["release_verdict"] == "not_assessed"
