"""Invented genre cues and author inventories, not measured real genre accuracy."""

from collections import Counter
from types import SimpleNamespace

from korean_humanizer.candidate_review import assess_candidate, assess_np_candidate


def test_personal_past_events_and_first_paragraph_are_not_lost_as_a_heading():
    text = (
        "나는 그날 처음 학교에 갔다. 내가 새 친구와 인사했고 함께 걸었다.\n집에 오니 기분이 좋았다."
    )
    result = assess_candidate(text)
    assert result["genre_hint"] == "experience_review" and not result["genre_verified"]
    assert result["features"]["first_person"] == 2


def test_quoted_first_person_does_not_turn_reporting_into_experience():
    text = "취재 소식\n그는 “나는 그곳에 갔다. 내가 친구를 만났고 기쁨을 느꼈다”고 말했다."
    assert not assess_candidate(text)["eligible"]


def test_direct_public_instructions_are_distinct_from_petition_reporting():
    text = (
        "수업 안내\n신청 방법은 온라인 접수입니다. "
        "제출 서류를 준비하세요. 문의는 홈페이지에서 받습니다."
    )
    assert assess_candidate(text)["genre_hint"] == "practical_notice"
    assert not assess_candidate(
        "법원에 탄원서를 제출했다. 접수 기간에 맞춰 신청 방법을 확인했고 온라인으로 문의했다."
    )["eligible"]
    assert not assess_candidate("신청 방법이 바뀌었다. 위원회가 접수 방안을 논의했다.")["eligible"]
    assert not assess_candidate(
        "참가자는 신청서를 작성할 수 있었다. 접수 기간에 홈페이지가 열렸다."
    )["eligible"]


def test_conflicting_labels_and_markup_are_queued_for_review():
    narrative = "나는 친구를 만났다. 내가 함께 학교에 갔고 집에 돌아왔다."
    assert (
        assess_candidate(narrative, source_label="explanation_information")["reason"]
        == "purpose_cues_conflict_with_prior_source_label"
    )
    assert not assess_candidate(narrative, source_label="unclassified")["eligible"]
    assert not assess_candidate('<span style="color: red">' + narrative + "</span>")["eligible"]


def test_reselection_joint_author_cap_and_limit(monkeypatch):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path("scripts").resolve()))
    from candidate_selection import select_candidates

    def row(key, author):
        return SimpleNamespace(
            split="train", source=SimpleNamespace(source_doc_id=key, author_group=author)
        )

    rows = [row("a", "shared"), row("b", "shared"), row("c", "other")]
    evidence = {
        r.source.source_doc_id: {
            "genre_hint": "experience_review" if i == 0 else "practical_notice",
            "score": 10,
        }
        for i, r in enumerate(rows)
    }
    picked, counts = select_candidates(
        rows, evidence, Counter({("train", "shared"): 19}), {}, limit=3
    )
    assert len(picked) == 2 and sum(counts.values()) == 2
    assert sum(r.source.author_group == "shared" for r in picked) == 1
    assert len(select_candidates(rows, evidence, Counter(), {}, limit=1)[0]) == 1


def test_np_metadata_recovers_omitted_subject_without_accepting_a_pair():
    text = "친구를 만났고 함께 걸었다. 집에 돌아왔고 많은 것을 배웠다."
    assert not assess_candidate(text)["eligible"]
    result = assess_np_candidate(text, "비출판물 > 일기")
    assert result["eligible"] and result["genre_hint"] == "experience_review"
    assert result["metadata_assisted_hint"] and not result["genre_verified"]
    assert not assess_np_candidate(text, "비출판물 > 기타 > 설명문")["eligible"]
    assert not assess_np_candidate(text, "비출판물 > 일기", source_label="unclassified")["eligible"]


def test_np_category_does_not_rescue_reported_speech_fiction_or_prior_conflicts():
    text = "나는 친구를 만났다. 내가 학교에 갔고 기쁨을 느꼈다."
    for category in ["비출판물 > 기타 > 희곡", "비출판물 > 기타 > 일기, 상상하는 글"]:
        assert not assess_np_candidate(text, category)["eligible"]
    assert not assess_np_candidate(text, "비출판물 > 일기", source_label="explanation_information")[
        "eligible"
    ]
    assert not assess_np_candidate(
        "그는 “친구를 만났고 함께 걸었고 돌아왔다”고 말했다.", "비출판물 > 수필"
    )["eligible"]
