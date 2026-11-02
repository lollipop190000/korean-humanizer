"""Synthetic adversarial cases, not corpus quality measurements."""

import json

import pytest

from korean_humanizer.automated_quality import screen
from korean_humanizer.protection import atoms, check_protection, protected_spans
from korean_humanizer.revision_protocols import draft_input, evidence_units, resolve_evidence


def test_url_boundary_ignores_prose_but_preserves_balanced_url_path_and_query():
    assert check_protection(
        "주소(https://example.org/a)는 안내다.", "주소(https://example.org/a)에서 확인한다."
    )["mechanical_pass"]
    assert atoms("https://example.org/a(b)?x=3)에서") == atoms("https://example.org/a(b)?x=3)도")
    for changed in [
        "http://example.org/a(b)?x=3",
        "https://example.org/a(b)?x=4",
        "https://example.org/a(c)?x=3",
        "https://example.org/다른경로",
    ]:
        assert not check_protection("https://example.org/a(b)?x=3)에서", changed)["mechanical_pass"]
    assert list(protected_spans("(https://example.org/a)ABC"))[-1][2] == "ABC"


def test_protected_input_preserves_raw_spans_and_duplicate_occurrences():
    source = "“12 명”과 12 명, 12 명, ppt를 보존한다."
    data = json.loads(draft_input(source))
    assert data["source"] == source
    assert data["protected_spans"] == ["“12 명”", "12 명", "12 명", "ppt"]
    for changed in [
        source.replace("ppt", "PPT"),
        source.replace("12 명", "13 명", 1),
        source.replace("“12 명”", "“12명”"),
    ]:
        assert not check_protection(source, changed)["mechanical_pass"]


SOURCE = "우리는 인공지능이 만든 시험용 글을 읽었다. 실제 사람 글은 아니다. " * 3
DRAFT = "우리가 읽은 글은 인공지능이 만든 시험용 글이다. 실제 사람 글은 아니다. " * 3


def indexed_response(**changes):
    data = dict(
        genre="experience_review",
        standalone="pass",
        forward="pass",
        backward="pass",
        logic="pass",
        stance="pass",
        register_match="pass",
        source_evidence_ids=["S000"],
        draft_evidence_ids=["D000"],
        genre_evidence_ids=["S001"],
    )
    return json.dumps(data | changes)


def test_indexed_evidence_is_lossless_side_bound_and_not_human_acceptance():
    units = evidence_units(SOURCE, "S")
    assert "".join(u["text"] for u in units) == SOURCE
    assert all(u["text"] == SOURCE[u["start"] : u["end"]] and len(u["text"]) <= 30 for u in units)
    r = screen(SOURCE, DRAFT, indexed_response(), evidence_contract="indexed-v3")
    assert r["automatic_pass"] and not r["human_review_completed"]
    assert r["pair_review_status"] == "pending"
    assert r["resolved_evidence"]["genre_evidence_ids"][0]["text"] == SOURCE[30:60]


def test_evidence_tail_is_not_one_character_and_punctuation_is_not_substantive():
    assert [len(u["text"]) for u in evidence_units("가" * 31, "S")] == [29, 2]
    with pytest.raises(ValueError, match="substantive"):
        resolve_evidence(". " * 30, DRAFT, json.loads(indexed_response()))


@pytest.mark.parametrize(
    "change",
    [
        {"source_evidence_ids": ["D000"]},
        {"genre_evidence_ids": ["S999"]},
        {"source_evidence_ids": ["S000", "S000"]},
        {"draft_evidence_ids": []},
        {"source_evidence_ids": ["S000", "S001", "S002", "S003"]},
        {"source_evidence_ids": "S000"},
        {"forward": "unclear"},
        {"genre": "unclassified"},
        {"source_quote": "fabricated extra field"},
    ],
)
def test_indexed_protocol_rejects_invalid_evidence_and_uncertainty(change):
    assert not screen(SOURCE, DRAFT, indexed_response(**change), evidence_contract="indexed-v3")[
        "automatic_pass"
    ]


def test_protocol_cannot_silently_upgrade_archived_responses_or_rescue_number_change():
    assert not screen(SOURCE, DRAFT, indexed_response())["automatic_pass"]
    assert (
        screen(
            SOURCE + " 12명", DRAFT + " 13명", indexed_response(), evidence_contract="indexed-v3"
        )["reason"]
        == "mechanical_protection_failed"
    )
    with pytest.raises(ValueError):
        screen(SOURCE, DRAFT, "{}", evidence_contract="unknown")
