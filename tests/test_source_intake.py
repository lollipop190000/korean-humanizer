"""Synthetic examples exercising intake guards; no human corpus data."""

import pytest

from korean_humanizer.models import AssignedSource, Rights, Source
from korean_humanizer.source_intake import prepare_intake


def source(identifier, text, **updates):
    values = dict(
        source_doc_id=identifier,
        author_group=identifier,
        source_group=identifier,
        genre="unclassified",
        text=text,
        provenance="fixture:synthetic-test-not-real-evidence",
        origin="human",  # Exercise the guard, not an assertion of human authorship.
        source_year=2018,
        rights=Rights(),
    )
    return Source(**(values | updates))


def base():
    return [
        AssignedSource(source=source("a", "가" * 200), split="train", group_id="a"),
        AssignedSource(source=source("b", "나" * 200), split="test", group_id="b"),
    ]


def test_excludes_duplicates_and_bad_years_without_changing_rights_or_frozen_rows():
    frozen = base()
    additions = [
        source("copy", "가 " * 200),
        source("undated", "다" * 200, source_year=None),
        source("recent", "라" * 200, source_year=2020),
        source("new", "마" * 200, author_group="b"),
        frozen[0].source,
    ]
    extended, excluded, report = prepare_intake(frozen, additions)
    assert extended[:2] == frozen
    assert len(extended) == 4 and extended[-1].split == "test"
    assert report["new_sources"] == 1
    assert report["linkage_only_rows"] == 1
    assert extended[-1].source.rights == Rights()
    assert report["reused_frozen_sources"] == 1
    assert {r["reason"] for r in excluded} == {
        "normalized_duplicate",
        "missing_document_year",
        "outside_pre_2019_scope",
    }


def test_bridge_conflict_and_changed_history_cannot_enter_training():
    frozen = base()
    bridge = source("bridge", "다" * 200, author_group="a", source_group="b")
    extended, excluded, _ = prepare_intake(frozen, [bridge])
    assert extended == frozen
    assert excluded[0]["reason"] == "conflicting_frozen_splits"
    with pytest.raises(ValueError, match="conflicting metadata"):
        prepare_intake(frozen, [source("a", "변경" * 100)])
    with pytest.raises(ValueError, match="duplicate candidate"):
        prepare_intake(frozen, [bridge, bridge])


def test_repeated_intake_preserves_completed_sources():
    frozen = base()
    candidate = source("new", "다" * 200)
    first, _, _ = prepare_intake(frozen, [candidate])
    second, _, report = prepare_intake(first, [candidate])
    assert second == first
    assert report["new_sources"] == 0
    assert report["reused_frozen_sources"] == 1


def test_duplicate_identity_link_keeps_other_author_documents_out_of_training():
    frozen = base()
    duplicate = source("copy", "나 " * 200, author_group="other-author")
    related = source("related", "다" * 200, author_group="other-author")
    extended, excluded, report = prepare_intake(frozen, [duplicate, related])
    assert all(r.split == "test" for r in extended[2:])
    assert excluded == [{"source_doc_id": "copy", "reason": "normalized_duplicate"}]
    assert report["new_sources_by_split"] == {"test": 1}
