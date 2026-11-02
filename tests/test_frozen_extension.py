"""Split extension uses synthetic records; no actual human source text."""

import pytest

from korean_humanizer.data import extend_frozen_splits
from korean_humanizer.models import AssignedSource, Rights, Source


def source(identifier, author, group, text):
    return Source(
        source_doc_id=identifier,
        author_group=author,
        source_group=group,
        genre="fixture",
        text=text,
        provenance="fixture:synthetic-only",
        origin="synthetic_fixture",
        rights=Rights(),
    )


def frozen_rows():
    return [
        AssignedSource(
            source=source("a", "author-a", "document-a", "가" * 80),
            split="train",
            group_id="frozen-a",
        ),
        AssignedSource(
            source=source("b", "author-b", "document-b", "나" * 80),
            split="test",
            group_id="frozen-b",
        ),
    ]


def test_new_documents_inherit_frozen_author_splits_without_redrawing_old_rows():
    frozen = frozen_rows()
    original = [r.model_dump() for r in frozen]
    additions = [source("c", "author-b", "document-c", "다" * 80)]
    result, quarantine = extend_frozen_splits(frozen, additions)
    assert [r.model_dump() for r in result[:2]] == original
    assert result[-1].split == "test"
    assert result[-1].group_id == "frozen-b"
    assert quarantine == []


def test_bridging_two_frozen_splits_quarantines_new_sources_without_moving_old_data():
    frozen = frozen_rows()
    bridge = source("bridge", "author-a", "document-b", "라" * 80)
    result, quarantine = extend_frozen_splits(frozen, [bridge])
    assert result == frozen
    assert quarantine == [bridge]


def test_frozen_text_changes_are_rejected_and_normalized_duplicates_inherit_split():
    frozen = frozen_rows()
    with pytest.raises(ValueError, match="conflicting"):
        extend_frozen_splits(frozen, [source("a", "author-a", "document-a", "변경")])
    result, quarantine = extend_frozen_splits(
        frozen, [source("duplicate", "other", "other", "가 " * 80)]
    )
    assert quarantine == []
    assert result[-1].split == "train"
