import pytest

from korean_humanizer.data import assign_splits, source_errors
from korean_humanizer.models import Source


def test_optimized_grouping_matches_exhaustive_similarity_rules(source):
    import random
    from difflib import SequenceMatcher

    from korean_humanizer.data import normalized

    randomizer = random.Random(207)
    texts = []
    for _ in range(12):
        base = "합성검사공통문구" + "".join(
            chr(0xAC00 + randomizer.randrange(80)) for _ in range(190)
        )
        texts.extend(
            [
                base,
                base[:80] + "검사" + base[80:],
                base[:40] + "수정한문구" + base[45:],
                base[40:] + base[:40],
            ]
        )
    rows = [
        source.model_copy(
            update={
                "source_doc_id": str(i),
                "author_group": str(i),
                "source_group": str(i),
                "text": text,
            }
        )
        for i, text in enumerate(texts)
    ]
    texts = [normalized(t) for t in texts]
    shingles = [{t[i : i + 5] for i in range(len(t) - 4)} for t in texts]
    parents = list(range(len(rows)))

    def find(i):
        while parents[i] != i:
            i = parents[i]
        return i

    for i in range(len(rows)):
        for j in range(i):
            overlap = shingles[i] & shingles[j]
            if not overlap:
                continue
            if (
                len(overlap) / len(shingles[i] | shingles[j]) >= 0.85
                or SequenceMatcher(None, texts[i], texts[j], autojunk=False).ratio() >= 0.95
            ):
                parents[find(i)] = find(j)
    expected = {
        frozenset(str(j) for j in range(len(rows)) if find(j) == find(i)) for i in range(len(rows))
    }
    actual = assign_splits(rows)
    observed = {
        frozenset(r.source.source_doc_id for r in actual if r.group_id == x.group_id)
        for x in actual
    }
    assert observed == expected


def test_unknown_rights_and_dates_cannot_be_used(source):
    human = source.model_copy(update={"origin": "human", "source_year": None})
    assert "unverified_authorship_year" in source_errors(human)
    pending = human.model_copy(
        update={
            "rights": human.rights.model_copy(update={"training": "pending", "evidence_ref": ""})
        }
    )
    assert {"training_rights_not_approved", "missing_rights_evidence"} <= set(
        source_errors(pending)
    )


def test_fixture_and_third_party_permissions_are_separate(source):
    assert "synthetic_fixture_not_human_training_data" in source_errors(source)
    assert source_errors(source, allow_fixture=True) == []
    assert "third_party_transfer_not_approved" in source_errors(
        source, allow_fixture=True, external_transfer=True
    )


def test_year_policy_requires_explicit_override(source):
    human = source.model_copy(update={"origin": "human", "source_year": 2020})
    assert "outside_initial_pre_2019_policy" in source_errors(human)
    assert source_errors(human, allow_later_years=True) == []


def test_grouping_is_transitive_and_input_order_independent(source):
    b = source.model_copy(
        update={"source_doc_id": "b", "text": "서로 다른 문장입니다.", "source_group": "second"}
    )
    c = source.model_copy(
        update={
            "source_doc_id": "c",
            "author_group": "other",
            "text": "또 다른 문장입니다.",
            "source_group": "second",
        }
    )
    rows = [source, b, c]
    first = assign_splits(rows)
    assert len({row.group_id for row in first}) == 1
    assert len({row.split for row in first}) == 1
    assert {r.source.source_doc_id: r.split for r in first} == {
        r.source.source_doc_id: r.split for r in assign_splits(rows[::-1])
    }


def test_near_duplicates_stay_together(source):
    text = (
        "이 연구는 지역별 강수량과 토양 수분을 비교하고 계절에 따라 달라지는 관측 결과를 기록했다."
    )
    a = source.model_copy(update={"text": text})
    b = source.model_copy(
        update={"source_doc_id": "b", "author_group": "b", "source_group": "b", "text": text + "\n"}
    )
    c = source.model_copy(
        update={
            "source_doc_id": "c",
            "author_group": "c",
            "source_group": "c",
            "text": text.replace("기록했다", "기록하였다"),
        }
    )
    assert len({row.group_id for row in assign_splits([a, b, c])}) == 1


def test_duplicate_ids_and_invalid_fraction_fail(source):
    with pytest.raises(ValueError, match="unique"):
        assign_splits([source, source])
    with pytest.raises(ValueError, match="fractions"):
        assign_splits([source], dev_fraction=0.9, test_fraction=0.2)


def test_schema_rejects_unknown_fields_and_blank_text(source):
    from pydantic import ValidationError

    for update in ({"text": "  "}, {"unexpected": True}):
        with pytest.raises(ValidationError):
            Source.model_validate(source.model_dump() | update)
