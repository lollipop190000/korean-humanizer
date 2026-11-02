import pytest

from korean_humanizer.models import AssignedSource
from korean_humanizer.synthesis import generate_mock_pair
from korean_humanizer.training import sft_records


def pair_for(source, split="train"):
    return generate_mock_pair(AssignedSource(source=source, split=split, group_id="group"))


def test_test_split_is_locked_at_generation_and_export(source):
    with pytest.raises(ValueError, match="locked"):
        pair_for(source, "test")
    pair = pair_for(source).model_copy(update={"split": "test"})
    with pytest.raises(ValueError, match="test pairs"):
        sft_records([pair], allow_mock=True)


def test_mock_never_silently_becomes_real_training_data(source):
    pair = pair_for(source)
    with pytest.raises(ValueError, match="production export"):
        sft_records([pair])
    row = sft_records([pair], allow_mock=True)[0]
    assert row["purpose"] == "mock_smoke_only"
    assert row["messages"][-1]["content"] == source.text


def test_pending_review_and_protected_change_block_export(source):
    human = source.model_copy(update={"origin": "human", "source_year": 2017})
    pair = pair_for(source).model_copy(
        update={
            "source": human,
            "review_status": "pending",
            "generation": pair_for(source).generation.model_copy(update={"mock": False}),
        }
    )
    with pytest.raises(ValueError, match="documented acceptance"):
        sft_records([pair])
    accepted = pair.model_copy(update={"review_status": "accepted", "review_ref": "review:001"})
    assert sft_records([accepted])[0]["purpose"] == "sft"
    broken = accepted.model_copy(
        update={"synthetic_ai_text": accepted.synthetic_ai_text.replace("25", "26")}
    )
    with pytest.raises(ValueError, match="protected information"):
        sft_records([broken])


def test_cross_split_author_leak_blocks_export(source):
    second = source.model_copy(
        update={
            "source_doc_id": "another",
            "source_group": "other",
            "text": "전혀 다른 내용을 썼다.",
            "protected_spans": [],
        }
    )
    with pytest.raises(ValueError, match="leak"):
        sft_records([pair_for(source), pair_for(second, "dev")], allow_mock=True)
