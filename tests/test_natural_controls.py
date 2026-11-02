"""Synthetic fixtures only; these tests do not construct actual human training examples."""

import pytest
from pydantic import ValidationError

from korean_humanizer.models import Generation, Pair, Rights, Source
from korean_humanizer.training import sft_records


def pair():
    return Pair(
        pair_id="fixture-natural",
        source=Source(
            source_doc_id="fixture-source",
            source_group="fixture-doc",
            author_group="fixture-author",
            text="AI가 작성한 소프트웨어 시험 문장이다. 사람 학습 원문이 아니다.",
            genre="fixture",
            provenance="fixture:not-real-rights",
            origin="synthetic_fixture",
            rights=Rights(
                training="approved", derivative_creation="approved", evidence_ref="fixture"
            ),
        ),
        synthetic_ai_text="AI가 작성한 소프트웨어 시험 문장이다. 사람 학습 원문이 아니다.",
        split="train",
        generation=Generation(
            model="identity-control", prompt_version="natural-control-v1", seed=42, mock=False
        ),
        input_kind="natural_control",
    )


def test_control_serialization_is_explicit_and_old_ai_pair_serialization_is_unchanged():
    control = pair()
    assert control.model_dump()["input_kind"] == "natural_control"
    old = control.model_copy(
        update={
            "input_kind": "ai_draft",
            "generation": Generation(
                model="fixture-generator", prompt_version="fixture", seed=42, mock=True
            ),
        }
    )
    assert "input_kind" not in old.model_dump()
    assert Pair.model_validate(old.model_dump()).input_kind == "ai_draft"


def test_control_cannot_claim_an_ai_generator_or_change_text():
    control = pair().model_dump()
    control["synthetic_ai_text"] += " 추측을 추가한다."
    with pytest.raises(ValidationError, match="identical"):
        Pair.model_validate(control)
    control = pair().model_dump()
    control["generation"]["model"] = "Qwen/Qwen3-8B"
    with pytest.raises(ValidationError, match="non-AI"):
        Pair.model_validate(control)


def test_fixture_is_not_exportable_as_real_research_and_copied_invalid_control_is_rejected():
    with pytest.raises(ValueError, match="real, pending"):
        sft_records([pair()], allow_pending_research=True)
    changed = pair().model_copy(update={"synthetic_ai_text": "의미가 바뀌었다."})
    with pytest.raises(ValueError, match="identical"):
        sft_records([changed], allow_pending_research=True)


def test_natural_input_label_survives_export_without_claiming_ai_generation_or_human_acceptance():
    # Simulate source metadata to exercise the export branch; this remains an AI-written fixture.
    control = pair()
    control = control.model_copy(
        update={
            "source": control.source.model_copy(update={"origin": "human", "source_year": 2018})
        }
    )
    row = sft_records([control], allow_pending_research=True)[0]
    assert row["input_kind"] == "natural_control"
    assert row["messages"][1]["content"] == row["messages"][2]["content"]
    assert control.review_status == "pending"
    assert control.generation.model == "identity-control"
    with pytest.raises(ValueError, match="documented acceptance"):
        sft_records([control])
