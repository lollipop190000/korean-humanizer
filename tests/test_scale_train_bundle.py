"""Synthetic validation of the paid training boundary; no model or network calls."""

import pytest
from test_scale_draft_bundle import module
from test_sft import fixture_bundle

from korean_humanizer.models import Generation


def target_pairs(tmp_path):
    _, seeds, _, _ = fixture_bundle(tmp_path)
    validator = module("scale_train_bundle")
    result = []
    for split, quotas in validator.QUOTAS.items():
        for genre, count in quotas.items():
            for i in range(count):
                key = f"{split}-{genre}-{i}"
                source = seeds[0].source.model_copy(
                    update={
                        "source_doc_id": key,
                        "author_group": f"{split}-{genre}-{i // 20}",
                        "genre": genre,
                    }
                )
                control = i < count // 10
                result.append(
                    seeds[0].model_copy(
                        update={
                            "pair_id": key,
                            "source": source,
                            "split": split,
                            "synthetic_ai_text": source.text if control else "입력",
                            "input_kind": "natural_control" if control else "ai_draft",
                            "review_status": "pending",
                            "generation": Generation(
                                model="identity-control" if control else "fixture-generator",
                                prompt_version="natural-control-v1" if control else "fixture",
                                seed=0,
                                mock=False,
                            ),
                        }
                    )
                )
    return validator, result


def test_small_or_unbalanced_dataset_cannot_enter_scale_training(tmp_path):
    value, rows = target_pairs(tmp_path)
    value.validate_pairs(rows)
    with pytest.raises(ValueError, match="exactly 2000"):
        value.validate_pairs(rows[:-1])
    rows[-1] = rows[-1].model_copy(
        update={"source": rows[-1].source.model_copy(update={"genre": "experience_review"})}
    )
    with pytest.raises(ValueError, match="genre quotas"):
        value.validate_pairs(rows)


def test_controls_must_have_distinct_source_excerpts(tmp_path):
    value, rows = target_pairs(tmp_path)
    rows[1] = rows[1].model_copy(update={"source": rows[0].source})
    with pytest.raises(ValueError, match="disjoint excerpts"):
        value.validate_pairs(rows)
