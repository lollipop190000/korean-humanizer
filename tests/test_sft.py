"""Synthetic software fixtures only: no human corpus and no actual model training."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from korean_humanizer.io import sha256_file, write_json, write_jsonl
from korean_humanizer.models import Generation, Pair, Rights, Source
from korean_humanizer.sft import SFTConfig, encode_completion, load_sft, prepare_sft
from korean_humanizer.training import INSTRUCTION

ROOT = Path(__file__).resolve().parents[1]


def fixture_bundle(tmp_path):
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    tokenizer = Tokenizer(
        models.WordLevel(
            {"[UNK]": 0, "[EOS]": 1, "입력": 2, "정답": 3, "원문": 4}, unk_token="[UNK]"
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.add_special_tokens(["[EOS]"])
    tokenizer.save(str(metadata / "tokenizer.json"))
    config = {
        "eos_token": "[EOS]",
        "chat_template": "{% for m in messages %}{{ m['role'] }} {{ m['content'] }} "
        "{% if m['role']=='assistant' %}[EOS]{% endif %}{% endfor %}"
        "{% if add_generation_prompt %}assistant {% endif %}",
    }
    write_json(metadata / "tokenizer_config.json", config)
    spec = SFTConfig(min_train_examples=1, min_dev_examples=1)
    write_json(
        metadata / "metadata.json",
        dict(
            model_id=spec.model_id,
            revision=spec.revision,
            files={
                n: {"sha256": sha256_file(metadata / n)}
                for n in ["tokenizer.json", "tokenizer_config.json"]
            },
        ),
    )
    # These Source.origin values exercise production validation; they are not real source evidence.
    pairs = []
    for name, split, text in [("a", "train", "정답"), ("b", "dev", "원문")]:
        source = Source(
            source_doc_id=name,
            author_group=name,
            source_group=name,
            genre="fixture",
            text=text,
            provenance="fixture:software-only",
            origin="human",
            source_year=2017,
            rights=Rights(
                training="approved",
                derivative_creation="approved",
                third_party_transfer="approved",
                evidence_ref="fixture:not-a-real-license",
            ),
        )
        pairs.append(
            Pair(
                pair_id=name,
                source=source,
                synthetic_ai_text="입력",
                split=split,
                generation=Generation(
                    model="fixture", prompt_version="fixture", seed=42, mock=False
                ),
                review_status="accepted",
                review_ref="fixture:not-human-review",
            )
        )
    write_jsonl(tmp_path / "pairs.jsonl", pairs)
    write_json(tmp_path / "config.json", spec.model_dump())
    return metadata, pairs, tokenizer, config


def test_completion_mask_and_padding_do_not_train_on_prompt(tmp_path):
    _, _, tokenizer, config = fixture_bundle(tmp_path)
    messages = [
        {"role": "system", "content": "지시"},
        {"role": "user", "content": "입력"},
        {"role": "assistant", "content": "정답"},
    ]
    row = encode_completion(messages, tokenizer, config, 100)
    n = row["prompt_tokens"]
    assert row["labels"][:n] == [-100] * n
    assert row["labels"][n:] == row["input_ids"][n:]
    assert tokenizer.token_to_id("[EOS]") in row["labels"][n:]
    module_spec = importlib.util.spec_from_file_location("train_sft", ROOT / "scripts/train_sft.py")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    second = {k: v + [1] for k, v in row.items() if k in ["input_ids", "attention_mask", "labels"]}
    padded = module.pad_batch([row, second], 0)
    assert padded["labels"][0][-1] == -100
    assert padded["attention_mask"][0][-1] == 0
    with pytest.raises(ValueError, match="truncation"):
        encode_completion(messages, tokenizer, config, 2)
    messages[-1]["content"] = "정답 [EOS]"
    with pytest.raises(ValueError, match="control token"):
        encode_completion(messages, tokenizer, config, 100)


def test_production_bundle_and_dry_run_are_offline(tmp_path):
    metadata, _, _, _ = fixture_bundle(tmp_path)
    output = tmp_path / "bundle"
    report = prepare_sft(tmp_path / "pairs.jsonl", metadata, tmp_path / "config.json", output)
    assert report["training_executed"] is False
    repeat = tmp_path / "repeat"
    prepare_sft(tmp_path / "pairs.jsonl", metadata, tmp_path / "config.json", repeat)
    assert sha256_file(repeat / "manifest.json") == sha256_file(output / "manifest.json")
    _, config, splits = load_sft(output)
    assert len(splits["train"]) == len(splits["dev"]) == 1
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/train_sft.py"), "--bundle", str(output)],
        text=True,
        capture_output=True,
        check=True,
    )
    assert json.loads(completed.stdout)["network_calls"] == 0
    assert config.max_steps == 32
    (output / "train.tokens.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="integrity"):
        load_sft(output)


@pytest.mark.parametrize("fault", ["mock", "pending", "transfer", "counts", "revision"])
def test_unready_data_never_creates_a_training_bundle(tmp_path, fault):
    metadata, pairs, _, _ = fixture_bundle(tmp_path)
    if fault == "mock":
        pairs[0].generation.mock = True
    elif fault == "pending":
        pairs[0].review_status = "pending"
    elif fault == "transfer":
        pairs[0].source.rights.third_party_transfer = "pending"
    elif fault == "counts":
        (tmp_path / "config.json").write_text(SFTConfig().model_dump_json())
    else:
        meta = json.loads((metadata / "metadata.json").read_text())
        meta["revision"] = "wrong"
        (metadata / "metadata.json").write_text(json.dumps(meta))
    (tmp_path / "pairs.jsonl").unlink()
    write_jsonl(tmp_path / "pairs.jsonl", pairs)
    with pytest.raises(ValueError):
        prepare_sft(
            tmp_path / "pairs.jsonl", metadata, tmp_path / "config.json", tmp_path / "bundle"
        )
    assert not (tmp_path / "bundle").exists()


def test_first_sft_uses_same_instruction_as_baseline():
    spec = json.loads((ROOT / "configs/baselines/qwen3-4b.json").read_text())
    assert spec["instruction"] == INSTRUCTION


def test_scale_recipe_has_two_epochs_and_less_frequent_checkpoints():
    config = SFTConfig.model_validate_json((ROOT / "configs/sft-scale-001.v1.json").read_text())
    assert config.max_steps * config.gradient_accumulation_steps == config.min_train_examples * 2
    assert config.min_dev_examples == 200
    assert config.data_policy == "pending_research"
    assert config.save_steps == config.eval_steps == 100
    assert config.warmup_steps < config.max_steps
    # Older prepared bundles retain the exact runtime schedule they used.
    old = SFTConfig.model_validate_json((ROOT / "configs/sft-first-research.v1.json").read_text())
    assert (old.max_steps, old.eval_steps, old.save_steps, old.warmup_steps) == (16, 8, 8, 1)


def test_workflow_budget_covers_all_phases_and_generation_counts_include_calibration():
    plan = json.loads((ROOT / "configs/scale-001.workflow.json").read_text())
    budget = plan["budget_proposal"]
    maximum = budget["max_aggregate_gpu_hours"] * (
        budget["max_gpu_hourly_usd"] + budget["max_ephemeral_disk_gb"] * 0.1 / 720
    )
    assert maximum + budget["cost_reserve_usd"] <= budget["total_usd"]
    assert plan["generation"]["calibration_part_of_request_limits"] is True
    assert plan["generation"]["automatic_request_retries"] == 0
    assert plan["quality"]["survey_results_required_before_training"] is False


def test_pending_research_retains_status_and_needs_two_explicit_choices(tmp_path):
    metadata, pairs, _, _ = fixture_bundle(tmp_path)
    for pair in pairs:
        pair.review_status = "pending"
        pair.review_ref = ""
    (tmp_path / "pairs.jsonl").unlink()
    write_jsonl(tmp_path / "pairs.jsonl", pairs)
    config_path = tmp_path / "config.json"
    with pytest.raises(ValueError, match="opt-in"):
        prepare_sft(
            tmp_path / "pairs.jsonl",
            metadata,
            config_path,
            tmp_path / "out",
            allow_pending_research=True,
        )
    config_path.unlink()
    write_json(
        config_path,
        SFTConfig(
            data_policy="pending_research", min_train_examples=1, min_dev_examples=1
        ).model_dump(),
    )
    with pytest.raises(ValueError, match="opt-in"):
        prepare_sft(tmp_path / "pairs.jsonl", metadata, config_path, tmp_path / "out")
    original = (tmp_path / "pairs.jsonl").read_bytes()
    report = prepare_sft(
        tmp_path / "pairs.jsonl",
        metadata,
        config_path,
        tmp_path / "out",
        allow_pending_research=True,
    )
    assert report["review_status_counts"] == {"pending": 2}
    assert report["human_semantic_review_completed"] is False
    assert report["data_pair_acceptance_required"] is False
    assert (tmp_path / "pairs.jsonl").read_bytes() == original
    load_sft(tmp_path / "out")
    from korean_humanizer.training import sft_records

    with pytest.raises(ValueError, match="documented acceptance"):
        sft_records(pairs)


@pytest.mark.parametrize("fault", ["rejected", "test", "mock", "rights", "protection", "leakage"])
def test_research_policy_keeps_other_data_guards(tmp_path, fault):
    metadata, pairs, _, _ = fixture_bundle(tmp_path)
    for pair in pairs:
        pair.review_status = "pending"
        pair.review_ref = ""
    if fault == "rejected":
        pairs[0].review_status = "rejected"
    elif fault == "test":
        pairs[0].split = "test"
    elif fault == "mock":
        pairs[0].generation.mock = True
    elif fault == "rights":
        pairs[0].source.rights.third_party_transfer = "pending"
    elif fault == "protection":
        pairs[0].synthetic_ai_text = "입력 12명"
    else:
        pairs[0].source.author_group = pairs[1].source.author_group
    (tmp_path / "pairs.jsonl").unlink()
    write_jsonl(tmp_path / "pairs.jsonl", pairs)
    (tmp_path / "config.json").unlink()
    write_json(
        tmp_path / "config.json",
        SFTConfig(
            data_policy="pending_research", min_train_examples=1, min_dev_examples=1
        ).model_dump(),
    )
    with pytest.raises(ValueError):
        prepare_sft(
            tmp_path / "pairs.jsonl",
            metadata,
            tmp_path / "config.json",
            tmp_path / "out",
            allow_pending_research=True,
        )
    assert not (tmp_path / "out").exists()
