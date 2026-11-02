"""Prepare completion-only SFT inputs offline; never opens a GPU or accepts mock pairs."""

import json
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import Field

from korean_humanizer.data import source_errors
from korean_humanizer.io import read_jsonl, sha256_file, write_json, write_jsonl
from korean_humanizer.models import Pair, Record
from korean_humanizer.token_audit import render_chat
from korean_humanizer.training import sft_records


class SFTConfig(Record):
    data_policy: Literal["accepted_only", "pending_research"] = "accepted_only"
    model_id: str = "Qwen/Qwen3-4B"
    revision: str = "1cfa9a7208912126459214e8b04321603b3df60c"
    method: str = "bf16_lora"
    max_sequence_tokens: int = Field(default=2048, ge=64, le=8192)
    max_steps: int = Field(default=32, ge=1, le=10000)
    gradient_accumulation_steps: int = Field(default=8, ge=1, le=32)
    learning_rate: float = Field(default=0.0001, gt=0, le=0.001)
    lora_rank: int = Field(default=16, ge=1, le=64)
    lora_alpha: int = Field(default=32, ge=1, le=128)
    lora_dropout: float = Field(default=0.05, ge=0, lt=1)
    seed: int = 42
    min_train_examples: int = Field(default=240, ge=1)
    min_dev_examples: int = Field(default=30, ge=1)
    eval_steps: int = Field(default=8, ge=1, le=1000)
    save_steps: int = Field(default=8, ge=1, le=1000)
    warmup_steps: int = Field(default=1, ge=0, le=1000)


def encode_completion(messages, tokenizer, tokenizer_config, max_tokens):
    if [m["role"] for m in messages] != ["system", "user", "assistant"]:
        raise ValueError("expected exactly system/user/assistant messages")
    # Reserved model control markers must never be introduced by data content.
    for message in messages:
        if any(
            token in message["content"]
            for token in (str(t) for t in tokenizer.get_added_tokens_decoder().values())
        ):
            raise ValueError("data contains a tokenizer control token")
    # Qwen assistant rendering accesses these optional fields under StrictUndefined.
    rendered_messages = [dict(m, tool_calls=None, reasoning_content=None) for m in messages]
    body = {"messages": rendered_messages[:2], "chat_template_kwargs": {"enable_thinking": False}}
    prompt = render_chat(tokenizer_config["chat_template"], body, tokenizer_config)
    body["messages"] = rendered_messages
    full = render_chat(
        tokenizer_config["chat_template"], body, tokenizer_config, add_generation_prompt=False
    )
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    full_ids = tokenizer.encode(full, add_special_tokens=False).ids
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("token boundary does not match generation prompt; refusing label guess")
    if len(full_ids) > max_tokens:
        raise ValueError("example exceeds context limit; no silent truncation")
    eos_id = tokenizer.token_to_id(tokenizer_config["eos_token"])
    target = full_ids[len(prompt_ids) :]
    if eos_id not in target or len(target) <= 1:
        raise ValueError("completion needs answer tokens and EOS")
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": [-100] * len(prompt_ids) + target,
        "prompt_tokens": len(prompt_ids),
    }


def prepare_sft(
    pairs_path: Path,
    metadata: Path,
    config_path: Path,
    output: Path,
    *,
    allow_pending_research: bool = False,
) -> dict:
    from tokenizers import Tokenizer

    config = SFTConfig.model_validate_json(config_path.read_text())
    if (config.data_policy == "pending_research") != allow_pending_research:
        raise ValueError("pending research requires both its config policy and explicit opt-in")
    if (config.model_id, config.revision, config.method) != (
        "Qwen/Qwen3-4B",
        "1cfa9a7208912126459214e8b04321603b3df60c",
        "bf16_lora",
    ):
        raise ValueError("first pilot supports only pinned 4B BF16 LoRA")
    pairs = read_jsonl(pairs_path, Pair)
    records = sft_records(pairs, allow_pending_research=allow_pending_research)
    for pair in pairs:
        if source_errors(pair.source, external_transfer=True):
            raise ValueError("RunPod requires documented external processing permission")
    counts = Counter(r["split"] for r in records)
    if counts["train"] < config.min_train_examples or counts["dev"] < config.min_dev_examples:
        raise ValueError("insufficient eligible train/dev examples for this pilot")
    provenance = json.loads((metadata / "metadata.json").read_text())
    if (provenance["model_id"], provenance["revision"]) != (config.model_id, config.revision):
        raise ValueError("tokenizer revision mismatch")
    for name in ["tokenizer.json", "tokenizer_config.json"]:
        if sha256_file(metadata / name) != provenance["files"][name]["sha256"]:
            raise ValueError("tokenizer integrity failure")
    tokenizer = Tokenizer.from_file(str(metadata / "tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    tokenizer_config = json.loads((metadata / "tokenizer_config.json").read_text())
    data = {"train": [], "dev": []}
    for row in records:
        encoded = encode_completion(
            row["messages"], tokenizer, tokenizer_config, config.max_sequence_tokens
        )
        data[row["split"]].append({"pair_id": row["pair_id"], **encoded})
        if row.get("input_kind") == "natural_control":
            data[row["split"]][-1]["input_kind"] = "natural_control"
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    for split, rows in data.items():
        write_jsonl(output / (split + ".tokens.jsonl"), rows)
    stored_config = config.model_dump()
    # Older frozen recipes retain byte-identical configs and manifest hashes.
    for field in ("eval_steps", "save_steps", "warmup_steps"):
        if field not in config.model_fields_set:
            stored_config.pop(field)
    write_json(output / "config.json", stored_config)
    for name in ["tokenizer.json", "tokenizer_config.json", "metadata.json"]:
        (output / name).write_bytes((metadata / name).read_bytes())
    manifest = {
        "status": "prepared_not_trained",
        "training_executed": False,
        "data_policy": config.data_policy,
        "human_semantic_review_completed": not allow_pending_research,
        "review_status_counts": dict(Counter(p.review_status for p in pairs)),
        "pair_source_sha256": sha256_file(pairs_path),
        "counts": dict(counts),
        "config_sha256": sha256_file(config_path),
        "files": {p.name: sha256_file(p) for p in sorted(output.iterdir()) if p.is_file()},
        "loss": "assistant_completion_only; includes EOS; prompts and padding masked",
        "baseline_survey_required_before_training": False,
        "data_pair_acceptance_required": not allow_pending_research,
        "quality_verdict": "not_assessed",
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def load_sft(bundle: Path):
    manifest = json.loads((bundle / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if Path(name).name != name or sha256_file(bundle / name) != expected:
            raise ValueError("SFT bundle integrity failure")
    config = SFTConfig.model_validate_json((bundle / "config.json").read_text())
    if manifest.get("data_policy", "accepted_only") != config.data_policy:
        raise ValueError("SFT data policy mismatch")
    if config.data_policy == "pending_research" and (
        manifest.get("human_semantic_review_completed") is not False
        or manifest.get("data_pair_acceptance_required") is not False
        or manifest.get("review_status_counts") != {"pending": sum(manifest["counts"].values())}
    ):
        raise ValueError("pending research must retain its unreviewed status")
    splits = {}
    for split in ["train", "dev"]:
        rows = [
            json.loads(line)
            for line in (bundle / (split + ".tokens.jsonl")).read_text().splitlines()
        ]
        if len(rows) != manifest["counts"][split] or not rows:
            raise ValueError("SFT example count mismatch")
        for row in rows:
            ids, labels, mask = row["input_ids"], row["labels"], row["attention_mask"]
            boundary = row["prompt_tokens"]
            if not (
                0 < boundary < len(ids) <= config.max_sequence_tokens
                and len(ids) == len(labels) == len(mask)
                and labels == [-100] * boundary + ids[boundary:]
                and mask == [1] * len(ids)
            ):
                raise ValueError("invalid completion labels or sequence length")
        splits[split] = rows
    if set(r["pair_id"] for r in splits["train"]) & set(r["pair_id"] for r in splits["dev"]):
        raise ValueError("pair IDs leak between train and dev")
    return manifest, config, splits
