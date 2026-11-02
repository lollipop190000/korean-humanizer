"""Freeze or verify the substantive scale SFT and comparison bundle on CPU."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from korean_humanizer.benchmark import BaselineSpec, digest, request_body
from korean_humanizer.io import read_jsonl, sha256_file, write_json, write_jsonl
from korean_humanizer.models import AssignedSource, Pair
from korean_humanizer.sft import load_sft, prepare_sft
from korean_humanizer.source_scope import exclusion_reason
from korean_humanizer.token_audit import render_chat
from korean_humanizer.training import INSTRUCTION

QUOTAS = {
    "train": {
        "experience_review": 600,
        "explanation_information": 800,
        "opinion_column": 400,
        "practical_notice": 200,
    },
    "dev": {
        "experience_review": 60,
        "explanation_information": 80,
        "opinion_column": 40,
        "practical_notice": 20,
    },
}


def validate_pairs(pairs):
    if Counter(p.split for p in pairs) != {"train": 2000, "dev": 200}:
        raise ValueError("scale SFT requires exactly 2000 train and 200 dev")
    if len({p.source.source_doc_id for p in pairs}) != 2200:
        raise ValueError("control and reverse examples require disjoint excerpts")
    for split, genres in QUOTAS.items():
        selected = [p for p in pairs if p.split == split]
        if Counter(p.source.genre for p in selected) != genres:
            raise ValueError("scale genre quotas are not satisfied")
        if sum(p.input_kind == "natural_control" for p in selected) != (
            200 if split == "train" else 20
        ):
            raise ValueError("scale natural-control quota differs")
        if split == "train" and max(Counter(p.source.author_group for p in selected).values()) > 20:
            raise ValueError("scale per-author train cap exceeded")
    if any(exclusion_reason(p.source.text) for p in pairs):
        raise ValueError("source outside the reviewed initial text scope")
    if any(p.review_status != "pending" or p.generation.mock for p in pairs):
        raise ValueError("scale research must not manufacture human acceptance")


def load(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["job_id"] != "scale-001-train":
        raise ValueError("unexpected training wrapper job")
    actual = {
        str(p.relative_to(directory))
        for p in directory.rglob("*")
        if p.is_file() and p != directory / "manifest.json"
    }
    if set(manifest["files"]) != actual:
        raise ValueError("training wrapper must cover every input")
    for name, expected in manifest["files"].items():
        if (
            Path(name).is_absolute()
            or ".." in Path(name).parts
            or sha256_file(directory / name) != expected
        ):
            raise ValueError("training wrapper checksum mismatch")
    pairs = read_jsonl(directory / "pairs.pending-research.jsonl", Pair)
    validate_pairs(pairs)
    inner, config, splits = load_sft(directory / "bundle")
    if inner["pair_source_sha256"] != sha256_file(directory / "pairs.pending-research.jsonl"):
        raise ValueError("training pairs differ from the tokenized source")
    if (
        config.max_steps != 1000
        or config.gradient_accumulation_steps != 4
        or config.learning_rate != 0.00005
        or config.data_policy != "pending_research"
        or config.model_id != "Qwen/Qwen3-4B"
        or config.revision != "1cfa9a7208912126459214e8b04321603b3df60c"
        or config.method != "bf16_lora"
        or config.max_sequence_tokens != 2048
        or (config.lora_rank, config.lora_alpha, config.lora_dropout) != (16, 32, 0.05)
        or (config.eval_steps, config.save_steps, config.warmup_steps) != (100, 100, 30)
    ):
        raise ValueError("scale training recipe differs")
    if {k: len(v) for k, v in splits.items()} != {"train": 2000, "dev": 200}:
        raise ValueError("tokenized training counts differ")
    job = json.loads((directory / "job.json").read_text())
    if job["bundle_manifest_sha256"] != sha256_file(directory / "bundle/manifest.json") or job[
        "evaluation_sha256"
    ] != sha256_file(directory / "evaluation.requests.jsonl"):
        raise ValueError("job inputs differ")
    requests = [
        json.loads(line)
        for line in (directory / "evaluation.requests.jsonl").read_text().splitlines()
    ]
    if (
        len(requests) != 172
        or len({r["case_id"] for r in requests}) != 172
        or Counter(r["input_origin"] for r in requests)
        != {"synthetic_development": 72, "corpus_natural_holdout": 100}
    ):
        raise ValueError("comparison requires 72 regression and 100 natural holdout inputs")
    return manifest, job, pairs


def prepare(pairs_path, inventory, metadata, output):
    from tokenizers import Tokenizer

    pairs = read_jsonl(pairs_path, Pair)
    validate_pairs(pairs)
    assigned = read_jsonl(inventory, AssignedSource)
    holdout = sorted(
        [r for r in assigned if r.split == "test" and not exclusion_reason(r.source.text)],
        key=lambda r: hashlib.sha256(r.source.source_doc_id.encode()).hexdigest(),
    )[:100]
    if len(holdout) != 100 or {r.source.source_doc_id for r in holdout} & {
        p.source.source_doc_id for p in pairs
    }:
        raise ValueError("insufficient disjoint natural holdout")
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    write_jsonl(output / "pairs.pending-research.jsonl", pairs)
    write_jsonl(output / "natural-holdout.sources.jsonl", holdout)
    prepare_sft(
        pairs_path,
        metadata,
        Path("configs/sft-scale-001.v1.json"),
        output / "bundle",
        allow_pending_research=True,
    )
    spec = BaselineSpec.model_validate_json(
        Path("configs/baselines/qwen3-4b.json").read_text()
    ).model_copy(
        update={
            "candidate": "qwen3-4b-scale-evaluation-v1",
            "instruction": INSTRUCTION,
            "max_output_tokens": 1024,
        }
    )
    dev = [json.loads(line) for line in Path("eval/dev_seed.v1.jsonl").read_text().splitlines()]
    if len(dev) != 72:
        raise ValueError("expected frozen 72 regression inputs")
    inputs = [(r["case_id"], r["text"], "synthetic_development") for r in dev] + [
        ("natural-" + r.source.source_doc_id, r.source.text, "corpus_natural_holdout")
        for r in holdout
    ]
    tc = json.loads((metadata / "tokenizer_config.json").read_text())
    tokenizer = Tokenizer.from_file(str(metadata / "tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    requests = []
    for key, text, origin in inputs:
        body = request_body(SimpleNamespace(text=text), spec)
        rendered = render_chat(tc["chat_template"], body, tc)
        count = len(tokenizer.encode(rendered, add_special_tokens=False).ids)
        if count + 1024 > 8192:
            raise ValueError("evaluation context overflow")
        row = {
            "case_id": key,
            "candidate": spec.candidate,
            "revision": spec.revision,
            "protocol": "chat",
            "input_origin": origin,
            "body": body,
            "prompt_tokens": count,
            "rendered_prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
        }
        row["request_sha256"] = digest(row)
        requests.append(row)
    write_jsonl(output / "evaluation.requests.jsonl", requests)
    write_json(
        output / "job.json",
        {
            "job_id": "scale-001-train",
            "bundle_manifest_sha256": sha256_file(output / "bundle/manifest.json"),
            "evaluation_sha256": sha256_file(output / "evaluation.requests.jsonl"),
            "comparison_inputs": 172,
            "comparison_max_output_tokens": 1024,
            "human_semantic_review_completed": False,
            "natural_holdout_is_publisher_attributed_not_human_validated": True,
        },
    )
    write_json(
        output / "manifest.json",
        {
            "job_id": "scale-001-train",
            "inventory_sha256": sha256_file(inventory),
            "files": {
                str(p.relative_to(output)): sha256_file(p)
                for p in sorted(output.rglob("*"))
                if p.is_file()
            },
        },
    )
    load(output)
    return {
        "manifest_sha256": sha256_file(output / "manifest.json"),
        "train": 2000,
        "dev": 200,
        "comparison_inputs": 172,
        "gpu_calls": 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--pairs", type=Path)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.verify:
        load(args.verify)
        print("Scale training wrapper verified.")
    elif all([args.pairs, args.inventory, args.metadata, args.output_dir]):
        print(json.dumps(prepare(args.pairs, args.inventory, args.metadata, args.output_dir)))
    else:
        parser.error("verify or every preparation input is required")
