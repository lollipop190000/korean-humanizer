"""Run an integrity-checked draft batch on an explicitly approved GPU. Default is offline."""

import argparse
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

from korean_humanizer.automated_quality import INSTRUCTION as JUDGE_INSTRUCTION
from korean_humanizer.automated_quality import (
    INSTRUCTION_V2,
    INSTRUCTION_V3,
    INSTRUCTION_V4,
    judge_input,
)
from korean_humanizer.batch_generation import batch_plan, generated_tokens, verify_prompt
from korean_humanizer.benchmark import BaselineSpec, digest, request_body
from korean_humanizer.draft_pilot import INSTRUCTION as DRAFT_INSTRUCTION
from korean_humanizer.io import read_jsonl, sha256_file, write_json
from korean_humanizer.models import AssignedSource
from korean_humanizer.revision_protocols import DRAFT_V2, draft_input, indexed_judge_input
from korean_humanizer.source_genre import INSTRUCTION as GENRE_INSTRUCTION
from korean_humanizer.source_scope import scale_year_allowed


def load(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["job_id"] not in {"scale-001-drafts", "scale-001-judge", "scale-001-source-genre"}:
        raise ValueError("unexpected job ID")
    is_judge = manifest["job_id"] == "scale-001-judge"
    is_genre = manifest["job_id"] == "scale-001-source-genre"
    draft_contract = manifest.get("draft_contract", "verbatim-v1")
    evidence_contract = manifest.get("evidence_contract", "verbatim-v2")
    if draft_contract not in {"verbatim-v1", "protected-v2"} or evidence_contract not in {
        "verbatim-v2",
        "indexed-v3",
        "indexed-v4",
    }:
        raise ValueError("unknown versioned request protocol")
    required = {
        "spec.json",
        "requests.jsonl",
        "tokens.json",
        "selected.sources.jsonl",
    }
    if is_judge:
        required.add("drafts.jsonl")
    if set(manifest["files"]) != required:
        raise ValueError("bundle must cover every required input file")
    for name, expected in manifest["files"].items():
        if Path(name).name != name or sha256_file(directory / name) != expected:
            raise ValueError("bundle checksum mismatch")
    spec = BaselineSpec.model_validate_json((directory / "spec.json").read_text())
    if (spec.model_id, spec.revision) != (
        "Qwen/Qwen3-8B",
        "b968826d9c46dd6066d109eabc6255188de91218",
    ) or spec.max_output_tokens > 1024:
        raise ValueError("generator revision or output cap outside scale scope")
    if is_judge:
        instructions = (
            {INSTRUCTION_V4}
            if evidence_contract == "indexed-v4"
            else {INSTRUCTION_V3}
            if evidence_contract == "indexed-v3"
            else {JUDGE_INSTRUCTION, INSTRUCTION_V2}
        )
        if spec.max_output_tokens > 384 or spec.instruction not in instructions:
            raise ValueError("judge instruction or output cap differs from scale scope")
        if evidence_contract in {"indexed-v3", "indexed-v4"} and spec.candidate != (
            "qwen3-8b-scale-judge-v4"
            if evidence_contract == "indexed-v4"
            else "qwen3-8b-scale-judge-v3"
        ):
            raise ValueError("indexed evidence candidate differs")
    if is_genre and (
        spec.instruction != GENRE_INSTRUCTION or spec.max_output_tokens != 8 or spec.top_k != 1
    ):
        raise ValueError("source genre triage differs from frozen protocol")
    if not is_judge and not is_genre:
        expected_instruction = DRAFT_V2 if draft_contract == "protected-v2" else DRAFT_INSTRUCTION
        if spec.instruction != expected_instruction:
            raise ValueError("draft instruction differs from scale scope")
        if draft_contract == "protected-v2" and spec.candidate != "qwen3-8b-scale-draft-v2":
            raise ValueError("protected draft candidate differs")
    requests = [
        json.loads(line) for line in (directory / "requests.jsonl").read_text().splitlines()
    ]
    audit = json.loads((directory / "tokens.json").read_text())
    for row in requests:
        expected = row["request_sha256"]
        actual = digest({key: value for key, value in row.items() if key != "request_sha256"})
        if expected != actual or row["body"]["max_tokens"] != spec.max_output_tokens:
            raise ValueError("request identity or output cap mismatch")
    batches = batch_plan(
        requests, audit["cases"], batch_size=manifest["batch_size"], seed=spec.seed
    )
    if batches != manifest["batches"] or len(requests) != manifest["requests"]:
        raise ValueError("batch plan differs from manifest")
    seeds = {key: batch["seed"] for batch in batches for key in batch["case_ids"]}
    if any(r["body"]["seed"] != seeds[r["case_id"]] for r in requests):
        raise ValueError("request seed differs from its deterministic batch seed")
    sources = read_jsonl(directory / "selected.sources.jsonl", AssignedSource)
    indexed = {row.source.source_doc_id: row for row in sources}
    if len(indexed) != len(sources) or set(indexed) != {r["case_id"] for r in requests}:
        raise ValueError("requests must correspond to the frozen source selection")
    drafts = {}
    if is_judge:
        values = [
            json.loads(line) for line in (directory / "drafts.jsonl").read_text().splitlines()
        ]
        drafts = {v["case_id"]: v for v in values}
        if len(drafts) != len(values) or set(drafts) != set(indexed):
            raise ValueError("judge drafts must match selected sources one-to-one")
    for row in requests:
        source = indexed[row["case_id"]]
        if source.split == "test" or source.source.origin != "human":
            raise ValueError("test/fixture sources cannot enter scale generation")
        if not scale_year_allowed(source.source):
            raise ValueError("source year outside scale scope")
        text_input = source.source
        if is_judge:
            draft = drafts[row["case_id"]]
            if draft["finish_reason"] != "stop" or draft["error_code"]:
                raise ValueError("incomplete draft cannot enter judge")
            input_fn = (
                indexed_judge_input
                if evidence_contract in {"indexed-v3", "indexed-v4"}
                else judge_input
            )
            text_input = SimpleNamespace(text=input_fn(source.source.text, draft["text"]))
        elif not is_genre and draft_contract == "protected-v2":
            text_input = SimpleNamespace(text=draft_input(source.source.text))
        expected_body = request_body(text_input, spec)
        expected_body["seed"] = seeds[row["case_id"]]
        if row["body"] != expected_body or (row["candidate"], row["revision"], row["protocol"]) != (
            spec.candidate,
            spec.revision,
            "chat",
        ):
            raise ValueError("request does not match the selected source and spec")
    return manifest, spec, requests, audit, batches


def run(bundle, output, approval, max_seconds):
    manifest, spec, requests, audit, batches = load(bundle)
    permission = json.loads(approval.read_text())
    if (
        permission.get("job_id") != "scale-001"
        or permission.get("bundle_manifest_sha256") != sha256_file(bundle / "manifest.json")
        or permission.get("source_use_and_external_transfer_approved") is not True
        or not permission.get("user_approval_ref")
        or permission.get("max_total_usd") != 30
        or not 1 <= max_seconds <= 40 * 3600
    ):
        raise ValueError("matching source-use and spending approval required")
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    # Installed only in the approved ephemeral GPU environment, never on the CPU workspace.
    import torch
    import transformers
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    if not torch.cuda.is_available():
        raise ValueError("CUDA unavailable")
    started = time.monotonic()
    snapshot = Path(
        snapshot_download(
            spec.model_id,
            revision=spec.revision,
            allow_patterns=["*.json", "*.safetensors", "*.model"],
        )
    )
    if (
        snapshot.name != spec.revision
        or sha256_file(snapshot / "tokenizer.json") != audit["tokenizer_sha256"]
    ):
        raise ValueError("downloaded revision/tokenizer differs from audited bundle")
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, trust_remote_code=False, padding_side="left"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        trust_remote_code=False,
        use_safetensors=True,
        attn_implementation="sdpa",
    ).eval()
    eos = model.generation_config.eos_token_id
    eos = {eos} if isinstance(eos, int) else set(eos or [])
    if not eos:
        raise ValueError("model needs explicit stop tokens")
    indexed = {r["case_id"]: r for r in requests}
    audits = {r["case_id"]: r for r in audit["cases"]}
    tokenizer_data = None
    if (
        manifest.get("evidence_contract") == "indexed-v4"
        and manifest["job_id"] == "scale-001-judge"
    ):
        from importlib.metadata import version

        from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data

        if version("lm-format-enforcer") != "0.11.3":
            raise ValueError("unreviewed constrained decoder version")
        tokenizer_data = build_token_enforcer_tokenizer_data(tokenizer)
    completed = 0
    path = output / "outputs.jsonl"
    with path.open("x") as handle:
        path.chmod(0o600)
        for number, batch in enumerate(batches):
            remaining = max_seconds - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("batch phase reached its deadline; no retry")
            rows = [indexed[key] for key in batch["case_ids"]]
            prompts = [
                tokenizer.apply_chat_template(
                    r["body"]["messages"],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                for r in rows
            ]
            for row, prompt in zip(rows, prompts, strict=True):
                count = len(tokenizer.encode(prompt, add_special_tokens=False))
                verify_prompt(
                    prompt,
                    audits[row["case_id"]],
                    count,
                    spec.max_output_tokens,
                    spec.context_tokens,
                )
            inputs = tokenizer(
                prompts,
                add_special_tokens=False,
                padding=True,
                truncation=False,
                return_tensors="pt",
            ).to("cuda")
            width = inputs.input_ids.shape[-1]
            set_seed(batch["seed"])
            batch_started = time.monotonic()
            constraints = {}
            if tokenizer_data is not None:
                from korean_humanizer.indexed_constraints import (
                    batch_prefix_filter,
                    judgment_schema,
                )

                inputs_data = [json.loads(r["body"]["messages"][1]["content"]) for r in rows]
                schemas = [judgment_schema(d["source"], d["draft"]) for d in inputs_data]
                constraints["prefix_allowed_tokens_fn"] = batch_prefix_filter(
                    tokenizer_data, schemas
                )
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    **constraints,
                    do_sample=True,
                    max_new_tokens=spec.max_output_tokens,
                    temperature=spec.temperature,
                    top_p=spec.top_p,
                    top_k=spec.top_k,
                    min_p=spec.min_p,
                    repetition_penalty=spec.repetition_penalty,
                    pad_token_id=tokenizer.pad_token_id,
                    max_time=min(300, remaining),
                )
            torch.cuda.synchronize()
            elapsed = time.monotonic() - batch_started
            for row, sequence in zip(rows, generated.tolist(), strict=True):
                tokens, reason = generated_tokens(sequence, width, eos)
                text = tokenizer.decode(tokens, skip_special_tokens=True)
                result = {
                    "case_id": row["case_id"],
                    "candidate": spec.candidate,
                    "request_sha256": row["request_sha256"],
                    "text": text,
                    "finish_reason": reason if text.strip() else "error",
                    "error_code": "" if text.strip() else "empty_output",
                    "output_tokens": len(tokens),
                    "batch_index": number,
                    "batch_seed": batch["seed"],
                    "batch_elapsed_seconds": elapsed,
                }
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                completed += 1
            del inputs, generated
            print(
                json.dumps({"completed": completed, "total": len(requests), "batch": number}),
                flush=True,
            )
    write_json(
        output / "run.json",
        {
            "job_id": manifest["job_id"],
            "bundle_manifest_sha256": sha256_file(bundle / "manifest.json"),
            "outputs_sha256": sha256_file(path),
            "requests": len(requests),
            "completed": completed,
            "elapsed_seconds": time.monotonic() - started,
            "revision": spec.revision,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "human_semantic_reviews": 0,
            "batch_plan_sha256": hashlib.sha256(
                json.dumps(batches, sort_keys=True).encode()
            ).hexdigest(),
        },
    )
    # Recovery executes draft and judge phases in one process; release model cycles first.
    del model, tokenizer
    __import__("gc").collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--approval", type=Path)
    parser.add_argument("--max-seconds", type=int, default=3600)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.execute:
        if args.output_dir is None or args.approval is None:
            parser.error("execution requires output directory and explicit matching approval")
        run(args.bundle, args.output_dir, args.approval, args.max_seconds)
    else:
        manifest, _, requests, _, batches = load(args.bundle)
        print(
            json.dumps(
                {
                    "mode": "dry_run",
                    "network_calls": 0,
                    "requests": len(requests),
                    "batches": len(batches),
                }
            )
        )
