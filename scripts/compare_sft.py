"""Generate the frozen 12-input base/adapter comparison inside the authorized GPU job."""

import argparse
import hashlib
import json
import time
from pathlib import Path

from korean_humanizer.benchmark import digest
from korean_humanizer.io import sha256_file, write_json, write_jsonl
from korean_humanizer.sft import load_sft


def compare(ready, run, output, offset=0, limit=None):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    job = json.loads((ready / "job.json").read_text())
    bundle = ready / "bundle"
    _, config, _ = load_sft(bundle)
    if (
        sha256_file(bundle / "manifest.json") != job["bundle_manifest_sha256"]
        or sha256_file(ready / "evaluation.requests.jsonl") != job["evaluation_sha256"]
    ):
        raise ValueError("comparison input integrity failure")
    if job["job_id"] not in {"sft-first-research-001", "scale-001-train"}:
        raise ValueError("unexpected comparison job")
    scale = job["job_id"] == "scale-001-train"
    steps = 1000 if scale else 16
    count_expected = 172 if scale else 12
    output_cap = 1024 if scale else 2048
    result = json.loads((run / f"result-step-{steps}.json").read_text())
    if not result["completed"] or result["global_step"] != steps:
        raise ValueError("comparison requires the completed final adapter")
    requests = [
        json.loads(s) for s in (ready / "evaluation.requests.jsonl").read_text().splitlines()
    ]
    if len(requests) != count_expected or len({r["case_id"] for r in requests}) != count_expected:
        raise ValueError("comparison must retain every frozen input")
    for row in requests:
        unsigned = {k: v for k, v in row.items() if k != "request_sha256"}
        if (
            digest(unsigned) != row["request_sha256"]
            or row["revision"] != config.revision
            or row["body"]["model"] != config.model_id
            or row["body"]["max_tokens"] != output_cap
            or row["body"]["chat_template_kwargs"] != {"enable_thinking": False}
        ):
            raise ValueError("comparison request differs from its frozen contract")
    if offset < 0 or offset >= len(requests) or (limit is not None and limit < 1):
        raise ValueError("invalid comparison shard")
    if limit is None and offset != 0:
        raise ValueError("partial comparisons require an explicit limit")
    selected = requests[offset : offset + limit] if limit is not None else requests
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    tokenizer = AutoTokenizer.from_pretrained(
        bundle, local_files_only=True, trust_remote_code=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        revision=config.revision,
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
        use_safetensors=True,
        attn_implementation="sdpa",
        device_map={"": "cuda:0"},
    ).eval()
    rows = []
    for candidate in ["base", f"adapter-step-{steps}"]:
        if candidate != "base":
            adapter = run / f"adapter-step-{steps}"
            adapter_config = json.loads((adapter / "adapter_config.json").read_text())
            if adapter_config["base_model_name_or_path"] != config.model_id:
                raise ValueError("adapter base model differs")
            model = PeftModel.from_pretrained(model, adapter, is_trainable=False).eval()
        for request in selected:
            body = request["body"]
            started = time.monotonic()
            row = dict(
                case_id=request["case_id"],
                candidate=candidate,
                request_sha256=request["request_sha256"],
                text="",
                finish_reason="error",
                error_code="",
            )
            try:
                prompt = tokenizer.apply_chat_template(
                    body["messages"],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to("cuda")
                count = inputs.input_ids.shape[-1]
                if count + body["max_tokens"] > 8192:
                    raise ValueError("comparison context overflow")
                set_seed(body["seed"])
                torch.cuda.reset_peak_memory_stats()
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs,
                        do_sample=True,
                        max_new_tokens=body["max_tokens"],
                        temperature=body["temperature"],
                        top_p=body["top_p"],
                        top_k=body["top_k"],
                        min_p=body["min_p"],
                        repetition_penalty=body["repetition_penalty"],
                        pad_token_id=tokenizer.eos_token_id,
                    )[0, count:]
                torch.cuda.synchronize()
                eos = model.generation_config.eos_token_id
                eos = [eos] if isinstance(eos, int) else eos or []
                text = tokenizer.decode(generated, skip_special_tokens=True)
                if not text.strip():
                    raise ValueError("empty comparison output")
                row.update(
                    text=text,
                    finish_reason="stop" if int(generated[-1]) in eos else "length",
                    output_tokens=len(generated),
                    prompt_tokens=count,
                    prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                )
                del inputs, generated
            except Exception as exc:
                row["error_code"] = type(exc).__name__
            row["latency_seconds"] = time.monotonic() - started
            rows.append(row)
            print(
                json.dumps({k: row[k] for k in ["candidate", "case_id", "finish_reason"]}),
                flush=True,
            )
        write_jsonl(
            output / (candidate + ".outputs.jsonl"),
            [r for r in rows if r["candidate"] == candidate],
        )
    report = dict(
        job_id=job["job_id"],
        outputs=len(rows),
        errors=sum(r["finish_reason"] == "error" for r in rows),
        length_limited=sum(r["finish_reason"] == "length" for r in rows),
        evaluation_sha256=job["evaluation_sha256"],
        input_offset=offset,
        input_count=len(selected),
        evaluation_total=len(requests),
        quality_verdict="not_assessed",
        human_semantic_review_completed=False,
        files={p.name: sha256_file(p) for p in sorted(output.iterdir())},
    )
    write_json(output / "summary.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    compare(args.ready, args.run, args.output_dir, args.offset, args.limit)
