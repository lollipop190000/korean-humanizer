"""Freeze AI quality-screening requests from verified draft outputs. No model calls."""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import generate_scale_batch as batch

from korean_humanizer.automated_quality import (
    INSTRUCTION_V2,
    INSTRUCTION_V3,
    INSTRUCTION_V4,
    judge_input,
)
from korean_humanizer.batch_generation import batch_plan
from korean_humanizer.benchmark import digest, request_body
from korean_humanizer.data import normalized
from korean_humanizer.io import read_jsonl, sha256_file, write_json, write_jsonl
from korean_humanizer.models import AssignedSource
from korean_humanizer.protection import check_protection
from korean_humanizer.revision_protocols import indexed_judge_input
from korean_humanizer.token_audit import render_chat


def prepare(drafts_bundle, run, metadata, output, *, batch_size=4, evidence_contract=None):
    from tokenizers import Tokenizer

    draft_manifest, spec, requests, _, _ = batch.load(drafts_bundle)
    if evidence_contract is None:
        evidence_contract = draft_manifest.get("evidence_contract") or (
            "indexed-v3"
            if draft_manifest.get("draft_contract") == "protected-v2"
            else "verbatim-v2"
        )
    if evidence_contract not in {"verbatim-v2", "indexed-v3", "indexed-v4"}:
        raise ValueError("unknown evidence contract")
    draft_prompt_version = "ai-polish:" + digest({"instruction": spec.instruction})
    result = json.loads((run / "run.json").read_text())
    if result["bundle_manifest_sha256"] != sha256_file(drafts_bundle / "manifest.json") or result[
        "outputs_sha256"
    ] != sha256_file(run / "outputs.jsonl"):
        raise ValueError("draft outputs are not bound to their frozen input bundle")
    outputs = [json.loads(line) for line in (run / "outputs.jsonl").read_text().splitlines()]
    indexed = {r["case_id"]: r for r in outputs}
    if len(indexed) != len(outputs) or set(indexed) != {r["case_id"] for r in requests}:
        raise ValueError("missing or duplicate draft outputs")
    request_index = {r["case_id"]: r for r in requests}
    sources, drafts, excluded = [], [], []
    for row in read_jsonl(drafts_bundle / "selected.sources.jsonl", AssignedSource):
        key = row.source.source_doc_id
        draft = indexed[key]
        if (
            draft["request_sha256"] != request_index[key]["request_sha256"]
            or draft["candidate"] != spec.candidate
        ):
            raise ValueError("draft request identity mismatch")
        reason = ""
        if draft["finish_reason"] != "stop" or draft["error_code"]:
            reason = "incomplete_draft"
        elif normalized(row.source.text) == normalized(draft["text"]):
            reason = "unchanged_not_a_reverse_example"
        elif not check_protection(row.source.text, draft["text"])["mechanical_pass"]:
            reason = "mechanical_protection_failed"
        if reason:
            excluded.append({"case_id": key, "reason": reason})
        else:
            sources.append(row)
            drafts.append(draft)
    if not sources:
        raise ValueError("no mechanically eligible completed drafts")
    spec = spec.model_copy(
        update={
            "candidate": "qwen3-8b-scale-judge-v4"
            if evidence_contract == "indexed-v4"
            else "qwen3-8b-scale-judge-v3"
            if evidence_contract == "indexed-v3"
            else "qwen3-8b-scale-judge-v2",
            "instruction": INSTRUCTION_V4
            if evidence_contract == "indexed-v4"
            else INSTRUCTION_V3
            if evidence_contract == "indexed-v3"
            else INSTRUCTION_V2,
            "max_output_tokens": 384,
            "temperature": 0.1,
        }
    )
    meta = json.loads((metadata / "metadata.json").read_text())
    if (meta["model_id"], meta["revision"]) != (spec.model_id, spec.revision):
        raise ValueError("judge tokenizer revision mismatch")
    for name in ["tokenizer.json", "tokenizer_config.json"]:
        if sha256_file(metadata / name) != meta["files"][name]["sha256"]:
            raise ValueError("judge tokenizer integrity failure")
    tokenizer = Tokenizer.from_file(str(metadata / "tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    tc = json.loads((metadata / "tokenizer_config.json").read_text())
    requests, audits = [], []
    for row, draft in zip(sources, drafts, strict=True):
        key = row.source.source_doc_id
        input_fn = (
            indexed_judge_input
            if evidence_contract in {"indexed-v3", "indexed-v4"}
            else judge_input
        )
        body = request_body(SimpleNamespace(text=input_fn(row.source.text, draft["text"])), spec)
        rendered = render_chat(tc["chat_template"], body, tc)
        count = len(tokenizer.encode(rendered, add_special_tokens=False).ids)
        if count + spec.max_output_tokens > spec.context_tokens:
            raise ValueError("judge context overflow; do not truncate")
        audits.append(
            {
                "case_id": key,
                "prompt_tokens": count,
                "rendered_prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
            }
        )
        requests.append(
            {
                "case_id": key,
                "candidate": spec.candidate,
                "revision": spec.revision,
                "protocol": "chat",
                "body": body,
            }
        )
    batches = batch_plan(requests, audits, batch_size=batch_size, seed=spec.seed)
    seeds = {key: b["seed"] for b in batches for key in b["case_ids"]}
    for request in requests:
        request["body"]["seed"] = seeds[request["case_id"]]
        request["request_sha256"] = digest(request)
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    write_jsonl(output / "selected.sources.jsonl", sources)
    write_jsonl(output / "drafts.jsonl", drafts)
    write_jsonl(output / "requests.jsonl", requests)
    write_json(output / "spec.json", spec.model_dump())
    write_json(
        output / "tokens.json",
        {
            "model_id": spec.model_id,
            "revision": spec.revision,
            "tokenizer_sha256": meta["files"]["tokenizer.json"]["sha256"],
            "cases": audits,
        },
    )
    write_json(
        output / "manifest.json",
        {
            "job_id": "scale-001-judge",
            "evidence_contract": evidence_contract,
            "draft_prompt_version": draft_prompt_version,
            "draft_manifest_sha256": sha256_file(drafts_bundle / "manifest.json"),
            "draft_outputs_sha256": sha256_file(run / "outputs.jsonl"),
            "requests": len(requests),
            "batch_size": batch_size,
            "batches": batches,
            "mechanical_exclusions": excluded,
            "files": {p.name: sha256_file(p) for p in sorted(output.iterdir())},
            "human_semantic_review_completed": False,
        },
    )
    return {
        "requests": len(requests),
        "excluded_before_judge": len(excluded),
        "manifest_sha256": sha256_file(output / "manifest.json"),
        "model_calls": 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drafts-bundle", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--evidence-contract", choices=["verbatim-v2", "indexed-v3", "indexed-v4"], default=None
    )
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.drafts_bundle,
                args.run,
                args.metadata,
                args.output_dir,
                evidence_contract=args.evidence_contract,
            )
        )
    )
