"""Freeze bounded scale draft requests on CPU. Does not approve, transfer or generate text."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from korean_humanizer.batch_generation import batch_plan
from korean_humanizer.benchmark import BaselineSpec, digest, request_body
from korean_humanizer.draft_pilot import INSTRUCTION
from korean_humanizer.io import read_jsonl, sha256_file, write_json, write_jsonl
from korean_humanizer.models import AssignedSource
from korean_humanizer.revision_protocols import DRAFT_V2, draft_input
from korean_humanizer.source_genre import INSTRUCTION as GENRE_INSTRUCTION
from korean_humanizer.source_scope import exclusion_reason, scale_year_allowed
from korean_humanizer.token_audit import render_chat


def prepare(
    sources,
    metadata,
    output,
    *,
    count=32,
    exclude_ids=None,
    selection_seed=None,
    source_genre_only=False,
    draft_contract="verbatim-v1",
    evidence_contract=None,
):
    from tokenizers import Tokenizer

    if draft_contract not in {"verbatim-v1", "protected-v2"}:
        raise ValueError("unknown draft contract")
    if source_genre_only and draft_contract != "verbatim-v1":
        raise ValueError("genre-only requests cannot use a rewriting protocol")
    if evidence_contract is not None and (
        draft_contract != "protected-v2" or evidence_contract != "indexed-v4"
    ):
        raise ValueError("unknown planned evidence contract")
    if not 1 <= count <= 6000:
        raise ValueError("request count outside the scale scope")
    items = [r for r in read_jsonl(sources, AssignedSource) if r.split in {"train", "dev"}]
    items = [r for r in items if r.source.source_doc_id not in (exclude_ids or set())]
    items = [r for r in items if not exclusion_reason(r.source.text)]
    counts = Counter()
    selected = []
    remaining = {r.source.source_doc_id: r for r in items}
    if len(remaining) != len(items):
        raise ValueError("duplicate source IDs")
    while len(selected) < count:
        if not remaining:
            raise ValueError("not enough non-test sources")
        key = min(
            remaining,
            key=lambda key: (
                counts[remaining[key].source.author_group],
                hashlib.sha256(f"{selection_seed}:{key}".encode()).hexdigest()
                if selection_seed is not None
                else key,
            ),
        )
        row = remaining.pop(key)
        if row.source.origin != "human" or not scale_year_allowed(row.source):
            raise ValueError("source outside approved year/origin scope")
        if not 200 <= len(row.source.text) <= 600:
            raise ValueError("source outside the approved excerpt length")
        selected.append(row)
        counts[row.source.author_group] += 1
    spec = BaselineSpec.model_validate_json(Path("configs/baselines/qwen3-8b.json").read_text())
    spec = spec.model_copy(
        update={
            "candidate": "qwen3-8b-scale-draft-v1",
            "instruction": INSTRUCTION,
            "max_output_tokens": 1024,
        }
    )
    if source_genre_only:
        spec = spec.model_copy(
            update={
                "candidate": "qwen3-8b-source-genre-v1",
                "instruction": GENRE_INSTRUCTION,
                "max_output_tokens": 8,
                "top_k": 1,
                "temperature": 0.2,
            }
        )
    elif draft_contract == "protected-v2":
        spec = spec.model_copy(
            update={"candidate": "qwen3-8b-scale-draft-v2", "instruction": DRAFT_V2}
        )
    meta = json.loads((metadata / "metadata.json").read_text())
    if (meta["model_id"], meta["revision"]) != (spec.model_id, spec.revision):
        raise ValueError("tokenizer revision mismatch")
    for name in ["tokenizer.json", "tokenizer_config.json"]:
        if sha256_file(metadata / name) != meta["files"][name]["sha256"]:
            raise ValueError("tokenizer integrity failure")
    tokenizer = Tokenizer.from_file(str(metadata / "tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    tc = json.loads((metadata / "tokenizer_config.json").read_text())
    requests, audits = [], []
    for row in selected:
        source = row.source
        request = {
            "case_id": source.source_doc_id,
            "candidate": spec.candidate,
            "revision": spec.revision,
            "protocol": "chat",
            "body": request_body(
                SimpleNamespace(text=draft_input(source.text))
                if draft_contract == "protected-v2"
                else source,
                spec,
            ),
        }
        rendered = render_chat(tc["chat_template"], request["body"], tc)
        tokens = len(tokenizer.encode(rendered, add_special_tokens=False).ids)
        if tokens + spec.max_output_tokens > spec.context_tokens:
            raise ValueError("context overflow; do not truncate")
        audits.append(
            {
                "case_id": source.source_doc_id,
                "prompt_tokens": tokens,
                "rendered_prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
            }
        )
        requests.append(request)
    batches = batch_plan(requests, audits, batch_size=4, seed=spec.seed)
    seeds = {key: b["seed"] for b in batches for key in b["case_ids"]}
    for request in requests:
        request["body"]["seed"] = seeds[request["case_id"]]
        request["request_sha256"] = digest(request)
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    write_jsonl(output / "selected.sources.jsonl", selected)
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
    manifest = {
        "job_id": "scale-001-source-genre" if source_genre_only else "scale-001-drafts",
        "status": "prepared_source_use_and_spending_approval_pending",
        "source_inventory_sha256": sha256_file(sources),
        "requests": len(requests),
        "batch_size": 4,
        "batches": batches,
        "calibration_included_in_workflow_request_limit": True,
        "files": {p.name: sha256_file(p) for p in sorted(output.iterdir())},
        "source_rights_approval_executed": False,
        "generation_executed": False,
    }
    if selection_seed is not None:
        manifest["selection_seed"] = selection_seed
    if draft_contract == "protected-v2":
        manifest["draft_contract"] = draft_contract
    if evidence_contract is not None:
        manifest["evidence_contract"] = evidence_contract
    write_json(output / "manifest.json", manifest)
    return {
        "requests": len(requests),
        "authors": len(counts),
        "batches": len(batches),
        "manifest_sha256": sha256_file(output / "manifest.json"),
        "gpu_calls": 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--exclude-sources", type=Path)
    parser.add_argument("--selection-seed", type=int)
    parser.add_argument(
        "--draft-contract", choices=["verbatim-v1", "protected-v2"], default="verbatim-v1"
    )
    args = parser.parse_args()
    exclusions = set()
    if args.exclude_sources:
        exclusions = {
            r.source.source_doc_id for r in read_jsonl(args.exclude_sources, AssignedSource)
        }
    print(
        json.dumps(
            prepare(
                args.sources,
                args.metadata,
                args.output_dir,
                count=args.count,
                exclude_ids=exclusions,
                selection_seed=args.selection_seed,
                draft_contract=args.draft_contract,
            )
        )
    )
