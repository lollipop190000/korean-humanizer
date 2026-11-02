"""Verify judge output identities and screen research pairs offline; no human acceptance."""

import argparse
import json
from collections import Counter
from pathlib import Path

import generate_scale_batch as batch

from korean_humanizer.automated_quality import screen
from korean_humanizer.io import read_jsonl, sha256_file, write_json, write_jsonl
from korean_humanizer.models import AssignedSource


def process(bundle, run, output):
    manifest, spec, requests, _, _ = batch.load(bundle)
    if manifest["job_id"] != "scale-001-judge":
        raise ValueError("a judge bundle is required")
    result = json.loads((run / "run.json").read_text())
    if result["bundle_manifest_sha256"] != sha256_file(bundle / "manifest.json") or result[
        "outputs_sha256"
    ] != sha256_file(run / "outputs.jsonl"):
        raise ValueError("judge result hashes differ from frozen inputs/outputs")
    outputs = [json.loads(line) for line in (run / "outputs.jsonl").read_text().splitlines()]
    indexed = {r["case_id"]: r for r in outputs}
    if len(indexed) != len(outputs) or set(indexed) != {r["case_id"] for r in requests}:
        raise ValueError("judge outputs must match requests one-to-one")
    sources = {
        r.source.source_doc_id: r
        for r in read_jsonl(bundle / "selected.sources.jsonl", AssignedSource)
    }
    drafts = {
        r["case_id"]: r
        for r in [json.loads(line) for line in (bundle / "drafts.jsonl").read_text().splitlines()]
    }
    screened = []
    for request in requests:
        key = request["case_id"]
        response = indexed[key]
        row = sources[key]
        if (
            response["request_sha256"] != request["request_sha256"]
            or response["candidate"] != spec.candidate
        ):
            raise ValueError("judge request identity differs")
        if response["finish_reason"] != "stop" or response["error_code"]:
            verdict = {
                "automatic_pass": False,
                "reason": "incomplete_judge",
                "rater_kind": "ai",
                "human_review_completed": False,
                "pair_review_status": "pending",
            }
        else:
            verdict = screen(
                row.source.text,
                drafts[key]["text"],
                response["text"],
                evidence_contract=manifest.get("evidence_contract", "verbatim-v2"),
            )
        screened.append(
            {
                "case_id": key,
                "split": row.split,
                "judge_request_sha256": request["request_sha256"],
                **verdict,
            }
        )
    passed = [r for r in screened if r["automatic_pass"]]
    report = {
        "job_id": "scale-001",
        "judge_manifest_sha256": sha256_file(bundle / "manifest.json"),
        "judge_outputs_sha256": sha256_file(run / "outputs.jsonl"),
        "drafts_submitted": len(requests) + len(manifest["mechanical_exclusions"]),
        "judged": len(screened),
        "automatic_pass": len(passed),
        "pre_judge_exclusions": dict(
            Counter(r["reason"] for r in manifest["mechanical_exclusions"])
        ),
        "judge_reasons": dict(Counter(r["reason"] for r in screened)),
        "passing_genres_by_split": {
            split: dict(Counter(r["judgment"]["genre"] for r in passed if r["split"] == split))
            for split in ["train", "dev"]
        },
        "human_semantic_reviews": 0,
        "accepted_pairs": 0,
        "training_executed": False,
        "same_model_self_screening_is_not_independent_quality_validation": True,
    }
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    write_jsonl(output / "screened.private.jsonl", screened)
    write_json(output / "summary.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(process(args.bundle, args.run, args.output_dir)))
