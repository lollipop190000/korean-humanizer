"""Freeze the first unreviewed research run. Offline; no survey ingestion or GPU calls."""

import argparse
import json
from collections import Counter
from pathlib import Path

from korean_humanizer.benchmark import load_bundle
from korean_humanizer.io import read_jsonl, sha256_file, write_json, write_jsonl
from korean_humanizer.models import Pair
from korean_humanizer.protection import check_protection
from korean_humanizer.sft import prepare_sft


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    evaluation_bundle = Path("artifacts/baseline-smoke-001-inputs/qwen3-4b")
    _, _, requests, spec = load_bundle(evaluation_bundle)
    original = Path("artifacts/draft-pilot-001-pairs/pairs.pending.jsonl")
    expected = "71efbe08115e342a9a6cb46b4fcac2b00546b2aaa1c2c0432f489c5ac523731f"
    if sha256_file(original) != expected:
        raise ValueError("first research run requires the original frozen 24 pending pairs")
    pairs = read_jsonl(original, Pair)
    selected = [
        p
        for p in pairs
        if check_protection(p.synthetic_ai_text, p.source.text, p.source.protected_spans)[
            "mechanical_pass"
        ]
    ]
    if Counter(p.split for p in selected) != {"train": 9, "dev": 1}:
        raise ValueError("unexpected research selection; do not redraw splits")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    selection = output / "pairs.pending-research.jsonl"
    write_jsonl(selection, selected)
    # Reuse the baseline's frozen 12 AI-written development inputs, never training targets.
    evaluation = evaluation_bundle / "requests.jsonl"
    if (
        len(requests) != 12
        or len({r["case_id"] for r in requests}) != 12
        or spec.model_id != "Qwen/Qwen3-4B"
        or spec.revision != "1cfa9a7208912126459214e8b04321603b3df60c"
        or spec.enable_thinking
    ):
        raise ValueError("expected the frozen 12 development inputs")
    (output / "evaluation.requests.jsonl").write_bytes(evaluation.read_bytes())
    prepare_sft(
        selection,
        Path("artifacts/tokenizers/qwen3-4b"),
        Path("configs/sft-first-research.v1.json"),
        output / "bundle",
        allow_pending_research=True,
    )
    report = {
        "job_id": "sft-first-research-001",
        "status": "prepared_gpu_budget_pending",
        "training_executed": False,
        "survey_results_used": False,
        "human_semantic_review_completed": False,
        "user_direction": (
            "Train first, apply survey findings later, survey outputs after training."
        ),
        "original_pairs_sha256": expected,
        "selected_pairs_sha256": sha256_file(selection),
        "bundle_manifest_sha256": sha256_file(output / "bundle/manifest.json"),
        "evaluation_sha256": sha256_file(output / "evaluation.requests.jsonl"),
        "counts": {"train": 9, "dev": 1, "excluded_mechanical": 14},
        "review_status": "pending throughout; no acceptance inferred or fabricated",
        "evaluation": {
            "inputs": 12,
            "models": ["base", "adapter-step-16"],
            "outputs": 24,
            "max_output_tokens_each": 2048,
            "source": "AI-authored development inputs, not independent final holdout",
            "sampling": "same frozen requests and non-thinking instruction for both models",
        },
        "budget_proposal_usd": 3,
        "spending_authorized": False,
        "gpu_max_seconds": 3300,
        "gpu_max_hourly_usd": 1.5,
        "quality_verdict": "not_assessed",
    }
    write_json(output / "job.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
