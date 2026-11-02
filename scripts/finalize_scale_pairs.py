"""Select exact quotas from verified AI screens. Pending research only; no model calls."""

import argparse
import json
from collections import Counter
from pathlib import Path

import screen_scale_outputs as screening

from korean_humanizer.benchmark import digest
from korean_humanizer.draft_pilot import INSTRUCTION as DRAFT_INSTRUCTION
from korean_humanizer.io import read_jsonl, sha256_file, write_json, write_jsonl
from korean_humanizer.models import AssignedSource, Generation, Pair
from korean_humanizer.protection import check_protection
from korean_humanizer.scale_selection import select_exact


def finalize(inventory, bundle_runs, approval, output):
    permission = json.loads(approval.read_text())
    if (
        permission.get("job_id") != "scale-001"
        or permission.get("max_total_usd") != 30
        or permission.get("source_use_and_external_transfer_approved") is not True
        or not permission.get("user_approval_ref")
    ):
        raise ValueError("recorded source-use and spending approval required")
    inventory_sha = sha256_file(inventory)
    assigned = read_jsonl(inventory, AssignedSource)
    originals = {r.source.source_doc_id: r for r in assigned}
    if len(originals) != len(assigned):
        raise ValueError("duplicate inventory IDs")
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    available = {}
    proofs = {}
    for number, (bundle, run) in enumerate(bundle_runs):
        bundle_manifest = json.loads((bundle / "manifest.json").read_text())
        screened_dir = output / f"screen-{number}"
        screening.process(bundle, run, screened_dir)
        selected = {
            r.source.source_doc_id: r
            for r in read_jsonl(bundle / "selected.sources.jsonl", AssignedSource)
        }
        drafts = {
            r["case_id"]: r
            for r in [
                json.loads(line) for line in (bundle / "drafts.jsonl").read_text().splitlines()
            ]
        }
        for record in [
            json.loads(line)
            for line in (screened_dir / "screened.private.jsonl").read_text().splitlines()
        ]:
            key = record["case_id"]
            if key in proofs:
                raise ValueError("source judged more than once")
            proofs[key] = record
            if originals.get(key) != selected[key]:
                raise ValueError("judge source differs from final frozen inventory")
            if record["automatic_pass"]:
                available[key] = {
                    "assigned": selected[key],
                    "draft": drafts[key],
                    "genre": record["judgment"]["genre"],
                    "proof_ref": str(screened_dir / "screened.private.jsonl"),
                    "prompt_version": bundle_manifest.get(
                        "draft_prompt_version",
                        "ai-polish-v1:" + digest({"instruction": DRAFT_INSTRUCTION}),
                    ),
                }
    quotas = {
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
    automatic_pass_count = len(available)
    excluded = permission.get("selection_excluded_case_ids", [])
    if (
        not isinstance(excluded, list)
        or any(not isinstance(key, str) for key in excluded)
        or len(excluded) != len(set(excluded))
        or not set(excluded).issubset(available)
    ):
        raise ValueError("selection exclusions must identify distinct existing automatic passes")
    for key in excluded:
        del available[key]
    protection_excluded = []
    for key, value in list(available.items()):
        source = value["assigned"].source
        if not check_protection(value["draft"]["text"], source.text, source.protected_spans)[
            "mechanical_pass"
        ]:
            protection_excluded.append(key)
            del available[key]
    chosen = {}
    report = {
        "job_id": "scale-001",
        "inventory_sha256": inventory_sha,
        "approval_ref": permission["user_approval_ref"],
        "judged_sources": len(proofs),
        "automatic_pass": automatic_pass_count,
        "human_semantic_reviews": 0,
        "accepted_pairs": 0,
        "training_executed": False,
        "quota_supply": {},
        "target_counts": {"train": 2000, "dev": 200},
    }
    if excluded:
        report["selection_excluded_case_ids"] = sorted(excluded)
    if protection_excluded:
        report["training_protection_excluded_case_ids"] = sorted(protection_excluded)
    if excluded or protection_excluded:
        report["eligible_after_exclusions"] = len(available)
    for split, target in quotas.items():
        pool = [
            {"case_id": key, "author_group": v["assigned"].source.author_group, "genre": v["genre"]}
            for key, v in available.items()
            if v["assigned"].split == split
        ]
        report["quota_supply"][split] = dict(Counter(r["genre"] for r in pool))
        chosen[split] = select_exact(pool, target, author_cap=20 if split == "train" else 200)
    if any(value is None for value in chosen.values()):
        report["status"] = "blocked_insufficient_eligible_genres_or_author_capacity"
        write_json(output / "summary.json", report)
        return report
    pairs = []
    for split, ids in chosen.items():
        controls_left = {g: n // 10 for g, n in quotas[split].items()}
        for key in sorted(ids):
            value = available[key]
            source = value["assigned"].source
            genre = value["genre"]
            is_control = controls_left[genre] > 0
            if is_control:
                controls_left[genre] -= 1
            rights = source.rights.model_copy(
                update={
                    "training": "approved",
                    "derivative_creation": "approved",
                    "third_party_transfer": "approved",
                    "evidence_ref": source.rights.evidence_ref
                    + "; "
                    + permission["user_approval_ref"]
                    + "; frozen-inventory-sha256:"
                    + inventory_sha,
                }
            )
            source = source.model_copy(update={"genre": genre, "rights": rights})
            pairs.append(
                Pair(
                    pair_id="scale-001-" + key,
                    source=source,
                    synthetic_ai_text=source.text if is_control else value["draft"]["text"],
                    split=split,
                    generation=Generation(
                        model="identity-control"
                        if is_control
                        else "Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218",
                        prompt_version="natural-control-v1"
                        if is_control
                        else value["prompt_version"],
                        seed=0 if is_control else value["draft"]["batch_seed"],
                        mock=False,
                    ),
                    review_status="pending",
                    review_ref=value["proof_ref"],
                    input_kind="natural_control" if is_control else "ai_draft",
                )
            )
    write_jsonl(output / "pairs.pending-research.jsonl", pairs)
    report.update(
        status="exact_quotas_selected_pending_research",
        pairs_sha256=sha256_file(output / "pairs.pending-research.jsonl"),
        input_kinds={
            split: dict(Counter(p.input_kind for p in pairs if p.split == split))
            for split in quotas
        },
    )
    write_json(output / "summary.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--bundle-run", nargs=2, type=Path, action="append", required=True)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(finalize(args.inventory, args.bundle_run, args.approval, args.output_dir)))
