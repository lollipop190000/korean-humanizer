"""Offline, paired prompt study preparation. No cloud or model calls."""

import json
from pathlib import Path

from korean_humanizer.benchmark import (
    BaselineSpec,
    DevCase,
    load_bundle,
    prepare_bundle,
    validate_cases,
)
from korean_humanizer.io import read_jsonl, sha256_file, write_json, write_jsonl
from korean_humanizer.token_audit import audit_tokens


def inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("study path escapes root")
    return path


def read_rubric(path: Path, cases: list[DevCase]) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    indexed = {row["case_id"]: row for row in rows}
    if len(indexed) != len(rows) or set(indexed) != {c.case_id for c in cases}:
        raise ValueError("rubric must cover every case exactly once")
    for row in rows:
        props = row["propositions"]
        if (
            not props
            or len({p["id"] for p in props}) != len(props)
            or any(not p["id"].strip() or not p["claim"].strip() for p in props)
        ):
            raise ValueError("rubric needs unique nonempty proposition IDs and claims")
    return indexed


def prepare_study(root: Path, config_path: Path, metadata_root: Path, output: Path) -> dict:
    config = json.loads(config_path.read_text())
    if config["study_id"] != "prompt-study-002" or config["approved_for_execution"]:
        raise ValueError("preparation must not imply execution approval")
    prompts = config["prompts"]
    if (
        [p["id"] for p in prompts] != ["p0", "p1"]
        or prompts[0]["instruction"] is not None
        or not isinstance(prompts[1]["instruction"], str)
        or not prompts[1]["instruction"].strip()
    ):
        raise ValueError("expected unchanged baseline and one editing prompt")
    selection_path = inside(root, config["anchor_selection"])
    selection = json.loads(selection_path.read_text())
    anchor_path = inside(root, config["anchor_dataset"])
    if sha256_file(anchor_path) != selection["dataset_sha256"]:
        raise ValueError("anchor dataset checksum mismatch")
    anchors = {c.case_id: c for c in read_jsonl(anchor_path, DevCase)}
    ids = selection["case_ids"]
    if len(ids) != 12 or len(set(ids)) != 12 or not set(ids) <= anchors.keys():
        raise ValueError("exactly 12 unique known anchors required")
    cases = [anchors[k] for k in ids] + read_jsonl(
        inside(root, config["challenge_dataset"]), DevCase
    )
    coverage = validate_cases(cases)
    if (
        coverage["cases"] != config["expected_cases"]
        or coverage["cases"] != 36
        or coverage["groups"] != config["expected_groups"]
        or coverage["groups"] != 24
    ):
        raise ValueError("unexpected study coverage")
    rubric_path = inside(root, config["rubric"])
    read_rubric(rubric_path, cases)
    bases = [
        BaselineSpec.model_validate_json(inside(root, p).read_text()) for p in config["models"]
    ]
    if [b.candidate for b in bases] != ["qwen3-4b", "qwen3-8b"]:
        raise ValueError("study needs the two Qwen baselines")
    if len(prompts) * len(bases) * len(cases) != config["expected_requests"]:
        raise ValueError("request count mismatch")
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    write_json(output / "config.json", config)
    write_jsonl(output / "cases.jsonl", cases)
    write_jsonl(output / "rubric.jsonl", list(read_rubric(rubric_path, cases).values()))
    input_review = [
        "# 실행 전 입력 검토",
        "",
        "AI 작성 개발 초안입니다. 의미가 같은 짝인지, 수정 기대와 보존 기준이 타당한지 검토하세요.",
        "모든 항목은 아직 사람 검수 전입니다. 수정하면 새 버전으로 실험을 다시 준비합니다.",
        "",
    ]
    for case in cases:
        input_review.extend(
            [
                f"## {case.case_id}",
                "",
                f"그룹: {case.group_id} / 수정 기대: {case.expected_edit}",
                "",
                case.text,
                "",
                "보존할 내용:",
                "",
                *[
                    f"- {p['claim']}"
                    for p in read_rubric(rubric_path, cases)[case.case_id]["propositions"]
                ],
                "",
                "검토자 / 판단 / 수정 의견: 미작성",
                "",
            ]
        )
    (output / "input-review.ko.md").write_text("\n".join(input_review))
    source_paths = [
        config["anchor_dataset"],
        config["anchor_selection"],
        config["challenge_dataset"],
        config["rubric"],
        *config["models"],
    ]
    source_hashes = {name: sha256_file(inside(root, name)) for name in source_paths}
    cells = []
    for base in bases:
        write_json(output / "bases" / (base.candidate + ".json"), base.model_dump())
        for prompt in prompts:
            candidate = base.candidate + "-" + prompt["id"]
            spec = base.model_dump() | {"candidate": candidate}
            if prompt["instruction"] is not None:
                spec["instruction"] = prompt["instruction"]
            spec_path = output / "specs" / (candidate + ".json")
            write_json(spec_path, spec)
            bundle = output / "bundles" / candidate
            prepare_bundle(output / "cases.jsonl", spec_path, bundle, allow_unreviewed=True)
            audit = audit_tokens(bundle, metadata_root / base.candidate)
            if not audit["all_within_context"]:
                raise ValueError("study exceeds context")
            write_json(bundle / "tokens.json", audit)
            cells.append(
                {
                    "candidate": candidate,
                    "base": base.candidate,
                    "prompt": prompt["id"],
                    "bundle": str(bundle.relative_to(output)),
                    "requests": len(cases),
                    "prompt_tokens_max": audit["prompt_tokens_max"],
                }
            )
    files = {
        str(p.relative_to(output)): sha256_file(p) for p in sorted(output.rglob("*")) if p.is_file()
    }
    manifest = {
        "study_id": config["study_id"],
        "status": "prepared_not_executed",
        "execution_approved": False,
        "coverage": coverage,
        "cells": cells,
        "max_requests": sum(c["requests"] for c in cells),
        "max_output_tokens_total": sum(
            b.max_output_tokens * len(cases) * len(prompts) for b in bases
        ),
        "source_config_sha256": sha256_file(config_path),
        "source_files": source_hashes,
        "files": files,
        "network_calls": 0,
    }
    write_json(output / "study.json", manifest)
    return manifest


def load_study(directory: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads((directory / "study.json").read_text())
    for name, expected in manifest["files"].items():
        if sha256_file(inside(directory, name)) != expected:
            raise ValueError("study integrity mismatch")
    config = json.loads((directory / "config.json").read_text())
    if manifest["study_id"] != "prompt-study-002" or manifest["max_requests"] != 144:
        raise ValueError("unexpected study scope")
    cases = read_jsonl(directory / "cases.jsonl", DevCase)
    read_rubric(directory / "rubric.jsonl", cases)
    jobs = []
    expected_cells = [
        base + "-" + prompt for base in ["qwen3-4b", "qwen3-8b"] for prompt in ["p0", "p1"]
    ]
    if [cell["candidate"] for cell in manifest["cells"]] != expected_cells:
        raise ValueError("missing, duplicate, or reordered study cell")
    for cell in manifest["cells"]:
        bundle = inside(directory, cell["bundle"])
        _, actual_cases, requests, spec = load_bundle(bundle)
        base = BaselineSpec.model_validate_json(
            (directory / "bases" / (cell["base"] + ".json")).read_text()
        )
        prompt = next(p for p in config["prompts"] if p["id"] == cell["prompt"])
        expected = base.model_dump() | {"candidate": cell["candidate"]}
        if prompt["instruction"] is not None:
            expected["instruction"] = prompt["instruction"]
        if spec.model_dump() != expected or actual_cases != cases or len(requests) != 36:
            raise ValueError("study changed more than the instruction")
        audit = json.loads((bundle / "tokens.json").read_text())
        if (
            audit["bundle_manifest_sha256"] != sha256_file(bundle / "manifest.json")
            or audit["candidate"] != spec.candidate
            or audit["revision"] != spec.revision
            or not audit["all_within_context"]
        ):
            raise ValueError("study token audit mismatch")
        jobs.append(
            {
                "spec": spec.model_dump(),
                "requests": requests,
                "audit": audit,
                "manifest_sha256": sha256_file(bundle / "manifest.json"),
            }
        )
    return manifest, jobs
