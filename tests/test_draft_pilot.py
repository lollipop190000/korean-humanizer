"""Frozen-job validation with synthetic software fixtures, never real inference."""

import json
from pathlib import Path

import pytest

from korean_humanizer.benchmark import BaselineSpec, digest
from korean_humanizer.draft_pilot import INSTRUCTION, generation_spec, import_drafts, load_drafts
from korean_humanizer.io import sha256_file, write_json, write_jsonl
from korean_humanizer.models import AssignedSource, Rights, Source

ROOT = Path(__file__).resolve().parents[1]


def bundle_fixture(tmp_path):
    out = tmp_path / "bundle"
    out.mkdir()
    spec = generation_spec(
        BaselineSpec.model_validate_json((ROOT / "configs/baselines/qwen3-8b.json").read_text())
    )
    rows = []
    requests = []
    audit = []
    for i in range(24):
        s = Source(
            source_doc_id=f"s{i}",
            author_group=f"a{i}",
            source_group=f"g{i}",
            genre="fixture",
            text="소프트웨어 검사 문자열이다.",
            provenance="fixture:not-real-corpus",
            origin="human",
            source_year=2017,
            rights=Rights(evidence_ref="fixture:not-real-rights"),
        )
        rows.append(AssignedSource(source=s, split="train", group_id=f"g{i}"))
        body = {
            "model": spec.model_id,
            "temperature": spec.temperature,
            "top_p": spec.top_p,
            "top_k": spec.top_k,
            "min_p": spec.min_p,
            "repetition_penalty": spec.repetition_penalty,
            "max_tokens": 2048,
            "seed": spec.seed,
            "messages": [
                {"role": "system", "content": INSTRUCTION},
                {"role": "user", "content": s.text},
            ],
            "chat_template_kwargs": {"enable_thinking": False},
        }
        request = {
            "case_id": s.source_doc_id,
            "candidate": spec.candidate,
            "revision": spec.revision,
            "protocol": "chat",
            "body": body,
        }
        requests.append({**request, "request_sha256": digest(request)})
        audit.append(
            {"case_id": s.source_doc_id, "prompt_tokens": 20, "rendered_prompt_sha256": "0" * 64}
        )
    write_jsonl(out / "selected.sources.jsonl", rows)
    write_jsonl(out / "requests.jsonl", requests)
    write_json(out / "spec.json", spec.model_dump())
    write_json(
        out / "tokens.json",
        {
            "model_id": spec.model_id,
            "revision": spec.revision,
            "tokenizer_sha256": "0" * 64,
            "cases": audit,
        },
    )
    freeze(out)
    review = tmp_path / "review.json"
    write_json(
        review,
        {
            "sources_sha256": sha256_file(out / "selected.sources.jsonl"),
            "reviewer_kind": "human",
            "reviewer_id": "fixture-not-a-real-rater",
            "review_ref": "fixture",
            "approved_uses": ["training", "derivative_creation", "third_party_transfer"],
        },
    )
    return out, review, requests


def freeze(out):
    p = out / "manifest.json"
    if p.exists():
        p.unlink()
    write_json(
        p,
        {
            "job_id": "draft-pilot-001",
            "requests": 24,
            "files": {x.name: sha256_file(x) for x in out.iterdir()},
        },
    )


def test_execution_requires_bound_human_rights_review(tmp_path):
    out, review, _ = bundle_fixture(tmp_path)
    assert len(load_drafts(out)["jobs"][0]["requests"]) == 24
    with pytest.raises(ValueError, match="human source-rights"):
        load_drafts(out, execute=True)
    assert load_drafts(out, review, execute=True)["job_id"] == "draft-pilot-001"
    record = json.loads(review.read_text())
    record["sources_sha256"] = "wrong"
    review.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="does not cover"):
        load_drafts(out, review, execute=True)


def test_rejects_tampering_and_test_split_even_if_files_rehashed(tmp_path):
    out, _, _ = bundle_fixture(tmp_path)
    p = out / "selected.sources.jsonl"
    rows = [json.loads(line) for line in p.read_text().splitlines()]
    rows[0]["split"] = "test"
    p.unlink()
    write_jsonl(p, rows)
    with pytest.raises(ValueError, match="checksum"):
        load_drafts(out)
    freeze(out)
    with pytest.raises(ValueError, match="test sources are locked"):
        load_drafts(out)


def test_completed_generation_is_still_pending_not_sft_accepted(tmp_path):
    out, review, requests = bundle_fixture(tmp_path)
    rows = [
        {
            "case_id": r["case_id"],
            "candidate": r["candidate"],
            "request_sha256": r["request_sha256"],
            "text": "가짜 생성 결과.",
            "finish_reason": "stop",
        }
        for r in requests
    ]
    rows[0].update(text="", finish_reason="error", error_code="fixture-failed")
    path = tmp_path / "outputs.jsonl"
    write_jsonl(path, rows)
    result = import_drafts(out, path, review, tmp_path / "results")
    assert result["complete"] == 23 and result["incomplete"] == 1 and result["accepted_pairs"] == 0
    pairs = [
        json.loads(line)
        for line in (tmp_path / "results/pairs.pending.jsonl").read_text().splitlines()
    ]
    assert all(p["review_status"] == "pending" for p in pairs)
    assert all(p["generation"]["mock"] is False for p in pairs)


def test_gpu_controller_refuses_transfer_without_review(tmp_path, monkeypatch):
    import importlib.util
    import sys

    out, _, _ = bundle_fixture(tmp_path)
    spec = importlib.util.spec_from_file_location(
        "draft_controller", ROOT / "scripts/run_gpu_smoke.py"
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    def forbidden(*args, **kwargs):
        raise AssertionError("must not reach any deployment call")

    monkeypatch.setattr(runner, "deploy", forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_gpu_smoke.py",
            "--draft-pilot",
            str(out),
            "--draft-manifest-sha256",
            sha256_file(out / "manifest.json"),
            "--output-dir",
            str(tmp_path / "run"),
            "--execute",
        ],
    )
    with pytest.raises(ValueError, match="human source-rights"):
        runner.main()
    assert not (tmp_path / "run").exists()
