"""Invented source metadata/tokenizer fixtures. No actual corpus, GPU or network calls."""

import importlib.util
import json
from pathlib import Path

import pytest
from test_sft import fixture_bundle

from korean_humanizer.io import sha256_file, write_jsonl
from korean_humanizer.models import AssignedSource

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def bundle(tmp_path):
    metadata, pairs, _, _ = fixture_bundle(tmp_path)
    meta_path = metadata / "metadata.json"
    meta = json.loads(meta_path.read_text())
    meta.update(model_id="Qwen/Qwen3-8B", revision="b968826d9c46dd6066d109eabc6255188de91218")
    meta_path.write_text(json.dumps(meta))
    rows = [
        AssignedSource(
            source=p.source.model_copy(
                update={
                    "text": (
                        "인공지능이 만든 가짜 시험 자료다. 실제 사람 글이나 학습 자료가 아니다. "
                    )
                    * 6
                }
            ),
            split=p.split,
            group_id=p.source.source_group,
        )
        for p in pairs
    ]
    sources = tmp_path / "sources.jsonl"
    write_jsonl(sources, rows)
    out = tmp_path / "bundle"
    result = module("prepare_scale_drafts").prepare(sources, metadata, out, count=2)
    assert result["authors"] == 2 and result["gpu_calls"] == 0
    return out


def test_bundle_roundtrip_and_absent_permission_fail_before_gpu_imports(tmp_path):
    out = bundle(tmp_path)
    worker = module("generate_scale_batch")
    manifest, _, requests, _, batches = worker.load(out)
    assert manifest["generation_executed"] is False
    assert len(requests) == 2 and len(batches) == 1
    approval = tmp_path / "denied.json"
    approval.write_text("{}")
    output = tmp_path / "outputs"
    with pytest.raises(ValueError, match="approval required"):
        worker.run(out, output, approval, 600)
    assert not output.exists()


def test_tampered_inputs_and_rehashed_test_sources_are_rejected(tmp_path):
    out = bundle(tmp_path)
    worker = module("generate_scale_batch")
    path = out / "selected.sources.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["split"] = "test"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="checksum"):
        worker.load(out)
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][path.name] = sha256_file(path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="test/fixture"):
        worker.load(out)
