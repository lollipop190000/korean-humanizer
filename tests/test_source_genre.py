"""Source-only labels cannot masquerade as semantic acceptance."""

import json

import pytest
from test_scale_draft_bundle import bundle, module

from korean_humanizer.io import sha256_file
from korean_humanizer.source_genre import parse_label
from korean_humanizer.source_scope import exclusion_reason


def test_genre_labels_fail_closed_without_inventing_acceptance():
    assert parse_label(" D\n") == "practical_notice"
    for text in ["D because", "B/D", "실용 안내", "", "d"]:
        assert parse_label(text) == "unclassified"


def test_newly_observed_adjudication_forms_are_outside_initial_scope():
    assert exclusion_reason("합성 검사 문장: 조정 결정에 관한 사례 판단.")
    assert exclusion_reason("합성 검사 문장: 민사상 책임 판단.")
    assert not exclusion_reason("행사 신청 방법과 접수 기간을 안내합니다.")


def test_source_genre_bundle_is_bound_and_cannot_be_used_as_pair_judgment(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(module("prepare_scale_drafts").Path("scripts").resolve()))
    original = bundle(tmp_path)
    output = tmp_path / "source-genres"
    module("prepare_scale_drafts").prepare(
        original / "selected.sources.jsonl",
        tmp_path / "metadata",
        output,
        count=2,
        source_genre_only=True,
    )
    manifest, spec, requests, _, _ = module("generate_scale_batch").load(output)
    assert manifest["job_id"] == "scale-001-source-genre"
    assert spec.max_output_tokens == 8 and len(requests) == 2
    with pytest.raises(ValueError, match="judge bundle"):
        module("screen_scale_outputs").process(output, tmp_path / "missing", tmp_path / "not-made")
    path = output / "spec.json"
    value = json.loads(path.read_text())
    value["max_output_tokens"] = 100
    path.write_text(json.dumps(value))
    manifest["files"]["spec.json"] = sha256_file(path)
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="source genre"):
        module("generate_scale_batch").load(output)
