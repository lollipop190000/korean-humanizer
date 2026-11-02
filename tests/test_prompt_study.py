import importlib.util
import json
import shutil
from collections import defaultdict
from pathlib import Path

import pytest

from korean_humanizer.benchmark import BaselineSpec, DevCase, OutputRow, request_body
from korean_humanizer.io import read_jsonl, sha256_file, write_jsonl
from korean_humanizer.prompt_study import load_study, prepare_study
from korean_humanizer.semantic_review import review_pack

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def study(tmp_path, monkeypatch):
    def fake_token_audit(bundle, _metadata):
        spec = json.loads((bundle / "spec.json").read_text())
        return {
            "candidate": spec["candidate"],
            "revision": spec["revision"],
            "bundle_manifest_sha256": sha256_file(bundle / "manifest.json"),
            "all_within_context": True,
            "prompt_tokens_max": 100,
        }

    monkeypatch.setattr("korean_humanizer.prompt_study.audit_tokens", fake_token_audit)
    output = tmp_path / "study"
    prepare_study(ROOT, ROOT / "configs/prompt_study.v1.json", tmp_path / "unused", output)
    return output


def test_matched_pairs_are_grouped_and_remain_drafts():
    cases = read_jsonl(ROOT / "eval/dev_challenge.v1.jsonl", DevCase)
    groups = defaultdict(list)
    for case in cases:
        groups[case.group_id].append(case)
        assert case.review_status == "draft" and case.origin == "ai_authored_development"
    assert len(groups) == 12
    for pair in groups.values():
        assert {c.expected_edit for c in pair} == {"minimal", "rewrite"}
        assert pair[0].invariants == pair[1].invariants
        assert pair[0].genre == pair[1].genre


def test_study_only_changes_instruction_and_preserves_anchor_requests(study):
    manifest, jobs = load_study(study)
    assert manifest["max_requests"] == 144
    assert manifest["coverage"]["groups"] == 24
    assert not manifest["execution_approved"]
    anchors = {c.case_id: c for c in read_jsonl(ROOT / "eval/dev_seed.v1.jsonl", DevCase)}
    for a, b in [(jobs[0], jobs[1]), (jobs[2], jobs[3])]:
        original = BaselineSpec.model_validate_json(
            (
                ROOT / "configs/baselines" / (a["spec"]["candidate"].removesuffix("-p0") + ".json")
            ).read_text()
        )
        unchanged = a["spec"] | {
            "candidate": b["spec"]["candidate"],
            "instruction": b["spec"]["instruction"],
        }
        assert unchanged == b["spec"]
        for request in a["requests"]:
            if request["case_id"] in anchors:
                assert request["body"] == request_body(anchors[request["case_id"]], original)
            assert "propositions" not in request["body"]


def test_study_rejects_modified_or_missing_condition(study):
    path = study / "specs/qwen3-4b-p1.json"
    path.write_text(path.read_text().replace('"seed": 42', '"seed": 43'))
    with pytest.raises(ValueError, match="integrity"):
        load_study(study)


def test_source_edit_or_rubric_omission_blocks_preparation(tmp_path, monkeypatch):
    config = json.loads((ROOT / "configs/prompt_study.v1.json").read_text())
    source = tmp_path / "source"
    for name in [
        config["anchor_dataset"],
        config["anchor_selection"],
        config["challenge_dataset"],
        config["rubric"],
        *config["models"],
        "configs/prompt_study.v1.json",
    ]:
        dest = source / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
    rubric = source / config["rubric"]
    rubric.write_text("\n".join(rubric.read_text().splitlines()[:-1]) + "\n")
    with pytest.raises(ValueError, match="cover every case"):
        prepare_study(
            source, source / "configs/prompt_study.v1.json", tmp_path, tmp_path / "blocked"
        )
    assert not (tmp_path / "blocked").exists()


def test_review_pack_hides_conditions_and_preserves_failures(study, tmp_path):
    _, jobs = load_study(study)
    result_dir = tmp_path / "fixture-results"
    cases = {c.case_id: c for c in read_jsonl(study / "cases.jsonl", DevCase)}
    for job in jobs:
        candidate = job["spec"]["candidate"]
        rows = [
            OutputRow(
                case_id=r["case_id"],
                candidate=candidate,
                request_sha256=r["request_sha256"],
                text=cases[r["case_id"]].text,
                finish_reason="stop",
            )
            for r in job["requests"]
        ]
        rows[-1] = rows[-1].model_copy(
            update={"text": "", "finish_reason": "error", "error_code": "fixture_error"}
        )
        write_jsonl(result_dir / (candidate + ".outputs.jsonl"), rows)
    target = tmp_path / "review"
    report = review_pack(study, result_dir, target, seed=17)
    assert report["tasks"] == 144 and report["ratings_completed"] == 0
    tasks = [json.loads(s) for s in (target / "reviewer/tasks.jsonl").read_text().splitlines()]
    assert sum(t["finish_reason"] == "error" for t in tasks) == 4
    for task in tasks:
        assert set(task) == {"task_id", "input", "output", "finish_reason", "propositions"}
    key = json.loads((target / "private-key.json").read_text())
    assert not key["share_with_raters"]
    (result_dir / (jobs[0]["spec"]["candidate"] + ".outputs.jsonl")).unlink()
    with pytest.raises(FileNotFoundError):
        review_pack(study, result_dir, tmp_path / "missing")


@pytest.mark.parametrize("extra", [[], ["--execute"], ["--execute", "--study-sha256", "0" * 64]])
def test_study_dry_run_and_missing_reviewed_hash_never_call_cloud(
    study, tmp_path, monkeypatch, extra
):
    spec = importlib.util.spec_from_file_location("study_runner", ROOT / "scripts/run_gpu_smoke.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    def forbidden(*args, **kwargs):
        raise AssertionError("unexpected cloud call")

    monkeypatch.setattr(runner, "api", forbidden)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_gpu_smoke.py",
            "--study",
            str(study),
            "--output-dir",
            str(tmp_path / "must-not-exist"),
            *extra,
        ],
    )
    if extra:
        with pytest.raises(SystemExit) as exc:
            runner.main()
        assert exc.value.code == 2
    else:
        runner.main()
    assert not (tmp_path / "must-not-exist").exists()
