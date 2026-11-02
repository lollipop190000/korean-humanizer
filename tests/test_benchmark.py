import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from korean_humanizer.benchmark import (
    BaselineSpec,
    DevCase,
    OutputRow,
    identity_benchmark,
    import_outputs,
    load_bundle,
    prepare_bundle,
    request_body,
    validate_cases,
)
from korean_humanizer.io import read_jsonl, write_json, write_jsonl

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "eval/dev_seed.v1.jsonl"
SPEC = ROOT / "configs/baselines/humanizer.json"


@pytest.fixture
def bundle(tmp_path):
    directory = tmp_path / "bundle"
    prepare_bundle(DATA, SPEC, directory, allow_unreviewed=True)
    return directory


def output_rows(bundle):
    _, cases, requests, spec = load_bundle(bundle)
    return [
        OutputRow(
            case_id=case.case_id,
            request_sha256=request["request_sha256"],
            candidate=spec.candidate,
            text=case.text,
            finish_reason="stop",
        )
        for case, request in zip(cases, requests, strict=True)
    ]


def test_draft_data_cannot_silently_become_reviewed_or_holdout(tmp_path):
    with pytest.raises(ValueError, match="human review"):
        prepare_bundle(DATA, SPEC, tmp_path / "denied")
    assert not (tmp_path / "denied").exists()
    case = read_jsonl(DATA, DevCase)[0].model_dump()
    for update in ({"split": "test"}, {"origin": "human"}, {"review_status": "human_reviewed"}):
        with pytest.raises(ValidationError):
            DevCase.model_validate(case | update)


def test_unique_dataset_and_span_checks():
    cases = read_jsonl(DATA, DevCase)
    assert validate_cases(cases)["cases"] == 72
    with pytest.raises(ValueError, match="case_id"):
        validate_cases(cases + cases[:1])
    with pytest.raises(ValueError, match="normalized"):
        validate_cases(cases + [cases[0].model_copy(update={"case_id": "duplicate"})])
    with pytest.raises(ValidationError):
        DevCase.model_validate(cases[0].model_dump() | {"protected_spans": ["ABSENT"]})


def test_official_prompt_fingerprint_and_rubric_isolation():
    spec = BaselineSpec.model_validate_json(SPEC.read_bytes())
    with pytest.raises(ValidationError, match="fingerprint"):
        BaselineSpec.model_validate(spec.model_dump() | {"separator": "\n### Rewritten:\n"})
    case = read_jsonl(DATA, DevCase)[0].model_copy(update={"invariants": ["DO_NOT_SEND_RUBRIC"]})
    body = request_body(case, spec)
    assert body["prompt"] == spec.instruction + "\n\n" + case.text + spec.separator
    assert "DO_NOT_SEND_RUBRIC" not in json.dumps(body)
    assert "stop" not in body
    assert body["top_k"] == 0 and body["min_p"] == 0


def test_qwen_uses_non_thinking_chat_protocol():
    spec = BaselineSpec.model_validate_json((ROOT / "configs/baselines/qwen3-4b.json").read_bytes())
    body = request_body(read_jsonl(DATA, DevCase)[0], spec)
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert [row["role"] for row in body["messages"]] == ["system", "user"]
    assert "prompt" not in body


def test_bundle_tamper_detected_before_outputs_are_accepted(bundle, tmp_path):
    path = bundle / "requests.jsonl"
    path.write_text(path.read_text().replace("48명", "49명"))
    with pytest.raises(ValueError, match="integrity"):
        load_bundle(bundle)
    assert not (tmp_path / "report").exists()


@pytest.mark.parametrize("failure", ["missing", "duplicate", "wrong_hash", "wrong_candidate"])
def test_results_must_match_every_request(bundle, tmp_path, failure):
    rows = output_rows(bundle)
    if failure == "missing":
        rows.pop()
    elif failure == "duplicate":
        rows.append(rows[0])
    elif failure == "wrong_hash":
        rows[0] = rows[0].model_copy(update={"request_sha256": "0" * 64})
    else:
        rows[0] = rows[0].model_copy(update={"candidate": "wrong"})
    outputs = tmp_path / "outputs.jsonl"
    write_jsonl(outputs, rows)
    with pytest.raises(ValueError):
        import_outputs(bundle, outputs, tmp_path / "report")
    assert not (tmp_path / "report").exists()


def test_failures_and_truncation_do_not_disappear_from_report(bundle, tmp_path):
    rows = output_rows(bundle)
    rows[0] = rows[0].model_copy(
        update={"finish_reason": "error", "error_code": "timeout", "text": ""}
    )
    rows[1] = rows[1].model_copy(update={"finish_reason": "length"})
    outputs = tmp_path / "outputs.jsonl"
    write_jsonl(outputs, rows)
    report = import_outputs(bundle, outputs, tmp_path / "report")
    assert report["complete"] == 70
    assert report["finish_reasons"] == {"error": 1, "length": 1, "stop": 70}
    assert not report["all_complete"]
    assert not (tmp_path / "report/texts.jsonl").exists()


def test_fact_change_is_flagged_but_never_removed_from_complete_comparison(bundle, tmp_path):
    rows = output_rows(bundle)
    index = next(i for i, row in enumerate(rows) if "48명" in row.text)
    rows[index] = rows[index].model_copy(update={"text": rows[index].text.replace("48명", "49명")})
    outputs = tmp_path / "outputs.jsonl"
    write_jsonl(outputs, rows)
    report = import_outputs(bundle, outputs, tmp_path / "report")
    assert report["complete"] == 72
    assert report["mechanical_pass_on_complete"] == 71
    assert report["naturalness"] == "not_assessed"
    assert len((tmp_path / "report/texts.jsonl").read_text().splitlines()) == 72


def test_empty_output_and_hidden_failure_rejected():
    base = {"case_id": "x", "request_sha256": "0" * 64, "candidate": "reference"}
    for extra in (
        {"finish_reason": "stop", "text": " "},
        {"finish_reason": "error", "text": ""},
        {"finish_reason": "stop", "text": "결과", "error_code": "timeout"},
    ):
        with pytest.raises(ValidationError):
            OutputRow.model_validate(base | extra)


def test_identity_is_labelled_without_claiming_model_quality(tmp_path):
    report = identity_benchmark(DATA, tmp_path / "identity")
    assert report["candidate"] == "identity"
    assert report["complete"] == report["mechanical_pass"] == 72
    assert not report["rewrite_model_executed"]
    assert report["naturalness"] == "not_assessed"


def test_saved_bundle_reconstructs_requests_after_reloading(bundle):
    manifest, cases, requests, spec = load_bundle(bundle)
    assert manifest["max_output_tokens_total"] == 72 * 2048
    assert len(cases) == len(requests) == 72
    assert not manifest["execution_approved"]
    assert spec.candidate == "humanizer-reference"


def test_manipulated_manifest_does_not_allow_different_request(bundle):
    from korean_humanizer.io import sha256_file

    path = bundle / "requests.jsonl"
    path.write_text(path.read_text().replace('"seed": 42', '"seed": 43'))
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["requests.jsonl"] = sha256_file(path)
    manifest_path.unlink()
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="does not match"):
        load_bundle(bundle)


def test_smoke_selection_is_bound_to_dataset_version(tmp_path):
    selection = ROOT / "eval/smoke_selection.v1.json"
    manifest = prepare_bundle(
        DATA, SPEC, tmp_path / "smoke", allow_unreviewed=True, selection=selection
    )
    assert manifest["max_requests"] == 12
    assert set(manifest["coverage"]["genres"].values()) == {2}
    wrong = json.loads(selection.read_text()) | {"dataset_sha256": "0" * 64}
    path = tmp_path / "wrong.json"
    write_json(path, wrong)
    with pytest.raises(ValueError, match="different dataset"):
        prepare_bundle(DATA, SPEC, tmp_path / "denied", allow_unreviewed=True, selection=path)
