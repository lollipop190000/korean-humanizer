"""Offline benchmark requests, integrity checks and result accounting."""

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from korean_humanizer.data import normalized
from korean_humanizer.io import read_jsonl, sha256_file, write_json, write_jsonl
from korean_humanizer.models import Record
from korean_humanizer.protection import check_protection


class DevCase(Record):
    case_id: str = Field(min_length=1)
    group_id: str = Field(min_length=1)
    genre: Literal["report", "explanation", "email", "essay", "post", "official"]
    text: str = Field(min_length=1)
    origin: Literal["ai_authored_development"]
    split: Literal["dev"]
    review_status: Literal["draft", "human_reviewed"]
    review_ref: str = ""
    focus: list[str] = Field(min_length=1)
    expected_edit: Literal["minimal", "rewrite"]
    invariants: list[str] = Field(min_length=1)
    protected_spans: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_review(self):
        if self.review_status == "human_reviewed" and not self.review_ref.strip():
            raise ValueError("human review needs a reference")
        if any(not span or span not in self.text for span in self.protected_spans):
            raise ValueError("protected span is absent")
        if any(not item.strip() for item in self.focus + self.invariants):
            raise ValueError("empty evaluation criterion")
        return self


class BaselineSpec(BaseModel):
    # Prompt whitespace is significant, especially the reference model's separator.
    model_config = ConfigDict(extra="forbid")
    candidate: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]+$")
    model_id: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    protocol: Literal["completion", "chat"]
    instruction: str = Field(min_length=1)
    separator: str = ""
    official_prompt_fingerprint: str | None = None
    temperature: float = Field(gt=0, le=2)
    top_p: float = Field(gt=0, le=1)
    top_k: int = Field(ge=0)
    min_p: float = Field(ge=0, le=1)
    repetition_penalty: float = Field(gt=0)
    enable_thinking: bool = False
    context_tokens: int = Field(gt=0)
    max_output_tokens: int = Field(gt=0)
    seed: int = Field(ge=0)
    source_url: str

    @model_validator(mode="after")
    def valid_context_and_fingerprint(self):
        if self.max_output_tokens >= self.context_tokens:
            raise ValueError("output budget must leave room for the prompt")
        if self.protocol == "completion" and not self.separator:
            raise ValueError("completion protocol needs a separator")
        if self.official_prompt_fingerprint:
            probe = self.instruction + "\n\nX" + self.separator
            if hashlib.sha256(probe.encode()).hexdigest()[:16] != self.official_prompt_fingerprint:
                raise ValueError("official prompt fingerprint mismatch")
        return self


class OutputRow(Record):
    case_id: str
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate: str
    text: str = ""
    finish_reason: Literal["stop", "length", "error"]
    error_code: str = ""
    latency_seconds: float | None = Field(default=None, ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def explicit_failure(self):
        if self.finish_reason == "error" and not self.error_code:
            raise ValueError("failed output needs an error_code")
        if self.finish_reason != "error" and not self.text.strip():
            raise ValueError("non-error output must not be empty")
        if self.finish_reason != "error" and self.error_code:
            raise ValueError("successful output cannot have an error_code")
        return self


def digest(value: dict) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def validate_cases(cases: list[DevCase]) -> dict:
    if not cases:
        raise ValueError("empty development dataset")
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("duplicate case_id")
    if len({normalized(case.text) for case in cases}) != len(cases):
        raise ValueError("duplicate normalized development text")
    return {
        "cases": len(cases),
        "groups": len({case.group_id for case in cases}),
        "genres": dict(Counter(case.genre for case in cases)),
        "edit_expectations": dict(Counter(case.expected_edit for case in cases)),
        "reviews": dict(Counter(case.review_status for case in cases)),
        "origins": dict(Counter(case.origin for case in cases)),
        "independent_holdout": False,
        "human_targets": False,
    }


def request_body(case: DevCase, spec: BaselineSpec) -> dict:
    body = {
        "model": spec.model_id,
        "temperature": spec.temperature,
        "top_p": spec.top_p,
        "top_k": spec.top_k,
        "min_p": spec.min_p,
        "repetition_penalty": spec.repetition_penalty,
        "max_tokens": spec.max_output_tokens,
        "seed": spec.seed,
    }
    if spec.protocol == "completion":
        body["prompt"] = spec.instruction + "\n\n" + case.text.strip() + spec.separator
    else:
        body["messages"] = [
            {"role": "system", "content": spec.instruction},
            {"role": "user", "content": case.text},
        ]
        body["chat_template_kwargs"] = {"enable_thinking": spec.enable_thinking}
    # Rubrics/invariants/genre labels never enter the model request.
    return body


def prepare_bundle(
    dataset: Path,
    spec_path: Path,
    output: Path,
    *,
    allow_unreviewed: bool = False,
    selection: Path | None = None,
) -> dict:
    cases = read_jsonl(dataset, DevCase)
    validate_cases(cases)
    if selection is not None:
        chosen = json.loads(selection.read_text())
        ids = chosen.get("case_ids", [])
        if chosen.get("dataset_sha256") != sha256_file(dataset):
            raise ValueError("selection was made for a different dataset version")
        if not ids or len(ids) != len(set(ids)) or not set(ids) <= {c.case_id for c in cases}:
            raise ValueError("selection IDs must be unique, nonempty and present in the dataset")
        wanted = set(ids)
        cases = [case for case in cases if case.case_id in wanted]
    coverage = validate_cases(cases)
    if not allow_unreviewed and any(case.review_status != "human_reviewed" for case in cases):
        raise ValueError("draft cases need human review or explicit --allow-unreviewed")
    spec = BaselineSpec.model_validate_json(spec_path.read_bytes())
    requests = []
    for case in cases:
        row = {
            "case_id": case.case_id,
            "candidate": spec.candidate,
            "revision": spec.revision,
            "protocol": spec.protocol,
            "body": request_body(case, spec),
        }
        requests.append(row | {"request_sha256": digest(row)})
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    write_jsonl(output / "cases.jsonl", cases)
    write_jsonl(output / "requests.jsonl", requests)
    write_json(output / "spec.json", spec.model_dump())
    manifest = {
        "schema_version": 1,
        "status": "prepared_not_executed",
        "candidate": spec.candidate,
        "model_id": spec.model_id,
        "revision": spec.revision,
        "coverage": coverage,
        "files": {
            name: sha256_file(output / name)
            for name in ("cases.jsonl", "requests.jsonl", "spec.json")
        },
        "source_dataset_sha256": sha256_file(dataset),
        "source_spec_sha256": sha256_file(spec_path),
        "selection_sha256": sha256_file(selection) if selection else None,
        "allow_unreviewed": allow_unreviewed,
        "max_requests": len(requests),
        "max_output_tokens_total": len(requests) * spec.max_output_tokens,
        "tokenization_checked": False,
        "execution_approved": False,
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def load_bundle(bundle: Path) -> tuple[dict, list[DevCase], list[dict], BaselineSpec]:
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf8"))
    for name in ("cases.jsonl", "requests.jsonl", "spec.json"):
        if manifest.get("files", {}).get(name) != sha256_file(bundle / name):
            raise ValueError(f"bundle integrity failure: {name}")
    cases = read_jsonl(bundle / "cases.jsonl", DevCase)
    validate_cases(cases)
    requests = [json.loads(line) for line in (bundle / "requests.jsonl").read_text().splitlines()]
    spec = BaselineSpec.model_validate_json((bundle / "spec.json").read_bytes())
    if len(requests) != len(cases) or manifest["max_requests"] != len(cases):
        raise ValueError("bundle record count mismatch")
    for case, row in zip(cases, requests, strict=True):
        expected = {
            "case_id": case.case_id,
            "candidate": spec.candidate,
            "revision": spec.revision,
            "protocol": spec.protocol,
            "body": request_body(case, spec),
        }
        if row != expected | {"request_sha256": digest(expected)}:
            raise ValueError("request does not match its case and specification")
    if manifest["candidate"] != spec.candidate or manifest["revision"] != spec.revision:
        raise ValueError("manifest model mismatch")
    return manifest, cases, requests, spec


def _observation(before: str, after: str) -> dict:
    return {
        "unchanged": before.strip() == after.strip(),
        "character_length_ratio": len(after) / len(before),
        "hangul_syllables": len(re.findall(r"[가-힣]", after)),
        "latin_letters": len(re.findall(r"[A-Za-z]", after)),
        "contains_think_marker": "<think>" in after or "</think>" in after,
        "semantic_status": "not_assessed",
    }


def import_outputs(bundle: Path, outputs: Path, output_dir: Path) -> dict:
    manifest, cases, requests, spec = load_bundle(bundle)
    rows = read_jsonl(outputs, OutputRow)
    indexed = {row.case_id: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("duplicate output case_id")
    if set(indexed) != {case.case_id for case in cases}:
        raise ValueError("all requested IDs need an output or an explicit error record")
    checks, text_results = [], []
    for case, request in zip(cases, requests, strict=True):
        row = indexed[case.case_id]
        if row.candidate != spec.candidate or row.request_sha256 != request["request_sha256"]:
            raise ValueError("output candidate or request hash mismatch")
        completed = row.finish_reason == "stop"
        protected = check_protection(case.text, row.text, case.protected_spans)
        checks.append(
            {
                "case_id": case.case_id,
                "group_id": case.group_id,
                "genre": case.genre,
                "finish_reason": row.finish_reason,
                "complete": completed,
                "protection": protected,
                "observations": _observation(case.text, row.text),
            }
        )
        if completed:
            text_results.append({"item_id": case.case_id, "text": row.text})
    report = {
        "candidate": spec.candidate,
        "revision": spec.revision,
        "expected": len(cases),
        "received": len(rows),
        "finish_reasons": dict(Counter(row.finish_reason for row in rows)),
        "complete": sum(item["complete"] for item in checks),
        "mechanical_pass_on_complete": sum(
            item["complete"] and item["protection"]["mechanical_pass"] for item in checks
        ),
        "all_complete": all(item["complete"] for item in checks),
        "semantic_quality": "not_assessed",
        "naturalness": "not_assessed",
        "source_manifest_sha256": sha256_file(bundle / "manifest.json"),
        "outputs_sha256": sha256_file(outputs),
        "coverage": manifest["coverage"],
    }
    output_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    write_jsonl(output_dir / "checks.jsonl", checks)
    # Incomplete runs cannot silently produce a passing-only blind comparison dataset.
    if report["all_complete"]:
        write_jsonl(output_dir / "texts.jsonl", text_results)
        write_jsonl(
            output_dir / "inputs.jsonl",
            [{"item_id": case.case_id, "text": case.text} for case in cases],
        )
    write_json(output_dir / "report.json", report)
    return report


def identity_benchmark(dataset: Path, output: Path) -> dict:
    cases = read_jsonl(dataset, DevCase)
    coverage = validate_cases(cases)
    checks = [check_protection(case.text, case.text, case.protected_spans) for case in cases]
    report = {
        "candidate": "identity",
        "coverage": coverage,
        "complete": len(cases),
        "mechanical_pass": sum(check["mechanical_pass"] for check in checks),
        "naturalness": "not_assessed",
        "semantic_quality": "not_assessed",
        "rewrite_model_executed": False,
        "actual_external_spend_usd": "0",
    }
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    write_jsonl(output / "texts.jsonl", [{"item_id": c.case_id, "text": c.text} for c in cases])
    write_json(output / "report.json", report)
    return report
