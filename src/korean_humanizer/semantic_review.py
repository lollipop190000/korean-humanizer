"""Blinded, proposition-level review records. No automated semantic verdicts."""

import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from korean_humanizer.benchmark import DevCase, OutputRow
from korean_humanizer.io import read_jsonl, sha256_file, write_json, write_jsonl
from korean_humanizer.models import Record
from korean_humanizer.prompt_study import load_study, read_rubric


class PropositionRating(Record):
    id: str = Field(min_length=1)
    verdict: Literal["preserved", "omitted", "changed", "unassessable"]
    evidence: str = ""

    @model_validator(mode="after")
    def require_evidence(self):
        if self.verdict != "preserved" and not self.evidence.strip():
            raise ValueError("non-preserved proposition needs evidence")
        return self


class SemanticRating(Record):
    task_id: str = Field(min_length=1)
    rater_id: str = Field(min_length=1)
    rater_kind: Literal["human", "ai"]
    review_ref: str = Field(min_length=1)
    propositions: list[PropositionRating] = Field(min_length=1)
    unsupported_addition: Literal["absent", "present", "unassessable"]
    register_drift: Literal["absent", "present", "unassessable"]
    naturalness_vs_input: Literal["improved", "same", "worse", "unassessable"]
    evidence: str = ""

    @model_validator(mode="after")
    def complete_rating(self):
        if len({p.id for p in self.propositions}) != len(self.propositions):
            raise ValueError("duplicate proposition judgment")
        if (
            self.unsupported_addition != "absent"
            or self.register_drift != "absent"
            or self.naturalness_vs_input != "same"
        ) and not self.evidence.strip():
            raise ValueError("changed or unassessable quality needs evidence")
        return self


def review_pack(study: Path, results: Path, output: Path, *, seed: int | None = None) -> dict:
    manifest, jobs = load_study(study)
    cases = {c.case_id: c for c in read_jsonl(study / "cases.jsonl", DevCase)}
    rubric = read_rubric(study / "rubric.jsonl", list(cases.values()))
    pending, result_hashes = [], {}
    for job in jobs:
        candidate = job["spec"]["candidate"]
        path = results / (candidate + ".outputs.jsonl")
        rows = read_jsonl(path, OutputRow)
        indexed = {r.case_id: r for r in rows}
        if len(indexed) != len(rows) or set(indexed) != set(cases):
            raise ValueError("review requires every requested output or explicit failure")
        result_hashes[candidate] = sha256_file(path)
        for request in job["requests"]:
            row = indexed[request["case_id"]]
            if row.candidate != candidate or row.request_sha256 != request["request_sha256"]:
                raise ValueError("review output identity mismatch")
            pending.append((candidate, cases[row.case_id], row))
    # Normal use has a private random ordering; a reproducible seed is for tests only.
    rng = random.SystemRandom() if seed is None else random.Random(seed)
    rng.shuffle(pending)
    tasks, forms, mapping = [], [], {}
    for i, (candidate, case, row) in enumerate(pending, 1):
        task_id = f"review-{i:04d}"
        props = rubric[case.case_id]["propositions"]
        tasks.append(
            {
                "task_id": task_id,
                "input": case.text,
                "output": row.text,
                "finish_reason": row.finish_reason,
                "propositions": props,
            }
        )
        forms.append(
            {
                "task_id": task_id,
                "rater_id": "",
                "rater_kind": None,
                "review_ref": "",
                "propositions": [{"id": p["id"], "verdict": None, "evidence": ""} for p in props],
                "unsupported_addition": None,
                "register_drift": None,
                "naturalness_vs_input": None,
                "evidence": "",
            }
        )
        mapping[task_id] = {
            "candidate": candidate,
            "case_id": case.case_id,
            "group_id": case.group_id,
            "expected_edit": case.expected_edit,
            "finish_reason": row.finish_reason,
            "proposition_ids": [p["id"] for p in props],
        }
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    write_jsonl(output / "reviewer" / "tasks.jsonl", tasks)
    write_jsonl(output / "reviewer" / "ratings.blank.jsonl", forms)
    key = {
        "share_with_raters": False,
        "mapping": mapping,
        "outputs_sha256": result_hashes,
        "study_sha256": sha256_file(study / "study.json"),
        "candidates": [c["candidate"] for c in manifest["cells"]],
        "tasks_sha256": sha256_file(output / "reviewer" / "tasks.jsonl"),
    }
    write_json(output / "private-key.json", key)
    return {"tasks": len(tasks), "ratings_completed": 0, "semantic_verdict": "not_assessed"}


def summarize_semantics(ratings: list[SemanticRating], key: dict) -> dict:
    if not ratings:
        raise ValueError("no completed review records")
    mapping = key["mapping"]
    seen, rater_kinds = set(), {}
    counts = defaultdict(Counter)
    groups = defaultdict(set)
    for row in ratings:
        identity = (row.task_id, row.rater_id)
        if identity in seen or row.task_id not in mapping:
            raise ValueError("duplicate review or unknown task")
        seen.add(identity)
        if row.rater_id in rater_kinds and rater_kinds[row.rater_id] != row.rater_kind:
            raise ValueError("rater kind changed")
        rater_kinds[row.rater_id] = row.rater_kind
        target = mapping[row.task_id]
        if {p.id for p in row.propositions} != set(target["proposition_ids"]):
            raise ValueError("every proposition needs a judgment")
        outcome = (row.rater_kind, row.rater_id, target["candidate"], target["expected_edit"])
        counter = counts[outcome]
        groups[outcome].add(target["group_id"])
        counter["rated"] += 1
        incomplete = target["finish_reason"] != "stop"
        counter["incomplete_generation"] += incomplete
        missing = any(p.verdict == "omitted" for p in row.propositions)
        changed = any(p.verdict == "changed" for p in row.propositions)
        unknown = any(p.verdict == "unassessable" for p in row.propositions)
        unknown |= "unassessable" in (
            row.unsupported_addition,
            row.register_drift,
            row.naturalness_vs_input,
        )
        counter["with_omission"] += missing
        counter["with_changed_claim"] += changed
        counter["with_addition"] += row.unsupported_addition == "present"
        counter["with_register_drift"] += row.register_drift == "present"
        counter["unassessable"] += unknown
        counter["naturalness_" + row.naturalness_vs_input] += 1
        preserved = (
            not incomplete
            and not missing
            and not changed
            and not unknown
            and row.unsupported_addition == "absent"
            and row.register_drift == "absent"
        )
        counter["preserved_and_assessable"] += preserved
        appropriate = (
            row.naturalness_vs_input == "improved"
            if target["expected_edit"] == "rewrite"
            else row.naturalness_vs_input in {"improved", "same"}
        )
        counter["preserved_with_appropriate_edit"] += preserved and appropriate
    cells = []
    # A completely unreviewed condition must remain visible for each rater.
    planned_cells = {(m["candidate"], m["expected_edit"]) for m in mapping.values()}
    for rater, kind in rater_kinds.items():
        for candidate, edit in planned_cells:
            counts[(kind, rater, candidate, edit)]
    for identity, counter in sorted(counts.items()):
        kind, rater, candidate, edit = identity
        expected = sum(
            m["candidate"] == candidate and m["expected_edit"] == edit for m in mapping.values()
        )
        cells.append(
            {
                "rater_kind": kind,
                "rater_id": rater,
                "candidate": candidate,
                "expected_edit": edit,
                "expected": expected,
                "unrated": expected - counter["rated"],
                "rated_groups": len(groups[identity]),
                "counts": dict(counter),
            }
        )
    return {
        "tasks_expected": len(mapping),
        "tasks_rated": len({r.task_id for r in ratings}),
        "records": len(ratings),
        "cells": cells,
        "release_verdict": "not_assessed",
        "note": (
            "Human/AI and individual raters remain separate. Paired cases are correlated; "
            "no independent-sample CI or automatic winner."
        ),
    }
