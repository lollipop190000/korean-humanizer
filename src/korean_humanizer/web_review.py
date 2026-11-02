"""Recover private web ratings, bind them to frozen pairs, and record human decisions."""

import hashlib
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, ValidationError, field_validator

from korean_humanizer.io import read_jsonl, sha256_file, write_json, write_jsonl
from korean_humanizer.models import Pair
from korean_humanizer.protection import check_protection
from korean_humanizer.training import sft_records

DIMENSIONS = ("meaning", "register", "sourceQuality", "naturalness")
PRESERVATION = DIMENSIONS[:3]
Choice = Literal["yes", "no", "unclear"]


def digest(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def web_content_hash(study: dict) -> str:
    """Use the same explicit field order as web/lib/validation.ts, not sorted JSON."""
    tasks = []
    for row in study["tasks"]:
        task = {key: row[key] for key in ("id", "original", "rewritten")}
        if "attribution" in row:
            task["attribution"] = {
                key: row["attribution"][key] for key in ("author", "published", "url", "license")
            }
        tasks.append(task)
    content = {key: study[key] for key in ("id", "title", "description")}
    content["tasks"] = tasks
    return hashlib.sha256(
        json.dumps(content, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


class WebRating(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schemaVersion: Literal[1]
    studyId: str = Field(min_length=1)
    contentHash: str = Field(pattern=r"^[a-f0-9]{64}$")
    taskId: str = Field(min_length=1)
    participantId: str = Field(pattern=r"^[a-f0-9-]{36}$")
    meaning: Choice
    register_choice: Choice = Field(alias="register")
    sourceQuality: Choice
    naturalness: Literal["better", "same", "worse", "unclear"]
    notes: str = Field(max_length=2000)
    humanConfirmed: StrictBool
    submittedAt: str

    @field_validator("humanConfirmed")
    @classmethod
    def confirmed(cls, value):
        if value is not True:
            raise ValueError("missing human attestation")
        return value

    @field_validator("submittedAt")
    @classmethod
    def timestamp(cls, value):
        if datetime.fromisoformat(value.replace("Z", "+00:00")).tzinfo is None:
            raise ValueError("timestamp needs a timezone")
        return value


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def fetch_snapshot(base_url: str, study_id: str, password: str, output: Path) -> dict:
    url = urllib.parse.urlsplit(base_url)
    if (
        url.scheme != "https"
        or not url.hostname
        or url.username
        or url.password
        or url.query
        or url.fragment
        or url.path not in ("", "/")
        or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", study_id)
    ):
        raise ValueError("use the owner's HTTPS origin and a valid study ID")
    if len(password) < 24:
        raise ValueError("administrator credential is missing or too short")
    if output.exists():
        raise FileExistsError(output)
    endpoint = (
        base_url.rstrip("/") + "/api/admin/results?" + urllib.parse.urlencode({"study": study_id})
    )
    request = urllib.request.Request(
        endpoint,
        headers={"Authorization": "Bearer " + password, "User-Agent": "KH-Review-Recovery/1.0"},
    )
    try:
        with urllib.request.build_opener(NoRedirect()).open(request, timeout=45) as response:
            raw = response.read(16 * 1024 * 1024 + 1)
    except urllib.error.HTTPError as error:
        raise ValueError(
            f"survey retrieval failed (HTTP {error.code}); no redirect followed"
        ) from None
    except urllib.error.URLError:
        raise ValueError("survey retrieval failed; check the network connection") from None
    if len(raw) > 16 * 1024 * 1024:
        raise ValueError("snapshot exceeds the 16 MiB recovery limit")
    payload = json.loads(raw)
    if payload["study"]["id"] != study_id or not isinstance(payload["ratings"], list):
        raise ValueError("unexpected survey response")
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    with (output / "snapshot.private.json").open("xb") as handle:
        handle.write(raw)
    (output / "snapshot.private.json").chmod(0o600)
    report = {
        "study_id": study_id,
        "endpoint": endpoint,
        "fetched_at": datetime.now(UTC).isoformat(),
        "snapshot_sha256": hashlib.sha256(raw).hexdigest(),
        "ratings_recovered": len(payload["ratings"]),
    }
    write_json(output / "provenance.json", report)
    return report


def _bind(snapshot: Path, mapping_path: Path, pairs_path: Path):
    payload = json.loads(snapshot.read_text())
    study = payload["study"]
    mapping = json.loads(mapping_path.read_text())
    if mapping["pairsSha256"] != sha256_file(pairs_path):
        raise ValueError("source Pair checksum mismatch")
    pairs = read_jsonl(pairs_path, Pair)
    if (
        study["schemaVersion"] != 1
        or study["id"] != mapping["studyId"]
        or study["contentHash"] != mapping["contentHash"]
        or web_content_hash(study) != mapping["contentHash"]
    ):
        raise ValueError("web study content checksum mismatch")
    pair_index = {p.pair_id: p for p in pairs}
    task_index = {t["id"]: t for t in study["tasks"]}
    links = mapping["mapping"]
    linked = {link["taskId"]: link for link in links}
    if (
        len(pair_index) != len(pairs)
        or len(task_index) != len(study["tasks"])
        or len(linked) != len(links)
        or len({link["pairId"] for link in links}) != len(links)
        or set(linked) != set(task_index)
        or {link["pairId"] for link in links} != set(pair_index)
    ):
        raise ValueError("duplicate or incomplete web/Pair mapping")
    for task_id, link in linked.items():
        pair, task = pair_index[link["pairId"]], task_index[task_id]
        if (
            pair.source.source_doc_id != link["sourceDocId"]
            or pair.source.text != task["original"]
            or pair.synthetic_ai_text != task["rewritten"]
            or pair.generation.mock
            or pair.source.origin != "human"
        ):
            raise ValueError("web text or provenance differs from its source Pair")
    return payload, mapping, pairs, linked


def _ratings(payload: dict, links: dict):
    groups = defaultdict(list)
    issues, valid = [], []
    study = payload["study"]
    for index, raw in enumerate(payload["ratings"], 1):
        try:
            rating = WebRating.model_validate(raw)
        except ValidationError as error:
            issues.append(
                {
                    "row": index,
                    "reason": "invalid_fields",
                    "fields": [e["loc"] for e in error.errors()],
                }
            )
            continue
        if rating.studyId != study["id"] or rating.contentHash != study["contentHash"]:
            issues.append({"row": index, "reason": "wrong_study_or_content_hash"})
            continue
        if rating.taskId not in links:
            issues.append({"row": index, "reason": "unknown_task"})
            continue
        notes = rating.notes.strip()
        problem = any(rating.model_dump(by_alias=True)[key] != "yes" for key in PRESERVATION)
        if problem and (
            len(notes) < 5
            or re.fullmatch(r"(없음|모름|특이사항 없음|잘 모르겠음|모르겠음)[.!\s]*", notes)
        ):
            issues.append({"row": index, "reason": "missing_specific_evidence"})
            continue
        groups[(rating.participantId, rating.taskId)].append((index, rating, raw))
    for entries in groups.values():
        signatures = {
            digest({key: raw[key] for key in (*DIMENSIONS, "notes", "humanConfirmed")})
            for _, _, raw in entries
        }
        if len(signatures) > 1:
            issues.append(
                {"rows": [e[0] for e in entries], "reason": "conflicting_duplicate_quarantined"}
            )
            continue
        index, rating, raw = min(entries, key=lambda e: datetime.fromisoformat(e[1].submittedAt))
        valid.append(
            {
                **rating.model_dump(by_alias=True),
                "response_id": digest(raw),
                "source_row": index,
                "pair_id": links[rating.taskId]["pairId"],
                "rater_kind": "human_attested_anonymous",
            }
        )
        if len(entries) > 1:
            issues.append(
                {"rows": [e[0] for e in entries], "reason": "identical_duplicate_counted_once"}
            )
    return valid, issues


def import_web_review(
    snapshot: Path, mapping_path: Path, pairs_path: Path, output: Path, *, target_reviews: int = 2
) -> dict:
    if target_reviews < 1:
        raise ValueError("target_reviews must be positive")
    payload, mapping, pairs, links = _bind(snapshot, mapping_path, pairs_path)
    ratings, issues = _ratings(payload, links)
    tasks = []
    for pair in pairs:
        rows = [r for r in ratings if r["pair_id"] == pair.pair_id]
        differences = [key for key in DIMENSIONS if len({r[key] for r in rows}) > 1]
        mechanical = check_protection(
            pair.synthetic_ai_text, pair.source.text, pair.source.protected_spans
        )
        tasks.append(
            {
                "pair_id": pair.pair_id,
                "source_doc_id": pair.source.source_doc_id,
                "review_status": pair.review_status,
                "responses": len(rows),
                "response_ids": [r["response_id"] for r in rows],
                "needs_more_reviews": len(rows) < target_reviews,
                "differing_dimensions": differences,
                "preservation_disagreement": any(key in PRESERVATION for key in differences),
                "naturalness_disagreement": "naturalness" in differences,
                "reported_problem_or_uncertainty": any(
                    r[key] != "yes" for r in rows for key in PRESERVATION
                ),
                "dimension_counts": {
                    key: dict(Counter(r[key] for r in rows)) for key in DIMENSIONS
                },
                "mechanical": mechanical,
                "final_decision": "not_adjudicated",
            }
        )
    manifest = {
        "schema_version": 1,
        "study_id": mapping["studyId"],
        "content_hash": mapping["contentHash"],
        "snapshot_sha256": sha256_file(snapshot),
        "mapping_sha256": sha256_file(mapping_path),
        "pairs_sha256": sha256_file(pairs_path),
        "target_reviews": target_reviews,
    }
    report = {
        "study_id": mapping["studyId"],
        "tasks": len(tasks),
        "raw_ratings": len(payload["ratings"]),
        "valid_ratings": len(ratings),
        "anonymous_participant_ids": len({r["participantId"] for r in ratings}),
        "tasks_rated": sum(t["responses"] > 0 for t in tasks),
        "unrated_tasks": sum(t["responses"] == 0 for t in tasks),
        "tasks_at_target": sum(not t["needs_more_reviews"] for t in tasks),
        "disagreement_tasks": sum(bool(t["differing_dimensions"]) for t in tasks),
        "preservation_disagreement_tasks": sum(t["preservation_disagreement"] for t in tasks),
        "naturalness_disagreement_tasks": sum(t["naturalness_disagreement"] for t in tasks),
        "mechanical_pass_tasks": sum(t["mechanical"]["mechanical_pass"] for t in tasks),
        "issue_events": len(issues),
        "automatically_accepted": 0,
        "training_executed": False,
    }
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    write_jsonl(output / "ratings.private.jsonl", ratings)
    write_jsonl(output / "issues.private.jsonl", issues)
    write_jsonl(output / "tasks.private.jsonl", tasks)
    manifest["ratings_sha256"] = sha256_file(output / "ratings.private.jsonl")
    manifest["tasks_sha256"] = sha256_file(output / "tasks.private.jsonl")
    write_json(output / "manifest.json", manifest)
    write_json(output / "summary.json", report)
    write_json(
        output / "adjudication.blank.json",
        {
            "manifest_sha256": sha256_file(output / "manifest.json"),
            "reviewer_kind": None,
            "reviewer_id": "",
            "review_ref": "",
            "decisions": [
                {
                    "pair_id": p.pair_id,
                    "decision": "pending",
                    "meaning_equivalent": "unclear",
                    "register_preserved": "unclear",
                    "source_quality": "unclear",
                    "independent_reviewers_confirmed": False,
                    "disagreements_reviewed": False,
                    "evidence_response_ids": [],
                    "reason": "",
                }
                for p in pairs
            ],
        },
    )
    lines = ["# 웹 응답 검수 현황", "", "익명 쿠키 수는 사람 신원 인증이 아닙니다.", ""]
    lines += [f"- {key}: {value}" for key, value in report.items()]
    lines += ["", "| 문장 쌍 | 평가 수 | 의견 차이 항목 | 기계 검사 |", "|---|---:|---|---|"]
    lines += [
        f"| {t['pair_id']} | {t['responses']} | {', '.join(t['differing_dimensions']) or '—'} | "
        f"{'통과' if t['mechanical']['mechanical_pass'] else '확인 필요'} |"
        for t in sorted(tasks, key=lambda t: (not t["differing_dimensions"], t["responses"]))
    ]
    (output / "review-summary.ko.md").write_text("\n".join(lines) + "\n")
    return report


def adjudicate_web_review(bundle: Path, pairs_path: Path, decisions: Path, output: Path) -> dict:
    manifest = json.loads((bundle / "manifest.json").read_text())
    receipt = json.loads(decisions.read_text())
    if receipt["manifest_sha256"] != sha256_file(bundle / "manifest.json"):
        raise ValueError("adjudication belongs to another review snapshot")
    if manifest["pairs_sha256"] != sha256_file(pairs_path):
        raise ValueError("source Pair checksum mismatch")
    for name, key in [
        ("ratings.private.jsonl", "ratings_sha256"),
        ("tasks.private.jsonl", "tasks_sha256"),
    ]:
        if sha256_file(bundle / name) != manifest[key]:
            raise ValueError("review bundle checksum mismatch")
    if (
        receipt.get("reviewer_kind") != "human"
        or not isinstance(receipt.get("reviewer_id"), str)
        or len(receipt["reviewer_id"].strip()) < 4
        or not isinstance(receipt.get("review_ref"), str)
        or len(receipt["review_ref"].strip()) < 8
    ):
        raise ValueError("a named human adjudicator and review reference are required")
    pairs = read_jsonl(pairs_path, Pair)
    index = {p.pair_id: p for p in pairs}
    tasks = {
        t["pair_id"]: t
        for t in map(json.loads, (bundle / "tasks.private.jsonl").read_text().splitlines())
    }
    ratings = {
        r["response_id"]: r
        for r in map(json.loads, (bundle / "ratings.private.jsonl").read_text().splitlines())
    }
    seen = set()
    ref = f"{receipt['review_ref']}#sha256={sha256_file(decisions)}"
    for decision in receipt["decisions"]:
        pair_id = decision["pair_id"]
        if pair_id not in index or pair_id in seen:
            raise ValueError("unknown or duplicate adjudication Pair")
        seen.add(pair_id)
        verdict = decision["decision"]
        if verdict == "pending":
            continue
        if verdict not in ("accepted", "rejected") or len(decision.get("reason", "").strip()) < 10:
            raise ValueError("human decisions require a concrete reason")
        evidence = decision.get("evidence_response_ids", [])
        if len(set(evidence)) != len(evidence) or any(
            e not in ratings or ratings[e]["pair_id"] != pair_id for e in evidence
        ):
            raise ValueError("decision evidence belongs to another Pair")
        if verdict == "accepted":
            participants = {ratings[e]["participantId"] for e in evidence}
            if (
                len(participants) < manifest["target_reviews"]
                or decision.get("independent_reviewers_confirmed") is not True
                or any(
                    decision.get(k) != "yes"
                    for k in ("meaning_equivalent", "register_preserved", "source_quality")
                )
                or (
                    tasks[pair_id]["differing_dimensions"]
                    and decision.get("disagreements_reviewed") is not True
                )
            ):
                raise ValueError(
                    "acceptance requires independent evidence and resolved human review"
                )
        index[pair_id] = index[pair_id].model_copy(
            update={"review_status": verdict, "review_ref": ref}
        )
    accepted = [p for p in index.values() if p.review_status == "accepted"]
    if accepted:
        # Retain all existing rights, protected information and leakage gates.
        sft_records(accepted)
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    write_jsonl(output / "pairs.reviewed.jsonl", list(index.values()))
    write_jsonl(output / "pairs.accepted.jsonl", accepted)
    receipt_path = output / "adjudication.private.json"
    with receipt_path.open("xb") as handle:
        handle.write(decisions.read_bytes())
    receipt_path.chmod(0o600)
    report = {
        "pairs": len(index),
        "accepted": len(accepted),
        "rejected": sum(p.review_status == "rejected" for p in index.values()),
        "pending": sum(p.review_status == "pending" for p in index.values()),
        "training_executed": False,
        "decisions_sha256": sha256_file(receipt_path),
    }
    write_json(output / "summary.json", report)
    return report
