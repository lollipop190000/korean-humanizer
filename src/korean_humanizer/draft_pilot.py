"""Frozen, bounded AI-draft generation job. Preparation never approves sources or spends."""

import hashlib
import html
import json
from collections import Counter
from pathlib import Path

from korean_humanizer.benchmark import BaselineSpec, digest, request_body
from korean_humanizer.io import read_jsonl, sha256_file, write_json, write_jsonl
from korean_humanizer.models import AssignedSource
from korean_humanizer.token_audit import render_chat

INSTRUCTION = (
    "아래 글을 작성자가 AI 도우미에게 문장 정리와 윤문을 부탁한 상황으로 보고 다시 쓰세요. "
    "고른 문장 길이, 정돈된 연결, 평이하고 차분한 설명으로 표현하되 일부러 어색하게 만들거나 "
    "상투어를 삽입하지 마세요. "
    "사실, 인명, 날짜, 숫자, 단위, 인용, 부정, 조건, 불확실성을 보존하세요. "
    "필자의 입장, 감정, 판단, 사과와 확신의 정도를 지우거나 추측하지 마세요. "
    "1인칭과 원래 높임말 수준을 유지하세요. 새로운 정보, 사건, 평가를 추가하지 마세요. "
    "내용이 허용하는 범위에서 문장 구조와 표현을 다시 구성하세요. "
    "설명 없이 다시 쓴 글만 출력하세요."
)


def generation_spec(base):
    return base.model_copy(update={"candidate": "qwen3-8b-draft-v1", "instruction": INSTRUCTION})


def prepare_drafts(splits: Path, metadata: Path, base_spec: Path, output: Path):
    from tokenizers import Tokenizer

    items = read_jsonl(splits, AssignedSource)
    eligible = [s for s in items if s.split in {"train", "dev"}]
    # Prefer distinct source/author groups before a second excerpt from the same group.
    selected = []
    group_counts = Counter()
    while len(selected) < 24:
        available = [
            s
            for s in eligible
            if s.source.source_doc_id not in {x.source.source_doc_id for x in selected}
        ]
        if not available:
            raise ValueError("need 24 non-test source excerpts")
        item = min(available, key=lambda s: (group_counts[s.group_id], s.source.source_doc_id))
        selected.append(item)
        group_counts[item.group_id] += 1
    for item in selected:
        if (
            item.source.origin != "human"
            or item.source.source_year is None
            or item.source.source_year >= 2019
        ):
            raise ValueError("draft pilot needs pre-2019 attributed human source candidates")
    spec = generation_spec(BaselineSpec.model_validate_json(base_spec.read_text()))
    if (
        spec.model_id != "Qwen/Qwen3-8B"
        or spec.revision != "b968826d9c46dd6066d109eabc6255188de91218"
    ):
        raise ValueError("first generation pilot uses the pinned 8B revision")
    meta = json.loads((metadata / "metadata.json").read_text())
    if (meta["model_id"], meta["revision"]) != (spec.model_id, spec.revision):
        raise ValueError("tokenizer does not match generator")
    for n in ["tokenizer.json", "tokenizer_config.json"]:
        if sha256_file(metadata / n) != meta["files"][n]["sha256"]:
            raise ValueError("tokenizer checksum mismatch")
    tokenizer = Tokenizer.from_file(str(metadata / "tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    tc = json.loads((metadata / "tokenizer_config.json").read_text())
    requests = []
    audit = []
    for item in selected:
        # request_body reads only .text; no synthetic DevCase provenance is invented.
        row = {
            "case_id": item.source.source_doc_id,
            "candidate": spec.candidate,
            "revision": spec.revision,
            "protocol": "chat",
            "body": request_body(item.source, spec),
        }
        row["request_sha256"] = digest(row)
        requests.append(row)
        rendered = render_chat(tc["chat_template"], row["body"], tc)
        count = len(tokenizer.encode(rendered, add_special_tokens=False).ids)
        if count + spec.max_output_tokens > spec.context_tokens:
            raise ValueError("generation context overflow")
        audit.append(
            {
                "case_id": row["case_id"],
                "prompt_tokens": count,
                "rendered_prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
            }
        )
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    write_jsonl(output / "selected.sources.jsonl", selected)
    write_jsonl(output / "requests.jsonl", requests)
    write_json(output / "spec.json", spec.model_dump())
    write_json(
        output / "tokens.json",
        {
            "model_id": spec.model_id,
            "revision": spec.revision,
            "tokenizer_sha256": meta["files"]["tokenizer.json"]["sha256"],
            "cases": audit,
        },
    )
    manifest = {
        "job_id": "draft-pilot-001",
        "status": "prepared_rights_review_and_spending_approval_pending",
        "sources_split_sha256": sha256_file(splits),
        "requests": 24,
        "max_output_tokens_total": 24 * spec.max_output_tokens,
        "budget_usd": 3,
        "files": {p.name: sha256_file(p) for p in output.iterdir()},
        "source_reviews_completed": 0,
        "generation_executed": False,
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def load_drafts(directory: Path, rights_review: Path | None = None, *, execute=False):
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["job_id"] != "draft-pilot-001" or manifest["requests"] != 24:
        raise ValueError("unexpected draft pilot scope")
    for name, value in manifest["files"].items():
        if Path(name).name != name or sha256_file(directory / name) != value:
            raise ValueError("draft bundle checksum mismatch")
    items = read_jsonl(directory / "selected.sources.jsonl", AssignedSource)
    requests = [
        json.loads(line) for line in (directory / "requests.jsonl").read_text().splitlines()
    ]
    spec = BaselineSpec.model_validate_json((directory / "spec.json").read_text())
    if (
        spec.model_id,
        spec.revision,
        spec.instruction,
        spec.max_output_tokens,
        spec.context_tokens,
        spec.enable_thinking,
        spec.candidate,
    ) != (
        "Qwen/Qwen3-8B",
        "b968826d9c46dd6066d109eabc6255188de91218",
        INSTRUCTION,
        2048,
        8192,
        False,
        "qwen3-8b-draft-v1",
    ):
        raise ValueError("generator spec outside fixed pilot")
    if (
        spec.temperature,
        spec.top_p,
        spec.top_k,
        spec.min_p,
        spec.repetition_penalty,
        spec.seed,
        spec.protocol,
    ) != (0.7, 0.8, 20, 0, 1, 42, "chat"):
        raise ValueError("sampling differs from fixed generation pilot")
    if (
        len(items) != 24
        or len({x.source.source_doc_id for x in items}) != 24
        or len(requests) != 24
    ):
        raise ValueError("exactly 24 unique source requests required")
    for item in items:
        source = item.source
        if (
            source.origin != "human"
            or source.source_year is None
            or source.source_year >= 2019
            or "denied"
            in (
                source.rights.training,
                source.rights.derivative_creation,
                source.rights.third_party_transfer,
            )
        ):
            raise ValueError("source is outside date/origin/rights scope")
    for item, row in zip(items, requests, strict=True):
        expected = {
            "case_id": item.source.source_doc_id,
            "candidate": spec.candidate,
            "revision": spec.revision,
            "protocol": "chat",
            "body": request_body(item.source, spec),
        }
        if item.split == "test" or row != {**expected, "request_sha256": digest(expected)}:
            raise ValueError("source/split/request mismatch; test sources are locked")
    audit = json.loads((directory / "tokens.json").read_text())
    if (audit["model_id"], audit["revision"]) != (spec.model_id, spec.revision):
        raise ValueError("audit revision mismatch")
    if [r["case_id"] for r in audit["cases"]] != [r["case_id"] for r in requests] or any(
        r["prompt_tokens"] + 2048 > 8192 for r in audit["cases"]
    ):
        raise ValueError("token audit mismatch")
    if execute:
        if not rights_review:
            raise ValueError("documented human source-rights review is required before transfer")
        approval = json.loads(rights_review.read_text())
        if (
            approval.get("sources_sha256") != sha256_file(directory / "selected.sources.jsonl")
            or approval.get("reviewer_kind") != "human"
            or not approval.get("reviewer_id", "").strip()
            or not approval.get("review_ref", "").strip()
            or set(approval.get("approved_uses", []))
            != {"training", "derivative_creation", "third_party_transfer"}
        ):
            raise ValueError(
                "source-rights review does not cover this source set and required uses"
            )
    return {
        "job_id": "draft-pilot-001",
        "draft_manifest_sha256": sha256_file(directory / "manifest.json"),
        "jobs": [
            {
                "spec": spec.model_dump(),
                "requests": requests,
                "audit": audit,
                "manifest_sha256": sha256_file(directory / "manifest.json"),
            }
        ],
    }


def review_html(directory: Path, catalog: Path):
    articles = {
        a["news_id"]: a for a in [json.loads(line) for line in catalog.read_text().splitlines()]
    }
    items = read_jsonl(directory / "selected.sources.jsonl", AssignedSource)
    parts = [
        '<!doctype html><html lang="ko"><meta charset="utf-8"><title>생성 전 원문 검토</title>',
        "<style>body{font:17px/1.75 system-ui;max-width:900px;margin:40px auto;padding:20px}"
        "article{border:1px solid #ddd;border-radius:12px;padding:24px;margin:20px 0}"
        "pre{white-space:pre-wrap;font:inherit}</style><h1>AI 초안 생성 전 원문 24개 검토</h1>",
        "<p>출처·작성자·작성일·공공누리 텍스트 제1유형 증빙과 발췌 문맥을 확인하세요. "
        "아직 사람 검수·초안 생성·학습을 완료하지 않았습니다. "
        "사진·표·캡션은 사용하지 않습니다.</p>",
    ]
    for item in items:
        s = item.source
        a = articles[s.source_group.rsplit("/", 1)[1]]
        parts.append(
            "<article><h2>"
            + html.escape(s.source_doc_id)
            + "</h2><p>"
            + html.escape(a["author"] + " / " + a["published"] + " / " + item.split)
            + '</p><p><a href="'
            + html.escape(a["url"])
            + '">공식 원문 및 하단 이용 조건 확인</a></p><p>'
            + html.escape(a["license_excerpt"])
            + "</p><pre>"
            + html.escape(s.text)
            + "</pre></article>"
        )
    return "\n".join(parts) + "</html>"


def import_drafts(directory: Path, outputs_path: Path, rights_review: Path, output: Path):
    """Preserve all statuses; complete outputs become pending pairs, never accepted pairs."""
    from korean_humanizer.benchmark import OutputRow
    from korean_humanizer.models import Generation, Pair, Rights
    from korean_humanizer.protection import check_protection

    payload = load_drafts(directory, rights_review, execute=True)
    job = payload["jobs"][0]
    rows = read_jsonl(outputs_path, OutputRow)
    indexed = {row.case_id: row for row in rows}
    if len(indexed) != len(rows) or set(indexed) != {r["case_id"] for r in job["requests"]}:
        raise ValueError("every requested ID needs exactly one output or explicit error")
    sources = read_jsonl(directory / "selected.sources.jsonl", AssignedSource)
    review = json.loads(rights_review.read_text())
    pairs = []
    checks = []
    for item, request in zip(sources, job["requests"], strict=True):
        row = indexed[request["case_id"]]
        if (
            row.candidate != job["spec"]["candidate"]
            or row.request_sha256 != request["request_sha256"]
        ):
            raise ValueError("generated output identity mismatch")
        check = {
            "source_doc_id": row.case_id,
            "finish_reason": row.finish_reason,
            "pair_created": False,
        }
        if row.finish_reason == "stop":
            source = item.source.model_copy(
                update={
                    "rights": Rights(
                        training="approved",
                        derivative_creation="approved",
                        third_party_transfer="approved",
                        evidence_ref=item.source.rights.evidence_ref
                        + "; human-review:"
                        + review["review_ref"],
                    )
                }
            )
            pairs.append(
                Pair(
                    pair_id="draft-pilot-001-" + row.case_id,
                    source=source,
                    split=item.split,
                    synthetic_ai_text=row.text,
                    generation=Generation(
                        model=job["spec"]["model_id"] + "@" + job["spec"]["revision"],
                        prompt_version="ai-polish-v1:" + digest({"instruction": INSTRUCTION}),
                        seed=42,
                        mock=False,
                    ),
                    review_status="pending",
                )
            )
            check.update(
                pair_created=True,
                mechanical=check_protection(source.text, row.text, source.protected_spans),
            )
        checks.append(check)
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    if pairs:
        write_jsonl(output / "pairs.pending.jsonl", pairs)
    write_jsonl(output / "checks.jsonl", checks)
    report = {
        "requested": 24,
        "complete": len(pairs),
        "incomplete": 24 - len(pairs),
        "accepted_pairs": 0,
        "semantic_review": "pending",
        "training_executed": False,
        "outputs_sha256": sha256_file(outputs_path),
    }
    write_json(output / "report.json", report)
    return report
