"""Portable, blinded survey packages; never calls Google or publishes data."""

import csv
import hashlib
import html
import json
import math
import random
import re
from collections import Counter
from pathlib import Path

from korean_humanizer.io import sha256_file, write_json, write_jsonl
from korean_humanizer.semantic_review import SemanticRating, summarize_semantics


def digest(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def prepare_surveys(
    review: Path,
    output: Path,
    study_id: str,
    template: Path,
    *,
    batch_size: int = 6,
    seed: int | None = None,
) -> dict:
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", study_id):
        raise ValueError("invalid study ID")
    if not 1 <= batch_size <= 12:
        raise ValueError("batch size must be between 1 and 12")
    key = json.loads((review / "private-key.json").read_text())
    task_path = review / "reviewer/tasks.jsonl"
    if sha256_file(task_path) != key["tasks_sha256"]:
        raise ValueError("review tasks checksum mismatch")
    tasks = [json.loads(line) for line in task_path.read_text().splitlines()]
    if not tasks or len({t["task_id"] for t in tasks}) != len(tasks):
        raise ValueError("empty or duplicate review tasks")
    if {t["task_id"] for t in tasks} != set(key["mapping"]):
        raise ValueError("review key coverage mismatch")
    groups = Counter(key["mapping"][t["task_id"]]["group_id"] for t in tasks)
    rng = random.SystemRandom() if seed is None else random.Random(seed)
    rng.shuffle(tasks)
    tasks.sort(key=lambda t: -groups[key["mapping"][t["task_id"]]["group_id"]])
    count = max(math.ceil(len(tasks) / batch_size), max(groups.values()))
    batches = [[] for _ in range(count)]
    for task in tasks:
        meta = key["mapping"][task["task_id"]]
        if {p["id"] for p in task["propositions"]} != set(meta["proposition_ids"]):
            raise ValueError("proposition IDs mismatch")
        if task["finish_reason"] != meta["finish_reason"]:
            raise ValueError("finish reason mismatch")
        eligible = [
            b
            for b in batches
            if len(b) < batch_size and all(x["meta"]["group_id"] != meta["group_id"] for x in b)
        ]
        if not eligible:
            eligible = [[]]
            batches.extend(eligible)
        rng.shuffle(eligible)
        target = min(
            eligible,
            key=lambda b: (
                len(b),
                sum(x["meta"]["candidate"] == meta["candidate"] for x in b),
                sum(x["meta"]["expected_edit"] == meta["expected_edit"] for x in b),
            ),
        )
        target.append({"task": task, "meta": meta})
    package = {
        "schema_version": 1,
        "questionnaire_version": "semantic-v1",
        "template_sha256": sha256_file(template),
        "study_id": study_id,
        "title": "한국어 문장 다듬기 평가",
        "study_sha256": key["study_sha256"],
        "tasks_sha256": key["tasks_sha256"],
        "target_raters_per_task": 2,
        "batches": [],
    }
    for i, batch in enumerate(batches, 1):
        if batch:
            rng.shuffle(batch)
            package["batches"].append({"id": f"batch-{i:03d}", "entries": batch})
    package["package_sha256"] = digest(package)
    source = template.read_text()
    if source.count("/*__SURVEY_PACKAGE__*/null") != 1:
        raise ValueError("template needs exactly one package placeholder")
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    write_json(output / "package.private.json", package)
    write_json(output / "review-key.private.json", key)
    installer = source.replace(
        "/*__SURVEY_PACKAGE__*/null", json.dumps(package, ensure_ascii=False)
    )
    (output / "Install.gs").write_text(installer)
    setup_html = """<!doctype html><html lang="ko"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>한국어 평가 설문 시작하기</title>
<style>body{font:17px/1.7 system-ui;max-width:840px;margin:48px auto;padding:24px;
background:#f4f7fc;color:#172033}section{background:white;border-radius:18px;padding:28px;
margin:20px 0}button,a.button{font:inherit;background:#234bbb;color:white;border:0;
border-radius:9px;padding:12px 18px;cursor:pointer;text-decoration:none;display:inline-block}
textarea{width:100%;height:200px}small{color:#53637c}li{margin-bottom:14px}</style>
<h1>한국어 평가 설문 시작하기</h1>
<p>설문 · 공유 입구 · 실험별 통합 집계를 Google 계정에 만듭니다.</p>
<section><h2>1. 생성 코드를 복사하세요</h2>
<p>이 파일에는 실험 자료와 모델 대응표가 있습니다. 평가자에게는 생성된 설문 링크만 공유하세요.</p>
<button id="copy">생성 코드 복사</button> <span id="status" role="status"></span>
<details><summary>코드 직접 선택</summary>
<textarea id="source" readonly>__SOURCE__</textarea></details>
</section><section><h2>2. Google에서 한 번 실행하세요</h2>
<ol><li><a href="https://script.google.com/home/start"
target="_blank" rel="noopener">Apps Script</a>에서
새 프로젝트를 만들고 Code.gs의 기본 내용을 지운 뒤 복사한 코드를 붙여넣으세요.</li>
<li>저장하고 위쪽 함수 선택에서 <b>setup</b>을 선택해 실행하세요.
이 스크립트가 Forms·Sheets와 자동 실행을 사용할 수 있도록 Google 권한 요청을 확인하세요.</li>
<li>실행 로그의 <b>관리 시트</b> 링크를 여세요. Forms 탭에 공유용 responder_url이
1분마다 한 묶음씩 생깁니다. 생성 중에는 브라우저를 닫아도 됩니다.</li></ol>
</section><section><h2>3. 공유 링크 하나로 묶으세요 (선택)</h2>
<p>Apps Script의 <b>배포 → 새 배포 → 웹 앱</b>을 선택하세요.
실행 사용자는 <b>나</b>, 접근 권한은 평가자가 접속할 수 있는 범위로 설정하세요.
익명 참여는 계정에서 <b>모든 사용자</b>를 허용할 때 가능합니다.</p>
<p>생성된 웹 앱 URL을 공유하면 설문 목록으로 들어옵니다.
웹 앱을 배포하지 않아도 Forms 탭의 개별 설문 링크를 공유할 수 있습니다.</p>
</section><section><h2>처음에는 작은 시험부터</h2>
<p>첫 설문을 열어 실제 평가 1회를 제출하고 <b>refresh</b>를 실행해
Dashboard·Coverage·Export에 들어오는지 확인하세요.
평가용으로 제출한 응답과 기능 확인용 응답은 구분하세요.</p>
<p>다음 실험도 <b>같은 Apps Script 프로젝트</b>에 새 생성 코드를 붙여넣고 setup을 실행하면
같은 관리 시트에 누적됩니다. 기존 결과는 유지됩니다.</p>
<a href="preview.html">이번 설문의 모든 문장과 질문 미리보기</a>
<p><small>아직 Google Forms 생성·배포 전입니다.
이 설치 파일에서는 응답을 수집하지 않습니다.</small></p>
</section><script>
document.getElementById('copy').onclick=async()=>{
 const source=document.getElementById('source');
 try { await navigator.clipboard.writeText(source.value); }
 catch(e) {
  source.parentElement.open=true;source.focus();source.select();
  if(!document.execCommand('copy')){
   document.getElementById('status').textContent='코드를 선택했습니다. 직접 복사해 주세요.';return;
  }
 }
 document.getElementById('status').textContent='복사했습니다. Google 편집기에 붙여넣으세요.';
};
</script></html>"""
    (output / "시작하기.html").write_text(setup_html.replace("__SOURCE__", html.escape(installer)))
    fragments = [
        '<!doctype html><html lang="ko"><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        "<title>한국어 문장 평가 · 설문 미리보기</title>",
        "<style>body{font:17px/1.7 system-ui;max-width:850px;margin:40px auto;padding:20px;"
        "background:#f5f7fb;color:#172033}article{background:white;padding:24px;"
        "border-radius:14px;margin:20px 0}pre{white-space:pre-wrap;font:inherit}"
        "h2{margin-top:48px}li{margin-bottom:8px}</style>",
        "<h1>한국어 문장 다듬기 평가</h1>",
        "<p>설문 내용 미리보기입니다. 이 파일은 응답을 전송하거나 저장하지 않습니다.</p>",
        "<p>모델을 추측하지 말고 문장을 평가해 주세요. "
        "원문이 자연스러우면 그대로 유지해도 좋습니다. "
        "개인정보 대신 같은 평가자 코드를 사용하세요. "
        "각 묶음에는 서로 다른 내용만 들어 있습니다.</p>",
    ]
    for batch in package["batches"]:
        fragments.append(f"<h2>평가 묶음 {html.escape(batch['id'])}</h2>")
        for i, entry in enumerate(batch["entries"], 1):
            task = entry["task"]
            fragments += [
                f"<article><h3>문장 {i}</h3><b>원문</b>",
                f"<pre>{html.escape(task['input'])}</pre><b>수정문</b>",
                f"<pre>{html.escape(task['output'])}</pre>",
                "<p>① 자연스러움: 좋아짐 / 비슷함 / 나빠짐 / 판단 어려움</p>",
                "<p>② 아래 내용 각각: 유지됨 / 빠짐 / 뜻이 바뀜 / 판단 어려움</p><ul>",
                *[f"<li>{html.escape(p['claim'])}</li>" for p in task["propositions"]],
                "</ul><p>③ 원문에 없는 내용 추가: 없음 / 있음 / 판단 어려움</p>",
                "<p>④ 어투·높임말 변화: 없음 / 있음 / 판단 어려움</p>",
                "<p>⑤ 판단 근거: 문제가 있거나 좋아졌다면 "
                "해당 표현과 이유를 적어 주세요.</p></article>",
            ]
    fragments.append("</html>")
    (output / "preview.html").write_text("\n".join(fragments))
    return {
        "study_id": study_id,
        "tasks": len(tasks),
        "forms": len(package["batches"]),
        "package_sha256": package["package_sha256"],
        "published": False,
        "responses": 0,
    }


def import_survey_csv(package_path: Path, key_path: Path, csv_path: Path, output: Path) -> dict:
    package = json.loads(package_path.read_text())
    if (
        digest({k: v for k, v in package.items() if k != "package_sha256"})
        != package["package_sha256"]
    ):
        raise ValueError("package checksum mismatch")
    key = json.loads(key_path.read_text())
    if (key["study_sha256"], key["tasks_sha256"]) != (
        package["study_sha256"],
        package["tasks_sha256"],
    ):
        raise ValueError("review key belongs to another package")
    ratings, skipped = [], 0
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["study_id"] != package["study_id"]:
                skipped += 1
                continue
            if row["package_sha256"] != package["package_sha256"]:
                raise ValueError("response package mismatch")
            ratings.append(SemanticRating.model_validate_json(row["rating_json"]))
    summary = summarize_semantics(ratings, key)
    summary["other_study_rows_skipped"] = skipped
    summary["rater_identity"] = "self_reported; check assigned reviewer codes separately"
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    write_jsonl(output / "ratings.jsonl", ratings)
    write_json(output / "summary.json", summary)
    return summary
