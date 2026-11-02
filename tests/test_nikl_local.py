"""Local transfer selector tests use synthetic text, never human training examples."""

import importlib.util
import io
import json
import zipfile
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "nikl_local", Path(__file__).resolve().parents[1] / "scripts/select_nikl_local.py"
)
selector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(selector)


def doc(identifier="NIRW1.1", date="20180000"):
    text = "이것은 소프트웨어 시험용 합성 문장입니다. 신청 기간과 제출 서류를 확인하세요. " * 6
    return {
        "id": identifier,
        "metadata": {"author": "가상인", "date": date},
        "paragraph": [{"id": identifier + ".1", "form": text}],
    }


def bundle(rows):
    return {"id": "NIRW1", "metadata": {"year": "2020", "category": "신문"}, "document": rows}


def test_stream_reads_complete_documents_across_chunks_and_ignores_root_year():
    class SmallReads(io.StringIO):
        def read(self, size=-1):
            return super().read(min(size, 7))

    text = json.dumps(bundle([doc(), doc("NIRW1.2", "20200000")]), ensure_ascii=False)
    records = list(selector.documents(SmallReads(text)))
    assert len(records) == 2
    assert selector.candidates(*records[0])
    assert not selector.candidates(*records[1])


def test_truncated_preview_never_produces_incomplete_document():
    raw = json.dumps(bundle([doc(), doc("NIRW1.2")]))
    stream = selector.documents(io.StringIO(raw[:-30]))
    assert next(stream)[1]["id"] == "NIRW1.1"
    with pytest.raises(ValueError):
        next(stream)


def test_selector_round_trip_and_dedup_without_mutating_archives(tmp_path):
    for name in selector.TARGETS:
        with zipfile.ZipFile(tmp_path / name, "w") as z:
            z.writestr("sample.json", json.dumps(bundle([doc()])))
    originals = [(tmp_path / name).read_bytes() for name in selector.TARGETS]
    output = tmp_path / "output"
    report = selector.select(tmp_path, output)
    assert report["selected_excerpts"] == 1
    assert report["unverified_genre_hints"] == {"practical_notice": 1}
    assert not report["rights_approved"]
    assert originals == [(tmp_path / name).read_bytes() for name in selector.TARGETS]
    with zipfile.ZipFile(output / "NIKL-selected-001.zip") as z:
        rows = [json.loads(x) for x in z.read("candidates.private.jsonl").splitlines()]
        assert rows[0]["document_date"] == "20180000"
        assert "author" not in rows[0]
        assert (
            rows[0]["text_sha256"] == selector.hashlib.sha256(rows[0]["text"].encode()).hexdigest()
        )


def test_part_sizes_and_literature_filter(tmp_path):
    selector.write_parts([{"text": "가" * 100}, {"text": "나" * 100}], tmp_path, {}, byte_limit=400)
    assert len(list(tmp_path.glob("*.zip"))) == 2
    assert selector.candidates({"metadata": {"category": "문어 > 책-상상 > 문학"}}, doc()) == []


def test_focused_purpose_cues_do_not_treat_reported_speech_as_author_experience():
    reported = '담당자는 "나는 처음 참여해서 감동을 느꼈다"고 말했다. 작년 신청 건수는 늘었다.'
    assert selector.focused_triage(reported, "행사 실적 보도")[0] is None
    notice = "참가를 희망하는 분은 신청 방법과 제출 서류를 확인하세요. 접수 기간은 5일까지입니다."
    assert selector.focused_triage(notice, "참가 안내")[0] == "practical_notice"
    experience = "나는 처음 그곳을 방문했을 때 걱정을 느꼈다. 돌아보니 내가 겪었던 일이 기억난다."
    assert selector.focused_triage(experience, "방문 후기")[0] == "experience_review"


def test_received_text_hashes_are_excluded_from_new_local_selection(tmp_path, monkeypatch):
    row = selector.candidates(bundle([]), doc())[0]
    value = selector.hashlib.sha256(selector.normalized(row["text"]).encode()).hexdigest()
    monkeypatch.setattr(selector, "EXCLUDED_TEXT_HASHES", {value})
    for name in selector.TARGETS:
        with zipfile.ZipFile(tmp_path / name, "w") as archive:
            archive.writestr("sample.json", json.dumps(bundle([doc()])))
    report = selector.select(tmp_path, tmp_path / "new", focused=True)
    assert report["selected_excerpts"] == 0
    assert report["previously_received_texts_excluded"] == 2


def test_supplement_preserves_four_excerpts_per_document_across_transfers(tmp_path, monkeypatch):
    monkeypatch.setattr(selector, "EXISTING_DOCUMENT_COUNTS", {"NIRW1.1": 4})
    for name in selector.TARGETS:
        with zipfile.ZipFile(tmp_path / name, "w") as archive:
            archive.writestr("sample.json", json.dumps(bundle([doc()])))
    assert selector.select(tmp_path, tmp_path / "new", focused=True)["selected_excerpts"] == 0
