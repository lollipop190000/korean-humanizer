"""Synthetic exports verify import integrity, dates and pending rights."""

import hashlib
import importlib.util
import json
import zipfile
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "nikl_import", Path(__file__).resolve().parents[1] / "scripts/import_nikl_selection.py"
)
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def row():
    text = ("소프트웨어 시험을 위한 합성 문장이며 실제 사람의 글이 아닙니다. " * 8).strip()
    return {
        "text": text,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "document_date": "20180000",
        "author_group_hash": "a" * 64,
        "document_id": "NIRW1.1",
        "source_zip": "NIKL_NEWSPAPER_v2.0_JSON.zip",
        "source_member": "NIRW1.json",
        "category": "신문",
        "rights_status": "pending_agreement_review",
    }


def test_normalization_keeps_pending_rights_and_known_author_identity():
    r = row()
    result = m.normalize_row(r, "fixture", {"a" * 64: {"known-author"}})
    assert result.author_group == "known-author"
    assert result.text == r["text"] and result.source_year == 2018
    assert result.rights.training == "pending" and result.genre == "unclassified"
    with pytest.raises(ValueError, match="ambiguous"):
        m.normalize_row(r, "fixture", {"a" * 64: {"author1", "author2"}})


@pytest.mark.parametrize(
    "change",
    [
        {"text_sha256": "0" * 64},
        {"document_date": "20200000"},
        {"document_date": "20180230"},
        {"rights_status": "approved"},
    ],
)
def test_corrupted_or_out_of_scope_metadata_is_rejected(change):
    with pytest.raises(ValueError):
        m.normalize_row(row() | change, "fixture", {})


def test_package_content_hash_and_external_summary_must_match(tmp_path):
    raw = (json.dumps(row()) + "\n").encode()
    summary = {"schema": "nikl-local-selection-v1", "selected_excerpts": 1}
    manifest = summary | {"content_sha256": hashlib.sha256(raw).hexdigest(), "rows_in_part": 1}
    path = tmp_path / "export.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("manifest.json", json.dumps(manifest))
        z.writestr("candidates.private.jsonl", raw)
    assert len(m.verified_rows(path, summary)[1]) == 1
    with pytest.raises(ValueError, match="summary"):
        m.verified_rows(path, summary | {"selected_excerpts": 2})
    with zipfile.ZipFile(tmp_path / "corrupt.zip", "w") as z:
        z.writestr("manifest.json", json.dumps(manifest))
        z.writestr("candidates.private.jsonl", raw.replace(b"NIRW1", b"NIRW2"))
    with pytest.raises(ValueError, match="hash"):
        m.verified_rows(tmp_path / "corrupt.zip", summary)
