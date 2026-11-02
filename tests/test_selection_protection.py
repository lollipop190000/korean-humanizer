"""Explicit source spans remain protected before training selection."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_quarantine_preserves_auto_verdict_but_excludes_training_supply(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    import finalize_scale_pairs as selection

    rows = [
        SimpleNamespace(
            source=SimpleNamespace(
                source_doc_id=k, author_group=k, text="원문", protected_spans=[]
            ),
            split="train",
        )
        for k in ["a", "b"]
    ]
    monkeypatch.setattr(selection, "read_jsonl", lambda *_: rows)
    monkeypatch.setattr(selection, "select_exact", lambda *_args, **_kwargs: None)

    def screen(_bundle, _run, output):
        output.mkdir()
        records = [
            {"case_id": k, "automatic_pass": True, "judgment": {"genre": "experience_review"}}
            for k in ["a", "b"]
        ]
        (output / "screened.private.jsonl").write_text("\n".join(map(json.dumps, records)))

    monkeypatch.setattr(selection.screening, "process", screen)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text("{}")
    (bundle / "drafts.jsonl").write_text(
        "\n".join(json.dumps({"case_id": k, "text": "초안"}) for k in ["a", "b"])
    )
    inventory = tmp_path / "inventory"
    inventory.write_text("fixture")
    approval = tmp_path / "approval.json"
    permission = {
        "job_id": "scale-001",
        "max_total_usd": 30,
        "source_use_and_external_transfer_approved": True,
        "user_approval_ref": "test",
        "selection_excluded_case_ids": ["a"],
    }
    approval.write_text(json.dumps(permission))
    report = selection.finalize(inventory, [(bundle, tmp_path)], approval, tmp_path / "out")
    assert report["automatic_pass"] == 2
    assert report["eligible_after_exclusions"] == 1
    assert report["quota_supply"]["train"] == {"experience_review": 1}
    assert report["accepted_pairs"] == 0
    rows[1].source.text = "&name1&의 원문"
    rows[1].source.protected_spans = ["&name1&"]
    (bundle / "drafts.jsonl").write_text(
        "\n".join(json.dumps({"case_id": k, "text": "name1의 초안"}) for k in ["a", "b"])
    )
    protected = selection.finalize(
        inventory, [(bundle, tmp_path)], approval, tmp_path / "protected"
    )
    assert protected["automatic_pass"] == 2
    assert protected["training_protection_excluded_case_ids"] == ["b"]
    assert protected["eligible_after_exclusions"] == 0
    assert protected["quota_supply"]["train"] == {}
    approval.write_text(json.dumps(permission | {"selection_excluded_case_ids": ["unknown"]}))
    with pytest.raises(ValueError, match="selection exclusions"):
        selection.finalize(inventory, [(bundle, tmp_path)], approval, tmp_path / "invalid")
