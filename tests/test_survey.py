import csv
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from korean_humanizer.io import sha256_file, write_json, write_jsonl
from korean_humanizer.survey import import_survey_csv, prepare_surveys

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "integrations/google_forms/Code.gs"


def review_fixture(tmp_path):
    review = tmp_path / "review"
    tasks, mapping = [], {}
    for group in range(6):
        for candidate in ["secret-model-a", "secret-model-b"]:
            task_id = f"t-{group}-{candidate[-1]}"
            tasks.append(
                dict(
                    task_id=task_id,
                    input="<script>fixture</script>",
                    output="수정문",
                    finish_reason="stop",
                    propositions=[{"id": "p1", "claim": "내용"}],
                )
            )
            mapping[task_id] = dict(
                candidate=candidate,
                case_id=f"case-{group}",
                group_id=f"group-{group}",
                expected_edit="minimal",
                finish_reason="stop",
                proposition_ids=["p1"],
            )
    write_jsonl(review / "reviewer/tasks.jsonl", tasks)
    write_json(
        review / "private-key.json",
        dict(
            mapping=mapping,
            study_sha256="a" * 64,
            tasks_sha256=sha256_file(review / "reviewer/tasks.jsonl"),
        ),
    )
    return review


def prepare(tmp_path):
    review = review_fixture(tmp_path)
    output = tmp_path / "survey"
    report = prepare_surveys(review, output, "fixture-study", TEMPLATE, batch_size=3, seed=42)
    return review, output, report


def test_survey_is_complete_group_separated_and_blinded(tmp_path):
    _, output, report = prepare(tmp_path)
    assert report["tasks"] == 12
    assert report["published"] is False
    package = json.loads((output / "package.private.json").read_text())
    entries = [e for b in package["batches"] for e in b["entries"]]
    assert len({e["task"]["task_id"] for e in entries}) == 12
    for batch in package["batches"]:
        assert len(batch["entries"]) <= 3
        assert len({e["meta"]["group_id"] for e in batch["entries"]}) == len(batch["entries"])
    preview = (output / "preview.html").read_text()
    assert "secret-model" not in preview
    assert "group-" not in preview
    assert "<script>fixture</script>" not in preview
    assert "&lt;script&gt;fixture&lt;/script&gt;" in preview
    assert "/*__SURVEY_PACKAGE__*/null" not in (output / "Install.gs").read_text()


def test_rejects_tampering_and_never_overwrites(tmp_path):
    review, output, _ = prepare(tmp_path)
    with pytest.raises(FileExistsError):
        prepare_surveys(review, output, "fixture-study", TEMPLATE)
    with (review / "reviewer/tasks.jsonl").open("a") as f:
        f.write("\n")
    with pytest.raises(ValueError, match="checksum"):
        prepare_surveys(review, tmp_path / "new", "fixture-study", TEMPLATE)


def csv_fixture(output, rows):
    path = output / "responses.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["study_id", "package_sha256", "rating_json"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def exported_row(output):
    package = json.loads((output / "package.private.json").read_text())
    task = package["batches"][0]["entries"][0]["task"]
    rating = dict(
        task_id=task["task_id"],
        rater_id="fixture-human",
        rater_kind="human",
        review_ref="fixture:only",
        propositions=[{"id": "p1", "verdict": "preserved"}],
        naturalness_vs_input="same",
        unsupported_addition="absent",
        register_drift="absent",
    )
    return dict(
        study_id="fixture-study",
        package_sha256=package["package_sha256"],
        rating_json=json.dumps(rating),
    )


def test_import_roundtrip_keeps_unrated_and_other_studies_separate(tmp_path):
    _, output, _ = prepare(tmp_path)
    row = exported_row(output)
    path = csv_fixture(output, [row, dict(row, study_id="another-study")])
    summary = import_survey_csv(
        output / "package.private.json",
        output / "review-key.private.json",
        path,
        tmp_path / "results",
    )
    assert summary["records"] == 1
    assert summary["tasks_rated"] == 1
    assert summary["tasks_expected"] == 12
    assert summary["other_study_rows_skipped"] == 1
    assert summary["release_verdict"] == "not_assessed"


@pytest.mark.parametrize("fault", ["duplicate", "package", "unknown_task"])
def test_import_rejects_mixed_identity_and_duplicates(tmp_path, fault):
    _, output, _ = prepare(tmp_path)
    row = exported_row(output)
    rows = [row]
    if fault == "duplicate":
        rows.append(row)
    elif fault == "package":
        row["package_sha256"] = "wrong"
    else:
        rating = json.loads(row["rating_json"])
        rating["task_id"] = "unknown"
        row["rating_json"] = json.dumps(rating)
    with pytest.raises(ValueError):
        import_survey_csv(
            output / "package.private.json",
            output / "review-key.private.json",
            csv_fixture(output, rows),
            tmp_path / "results",
        )
    assert not (tmp_path / "results").exists()


def test_google_forms_logic_with_node():
    if not shutil.which("node"):
        pytest.skip("Node needed for Apps Script pure-logic tests")
    subprocess.run(["node", str(ROOT / "tests/google_forms.test.cjs")], cwd=ROOT, check=True)
