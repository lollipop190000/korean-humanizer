"""AI-authored software fixtures only; these are never real human ratings or corpus."""

import json
import urllib.error
from copy import deepcopy

import pytest

from korean_humanizer.io import read_jsonl, sha256_file, write_json, write_jsonl
from korean_humanizer.models import Generation, Pair, Rights, Source
from korean_humanizer.training import sft_records
from korean_humanizer.web_review import (
    NoRedirect,
    adjudicate_web_review,
    fetch_snapshot,
    import_web_review,
    web_content_hash,
)


def fixture(tmp_path):
    originals = ["도서관에서는 책을 빌릴 수 있다.", "기온은 25 ℃이며 비가 올 수 있다."]
    rewrites = ["책은 도서관에서 빌릴 수 있다.", "비가 올 수 있으며 기온은 25 ℃이다."]
    pairs = [
        Pair(
            pair_id=f"pair-{i}",
            source=Source(
                source_doc_id=f"doc-{i}",
                author_group=f"author-{i}",
                source_group=f"group-{i}",
                genre="fixture",
                text=text,
                provenance="fixture:AI-authored-software-test",
                origin="human",  # Simulated production metadata, never a real corpus claim.
                source_year=2018,
                rights=Rights(
                    training="approved",
                    derivative_creation="approved",
                    third_party_transfer="approved",
                    evidence_ref="fixture:only",
                ),
            ),
            synthetic_ai_text=rewrites[i],
            split="train" if i == 0 else "dev",
            generation=Generation(
                model="fixture-model", prompt_version="fixture", seed=1, mock=False
            ),
        )
        for i, text in enumerate(originals)
    ]
    study = {
        "schemaVersion": 1,
        "id": "fixture-study",
        "title": "Fixture · 한국어 🙂",
        "description": "AI-authored fixtures only",
        "tasks": [
            {"id": f"task-{i}", "original": p.source.text, "rewritten": p.synthetic_ai_text}
            for i, p in enumerate(pairs)
        ],
    }
    study["contentHash"] = web_content_hash(study)
    payload = {"study": study, "ratings": [], "summary": {"responses": 99999}}
    pair_path = tmp_path / "pairs.jsonl"
    write_jsonl(pair_path, pairs)
    mapping = {
        "studyId": study["id"],
        "contentHash": study["contentHash"],
        "pairsSha256": sha256_file(pair_path),
        "mapping": [
            {"taskId": f"task-{i}", "pairId": p.pair_id, "sourceDocId": p.source.source_doc_id}
            for i, p in enumerate(pairs)
        ],
    }
    mapping_path, snapshot = tmp_path / "mapping.json", tmp_path / "snapshot.json"
    write_json(mapping_path, mapping)
    write_json(snapshot, payload)
    return payload, mapping_path, pair_path, snapshot


def rating(payload, participant=1, task=0, **changes):
    return {
        "schemaVersion": 1,
        "studyId": payload["study"]["id"],
        "contentHash": payload["study"]["contentHash"],
        "taskId": f"task-{task}",
        "participantId": f"00000000-0000-4000-8000-{participant:012d}",
        "meaning": "yes",
        "register": "yes",
        "sourceQuality": "yes",
        "naturalness": "same",
        "notes": "",
        "humanConfirmed": True,
        "submittedAt": "2026-10-22T01:02:03.123Z",
        **changes,
    }


def save_snapshot(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False))


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def import_fixture(tmp_path, rows=2):
    payload, mapping, pairs, snapshot = fixture(tmp_path)
    payload["ratings"] = [rating(payload, participant=i + 1) for i in range(rows)]
    save_snapshot(snapshot, payload)
    output = tmp_path / "recovered"
    report = import_web_review(snapshot, mapping, pairs, output)
    return payload, mapping, pairs, snapshot, output, report


def human_decision(bundle, path, **changes):
    receipt = json.loads((bundle / "adjudication.blank.json").read_text())
    rows = load_jsonl(bundle / "ratings.private.jsonl")
    receipt.update(
        reviewer_kind="human", reviewer_id="fixture-reviewer", review_ref="fixture:review"
    )
    receipt["decisions"][0].update(
        decision="accepted",
        meaning_equivalent="yes",
        register_preserved="yes",
        source_quality="yes",
        independent_reviewers_confirmed=True,
        evidence_response_ids=[r["response_id"] for r in rows if r["pair_id"] == "pair-0"],
        reason="Fixture adjudication only, not a real human judgment.",
    )
    receipt["decisions"][0].update(changes)
    write_json(path, receipt)
    return receipt


def test_recovery_binds_pairs_keeps_unrated_and_never_accepts_clicks(tmp_path):
    _, _, pair_path, _, output, report = import_fixture(tmp_path)
    assert report["valid_ratings"] == 2
    assert report["tasks_rated"] == 1
    assert report["unrated_tasks"] == 1
    assert report["tasks_at_target"] == 1
    assert report["automatically_accepted"] == 0
    assert report["training_executed"] is False
    tasks = load_jsonl(output / "tasks.private.jsonl")
    assert tasks[1]["responses"] == 0
    assert all(t["final_decision"] == "not_adjudicated" for t in tasks)
    with pytest.raises(ValueError, match="documented acceptance"):
        sft_records(read_jsonl(pair_path, Pair))
    with pytest.raises(ValueError, match="named human"):
        adjudicate_web_review(
            output, pair_path, output / "adjudication.blank.json", tmp_path / "bad"
        )


def test_naturalness_difference_is_separate_from_preservation(tmp_path):
    payload, mapping, pairs, snapshot = fixture(tmp_path)
    payload["ratings"] = [rating(payload), rating(payload, 2, naturalness="better")]
    save_snapshot(snapshot, payload)
    report = import_web_review(snapshot, mapping, pairs, tmp_path / "out")
    assert report["naturalness_disagreement_tasks"] == 1
    assert report["preservation_disagreement_tasks"] == 0
    assert report["disagreement_tasks"] == 1


def test_tampered_text_hash_mapping_and_pairs_cannot_be_imported(tmp_path):
    payload, mapping, pairs, snapshot = fixture(tmp_path)
    payload["study"]["tasks"][0]["original"] += " 변조"
    save_snapshot(snapshot, payload)
    with pytest.raises(ValueError, match="content checksum"):
        import_web_review(snapshot, mapping, pairs, tmp_path / "out")
    payload["study"]["contentHash"] = web_content_hash(payload["study"])
    link = json.loads(mapping.read_text())
    link["contentHash"] = payload["study"]["contentHash"]
    mapping.write_text(json.dumps(link))
    save_snapshot(snapshot, payload)
    with pytest.raises(ValueError, match="web text"):
        import_web_review(snapshot, mapping, pairs, tmp_path / "out")
    pairs.write_text(pairs.read_text() + "\n")
    with pytest.raises(ValueError, match="checksum"):
        import_web_review(snapshot, mapping, pairs, tmp_path / "out")


def test_invalid_rows_quarantined_and_duplicate_identities_not_double_counted(tmp_path):
    payload, mapping, pairs, snapshot = fixture(tmp_path)
    good = rating(payload)
    payload["ratings"] = [
        good,
        deepcopy(good),
        rating(payload, 2, humanConfirmed=1),
        rating(payload, 3, meaning="no", notes="없음"),
        rating(payload, 4, contentHash="0" * 64),
        rating(payload, 5, task=99),
        rating(payload, 6),
        rating(payload, 6, naturalness="worse"),
    ]
    save_snapshot(snapshot, payload)
    output = tmp_path / "out"
    report = import_web_review(snapshot, mapping, pairs, output)
    assert report["raw_ratings"] == 8
    assert report["valid_ratings"] == 1
    assert report["anonymous_participant_ids"] == 1
    assert {i["reason"] for i in load_jsonl(output / "issues.private.jsonl")} == {
        "invalid_fields",
        "missing_specific_evidence",
        "wrong_study_or_content_hash",
        "unknown_task",
        "identical_duplicate_counted_once",
        "conflicting_duplicate_quarantined",
    }


def test_empty_snapshot_and_exclusive_output_are_supported(tmp_path):
    payload, mapping, pairs, snapshot = fixture(tmp_path)
    out = tmp_path / "out"
    report = import_web_review(snapshot, mapping, pairs, out)
    assert report["valid_ratings"] == 0 and report["unrated_tasks"] == 2
    assert (out / "ratings.private.jsonl").read_text() == ""
    with pytest.raises(FileExistsError):
        import_web_review(snapshot, mapping, pairs, out)


def test_explicit_human_decision_only_changes_named_pair_and_preserves_source(tmp_path):
    _, _, pair_path, _, bundle, _ = import_fixture(tmp_path)
    decisions = tmp_path / "decisions.json"
    human_decision(bundle, decisions)
    out = tmp_path / "adjudicated"
    report = adjudicate_web_review(bundle, pair_path, decisions, out)
    assert report["accepted"] == 1 and report["pending"] == 1
    original = read_jsonl(pair_path, Pair)[0]
    accepted = read_jsonl(out / "pairs.accepted.jsonl", Pair)[0]
    assert accepted.source == original.source
    assert accepted.synthetic_ai_text == original.synthetic_ai_text
    assert "sha256=" in accepted.review_ref
    assert sha256_file(out / "adjudication.private.json") == sha256_file(decisions)
    assert sft_records([accepted])[0]["purpose"] == "sft"


def test_insufficient_independent_evidence_and_changed_bundle_block_acceptance(tmp_path):
    _, _, pair_path, _, bundle, _ = import_fixture(tmp_path, rows=1)
    decisions = tmp_path / "decisions.json"
    human_decision(bundle, decisions)
    with pytest.raises(ValueError, match="independent evidence"):
        adjudicate_web_review(bundle, pair_path, decisions, tmp_path / "out")
    (bundle / "tasks.private.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="bundle checksum"):
        adjudicate_web_review(bundle, pair_path, decisions, tmp_path / "out")


def test_protected_information_failure_cannot_be_overridden_by_human_checkbox(tmp_path):
    payload, mapping, pair_path, snapshot = fixture(tmp_path)
    pairs = read_jsonl(pair_path, Pair)
    pairs[0] = pairs[0].model_copy(update={"synthetic_ai_text": "책은 12권 빌릴 수 있다."})
    pair_path.unlink()
    write_jsonl(pair_path, pairs)
    payload["study"]["tasks"][0]["rewritten"] = pairs[0].synthetic_ai_text
    payload["study"]["contentHash"] = web_content_hash(payload["study"])
    link = json.loads(mapping.read_text())
    link.update(contentHash=payload["study"]["contentHash"], pairsSha256=sha256_file(pair_path))
    mapping.write_text(json.dumps(link))
    payload["ratings"] = [rating(payload), rating(payload, 2)]
    save_snapshot(snapshot, payload)
    bundle = tmp_path / "bundle"
    import_web_review(snapshot, mapping, pair_path, bundle)
    decisions = tmp_path / "decisions.json"
    human_decision(bundle, decisions)
    with pytest.raises(ValueError, match="protected information"):
        adjudicate_web_review(bundle, pair_path, decisions, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_fetch_rejects_unsafe_origins_without_sending_credentials(tmp_path):
    for url in [
        "http://owner.test",
        "https://user:pass@owner.test",
        "https://owner.test/path",
        "https://owner.test?key=value",
    ]:
        with pytest.raises(ValueError, match="HTTPS origin"):
            fetch_snapshot(url, "study", "x" * 32, tmp_path / "out")


def test_fetch_is_read_only_keeps_snapshot_and_never_follows_redirect(tmp_path, monkeypatch):
    raw = json.dumps({"study": {"id": "fixture-study"}, "ratings": []}).encode()
    requests = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, size):
            return raw

    class Opener:
        redirected = False

        def open(self, request, timeout):
            requests.append(request)
            if self.redirected:
                raise urllib.error.HTTPError(request.full_url, 302, "moved", {}, None)
            return Response()

    opener = Opener()

    def build(handler):
        assert isinstance(handler, NoRedirect)
        return opener

    monkeypatch.setattr("korean_humanizer.web_review.urllib.request.build_opener", build)
    secret = "fixture-password-with-at-least-24-characters"
    output = tmp_path / "snapshot"
    report = fetch_snapshot("https://owner.test", "fixture-study", secret, output)
    assert requests[0].get_method() == "GET" and requests[0].data is None
    assert (output / "snapshot.private.json").read_bytes() == raw
    assert report["snapshot_sha256"] == sha256_file(output / "snapshot.private.json")
    assert secret not in (output / "provenance.json").read_text()
    opener.redirected = True
    with pytest.raises(ValueError, match="no redirect followed"):
        fetch_snapshot("https://owner.test", "fixture-study", secret, tmp_path / "redirect")
    assert not (tmp_path / "redirect").exists()


def test_disagreements_require_explicit_review_before_acceptance(tmp_path):
    payload, mapping, pair_path, snapshot = fixture(tmp_path)
    payload["ratings"] = [rating(payload), rating(payload, 2, naturalness="better")]
    save_snapshot(snapshot, payload)
    bundle = tmp_path / "bundle"
    import_web_review(snapshot, mapping, pair_path, bundle)
    decisions = tmp_path / "decisions.json"
    receipt = human_decision(bundle, decisions)
    with pytest.raises(ValueError, match="resolved human review"):
        adjudicate_web_review(bundle, pair_path, decisions, tmp_path / "out")
    receipt["decisions"][0]["disagreements_reviewed"] = True
    decisions.write_text(json.dumps(receipt))
    assert adjudicate_web_review(bundle, pair_path, decisions, tmp_path / "out")["accepted"] == 1
