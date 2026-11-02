"""Verify a local selector export and normalize it without approving corpus rights."""

import argparse
import hashlib
import json
import re
import zipfile
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

from korean_humanizer.data import normalized
from korean_humanizer.io import read_jsonl, sha256_file, write_json, write_jsonl
from korean_humanizer.models import AssignedSource, Rights, Source
from korean_humanizer.source_scope import exclusion_reason


def verified_rows(path, supplied_summary):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if sorted(names) != ["candidates.private.jsonl", "manifest.json"]:
            raise ValueError("unexpected or duplicate export members")
        if any(e.file_size > 20 * 1024 * 1024 for e in archive.infolist()):
            raise ValueError("export member exceeds inspection bound")
        manifest = json.loads(archive.read("manifest.json"))
        raw = archive.read("candidates.private.jsonl")
    if manifest.get("schema") != "nikl-local-selection-v1":
        raise ValueError("unknown selection schema")
    if any(manifest.get(k) != v for k, v in supplied_summary.items()):
        raise ValueError("supplied summary disagrees with archive manifest")
    if hashlib.sha256(raw).hexdigest() != manifest.get("content_sha256"):
        raise ValueError("export content hash mismatch")
    rows = [json.loads(line) for line in raw.splitlines()]
    if len(rows) != manifest.get("rows_in_part"):
        raise ValueError("export row count mismatch")
    return manifest, rows


def normalize_row(row, provenance, alias_map):
    text = row["text"]
    if hashlib.sha256(text.encode()).hexdigest() != row["text_sha256"]:
        raise ValueError("source text hash mismatch")
    if text != text.strip():
        raise ValueError("noncanonical boundary whitespace")
    if not 200 <= len(text) <= 600:
        raise ValueError("source length outside scope")
    stamp = row["document_date"]
    if not re.fullmatch(r"\d{8}", stamp) or not 1000 <= int(stamp[:4]) < 2019:
        raise ValueError("source year outside scope")
    year, month, day = int(stamp[:4]), int(stamp[4:6]), int(stamp[6:])
    if month > 12 or day > 31 or (not month and day):
        raise ValueError("invalid partial date")
    if month and day:
        date(year, month, day)
    author_hash = row["author_group_hash"]
    if not re.fullmatch(r"[a-f0-9]{64}", author_hash):
        raise ValueError("invalid author group hash")
    doc_id = row["document_id"]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", doc_id):
        raise ValueError("invalid document ID")
    dataset = {
        "NIKL_NEWSPAPER_v2.0_JSON.zip": "newspaper-v2",
        "NIKL_WRITTEN_v1.2_JSON.zip": "written-v1.2",
    }.get(row["source_zip"])
    if not dataset or not row["source_member"].endswith(".json"):
        raise ValueError("unknown source archive/member")
    if re.search(r"책-상상|문학|소설|동화|시집|희곡", row["category"]):
        raise ValueError("literary category outside scope")
    if row.get("rights_status") != "pending_agreement_review":
        raise ValueError("selector cannot grant usage rights")
    aliases = alias_map.get(author_hash, set())
    if len(aliases) > 1:
        raise ValueError("ambiguous_existing_author_alias")
    author = next(iter(aliases)) if aliases else f"nikl/byline-sha256/{author_hash}"
    return Source(
        source_doc_id=f"nikl/{dataset}/{doc_id}/{row['text_sha256'][:16]}",
        author_group=author,
        source_group=f"nikl/document/{doc_id}",
        genre="unclassified",
        text=text,
        provenance=provenance,
        origin="human",
        source_year=year,
        rights=Rights(),
        protected_spans=list(dict.fromkeys(re.findall(r"(?:&name\d+&|<name\d+>)", text))),
    )


def run(archives, summary_path, frozen_path, np_path, output):
    if output.exists():
        raise ValueError("output already exists; preserve completed import")
    supplied = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    frozen = read_jsonl(frozen_path, AssignedSource)
    np_sources = read_jsonl(np_path, Source)
    aliases = defaultdict(set)
    for row in frozen:
        group = row.source.author_group
        if group.startswith(("gnews.gg.go.kr/byline/", "korea.kr/policy-reporter/")):
            key = hashlib.sha256(normalized(group.rsplit("/", 1)[-1]).encode()).hexdigest()
            aliases[key].add(group)
    seen_text = {normalized(r.source.text) for r in frozen} | {
        normalized(s.text) for s in np_sources
    }
    rows, manifests = [], []
    for path in archives:
        manifest, part_rows = verified_rows(path, supplied)
        manifests.append(manifest)
        rows.extend(part_rows)
    parts = manifests[0]["parts"]
    if any(m["parts"] != parts for m in manifests) or sorted(m["part"] for m in manifests) != list(
        range(1, parts + 1)
    ):
        raise ValueError("missing, duplicate, or inconsistent parts")
    if len(rows) != supplied["selected_excerpts"]:
        raise ValueError("total candidate count mismatch")
    counts = Counter(r["genre_hint"] for r in rows)
    if dict(counts) != supplied["unverified_genre_hints"]:
        raise ValueError("genre count mismatch")
    author_counts = Counter(r["author_group_hash"] for r in rows)
    if (
        len(author_counts) != supplied["author_groups"]
        or max(author_counts.values(), default=0) > 20
    ):
        raise ValueError("author count/cap mismatch")
    if max(Counter(r["document_id"] for r in rows).values(), default=0) > 4:
        raise ValueError("document cap exceeded")
    sources, evidence, excluded, ids = [], [], [], set()
    linkage = []
    alias_count = 0
    for i, row in enumerate(rows):
        # Hash failures invalidate the entire package; metadata exclusions are quarantined.
        if hashlib.sha256(row["text"].encode()).hexdigest() != row["text_sha256"]:
            raise ValueError("source text hash mismatch")
        try:
            source = normalize_row(
                row, f"nikl-local-selection:{sha256_file(summary_path)}#row-{i + 1}", aliases
            )
        except (ValueError, KeyError, TypeError) as exc:
            excluded.append({"row": i + 1, "reason": str(exc)})
            continue
        reason = exclusion_reason(source.text)
        if normalized(source.text) in seen_text:
            reason = "duplicate_existing_or_np_or_current_text"
            linkage.append(source)
        if reason:
            excluded.append({"row": i + 1, "reason": reason})
            continue
        if source.source_doc_id in ids:
            raise ValueError("conflicting source ID")
        seen_text.add(normalized(source.text))
        ids.add(source.source_doc_id)
        sources.append(source)
        alias_count += not source.author_group.startswith("nikl/")
        evidence.append(row | {"source_doc_id": source.source_doc_id})
    report = {
        "status": "normalized_pending_agreement_genre_and_semantic_review",
        "received_rows": len(rows),
        "normalized_candidates": len(sources),
        "excluded_reasons": dict(Counter(r["reason"] for r in excluded)),
        "known_existing_author_aliases": alias_count,
        "duplicate_linkage_records": len(linkage),
        "remaining_hints_not_verified_genres": dict(Counter(r["genre_hint"] for r in evidence)),
        "author_groups": len({s.author_group for s in sources}),
        "documents": len({s.source_group for s in sources}),
        "np_candidates_before_final_filtering": len(np_sources),
        "frozen_rows_unchanged": True,
        "splitting_executed": False,
        "rights_approved": 0,
        "model_calls": 0,
        "input_sha256": {
            str(p): sha256_file(p) for p in [*archives, summary_path, frozen_path, np_path]
        },
    }
    output.mkdir(parents=True, mode=0o700)
    write_jsonl(output / "sources.normalized.private.jsonl", sources)
    write_jsonl(output / "sources.linkage.private.jsonl", linkage)
    write_jsonl(output / "source-evidence.private.jsonl", evidence)
    write_jsonl(output / "excluded.private.jsonl", excluded)
    write_json(output / "summary.json", report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archives", type=Path, nargs="+", required=True)
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--frozen", type=Path, required=True)
    p.add_argument("--np-sources", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(
        json.dumps(run(a.archives, a.summary, a.frozen, a.np_sources, a.output), ensure_ascii=False)
    )
