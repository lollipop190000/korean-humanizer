"""Inspect the supplied NP formats and stage local candidates without granting rights."""

import argparse
import csv
import io
import json
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from korean_humanizer.io import sha256_file, write_json, write_jsonl
from korean_humanizer.nikl_np import candidate_sources, comparison_text, parse_document


def inspect(sjml_zip, csv_zip, output):
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    sjml_sha = sha256_file(sjml_zip)
    csv_docs = defaultdict(list)
    csv_meta = defaultdict(set)
    with zipfile.ZipFile(csv_zip) as z:
        names = [n for n in z.namelist() if n.endswith(".csv")]
        if len(names) != 1 or z.getinfo(names[0]).file_size > 100_000_000:
            raise ValueError("unexpected CSV archive")
        for row in csv.DictReader(io.StringIO(z.read(names[0]).decode("utf-8-sig"))):
            csv_docs[row["file_id"]].append(row["sentence"])
            csv_meta[row["file_id"]].add((row["author_id"], row["text_date"], row["category"]))
    years, categories, disposition = Counter(), Counter(), Counter()
    documents, candidates, failures, seen, author_ids = [], [], [], set(), set()
    repaired = matched = count = 0
    with zipfile.ZipFile(sjml_zip) as z:
        for entry in z.infolist():
            if not entry.filename.endswith(".sjml"):
                continue
            count += 1
            if entry.file_size > 2_000_000:
                raise ValueError("SJML member too large")
            try:
                doc = parse_document(z.read(entry))
                fid = doc["file_id"]
                if fid != Path(entry.filename).stem:
                    raise ValueError("filename and document ID disagree")
                if fid in seen:
                    raise ValueError("duplicate document ID")
                seen.add(fid)
            except (ValueError, ET.ParseError, UnicodeError) as exc:
                failures.append({"member": Path(entry.filename).name, "error": type(exc).__name__})
                continue
            repaired += doc["literal_ampersands_escaped"]
            years[str(doc["year"])] += 1
            categories[doc["category"]] += 1
            author_ids.add(doc["author_id"])
            same = fid in csv_docs and comparison_text(
                "".join(doc["paragraphs"])
            ) == comparison_text("".join(csv_docs[fid]))
            same_meta = csv_meta[fid] == {(doc["author_id"], doc["date"], doc["category"])}
            matched += same and same_meta
            if same and same_meta:
                sources, reason = candidate_sources(doc, f"nikl-np-v1.2:{sjml_sha}#{fid}")
                candidates.extend(sources)
            else:
                reason = "format_mismatch_quarantined"
            disposition[reason] += 1
            documents.append(
                {k: v for k, v in doc.items() if k != "paragraphs"} | {"status": reason}
            )
    report = {
        "sjml_members": count,
        "parsed_documents": len(seen),
        "parse_errors": len(failures),
        "literal_ampersand_repairs": repaired,
        "csv_documents": len(csv_docs),
        "csv_rows": sum(map(len, csv_docs.values())),
        "matched_documents_text_and_metadata": matched,
        "years": dict(years),
        "top_categories": categories.most_common(12),
        "author_ids": len(author_ids),
        "document_disposition": dict(disposition),
        "candidate_excerpts": len(candidates),
        "candidate_authors": len({s.author_group for s in candidates}),
        "rights_approved": 0,
        "gpu_calls": 0,
        "input_sha256": {"sjml": sha256_file(sjml_zip), "csv": sha256_file(csv_zip)},
    }
    write_jsonl(output / "documents.private.jsonl", documents)
    write_jsonl(output / "sources.candidates.private.jsonl", candidates)
    write_jsonl(output / "failures.private.jsonl", failures)
    write_json(output / "summary.json", report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sjml-zip", type=Path, required=True)
    p.add_argument("--csv-zip", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(inspect(a.sjml_zip, a.csv_zip, a.output_dir), ensure_ascii=False))
