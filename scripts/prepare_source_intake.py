"""Prepare already-normalized sources locally; no native corpus schema assumptions."""

import argparse
import json
from pathlib import Path

from korean_humanizer.io import read_jsonl, sha256_file, write_json, write_jsonl
from korean_humanizer.models import AssignedSource, Source
from korean_humanizer.source_intake import prepare_intake


def run(frozen_path, source_paths, output):
    hashes = {str(p): sha256_file(p) for p in [frozen_path, *source_paths]}
    frozen = read_jsonl(frozen_path, AssignedSource)
    sources = [s for p in source_paths for s in read_jsonl(p, Source)]
    extended, excluded, report = prepare_intake(frozen, sources)
    if hashes != {str(p): sha256_file(p) for p in [frozen_path, *source_paths]}:
        raise ValueError("input changed during intake")
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    write_jsonl(output / "sources.extended.private.jsonl", extended)
    excluded_ids = {r["source_doc_id"] for r in excluded}
    additions = [r for r in extended[len(frozen) :] if r.source.source_doc_id not in excluded_ids]
    write_jsonl(output / "sources.additions.private.jsonl", additions)
    write_jsonl(output / "exclusions.private.jsonl", excluded)
    report["input_sha256"] = hashes
    report["extended_sha256"] = sha256_file(output / "sources.extended.private.jsonl")
    write_json(output / "summary.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-split", type=Path, required=True)
    parser.add_argument("--sources", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.frozen_split, args.sources, args.output_dir)))
