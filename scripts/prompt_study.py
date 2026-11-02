"""Prepare a CPU-only study or blinded reviewer files; never provisions a GPU."""

import argparse
import json
from pathlib import Path

from korean_humanizer.io import read_jsonl, write_json
from korean_humanizer.prompt_study import prepare_study
from korean_humanizer.semantic_review import SemanticRating, review_pack, summarize_semantics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    prepare.add_argument("--metadata-root", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    pack = commands.add_parser("review-pack")
    pack.add_argument("--study", type=Path, required=True)
    pack.add_argument("--results", type=Path, required=True)
    pack.add_argument("--output-dir", type=Path, required=True)
    summary = commands.add_parser("review-summary")
    summary.add_argument("--ratings", type=Path, required=True)
    summary.add_argument("--key", type=Path, required=True)
    summary.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_study(
            args.root,
            args.root / "configs/prompt_study.v1.json",
            args.metadata_root,
            args.output_dir,
        )
        result = {k: v for k, v in result.items() if k not in {"files", "source_files"}}
    elif args.command == "review-pack":
        result = review_pack(args.study, args.results, args.output_dir)
    else:
        result = summarize_semantics(
            read_jsonl(args.ratings, SemanticRating), json.loads(args.key.read_text())
        )
        write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
