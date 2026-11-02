"""Freeze the first 24 real-source generation requests and tokenizer audit. Offline only."""

import argparse
import json
from pathlib import Path

from korean_humanizer.draft_pilot import prepare_drafts, review_html

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--sources-split", type=Path, required=True)
parser.add_argument("--catalog", type=Path, required=True)
parser.add_argument("--metadata", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
args = parser.parse_args()
report = prepare_drafts(
    args.sources_split, args.metadata, Path("configs/baselines/qwen3-8b.json"), args.output_dir
)
(args.output_dir / "원문검토.html").write_text(review_html(args.output_dir, args.catalog))
print(json.dumps(report, ensure_ascii=False, indent=2))
