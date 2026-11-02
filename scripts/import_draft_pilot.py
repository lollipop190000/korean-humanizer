"""Recover all draft statuses and create pending, never auto-accepted, training pairs."""

import argparse
import json
from pathlib import Path

from korean_humanizer.draft_pilot import import_drafts

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--bundle", type=Path, required=True)
parser.add_argument("--outputs", type=Path, required=True)
parser.add_argument("--rights-review", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
args = parser.parse_args()
print(
    json.dumps(
        import_drafts(args.bundle, args.outputs, args.rights_review, args.output_dir),
        ensure_ascii=False,
        indent=2,
    )
)
