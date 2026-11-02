"""Read-only web recovery, CPU pair review, and explicit human adjudication."""

import argparse
import json
import os
from pathlib import Path

from korean_humanizer.web_review import (
    adjudicate_web_review,
    fetch_snapshot,
    import_web_review,
)

parser = argparse.ArgumentParser(description=__doc__)
commands = parser.add_subparsers(dest="command", required=True)
fetch = commands.add_parser("fetch", help="Read the owner's API; never creates survey responses")
fetch.add_argument("--base-url", required=True)
fetch.add_argument("--study-id", required=True)
fetch.add_argument("--credentials-json", type=Path)
fetch.add_argument("--output-dir", type=Path, required=True)
recover = commands.add_parser("import")
recover.add_argument("--snapshot", type=Path, required=True)
recover.add_argument("--mapping", type=Path, required=True)
recover.add_argument("--pairs", type=Path, required=True)
recover.add_argument("--target-reviews", type=int, default=2)
recover.add_argument("--output-dir", type=Path, required=True)
judge = commands.add_parser("adjudicate")
judge.add_argument("--bundle", type=Path, required=True)
judge.add_argument("--pairs", type=Path, required=True)
judge.add_argument("--decisions", type=Path, required=True)
judge.add_argument("--output-dir", type=Path, required=True)
args = parser.parse_args()
if args.command == "fetch":
    password = (
        json.loads(args.credentials_json.read_text())["ADMIN_PASSWORD"]
        if args.credentials_json
        else os.environ.get("KH_SURVEY_ADMIN_PASSWORD", "")
    )
    report = fetch_snapshot(args.base_url, args.study_id, password, args.output_dir)
elif args.command == "import":
    report = import_web_review(
        args.snapshot, args.mapping, args.pairs, args.output_dir, target_reviews=args.target_reviews
    )
else:
    report = adjudicate_web_review(args.bundle, args.pairs, args.decisions, args.output_dir)
print(json.dumps(report, ensure_ascii=False, indent=2))
