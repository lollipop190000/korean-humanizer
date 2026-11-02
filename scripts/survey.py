"""Build a Google Forms installer or import its consolidated Export CSV. Offline only."""

import argparse
import json
from pathlib import Path

from korean_humanizer.survey import import_survey_csv, prepare_surveys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--review", type=Path, required=True)
    prepare.add_argument("--study-id", required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--batch-size", type=int, default=6)
    collect = commands.add_parser("import")
    collect.add_argument("--package", type=Path, required=True)
    collect.add_argument("--key", type=Path, required=True)
    collect.add_argument("--csv", type=Path, required=True)
    collect.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        root = Path(__file__).resolve().parents[1]
        result = prepare_surveys(
            args.review,
            args.output_dir,
            args.study_id,
            root / "integrations/google_forms/Code.gs",
            batch_size=args.batch_size,
        )
    else:
        result = import_survey_csv(args.package, args.key, args.csv, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
