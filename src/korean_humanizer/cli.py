"""Data validation, experiment preparation, and model evaluation CLI."""

import argparse
import json
import sys
import tomllib
from collections import Counter
from pathlib import Path

from pydantic import ValidationError

from korean_humanizer.benchmark import (
    DevCase,
    identity_benchmark,
    import_outputs,
    prepare_bundle,
    validate_cases,
)
from korean_humanizer.costs import CostConfig, estimate
from korean_humanizer.data import assign_splits, source_errors
from korean_humanizer.evaluation import Rating, blind_compare, summarize_ratings
from korean_humanizer.inference import IdentityBaseline, inspect_rewrite
from korean_humanizer.io import read_jsonl, sha256_file, write_json, write_jsonl
from korean_humanizer.models import Pair, Source, TextResult
from korean_humanizer.protection import check_protection
from korean_humanizer.synthesis import generate_mock_pair
from korean_humanizer.training import sft_records, training_plan


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kh", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-sources")
    validate.add_argument("input", type=Path)
    validate.add_argument("--allow-fixtures", action="store_true")
    validate.add_argument("--external-transfer", action="store_true")
    validate.add_argument("--allow-later-years", action="store_true")
    split = commands.add_parser("split")
    split.add_argument("input", type=Path)
    split.add_argument("--output", type=Path, required=True)
    split.add_argument("--seed", type=int, default=42)
    mock = commands.add_parser("mock-pipeline")
    mock.add_argument("input", type=Path)
    mock.add_argument("--output-dir", type=Path, required=True)
    mock.add_argument("--seed", type=int, default=42)
    costs = commands.add_parser("estimate")
    costs.add_argument("config", type=Path)
    check = commands.add_parser("check-protection")
    check.add_argument("original", type=Path)
    check.add_argument("rewritten", type=Path)
    check.add_argument("--span", action="append", default=[])
    plan = commands.add_parser("training-plan")
    plan.add_argument("pairs", type=Path)
    plan.add_argument("--allow-mock", action="store_true")
    export = commands.add_parser("export-sft")
    export.add_argument("pairs", type=Path)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--allow-mock", action="store_true")
    baseline = commands.add_parser("identity")
    baseline.add_argument("input", type=Path)
    blind = commands.add_parser("blind-compare")
    for name in ("inputs", "left", "right"):
        blind.add_argument(f"--{name}", type=Path, required=True)
    blind.add_argument("--output-dir", type=Path, required=True)
    blind.add_argument("--seed", type=int, default=42)
    ratings = commands.add_parser("summarize-ratings")
    ratings.add_argument("ratings", type=Path)
    ratings.add_argument("--key", type=Path, required=True)
    dev = commands.add_parser("benchmark-validate")
    dev.add_argument("dataset", type=Path)
    prepare = commands.add_parser("benchmark-prepare")
    prepare.add_argument("dataset", type=Path)
    prepare.add_argument("--spec", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--allow-unreviewed", action="store_true")
    prepare.add_argument("--selection", type=Path)
    identity = commands.add_parser("benchmark-identity")
    identity.add_argument("dataset", type=Path)
    identity.add_argument("--output-dir", type=Path, required=True)
    results = commands.add_parser("benchmark-import")
    results.add_argument("--bundle", type=Path, required=True)
    results.add_argument("--outputs", type=Path, required=True)
    results.add_argument("--output-dir", type=Path, required=True)
    tokens = commands.add_parser("benchmark-tokens")
    tokens.add_argument("--bundle", type=Path, required=True)
    tokens.add_argument("--metadata-dir", type=Path, required=True)
    tokens.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("benchmark-run-local")
    run.add_argument("--bundle", type=Path, required=True)
    run.add_argument("--base-url", default="http://127.0.0.1:8000")
    run.add_argument("--output-dir", type=Path)
    run.add_argument("--token-audit", type=Path)
    run.add_argument("--execute", action="store_true")
    run.add_argument("--max-seconds", type=float, default=900)
    run.add_argument("--request-timeout", type=float, default=120)
    return parser


def _emit(value: dict) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def _mock_pipeline(args: argparse.Namespace) -> dict:
    sources = read_jsonl(args.input, Source)
    assigned = assign_splits(sources, seed=args.seed)
    for source in sources:
        if source.origin != "synthetic_fixture" or source_errors(source, allow_fixture=True):
            raise ValueError("mock pipeline requires valid synthetic fixtures exclusively")
    pairs = [generate_mock_pair(item, seed=args.seed) for item in assigned if item.split != "test"]
    plan = training_plan(pairs, allow_mock=True)
    records = sft_records(pairs, allow_mock=True)
    checks = [
        check_protection(pair.synthetic_ai_text, pair.source.text, pair.source.protected_spans)
        for pair in pairs
    ]
    # Validate everything before creating the output directory; never overwrite a prior run.
    args.output_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    write_jsonl(args.output_dir / "sources.split.jsonl", assigned)
    write_jsonl(args.output_dir / "pairs.mock.jsonl", pairs)
    for split in ("train", "dev"):
        write_jsonl(
            args.output_dir / f"{split}.sft.jsonl",
            [row for row in records if row["split"] == split],
        )
    write_json(args.output_dir / "training-plan.json", plan)
    report = {
        "purpose": "mock_smoke_only",
        "input_sha256": sha256_file(args.input),
        "seed": args.seed,
        "source_count": len(sources),
        "splits": dict(Counter(item.split for item in assigned)),
        "pair_count": len(pairs),
        "protection_passed": sum(c["mechanical_pass"] for c in checks),
        "test_sources_generated": 0,
        "model_training_executed": False,
        "network_calls": 0,
        "actual_external_spend_usd": "0",
        "semantic_quality": "not_assessed",
        "model_quality": "not_assessed",
    }
    write_json(args.output_dir / "report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "validate-sources":
            rows = read_jsonl(args.input, Source)
            failures = {
                row.source_doc_id: errors
                for row in rows
                if (
                    errors := source_errors(
                        row,
                        allow_fixture=args.allow_fixtures,
                        external_transfer=args.external_transfer,
                        allow_later_years=args.allow_later_years,
                    )
                )
            }
            duplicates = len(rows) != len({row.source_doc_id for row in rows})
            _emit({"records": len(rows), "failures": failures, "duplicate_ids": duplicates})
            return int(bool(failures) or duplicates)
        if args.command == "split":
            rows = assign_splits(read_jsonl(args.input, Source), seed=args.seed)
            write_jsonl(args.output, rows)
            _emit({"records": len(rows), "splits": dict(Counter(row.split for row in rows))})
        elif args.command == "mock-pipeline":
            _emit(_mock_pipeline(args))
        elif args.command == "estimate":
            result = estimate(CostConfig.model_validate(tomllib.loads(args.config.read_text())))
            _emit(result)
            return 0 if result["within_budget"] else 1
        elif args.command == "check-protection":
            result = check_protection(
                args.original.read_text(encoding="utf-8"),
                args.rewritten.read_text(encoding="utf-8"),
                args.span,
            )
            _emit(result)
            return 0 if result["mechanical_pass"] else 1
        elif args.command == "training-plan":
            _emit(training_plan(read_jsonl(args.pairs, Pair), allow_mock=args.allow_mock))
        elif args.command == "export-sft":
            rows = sft_records(read_jsonl(args.pairs, Pair), allow_mock=args.allow_mock)
            write_jsonl(args.output, rows)
            _emit({"records": len(rows), "purpose": "mock" if args.allow_mock else "sft"})
        elif args.command == "identity":
            _emit(inspect_rewrite(args.input.read_text(encoding="utf-8"), IdentityBaseline()))
        elif args.command == "blind-compare":
            tasks, key = blind_compare(
                read_jsonl(args.inputs, TextResult),
                read_jsonl(args.left, TextResult),
                read_jsonl(args.right, TextResult),
                seed=args.seed,
            )
            args.output_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
            write_jsonl(args.output_dir / "tasks.jsonl", tasks)
            write_json(args.output_dir / "key.private.json", key)
            _emit({"tasks": len(tasks), "key_must_be_hidden_from_raters": True})
        elif args.command == "summarize-ratings":
            _emit(
                summarize_ratings(
                    read_jsonl(args.ratings, Rating),
                    json.loads(args.key.read_text(encoding="utf-8")),
                )
            )
        elif args.command == "benchmark-validate":
            _emit(validate_cases(read_jsonl(args.dataset, DevCase)))
        elif args.command == "benchmark-prepare":
            _emit(
                prepare_bundle(
                    args.dataset,
                    args.spec,
                    args.output_dir,
                    allow_unreviewed=args.allow_unreviewed,
                    selection=args.selection,
                )
            )
        elif args.command == "benchmark-identity":
            _emit(identity_benchmark(args.dataset, args.output_dir))
        elif args.command == "benchmark-import":
            report = import_outputs(args.bundle, args.outputs, args.output_dir)
            _emit(report)
            return 0 if report["all_complete"] else 1
        elif args.command == "benchmark-tokens":
            from korean_humanizer.token_audit import audit_tokens

            report = audit_tokens(args.bundle, args.metadata_dir)
            write_json(args.output, report)
            _emit({key: value for key, value in report.items() if key != "cases"})
            return 0 if report["all_within_context"] else 1
        elif args.command == "benchmark-run-local":
            from korean_humanizer.local_runner import run_local

            report = run_local(
                args.bundle,
                args.base_url,
                output_dir=args.output_dir,
                token_audit=args.token_audit,
                execute=args.execute,
                max_seconds=args.max_seconds,
                request_timeout=args.request_timeout,
            )
            _emit(report)
            if args.execute and report["complete"] != report["requests"]:
                return 1
        return 0
    except ImportError:
        print(
            "error: install CPU benchmark extras with uv sync --frozen --extra benchmark",
            file=sys.stderr,
        )
        return 2
    except ValidationError:
        print(
            "error: invalid configuration; check required fields, types and nonnegative prices",
            file=sys.stderr,
        )
        return 2
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
