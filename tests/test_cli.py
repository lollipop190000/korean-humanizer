import json
import subprocess
import sys
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures" / "sources.jsonl"


def run(*args):
    return subprocess.run(
        [sys.executable, "-m", "korean_humanizer.cli", *map(str, args)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_end_to_end_mock_run_is_repeatable_and_test_sources_are_locked(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    for output in (first, second):
        result = run("mock-pipeline", FIXTURES, "--output-dir", output)
        assert result.returncode == 0, result.stderr
        report = json.loads(result.stdout)
        assert set(report["splits"]) == {"train", "dev", "test"}
        assert report["pair_count"] == report["splits"]["train"] + report["splits"]["dev"]
        assert report["protection_passed"] == report["pair_count"]
        assert report["test_sources_generated"] == 0
        assert not report["model_training_executed"]
        assert report["actual_external_spend_usd"] == "0"
        pairs = [
            json.loads(line) for line in (output / "pairs.mock.jsonl").read_text().splitlines()
        ]
        assert all(pair["split"] != "test" for pair in pairs)
    assert (first / "report.json").read_bytes() == (second / "report.json").read_bytes()
    assert run("mock-pipeline", FIXTURES, "--output-dir", first).returncode != 0


def test_invalid_and_non_fixture_data_fail_without_exposing_text(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"text": "PRIVATE_CONTENT"}\n')
    result = run("validate-sources", path)
    assert result.returncode == 2
    assert "PRIVATE_CONTENT" not in result.stderr
    assert run("validate-sources", FIXTURES).returncode == 1
    assert run("validate-sources", FIXTURES, "--allow-fixtures").returncode == 0


def test_budget_overrun_is_nonzero_and_paid_execution_unavailable(tmp_path):
    config = tmp_path / "cost.toml"
    config.write_text(
        'pricing_status="test"\nbudget_usd="0"\ngpu_hourly_usd="1"\ngpu_max_hours="1"'
    )
    result = run("estimate", config)
    assert result.returncode == 1
    assert not json.loads(result.stdout)["execution_enabled"]
    assert (
        run("mock-pipeline", FIXTURES, "--output-dir", tmp_path / "out", "--execute").returncode
        != 0
    )
