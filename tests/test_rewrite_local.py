import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("rewrite_local", Path("scripts/rewrite_local.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
prepare = module.prepare


def test_dry_run_needs_no_weights_or_inference_dependencies(tmp_path):
    text = tmp_path / "input.txt"
    text.write_text("회의는 오후 3시입니다.", encoding="utf-8-sig")
    result = subprocess.run(
        [sys.executable, "scripts/rewrite_local.py", "--text-file", str(text)],
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(result.stdout)
    assert report["mode"] == "dry-run"
    assert report["candidate"] == "qwen3-4b-base"
    assert report["model_downloaded"] is False


@pytest.mark.parametrize("content,limit", [(" ", 256), ("글", 0), ("글", 1025)])
def test_invalid_input_rejected_before_loading_model(tmp_path, content, limit):
    text = tmp_path / "input.txt"
    text.write_text(content)
    with pytest.raises(ValueError):
        prepare(text, max_new_tokens=limit)


def test_incompatible_adapter_rejected_before_loading_weights(tmp_path):
    text = tmp_path / "input.txt"
    text.write_text("글")
    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "Qwen/Qwen3-8B"})
    )
    with pytest.raises(ValueError, match="same"):
        prepare(text, tmp_path)
