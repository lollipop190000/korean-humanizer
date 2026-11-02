import hashlib
import json
from pathlib import Path

import pytest

from korean_humanizer.benchmark import prepare_bundle
from korean_humanizer.io import sha256_file, write_json

ROOT = Path(__file__).resolve().parents[1]


def test_template_rendering_does_not_leak_or_enable_thinking():
    pytest.importorskip("jinja2")
    from korean_humanizer.token_audit import render_chat

    template = "{{ messages[0].content }}|{{ enable_thinking }}|{{ add_generation_prompt }}"
    body = {
        "messages": [{"role": "user", "content": "한국어"}],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    assert render_chat(template, body, {}) == "한국어|False|True"


def test_token_count_disables_saved_truncation_and_padding(tmp_path):
    tokenizers = pytest.importorskip("tokenizers")
    from korean_humanizer.token_audit import audit_tokens

    bundle = tmp_path / "bundle"
    spec_path = ROOT / "configs/baselines/humanizer.json"
    prepare_bundle(ROOT / "eval/dev_seed.v1.jsonl", spec_path, bundle, allow_unreviewed=True)
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer.enable_truncation(max_length=2)
    tokenizer.enable_padding(length=9000)
    tokenizer.save(str(metadata / "tokenizer.json"))
    (metadata / "tokenizer_config.json").write_text("{}")
    spec = json.loads(spec_path.read_text())
    write_json(
        metadata / "metadata.json",
        {
            "model_id": spec["model_id"],
            "revision": spec["revision"],
            "files": {
                name: {"sha256": sha256_file(metadata / name)}
                for name in ("tokenizer.json", "tokenizer_config.json")
            },
        },
    )
    report = audit_tokens(bundle, metadata)
    assert 2 < report["prompt_tokens_max"] < 9000
    assert report["all_within_context"]
    assert not report["model_weights_loaded"]
    (metadata / "tokenizer.json").write_text("{}")
    with pytest.raises(ValueError, match="integrity"):
        audit_tokens(bundle, metadata)


def test_download_verifies_upstream_git_blob_and_lfs_hashes():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "fetch_tokenizers", ROOT / "scripts/fetch_tokenizers.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = b"test tokenizer"
    blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
    module.verify_object(data, {"blobId": blob})
    module.verify_object(data, {"lfs": {"sha256": hashlib.sha256(data).hexdigest()}})
    with pytest.raises(ValueError, match="upstream"):
        module.verify_object(data + b"changed", {"blobId": blob})
