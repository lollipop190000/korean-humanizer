import hashlib
import io
import json
from urllib.error import HTTPError

import pytest

from korean_humanizer.model_download import download_model, read_manifest


def manifest(tmp_path):
    content = {"adapter_config.json": b"{}", "adapter_model.safetensors": b"fixture-not-real"}
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "model_id": "nikl-practical-001",
                "release_tag": "model-nikl-practical-001",
                "files": {
                    name: {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                    for name, data in content.items()
                },
            }
        )
    )
    return path, content


def test_download_checks_all_bytes_then_reuses_cache(tmp_path, monkeypatch):
    path, content = manifest(tmp_path)
    urls = []

    def open_url(request, **kwargs):
        urls.append(request.full_url)
        return io.BytesIO(content[request.full_url.rsplit("/", 1)[1]])

    monkeypatch.setattr("urllib.request.urlopen", open_url)
    target = download_model(tmp_path / "model", path)
    assert len(urls) == 2
    assert (target / "adapter_model.safetensors").read_bytes() == content[
        "adapter_model.safetensors"
    ]
    download_model(target, path)
    assert len(urls) == 2


@pytest.mark.parametrize("bad", [b"x", b"garbage-oversize", b"[]"])
def test_corrupt_download_is_never_installed(tmp_path, monkeypatch, bad):
    path, _ = manifest(tmp_path)
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: io.BytesIO(bad))
    target = tmp_path / "model"
    with pytest.raises(ValueError, match="mismatch"):
        download_model(target, path)
    assert not list(target.iterdir())


def test_existing_corrupt_file_is_not_silently_overwritten(tmp_path):
    path, _ = manifest(tmp_path)
    target = tmp_path / "model"
    target.mkdir()
    (target / "adapter_config.json").write_bytes(b"[]")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        download_model(target, path)
    assert (target / "adapter_config.json").read_bytes() == b"[]"


def test_missing_release_fails_without_substituting_base(tmp_path, monkeypatch):
    path, _ = manifest(tmp_path)

    def missing(*args, **kwargs):
        raise HTTPError("https://example.test", 404, "Not Found", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", missing)
    with pytest.raises(RuntimeError, match="base model will not be substituted"):
        download_model(tmp_path / "model", path)


def test_manifest_rejects_path_traversal(tmp_path):
    path, _ = manifest(tmp_path)
    data = json.loads(path.read_text())
    data["files"]["../outside"] = data["files"]["adapter_config.json"]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Unsafe"):
        read_manifest(path)
