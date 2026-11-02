"""CPU archive/transfer tests with synthetic bytes; no cloud calls or GPU imports."""

import hashlib
import importlib.util
import io
import json
import tarfile
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def module(name, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def archive(files):
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w:gz") as tar:
        for name, content in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    return data.getvalue()


@pytest.mark.parametrize("fault", [None, "archive_hash", "file_hash", "traversal"])
def test_binary_retrieval_requires_archive_and_each_file_integrity(tmp_path, monkeypatch, fault):
    controller = module("run_first_sft", monkeypatch)
    files = {"run/adapter/model.safetensors": b"fixture-binary\x00\xff"}
    if fault == "traversal":
        files["../escape"] = b"never-write-outside-target"
    hashes = {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}
    if fault == "file_hash":
        hashes[next(iter(hashes))] = "0" * 64
    files["retrieval-manifest.json"] = json.dumps({"success": True, "files": hashes}).encode()
    raw = archive(files)

    class Response(io.BytesIO):
        headers = {"Content-Length": str(len(raw))}

    class Opener:
        def open(self, request, timeout):
            assert request.get_header("Authorization") == "Bearer fixture-only"
            return Response(raw)

    monkeypatch.setattr(controller.urllib.request, "build_opener", lambda *a: Opener())
    status = {"artifact_bytes": len(raw), "artifact_sha256": hashlib.sha256(raw).hexdigest()}
    if fault == "archive_hash":
        status["artifact_sha256"] = "0" * 64
    if fault:
        with pytest.raises(ValueError):
            controller.retrieve(
                "https://fixture.invalid/artifact",
                "fixture-only",
                status,
                tmp_path,
                time.time() + 30,
            )
        assert not (tmp_path / "retrieval.json").exists()
        assert not (tmp_path / "escape").exists()
    else:
        controller.retrieve(
            "https://fixture.invalid/artifact", "fixture-only", status, tmp_path, time.time() + 30
        )
        assert json.loads((tmp_path / "retrieval.json").read_text())["verified"] is True
        assert (tmp_path / "recovered/run/adapter/model.safetensors").read_bytes() == files[
            "run/adapter/model.safetensors"
        ]


def test_worker_rejects_input_traversal_before_writing(tmp_path, monkeypatch):
    worker = module("sft_gpu_worker", monkeypatch)
    path = tmp_path / "bad.tar.gz"
    path.write_bytes(archive({"okay": b"fixture", "../escape": b"bad"}))
    target = tmp_path / "target"
    target.mkdir()
    with pytest.raises(ValueError, match="unsafe"):
        worker.unpack(path, target)
    assert not list(target.iterdir())
