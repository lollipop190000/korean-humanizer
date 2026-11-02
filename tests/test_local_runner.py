import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from korean_humanizer.benchmark import DevCase, import_outputs, prepare_bundle
from korean_humanizer.io import read_jsonl, sha256_file, write_json, write_jsonl
from korean_humanizer.local_runner import local_base_url, run_local

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def local_job(tmp_path):
    dataset = tmp_path / "dev.jsonl"
    cases = read_jsonl(ROOT / "eval/dev_seed.v1.jsonl", DevCase)[:3]
    write_jsonl(dataset, cases)
    bundle = tmp_path / "bundle"
    prepare_bundle(
        dataset, ROOT / "configs/baselines/humanizer.json", bundle, allow_unreviewed=True
    )
    manifest = json.loads((bundle / "manifest.json").read_text())
    audit = tmp_path / "audit.json"
    write_json(
        audit,
        {
            "bundle_manifest_sha256": sha256_file(bundle / "manifest.json"),
            "candidate": manifest["candidate"],
            "revision": manifest["revision"],
            "all_within_context": True,
        },
    )
    return bundle, audit


@pytest.mark.parametrize(
    "url",
    [
        "https://api.example.com",
        "http://example.com",
        "http://localhost@evil.test",
        "http://localhost/path",
        "http://localhost?target=external",
    ],
)
def test_hosted_endpoints_are_not_part_of_the_local_runner(url):
    with pytest.raises(ValueError, match="local HTTP"):
        local_base_url(url)


def test_default_dry_run_makes_no_request_or_output(local_job, tmp_path, monkeypatch):
    import urllib.request

    def forbidden(*args, **kwargs):
        raise AssertionError("dry run attempted network access")

    monkeypatch.setattr(urllib.request, "build_opener", forbidden)
    bundle, _ = local_job
    output = tmp_path / "run"
    report = run_local(bundle, "http://127.0.0.1:1", output_dir=output)
    assert report["mode"] == "dry_run" and report["network_calls"] == 0
    assert not output.exists()


def test_local_http_results_include_failures_and_truncation(local_job, tmp_path):
    bundle, audit = local_job
    seen = []
    followed_redirect = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((self.path, payload))
            if len(seen) == 3:
                self.send_response(302)
                self.send_header(
                    "Location", f"http://127.0.0.1:{self.server.server_port}/redirected"
                )
                self.end_headers()
                return
            response = {
                "model": "jialinyyzz/humanizer",
                "choices": [
                    {
                        "text": "테스트 서버의 고정 응답입니다.",
                        "finish_reason": "stop" if len(seen) == 1 else "length",
                    }
                ],
            }
            data = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

        def do_GET(self):
            followed_redirect.append(self.path)
            self.send_error(500)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        report = run_local(
            bundle,
            f"http://127.0.0.1:{server.server_port}",
            output_dir=tmp_path / "run",
            token_audit=audit,
            execute=True,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert report["network_calls"] == 3
    assert (report["complete"], report["truncated"], report["errors"]) == (1, 1, 1)
    assert seen[0][0] == "/v1/completions"
    assert not followed_redirect
    assert "stop" not in seen[0][1]
    imported = import_outputs(bundle, tmp_path / "run/outputs.jsonl", tmp_path / "report")
    assert not imported["all_complete"]


def test_execution_requires_matching_context_audit(local_job, tmp_path):
    bundle, audit = local_job
    with pytest.raises(ValueError, match="matching token audit"):
        run_local(bundle, "http://127.0.0.1", output_dir=tmp_path / "run", execute=True)
    bad = json.loads(audit.read_text()) | {"all_within_context": False}
    audit.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="context lengths"):
        run_local(
            bundle, "http://127.0.0.1", output_dir=tmp_path / "run", token_audit=audit, execute=True
        )


def test_deadline_records_unattempted_items_as_failures(local_job, tmp_path, monkeypatch):
    import korean_humanizer.local_runner as runner

    bundle, audit = local_job
    clock = iter([0.0] + [10.0] * 20)
    monkeypatch.setattr(runner.time, "monotonic", lambda: next(clock))
    report = run_local(
        bundle,
        "http://127.0.0.1:1",
        output_dir=tmp_path / "run",
        token_audit=audit,
        execute=True,
        max_seconds=1,
        request_timeout=1,
    )
    assert report["network_calls"] == 0
    assert report["errors"] == report["requests"] == 3
