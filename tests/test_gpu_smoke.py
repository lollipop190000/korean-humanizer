import importlib.util
import json
import urllib.error
from pathlib import Path

import pytest


def load_script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("failure", ["create_timeout", "price_changed", "none"])
def test_created_pod_is_deleted_even_on_ambiguous_create(tmp_path, monkeypatch, failure):
    runner = load_script("run_gpu_smoke")
    calls, created = [], {}
    monkeypatch.setenv("RUNPOD_API_KEY", "account-secret-must-not-reach-worker")

    def fake_api(method, path, body=None):
        calls.append((method, path))
        if path.startswith("/v2/catalog"):
            return {"gpus": [{"id": runner.GPU, "availability": "LOW", "price": {"secure": 0.59}}]}
        if method == "POST":
            assert "account-secret-must-not-reach-worker" not in json.dumps(body)
            created.update(body)
            if failure == "create_timeout":
                raise TimeoutError("ambiguous transport failure")
            return {"id": "test-pod", "cost": 2 if failure == "price_changed" else 0.59}
        if method == "GET" and path == "/v2/pods":
            return {
                "pods": [
                    {"id": "test-pod", "name": created["name"]},
                    {"id": "unrelated", "name": "user-existing-pod"},
                ]
            }
        if method == "DELETE":
            assert path == "/v2/pods/test-pod"
            return None
        if path == "/v2/pods/test-pod":
            raise urllib.error.HTTPError(path, 404, "not found", {}, None)
        if path.startswith("/v2/billing"):
            return {"metadata": {"totals": {"totalAmount": 0.01}}}
        raise AssertionError((method, path))

    monkeypatch.setattr(runner, "api", fake_api)
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: None)
    monkeypatch.setattr(runner, "request", lambda *a, **k: {"status": "done", "models": {}})
    output = tmp_path / "run"
    if failure == "none":
        runner.deploy(output, {"jobs": []})
    else:
        with pytest.raises((TimeoutError, RuntimeError)):
            runner.deploy(output, {"jobs": []})
    assert calls.count(("POST", "/v2/pods")) == 1
    assert ("DELETE", "/v2/pods/test-pod") in calls
    assert json.loads((output / "termination.json").read_text())["confirmed"] is True


def test_worker_status_requires_per_job_token(monkeypatch):
    import threading
    import urllib.request
    from http.server import ThreadingHTTPServer

    worker = load_script("gpu_smoke_worker")
    monkeypatch.setenv("KH_RESULT_TOKEN", "random-job-token")
    server = ThreadingHTTPServer(("127.0.0.1", 0), worker.StatusHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    url = f"http://127.0.0.1:{server.server_port}/status"
    try:
        for token in ["", "wrong"]:
            with pytest.raises(urllib.error.HTTPError) as exc:
                opener.open(urllib.request.Request(url, headers={"Authorization": token}))
            assert exc.value.code == 401
        with opener.open(
            urllib.request.Request(url, headers={"Authorization": "Bearer random-job-token"})
        ) as response:
            assert json.load(response)["status"] == "booting"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
