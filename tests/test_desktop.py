import json
import re
import threading
import time
import urllib.error
import urllib.request

import pytest

from korean_humanizer.desktop import Application, make_server


class FakeModel:
    def prompt_token_count(self, text):
        return 100 + len(text)

    def rewrite(self, text, max_new_tokens):
        return {
            "output": "합성 시험 결과: " + text,
            "finish_reason": "stop",
            "checks": {"mechanical_pass": True},
            "device": "cpu",
            "quantization": "none",
        }


@pytest.fixture
def local_app():
    loads = []

    def loader():
        loads.append(True)
        return FakeModel()

    app = Application(loader)
    server, _ = make_server(0, app)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    with urllib.request.urlopen(url) as r:
        html = r.read().decode()
    token = re.search(r'const token="([^"]+)"', html)[1]
    yield url, token, app, loads
    server.shutdown()
    server.server_close()
    app.pool.shutdown()
    thread.join()


def request(url, token, text, origin=None):
    headers = {"Content-Type": "application/json", "X-Local-Token": token}
    if origin:
        headers["Origin"] = origin
    req = urllib.request.Request(url + "/api/rewrite", json.dumps({"text": text}).encode(), headers)
    with urllib.request.urlopen(req) as response:
        return response.status


def finished(app):
    deadline = time.monotonic() + 3
    while app.snapshot()["status"] == "busy":
        assert time.monotonic() < deadline
        time.sleep(0.01)
    return app.snapshot()


def test_paste_rewrite_reuse_model_without_saving_text(local_app):
    url, token, app, loads = local_app
    for text in ["첫 시험 글", "다음 시험 글"]:
        assert request(url, token, text, url) == 202
        result = finished(app)
        assert result["result"]["output"].endswith(text)
    assert len(loads) == 1


@pytest.mark.parametrize("token,origin", [("bad", None), (None, "https://unrelated.test")])
def test_other_pages_cannot_trigger_local_inference(local_app, token, origin):
    url, real_token, app, loads = local_app
    with pytest.raises(urllib.error.HTTPError) as error:
        request(url, token or real_token, "시험", origin)
    assert error.value.code == 403
    assert not loads


def test_rebinding_host_is_rejected(local_app):
    url, _, _, _ = local_app
    req = urllib.request.Request(url, headers={"Host": "unrelated.test"})
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(req)
    assert error.value.code == 403


def test_only_one_generation_runs_at_a_time():
    started, release = threading.Event(), threading.Event()

    def loader():
        started.set()
        release.wait(3)
        return FakeModel()

    app = Application(loader)
    assert app.submit("첫 글")
    assert started.wait(2)
    assert not app.submit("두 번째 글")
    release.set()
    assert finished(app)["status"] == "done"
    app.pool.shutdown()


def test_failed_model_never_returns_base_model_result():
    def loader():
        raise RuntimeError("missing trained weights")

    app = Application(loader)
    app.submit("시험")
    result = finished(app)
    assert result["status"] == "error"
    assert "result" not in result
    app.pool.shutdown()


def test_long_korean_json_request_and_validation(local_app):
    url, token, app, loads = local_app
    text = "한글 문장. " * 3000
    assert len(json.dumps({"text": text}).encode()) > 65536
    assert request(url, token, text) == 202
    state = finished(app)
    assert state["status"] == "done"
    assert state["result"]["chunks_total"] > 1
    assert len(loads) == 1
    with pytest.raises(urllib.error.HTTPError) as error:
        request(url, token, "가" * 50001)
    assert error.value.code == 400


def test_identity_is_visible_and_failed_document_has_no_result():
    from test_long_text import EchoModel

    app = Application(EchoModel)
    app.submit("  원래 글\n\n다음 문단  ")
    state = finished(app)
    assert state["status"] == "done"
    assert "변경 없음" in state["message"]
    assert state["result"]["output"] == "  원래 글\n\n다음 문단  "
    app.model.rewrite = lambda *args, **kwargs: {"finish_reason": "stop", "output": ""}
    app.submit("실패 입력")
    state = finished(app)
    assert state["status"] == "error"
    assert "전체 미완료" in state["message"]
    assert "result" not in state
    app.pool.shutdown()
