"""Loopback-only text editor; loads one trained model and reuses it between requests."""

import argparse
import json
import secrets
import threading
import urllib.error
import urllib.request
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from korean_humanizer.long_text import MAX_CHARACTERS, rewrite_document
from korean_humanizer.model_download import ROOT, download_model


class Application:
    def __init__(self, loader=None):
        self.loader = loader or self.load_model
        self.model = None
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.state = {"status": "idle", "message": "글을 입력하고 다듬기를 눌러 주세요."}

    @staticmethod
    def load_model():
        from korean_humanizer.local_model import LocalModel

        adapter = download_model(ROOT / "artifacts/models/nikl-practical-001")
        spec = json.loads((ROOT / "configs/baselines/qwen3-4b.json").read_text())
        config = json.loads((adapter / "adapter_config.json").read_text())
        if (
            config["base_model_name_or_path"] != spec["model_id"]
            or config["revision"] != spec["revision"]
        ):
            raise ValueError("학습 모델과 기반 모델 버전이 일치하지 않습니다.")
        return LocalModel(spec, adapter, memory_mode="auto")

    def snapshot(self):
        with self.lock:
            return dict(self.state)

    def submit(self, text):
        if not isinstance(text, str) or not text.strip() or len(text) > MAX_CHARACTERS:
            raise ValueError("1~50,000자의 글을 입력해 주세요.")
        with self.lock:
            if self.state["status"] == "busy":
                return False
            self.state = {
                "status": "busy",
                "message": "모델을 준비하고 있습니다. 첫 실행은 다운로드로 오래 걸릴 수 있어요."
                if self.model is None
                else "글을 다듬고 있습니다.",
            }
        self.pool.submit(self.process, text)
        return True

    def progress(self, completed, total, retries):
        with self.lock:
            self.state = {
                "status": "busy",
                "message": f"{completed}/{total}구간 완료 · 내 컴퓨터에서 처리 중"
                + (f" · 출력이 잘린 구간을 더 나눠 처리 중 ({retries}회)" if retries else ""),
                "progress": {"completed": completed, "total": total},
            }

    def process(self, text):
        try:
            if self.model is None:
                self.model = self.loader()
            result = rewrite_document(self.model, text, self.progress)
            message = {
                "unchanged": "변경 없음: 모델이 원문을 그대로 반환했습니다.",
                "whitespace_only": "공백만 변경됐습니다. 문장 내용은 같습니다.",
                "changed": "문장이 변경됐습니다. 의미와 자연스러움을 직접 확인해 주세요.",
            }[result["change_kind"]]
            with self.lock:
                self.state = {
                    "status": "done",
                    "result": result,
                    "message": f"{result['chunks_total']}구간 처리 완료. {message}",
                }
        except Exception as error:
            if isinstance(error, ValueError):
                message = str(error)
            elif "out of memory" in str(error).lower():
                message = (
                    "메모리가 부족합니다. 다른 앱을 닫고 앱을 다시 실행하거나 "
                    "Colab을 이용해 주세요."
                )
            else:
                message = (
                    "모델을 실행하지 못했습니다. 인터넷 연결과 설치 창의 오류를 확인해 주세요. "
                    "원래 Qwen 모델로 대신 실행하지 않았습니다."
                )
            # No input/output text is logged or persisted.
            print(f"Local inference error: {type(error).__name__}", flush=True)
            with self.lock:
                self.state = {"status": "error", "message": message}


def make_server(port=8765, app=None):
    app = app or Application()
    token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send_data(self, status, body, content_type="application/json; charset=utf-8"):
            data = body.encode() if isinstance(body, str) else json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.end_headers()
            self.wfile.write(data)

        def valid_host(self):
            return self.headers.get("Host") == f"127.0.0.1:{self.server.server_port}"

        def do_GET(self):
            if not self.valid_host():
                self.send_data(403, {"error": "Local host only"})
            elif self.path == "/":
                html = (ROOT / "desktop/index.html").read_text(encoding="utf-8")
                self.send_data(
                    200, html.replace("__LOCAL_TOKEN__", token), "text/html; charset=utf-8"
                )
            elif self.path == "/api/status":
                self.send_data(200, app.snapshot())
            elif self.path == "/api/health":
                self.send_data(200, {"application": "korean-humanizer-local-v1"})
            else:
                self.send_data(404, {"error": "Not found"})

        def do_POST(self):
            origin = self.headers.get("Origin")
            expected = f"http://127.0.0.1:{self.server.server_port}"
            if (
                not self.valid_host()
                or self.headers.get("X-Local-Token") != token
                or origin not in (None, expected)
            ):
                self.send_data(403, {"error": "Local request required"})
                return
            if self.path != "/api/rewrite":
                self.send_data(404, {"error": "Not found"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 1024 * 1024:
                    raise ValueError("요청 크기가 올바르지 않습니다.")
                body = json.loads(self.rfile.read(size))
                if not isinstance(body, dict):
                    raise ValueError("요청 형식이 올바르지 않습니다.")
                if app.submit(body.get("text")):
                    self.send_data(202, {"status": "busy"})
                else:
                    self.send_data(409, {"error": "이전 글을 다듬는 중입니다."})
            except (ValueError, UnicodeError) as error:
                self.send_data(400, {"error": str(error)})

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    return server, app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    url = f"http://127.0.0.1:{args.port}"
    try:
        with urllib.request.urlopen(url + "/api/health", timeout=2) as response:
            running = json.load(response).get("application") == "korean-humanizer-local-v1"
    except (OSError, ValueError):
        running = False
    if running:
        if not args.no_browser:
            webbrowser.open(url)
        print("Korean Humanizer is already running:", url)
        return
    server, app = make_server(args.port)
    print("Korean Humanizer:", url, "— keep this window open; Ctrl+C to stop.", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        app.pool.shutdown(wait=True)


if __name__ == "__main__":
    main()
