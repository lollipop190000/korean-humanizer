"""Single-upload authenticated SFT worker. Never receives the RunPod account key."""

import hashlib
import hmac
import json
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path("/tmp/kh-sft")
STATE = {"status": "awaiting_input", "input_upload_started": False}
LOCK = threading.Lock()


def update(**values):
    with LOCK:
        STATE.update(values)


def unpack(archive, target):
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        if len(members) > 2000 or sum(m.size for m in members) > 64 * 1024 * 1024:
            raise ValueError("input archive exceeds scope")
        for member in members:
            if (
                not member.isfile()
                or Path(member.name).is_absolute()
                or ".." in Path(member.name).parts
            ):
                raise ValueError("unsafe input archive member")
        for member in members:
            dest = target / member.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, dest.open("xb") as output:
                shutil.copyfileobj(source, output)


def run():
    success = False
    try:
        project = ROOT / "project"
        unpack(ROOT / "input.tar.gz", project)
        ready = project / "ready"
        manifest_hash = os.environ["KH_MANIFEST_SHA256"]
        if (
            hashlib.sha256((ready / "bundle/manifest.json").read_bytes()).hexdigest()
            != manifest_hash
        ):
            raise ValueError("input manifest mismatch")
        logs = project / "logs"
        logs.mkdir()

        def command(name, args, timeout):
            update(status=name)
            env = dict(os.environ, PYTHONPATH=str(project / "src"))
            env.pop("KH_RESULT_TOKEN", None)
            with (logs / (name + ".log")).open("w") as log:
                subprocess.run(
                    args,
                    cwd=project,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=timeout,
                    check=True,
                )

        command(
            "installing",
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "-r",
                "requirements/sft-gpu.txt",
                "pydantic==2.12.3",
                "jinja2==3.1.6",
            ],
            480,
        )
        common = [
            sys.executable,
            "scripts/train_sft.py",
            "--bundle",
            "ready/bundle",
            "--output-dir",
            "run",
            "--manifest-sha256",
            manifest_hash,
            "--execute",
        ]
        kind = os.environ.get("KH_JOB_KIND", "sft-first-research-001")
        if kind in {"scale-001-drafts", "scale-001-judge"}:
            duration = int(os.environ["KH_PHASE_MAX_SECONDS"])
            if not 600 <= duration <= 40 * 3600:
                raise ValueError("phase duration outside scale scope")
            command(
                "batch_generating",
                [
                    sys.executable,
                    "scripts/generate_scale_batch.py",
                    "--bundle",
                    "ready/bundle",
                    "--output-dir",
                    "run",
                    "--approval",
                    "ready/approval.json",
                    "--max-seconds",
                    str(max(1, duration - 480)),
                    "--execute",
                ],
                duration,
            )
        elif kind == "scale-001-train":
            duration = int(os.environ["KH_PHASE_MAX_SECONDS"])
            if not 1800 <= duration <= 40 * 3600:
                raise ValueError("scale training phase duration outside scope")
            if (
                hashlib.sha256((ready / "manifest.json").read_bytes()).hexdigest()
                != os.environ["KH_WRAPPER_SHA256"]
            ):
                raise ValueError("scale training wrapper differs")
            command(
                "verifying_scale_recipe",
                [sys.executable, "scripts/scale_train_bundle.py", "--verify", "ready"],
                600,
            )
            command("training_scale_1000", common, max(1, (duration - 600) * 2 // 3))
            command(
                "comparing",
                [
                    sys.executable,
                    "scripts/compare_sft.py",
                    "--ready",
                    "ready",
                    "--run",
                    "run",
                    "--output-dir",
                    "comparison",
                ],
                max(1, (duration - 600) // 3),
            )
        elif kind == "sft-first-research-001":
            command("training_to_8", common + ["--stop-after-step", "8"], 900)
            command("resuming_to_16", common + ["--resume-checkpoint", "run/checkpoint-8"], 600)
            command(
                "comparing",
                [
                    sys.executable,
                    "scripts/compare_sft.py",
                    "--ready",
                    "ready",
                    "--run",
                    "run",
                    "--output-dir",
                    "comparison",
                ],
                700,
            )
        else:
            raise ValueError("unsupported worker phase")
        success = True
    except Exception:
        update(error=traceback.format_exc()[-5000:])
    finally:
        update(status="packaging")
        paths = []
        project = ROOT / "project"
        for name in ["run", "logs", "comparison"]:
            if (project / name).exists():
                paths.extend(p for p in (project / name).rglob("*") if p.is_file())
        manifest = {
            "success": success,
            "files": {
                str(p.relative_to(project)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(paths)
            },
        }
        (project / "retrieval-manifest.json").write_text(json.dumps(manifest, indent=2))
        paths.append(project / "retrieval-manifest.json")
        archive = ROOT / "result.tar.gz"
        with tarfile.open(archive, "w:gz", compresslevel=1) as tar:
            for path in sorted(paths):
                tar.add(path, arcname=str(path.relative_to(project)), recursive=False)
        sha = hashlib.file_digest(archive.open("rb"), "sha256").hexdigest()
        update(
            status="done" if success else "failed",
            artifact_sha256=sha,
            artifact_bytes=archive.stat().st_size,
        )


class Handler(BaseHTTPRequestHandler):
    def authorized(self):
        if not hmac.compare_digest(
            self.headers.get("Authorization", ""), "Bearer " + os.environ["KH_RESULT_TOKEN"]
        ):
            self.send_error(401)
            return False
        return True

    def do_GET(self):
        if not self.authorized():
            return
        path = self.path.split("?", 1)[0]
        if path == "/status":
            with LOCK:
                body = json.dumps(STATE).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        elif path == "/artifact" and STATE.get("artifact_sha256"):
            path = ROOT / "result.tar.gz"
            self.send_response(200)
            self.send_header("Content-Length", str(path.stat().st_size))
            self.send_header("Content-Type", "application/gzip")
            self.end_headers()
            with path.open("rb") as source:
                shutil.copyfileobj(source, self.wfile, length=1024 * 1024)
        else:
            self.send_error(404)

    def do_POST(self):
        if not self.authorized():
            return
        with LOCK:
            if self.path != "/input" or STATE["status"] != "awaiting_input":
                self.send_error(409)
                return
            STATE["status"] = "receiving"
            STATE["input_upload_started"] = True
        try:
            count = int(self.headers.get("Content-Length", "0"))
            if not 0 < count <= 32 * 1024 * 1024:
                raise ValueError("input size outside bounds")
            self.connection.settimeout(90)
            ROOT.mkdir(parents=True, exist_ok=True)
            path = ROOT / "input.tar.gz"
            digest = hashlib.sha256()
            with path.open("xb") as target:
                remaining = count
                while remaining:
                    block = self.rfile.read(min(1024 * 1024, remaining))
                    if not block:
                        raise ValueError("incomplete upload")
                    digest.update(block)
                    target.write(block)
                    remaining -= len(block)
            if digest.hexdigest() != os.environ["KH_INPUT_SHA256"]:
                raise ValueError("input checksum mismatch")
            update(status="uploaded")
            threading.Thread(target=run, daemon=True).start()
            self.send_response(202)
            self.send_header("Content-Length", "0")
            self.end_headers()
        except Exception:
            update(status="upload_failed")
            self.send_error(400)

    def log_message(self, *_args):
        pass


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", 8080), Handler).serve_forever()
