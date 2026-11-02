"""Finite first SFT controller. Default is offline; execute only under the job's approval."""

import argparse
import base64
import hashlib
import json
import secrets
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path

import run_gpu_smoke as cloud

from korean_humanizer.io import sha256_file
from korean_humanizer.sft import load_sft

ROOT = Path(__file__).resolve().parents[1]
READY = ROOT / "artifacts/sft-first-research-001-ready"
MANIFEST = "42d4b785732a2f51a46f8cb9f2a404dd68cad1c7e9827f815163d79d785b4fd4"
EVALUATION = "d760d429569b92b6213d3637ac0012e34fff4cafde102ef839d6ef3937f38463"
MAX_ARTIFACT = 2 * 1024 * 1024 * 1024


def input_archive(output):
    manifest, config, _ = load_sft(READY / "bundle")
    if (
        sha256_file(READY / "bundle/manifest.json") != MANIFEST
        or sha256_file(READY / "evaluation.requests.jsonl") != EVALUATION
        or config.data_policy != "pending_research"
        or config.max_steps != 16
        or manifest["counts"] != {"train": 9, "dev": 1}
    ):
        raise ValueError("job differs from the approved first SFT scope")
    files = []
    for path in sorted((ROOT / "src/korean_humanizer").glob("*.py")):
        files.append((path, str(path.relative_to(ROOT))))
    for name in ["train_sft.py", "compare_sft.py"]:
        files.append((ROOT / "scripts" / name, "scripts/" + name))
    files.append((ROOT / "requirements/sft-gpu.txt", "requirements/sft-gpu.txt"))
    for name in ["job.json", "evaluation.requests.jsonl"]:
        files.append((READY / name, "ready/" + name))
    for path in sorted((READY / "bundle").iterdir()):
        files.append((path, "ready/bundle/" + path.name))
    with tarfile.open(output, "w:gz", compresslevel=1) as tar:
        for path, name in files:
            tar.add(path, arcname=name, recursive=False)
    output.chmod(0o600)
    return {name: sha256_file(path) for path, name in files}


def upload(url, token, path):
    req = urllib.request.Request(
        url,
        method="POST",
        data=path.read_bytes(),
        headers={
            "Authorization": "Bearer " + token,
            "User-Agent": cloud.USER_AGENT,
            "Content-Type": "application/gzip",
        },
    )
    with urllib.request.build_opener(cloud.NoRedirects()).open(req, timeout=120) as result:
        if result.status != 202:
            raise RuntimeError("worker did not accept input")


def retrieve(url, token, status, directory, deadline):
    size = status["artifact_bytes"]
    if not 0 < size <= MAX_ARTIFACT:
        raise ValueError("artifact size outside bounds")
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": "Bearer " + token,
            "User-Agent": cloud.USER_AGENT,
        },
    )
    archive = directory / "result.tar.gz"
    digest = hashlib.sha256()
    count = 0
    with urllib.request.build_opener(cloud.NoRedirects()).open(req, timeout=45) as response:
        if int(response.headers.get("Content-Length", "-1")) != size:
            raise ValueError("artifact Content-Length differs")
        with archive.open("xb") as target:
            archive.chmod(0o600)
            while block := response.read(1024 * 1024):
                count += len(block)
                if count > size or time.time() >= deadline:
                    raise ValueError("retrieval size/time limit exceeded")
                target.write(block)
                digest.update(block)
    if count != size or digest.hexdigest() != status["artifact_sha256"]:
        raise ValueError("artifact checksum mismatch")
    target = directory / "recovered"
    target.mkdir(mode=0o700)
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        if len(members) > 2000 or sum(m.size for m in members) > 5 * 1024**3:
            raise ValueError("result archive exceeds scope")
        names = [m.name for m in members]
        if len(names) != len(set(names)):
            raise ValueError("duplicate result archive members")
        for member in members:
            if (
                not member.isfile()
                or Path(member.name).is_absolute()
                or ".." in Path(member.name).parts
            ):
                raise ValueError("unsafe result archive")
        for member in members:
            path = target / member.name
            path.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, path.open("xb") as dest:
                shutil.copyfileobj(source, dest)
    receipt = json.loads((target / "retrieval-manifest.json").read_text())
    if set(receipt["files"]) != set(names) - {"retrieval-manifest.json"}:
        raise ValueError("result manifest file coverage differs")
    for name, expected in receipt["files"].items():
        if sha256_file(target / name) != expected:
            raise ValueError("recovered file checksum mismatch")
    cloud.write_json(
        directory / "retrieval.json",
        {
            "verified": True,
            "archive_bytes": size,
            "archive_sha256": digest.hexdigest(),
            "files": len(receipt["files"]),
            "worker_success": receipt["success"],
        },
    )


def execute(directory):
    directory.mkdir(parents=True, exist_ok=False, mode=0o700)
    archive = directory / "input.tar.gz"
    files = input_archive(archive)
    job_name = "kh-sft-first-research-001-" + secrets.token_hex(4)
    token = secrets.token_urlsafe(32)
    worker = (ROOT / "scripts/sft_gpu_worker.py").read_bytes()
    bootstrap = (
        "import base64,zlib;exec(zlib.decompress(base64.b64decode('"
        + base64.b64encode(zlib.compress(worker)).decode()
        + "')))"
    )
    catalog = cloud.api(
        "GET", "/v2/catalog/gpus?include=AVAILABILITY&product=POD&cloud=SECURE&count=1"
    )
    gpu = next(g for g in catalog["gpus"] if g["id"] == cloud.GPU)
    if gpu["availability"] == "NONE" or not 0 < gpu["price"]["secure"] <= 1.5:
        raise RuntimeError("approved GPU unavailable or above price cap")
    cloud.write_json(directory / "catalog.json", gpu)
    cloud.write_json(
        directory / "plan.json",
        {
            "job_id": "sft-first-research-001",
            "budget_usd": 3,
            "approved": True,
            "job_name": job_name,
            "image": cloud.IMAGE,
            "disk_gb": 120,
            "persistent_volumes": 0,
            "max_seconds": 3300,
            "input_sha256": sha256_file(archive),
            "input_files": files,
            "worker_sha256": hashlib.sha256(worker).hexdigest(),
            "manifest_sha256": MANIFEST,
            "evaluation_sha256": EVALUATION,
        },
    )
    body = {
        "name": job_name,
        "image": cloud.IMAGE,
        "cloud": "SECURE",
        "disk": 120,
        "gpu": {
            "id": cloud.GPU,
            "count": 1,
            "minRamPerGpu": 32,
            "minVcpuCountPerGpu": 4,
            "minCudaVersion": "12.4",
        },
        "entrypoint": ["python", "-u", "-c"],
        "cmd": [bootstrap],
        "ports": ["8080/http"],
        "env": {
            "KH_RESULT_TOKEN": token,
            "KH_INPUT_SHA256": sha256_file(archive),
            "KH_MANIFEST_SHA256": MANIFEST,
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HOME": "/tmp/hf-cache",
        },
    }
    pod_id = None
    start = time.time()
    deadline = start + 3300
    try:
        pod = cloud.api("POST", "/v2/pods", body)
        pod_id = pod["id"]
        cloud.write_json(
            directory / "deployment.json",
            {
                **{k: pod.get(k) for k in ["id", "name", "cost", "createdAt", "dataCenterId"]},
                "deadline_epoch": deadline,
                "creation_started_epoch": start,
            },
        )
        with (directory / "watchdog.log").open("a") as log:
            subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "--watchdog", str(directory)],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        if not isinstance(pod.get("cost"), (int, float)) or not 0 < pod["cost"] <= 1.5:
            raise RuntimeError("actual Pod price above approved cap")
        cloud.log("created", pod_id=pod_id, hourly_usd=pod["cost"])
        base = "https://" + pod_id + "-8080.proxy.runpod.net"
        uploaded = False
        last = None
        while time.time() < deadline:
            try:
                status = cloud.request("GET", base + "/status", token, timeout=20)
            except (urllib.error.URLError, OSError, ValueError, TimeoutError) as exc:
                if time.time() - start > 900 and last is None:
                    raise RuntimeError("worker readiness exceeded 15 minutes") from exc
                time.sleep(10)
                continue
            cloud.write_json(directory / "worker-status.json", status)
            if status["status"] != last:
                cloud.log(
                    "progress", status=status["status"], elapsed_seconds=round(time.time() - start)
                )
                last = status["status"]
            if status["status"] == "awaiting_input" and not uploaded:
                uploaded = True
                upload(base + "/input", token, archive)
            elif status["status"] in {"done", "failed"}:
                retrieve(base + "/artifact", token, status, directory, deadline)
                if status["status"] != "done":
                    raise RuntimeError("worker failed; recovered partial artifacts")
                break
            elif status["status"] == "upload_failed":
                raise RuntimeError("worker rejected input; no automatic retry")
            time.sleep(10)
        else:
            raise RuntimeError("job reached its hard controller deadline")
    finally:
        if pod_id is None:
            listing = cloud.api("GET", "/v2/pods")
            pods = listing["pods"] if isinstance(listing, dict) else listing
            matches = [p for p in pods if p.get("name") == job_name]
            if len(matches) > 1:
                raise RuntimeError("ambiguous Pod creation requires inspection")
            if matches:
                pod_id = matches[0]["id"]
        if pod_id:
            for attempt in range(6):
                try:
                    if cloud.delete_and_confirm(pod_id):
                        cloud.write_json(
                            directory / "termination.json",
                            {
                                "confirmed": True,
                                "actor": "controller",
                                "pod_id": pod_id,
                                "elapsed_seconds": time.time() - start,
                                "epoch": time.time(),
                            },
                        )
                        cloud.log("terminated", pod_id=pod_id)
                        break
                except Exception as exc:
                    cloud.log("delete_error", error=type(exc).__name__, attempt=attempt + 1)
                time.sleep(5)
            else:
                raise RuntimeError("deletion not confirmed; watchdog remains active")
    if pod_id:
        cloud.write_json(
            directory / "billing.json",
            cloud.api("GET", "/v2/billing/pods?bucketSize=hour&lastN=2&podId=" + pod_id),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--watchdog", type=Path)
    args = parser.parse_args()
    if args.watchdog:
        cloud.watchdog(args.watchdog)
    elif args.execute:
        if not args.output_dir:
            parser.error("--output-dir is required for execution")
        execute(args.output_dir.resolve())
    else:
        manifest, config, _ = load_sft(READY / "bundle")
        if sha256_file(READY / "bundle/manifest.json") != MANIFEST:
            raise ValueError("unexpected SFT manifest")
        print(
            json.dumps(
                {
                    "training_executed": False,
                    "network_calls": 0,
                    "config": config.model_dump(),
                    "counts": manifest["counts"],
                }
            )
        )


if __name__ == "__main__":
    main()
