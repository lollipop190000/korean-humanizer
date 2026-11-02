"""Deploy a separately authorized baseline or prompt-study job; defaults to dry-run.

An execution flag or manifest hash is not user authorization. Requires the CPU environment.
Never writes the RunPod account key to disk
or forwards it to the Pod. A separate local watchdog terminates only this job's Pod.
"""

import argparse
import base64
import hashlib
import json
import os
import secrets
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
from pathlib import Path

from korean_humanizer.benchmark import load_bundle
from korean_humanizer.draft_pilot import load_drafts
from korean_humanizer.io import sha256_file
from korean_humanizer.prompt_study import load_study

API = "https://api.runpod.io"
IMAGE = "pytorch/pytorch@sha256:77f17f843507062875ce8be2a6f76aa6aa3df7f9ef1e31d9d7432f4b0f563dee"
GPU = "NVIDIA A40"
MAX_SECONDS = 3300  # Leave five minutes for deletion before the approved one-hour limit.
USER_AGENT = "korean-humanizer/0.1 (RunPod REST client)"


class NoRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.URLError("redirect rejected")


def request(method, url, token, body=None, timeout=30):
    req = urllib.request.Request(
        url,
        method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={
            "Authorization": "Bearer " + token,
            "User-Agent": USER_AGENT,
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    with urllib.request.build_opener(NoRedirects()).open(req, timeout=timeout) as response:
        raw = response.read(8 * 1024 * 1024)
        return json.loads(raw) if raw else None


def api(method, path, body=None):
    return request(method, API + path, os.environ["RUNPOD_API_KEY"], body)


def write_json(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf8") as handle:
        tmp.chmod(0o600)
        json.dump(data, handle, ensure_ascii=False, indent=2)
    tmp.replace(path)


def log(event, **values):
    print(json.dumps({"event": event, **values}, ensure_ascii=False), flush=True)


def delete_and_confirm(pod_id):
    path = "/v2/pods/" + urllib.parse.quote(pod_id, safe="")
    try:
        api("DELETE", path)
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
    try:
        api("GET", path)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return True
        raise
    return False


def watchdog(directory):
    state = json.loads((directory / "deployment.json").read_text())
    while time.time() < state["deadline_epoch"]:
        if (directory / "termination.json").exists():
            return
        time.sleep(min(15, max(0, state["deadline_epoch"] - time.time())))
    while not (directory / "termination.json").exists():
        try:
            if delete_and_confirm(state["id"]):
                write_json(
                    directory / "termination.json",
                    {
                        "confirmed": True,
                        "actor": "watchdog",
                        "epoch": time.time(),
                        "pod_id": state["id"],
                    },
                )
                return
        except Exception as exc:
            log("watchdog_delete_error", error=type(exc).__name__)
        time.sleep(15)


def job_payload(bundle_paths):
    jobs = []
    for path in bundle_paths:
        _, _, requests, spec = load_bundle(path)
        audit = json.loads((path / "tokens.json").read_text())
        if (
            spec.candidate not in {"qwen3-4b", "qwen3-8b"}
            or spec.protocol != "chat"
            or spec.enable_thinking
            or len(requests) != 12
            or spec.max_output_tokens != 2048
            or audit["bundle_manifest_sha256"] != sha256_file(path / "manifest.json")
            or not audit["all_within_context"]
            or audit["revision"] != spec.revision
        ):
            raise ValueError("bundle outside approved smoke scope")
        jobs.append(
            {
                "spec": spec.model_dump(),
                "requests": requests,
                "audit": audit,
                "manifest_sha256": sha256_file(path / "manifest.json"),
            }
        )
    if [j["spec"]["candidate"] for j in jobs] != ["qwen3-4b", "qwen3-8b"]:
        raise ValueError("expected 4B then 8B exactly once")
    if [r["case_id"] for r in jobs[0]["requests"]] != [r["case_id"] for r in jobs[1]["requests"]]:
        raise ValueError("candidate input sets differ")
    return {"job_id": "baseline-smoke-001", "jobs": jobs}


def deploy(directory, payload):
    directory.mkdir(parents=True, exist_ok=False, mode=0o700)
    job_name = "kh-" + payload.get("job_id", "baseline-smoke-001") + "-" + secrets.token_hex(4)
    result_token = secrets.token_urlsafe(32)
    source = Path(__file__).with_name("gpu_smoke_worker.py").read_bytes()
    code = base64.b64encode(zlib.compress(source)).decode()
    bootstrap = "import base64,zlib;exec(zlib.decompress(base64.b64decode('" + code + "')))"
    body = {
        "name": job_name,
        "image": IMAGE,
        "cloud": "SECURE",
        "disk": 80,
        "gpu": {
            "id": GPU,
            "count": 1,
            "minRamPerGpu": 32,
            "minVcpuCountPerGpu": 4,
            "minCudaVersion": "12.4",
        },
        "entrypoint": ["python", "-u", "-c"],
        "cmd": [bootstrap],
        "ports": ["8080/http"],
        "env": {
            "KH_JOB": base64.b64encode(zlib.compress(json.dumps(payload).encode())).decode(),
            "KH_RESULT_TOKEN": result_token,
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TOKENIZERS_PARALLELISM": "false",
        },
    }
    if len(json.dumps(body).encode()) > 100000:
        raise ValueError("deployment payload too large")
    catalog = api("GET", "/v2/catalog/gpus?include=AVAILABILITY&product=POD&cloud=SECURE&count=1")
    gpu = next(g for g in catalog["gpus"] if g["id"] == GPU)
    if gpu["availability"] == "NONE" or not 0 < gpu["price"]["secure"] <= 1.50:
        raise RuntimeError("approved GPU unavailable or above price limit")
    write_json(directory / "catalog.json", gpu)
    write_json(
        directory / "plan.json",
        {
            "job_name": job_name,
            "image": IMAGE,
            "gpu": GPU,
            "disk_gb": 80,
            "persistent_volumes": 0,
            "budget_usd": 3,
            "max_seconds": MAX_SECONDS,
            "worker_sha256": hashlib.sha256(source).hexdigest(),
            "payload_sha256": hashlib.sha256(json.dumps(payload).encode()).hexdigest(),
            "catalog_compute_hourly_usd": gpu["price"]["secure"],
        },
    )
    # This is an explicit, single create request. A failed/ambiguous create is
    # reconciled by the unique job name, never retried blindly.
    pod_id = None
    creation_started = time.time()
    deadline = creation_started + MAX_SECONDS
    try:
        try:
            pod = api("POST", "/v2/pods", body)
        except urllib.error.HTTPError as exc:
            detail = exc.read(8000).decode(errors="replace")
            for secret in (result_token, os.environ.get("RUNPOD_API_KEY", "")):
                if secret:
                    detail = detail.replace(secret, "[redacted]")
            write_json(directory / "create-error.json", {"http_status": exc.code, "detail": detail})
            log("create_rejected", http_status=exc.code, detail=detail)
            raise
        pod_id = pod["id"]
        state = {
            k: pod.get(k)
            for k in ["id", "name", "status", "cost", "createdAt", "dataCenterId", "cudaVersion"]
        }
        state.update(deadline_epoch=deadline, creation_started_epoch=creation_started)
        write_json(directory / "deployment.json", state)
        with (directory / "watchdog.log").open("a") as output:
            subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "--watchdog", str(directory)],
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        log("created", **state)
        # The returned cost is authoritative for this placement. Disk adds
        # $0.10/GB/month (~$0.012/h for 80GB); the $3 budget has ample margin.
        if not isinstance(pod.get("cost"), (int, float)) or not 0 < pod["cost"] <= 1.50:
            raise RuntimeError("deployment hourly cost outside approval")
        proxy = "https://" + pod_id + "-8080.proxy.runpod.net/status"
        last = None
        while time.time() < deadline:
            try:
                result = request("GET", proxy, result_token, timeout=20)
                write_json(directory / "remote-results.json", result)
                progress = {
                    "status": result["status"],
                    "candidate": result.get("candidate"),
                    "outputs": {k: len(v["outputs"]) for k, v in result["models"].items()},
                }
                if progress != last:
                    log("progress", **progress)
                    last = progress
                if result["status"] in {"done", "failed"}:
                    break
            except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
                log("waiting_for_worker", error=type(exc).__name__)
            if (
                time.time() - creation_started > 900
                and not (directory / "remote-results.json").exists()
            ):
                raise RuntimeError("worker readiness exceeded 15 minutes")
            time.sleep(10)
    finally:
        if pod_id is None:
            # Reconcile a transport failure without touching any other Pod.
            listing = api("GET", "/v2/pods")
            pods = listing["pods"] if isinstance(listing, dict) else listing
            matches = [p for p in pods if p.get("name") == job_name]
            if len(matches) == 1:
                pod_id = matches[0]["id"]
            elif matches:
                raise RuntimeError("multiple Pods share unique job name; inspect immediately")
            else:
                write_json(
                    directory / "no-pod-created.json",
                    {"job_name": job_name, "confirmed_by_listing": True, "epoch": time.time()},
                )
        if pod_id:
            for attempt in range(6):
                try:
                    if delete_and_confirm(pod_id):
                        write_json(
                            directory / "termination.json",
                            {
                                "confirmed": True,
                                "actor": "controller",
                                "pod_id": pod_id,
                                "epoch": time.time(),
                                "elapsed_seconds": time.time() - creation_started,
                            },
                        )
                        log("terminated", pod_id=pod_id)
                        break
                except Exception as exc:
                    log("delete_error", error=type(exc).__name__, attempt=attempt + 1)
                time.sleep(5)
            else:
                raise RuntimeError("Pod deletion not confirmed; watchdog remains active")
    if pod_id:
        billing = api("GET", "/v2/billing/pods?bucketSize=hour&lastN=2&podId=" + pod_id)
        write_json(directory / "billing.json", billing)
        log("billing", metadata=billing.get("metadata"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundles", nargs=2, type=Path)
    parser.add_argument("--draft-pilot", type=Path)
    parser.add_argument("--draft-manifest-sha256")
    parser.add_argument("--rights-review", type=Path)
    parser.add_argument("--study", type=Path)
    parser.add_argument("--study-sha256")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--watchdog", type=Path)
    args = parser.parse_args()
    if args.watchdog:
        watchdog(args.watchdog)
        return
    if (
        not args.output_dir
        or sum(bool(x) for x in (args.bundles, args.study, args.draft_pilot)) != 1
    ):
        parser.error("--output-dir and exactly one input kind are required")
    if args.draft_pilot:
        if args.execute and args.draft_manifest_sha256 != sha256_file(
            args.draft_pilot / "manifest.json"
        ):
            parser.error("execution requires the exact reviewed --draft-manifest-sha256")
        payload = load_drafts(args.draft_pilot, args.rights_review, execute=args.execute)
    elif args.study:
        manifest, jobs = load_study(args.study)
        study_sha = sha256_file(args.study / "study.json")
        if args.execute and args.study_sha256 != study_sha:
            parser.error("execution requires the exact reviewed --study-sha256")
        payload = {"job_id": manifest["study_id"], "study_sha256": study_sha, "jobs": jobs}
    else:
        payload = job_payload(args.bundles)
    if not args.execute:
        log(
            "dry_run",
            candidates=[j["spec"]["candidate"] for j in payload["jobs"]],
            requests=sum(len(j["requests"]) for j in payload["jobs"]),
            max_output_tokens=2048,
            budget_usd=3,
            network_calls=0,
        )
        return
    deploy(args.output_dir.resolve(), payload)


if __name__ == "__main__":
    main()
