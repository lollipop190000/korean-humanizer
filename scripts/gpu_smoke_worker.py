"""One-shot Qwen smoke worker, run only inside the approved ephemeral GPU Pod.

Its HTTPS-proxied status endpoint requires a per-job random bearer token.
The RunPod account API key is never forwarded to this worker.
"""

import base64
import gc
import hashlib
import hmac
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import traceback
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

STATE = {"status": "booting", "models": {}, "errors": []}
LOCK = threading.Lock()


def update(**values):
    with LOCK:
        STATE.update(values)


class StatusHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        expected = "Bearer " + os.environ["KH_RESULT_TOKEN"]
        if not hmac.compare_digest(self.headers.get("Authorization", ""), expected):
            self.send_error(401)
            return
        if self.path != "/status":
            self.send_error(404)
            return
        with LOCK:
            body = json.dumps(STATE, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


def run_job(payload):
    update(status="installing")
    requirements = [
        "transformers==4.51.3",
        "accelerate==1.6.0",
        "safetensors==0.5.3",
        "huggingface-hub==0.30.2",
    ]
    with Path("/tmp/kh-install.log").open("w") as log:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", *requirements],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=480,
        )
    import torch
    import transformers
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.set_num_threads(4)
    update(
        runtime={
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "nvidia_smi": subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader",
                ],
                text=True,
            ).strip(),
            "pip_freeze": subprocess.check_output(
                [sys.executable, "-m", "pip", "freeze"], text=True
            ),
        }
    )
    for job in payload["jobs"]:
        spec = job["spec"]
        candidate = spec["candidate"]
        cache = Path("/tmp/kh-model-cache") / candidate
        info = {
            "revision": spec["revision"],
            "model_id": spec["model_id"],
            "outputs": [],
            "metrics": [],
            "manifest_sha256": job["manifest_sha256"],
        }
        with LOCK:
            STATE["models"][candidate] = info
        update(status="downloading", candidate=candidate)
        snapshot = Path(
            snapshot_download(
                spec["model_id"],
                revision=spec["revision"],
                cache_dir=cache,
                allow_patterns=["*.json", "*.safetensors", "*.model"],
                max_workers=4,
            )
        )
        if snapshot.name != spec["revision"]:
            raise RuntimeError("model revision mismatch")
        tokenizer_sha = hashlib.sha256((snapshot / "tokenizer.json").read_bytes()).hexdigest()
        if tokenizer_sha != job["audit"]["tokenizer_sha256"]:
            raise RuntimeError("tokenizer checksum mismatch")
        update(status="loading", candidate=candidate)
        started = time.monotonic()
        tokenizer = AutoTokenizer.from_pretrained(snapshot, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(
            snapshot,
            torch_dtype=torch.bfloat16,
            device_map={"": "cuda:0"},
            trust_remote_code=False,
            use_safetensors=True,
            attn_implementation="sdpa",
        ).eval()
        info["load_seconds"] = time.monotonic() - started
        info["revision_verified"] = True
        info["tokenizer_sha256"] = tokenizer_sha
        info["weight_files"] = [
            {"name": p.name, "bytes": p.stat().st_size, "cache_blob": p.resolve().name}
            for p in sorted(snapshot.glob("*.safetensors"))
        ]
        audits = {r["case_id"]: r for r in job["audit"]["cases"]}
        update(status="generating", candidate=candidate)
        for request in job["requests"]:
            body = request["body"]
            row = {
                "case_id": request["case_id"],
                "candidate": candidate,
                "request_sha256": request["request_sha256"],
                "text": "",
                "finish_reason": "error",
                "error_code": "generation_failed",
            }
            started = time.monotonic()
            try:
                prompt = tokenizer.apply_chat_template(
                    body["messages"],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                audit = audits[request["case_id"]]
                if hashlib.sha256(prompt.encode()).hexdigest() != audit["rendered_prompt_sha256"]:
                    raise RuntimeError("rendered prompt mismatch")
                inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to("cuda")
                prompt_count = inputs.input_ids.shape[-1]
                if prompt_count != audit["prompt_tokens"]:
                    raise RuntimeError("prompt token count mismatch")
                if prompt_count + body["max_tokens"] > spec["context_tokens"]:
                    raise RuntimeError("context overflow")
                set_seed(body["seed"])
                torch.cuda.reset_peak_memory_stats()
                with torch.inference_mode():
                    output = model.generate(
                        **inputs,
                        do_sample=True,
                        max_new_tokens=body["max_tokens"],
                        temperature=body["temperature"],
                        top_p=body["top_p"],
                        top_k=body["top_k"],
                        min_p=body["min_p"],
                        repetition_penalty=body["repetition_penalty"],
                        pad_token_id=tokenizer.eos_token_id,
                    )
                torch.cuda.synchronize()
                generated = output[0, prompt_count:]
                eos = model.generation_config.eos_token_id
                eos = [eos] if isinstance(eos, int) else eos
                stopped = int(generated[-1]) in (eos or [])
                text = tokenizer.decode(generated, skip_special_tokens=True)
                if not text.strip():
                    raise RuntimeError("empty model output")
                row.update(text=text, finish_reason="stop" if stopped else "length", error_code="")
                with LOCK:
                    info["metrics"].append(
                        {
                            "case_id": row["case_id"],
                            "prompt_tokens": prompt_count,
                            "output_tokens": len(generated),
                            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                        }
                    )
                del inputs, output, generated
            except Exception as exc:
                row["error_code"] = type(exc).__name__
                with LOCK:
                    STATE["errors"].append({"case_id": row["case_id"], "detail": str(exc)[:500]})
            row["latency_seconds"] = time.monotonic() - started
            with LOCK:
                info["outputs"].append(row)
            print(
                json.dumps(
                    {
                        "candidate": candidate,
                        "case_id": row["case_id"],
                        "finish_reason": row["finish_reason"],
                    }
                ),
                flush=True,
            )
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        shutil.rmtree(cache)
    update(status="done")


def main():
    payload = json.loads(zlib.decompress(base64.b64decode(os.environ.pop("KH_JOB"))))
    server = ThreadingHTTPServer(("0.0.0.0", 8080), StatusHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        run_job(payload)
    except Exception:
        update(status="failed", fatal_error=traceback.format_exc()[-6000:])
    # The controller retrieves results and deletes this Pod. Exiting this process
    # alone would not terminate RunPod billing.
    while True:
        time.sleep(15)


if __name__ == "__main__":
    main()
