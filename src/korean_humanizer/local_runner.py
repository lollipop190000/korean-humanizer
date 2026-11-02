"""Explicitly run a prepared benchmark against a local OpenAI-compatible server."""

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from korean_humanizer.benchmark import OutputRow, load_bundle
from korean_humanizer.io import sha256_file, write_json


class NoRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.URLError("local runner does not follow redirects")


def local_base_url(value: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ValueError(
            "only a local HTTP server root is supported; hosted API URLs are not accepted"
        )
    # Access port to reject malformed ports before making any requests.
    _ = parsed.port
    return value.rstrip("/")


def parse_completion(payload: dict, expected_model: str) -> tuple[str, str]:
    if payload.get("model") != expected_model:
        raise ValueError("server_model_mismatch")
    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("invalid_choices")
    choice = choices[0]
    reason = choice.get("finish_reason")
    if reason not in {"stop", "length"}:
        raise ValueError("unsupported_finish_reason")
    text = choice.get("text")
    if text is None:
        text = choice.get("message", {}).get("content")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("empty_or_nontext_response")
    return text, reason


def run_local(
    bundle: Path,
    base_url: str,
    *,
    output_dir: Path | None = None,
    token_audit: Path | None = None,
    execute: bool = False,
    max_seconds: float = 900,
    request_timeout: float = 120,
) -> dict:
    _, _, requests, spec = load_bundle(bundle)
    base_url = local_base_url(base_url)
    if not (0 < max_seconds <= 86400 and 0 < request_timeout <= max_seconds):
        raise ValueError("positive time limits required; request timeout must fit the total limit")
    plan = {
        "candidate": spec.candidate,
        "requests": len(requests),
        "base_url": base_url,
        "mode": "dry_run",
        "network_calls": 0,
        "retries": 0,
        "max_seconds": max_seconds,
        "max_output_tokens_per_request": spec.max_output_tokens,
        "note": "No cloud resources are provisioned. Timeouts do not stop or bill-cap a GPU Pod.",
    }
    if not execute:
        return plan
    if output_dir is None or token_audit is None:
        raise ValueError("execution requires a new output directory and matching token audit")
    audit = json.loads(token_audit.read_text())
    if (
        audit.get("bundle_manifest_sha256") != sha256_file(bundle / "manifest.json")
        or audit.get("candidate") != spec.candidate
        or not audit.get("all_within_context")
        or audit.get("revision") != spec.revision
    ):
        raise ValueError("token audit does not approve this bundle's context lengths")
    output_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    deadline = time.monotonic() + max_seconds
    endpoint = "/v1/completions" if spec.protocol == "completion" else "/v1/chat/completions"
    # These URLs are loopback-only; never forward local drafts to an environment HTTP proxy.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirects())
    output_rows, calls = [], 0
    with (output_dir / "outputs.jsonl").open("x", encoding="utf8") as handle:
        (output_dir / "outputs.jsonl").chmod(0o600)
        for request in requests:
            row = {
                "case_id": request["case_id"],
                "candidate": spec.candidate,
                "request_sha256": request["request_sha256"],
                "text": "",
                "finish_reason": "error",
                "error_code": "deadline_exceeded",
            }
            started = time.monotonic()
            if started < deadline:
                req = urllib.request.Request(
                    base_url + endpoint,
                    json.dumps(request["body"]).encode(),
                    {"Content-Type": "application/json"},
                )
                try:
                    calls += 1
                    with opener.open(
                        req, timeout=min(request_timeout, deadline - started)
                    ) as response:
                        content = response.read(1024 * 1024 + 1)
                    if len(content) > 1024 * 1024:
                        raise ValueError("response_too_large")
                    payload = json.loads(content)
                    if not isinstance(payload, dict):
                        raise ValueError("invalid_response_object")
                    text, reason = parse_completion(payload, spec.model_id)
                    row.update(text=text, finish_reason=reason, error_code="")
                except urllib.error.HTTPError as exc:
                    row["error_code"] = f"http_{exc.code}"
                except (urllib.error.URLError, TimeoutError, OSError):
                    row["error_code"] = "connection_or_timeout"
                except (ValueError, TypeError, AttributeError):
                    row["error_code"] = "invalid_model_response"
            row["latency_seconds"] = time.monotonic() - started
            validated = OutputRow.model_validate(row)
            handle.write(validated.model_dump_json() + "\n")
            handle.flush()
            output_rows.append(validated)
    report = plan | {
        "mode": "executed_local",
        "network_calls": calls,
        "complete": sum(row.finish_reason == "stop" for row in output_rows),
        "errors": sum(row.finish_reason == "error" for row in output_rows),
        "truncated": sum(row.finish_reason == "length" for row in output_rows),
        "requested_revision": spec.revision,
        "server_revision_verified": False,
        "manifest_sha256": sha256_file(bundle / "manifest.json"),
    }
    write_json(output_dir / "run.json", report)
    return report
