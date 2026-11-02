"""Conservative aggregate reservations for scale-001. This module provisions nothing."""

import copy

MAX_SECONDS = 40 * 3600
MAX_COMPUTE_HOURLY = 0.70
MAX_DISK_HOURLY = 120 * 0.10 / 720
RESERVE_USD = 1.30
HOURLY_CEILING = MAX_COMPUTE_HOURLY + MAX_DISK_HOURLY


def empty_ledger(user_approval_ref):
    if not user_approval_ref:
        raise ValueError("a recorded user approval is required")
    return {
        "job_id": "scale-001",
        "max_total_usd": 30,
        "user_approval_ref": user_approval_ref,
        "phases": [],
    }


def used_seconds(ledger):
    return sum(p.get("confirmed_elapsed_seconds", p["reserved_seconds"]) for p in ledger["phases"])


def reserve_phase(ledger, manifest_sha, kind, case_ids, reserved_seconds):
    if ledger.get("job_id") != "scale-001" or ledger.get("max_total_usd") != 30:
        raise ValueError("ledger differs from scale-001 spending scope")
    if not ledger.get("user_approval_ref"):
        raise ValueError("missing recorded user approval")
    if (
        kind not in {"drafts", "judge", "train", "evaluation"}
        or not 600 <= reserved_seconds <= MAX_SECONDS
    ):
        raise ValueError("invalid phase kind or reserved duration")
    if any(not p.get("termination_confirmed") for p in ledger["phases"]):
        raise ValueError("an active or unconfirmed phase blocks another GPU")
    phases_started = [p for p in ledger["phases"] if not p.get("provisioning_rejected")]
    if len(manifest_sha) != 64 or any(p["manifest_sha256"] == manifest_sha for p in phases_started):
        raise ValueError("invalid manifest or automatic rerun attempt")
    previous = {key for p in phases_started if p["kind"] == kind for key in p["case_ids"]}
    if len(set(case_ids)) != len(case_ids) or previous.intersection(case_ids):
        raise ValueError("duplicate requests or automatic retries are forbidden")
    if kind in {"drafts", "judge"} and len(previous) + len(case_ids) > 6000:
        raise ValueError("workflow request limit exceeded")
    if kind == "train" and any(p["kind"] == "train" for p in phases_started):
        raise ValueError("only one SFT run is authorized")
    total_seconds = used_seconds(ledger) + reserved_seconds
    total_bound = total_seconds / 3600 * HOURLY_CEILING + RESERVE_USD
    if total_seconds > MAX_SECONDS or total_bound > 30:
        raise ValueError("aggregate time/cost reservation exceeds approval")
    result = copy.deepcopy(ledger)
    result["phases"].append(
        {
            "manifest_sha256": manifest_sha,
            "kind": kind,
            "case_ids": list(case_ids),
            "reserved_seconds": reserved_seconds,
            "termination_confirmed": False,
        }
    )
    return result


def finish_phase(ledger, manifest_sha, elapsed_seconds, *, termination_confirmed):
    if not termination_confirmed or elapsed_seconds < 0:
        raise ValueError("release a reservation only after confirmed termination")
    result = copy.deepcopy(ledger)
    matches = [
        p
        for p in result["phases"]
        if p["manifest_sha256"] == manifest_sha and not p.get("termination_confirmed")
    ]
    if len(matches) != 1 or matches[0].get("termination_confirmed"):
        raise ValueError("unknown or already finalized phase")
    matches[0].update(termination_confirmed=True, confirmed_elapsed_seconds=elapsed_seconds)
    result["conservative_compute_and_disk_estimate_usd"] = (
        used_seconds(result) / 3600 * HOURLY_CEILING
    )
    result["billing_finalized"] = False
    return result
