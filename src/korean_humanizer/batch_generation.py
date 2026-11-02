"""Offline batch planning and stop-token accounting, without GPU imports or spending."""

import hashlib


def batch_plan(requests, audits, *, batch_size=4, seed=42):
    if not 1 <= batch_size <= 8 or not requests or len(requests) > 6000:
        raise ValueError("batch size/request count outside the scale scope")
    indexed = {a["case_id"]: a for a in audits}
    ids = [r["case_id"] for r in requests]
    if len(set(ids)) != len(ids) or set(ids) != set(indexed) or len(indexed) != len(audits):
        raise ValueError("requests and token audits must correspond one-to-one")
    ordered = sorted(requests, key=lambda r: (indexed[r["case_id"]]["prompt_tokens"], r["case_id"]))
    batches = []
    for offset in range(0, len(ordered), batch_size):
        batch = ordered[offset : offset + batch_size]
        batches.append(
            {"seed": seed + offset // batch_size, "case_ids": [r["case_id"] for r in batch]}
        )
    return batches


def verify_prompt(prompt, audit, count, max_output_tokens, context_tokens):
    if hashlib.sha256(prompt.encode()).hexdigest() != audit["rendered_prompt_sha256"]:
        raise ValueError("rendered prompt mismatch")
    if count != audit["prompt_tokens"] or count + max_output_tokens > context_tokens:
        raise ValueError("prompt token count mismatch or context overflow")


def generated_tokens(sequence, padded_prompt_width, eos_ids):
    """All rows start after the padded width, including left-padded short prompts."""
    if padded_prompt_width < 1 or len(sequence) <= padded_prompt_width:
        raise ValueError("generation contains no completion tokens")
    completion = list(sequence[padded_prompt_width:])
    for index, token in enumerate(completion):
        if token in eos_ids:
            return completion[: index + 1], "stop"
    return completion, "length"
