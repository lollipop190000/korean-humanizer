"""Synthetic prompt/token fixtures; no model imports, GPU calls or human corpus."""

import hashlib

import pytest

from korean_humanizer.batch_generation import batch_plan, generated_tokens, verify_prompt


def test_short_left_padded_rows_slice_at_batch_width_and_stop_at_their_first_eos():
    short = [0, 0, 1, 2, 31, 9, 0, 0]
    long = [1, 2, 3, 4, 51, 52, 53, 9]
    assert generated_tokens(short, 4, {9}) == ([31, 9], "stop")
    assert generated_tokens(long, 4, {9}) == ([51, 52, 53, 9], "stop")
    assert generated_tokens(long[:-1], 4, {9}) == ([51, 52, 53], "length")
    with pytest.raises(ValueError, match="no completion"):
        generated_tokens(short[:4], 4, {9})


def test_batches_are_deterministic_independent_of_input_file_order_and_length_grouped():
    requests = [{"case_id": key} for key in ["b", "c", "a"]]
    audits = [
        {"case_id": key, "prompt_tokens": size} for key, size in [("a", 4), ("b", 40), ("c", 2)]
    ]
    plan = batch_plan(requests, audits, batch_size=2)
    assert plan == [{"seed": 42, "case_ids": ["c", "a"]}, {"seed": 43, "case_ids": ["b"]}]
    assert batch_plan(list(reversed(requests)), audits, batch_size=2) == plan
    with pytest.raises(ValueError, match="one-to-one"):
        batch_plan(requests + requests[:1], audits)


def test_prompt_and_context_checks_do_not_allow_silent_changes_or_truncation():
    text = "fixture-only"
    audit = {
        "rendered_prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "prompt_tokens": 4,
    }
    verify_prompt(text, audit, 4, 2, 6)
    with pytest.raises(ValueError, match="prompt mismatch"):
        verify_prompt(text + " changed", audit, 4, 2, 6)
    with pytest.raises(ValueError, match="overflow"):
        verify_prompt(text, audit, 4, 3, 6)


def test_recovery_eight_row_batches_preserve_unique_seeds_and_an_upper_bound():
    requests = [{"case_id": str(i)} for i in range(17)]
    audits = [{"case_id": str(i), "prompt_tokens": i + 1} for i in range(17)]
    plan = batch_plan(requests, audits, batch_size=8)
    assert [len(b["case_ids"]) for b in plan] == [8, 8, 1]
    assert len({key for batch in plan for key in batch["case_ids"]}) == 17
    assert [b["seed"] for b in plan] == [42, 43, 44]
    with pytest.raises(ValueError):
        batch_plan(requests, audits, batch_size=9)
