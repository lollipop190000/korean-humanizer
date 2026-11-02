"""Constrained token sampling blocks malformed evidence without forcing pass verdicts."""

import json
from types import SimpleNamespace

import pytest
from lmformatenforcer import TokenEnforcerTokenizerData

from korean_humanizer.indexed_constraints import batch_prefix_filter, judgment_schema


def tokenizer_data():
    return TokenEnforcerTokenizerData(
        [(i, chr(i), False) for i in range(32, 127)],
        lambda ids: "".join(chr(i) for i in ids if 32 <= i < 127),
        0,
        False,
        128,
    )


def response(**changes):
    return {
        "genre": "experience_review",
        "standalone": "pass",
        "forward": "fail",
        "backward": "unclear",
        "logic": "pass",
        "stance": "pass",
        "register_match": "pass",
        "source_evidence_ids": ["S000"],
        "draft_evidence_ids": ["D000"],
        "genre_evidence_ids": ["S000"],
    } | changes


def accepted(value, batch=0):
    schemas = [judgment_schema("a" * 120, "b" * 120), judgment_schema("a" * 20, "b" * 20)]
    allowed = batch_prefix_filter(tokenizer_data(), schemas)
    prefix = [1]
    for char in json.dumps(value, separators=(",", ":")):
        if ord(char) not in allowed(batch, SimpleNamespace(tolist=lambda: prefix)):
            return False
        prefix.append(ord(char))
    return 0 in allowed(batch, SimpleNamespace(tolist=lambda: prefix))


def test_constraints_allow_fail_unclear_and_valid_evidence():
    assert accepted(response())
    assert accepted(response(source_evidence_ids=["S000", "S001", "S003"]))


@pytest.mark.parametrize(
    "changes",
    [
        {"source_evidence_ids": ["S000", "S001", "S002", "S003"]},
        {"draft_evidence_ids": []},
        {"source_evidence_ids": ["D000"]},
        {"genre_evidence_ids": ["S999"]},
        {"forward": "invented"},
        {"genre": "invalid"},
        {"explanation": "extra field"},
    ],
)
def test_constraints_block_invalid_json_schema_values(changes):
    assert not accepted(response(**changes))


def test_constraints_are_bound_to_each_batch_rows_actual_text():
    value = response(source_evidence_ids=["S003"])
    assert accepted(value, batch=0)
    assert not accepted(value, batch=1)
