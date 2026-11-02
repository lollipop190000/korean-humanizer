"""SFT export and CPU training plans; no model training is performed here."""

from collections import Counter

from korean_humanizer.data import assign_splits, source_errors
from korean_humanizer.models import Pair
from korean_humanizer.protection import check_protection

INSTRUCTION = (
    "다음 글을 자연스러운 한국어로 다듬으세요. 사실, 부정, 조건, 불확실성과 원래 어투를 "
    "유지하세요. 숫자, 단위, 이름, 인용, 수식과 코드를 보존하세요. 이미 자연스러운 부분은 "
    "그대로 두세요. 설명 없이 수정한 글만 출력하세요."
)


def sft_records(
    pairs: list[Pair], *, allow_mock: bool = False, allow_pending_research: bool = False
) -> list[dict]:
    if allow_mock and allow_pending_research:
        raise ValueError("mock and pending research policies cannot be combined")
    if not pairs:
        raise ValueError("cannot export zero examples")
    identifiers = [pair.pair_id for pair in pairs]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("duplicate pair_id")
    sources = {}
    source_splits = {}
    for pair in pairs:
        doc_id = pair.source.source_doc_id
        if doc_id in sources and sources[doc_id] != pair.source:
            raise ValueError("conflicting metadata for the same source_doc_id")
        sources[doc_id] = pair.source
        source_splits.setdefault(doc_id, set()).add(pair.split)
    grouped_splits = {}
    for assigned in assign_splits(list(sources.values())):
        grouped_splits.setdefault(assigned.group_id, set()).update(
            source_splits[assigned.source.source_doc_id]
        )
    if any(len(splits) > 1 for splits in grouped_splits.values()):
        raise ValueError("related or duplicate sources leak across splits")
    rows = []
    for pair in pairs:
        # model_copy can bypass Pydantic validators; exports recheck the contract.
        pair.control_is_not_ai_generation()
        if pair.split == "test":
            raise ValueError("test pairs must never enter SFT export")
        mock = (
            pair.generation.mock
            and pair.source.origin == "synthetic_fixture"
            and pair.review_status == "mock_only"
        )
        if allow_mock:
            if not mock:
                raise ValueError("mock export requires explicit mock fixtures throughout")
        elif allow_pending_research:
            if (
                pair.generation.mock
                or pair.source.origin != "human"
                or pair.review_status != "pending"
            ):
                raise ValueError("pending research requires real, pending pairs throughout")
        elif (
            pair.generation.mock
            or pair.source.origin != "human"
            or pair.review_status != "accepted"
            or not pair.review_ref
        ):
            raise ValueError("production export requires human sources and documented acceptance")
        errors = source_errors(pair.source, allow_fixture=allow_mock)
        if errors:
            raise ValueError(f"pair {pair.pair_id}: {errors}")
        check = check_protection(
            pair.synthetic_ai_text, pair.source.text, pair.source.protected_spans
        )
        if not check["mechanical_pass"]:
            raise ValueError(f"pair {pair.pair_id}: protected information differs")
        rows.append(
            {
                "pair_id": pair.pair_id,
                "split": pair.split,
                "purpose": (
                    "mock_smoke_only"
                    if allow_mock
                    else "unreviewed_research_sft"
                    if allow_pending_research
                    else "sft"
                ),
                "messages": [
                    {"role": "system", "content": INSTRUCTION},
                    {"role": "user", "content": pair.synthetic_ai_text},
                    {"role": "assistant", "content": pair.source.text},
                ],
            }
        )
        if pair.input_kind == "natural_control":
            rows[-1]["input_kind"] = "natural_control"
    return rows


def training_plan(pairs: list[Pair], *, allow_mock: bool = False) -> dict:
    rows = sft_records(pairs, allow_mock=allow_mock)
    splits = Counter(row["split"] for row in rows)
    if not splits["train"] or not splits["dev"]:
        raise ValueError("a training plan requires both train and dev examples")
    return {
        "status": "dry_run_only",
        "training_executed": False,
        "purpose": "mock_smoke_only" if allow_mock else "sft_preparation",
        "method": "sft",
        "examples": dict(splits),
        "model": "not_selected",
        "gpu_dependencies": "not_installed",
        "prerequisites": [
            "select model from baseline evidence",
            "verify tokenizer and chat template",
            "tokenize and check lengths without silent truncation",
            "approve GPU job and budget",
            "test checkpoint save and resume",
        ],
    }
