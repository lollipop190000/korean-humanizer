"""Mock wiring only. No paid provider or network calls are implemented."""

from typing import Protocol

from korean_humanizer.data import source_errors
from korean_humanizer.models import AssignedSource, Generation, Pair


class DraftProvider(Protocol):
    def generate(self, text: str, *, seed: int) -> str: ...


class MockProvider:
    def generate(self, text: str, *, seed: int) -> str:
        return "다음 내용을 정리했습니다. " + text


def generate_mock_pair(item: AssignedSource, *, seed: int = 42) -> Pair:
    if item.split == "test":
        raise ValueError("test sources are locked; synthesis is forbidden")
    if item.source.origin != "synthetic_fixture":
        raise ValueError("mock pipeline accepts synthetic fixtures only")
    errors = source_errors(item.source, allow_fixture=True)
    if errors:
        raise ValueError(f"source {item.source.source_doc_id}: {errors}")
    return Pair(
        pair_id=f"mock-{item.source.source_doc_id}",
        source=item.source,
        split=item.split,
        synthetic_ai_text=MockProvider().generate(item.source.text, seed=seed),
        generation=Generation(
            model="mock-prefix-v1", prompt_version="mock-v1", seed=seed, mock=True
        ),
        review_status="mock_only",
    )
