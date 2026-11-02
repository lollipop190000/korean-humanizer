"""An explicit identity baseline, not a trained Humanizer implementation."""

from typing import Protocol

from korean_humanizer.protection import check_protection


class RewriteBackend(Protocol):
    name: str

    def rewrite(self, text: str) -> str: ...


class IdentityBaseline:
    name = "identity-baseline"

    def rewrite(self, text: str) -> str:
        return text


def inspect_rewrite(text: str, backend: RewriteBackend, spans: list[str] | None = None) -> dict:
    if not text.strip():
        raise ValueError("input must not be empty")
    output = backend.rewrite(text)
    return {
        "backend": backend.name,
        "output": output,
        "checks": check_protection(text, output, spans),
    }
