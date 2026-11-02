"""Conservative mechanical checks, never a semantic-fidelity verdict."""

import re
from collections import Counter

PATTERN = re.compile(
    r"```[\s\S]*?```|`[^`\n]+`|\$\$[\s\S]*?\$\$|\$[^$\n]+\$"
    r'|“[^”]+”|「[^」]+」|"[^"\n]+"'
    r"|https?://[^\s<>]+"
    r"|\d+(?:,\d{3})*(?:\.\d+)?(?:\s?(?:kg|km|mg|mm|cm|mL|GB|MB|Hz|°C|℃|%|원|명|개|년|월|일|초))?"
    r"|[A-Za-z][A-Za-z0-9_.+\-]*|[\u3400-\u4dbf\u4e00-\u9fff]+"
)


def protected_spans(text: str):
    """Yield literal spans; a URL's unmatched closing bracket belongs to surrounding prose."""
    position = 0
    while match := PATTERN.search(text, position):
        start, end = match.span()
        raw = match.group()
        if raw.startswith(("https://", "http://")):
            stack = []
            for i, char in enumerate(raw):
                if char in "([{":
                    stack.append(char)
                elif char in ")]}":
                    if not stack or {")": "(", "]": "[", "}": "{"}[char] != stack[-1]:
                        end = start + i
                        break
                    stack.pop()
        yield start, end, text[start:end]
        position = end


def atoms(text: str) -> Counter[str]:
    result: Counter[str] = Counter()
    for _, _, atom in protected_spans(text):
        # Permit spacing around a numeric unit, but preserve quotes/code/math verbatim.
        if atom[0].isdigit():
            atom = re.sub(r"\s+", "", atom)
        result[atom] += 1
    return result


def check_protection(original: str, rewritten: str, spans: list[str] | None = None) -> dict:
    before, after = atoms(original), atoms(rewritten)
    missing = dict(before - after)
    added = dict(after - before)
    changed_spans = [
        span
        for span in spans or []
        if not span or span not in original or original.count(span) != rewritten.count(span)
    ]
    return {
        "mechanical_pass": bool(rewritten.strip()) and not (missing or added or changed_spans),
        "missing": missing,
        "added": added,
        "changed_explicit_spans": changed_spans,
        "semantic_status": "not_assessed",
    }
