"""Bounded local document inference. Never substitute source text for failed output."""

import re

from korean_humanizer.protection import check_protection

MAX_CHARACTERS = 50_000
MAX_CHUNKS = 256
PROMPT_TOKENS = 1024
OUTPUT_TOKENS = 1024
MAX_SPLIT_RETRIES = 3


def change_kind(source, output):
    if source == output:
        return "unchanged"
    if "".join(source.split()) == "".join(output.split()):
        return "whitespace_only"
    return "changed"


def boundary(text, limit):
    """Prefer sentence/word boundaries without discarding any characters."""
    matches = list(re.finditer(r"[.!?。！？](?:[ \t]+|$)|[ \t]+", text[:limit]))
    return next((m.end() for m in reversed(matches) if m.end() >= limit // 2), limit)


def split_document(text, count_tokens, budget=PROMPT_TOKENS):
    """Return exact source slices; whitespace-only slices need no inference.

    Newlines remain literal so blank lines, CRLF and indentation survive merging.
    Every nonblank slice is checked with the actual rendered chat prompt.
    """
    if count_tokens("") >= budget:
        raise ValueError("모델 지시문이 구간 토큰 한도를 초과했습니다.")
    parts = []
    for line in re.split(r"(\r\n|\r|\n)", text):
        while line:
            if not line.strip() or count_tokens(line.strip()) <= budget:
                parts.append(line)
                break
            low, high, best = 1, len(line) - 1, 0
            while low <= high:
                middle = (low + high) // 2
                if count_tokens(line[:middle].strip()) <= budget:
                    best, low = middle, middle + 1
                else:
                    high = middle - 1
            if not best:
                raise ValueError("한 글자도 처리 구간에 담을 수 없습니다.")
            cut = boundary(line, best)
            # Token counts are not strictly monotonic at tokenization boundaries.
            if count_tokens(line[:cut].strip()) > budget:
                cut = best
            parts.append(line[:cut])
            line = line[cut:]
    return parts


class DocumentError(ValueError):
    """A failed document is never exposed as a complete result."""


def rewrite_document(model, text, progress=None):
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_CHARACTERS:
        raise ValueError("1~50,000자의 글을 입력해 주세요.")
    parts = split_document(text, model.prompt_token_count)
    total = sum(bool(part.strip()) for part in parts)
    completed = retries = warnings = 0
    metadata = None

    def update():
        if total > MAX_CHUNKS:
            raise DocumentError("처리 구간이 256개를 넘습니다. 문서를 나눠 입력해 주세요.")
        if progress:
            progress(completed, total, retries)

    def process(part, depth=0):
        nonlocal total, completed, retries, warnings, metadata
        core = part.strip()
        if not core:
            return part
        leading = part[: len(part) - len(part.lstrip())]
        trailing = part[len(part.rstrip()) :]
        update()
        result = model.rewrite(core, max_new_tokens=OUTPUT_TOKENS)
        if result["finish_reason"] == "length":
            if depth >= MAX_SPLIT_RETRIES or len(core) < 2:
                raise DocumentError("작은 구간에서도 출력이 잘려 처리를 중단했습니다.")
            cut = boundary(core, len(core) // 2)
            smaller = []
            for half in (core[:cut], core[cut:]):
                smaller.extend(split_document(half, model.prompt_token_count))
            total += sum(bool(p.strip()) for p in smaller) - 1
            retries += 1
            update()
            return leading + "".join(process(p, depth + 1) for p in smaller) + trailing
        if result["finish_reason"] != "stop" or not result["output"].strip():
            raise DocumentError("모델이 정상적인 결과를 반환하지 않아 중단했습니다.")
        metadata = result
        output = leading + result["output"].strip() + trailing
        warnings += not check_protection(part, output)["mechanical_pass"]
        completed += 1
        update()
        return output

    try:
        update()
        output = "".join(process(part) for part in parts)
    except Exception as error:
        if isinstance(error, DocumentError):
            reason = str(error)
        elif "out of memory" in str(error).lower():
            reason = "GPU 메모리가 부족합니다. 다른 GPU 앱을 닫고 다시 실행해 주세요."
        else:
            reason = "모델 실행에 실패했습니다. 실행 환경과 모델 설치를 확인해 주세요."
        raise DocumentError(
            f"전체 미완료 ({completed}/{total}구간 완료). {reason} 원래 글은 유지됩니다."
        ) from error
    checks = check_protection(text, output)
    return {
        **metadata,
        "output": output,
        "finish_reason": "stop",
        "checks": checks,
        "chunk_warning_count": warnings,
        "chunks_completed": completed,
        "chunks_total": total,
        "split_retries": retries,
        "change_kind": change_kind(text, output),
        "human_review_completed": False,
    }
