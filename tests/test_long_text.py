import pytest

from korean_humanizer.long_text import (
    MAX_CHARACTERS,
    DocumentError,
    change_kind,
    rewrite_document,
    split_document,
)


class EchoModel:
    def __init__(self):
        self.calls = []

    def prompt_token_count(self, text):
        return 100 + len(text)

    def rewrite(self, text, max_new_tokens):
        assert self.prompt_token_count(text) <= 1024
        assert max_new_tokens == 1024
        self.calls.append(text)
        return {"output": text, "finish_reason": "stop", "device": "cpu"}


@pytest.mark.parametrize(
    "text",
    [
        "  첫 문단입니다.\r\n\r\n  두 번째 문단입니다. \n\t끝.  ",
        "한글🙂e\u0301 " * 1300,
        "띄어쓰기없는한글" * 500,
        "\t\n첫째\r둘째\r\n셋째\n\n  ",
    ],
)
def test_split_round_trip_and_exact_identity_including_whitespace(text):
    model = EchoModel()
    parts = split_document(text, model.prompt_token_count)
    assert "".join(parts) == text
    assert all(model.prompt_token_count(p.strip()) <= 1024 for p in parts if p.strip())
    progress = []
    result = rewrite_document(model, text, lambda *args: progress.append(args))
    assert result["output"] == text
    assert result["change_kind"] == "unchanged"
    assert result["chunks_completed"] == len(model.calls) == result["chunks_total"]
    assert progress[-1][:2] == (len(model.calls), len(model.calls))
    assert result["human_review_completed"] is False


def test_merge_changed_paragraphs_in_order_and_keep_layout():
    class EditModel(EchoModel):
        def rewrite(self, text, max_new_tokens):
            result = super().rewrite(text, max_new_tokens)
            result["output"] = "\n" + text.replace("시험", "검사") + "\n"
            return result

    text = "  첫 시험. \r\n\r\n\t두 번째 시험.\n"
    result = rewrite_document(EditModel(), text)
    assert result["output"] == text.replace("시험", "검사")
    assert result["change_kind"] == "changed"


def test_only_truncated_chunk_is_split_and_retried():
    class LimitedModel(EchoModel):
        def rewrite(self, text, max_new_tokens):
            result = super().rewrite(text, max_new_tokens)
            if len(text) > 20:
                result.update(output="잘린 결과", finish_reason="length")
            return result

    model = LimitedModel()
    text = "첫 구간.\n" + "한글" * 30 + "\n마지막 구간."
    result = rewrite_document(model, text)
    assert result["output"] == text
    assert model.calls.count("첫 구간.") == 1
    assert model.calls.count("마지막 구간.") == 1
    assert result["split_retries"] == 3
    assert result["chunks_completed"] == 6


@pytest.mark.parametrize("mode", ["length", "empty", "error", "other"])
def test_failed_segment_never_returns_incomplete_document_as_done(mode):
    class FailingModel(EchoModel):
        def rewrite(self, text, max_new_tokens):
            result = super().rewrite(text, max_new_tokens)
            if text == "정상":
                return result
            if mode == "error":
                raise RuntimeError("sensitive source must not appear in error")
            if mode == "empty":
                result["output"] = "  "
            else:
                result["finish_reason"] = mode
            return result

    model = FailingModel()
    with pytest.raises(DocumentError, match="전체 미완료") as error:
        rewrite_document(model, "정상\n" + "실패할구간" * 30)
    assert "sensitive" not in str(error.value)
    assert len(model.calls) <= 5  # first success, then bounded retry depth


def test_oversized_or_excessively_fragmented_documents_fail_before_generation():
    model = EchoModel()
    with pytest.raises(ValueError):
        rewrite_document(model, "가" * (MAX_CHARACTERS + 1))
    with pytest.raises(DocumentError, match="256"):
        rewrite_document(model, "가\n" * 257)
    assert not model.calls


def test_protection_checked_after_merge_and_per_chunk():
    class UnsafeModel(EchoModel):
        def rewrite(self, text, max_new_tokens):
            result = super().rewrite(text, max_new_tokens)
            result["output"] = text.replace("2026", "2027")
            return result

    result = rewrite_document(UnsafeModel(), "시작\n2026년 11월")
    assert not result["checks"]["mechanical_pass"]
    assert result["chunk_warning_count"] == 1


def test_change_classification_does_not_claim_quality():
    assert change_kind("같은 글", "같은 글") == "unchanged"
    assert change_kind("같은 글", "같은  글") == "whitespace_only"
    assert change_kind("같은 글", "다른 글") == "changed"


def test_chat_template_count_matches_generation_input():
    from korean_humanizer.local_model import LocalModel

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["enable_thinking"] is False
            return "SYSTEM:" + messages[0]["content"] + "USER:" + messages[1]["content"]

        def encode(self, text, add_special_tokens):
            assert add_special_tokens is False
            return list(text)

    model = object.__new__(LocalModel)
    model.tokenizer = Tokenizer()
    model.spec = {"instruction": "원문 의미 보존"}
    assert model.prompt_token_count("본문") == len(model.render_prompt("본문"))
