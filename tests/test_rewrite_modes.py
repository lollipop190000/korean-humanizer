from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from test_long_text import EchoModel

from korean_humanizer.local_model import LocalModel
from korean_humanizer.long_text import rewrite_document
from korean_humanizer.rewrite_modes import STRONG_INSTRUCTION


@pytest.mark.parametrize("mode,sampling", [("standard", False), ("strong", True)])
def test_actual_generation_uses_selected_prompt_and_sampling(mode, sampling):
    seen = {}

    class Inputs(dict):
        input_ids = SimpleNamespace(shape=(1, 3))

        def to(self, device):
            return self

    class Generated:
        def __getitem__(self, key):
            assert key == (0, slice(3, None))
            return [123, 2]

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 2

        def apply_chat_template(self, messages, **kwargs):
            seen["instruction"] = messages[0]["content"]
            return messages[0]["content"] + messages[1]["content"]

        def __call__(self, prompt, **kwargs):
            return Inputs()

        def decode(self, tokens, **kwargs):
            return "합성 모델 응답"

    class Model:
        generation_config = SimpleNamespace(eos_token_id=2)

        def generate(self, **kwargs):
            seen["generation"] = kwargs
            return Generated()

    engine = object.__new__(LocalModel)
    engine.tokenizer = Tokenizer()
    engine.model = Model()
    engine.torch = SimpleNamespace(inference_mode=nullcontext)
    engine.spec = {"instruction": "original instruction", "revision": "fixed revision"}
    engine.device, engine.dtype, engine.quantized, engine.adapter = "cpu", "float32", False, True
    result = engine.rewrite("입력", max_new_tokens=1024, rewrite_mode=mode)
    assert seen["instruction"] == (
        STRONG_INSTRUCTION if mode == "strong" else "original instruction"
    )
    assert seen["generation"]["do_sample"] is sampling
    assert seen["generation"]["max_new_tokens"] == 1024
    if sampling:
        assert seen["generation"]["temperature"] == 0.7
    assert result["rewrite_mode"] == mode
    assert result["candidate"] == "local-adapter"
    assert result["human_review_completed"] is False
    assert engine.spec["instruction"] == "original instruction"


def test_strong_prompt_overhead_is_counted_for_every_chunk_and_retry():
    class ModeModel(EchoModel):
        def __init__(self):
            super().__init__()
            self.modes = []

        def prompt_token_count(self, text, rewrite_mode="standard"):
            self.modes.append(rewrite_mode)
            return (600 if rewrite_mode == "strong" else 100) + len(text)

        def rewrite(self, text, max_new_tokens, rewrite_mode="standard"):
            assert rewrite_mode == "strong"
            assert self.prompt_token_count(text, rewrite_mode) <= 1024
            return {
                "output": text,
                "finish_reason": "length" if len(text) > 250 else "stop",
            }

    model = ModeModel()
    source = "한글 입력입니다. " * 100
    result = rewrite_document(model, source, rewrite_mode="strong")
    assert result["output"] == source
    assert result["change_kind"] == "unchanged"  # no forced or fabricated edits
    assert result["split_retries"] > 0
    assert set(model.modes) == {"strong"}


def test_invalid_mode_fails_before_any_model_request():
    model = EchoModel()
    with pytest.raises(ValueError):
        rewrite_document(model, "입력", rewrite_mode="invalid")
    assert not model.calls
