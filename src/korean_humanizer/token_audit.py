"""Tokenize exact prepared prompts without loading model weights."""

import hashlib
import json
from pathlib import Path

from korean_humanizer.benchmark import load_bundle
from korean_humanizer.io import sha256_file


def render_chat(
    template: str, body: dict, config: dict, *, add_generation_prompt: bool = True
) -> str:
    from jinja2 import StrictUndefined
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    env = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, undefined=StrictUndefined
    )

    def fail(message):
        raise ValueError(f"chat template rejected messages: {message}")

    env.globals["raise_exception"] = fail
    return env.from_string(template).render(
        messages=body["messages"],
        tools=None,
        documents=None,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=body["chat_template_kwargs"]["enable_thinking"],
        bos_token=config.get("bos_token", ""),
        eos_token=config.get("eos_token", ""),
    )


def audit_tokens(bundle: Path, metadata_dir: Path) -> dict:
    from tokenizers import Tokenizer

    _, _, requests, spec = load_bundle(bundle)
    metadata = json.loads((metadata_dir / "metadata.json").read_text())
    if metadata["model_id"] != spec.model_id or metadata["revision"] != spec.revision:
        raise ValueError("tokenizer model/revision does not match benchmark")
    for name in ("tokenizer.json", "tokenizer_config.json"):
        if sha256_file(metadata_dir / name) != metadata["files"][name]["sha256"]:
            raise ValueError(f"tokenizer integrity failure: {name}")
    tokenizer = Tokenizer.from_file(str(metadata_dir / "tokenizer.json"))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    config = json.loads((metadata_dir / "tokenizer_config.json").read_text())
    results = []
    for request in requests:
        if spec.protocol == "completion":
            prompt = request["body"]["prompt"]
            # Match tokenizer(prompt) defaults, including the model's BOS postprocessor.
            encoded = tokenizer.encode(prompt, add_special_tokens=True)
        else:
            template = config.get("chat_template")
            if not isinstance(template, str):
                raise ValueError("a single official chat template is required")
            prompt = render_chat(template, request["body"], config)
            encoded = tokenizer.encode(prompt, add_special_tokens=False)
        tokens = len(encoded.ids)
        results.append(
            {
                "case_id": request["case_id"],
                "prompt_tokens": tokens,
                "rendered_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "within_context": tokens + spec.max_output_tokens <= spec.context_tokens,
            }
        )
    return {
        "candidate": spec.candidate,
        "revision": spec.revision,
        "bundle_manifest_sha256": sha256_file(bundle / "manifest.json"),
        "tokenizer_sha256": metadata["files"]["tokenizer.json"]["sha256"],
        "cases": results,
        "prompt_tokens_total": sum(r["prompt_tokens"] for r in results),
        "prompt_tokens_max": max(r["prompt_tokens"] for r in results),
        "context_tokens": spec.context_tokens,
        "max_output_tokens": spec.max_output_tokens,
        "all_within_context": all(r["within_context"] for r in results),
        "gpu_compatibility": "not_assessed",
        "model_weights_loaded": False,
    }
