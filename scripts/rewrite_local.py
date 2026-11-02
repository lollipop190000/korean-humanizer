"""Run the trained Korean Humanizer adapter; dry-run unless --execute is set."""

import argparse
import json
import sys
from pathlib import Path

from korean_humanizer.model_download import download_model
from korean_humanizer.protection import check_protection

ROOT = Path(__file__).resolve().parents[1]


def prepare(text_file, adapter=None, max_new_tokens=256):
    text = Path(text_file).read_text(encoding="utf-8-sig").strip()
    if not text:
        raise ValueError("Input file is empty")
    if not 1 <= max_new_tokens <= 1024:
        raise ValueError("max-new-tokens must be between 1 and 1024")
    spec = json.loads((ROOT / "configs/baselines/qwen3-4b.json").read_text())
    if adapter:
        adapter = Path(adapter).resolve()
        config = json.loads((adapter / "adapter_config.json").read_text())
        if config.get("base_model_name_or_path") != spec["model_id"]:
            raise ValueError("Adapter must use the same Qwen/Qwen3-4B base model")
        if config.get("revision") not in (None, spec["revision"]):
            raise ValueError("Adapter revision differs from the pinned base revision")
        if not (adapter / "adapter_model.safetensors").is_file():
            raise ValueError("Local adapter_model.safetensors is required")
    return text, spec, adapter


def rewrite(text, spec, adapter, device, max_new_tokens):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA GPU unavailable; select a GPU runtime or use --device cpu")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Apple MPS unavailable")
    dtype = torch.float32 if device == "cpu" else torch.float16
    if device == "cuda" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    if adapter and (adapter / "tokenizer_config.json").is_file():
        tokenizer = AutoTokenizer.from_pretrained(
            str(adapter), local_files_only=True, trust_remote_code=False
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            spec["model_id"], revision=spec["revision"], trust_remote_code=False
        )
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": spec["instruction"]},
            {"role": "user", "content": text},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    count = inputs.input_ids.shape[-1]
    if count > 2048:
        raise ValueError("Input exceeds the demo's 2048-token limit; split the text")
    model = AutoModelForCausalLM.from_pretrained(
        spec["model_id"],
        revision=spec["revision"],
        torch_dtype=dtype,
        device_map={"": device},
        trust_remote_code=False,
        use_safetensors=True,
        attn_implementation="sdpa",
    ).eval()
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model, str(adapter), is_trainable=False, local_files_only=True
        ).eval()
    with torch.inference_mode():
        generated = model.generate(
            **inputs.to(device),
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    tokens = generated[0, count:]
    output = tokenizer.decode(tokens, skip_special_tokens=True)
    eos = model.generation_config.eos_token_id
    eos_ids = eos if isinstance(eos, list) else [eos]
    return {
        "candidate": "local-adapter" if adapter else "qwen3-4b-base",
        "base_revision": spec["revision"],
        "device": device,
        "dtype": str(dtype),
        "output": output,
        "finish_reason": "stop" if int(tokens[-1]) in eos_ids else "length",
        "checks": check_protection(text, output),
        "human_review_completed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-file", type=Path, required=True, help="UTF-8 input file")
    candidate = parser.add_mutually_exclusive_group()
    candidate.add_argument("--adapter", type=Path, help="Your local compatible LoRA adapter folder")
    candidate.add_argument("--base", action="store_true", help="Explicitly use untrained base only")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--execute", action="store_true", help="Download weights and run inference")
    args = parser.parse_args()
    text, spec, adapter = prepare(args.text_file, args.adapter, args.max_new_tokens)
    project_model = not args.base and args.adapter is None
    selected = (
        "nikl-practical-001" if project_model else "local-adapter" if adapter else "qwen3-4b-base"
    )
    if not args.execute:
        print(
            json.dumps(
                {
                    "mode": "dry-run",
                    "model": spec["model_id"],
                    "revision": spec["revision"],
                    "candidate": selected,
                    "device": args.device,
                    "model_downloaded": False,
                    "next": "Add --execute to download model weights and run inference",
                },
                indent=2,
            )
        )
        return
    if project_model:
        print("Downloading/verifying the trained project adapter.", file=sys.stderr)
        adapter = download_model(ROOT / "artifacts/models/nikl-practical-001")
        text, spec, adapter = prepare(args.text_file, adapter, args.max_new_tokens)
    print("Loading model locally; first run downloads several GB of weights.", file=sys.stderr)
    print(
        json.dumps(
            rewrite(text, spec, adapter, args.device, args.max_new_tokens)
            | {"candidate": selected},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
