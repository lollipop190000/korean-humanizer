"""Run the trained Korean Humanizer adapter; dry-run unless --execute is set."""

import argparse
import json
import sys
from pathlib import Path

from korean_humanizer.model_download import download_model

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
    from korean_humanizer.local_model import LocalModel

    return LocalModel(spec, adapter, device).rewrite(text, max_new_tokens)


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
