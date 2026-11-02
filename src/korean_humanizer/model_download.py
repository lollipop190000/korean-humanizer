"""Download and verify the pinned project adapter without falling back to a base model."""

import hashlib
import json
import os
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "configs/models/nikl-practical-001.json"
RELEASE_ROOT = "https://github.com/lollipop190000/korean-humanizer/releases/download/"


def read_manifest(path=MANIFEST):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest["model_id"] != "nikl-practical-001":
        raise ValueError("Unexpected project model")
    if manifest["release_tag"] != "model-nikl-practical-001":
        raise ValueError("Unexpected model release tag")
    for name, info in manifest["files"].items():
        if Path(name).name != name or "/" in name or "\\" in name or name in {".", ".."}:
            raise ValueError("Unsafe model file name")
        if info["size"] <= 0 or len(info["sha256"]) != 64:
            raise ValueError("Invalid model file integrity metadata")
    if not {"adapter_config.json", "adapter_model.safetensors"} <= manifest["files"].keys():
        raise ValueError("Adapter manifest is incomplete")
    return manifest


def verify(path, expected):
    path = Path(path)
    if not path.is_file() or path.stat().st_size != expected["size"]:
        return False
    with path.open("rb") as handle:
        actual = hashlib.file_digest(handle, "sha256").hexdigest()
    return actual == expected["sha256"]


def download_model(output_dir, manifest_path=MANIFEST):
    manifest = read_manifest(manifest_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, expected in manifest["files"].items():
        destination = output_dir / name
        if destination.exists():
            if not verify(destination, expected):
                raise ValueError(f"Checksum mismatch: {destination}; move it aside and retry")
            continue
        url = f"{RELEASE_ROOT}{manifest['release_tag']}/{name}"
        partial = None
        try:
            with tempfile.NamedTemporaryFile(dir=output_dir, delete=False) as target:
                partial = Path(target.name)
                request = urllib.request.Request(url, headers={"User-Agent": "korean-humanizer"})
                with urllib.request.urlopen(request, timeout=120) as response:
                    total = 0
                    while chunk := response.read(1024 * 1024):
                        total += len(chunk)
                        if total > expected["size"]:
                            raise ValueError(f"Download size mismatch for {name}")
                        target.write(chunk)
            if not verify(partial, expected):
                raise ValueError(f"Download checksum mismatch for {name}")
            os.replace(partial, destination)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                f"Model asset {name} unavailable (HTTP {exc.code}). "
                "The trained adapter is required; the base model will not be substituted."
            ) from None
        finally:
            if partial is not None:
                partial.unlink(missing_ok=True)
    return output_dir
