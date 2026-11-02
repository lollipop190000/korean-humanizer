"""Fetch only tokenizer metadata at a pinned revision; verify upstream object hashes."""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

from korean_humanizer.benchmark import BaselineSpec
from korean_humanizer.io import write_json

LIMIT = 40 * 1024 * 1024


def fetch(url: str, limit: int = LIMIT) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError("metadata download exceeded size limit")
    return data


def verify_object(data: bytes, sibling: dict) -> None:
    if sibling.get("lfs"):
        expected = sibling["lfs"]["sha256"]
        actual = hashlib.sha256(data).hexdigest()
    else:
        expected = sibling["blobId"]
        actual = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
    if actual != expected:
        raise ValueError("downloaded tokenizer does not match upstream object hash")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    spec = BaselineSpec.model_validate_json(args.spec.read_bytes())
    if args.output_dir.exists():
        raise ValueError("choose a new metadata output directory")
    url = f"https://huggingface.co/api/models/{spec.model_id}/revision/{spec.revision}?blobs=true"
    info = json.loads(fetch(url, limit=2 * 1024 * 1024))
    if info["sha"] != spec.revision:
        raise ValueError("upstream revision mismatch")
    siblings = {row["rfilename"]: row for row in info["siblings"]}
    downloaded, entries = {}, {}
    for name in ("tokenizer.json", "tokenizer_config.json"):
        data = fetch(f"https://huggingface.co/{spec.model_id}/resolve/{spec.revision}/{name}")
        verify_object(data, siblings[name])
        downloaded[name] = data
        entries[name] = {
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
            "upstream": siblings[name],
        }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, data in downloaded.items():
        with (args.output_dir / name).open("xb") as handle:
            handle.write(data)
    write_json(
        args.output_dir / "metadata.json",
        {"model_id": spec.model_id, "revision": spec.revision, "files": entries},
    )
    print(
        json.dumps(
            {
                "model_id": spec.model_id,
                "revision": spec.revision,
                "bytes": sum(len(value) for value in downloaded.values()),
                "weights_downloaded": False,
            }
        )
    )


if __name__ == "__main__":
    main()
