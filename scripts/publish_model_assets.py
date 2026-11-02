"""Verify an approved model-only transfer and upload it from repository-scoped Actions."""

import hashlib
import json
import os
import subprocess
import tempfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path


def main():
    url = os.environ["MODEL_SOURCE_URL"]
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in {
        "release-assets.githubusercontent.com",
        "objects.githubusercontent.com",
    }:
        raise ValueError("Unexpected model transfer host")
    # This short-lived link grants access only to the approved model package.
    print("::add-mask::" + url, flush=True)
    manifest = json.loads(Path("configs/models/nikl-practical-001.json").read_text())
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        archive = root / "model.zip"
        digest = hashlib.sha256()
        count = 0
        try:
            with urllib.request.urlopen(url, timeout=120) as source, archive.open("wb") as target:
                while chunk := source.read(1024 * 1024):
                    count += len(chunk)
                    if count > 148088066:
                        raise ValueError("Model transfer exceeds expected size")
                    digest.update(chunk)
                    target.write(chunk)
        except Exception:
            raise RuntimeError("Model-only transfer failed; obtain a fresh transfer URL") from None
        if digest.hexdigest() != "534f5aa81e5d1ed2cb47cb66bdf2488300db51896c3ab009b371898c48b43976":
            raise ValueError("Model-only ZIP checksum mismatch")
        with zipfile.ZipFile(archive) as z:
            if set(z.namelist()) != set(manifest["files"]) or len(z.namelist()) != len(
                manifest["files"]
            ):
                raise ValueError("Unexpected model package files")
            paths = []
            for name, info in manifest["files"].items():
                if Path(name).name != name:
                    raise ValueError("Unsafe model package name")
                data = z.read(name)
                if len(data) != info["size"] or hashlib.sha256(data).hexdigest() != info["sha256"]:
                    raise ValueError("Model asset checksum mismatch")
                path = root / name
                path.write_bytes(data)
                paths.append(str(path))
        subprocess.run(
            [
                "gh",
                "release",
                "upload",
                manifest["release_tag"],
                *paths,
                "--repo",
                "lollipop190000/korean-humanizer",
            ],
            check=True,
        )
        print("Uploaded 12 verified model assets. Release remains draft for final verification.")


if __name__ == "__main__":
    main()
