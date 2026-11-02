"""Create a source-only Windows installer ZIP from tracked files, never model caches."""

import hashlib
import json
import subprocess
import zipfile
from pathlib import Path


def main():
    files = subprocess.check_output(["git", "ls-files", "-z"]).decode().split("\0")
    output = Path("dist/Korean-Humanizer-Windows.zip")
    output.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name in sorted(filter(None, files)):
            if name.split("/")[0] in {".git", ".venv", ".venv-inference", "artifacts", "dist"}:
                raise ValueError("Private/cache file in package manifest")
            if name.endswith((".safetensors", ".bin", ".private.json")):
                raise ValueError("Unexpected binary or private state")
            data = Path(name).read_bytes()
            if name.endswith(".cmd"):
                data = data.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
            z.writestr("Korean-Humanizer/" + name, data)
    with zipfile.ZipFile(output) as z:
        assert z.testzip() is None
        assert "Korean-Humanizer/Install.cmd" in z.namelist()
        assert "Korean-Humanizer/desktop/index.html" in z.namelist()
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    Path("dist/SHA256SUMS.txt").write_text(f"{digest}  {output.name}\n")
    print(json.dumps({"file": str(output), "bytes": output.stat().st_size, "sha256": digest}))


if __name__ == "__main__":
    main()
