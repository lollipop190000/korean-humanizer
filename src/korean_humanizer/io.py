"""JSONL operations fail on malformed records and refuse accidental overwrites."""

import hashlib
import json
import os
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


def read_jsonl(path: Path, model: type[T]) -> list[T]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank JSONL record")
            try:
                rows.append(model.model_validate_json(line))
            except ValidationError as exc:
                # Do not echo potentially restricted text from Pydantic error payloads.
                fields = [".".join(map(str, error["loc"])) for error in exc.errors()]
                raise ValueError(f"{path}:{line_number}: invalid fields: {fields}") from None
    if not rows:
        raise ValueError(f"{path}: empty dataset")
    return rows


def write_jsonl(path: Path, rows: list[BaseModel | dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        os.chmod(path, 0o600)
        for row in rows:
            value = row.model_dump(mode="json") if isinstance(row, BaseModel) else row
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        os.chmod(path, 0o600)
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
