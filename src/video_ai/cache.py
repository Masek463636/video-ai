from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


def cache_key(*parts: object) -> str:
    payload = "::".join(str(part) for part in parts)
    return hashlib.sha1(payload.encode("utf-8", errors="replace")).hexdigest()[:24]


def file_signature(path: str | Path) -> str:
    file = Path(path)
    try:
        stat = file.stat()
        return f"{file.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
    except OSError:
        return str(file)


class DiskCache:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def get_json(self, key: str) -> dict[str, Any] | list[Any] | None:
        path = self.root / f"{key}.json"
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return None

    def set_json(self, key: str, value: Any) -> Path:
        path = self.root / f"{key}.json"
        temp = path.with_suffix(".tmp")
        temp.write_text(
            json.dumps(value, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temp.replace(path)
        return path

    def get_file(self, key: str, suffix: str) -> Path | None:
        path = self.root / f"{key}{suffix}"
        return path if path.is_file() and path.stat().st_size > 0 else None

    def put_file(self, key: str, source: str | Path, suffix: str) -> Path:
        source = Path(source)
        target = self.root / f"{key}{suffix}"
        temp = target.with_suffix(target.suffix + ".tmp")
        shutil.copy2(source, temp)
        temp.replace(target)
        return target
