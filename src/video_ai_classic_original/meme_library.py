from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
_TOKEN_RE = re.compile(r"[\w-]+", flags=re.UNICODE)


@dataclass(slots=True)
class MemeAsset:
    path: Path
    kind: str
    score: float
    title: str


def search_memes(directory: str | Path | None, query: str, *, limit: int = 12) -> list[MemeAsset]:
    if not directory:
        return []
    root = Path(directory)
    if not root.exists() or not root.is_dir():
        return []

    wanted = _tokens(query)
    results: list[MemeAsset] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in _VIDEO_EXTS | _IMAGE_EXTS:
            continue
        name_tokens = _tokens(path.stem.replace("_", " ").replace("-", " "))
        overlap = len(wanted & name_tokens)
        score = overlap * 8.0
        # Meme videos are usually more useful than static images for Shorts.
        kind = "video" if suffix in _VIDEO_EXTS else "image"
        if kind == "video":
            score += 2.5
        # Files in named folders can act like lightweight tags.
        parent_tokens = _tokens(" ".join(p.name for p in path.parents if p != root.parent))
        score += len(wanted & parent_tokens) * 2.0
        results.append(MemeAsset(path=path, kind=kind, score=score, title=path.stem))

    results.sort(key=lambda item: item.score, reverse=True)
    return results[:max(1, limit)]


def _tokens(value: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(value) if len(t) > 1}
