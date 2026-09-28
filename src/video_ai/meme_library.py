from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
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


@dataclass(slots=True)
class GiphyMemeAsset:
    title: str
    download_url: str
    page_url: str
    preview_url: str
    score: float = 0.0


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


def search_giphy_memes(query: str, *, limit: int = 10) -> list[GiphyMemeAsset]:
    """Search Giphy for meme/reaction MP4s used as primary Style 5 B-roll."""
    api_key = os.getenv("GIPHY_API_KEY", "").strip()
    if not api_key:
        return []

    cleaned = re.sub(
        r"\b(reaction|sticker|gif|meme|funny)\b",
        " ",
        str(query or ""),
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return []

    limit = max(1, min(12, int(limit)))
    url = (
        "https://api.giphy.com/v1/gifs/search"
        f"?api_key={urllib.parse.quote(api_key)}"
        f"&q={urllib.parse.quote(cleaned)}"
        f"&limit={limit + 4}"
        "&rating=pg-13&lang=en"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "video-ai/2.0"})
        with urllib.request.urlopen(req, timeout=10) as response:
            payload = json.load(response)
    except Exception:
        return []

    rows = payload.get("data", []) if isinstance(payload, dict) else []
    if not isinstance(rows, list):
        return []

    out: list[GiphyMemeAsset] = []
    for rank, item in enumerate(rows):
        if not isinstance(item, dict):
            continue
        images = item.get("images")
        if not isinstance(images, dict):
            continue

        mp4_url = ""
        preview_url = ""
        for rendition in ("downsized_medium", "original", "fixed_height"):
            media = images.get(rendition)
            if isinstance(media, dict):
                if not mp4_url and media.get("mp4"):
                    mp4_url = str(media.get("mp4"))
                if not preview_url and media.get("url"):
                    preview_url = str(media.get("url"))
        if not mp4_url:
            continue

        out.append(GiphyMemeAsset(
            title=str(item.get("title") or item.get("slug") or "Giphy reaction").strip(),
            download_url=mp4_url,
            page_url=str(item.get("url") or "").strip(),
            preview_url=preview_url,
            score=max(0.0, 18.0 - rank * 0.8),
        ))
        if len(out) >= limit:
            break
    return out
