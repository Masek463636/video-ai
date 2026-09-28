from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass


@dataclass(slots=True)
class StockVideo:
    title: str
    page_url: str
    download_url: str
    width: int
    height: int
    duration: float
    source: str
    license: str = ""
    preview_url: str = ""


def search_stock_videos(query: str, *, limit: int = 12) -> list[StockVideo]:
    """Search configured video providers.

    Providers are opt-in through environment variables:
    - PEXELS_API_KEY
    - PIXABAY_API_KEY

    Missing keys are not errors; the app simply falls back to open image sources.
    """
    out: list[StockVideo] = []
    pexels_key = os.getenv("PEXELS_API_KEY", "").strip()
    pixabay_key = os.getenv("PIXABAY_API_KEY", "").strip()
    if pexels_key:
        try:
            out.extend(_search_pexels(query, pexels_key, limit=limit))
        except Exception:
            pass
    if pixabay_key:
        try:
            out.extend(_search_pixabay(query, pixabay_key, limit=limit))
        except Exception:
            pass
    seen: set[str] = set()
    deduped: list[StockVideo] = []
    for item in out:
        if not item.download_url or item.download_url in seen:
            continue
        seen.add(item.download_url)
        deduped.append(item)
    return deduped


def _search_pexels(query: str, api_key: str, *, limit: int) -> list[StockVideo]:
    params = urllib.parse.urlencode({
        "query": query,
                "per_page": max(1, min(limit, 40)),
        "page": 1,
    })
    request = urllib.request.Request(
        "https://api.pexels.com/videos/search?" + params,
        headers={"Authorization": api_key, "User-Agent": "video-ai/2.0.0"},
    )
    with urllib.request.urlopen(request, timeout=25) as response:
        payload = json.load(response)

    results: list[StockVideo] = []
    for video in payload.get("videos", []):
        files = list(video.get("video_files") or [])
        # Prefer vertical-ish HD files but avoid huge 4K downloads for MVP.
        files.sort(key=lambda f: _pexels_file_score(f), reverse=True)
        selected = next((f for f in files if f.get("link")), None)
        if not selected:
            continue
        preview = _smallest_pexels_preview(files) or selected
        results.append(StockVideo(
            title=f"Pexels video {video.get('id', '')}",
            page_url=str(video.get("url") or ""),
            download_url=str(selected.get("link") or ""),
            width=int(selected.get("width") or video.get("width") or 0),
            height=int(selected.get("height") or video.get("height") or 0),
            duration=float(video.get("duration") or 0.0),
            source="pexels",
            license="Pexels License",
            preview_url=str(preview.get("link") or selected.get("link") or ""),
        ))
    return results


def _smallest_pexels_preview(files: list[dict]) -> dict | None:
    candidates = [
        f for f in files
        if f.get("link")
        and int(f.get("width") or 0) > 0
        and int(f.get("height") or 0) > 0
    ]
    if not candidates:
        return None
    # CLIP only needs a small decodable preview. Prefer the lowest-resolution
    # variant to avoid downloading dozens of full HD files during retrieval.
    candidates.sort(
        key=lambda f: (
            int(f.get("width") or 0) * int(f.get("height") or 0),
            max(int(f.get("width") or 0), int(f.get("height") or 0)),
        )
    )
    return candidates[0]


def _pexels_file_score(file: dict) -> float:
    width = int(file.get("width") or 0)
    height = int(file.get("height") or 0)
    if width <= 0 or height <= 0:
        return -1000.0
    ratio = width / height
    portrait_bonus = 8.0 if ratio < 0.8 else 3.0 if ratio < 1.0 else 0.0
    resolution = min(width, 1920) * min(height, 1920) / 1_000_000
    oversize_penalty = 2.0 if max(width, height) > 2160 else 0.0
    return portrait_bonus + resolution - oversize_penalty


def _search_pixabay(query: str, api_key: str, *, limit: int) -> list[StockVideo]:
    params = urllib.parse.urlencode({
        "key": api_key,
        "q": query,
        "video_type": "all",
        "per_page": max(3, min(limit, 50)),
        "safesearch": "true",
    })
    request = urllib.request.Request(
        "https://pixabay.com/api/videos/?" + params,
        headers={"User-Agent": "video-ai/1.5.3"},
    )
    with urllib.request.urlopen(request, timeout=25) as response:
        payload = json.load(response)

    results: list[StockVideo] = []
    for hit in payload.get("hits", []):
        videos = hit.get("videos") or {}
        selected = _best_pixabay_variant(videos)
        if not selected:
            continue
        preview = _smallest_pixabay_variant(videos) or selected
        results.append(StockVideo(
            title=str(hit.get("tags") or f"Pixabay video {hit.get('id', '')}"),
            page_url=str(hit.get("pageURL") or ""),
            download_url=str(selected.get("url") or ""),
            width=int(selected.get("width") or 0),
            height=int(selected.get("height") or 0),
            duration=float(hit.get("duration") or 0.0),
            source="pixabay",
            license="Pixabay Content License",
            preview_url=str(preview.get("url") or selected.get("url") or ""),
        ))
    return results


def _smallest_pixabay_variant(videos: dict) -> dict | None:
    variants = [videos.get(name) for name in ("tiny", "small", "medium", "large")]
    variants = [v for v in variants if isinstance(v, dict) and v.get("url")]
    if not variants:
        return None
    variants.sort(
        key=lambda v: (
            int(v.get("width") or 0) * int(v.get("height") or 0),
            max(int(v.get("width") or 0), int(v.get("height") or 0)),
        )
    )
    return variants[0]


def _best_pixabay_variant(videos: dict) -> dict | None:
    variants = [videos.get(name) for name in ("medium", "small", "large", "tiny")]
    variants = [v for v in variants if isinstance(v, dict) and v.get("url")]
    if not variants:
        return None
    variants.sort(
        key=lambda v: (
            1 if int(v.get("height") or 0) >= int(v.get("width") or 0) else 0,
            int(v.get("width") or 0) * int(v.get("height") or 0),
        ),
        reverse=True,
    )
    return variants[0]
