from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass


@dataclass(slots=True)
class GiphyMeme:
    title: str
    page_url: str
    download_url: str
    preview_url: str
    width: int
    height: int
    source: str = "giphy"


def enabled() -> bool:
    """GIPHY media copying is opt-in because API terms restrict caching/copies."""
    key = os.getenv("GIPHY_API_KEY", "").strip()
    approved = os.getenv("GIPHY_MEDIA_CACHE_APPROVED", "").strip().lower() in {"1", "true", "yes"}
    return bool(key and approved)


def search_giphy(query: str, *, limit: int = 8) -> list[GiphyMeme]:
    if not enabled():
        return []
    api_key = os.getenv("GIPHY_API_KEY", "").strip()
    query = " ".join(query.split())[:50]
    if not query:
        return []

    params = urllib.parse.urlencode({
        "api_key": api_key,
        "q": query,
        "limit": max(1, min(limit, 25)),
        "rating": "pg-13",
        "bundle": "messaging_non_clips",
    })
    req = urllib.request.Request(
        "https://api.giphy.com/v1/gifs/search?" + params,
        headers={"User-Agent": "video-ai/2.1.0"},
    )
    with urllib.request.urlopen(req, timeout=20) as response:
        payload = json.load(response)

    out: list[GiphyMeme] = []
    for item in payload.get("data", []):
        images = item.get("images") or {}
        original = images.get("original") or {}
        fixed = images.get("fixed_height") or images.get("fixed_width") or {}
        final_mp4 = str(original.get("mp4") or fixed.get("mp4") or "")
        preview_mp4 = str(fixed.get("mp4") or final_mp4)
        if not final_mp4:
            continue
        out.append(GiphyMeme(
            title=str(item.get("title") or "GIPHY reaction"),
            page_url=str(item.get("url") or ""),
            download_url=final_mp4,
            preview_url=preview_mp4,
            width=int(original.get("width") or fixed.get("width") or 0),
            height=int(original.get("height") or fixed.get("height") or 0),
        ))
    return out
