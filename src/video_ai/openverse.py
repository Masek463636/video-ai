from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass

OPENVERSE_API = "https://api.openverse.org/v1/images/"
USER_AGENT = "video-ai/0.4 (+https://github.com/Masek463636/video-ai)"


@dataclass(slots=True)
class OpenverseImage:
    title: str
    page_url: str
    download_url: str
    width: int
    height: int
    license: str = ""
    license_url: str = ""
    artist: str = ""
    description: str = ""
    preview_url: str = ""


def search_openverse(query: str, *, limit: int = 20, prefer_original: bool = False) -> list[OpenverseImage]:
    """Search Openverse anonymously for openly licensed images.

    Openverse supports anonymous API requests. We intentionally keep this
    provider keyless for the MVP and use conservative page sizes.
    """
    query = query.strip()
    if not query:
        return []
    params = {
        "q": query,
        "page_size": max(1, min(int(limit), 20)),
        "mature": "false",
    }
    request = urllib.request.Request(
        OPENVERSE_API + "?" + urllib.parse.urlencode(params),
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=25) as response:
        payload = json.load(response)

    out: list[OpenverseImage] = []
    for raw in payload.get("results", []) or []:
        # Prefer Openverse's thumbnail proxy because upstream originals can be
        # huge, hotlink-protected, or disappear. Fall back to the original URL.
        thumbnail = str(raw.get("thumbnail") or "").strip()
        original = str(raw.get("url") or "").strip()
        url = (original or thumbnail) if prefer_original else (thumbnail or original)
        if not url:
            continue
        title = str(raw.get("title") or "").strip()
        creator = str(raw.get("creator") or "").strip()
        tags = raw.get("tags") or []
        tag_text = " ".join(
            str(tag.get("name") or "") for tag in tags[:12] if isinstance(tag, dict)
        )
        description = " ".join(x for x in (title, creator, tag_text) if x)
        out.append(
            OpenverseImage(
                title=title or "Openverse image",
                page_url=str(raw.get("foreign_landing_url") or raw.get("detail_url") or ""),
                download_url=url,
                width=int(raw.get("width") or 0),
                height=int(raw.get("height") or 0),
                license=str(raw.get("license") or ""),
                license_url=str(raw.get("license_url") or ""),
                artist=creator,
                description=description,
                preview_url=thumbnail or url,
            )
        )
    return out
