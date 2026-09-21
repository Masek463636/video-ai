from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class FreesoundSfx:
    sound_id: int
    name: str
    preview_url: str
    page_url: str
    duration: float
    username: str
    license: str
    score: float = 0.0


def enabled() -> bool:
    key = os.getenv("FREESOUND_API_KEY", "").strip()
    if not key:
        return False
    commercial = os.getenv("VIDEO_AI_COMMERCIAL", "").strip().lower() in {"1", "true", "yes"}
    approved = os.getenv("FREESOUND_COMMERCIAL_APPROVED", "").strip().lower() in {"1", "true", "yes"}
    return (not commercial) or approved


def search_sfx(query: str, *, limit: int = 6) -> list[FreesoundSfx]:
    if not enabled():
        return []
    key = os.getenv("FREESOUND_API_KEY", "").strip()
    params = urllib.parse.urlencode({
        "query": " ".join(query.split())[:120],
        "filter": 'license:"Creative Commons 0" duration:[0.05 TO 5]',
        "fields": "id,name,username,license,previews,duration,url,score",
        "page_size": max(1, min(limit, 12)),
    })
    req = urllib.request.Request(
        "https://freesound.org/apiv2/search/?" + params,
        headers={
            "Authorization": f"Token {key}",
            "User-Agent": "video-ai/2.1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=20) as response:
        payload = json.load(response)

    out: list[FreesoundSfx] = []
    for item in payload.get("results", []):
        previews = item.get("previews") or {}
        preview = str(previews.get("preview-hq-mp3") or previews.get("preview-lq-mp3") or "")
        if not preview:
            continue
        out.append(FreesoundSfx(
            sound_id=int(item.get("id") or 0),
            name=str(item.get("name") or "sound effect"),
            preview_url=preview,
            page_url=str(item.get("url") or ""),
            duration=float(item.get("duration") or 0.0),
            username=str(item.get("username") or ""),
            license=str(item.get("license") or ""),
            score=float(item.get("score") or 0.0),
        ))
    return out


def materialize_best_sfx(query: str, output: str | Path) -> tuple[Path | None, FreesoundSfx | None]:
    results = search_sfx(query, limit=6)
    if not results:
        return None, None
    chosen = results[0]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        chosen.preview_url,
        headers={"User-Agent": "video-ai/2.1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as response, output.open("wb") as handle:
            handle.write(response.read())
    except Exception:
        output.unlink(missing_ok=True)
        return None, None
    if not output.exists() or output.stat().st_size < 1024:
        output.unlink(missing_ok=True)
        return None, None
    return output, chosen
