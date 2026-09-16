from __future__ import annotations

import html
import json
import re
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .models import ShotPlan

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "video-ai/0.2 (+https://github.com/Masek463636/video-ai)"
_MAX_DOWNLOAD_BYTES = 80 * 1024 * 1024
_TAG_RE = re.compile(r"<[^>]+>")
_TOKEN_RE = re.compile(r"[\w-]+", flags=re.UNICODE)


@dataclass(slots=True)
class AssetCandidate:
    title: str
    page_url: str
    download_url: str
    mime: str
    width: int
    height: int
    size: int
    kind: str
    license: str = ""
    license_url: str = ""
    artist: str = ""
    credit: str = ""
    score: float = 0.0


def search_commons(query: str, *, limit: int = 12) -> list[AssetCandidate]:
    """Search Wikimedia Commons for renderable images/videos, no API key required."""
    query = query.strip()
    if not query:
        return []
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": query,
        "gsrnamespace": 6,
        "gsrlimit": max(1, min(limit, 30)),
        "prop": "imageinfo",
        "iiprop": "url|mime|size|extmetadata",
        "iiurlwidth": 1600,
        "format": "json",
        "formatversion": 2,
    }
    payload = _json_get(COMMONS_API + "?" + urllib.parse.urlencode(params))
    pages = payload.get("query", {}).get("pages", [])
    out: list[AssetCandidate] = []
    for page in pages:
        infos = page.get("imageinfo") or []
        if not infos:
            continue
        info = infos[0]
        mime = str(info.get("mime") or "").lower()
        kind = _kind_from_mime(mime)
        if kind is None:
            continue
        size = int(info.get("size") or 0)
        if kind == "video" and size > _MAX_DOWNLOAD_BYTES:
            continue
        url = str(info.get("url") or "")
        if kind == "image":
            url = str(info.get("thumburl") or url)
        if not url:
            continue
        meta = info.get("extmetadata") or {}
        candidate = AssetCandidate(
            title=str(page.get("title") or "").removeprefix("File:"),
            page_url=str(info.get("descriptionurl") or ""),
            download_url=url,
            mime=mime,
            width=int(info.get("width") or 0),
            height=int(info.get("height") or 0),
            size=size,
            kind=kind,
            license=_meta(meta, "LicenseShortName"),
            license_url=_meta(meta, "LicenseUrl"),
            artist=_clean_html(_meta(meta, "Artist")),
            credit=_clean_html(_meta(meta, "Credit")),
        )
        candidate.score = _score(candidate, query)
        out.append(candidate)
    out.sort(key=lambda c: c.score, reverse=True)
    return out


def materialize_assets(
    plan: ShotPlan,
    out_dir: str | Path,
    *,
    limit: int = 12,
    overwrite: bool = False,
) -> list[dict]:
    """Fill blank scene assets from Commons and write a license/source manifest."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    used_urls: set[str] = set()
    manifest: list[dict] = []

    for index, scene in enumerate(plan.scenes):
        if scene.asset and not overwrite:
            p = Path(scene.asset)
            if p.exists():
                manifest.append({"scene": index, "status": "existing", "path": str(p)})
                continue

        chosen: AssetCandidate | None = None
        search_used = ""
        for variant in _query_variants(scene.query, scene.caption):
            candidates = search_commons(variant, limit=limit)
            chosen = next((c for c in candidates if c.download_url not in used_urls), None)
            if chosen:
                search_used = variant
                break

        if not chosen:
            scene.asset = None
            scene.asset_kind = "blank"
            manifest.append({"scene": index, "status": "not_found", "query": scene.query})
            continue

        suffix = _suffix(chosen)
        target = out_dir / f"scene_{index:03d}{suffix}"
        _download(chosen.download_url, target)
        used_urls.add(chosen.download_url)
        scene.asset = str(target.resolve())
        scene.asset_kind = chosen.kind  # type: ignore[assignment]
        manifest.append({
            "scene": index,
            "status": "downloaded",
            "query": scene.query,
            "search_used": search_used,
            "path": str(target),
            **asdict(chosen),
        })

    (out_dir / "assets_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def _json_get(url: str) -> dict:
    request = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=25) as response:
        return json.load(response)


def _download(url: str, target: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response, target.open("wb") as output:
        total = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > _MAX_DOWNLOAD_BYTES:
                raise RuntimeError(
                    f"asset exceeds {_MAX_DOWNLOAD_BYTES // (1024 * 1024)} MB"
                )
            output.write(chunk)


def _kind_from_mime(mime: str) -> str | None:
    if mime in {"image/jpeg", "image/png", "image/webp"}:
        return "image"
    if mime in {"video/webm", "video/mp4", "video/ogg"}:
        return "video"
    return None


def _suffix(candidate: AssetCandidate) -> str:
    path = urllib.parse.urlparse(candidate.download_url).path
    suffix = Path(path).suffix.lower()
    if candidate.kind == "image" and suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        return ".jpg"
    if candidate.kind == "video" and suffix not in {".webm", ".mp4", ".ogv", ".ogg"}:
        return ".webm"
    return suffix or (".jpg" if candidate.kind == "image" else ".webm")


def _score(candidate: AssetCandidate, query: str) -> float:
    q = {t.lower() for t in _TOKEN_RE.findall(query) if len(t) > 1}
    title = {t.lower() for t in _TOKEN_RE.findall(candidate.title) if len(t) > 1}
    overlap = len(q & title)
    score = overlap * 8.0
    if candidate.kind == "image":
        score += 2.0
    if candidate.width >= 1000 and candidate.height >= 700:
        score += 2.0
    if candidate.height > candidate.width:
        score += 1.5
    if candidate.license:
        score += 0.5
    return score


def _query_variants(query: str, caption: str | None) -> Iterable[str]:
    seen: set[str] = set()
    base = [query.strip(), (caption or "").strip()]
    english = _simple_ru_concepts(" ".join(base))
    if english:
        base.append(english)
    for value in base:
        value = re.sub(r"\s+", " ", value).strip()
        if value and value.lower() not in seen:
            seen.add(value.lower())
            yield value


def _simple_ru_concepts(text: str) -> str:
    lowered = text.lower()
    rules = {
        "экзам": "student exam",
        "тест": "student test",
        "универ": "university student",
        "учеб": "student studying",
        "школ": "school student",
        "расстро": "sad student",
        "груст": "sad person",
        "деньг": "money cash",
        "работ": "person working office",
        "телефон": "smartphone",
        "компьют": "computer",
        "машин": "car",
        "игр": "video game",
        "ютуб": "YouTube creator",
        "видео": "video camera",
    }
    concepts = [value for stem, value in rules.items() if stem in lowered]
    return " ".join(dict.fromkeys(concepts))


def _meta(meta: dict, key: str) -> str:
    raw = meta.get(key) or {}
    return str(raw.get("value") or "") if isinstance(raw, dict) else str(raw or "")


def _clean_html(value: str) -> str:
    return html.unescape(_TAG_RE.sub(" ", value)).strip()
