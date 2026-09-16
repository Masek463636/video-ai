from __future__ import annotations

import html
import json
import math
import re
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .models import Scene, ShotPlan
from .vision import detect_focus

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "video-ai/0.3 (+https://github.com/Masek463636/video-ai)"
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
    description: str = ""
    score: float = 0.0


def search_commons(query: str, *, limit: int = 20) -> list[AssetCandidate]:
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
        description = " ".join(
            x for x in (
                _clean_html(_meta(meta, "ImageDescription")),
                _clean_html(_meta(meta, "ObjectName")),
                _clean_html(_meta(meta, "Categories")),
            ) if x
        )
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
            description=description,
        )
        candidate.score = _score(candidate, query, query)
        out.append(candidate)
    out.sort(key=lambda c: c.score, reverse=True)
    return out


def materialize_assets(
    plan: ShotPlan,
    out_dir: str | Path,
    *,
    limit: int = 20,
    overwrite: bool = False,
) -> list[dict]:
    """Resolve each scene from a pool of candidates and persist attribution metadata."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    used_urls: set[str] = set()
    used_titles: list[str] = []
    manifest: list[dict] = []

    for index, scene in enumerate(plan.scenes):
        if scene.asset and not overwrite:
            p = Path(scene.asset)
            if p.exists():
                _apply_focus(scene, p)
                manifest.append({"scene": index, "status": "existing", "path": str(p)})
                continue

        pool: dict[str, tuple[AssetCandidate, str]] = {}
        variants = list(_query_variants(scene.query, scene.caption))
        for query_index, variant in enumerate(variants):
            try:
                candidates = search_commons(variant, limit=limit)
            except Exception:
                continue
            for candidate in candidates:
                if candidate.download_url in used_urls:
                    continue
                score = _score(candidate, scene.query, scene.caption or "")
                score -= query_index * 0.35
                score -= _repeat_penalty(candidate.title, used_titles)
                candidate.score = score
                existing = pool.get(candidate.download_url)
                if existing is None or candidate.score > existing[0].score:
                    pool[candidate.download_url] = (candidate, variant)

        ranked = sorted(pool.values(), key=lambda item: item[0].score, reverse=True)
        chosen_pair = ranked[0] if ranked else None
        if not chosen_pair:
            scene.asset = None
            scene.asset_kind = "blank"
            manifest.append({
                "scene": index,
                "status": "not_found",
                "query": scene.query,
                "queries_tried": variants,
            })
            continue

        chosen, search_used = chosen_pair
        suffix = _suffix(chosen)
        target = out_dir / f"scene_{index:03d}{suffix}"
        try:
            _download(chosen.download_url, target)
        except Exception as exc:
            scene.asset = None
            scene.asset_kind = "blank"
            manifest.append({
                "scene": index,
                "status": "download_failed",
                "query": scene.query,
                "error": str(exc),
            })
            continue

        used_urls.add(chosen.download_url)
        used_titles.append(chosen.title)
        scene.asset = str(target.resolve())
        scene.asset_kind = chosen.kind  # type: ignore[assignment]
        _apply_focus(scene, target)
        manifest.append({
            "scene": index,
            "status": "downloaded",
            "query": scene.query,
            "queries_tried": variants,
            "search_used": search_used,
            "path": str(target),
            "focus": {
                "x": scene.focus_x,
                "y": scene.focus_y,
                "source": scene.focus_source,
            },
            "top_candidates": [
                {"title": c.title, "score": round(c.score, 3), "search": q}
                for c, q in ranked[:5]
            ],
            **asdict(chosen),
        })

    (out_dir / "assets_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def _apply_focus(scene: Scene, path: Path) -> None:
    if scene.asset_kind != "image":
        return
    x, y, source = detect_focus(path)
    scene.focus_x = round(x, 4)
    scene.focus_y = round(y, 4)
    scene.focus_source = source


def _score(candidate: AssetCandidate, query: str, caption: str) -> float:
    wanted = _tokens(query + " " + caption)
    title = _tokens(candidate.title)
    description = _tokens(candidate.description)
    exact = len(wanted & title)
    contextual = len(wanted & description)
    score = exact * 7.0 + contextual * 2.2

    # Reward useful framing/resolution for Shorts rather than just search order.
    if candidate.width >= 1200 or candidate.height >= 1200:
        score += 2.5
    elif candidate.width >= 800 or candidate.height >= 800:
        score += 1.0
    ratio = candidate.width / candidate.height if candidate.height else 1.0
    if 0.48 <= ratio <= 0.75:  # already close to portrait
        score += 3.0
    elif 0.75 < ratio <= 1.45:
        score += 1.5
    elif ratio > 2.2:
        score -= 1.5
    if candidate.kind == "video":
        score += 1.5  # motion is generally more valuable B-roll than a still
    if candidate.license:
        score += 0.5
    if not wanted:
        score += 0.0
    return score


def _repeat_penalty(title: str, previous_titles: list[str]) -> float:
    current = _tokens(title)
    if not current:
        return 0.0
    worst = 0.0
    for previous in previous_titles[-5:]:
        other = _tokens(previous)
        if not other:
            continue
        similarity = len(current & other) / max(1, len(current | other))
        worst = max(worst, similarity)
    return worst * 8.0


def _tokens(value: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(value) if len(t) > 1}


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
                raise RuntimeError(f"asset exceeds {_MAX_DOWNLOAD_BYTES // (1024 * 1024)} MB")
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


def _query_variants(query: str, caption: str | None) -> Iterable[str]:
    """Generate several retrieval views instead of betting the scene on one query."""
    seen: set[str] = set()
    query = re.sub(r"\s+", " ", query).strip()
    caption = re.sub(r"\s+", " ", caption or "").strip()
    concepts = _simple_ru_concepts(query + " " + caption)
    keywords = " ".join(list(_tokens(query + " " + caption))[:6])
    base = [concepts, query, keywords, caption]
    for value in base:
        value = re.sub(r"\s+", " ", value).strip()
        lowered = value.lower()
        if value and lowered not in seen:
            seen.add(lowered)
            yield value


def _simple_ru_concepts(text: str) -> str:
    lowered = text.lower()
    rules = {
        "экзам": "student exam classroom",
        "тест": "student taking test classroom",
        "универ": "university student campus",
        "учеб": "student studying desk",
        "школ": "school student classroom",
        "расстро": "sad disappointed student",
        "груст": "sad disappointed person",
        "радост": "happy excited person",
        "деньг": "money cash finance",
        "работ": "person working office",
        "телефон": "person using smartphone",
        "компьют": "person using computer",
        "машин": "car driving road",
        "игр": "video game player gaming",
        "ютуб": "YouTube creator filming video",
        "видео": "video camera filming",
        "друг": "friends talking together",
        "парень": "young man portrait",
        "девуш": "young woman portrait",
        "отнош": "young couple relationship",
        "страш": "scared person dark",
        "ноч": "night city dark",
        "дом": "home apartment interior",
    }
    concepts = [value for stem, value in rules.items() if stem in lowered]
    return " ".join(dict.fromkeys(concepts))


def _meta(meta: dict, key: str) -> str:
    raw = meta.get(key) or {}
    return str(raw.get("value") or "") if isinstance(raw, dict) else str(raw or "")


def _clean_html(value: str) -> str:
    return html.unescape(_TAG_RE.sub(" ", value)).strip()
