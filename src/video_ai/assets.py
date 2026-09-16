from __future__ import annotations

import html
import json
import re
import tempfile
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .models import Scene, ShotPlan
from .vision import detect_focus

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "video-ai/0.4 (+https://github.com/Masek463636/video-ai)"
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
    semantic_score: float | None = None


def search_commons(query: str, *, limit: int = 20) -> list[AssetCandidate]:
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
    semantic: bool = False,
    semantic_top_k: int = 6,
    replace_scenes: set[int] | None = None,
    rank_offset: int = 0,
) -> list[dict]:
    """Resolve scene assets with heuristic + optional CLIP reranking.

    `replace_scenes` and `rank_offset` are used by the QC repair pass: failed scenes
    can be re-selected from the next-best candidate without disturbing good scenes.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    replace_scenes = replace_scenes or set()
    used_urls: set[str] = set()
    used_titles: list[str] = []
    manifest: list[dict] = []

    # Existing non-replaced assets count as already used, reducing visual repeats.
    for index, scene in enumerate(plan.scenes):
        if index in replace_scenes:
            continue
        if scene.asset and Path(scene.asset).exists():
            used_titles.append(Path(scene.asset).stem)

    for index, scene in enumerate(plan.scenes):
        force_replace = index in replace_scenes
        if scene.asset and not overwrite and not force_replace:
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
        if semantic and ranked:
            _semantic_rerank(scene, ranked, top_k=semantic_top_k)
            ranked.sort(key=lambda item: item[0].score, reverse=True)

        if not ranked:
            scene.asset = None
            scene.asset_kind = "blank"
            scene.asset_score = None
            scene.semantic_score = None
            manifest.append({
                "scene": index,
                "status": "not_found",
                "query": scene.query,
                "queries_tried": variants,
            })
            continue

        chosen: AssetCandidate | None = None
        search_used = ""
        target: Path | None = None
        # Try candidates in order. If a source fails to download, automatically
        # fall through to the next one instead of leaving a black scene.
        for candidate, search in ranked[max(0, rank_offset):]:
            suffix = _suffix(candidate)
            candidate_target = out_dir / f"scene_{index:03d}{suffix}"
            try:
                _download(candidate.download_url, candidate_target)
            except Exception:
                continue
            chosen = candidate
            search_used = search
            target = candidate_target
            break

        if chosen is None or target is None:
            scene.asset = None
            scene.asset_kind = "blank"
            scene.asset_score = None
            scene.semantic_score = None
            manifest.append({"scene": index, "status": "download_failed", "query": scene.query})
            continue

        used_urls.add(chosen.download_url)
        used_titles.append(chosen.title)
        scene.asset = str(target.resolve())
        scene.asset_kind = chosen.kind  # type: ignore[assignment]
        scene.asset_score = round(chosen.score, 4)
        scene.semantic_score = (
            round(chosen.semantic_score, 4) if chosen.semantic_score is not None else None
        )
        _apply_focus(scene, target)
        manifest.append({
            "scene": index,
            "status": "downloaded",
            "query": scene.query,
            "queries_tried": variants,
            "search_used": search_used,
            "path": str(target),
            "focus": {"x": scene.focus_x, "y": scene.focus_y, "source": scene.focus_source},
            "top_candidates": [
                {
                    "title": c.title,
                    "score": round(c.score, 3),
                    "semantic_score": (
                        round(c.semantic_score, 4) if c.semantic_score is not None else None
                    ),
                    "search": q,
                }
                for c, q in ranked[:8]
            ],
            **asdict(chosen),
        })

    (out_dir / "assets_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def _semantic_rerank(scene: Scene, ranked: list[tuple[AssetCandidate, str]], *, top_k: int) -> None:
    image_pairs = [(c, q) for c, q in ranked if c.kind == "image"][:max(1, top_k)]
    if not image_pairs:
        return
    try:
        from .multimodal import ClipRanker
        ranker = ClipRanker()
    except Exception:
        return

    prompt = (scene.caption or scene.query or "").strip()
    with tempfile.TemporaryDirectory(prefix="video-ai-clip-") as temp_dir:
        paths: list[Path] = []
        valid: list[AssetCandidate] = []
        for idx, (candidate, _) in enumerate(image_pairs):
            path = Path(temp_dir) / f"candidate_{idx:02d}{_suffix(candidate)}"
            try:
                _download(candidate.download_url, path)
            except Exception:
                continue
            paths.append(path)
            valid.append(candidate)
        if not paths:
            return
        try:
            similarities = ranker.score_images(prompt, paths)
        except Exception:
            return
        for candidate, similarity in zip(valid, similarities):
            candidate.semantic_score = float(similarity)
            # CLIP cosine similarities often cluster tightly, so give the semantic
            # signal meaningful weight without deleting the retrieval/composition score.
            candidate.score += float(similarity) * 24.0


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
    if candidate.width >= 1200 or candidate.height >= 1200:
        score += 2.5
    elif candidate.width >= 800 or candidate.height >= 800:
        score += 1.0
    ratio = candidate.width / candidate.height if candidate.height else 1.0
    if 0.48 <= ratio <= 0.75:
        score += 3.0
    elif 0.75 < ratio <= 1.45:
        score += 1.5
    elif ratio > 2.2:
        score -= 1.5
    if candidate.kind == "video":
        score += 1.5
    if candidate.license:
        score += 0.5
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
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
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
