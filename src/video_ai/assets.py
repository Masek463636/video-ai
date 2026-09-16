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
from .openverse import search_openverse
from .vision import detect_focus

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "video-ai/0.7 (+https://github.com/Masek463636/video-ai)"
_MAX_DOWNLOAD_BYTES = 80 * 1024 * 1024
_TAG_RE = re.compile(r"<[^>]+>")
_TOKEN_RE = re.compile(r"[\w-]+", flags=re.UNICODE)
_FALLBACK_MOTIONS = ("zoom_in", "pan_right", "zoom_out", "pan_left")


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
    source: str = "commons"


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
            source="commons",
        )
        candidate.score = _score(candidate, query, query)
        out.append(candidate)
    out.sort(key=lambda c: c.score, reverse=True)
    return out


def search_all(query: str, *, limit: int = 20) -> list[AssetCandidate]:
    pool: dict[str, AssetCandidate] = {}
    try:
        for candidate in search_commons(query, limit=limit):
            pool.setdefault(candidate.download_url, candidate)
    except Exception:
        pass
    try:
        for item in search_openverse(query, limit=limit):
            candidate = AssetCandidate(
                title=item.title,
                page_url=item.page_url,
                download_url=item.download_url,
                mime="image/jpeg",
                width=item.width,
                height=item.height,
                size=0,
                kind="image",
                license=item.license,
                license_url=item.license_url,
                artist=item.artist,
                credit=item.artist,
                description=item.description,
                source="openverse",
            )
            pool.setdefault(candidate.download_url, candidate)
    except Exception:
        pass
    return list(pool.values())


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
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    replace_scenes = replace_scenes or set()
    used_urls: set[str] = set()
    used_titles: list[str] = []
    manifest: list[dict] = []

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
        variants = list(_query_variants(scene))
        english_context = _simple_ru_concepts(f"{scene.query} {scene.caption or ''}")

        for query_index, variant in enumerate(variants):
            for candidate in search_all(variant, limit=limit):
                if candidate.download_url in used_urls:
                    continue
                ranking_query = " ".join(x for x in (variant, english_context) if x).strip()
                score = _score(candidate, ranking_query, english_context)
                score += _visual_mode_bonus(scene, candidate)
                score -= query_index * 0.28
                score -= _repeat_penalty(candidate.title, used_titles)
                if candidate.source == "openverse":
                    score += 0.4
                candidate.score = score
                existing = pool.get(candidate.download_url)
                if existing is None or candidate.score > existing[0].score:
                    pool[candidate.download_url] = (candidate, variant)

        ranked = sorted(pool.values(), key=lambda item: item[0].score, reverse=True)
        if semantic and ranked:
            _semantic_rerank(scene, ranked, top_k=semantic_top_k)
            ranked.sort(key=lambda item: item[0].score, reverse=True)

        chosen: AssetCandidate | None = None
        search_used = ""
        target: Path | None = None
        for candidate, search in ranked[max(0, rank_offset):]:
            suffix = _suffix(candidate)
            candidate_target = out_dir / f"scene_{index:03d}{suffix}"
            try:
                _download(candidate.download_url, candidate_target)
                if not candidate_target.exists() or candidate_target.stat().st_size < 1024:
                    candidate_target.unlink(missing_ok=True)
                    continue
            except Exception:
                candidate_target.unlink(missing_ok=True)
                continue
            chosen = candidate
            search_used = search
            target = candidate_target
            break

        if chosen is None or target is None:
            scene.asset = None
            scene.asset_kind = "blank"
            scene.focus_x = None
            scene.focus_y = None
            scene.focus_source = None
            scene.asset_score = None
            scene.semantic_score = None
            manifest.append({
                "scene": index,
                "status": "not_found",
                "query": scene.query,
                "visual_mode": scene.visual_mode,
                "queries_tried": variants,
            })
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
            "visual_mode": scene.visual_mode,
            "queries_tried": variants,
            "search_used": search_used,
            "path": str(target),
            "source": chosen.source,
            "kind": chosen.kind,
            "focus": {"x": scene.focus_x, "y": scene.focus_y, "source": scene.focus_source},
            "top_candidates": [
                {
                    "title": c.title,
                    "score": round(c.score, 3),
                    "semantic_score": round(c.semantic_score, 4) if c.semantic_score is not None else None,
                    "search": q,
                    "source": c.source,
                    "kind": c.kind,
                }
                for c, q in ranked[:8]
            ],
            **asdict(chosen),
        })

    ensure_visual_coverage(plan, manifest)
    (out_dir / "assets_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def ensure_visual_coverage(plan: ShotPlan, manifest: list[dict] | None = None) -> set[int]:
    good = [
        index
        for index, scene in enumerate(plan.scenes)
        if scene.asset and Path(scene.asset).exists() and scene.asset_kind != "blank"
    ]
    if not good:
        return set()

    filled: set[int] = set()
    by_scene = {
        int(item.get("scene")): item
        for item in (manifest or [])
        if isinstance(item.get("scene"), int)
    }
    for index, scene in enumerate(plan.scenes):
        if scene.asset and Path(scene.asset).exists() and scene.asset_kind != "blank":
            continue
        source_index = min(good, key=lambda other: (abs(other - index), 0 if other < index else 1, other))
        source = plan.scenes[source_index]
        scene.asset = source.asset
        scene.asset_kind = source.asset_kind
        scene.focus_x = source.focus_x
        scene.focus_y = source.focus_y
        scene.focus_source = f"fallback_nearest:{source_index}"
        scene.asset_score = source.asset_score
        scene.semantic_score = source.semantic_score
        scene.motion = _FALLBACK_MOTIONS[index % len(_FALLBACK_MOTIONS)]  # type: ignore[assignment]
        filled.add(index)
        item = by_scene.get(index)
        if item is not None:
            item["status"] = "fallback_nearest"
            item["fallback_from_scene"] = source_index
            item["path"] = scene.asset
    return filled


def _visual_mode_bonus(scene: Scene, candidate: AssetCandidate) -> float:
    if scene.visual_mode == "video":
        return 12.0 if candidate.kind == "video" else -2.5
    if scene.visual_mode == "image":
        return 3.0 if candidate.kind == "image" else -0.5
    if scene.visual_mode == "meme":
        haystack = f"{candidate.title} {candidate.description}".lower()
        reaction = any(word in haystack for word in ("meme", "reaction", "funny", "laugh", "surprise", "face"))
        return 7.0 if reaction else (1.5 if candidate.kind == "image" else 0.0)
    return 0.0


def _semantic_rerank(scene: Scene, ranked: list[tuple[AssetCandidate, str]], *, top_k: int) -> None:
    image_pairs = [(c, q) for c, q in ranked if c.kind == "image"][:max(1, top_k)]
    if not image_pairs:
        return
    try:
        from .multimodal import ClipRanker
        ranker = ClipRanker()
    except Exception:
        return
    prompt = _semantic_prompt(scene)
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
            candidate.score += float(similarity) * 24.0


def _semantic_prompt(scene: Scene) -> str:
    original = f"{scene.query} {scene.caption or ''}".strip()
    concepts = _simple_ru_concepts(original)
    return concepts or (scene.query or "people documentary photo").strip()


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


def _query_variants(scene: Scene) -> Iterable[str]:
    seen: set[str] = set()
    query = re.sub(r"\s+", " ", scene.query).strip()
    caption = re.sub(r"\s+", " ", scene.caption or "").strip()
    full_text = f"{query} {caption}".strip()
    concepts = _simple_ru_concepts(full_text)
    tokens = list(_tokens(full_text))
    keywords = " ".join(tokens[:5])
    broad = " ".join(tokens[:2])

    mode_queries: list[str] = []
    if scene.visual_mode == "video":
        mode_queries = [f"{concepts or query} video", f"{query} footage", f"{broad} motion"]
    elif scene.visual_mode == "meme":
        mode_queries = ["funny reaction face", "surprised reaction", f"{concepts or broad} reaction"]

    base = [
        *mode_queries,
        concepts,
        *_generic_visual_queries(full_text),
        query,
        keywords,
        broad,
        caption,
        "people documentary photo",
    ]
    for value in base:
        value = re.sub(r"\s+", " ", value).strip()
        lowered = value.lower()
        if value and lowered not in seen:
            seen.add(lowered)
            yield value


def _generic_visual_queries(text: str) -> list[str]:
    lowered = text.lower()
    queries: list[str] = []
    rules: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
        (("экзам", "тест", "учеб"), ("student exam", "student classroom", "student studying")),
        (("стресс", "нерв", "расстро", "груст"), ("stressed person", "worried young man", "sad person portrait")),
        (("чиновник", "правитель", "власт"), ("government office", "government official", "historic government building")),
        (("китай", "китайск"), ("China historical", "Chinese people historical", "China city")),
        (("сон", "спал", "снилось"), ("sleeping person dream", "surreal dream", "person sleeping")),
        (("галлюцин", "виден", "увидел"), ("surreal vision", "dream vision", "dramatic portrait")),
        (("иисус", "христ", "религи", "бог"), ("Jesus painting", "religious painting", "Christian art")),
        (("арм", "солдат", "войск"), ("soldiers army historical", "military formation", "historical soldiers")),
        (("восстан", "бунт", "битв"), ("historical rebellion", "battle painting", "crowd uprising")),
        (("деньг", "цена", "миллион", "тысяч"), ("money cash", "finance concept", "counting money")),
        (("телефон", "сообщен", "звон"), ("person using smartphone", "smartphone closeup", "phone notification")),
        (("машин", "авто"), ("car driving", "car road", "car interior")),
        (("компьют", "ютуб", "видео"), ("computer screen creator", "video creator desk", "camera filming")),
        (("дом", "квартир"), ("home interior", "apartment room", "house exterior")),
    )
    for stems, variants in rules:
        if any(stem in lowered for stem in stems):
            queries.extend(variants)
    return list(dict.fromkeys(queries))


def _simple_ru_concepts(text: str) -> str:
    lowered = text.lower()
    rules = {
        "экзам": "student exam classroom",
        "госэкзам": "student civil service exam",
        "чиновник": "government official office",
        "стресс": "stressed worried person",
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
        "китай": "China Chinese historical",
        "сон": "sleeping person dream",
        "галлюцин": "surreal hallucination vision",
        "иисус": "Jesus Christian religious painting",
        "христ": "Jesus Christian religious painting",
        "бог": "religious painting divine vision",
        "арм": "soldiers army historical",
        "солдат": "soldiers army historical",
        "восстан": "historical rebellion uprising",
        "битв": "historical battle painting",
    }
    concepts = [value for stem, value in rules.items() if stem in lowered]
    return " ".join(dict.fromkeys(concepts))


def _meta(meta: dict, key: str) -> str:
    raw = meta.get(key) or {}
    return str(raw.get("value") or "") if isinstance(raw, dict) else str(raw or "")


def _clean_html(value: str) -> str:
    return html.unescape(_TAG_RE.sub(" ", value)).strip()
