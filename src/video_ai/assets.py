from __future__ import annotations

import html
import json
import re
import shutil
import tempfile
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .meme_library import search_memes
from .models import Scene, ShotPlan
from .openverse import search_openverse
from .stock_video import search_stock_videos
from .vision import detect_focus

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "video-ai/0.8 (+https://github.com/Masek463636/video-ai)"
_MAX_DOWNLOAD_BYTES = 120 * 1024 * 1024
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
    local_path: str | None = None


def search_commons(query: str, *, limit: int = 20) -> list[AssetCandidate]:
    query = query.strip()
    if not query:
        return []
    params = {
        "action": "query", "generator": "search", "gsrsearch": query,
        "gsrnamespace": 6, "gsrlimit": max(1, min(limit, 30)),
        "prop": "imageinfo", "iiprop": "url|mime|size|extmetadata",
        "iiurlwidth": 1600, "format": "json", "formatversion": 2,
    }
    payload = _json_get(COMMONS_API + "?" + urllib.parse.urlencode(params))
    out: list[AssetCandidate] = []
    for page in payload.get("query", {}).get("pages", []):
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
        url = str(info.get("thumburl") or info.get("url") or "") if kind == "image" else str(info.get("url") or "")
        if not url:
            continue
        meta = info.get("extmetadata") or {}
        description = " ".join(x for x in (
            _clean_html(_meta(meta, "ImageDescription")),
            _clean_html(_meta(meta, "ObjectName")),
            _clean_html(_meta(meta, "Categories")),
        ) if x)
        out.append(AssetCandidate(
            title=str(page.get("title") or "").removeprefix("File:"),
            page_url=str(info.get("descriptionurl") or ""), download_url=url,
            mime=mime, width=int(info.get("width") or 0), height=int(info.get("height") or 0),
            size=size, kind=kind, license=_meta(meta, "LicenseShortName"),
            license_url=_meta(meta, "LicenseUrl"), artist=_clean_html(_meta(meta, "Artist")),
            credit=_clean_html(_meta(meta, "Credit")), description=description, source="commons",
        ))
    return out


def search_all(query: str, *, limit: int = 20, include_stock_video: bool = False) -> list[AssetCandidate]:
    pool: dict[str, AssetCandidate] = {}
    try:
        for c in search_commons(query, limit=limit):
            pool.setdefault(c.download_url, c)
    except Exception:
        pass
    try:
        for item in search_openverse(query, limit=limit):
            c = AssetCandidate(
                title=item.title, page_url=item.page_url, download_url=item.download_url,
                mime="image/jpeg", width=item.width, height=item.height, size=0, kind="image",
                license=item.license, license_url=item.license_url, artist=item.artist,
                credit=item.artist, description=item.description, source="openverse",
            )
            pool.setdefault(c.download_url, c)
    except Exception:
        pass
    if include_stock_video:
        try:
            for item in search_stock_videos(query, limit=min(limit, 16)):
                c = AssetCandidate(
                    title=item.title, page_url=item.page_url, download_url=item.download_url,
                    mime="video/mp4", width=item.width, height=item.height, size=0, kind="video",
                    license=item.license, description=query, source=item.source,
                )
                pool.setdefault(c.download_url, c)
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
    meme_dir: str | Path | None = None,
) -> list[dict]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    replace_scenes = replace_scenes or set()
    used_urls: set[str] = set()
    used_titles: list[str] = []
    manifest: list[dict] = []

    for index, scene in enumerate(plan.scenes):
        if index not in replace_scenes and scene.asset and Path(scene.asset).exists():
            used_titles.append(Path(scene.asset).stem)

    for index, scene in enumerate(plan.scenes):
        force_replace = index in replace_scenes
        if scene.asset and not overwrite and not force_replace and Path(scene.asset).exists():
            _apply_focus(scene, Path(scene.asset))
            manifest.append({"scene": index, "status": "existing", "path": scene.asset})
            continue

        pool: dict[str, tuple[AssetCandidate, str]] = {}
        variants = list(_query_variants(scene))

        if scene.visual_mode == "meme":
            for meme in search_memes(meme_dir, scene.visual_description or scene.query, limit=12):
                key = f"local:{meme.path.resolve()}"
                pool[key] = (AssetCandidate(
                    title=meme.title, page_url="", download_url=key, mime="video/mp4" if meme.kind == "video" else "image/jpeg",
                    width=0, height=0, size=meme.path.stat().st_size, kind=meme.kind,
                    license="local user library", source="local_meme", local_path=str(meme.path.resolve()), score=20.0 + meme.score,
                ), "local meme library")

        for query_index, variant in enumerate(variants):
            include_stock = scene.visual_mode == "video"
            for candidate in search_all(variant, limit=limit, include_stock_video=include_stock):
                if candidate.download_url in used_urls:
                    continue
                ranking_query = " ".join(x for x in (variant, scene.visual_description or "") if x).strip()
                candidate.score = (
                    _score(candidate, ranking_query, scene.visual_description or "")
                    + _visual_mode_bonus(scene, candidate)
                    - query_index * 0.32
                    - _repeat_penalty(candidate.title, used_titles)
                )
                existing = pool.get(candidate.download_url)
                if existing is None or candidate.score > existing[0].score:
                    pool[candidate.download_url] = (candidate, variant)

        ranked = sorted(pool.values(), key=lambda item: item[0].score, reverse=True)
        if semantic and ranked:
            _semantic_rerank(scene, ranked, top_k=semantic_top_k)
            ranked.sort(key=lambda item: item[0].score, reverse=True)

        chosen = None
        target = None
        search_used = ""
        for candidate, search in ranked[max(0, rank_offset):]:
            suffix = _suffix(candidate)
            candidate_target = out_dir / f"scene_{index:03d}{suffix}"
            try:
                if candidate.local_path:
                    shutil.copy2(candidate.local_path, candidate_target)
                else:
                    _download(candidate.download_url, candidate_target)
                if not candidate_target.exists() or candidate_target.stat().st_size < 1024:
                    candidate_target.unlink(missing_ok=True)
                    continue
            except Exception:
                candidate_target.unlink(missing_ok=True)
                continue
            chosen, target, search_used = candidate, candidate_target, search
            break

        if chosen is None or target is None:
            scene.asset = None; scene.asset_kind = "blank"; scene.focus_x = None; scene.focus_y = None
            scene.focus_source = None; scene.asset_score = None; scene.semantic_score = None
            manifest.append({
                "scene": index, "status": "not_found", "query": scene.query,
                "visual_mode": scene.visual_mode, "visual_description": scene.visual_description,
                "queries_tried": variants,
            })
            continue

        used_urls.add(chosen.download_url); used_titles.append(chosen.title)
        scene.asset = str(target.resolve()); scene.asset_kind = chosen.kind  # type: ignore[assignment]
        scene.asset_score = round(chosen.score, 4)
        scene.semantic_score = round(chosen.semantic_score, 4) if chosen.semantic_score is not None else None
        _apply_focus(scene, target)
        manifest.append({
            "scene": index, "status": "downloaded", "query": scene.query,
            "visual_mode": scene.visual_mode, "visual_description": scene.visual_description,
            "queries_tried": variants, "search_used": search_used, "path": str(target),
            "source": chosen.source, "kind": chosen.kind,
            "top_candidates": [{
                "title": c.title, "score": round(c.score, 3), "search": q,
                "source": c.source, "kind": c.kind,
                "semantic_score": round(c.semantic_score,4) if c.semantic_score is not None else None,
            } for c,q in ranked[:8]],
            **asdict(chosen),
        })

    ensure_visual_coverage(plan, manifest)
    (out_dir / "assets_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def ensure_visual_coverage(plan: ShotPlan, manifest: list[dict] | None = None) -> set[int]:
    good = [i for i,s in enumerate(plan.scenes) if s.asset and Path(s.asset).exists() and s.asset_kind != "blank"]
    if not good:
        return set()
    filled: set[int] = set()
    by_scene = {int(item.get("scene")): item for item in (manifest or []) if isinstance(item.get("scene"), int)}
    for index, scene in enumerate(plan.scenes):
        if scene.asset and Path(scene.asset).exists() and scene.asset_kind != "blank":
            continue
        source_index = min(good, key=lambda other: (abs(other-index), 0 if other<index else 1, other))
        source = plan.scenes[source_index]
        scene.asset=source.asset; scene.asset_kind=source.asset_kind; scene.focus_x=source.focus_x; scene.focus_y=source.focus_y
        scene.focus_source=f"fallback_nearest:{source_index}"; scene.asset_score=source.asset_score; scene.semantic_score=source.semantic_score
        scene.motion=_FALLBACK_MOTIONS[index % len(_FALLBACK_MOTIONS)]  # type: ignore[assignment]
        filled.add(index)
        if index in by_scene:
            by_scene[index].update({"status":"fallback_nearest","fallback_from_scene":source_index,"path":scene.asset})
    return filled


def _query_variants(scene: Scene) -> Iterable[str]:
    seen: set[str] = set()
    base = [*(scene.search_queries or []), scene.visual_description or "", scene.query]
    if scene.visual_mode == "video":
        base += [f"{scene.visual_description or scene.query} documentary footage", "historical documentary footage"]
    elif scene.visual_mode == "meme":
        base += [f"{scene.visual_description or scene.query} reaction", "funny reaction face meme"]
    else:
        base += [f"{scene.visual_description or scene.query} archival photo"]
    for value in base:
        value = re.sub(r"\s+", " ", value).strip()
        if value and value.lower() not in seen:
            seen.add(value.lower()); yield value


def _visual_mode_bonus(scene: Scene, candidate: AssetCandidate) -> float:
    if scene.visual_mode == "video":
        return 18.0 if candidate.kind == "video" else -5.0
    if scene.visual_mode == "image":
        return 4.0 if candidate.kind == "image" else -1.0
    if scene.visual_mode == "meme":
        if candidate.source == "local_meme": return 25.0
        haystack=f"{candidate.title} {candidate.description}".lower()
        return 8.0 if any(w in haystack for w in ("meme","reaction","funny","laugh","surprise")) else 0.0
    return 0.0


def _semantic_rerank(scene: Scene, ranked: list[tuple[AssetCandidate,str]], *, top_k: int) -> None:
    image_pairs=[(c,q) for c,q in ranked if c.kind=="image"][:max(1,top_k)]
    if not image_pairs: return
    try:
        from .multimodal import ClipRanker
        ranker=ClipRanker()
    except Exception:
        return
    prompt=(scene.visual_description or scene.query or "documentary scene").strip()
    with tempfile.TemporaryDirectory(prefix="video-ai-clip-") as td:
        paths=[]; valid=[]
        for idx,(candidate,_) in enumerate(image_pairs):
            p=Path(td)/f"candidate_{idx:02d}{_suffix(candidate)}"
            try:
                if candidate.local_path: shutil.copy2(candidate.local_path,p)
                else: _download(candidate.download_url,p)
            except Exception: continue
            paths.append(p); valid.append(candidate)
        if not paths: return
        try: scores=ranker.score_images(prompt,paths)
        except Exception: return
        for candidate,similarity in zip(valid,scores):
            candidate.semantic_score=float(similarity); candidate.score += float(similarity)*28.0


def _apply_focus(scene: Scene, path: Path) -> None:
    if scene.asset_kind != "image": return
    x,y,source=detect_focus(path); scene.focus_x=round(x,4); scene.focus_y=round(y,4); scene.focus_source=source


def _score(candidate: AssetCandidate, query: str, caption: str) -> float:
    wanted=_tokens(query+" "+caption); title=_tokens(candidate.title); description=_tokens(candidate.description)
    score=len(wanted & title)*7.0 + len(wanted & description)*2.5
    if candidate.width>=1200 or candidate.height>=1200: score+=2.5
    elif candidate.width>=800 or candidate.height>=800: score+=1.0
    if candidate.height and candidate.width:
        ratio=candidate.width/candidate.height
        if 0.45<=ratio<=0.8: score+=3.5
        elif ratio<=1.1: score+=1.5
    if candidate.license: score+=0.5
    return score


def _repeat_penalty(title: str, previous_titles: list[str]) -> float:
    current=_tokens(title)
    if not current: return 0.0
    worst=0.0
    for prev in previous_titles[-5:]:
        other=_tokens(prev)
        if other: worst=max(worst,len(current&other)/max(1,len(current|other)))
    return worst*8.0


def _tokens(value: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(value) if len(t)>1}


def _json_get(url: str) -> dict:
    req=urllib.request.Request(url,headers={"User-Agent":USER_AGENT,"Accept":"application/json"})
    with urllib.request.urlopen(req,timeout=25) as response: return json.load(response)


def _download(url: str, target: Path) -> None:
    req=urllib.request.Request(url,headers={"User-Agent":USER_AGENT})
    with urllib.request.urlopen(req,timeout=60) as response, target.open("wb") as output:
        total=0
        while True:
            chunk=response.read(1024*1024)
            if not chunk: break
            total+=len(chunk)
            if total>_MAX_DOWNLOAD_BYTES: raise RuntimeError("asset too large")
            output.write(chunk)


def _kind_from_mime(mime: str) -> str | None:
    if mime in {"image/jpeg","image/png","image/webp"}: return "image"
    if mime in {"video/webm","video/mp4","video/ogg"}: return "video"
    return None


def _suffix(candidate: AssetCandidate) -> str:
    if candidate.local_path:
        suffix=Path(candidate.local_path).suffix.lower()
    else:
        suffix=Path(urllib.parse.urlparse(candidate.download_url).path).suffix.lower()
    if candidate.kind=="image" and suffix not in {".jpg",".jpeg",".png",".webp"}: return ".jpg"
    if candidate.kind=="video" and suffix not in {".webm",".mp4",".ogv",".ogg",".mov",".mkv",".avi"}: return ".mp4"
    return suffix or (".jpg" if candidate.kind=="image" else ".mp4")


def _meta(meta: dict, key: str) -> str:
    raw=meta.get(key) or {}
    return str(raw.get("value") or "") if isinstance(raw,dict) else str(raw or "")


def _clean_html(value: str) -> str:
    return html.unescape(_TAG_RE.sub(" ", value)).strip()
