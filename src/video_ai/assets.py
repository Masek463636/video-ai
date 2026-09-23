from __future__ import annotations

import hashlib
import html
import json
import re
import shutil
import subprocess
import tempfile
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .meme_library import search_memes
from .models import Scene, ShotPlan
from .openverse import search_openverse
from .quality_guard import infer_tone, local_quality_guard
from .stock_video import search_stock_videos
from .vision import detect_focus

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "video-ai/2.0.0 (+https://github.com/Masek463636/video-ai)"
_MAX_DOWNLOAD_BYTES = 120 * 1024 * 1024
_TAG_RE = re.compile(r"<[^>]+>")
_TOKEN_RE = re.compile(r"[\w-]+", flags=re.UNICODE)
_DARK_TONES = {"tragic", "negative", "violent", "tense", "shocking"}


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
    preview_url: str = ""


@dataclass(slots=True)
class MaterialRegistry:
    """Persistent source usage across initial selection and every repair pass."""
    url_counts: dict[str, int] = None  # type: ignore[assignment]
    title_counts: dict[str, int] = None  # type: ignore[assignment]
    scene_urls: dict[int, str] = None  # type: ignore[assignment]
    scene_titles: dict[int, str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.url_counts = {} if self.url_counts is None else self.url_counts
        self.title_counts = {} if self.title_counts is None else self.title_counts
        self.scene_urls = {} if self.scene_urls is None else self.scene_urls
        self.scene_titles = {} if self.scene_titles is None else self.scene_titles

    def release_scene(self, index: int) -> None:
        url = self.scene_urls.pop(index, None)
        title = self.scene_titles.pop(index, None)
        if url:
            left = self.url_counts.get(url, 0) - 1
            if left > 0:
                self.url_counts[url] = left
            else:
                self.url_counts.pop(url, None)
        if title:
            key = title.casefold().strip()
            left = self.title_counts.get(key, 0) - 1
            if left > 0:
                self.title_counts[key] = left
            else:
                self.title_counts.pop(key, None)

    def register_scene(self, index: int, url: str, title: str) -> None:
        self.release_scene(index)
        self.scene_urls[index] = url
        self.scene_titles[index] = title
        self.url_counts[url] = self.url_counts.get(url, 0) + 1
        key = title.casefold().strip()
        if key:
            self.title_counts[key] = self.title_counts.get(key, 0) + 1

    def used_urls(self) -> set[str]:
        return set(self.url_counts)

    def used_titles(self) -> list[str]:
        return list(self.title_counts)


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
        description = " ".join(
            x for x in (
                _clean_html(_meta(meta, "ImageDescription")),
                _clean_html(_meta(meta, "ObjectName")),
                _clean_html(_meta(meta, "Categories")),
            ) if x
        )
        out.append(AssetCandidate(
            str(page.get("title") or "").removeprefix("File:"),
            str(info.get("descriptionurl") or ""), url, mime,
            int(info.get("width") or 0), int(info.get("height") or 0), size, kind,
            _meta(meta, "LicenseShortName"), _meta(meta, "LicenseUrl"),
            _clean_html(_meta(meta, "Artist")), _clean_html(_meta(meta, "Credit")),
            description, source="commons",
        ))
    return out


def search_all(query: str, *, limit: int = 20, include_stock_video: bool = False, archive_only: bool = False) -> list[AssetCandidate]:
    pool: dict[str, AssetCandidate] = {}
    try:
        for c in search_commons(query, limit=limit):
            pool.setdefault(c.download_url, c)
    except Exception:
        pass
    try:
        for item in search_openverse(query, limit=limit):
            c = AssetCandidate(
                item.title, item.page_url, item.download_url, "image/jpeg",
                item.width, item.height, 0, "image", item.license, item.license_url,
                item.artist, item.artist, item.description, source="openverse",
            )
            pool.setdefault(c.download_url, c)
    except Exception:
        pass
    if include_stock_video and not archive_only:
        try:
            for item in search_stock_videos(query, limit=min(max(limit, 24), 40)):
                c = AssetCandidate(
                    item.title, item.page_url, item.download_url, "video/mp4",
                    item.width, item.height, 0, "video", item.license,
                    description=query, source=item.source, preview_url=item.preview_url,
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
    registry: MaterialRegistry | None = None,
    allow_coverage_reuse: bool = True,
    judge_with_gemini: bool = True,
) -> list[dict]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    replace_scenes = replace_scenes or set()
    registry = registry or MaterialRegistry()
    for scene_index in replace_scenes:
        registry.release_scene(scene_index)
    used_urls = registry.used_urls()
    used_titles = registry.used_titles()
    manifest: list[dict] = []
    total_scenes = len(plan.scenes)

    gemini = None
    if judge_with_gemini:
        try:
            from .gemini_ai import get_gemini_client
            gemini = get_gemini_client()
        except Exception:
            gemini = None

    soft_story_context = _infer_soft_story_context(plan)
    for scene in plan.scenes:
        scene.tone = infer_tone(scene.caption)
        scene.required_context = _local_required_context(scene)
        if soft_story_context and not scene.semantic_lock and scene.visual_mode != "meme":
            marker = "Soft historical setting:"
            if marker not in (scene.visual_description or ""):
                base = (scene.visual_description or scene.query or "documentary scene").rstrip(" .")
                scene.visual_description = f"{base}. {marker} {soft_story_context}."

    if soft_story_context:
        print(f"[context] soft story setting: {soft_story_context}", flush=True)

    # Whole-roll Visual Director: one Gemini request for all unresolved donor
    # stock-video scenes. This replaces the old one-request-per-scene v2 path.
    v2_batch: dict[int, tuple[AssetCandidate, str, dict, list[str], int]] = {}
    if plan.director_source == "donor_gemini":
        batch_indexes = [
            index for index, scene in enumerate(plan.scenes)
            if (
                scene.visual_mode == "video"
                and scene.source_mode == "stock_video"
                and not scene.semantic_lock
                and (
                    overwrite
                    or index in replace_scenes
                    or not scene.asset
                    or not Path(scene.asset).exists()
                )
            )
        ]
        if batch_indexes:
            v2_batch = _v2_select_roll_batch(
                plan,
                batch_indexes,
                gemini=gemini,
                used_urls=used_urls,
                prefix="[v2-roll]",
            )

    for index, scene in enumerate(plan.scenes):
        prefix = f"[{index + 1}/{total_scenes}]"
        force_replace = index in replace_scenes
        print(
            f"{prefix} search | tone={scene.tone} | {scene.visual_mode}/{scene.source_mode}"
            f" | query={scene.query}",
            flush=True,
        )

        if scene.asset and not overwrite and not force_replace and Path(scene.asset).exists():
            _apply_focus(scene, Path(scene.asset))
            manifest.append({"scene": index, "status": "existing", "path": scene.asset, "tone": scene.tone, "soft_story_context": soft_story_context})
            print(f"{prefix} existing asset", flush=True)
            continue

        use_v2_stock = (
            plan.director_source == "donor_gemini"
            and scene.visual_mode == "video"
            and scene.source_mode == "stock_video"
            and not scene.semantic_lock
        )
        if use_v2_stock:
            batch_choice = v2_batch.get(index)
            if batch_choice is None:
                scene.asset = None
                scene.asset_kind = "blank"
                scene.focus_x = scene.focus_y = None
                scene.focus_source = None
                scene.asset_score = scene.semantic_score = None
                manifest.append({
                    "scene": index,
                    "status": "v2_not_found",
                    "query": scene.query,
                    "tone": scene.tone,
                    "visual_mode": scene.visual_mode,
                    "source_mode": scene.source_mode,
                    "queries_tried": list(scene.search_queries or [scene.query]),
                    "v2_previews": 0,
                })
                print(f"{prefix} [v2-roll] no usable batch choice; local rescue later", flush=True)
                continue

            chosen_v2, search_v2, info_v2, queries_v2, previews_v2 = batch_choice
            target_v2 = out_dir / f"_scene_{index:03d}_v2{_suffix(chosen_v2)}"
            try:
                _download(chosen_v2.download_url, target_v2)
            except Exception:
                target_v2.unlink(missing_ok=True)
                scene.asset = None
                scene.asset_kind = "blank"
                manifest.append({
                    "scene": index,
                    "status": "v2_download_failed",
                    "query": scene.query,
                    "queries_tried": queries_v2,
                    "v2_previews": previews_v2,
                })
                print(f"{prefix} [v2-roll] selected candidate download failed", flush=True)
                continue

            guard = local_quality_guard(
                target_v2,
                title=chosen_v2.title,
                description=chosen_v2.description,
                width=chosen_v2.width,
                height=chosen_v2.height,
                kind=chosen_v2.kind,
            )
            if guard.score < 38 or _has_severe_local_issue(guard.issues):
                target_v2.unlink(missing_ok=True)
                scene.asset = None
                scene.asset_kind = "blank"
                manifest.append({
                    "scene": index,
                    "status": "v2_local_quality_reject",
                    "query": scene.query,
                    "queries_tried": queries_v2,
                    "v2_previews": previews_v2,
                    "local_quality_rejected": [{
                        "title": chosen_v2.title,
                        "source": chosen_v2.source,
                        "quality_score": guard.score,
                        "issues": guard.issues,
                    }],
                })
                print(f"{prefix} [v2-roll] local quality rejected batch choice", flush=True)
                continue

            final_target = out_dir / f"scene_{index:03d}{target_v2.suffix.lower()}"
            if final_target != target_v2:
                final_target.unlink(missing_ok=True)
                target_v2.replace(final_target)
            _cleanup_scene_attempts(out_dir, index, keep=final_target)

            registry.register_scene(index, chosen_v2.download_url, chosen_v2.title)
            used_urls = registry.used_urls()
            used_titles = registry.used_titles()
            scene.asset = str(final_target.resolve())
            scene.asset_kind = "video"
            scene.asset_score = round(chosen_v2.score, 4)
            scene.semantic_score = round(chosen_v2.semantic_score or 0.0, 4)
            _apply_focus(scene, final_target)
            manifest.append({
                "scene": index,
                "status": "v2_roll_visual_director",
                "query": scene.query,
                "tone": scene.tone,
                "visual_mode": scene.visual_mode,
                "source_mode": scene.source_mode,
                "queries_tried": queries_v2,
                "search_used": search_v2,
                "path": str(final_target),
                "source": chosen_v2.source,
                "kind": chosen_v2.kind,
                "gemini_judge": info_v2,
                "v2_previews": previews_v2,
                **asdict(chosen_v2),
            })
            print(
                f"{prefix} [v2-roll] selected {chosen_v2.title[:55]} | fit={info_v2.get('score')}",
                flush=True,
            )
            continue

        ranked, variants = _build_ranked_pool(
            scene, limit=limit, meme_dir=meme_dir, used_urls=used_urls,
            used_titles=used_titles, semantic=semantic, semantic_top_k=semantic_top_k,
        )
        print(f"{prefix} candidates={len(ranked)}" + (" | deep visual CLIP ranked" if semantic else ""), flush=True)

        chosen, target, search_used, judge_info, rejected, local_rejected, checked = _select_best_candidate(
            scene, ranked[max(0, rank_offset):], out_dir, index=index, gemini=gemini,
            max_checks=5, prefix=prefix, match_level="exact",
        )

        recovery_used = False
        if chosen is None and not scene.semantic_lock:
            previous_queries = list(variants)
            rewritten: list[str] = []
            if gemini is not None:
                try:
                    rewritten = gemini.rewrite_search_queries(
                        scene,
                        rejected=rejected,
                        previous_queries=previous_queries,
                        mode="close",
                    )
                except Exception:
                    rewritten = []

            if rewritten:
                print(
                    f"{prefix} close fallback: " + " || ".join(rewritten),
                    flush=True,
                )
                original_queries = list(scene.search_queries)
                original_query = scene.query
                scene.search_queries = rewritten + [
                    q for q in original_queries if q.casefold() not in {x.casefold() for x in rewritten}
                ]
                scene.query = scene.search_queries[0]
                rejected_urls = {
                    str(item.get("download_url") or "")
                    for item in rejected
                    if item.get("download_url")
                }
                recovery_ranked, recovery_variants = _build_ranked_pool(
                    scene,
                    limit=max(limit, 24),
                    meme_dir=meme_dir,
                    used_urls=used_urls | rejected_urls,
                    used_titles=used_titles,
                    semantic=semantic,
                    semantic_top_k=max(semantic_top_k, 36),
                )
                print(
                    f"{prefix} close candidates={len(recovery_ranked)}"
                    + (" | deep visual CLIP ranked" if semantic else ""),
                    flush=True,
                )
            else:
                print(f"{prefix} recovery search: concrete current-beat fallback", flush=True)
                recovery_ranked = _build_recovery_pool(
                    scene,
                    limit=max(12, min(limit, 20)),
                    used_urls=used_urls | {
                        str(item.get("download_url") or "")
                        for item in rejected
                        if item.get("download_url")
                    },
                    used_titles=used_titles,
                    semantic=semantic,
                )

            chosen, target, search_used, recovery_judge, recovery_rejected, recovery_local, recovery_checked = _select_best_candidate(
                scene, recovery_ranked, out_dir, index=index, gemini=gemini,
                max_checks=4, prefix=prefix, attempt_offset=checked, match_level="close",
            )
            rejected.extend(recovery_rejected)
            local_rejected.extend(recovery_local)
            checked += recovery_checked
            if recovery_judge is not None:
                judge_info = recovery_judge
            recovery_used = chosen is not None

        # Final ladder level: broad contextual B-roll. The exact gesture/object
        # no longer needs to be present, but the footage must honestly support
        # the narration topic and remain usable.
        if chosen is None and not scene.semantic_lock:
            previous_queries = list(_query_variants(scene))
            context_queries: list[str] = []
            if gemini is not None:
                try:
                    context_queries = gemini.rewrite_search_queries(
                        scene,
                        rejected=rejected,
                        previous_queries=previous_queries,
                        mode="context",
                    )
                except Exception:
                    context_queries = []
            if context_queries:
                print(
                    f"{prefix} context fallback: " + " || ".join(context_queries),
                    flush=True,
                )
                scene.search_queries = context_queries
                scene.query = context_queries[0]
                rejected_urls = {
                    str(item.get("download_url") or "")
                    for item in rejected
                    if item.get("download_url")
                }
                context_ranked, _ = _build_ranked_pool(
                    scene,
                    limit=max(limit, 20),
                    meme_dir=meme_dir,
                    used_urls=used_urls | rejected_urls,
                    used_titles=used_titles,
                    semantic=semantic,
                    semantic_top_k=max(semantic_top_k, 24),
                )
            else:
                context_ranked = _build_recovery_pool(
                    scene,
                    limit=max(12, min(limit, 18)),
                    used_urls=used_urls | {
                        str(item.get("download_url") or "")
                        for item in rejected
                        if item.get("download_url")
                    },
                    used_titles=used_titles,
                    semantic=semantic,
                )
            print(
                f"{prefix} context candidates={len(context_ranked)}"
                + (" | deep visual CLIP ranked" if semantic else ""),
                flush=True,
            )
            chosen, target, search_used, context_judge, context_rejected, context_local, context_checked = _select_best_candidate(
                scene,
                context_ranked,
                out_dir,
                index=index,
                gemini=gemini,
                max_checks=3,
                prefix=prefix,
                attempt_offset=checked,
                match_level="context",
            )
            rejected.extend(context_rejected)
            local_rejected.extend(context_local)
            checked += context_checked
            if context_judge is not None:
                judge_info = context_judge
            recovery_used = chosen is not None

        if chosen is None or target is None:
            scene.asset = None
            scene.asset_kind = "blank"
            scene.focus_x = scene.focus_y = None
            scene.focus_source = None
            scene.asset_score = scene.semantic_score = None
            status = "not_found_locked" if scene.semantic_lock else "not_found_recovery"
            manifest.append({
                "scene": index, "status": status, "query": scene.query, "tone": scene.tone,
                "visual_mode": scene.visual_mode, "source_mode": scene.source_mode,
                "semantic_lock": scene.semantic_lock, "required_entities": scene.required_entities,
                "required_context": scene.required_context, "semantic_fallback": scene.semantic_fallback,
                "motion_preset": scene.motion_preset, "visual_description": scene.visual_description,
                "soft_story_context": soft_story_context, "queries_tried": variants,
                "local_quality_rejected": local_rejected, "gemini_rejected": rejected,
                "gemini_checks": checked,
            })
            print(f"{prefix} no safe candidate -> coverage recovery later", flush=True)
            continue

        final_target = out_dir / f"scene_{index:03d}{target.suffix.lower()}"
        if final_target != target:
            final_target.unlink(missing_ok=True)
            target.replace(final_target)
        _cleanup_scene_attempts(out_dir, index, keep=final_target)

        registry.register_scene(index, chosen.download_url, chosen.title)
        used_urls = registry.used_urls()
        used_titles = registry.used_titles()
        scene.asset = str(final_target.resolve())
        scene.asset_kind = chosen.kind  # type: ignore[assignment]
        scene.asset_score = round(chosen.score, 4)
        scene.semantic_score = round(chosen.semantic_score, 4) if chosen.semantic_score is not None else None
        _apply_focus(scene, final_target)
        manifest.append({
            "scene": index, "status": "downloaded_recovery" if recovery_used else "downloaded",
            "query": scene.query, "tone": scene.tone, "visual_mode": scene.visual_mode,
            "source_mode": scene.source_mode, "semantic_lock": scene.semantic_lock,
            "required_entities": scene.required_entities, "required_context": scene.required_context,
            "semantic_fallback": scene.semantic_fallback, "motion_preset": scene.motion_preset,
            "meme_filename": scene.meme_filename, "visual_description": scene.visual_description,
            "soft_story_context": soft_story_context, "queries_tried": variants,
            "search_used": search_used, "path": str(final_target),
            "source": chosen.source, "kind": chosen.kind, "gemini_judge": judge_info,
            "local_quality_rejected": local_rejected, "gemini_rejected": rejected,
            "gemini_checks": checked, "recovery_used": recovery_used,
            "top_candidates": [
                {"title": c.title, "score": round(c.score, 3), "search": q, "source": c.source,
                 "kind": c.kind, "semantic_score": round(c.semantic_score, 4) if c.semantic_score is not None else None}
                for c, q in ranked[:8]
            ],
            **asdict(chosen),
        })
        if judge_info:
            print(
                f"{prefix} selected | semantic={judge_info.get('score')} tone={judge_info.get('tone_match')} quality={judge_info.get('quality_score')}",
                flush=True,
            )
        else:
            print(f"{prefix} selected locally", flush=True)

    filled: set[int] = set()
    if allow_coverage_reuse:
        filled = ensure_visual_coverage(plan, manifest)
        if filled:
            print(f"[coverage] recovered {len(filled)} empty scene(s) without black frames", flush=True)
    else:
        unresolved = [
            i for i, scene in enumerate(plan.scenes)
            if not scene.asset or not Path(scene.asset).exists() or scene.asset_kind == "blank"
        ]
        if unresolved:
            print(f"[coverage] donor mode: reuse disabled; unresolved scenes={unresolved}", flush=True)
    (out_dir / "assets_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def _infer_soft_story_context(plan: ShotPlan) -> str:
    text = " ".join((scene.caption or "") for scene in plan.scenes).lower()
    parts: list[str] = []
    if any(token in text for token in ("тайпин", "сюцюан", "hong xiu", "taiping")):
        parts.extend(["19th-century China", "Taiping Rebellion era"])
    elif any(token in text for token in ("китай", "china", "chinese")):
        parts.append("China")
    if any(token in text for token in ("династия цин", "династии цин", "qing dynasty")):
        parts.append("Qing dynasty")
    return ", ".join(dict.fromkeys(parts))


def _local_required_context(scene: Scene) -> list[str]:
    caption = (scene.caption or "").lower()
    out: list[str] = []
    for context in scene.required_context:
        if _context_explicit_in_caption(context, caption):
            out.append(context)
    return list(dict.fromkeys(out))


def _context_explicit_in_caption(context: str, caption: str) -> bool:
    lowered = context.lower()
    if "china" in lowered or "chinese" in lowered:
        return any(token in caption for token in ("китай", "китайск", "china", "chinese"))
    if "qing" in lowered:
        return any(token in caption for token in ("цин", "qing"))
    if "19th century" in lowered:
        return bool(re.search(r"\b18\d{2}\b|\b19\s*(?:-|‑)?\s*(?:й|ый)?\s*век|\bxix\b", caption))
    if "20th century" in lowered:
        return bool(re.search(r"\b19\d{2}\b|\b20\s*(?:-|‑)?\s*(?:й|ый)?\s*век|\bxx\b", caption))
    if "21st century" in lowered:
        return bool(re.search(r"\b20\d{2}\b|\b21\s*(?:-|‑)?\s*(?:й|ый)?\s*век|\bxxi\b", caption))
    tokens = _tokens(context)
    caption_tokens = _tokens(caption)
    return bool(tokens) and tokens.issubset(caption_tokens)



def _v2_stock_candidates(
    queries: list[str],
    *,
    used_urls: set[str],
    max_candidates: int = 24,
    per_query: int = 8,
) -> list[tuple[AssetCandidate, str]]:
    """Query stock providers directly and keep a small query-balanced pool.

    v2 intentionally skips Commons/Openverse and skips CLIP. Provider relevance
    gives us the shortlist; Gemini compares the actual preview frames together.
    """
    out: list[tuple[AssetCandidate, str]] = []
    seen = set(used_urls)
    for query in queries[:3]:
        added = 0
        try:
            items = search_stock_videos(query, limit=max(per_query, 8))
        except Exception:
            items = []
        buckets: dict[str, list[Any]] = {}
        source_order: list[str] = []
        for item in items:
            if not item.download_url or item.download_url in seen:
                continue
            if item.source not in buckets:
                buckets[item.source] = []
                source_order.append(item.source)
            buckets[item.source].append(item)

        cursor = 0
        while added < per_query and len(out) < max_candidates and source_order:
            progressed = False
            for source in source_order:
                bucket = buckets[source]
                if cursor >= len(bucket):
                    continue
                item = bucket[cursor]
                if item.download_url in seen:
                    continue
                seen.add(item.download_url)
                out.append((
                    AssetCandidate(
                        item.title,
                        item.page_url,
                        item.download_url,
                        "video/mp4",
                        item.width,
                        item.height,
                        0,
                        "video",
                        item.license,
                        description=query,
                        source=item.source,
                        preview_url=item.preview_url,
                    ),
                    query,
                ))
                added += 1
                progressed = True
                if added >= per_query or len(out) >= max_candidates:
                    break
            if not progressed:
                break
            cursor += 1
        if len(out) >= max_candidates:
            break
    return out


def _v2_extract_preview(candidate: AssetCandidate, root: Path, index: int) -> Path | None:
    video_path = root / f"v2_{index:03d}{_suffix(candidate)}"
    frame_path = root / f"v2_{index:03d}.jpg"
    try:
        _download(candidate.preview_url or candidate.download_url, video_path)
        if not video_path.exists() or video_path.stat().st_size < 1024:
            return None
        for sec in (0.8, 1.8, 0.15):
            completed = subprocess.run(
                [
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-ss", str(sec), "-i", str(video_path),
                    "-frames:v", "1", "-vf", "scale=512:-2",
                    str(frame_path),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=15,
                check=False,
            )
            if completed.returncode == 0 and frame_path.exists() and frame_path.stat().st_size >= 1024:
                return frame_path
    except Exception:
        return None
    finally:
        video_path.unlink(missing_ok=True)
    return None


def _v2_select_roll_batch(
    plan: ShotPlan,
    indexes: list[int],
    *,
    gemini: Any,
    used_urls: set[str],
    prefix: str,
) -> dict[int, tuple[AssetCandidate, str, dict, list[str], int]]:
    """Local-first Material Brain 2 visual selector.

    Search up to nine balanced stock candidates per scene and score their ACTUAL
    preview frames with local CLIP. Gemini is optional refinement only. This path
    also runs when --material-v2-local is used, so local mode does not silently
    fall back to weak metadata-only selection.
    """
    candidate_maps: dict[int, dict[int, tuple[AssetCandidate, str]]] = {}
    rows_by_scene: dict[int, list[dict[str, Any]]] = {}
    queries_by_scene: dict[int, list[str]] = {}
    preview_counts: dict[int, int] = {}

    with tempfile.TemporaryDirectory(prefix="video-ai-v2-roll-") as d:
        root = Path(d)

        for scene_index in indexes:
            scene = plan.scenes[scene_index]
            base = [q for q in (scene.search_queries or []) if q.strip()]
            if scene.query:
                base.append(scene.query)

            queries: list[str] = []
            seen_q: set[str] = set()
            for raw in base:
                q = re.sub(r"\s+", " ", str(raw)).strip()
                key = q.casefold()
                if q and key not in seen_q:
                    seen_q.add(key)
                    queries.append(q)
                if len(queries) >= 3:
                    break
            queries_by_scene[scene_index] = queries
            if not queries:
                continue

            pairs = _v2_stock_candidates(
                queries,
                used_urls=used_urls,
                max_candidates=18,
                per_query=6,
            )
            if not pairs:
                continue

            # Round-robin across query variants so each scene gets genuinely
            # different visual ideas instead of 4 nearly identical provider hits.
            buckets: dict[str, list[tuple[AssetCandidate, str]]] = {}
            order: list[str] = []
            for candidate, search in pairs:
                if search not in buckets:
                    buckets[search] = []
                    order.append(search)
                buckets[search].append((candidate, search))

            balanced: list[tuple[AssetCandidate, str]] = []
            cursor = 0
            while len(balanced) < 9 and order:
                progressed = False
                for search in order:
                    bucket = buckets.get(search) or []
                    if cursor < len(bucket):
                        balanced.append(bucket[cursor])
                        progressed = True
                        if len(balanced) >= 9:
                            break
                if not progressed:
                    break
                cursor += 1

            rows: list[dict[str, Any]] = []
            by_candidate: dict[int, tuple[AssetCandidate, str]] = {}
            local_id = 1
            for candidate, search in balanced:
                frame = _v2_extract_preview(candidate, root, scene_index * 100 + local_id)
                if frame is None:
                    continue
                rows.append({
                    "candidate": local_id,
                    "preview_path": str(frame),
                    "title": candidate.title,
                    "source": candidate.source,
                    "search": search,
                })
                by_candidate[local_id] = (candidate, search)
                local_id += 1

            if rows:
                candidate_maps[scene_index] = by_candidate
                rows_by_scene[scene_index] = rows
                preview_counts[scene_index] = len(rows)

        if not rows_by_scene:
            return {}

        # CLIP is the baseline visual judge. Keep the prompt short and English:
        # long visual_description boilerplate made CLIP latch onto unrelated
        # aesthetics (bags, animals, forests, etc.).
        clip_ranker = None
        try:
            from .multimodal import get_clip_ranker
            clip_ranker = get_clip_ranker()
            print(
                f"{prefix} local CLIP baseline: {len(rows_by_scene)} scene(s), "
                f"{sum(preview_counts.values())} preview(s)",
                flush=True,
            )
        except Exception as exc:
            print(
                f"{prefix} local CLIP unavailable ({exc}); install with "
                'python -m pip install -e ".[semantic]"',
                flush=True,
            )

        local_rankings: dict[int, list[tuple[float, int]]] = {}
        if clip_ranker is not None:
            for scene_index in indexes:
                rows = rows_by_scene.get(scene_index) or []
                if not rows:
                    continue
                scene = plan.scenes[scene_index]
                primary_prompt = (
                    scene.query
                    or (scene.search_queries[0] if scene.search_queries else "")
                    or "person interacting with product"
                )
                paths = [Path(str(row["preview_path"])) for row in rows]

                try:
                    primary_scores = clip_ranker.score_images(primary_prompt, paths)
                except Exception as exc:
                    print(f"{prefix} CLIP scene {scene_index + 1} failed: {exc}", flush=True)
                    continue

                specific_scores: dict[int, float] = {}
                searches = list(dict.fromkeys(str(row.get("search") or "") for row in rows))
                for search in searches:
                    indexes_for_search = [
                        i for i, row in enumerate(rows)
                        if str(row.get("search") or "") == search
                    ]
                    if not indexes_for_search:
                        continue
                    try:
                        values = clip_ranker.score_images(
                            search,
                            [paths[i] for i in indexes_for_search],
                        )
                    except Exception:
                        continue
                    for i, value in zip(indexes_for_search, values):
                        specific_scores[i] = float(value)

                ranked_rows: list[tuple[float, int]] = []
                wanted_tokens = _tokens(
                    " ".join([primary_prompt, *(scene.search_queries or [])[:3]])
                )
                for i, row in enumerate(rows):
                    try:
                        candidate_id = int(row.get("candidate"))
                    except (TypeError, ValueError):
                        continue
                    primary = float(primary_scores[i]) if i < len(primary_scores) else 0.0
                    specific = specific_scores.get(i, primary)
                    title_tokens = _tokens(str(row.get("title") or ""))
                    lexical = (
                        len(wanted_tokens & title_tokens) / max(1, len(wanted_tokens))
                        if wanted_tokens else 0.0
                    )
                    # Specific query match dominates; primary intent prevents a
                    # provider result from winning just because it matches one
                    # generic word. Title overlap is only a small tie-breaker.
                    score = specific * 0.62 + primary * 0.34 + lexical * 0.04
                    ranked_rows.append((score, candidate_id))

                ranked_rows.sort(reverse=True)
                if ranked_rows:
                    local_rankings[scene_index] = ranked_rows
                    compact = ", ".join(
                        f"#{candidate_id}:{score:.3f}"
                        for score, candidate_id in ranked_rows[:4]
                    )
                    print(f"{prefix} CLIP scene {scene_index + 1}: {compact}", flush=True)

        # Complete unique local baseline.
        results: dict[int, tuple[AssetCandidate, str, dict, list[str], int]] = {}
        selected_urls: set[str] = set(used_urls)
        for scene_index in indexes:
            ranking = local_rankings.get(scene_index) or []
            choice_pair = None
            choice_score = None
            choice_id = None
            for score, candidate_id in ranking:
                pair = candidate_maps.get(scene_index, {}).get(candidate_id)
                if pair is None:
                    continue
                candidate, _ = pair
                if candidate.download_url in selected_urls:
                    continue
                choice_pair = pair
                choice_score = score
                choice_id = candidate_id
                break

            if choice_pair is None:
                for candidate_id, pair in candidate_maps.get(scene_index, {}).items():
                    candidate, _ = pair
                    if candidate.download_url not in selected_urls:
                        choice_pair = pair
                        choice_id = candidate_id
                        break

            if choice_pair is None:
                continue

            candidate, search = choice_pair
            fit = (
                int(max(0.0, min(100.0, ((choice_score or 0.0) + 0.10) * 190.0)))
                if choice_score is not None else 42
            )
            candidate.score = float(fit)
            candidate.semantic_score = float(choice_score) if choice_score is not None else None
            info = {
                "score": fit,
                "accept": True,
                "reason": "local CLIP multi-query preview ranking" if choice_score is not None else "provider fallback",
                "mismatch": "",
                "tone_match": 100,
                "quality_score": 0,
                "match_level": "v2_local_clip" if choice_score is not None else "v2_local_provider",
                "candidate": choice_id,
            }
            results[scene_index] = (
                candidate,
                search,
                info,
                queries_by_scene.get(scene_index, []),
                preview_counts.get(scene_index, 0),
            )
            selected_urls.add(candidate.download_url)

        print(
            f"{prefix} local baseline selected {len(results)}/{len(rows_by_scene)} scene(s)",
            flush=True,
        )

        # --material-v2-local or unavailable Gemini: stop here. No degradation to
        # metadata-only rescue just because the cloud model is down.
        if gemini is None:
            print(f"{prefix} Gemini skipped; using local CLIP roll", flush=True)
            return results

        # Gemini only sees the top 3 CLIP candidates per scene, which is smaller,
        # cheaper and semantically stronger than dumping every provider preview.
        gemini_groups: list[dict[str, Any]] = []
        for scene_index in indexes:
            rows = rows_by_scene.get(scene_index) or []
            if not rows:
                continue
            ranking = local_rankings.get(scene_index) or []
            preferred_ids = [candidate_id for _, candidate_id in ranking[:3]]
            if not preferred_ids:
                preferred_ids = [int(row["candidate"]) for row in rows[:3]]
            top_rows = [
                row for row in rows
                if int(row.get("candidate", -1)) in set(preferred_ids)
            ]
            if top_rows:
                gemini_groups.append({"scene": scene_index, "candidates": top_rows})

        batch_size = 4
        total_batches = (len(gemini_groups) + batch_size - 1) // batch_size
        gemini_selected = 0
        for batch_no, offset in enumerate(range(0, len(gemini_groups), batch_size), start=1):
            chunk = gemini_groups[offset:offset + batch_size]
            image_count = sum(len(group.get("candidates") or []) for group in chunk)
            print(
                f"{prefix} Gemini batch {batch_no}/{total_batches}: "
                f"{len(chunk)} scene(s), {image_count} CLIP-shortlisted preview(s)",
                flush=True,
            )
            choices = gemini.choose_roll_visuals(plan.scenes, chunk)
            if not choices and getattr(gemini, "last_error", None):
                print(
                    f"{prefix} Gemini unavailable; keeping local CLIP baseline",
                    flush=True,
                )
                break

            for choice in choices:
                try:
                    scene_index = int(choice.get("scene"))
                    candidate_index = int(choice.get("candidate"))
                    fit = int(choice.get("fit", 0))
                except (TypeError, ValueError):
                    continue
                if candidate_index == 0 or fit < 35:
                    continue
                pair = candidate_maps.get(scene_index, {}).get(candidate_index)
                if pair is None:
                    continue
                candidate, search = pair
                other_urls = {
                    value[0].download_url
                    for idx, value in results.items()
                    if idx != scene_index
                }
                if candidate.download_url in other_urls:
                    continue

                candidate.score = float(fit)
                candidate.semantic_score = fit / 100.0
                info = {
                    "score": fit,
                    "accept": fit >= 50,
                    "reason": str(choice.get("reason") or ""),
                    "mismatch": "",
                    "tone_match": 100,
                    "quality_score": 0,
                    "match_level": "v2_gemini_refined",
                }
                results[scene_index] = (
                    candidate,
                    search,
                    info,
                    queries_by_scene.get(scene_index, []),
                    preview_counts.get(scene_index, 0),
                )
                gemini_selected += 1

        print(
            f"{prefix} final roll choices {len(results)}/{len(rows_by_scene)} "
            f"(Gemini refinements={gemini_selected})",
            flush=True,
        )
        return results



def _v2_select_stock_video(
    scene: Scene,
    out_dir: Path,
    *,
    index: int,
    gemini: Any,
    used_urls: set[str],
    prefix: str,
) -> tuple[AssetCandidate | None, Path | None, str, dict | None, list[str], int]:
    """Quota-aware v2 Visual Director.

    Material Brain 2 already provides strong action-first search queries. Spend
    Gemini quota only once per scene to compare real preview frames instead of
    asking Gemini separately to rewrite queries and then judging exact+broad
    pools. If Gemini declines the shortlist, the caller's local stock-video
    rescue handles the scene without another API request.
    """
    base = [q for q in (scene.search_queries or []) if q.strip()]
    if scene.query:
        base.append(scene.query)
    queries: list[str] = []
    seen: set[str] = set()
    for raw in base:
        q = re.sub(r"\s+", " ", raw).strip()
        key = q.casefold()
        if q and key not in seen:
            seen.add(key)
            queries.append(q)
        if len(queries) >= 3:
            break

    if not queries:
        return None, None, "", None, [], 0

    print(f"{prefix} [v2] queries: " + " || ".join(queries), flush=True)

    pairs = _v2_stock_candidates(
        queries,
        used_urls=used_urls,
        max_candidates=24,
        per_query=8,
    )
    if not pairs:
        print(f"{prefix} [v2] no provider candidates", flush=True)
        return None, None, "", None, queries, 0

    with tempfile.TemporaryDirectory(prefix="video-ai-v2-director-") as d:
        root = Path(d)
        rows: list[dict[str, Any]] = []
        by_index: dict[int, tuple[AssetCandidate, str]] = {}
        label = 1
        for candidate, search in pairs:
            frame = _v2_extract_preview(candidate, root, label)
            if frame is None:
                continue
            rows.append({
                "index": label,
                "preview_path": str(frame),
                "title": candidate.title,
                "source": candidate.source,
                "search": search,
            })
            by_index[label] = (candidate, search)
            label += 1

        print(f"{prefix} [v2] previews={len(rows)}", flush=True)
        if not rows:
            return None, None, "", None, queries, 0

        try:
            choices = gemini.choose_visual_candidates(scene, rows, mode="exact")
        except Exception as exc:
            print(f"{prefix} [v2] Gemini choice unavailable: {exc}", flush=True)
            choices = []

        if not choices:
            print(f"{prefix} [v2] Gemini returned no choices; local rescue later", flush=True)
            return None, None, "", None, queries, len(rows)

        compact = ", ".join(
            f"#{x.get('index')}:{x.get('fit')}" for x in choices[:5]
        )
        print(f"{prefix} [v2] Gemini choices {compact}", flush=True)

        weak_choice: tuple[int, AssetCandidate, str, dict] | None = None
        for choice in choices:
            pair = by_index.get(int(choice.get("index", -1)))
            if pair is None:
                continue
            candidate, search = pair
            fit = int(choice.get("fit", 0))
            info = {
                "score": fit,
                "accept": fit >= 55,
                "reason": str(choice.get("reason") or ""),
                "mismatch": "",
                "tone_match": 100,
                "quality_score": 0,
                "match_level": "v2_exact",
            }
            if weak_choice is None or fit > weak_choice[0]:
                weak_choice = (fit, candidate, search, info)
            if fit < 55:
                continue

            target = out_dir / f"_scene_{index:03d}_v2{_suffix(candidate)}"
            try:
                _download(candidate.download_url, target)
            except Exception:
                target.unlink(missing_ok=True)
                continue
            guard = local_quality_guard(
                target,
                title=candidate.title,
                description=candidate.description,
                width=candidate.width,
                height=candidate.height,
                kind=candidate.kind,
            )
            if guard.score < 38 or _has_severe_local_issue(guard.issues):
                target.unlink(missing_ok=True)
                continue
            candidate.score = float(fit)
            candidate.semantic_score = fit / 100.0
            info["quality_score"] = guard.score
            info["accept"] = True
            return candidate, target, search, info, queries, len(rows)

        # Bounded emergency acceptance from the same single Gemini judgement.
        if weak_choice is not None and weak_choice[0] >= 30:
            fit, candidate, search, info = weak_choice
            target = out_dir / f"_scene_{index:03d}_v2_emergency{_suffix(candidate)}"
            try:
                _download(candidate.download_url, target)
                guard = local_quality_guard(
                    target,
                    title=candidate.title,
                    description=candidate.description,
                    width=candidate.width,
                    height=candidate.height,
                    kind=candidate.kind,
                )
                if guard.score >= 38 and not _has_severe_local_issue(guard.issues):
                    candidate.score = float(fit)
                    candidate.semantic_score = fit / 100.0
                    info["quality_score"] = guard.score
                    info["accept"] = True
                    info["match_level"] = "v2_emergency"
                    print(f"{prefix} [v2] emergency unique fallback fit={fit}", flush=True)
                    return candidate, target, search, info, queries, len(rows)
            except Exception:
                pass
            target.unlink(missing_ok=True)

    return None, None, "", None, queries, 0


def _build_ranked_pool(
    scene: Scene,
    *,
    limit: int,
    meme_dir: str | Path | None,
    used_urls: set[str],
    used_titles: list[str],
    semantic: bool,
    semantic_top_k: int,
) -> tuple[list[tuple[AssetCandidate, str]], list[str]]:
    pool: dict[str, tuple[AssetCandidate, str]] = {}
    variants = list(_query_variants(scene))

    if (scene.visual_mode == "meme" or scene.source_mode == "meme_library") and not scene.semantic_lock:
        memes = search_memes(meme_dir, scene.visual_description or scene.query, limit=20)
        if scene.meme_filename:
            memes.sort(key=lambda m: 0 if m.path.name == scene.meme_filename else 1)
        for meme in memes:
            key = f"local:{meme.path.resolve()}"
            exact = 100.0 if scene.meme_filename and meme.path.name == scene.meme_filename else 0.0
            pool[key] = (
                AssetCandidate(
                    meme.title, "", key, "video/mp4" if meme.kind == "video" else "image/jpeg",
                    0, 0, meme.path.stat().st_size, meme.kind, "local user library",
                    source="local_meme", local_path=str(meme.path.resolve()),
                    score=30.0 + meme.score + exact,
                ),
                "local meme library",
            )

    archive_only = scene.semantic_lock or scene.source_mode == "historical_archive"
    include_stock = (
        scene.source_mode == "stock_video" or (scene.source_mode == "auto" and scene.visual_mode == "video")
    ) and not scene.semantic_lock
    allow_public = scene.source_mode != "meme_library" or not pool or scene.semantic_lock

    if allow_public:
        for query_index, variant in enumerate(variants):
            for candidate in search_all(variant, limit=limit, include_stock_video=include_stock, archive_only=archive_only):
                if candidate.download_url in used_urls or not _source_allowed(scene, candidate):
                    continue
                duplicate = _title_similarity(candidate.title, used_titles[-4:])
                if duplicate >= 0.86 and not scene.semantic_lock:
                    continue
                ranking_query = " ".join(x for x in (variant, scene.visual_description or "") if x).strip()
                candidate.score = (
                    _score(candidate, ranking_query, scene.visual_description or "")
                    + _visual_mode_bonus(scene, candidate)
                    + _source_mode_bonus(scene, candidate)
                    + _semantic_lock_bonus(scene, candidate)
                    - query_index * 0.28
                    - _repeat_penalty(candidate.title, used_titles)
                )
                existing = pool.get(candidate.download_url)
                if existing is None or candidate.score > existing[0].score:
                    pool[candidate.download_url] = (candidate, variant)

    ranked = sorted(pool.values(), key=lambda item: item[0].score, reverse=True)
    if scene.visual_mode == "video" and scene.source_mode == "stock_video":
        video_ranked = [
            item for item in ranked
            if item[0].kind == "video" and item[0].source in {"pexels", "pixabay"}
        ]
        if video_ranked:
            ranked = video_ranked
    if semantic and ranked:
        _video_preview_rerank(scene, ranked, top_k=max(semantic_top_k, 60))
        _semantic_rerank(scene, ranked, top_k=semantic_top_k)
        ranked.sort(key=lambda item: item[0].score, reverse=True)
    return ranked, variants


def _build_recovery_pool(
    scene: Scene,
    *,
    limit: int,
    used_urls: set[str],
    used_titles: list[str],
    semantic: bool,
) -> list[tuple[AssetCandidate, str]]:
    pool: dict[str, tuple[AssetCandidate, str]] = {}
    include_stock = scene.visual_mode == "video" or scene.source_mode == "stock_video"
    for query in _recovery_queries(scene):
        for candidate in search_all(query, limit=limit, include_stock_video=include_stock, archive_only=False):
            if candidate.download_url in used_urls:
                continue
            if scene.visual_mode == "image" and candidate.kind != "image":
                continue
            if scene.visual_mode == "video" and candidate.kind not in {"video", "image"}:
                continue
            duplicate = _title_similarity(candidate.title, used_titles[-4:])
            if duplicate >= 0.9:
                continue
            candidate.score = _score(candidate, query, scene.caption or "") + _visual_mode_bonus(scene, candidate) - _repeat_penalty(candidate.title, used_titles)
            pool.setdefault(candidate.download_url, (candidate, query))
    ranked = sorted(pool.values(), key=lambda item: item[0].score, reverse=True)
    if semantic and ranked:
        _video_preview_rerank(scene, ranked, top_k=min(60, len(ranked)))
        _semantic_rerank(scene, ranked, top_k=min(4, len(ranked)))
        ranked.sort(key=lambda item: item[0].score, reverse=True)
    return ranked


def _select_best_candidate(
    scene: Scene,
    ranked: list[tuple[AssetCandidate, str]],
    out_dir: Path,
    *,
    index: int,
    gemini,
    max_checks: int,
    prefix: str,
    attempt_offset: int = 0,
    match_level: str = "exact",
) -> tuple[AssetCandidate | None, Path | None, str, dict | None, list[dict], list[dict], int]:
    rejected: list[dict] = []
    local_rejected: list[dict] = []
    evaluated: list[tuple[float, AssetCandidate, Path, str, dict | None]] = []
    checks = 0

    for candidate, search in ranked:
        if checks >= max_checks:
            break
        attempt = attempt_offset + checks + 1
        suffix = _suffix(candidate)
        candidate_target = out_dir / f"_scene_{index:03d}_try_{attempt:02d}{suffix}"
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

        local_guard = local_quality_guard(
            candidate_target, title=candidate.title, description=candidate.description,
            width=candidate.width, height=candidate.height, kind=candidate.kind,
        )
        if local_guard.score < 38 or _has_severe_local_issue(local_guard.issues):
            local_rejected.append({
                "title": candidate.title, "source": candidate.source,
                "quality_score": local_guard.score, "issues": local_guard.issues,
            })
            candidate_target.unlink(missing_ok=True)
            continue

        conflict = _historical_metadata_conflict(scene, candidate)
        if conflict:
            rejected.append({
                "title": candidate.title,
                "source": candidate.source,
                "download_url": candidate.download_url,
                "reason": conflict,
                "mismatch": conflict,
                "hard_reject": True,
                "metadata_guard": True,
            })
            candidate_target.unlink(missing_ok=True)
            print(f"{prefix} rejected metadata: {conflict}", flush=True)
            continue

        checks += 1
        judge_info = None
        combined = candidate.score + local_guard.score * 0.035
        if gemini is not None:
            print(f"{prefix} Gemini {checks}/{max_checks}: {candidate.title[:55]}", flush=True)
            judgement = gemini.judge_visual(
                scene,
                candidate_target,
                candidate_title=candidate.title,
                source=candidate.source,
                match_level=match_level,
            )
            if judgement is None:
                combined -= 3.0
            else:
                judge_info = {
                    "score": judgement.score, "accept": judgement.accept,
                    "reason": judgement.reason, "mismatch": judgement.mismatch,
                    "tone_match": judgement.tone_match, "quality_score": judgement.quality_score,
                    "quality_issues": judgement.quality_issues or [],
                    "match_level": match_level,
                }
                severe = _hard_judge_reject(scene, judgement)
                combined += judgement.score * 0.16 + judgement.tone_match * 0.075 + judgement.quality_score * 0.075
                if judgement.accept:
                    combined += 4.0
                else:
                    combined -= 4.0
                if severe:
                    rejected.append({"title": candidate.title, "source": candidate.source, "download_url": candidate.download_url, **judge_info, "hard_reject": True})
                    candidate_target.unlink(missing_ok=True)
                    print(f"{prefix} rejected hard: {judgement.mismatch or judgement.reason[:70]}", flush=True)
                    continue
                if not judgement.accept:
                    rejected.append({"title": candidate.title, "source": candidate.source, "download_url": candidate.download_url, **judge_info, "hard_reject": False})

        evaluated.append((combined, candidate, candidate_target, search, judge_info))

    if not evaluated:
        return None, None, "", None, rejected, local_rejected, checks

    evaluated.sort(key=lambda item: item[0], reverse=True)
    best_score, chosen, target, search_used, judge_info = evaluated[0]
    chosen.score = best_score
    for _, _, other_target, _, _ in evaluated[1:]:
        other_target.unlink(missing_ok=True)
    return chosen, target, search_used, judge_info, rejected, local_rejected, checks


def _hard_judge_reject(scene: Scene, judgement) -> bool:
    if judgement.quality_score < 28:
        return True
    if judgement.score < 30:
        return True
    if scene.tone in _DARK_TONES and judgement.tone_match < 28:
        return True
    if scene.semantic_lock:
        return (not judgement.accept) or judgement.score < 70
    mismatch = (judgement.mismatch or "").lower()
    factual_red_flags = (
        "wrong person", "wrong event", "wrong war", "wrong century", "wrong country",
        "wrong period", "different war", "different conflict", "unrelated conflict",
        "anachron", "modern substitute", "modern-day", "modern setting",
    )
    return any(flag in mismatch for flag in factual_red_flags)


def _has_severe_local_issue(issues: list[str]) -> bool:
    severe = {"too_short", "flat_gray_video"}
    return any(issue in severe for issue in issues)


def ensure_visual_coverage(plan: ShotPlan, manifest: list[dict] | None = None) -> set[int]:
    good = [i for i, s in enumerate(plan.scenes) if s.asset and Path(s.asset).exists() and s.asset_kind != "blank"]
    if not good:
        return set()
    filled: set[int] = set()
    by_scene = {int(i.get("scene")): i for i in (manifest or []) if isinstance(i.get("scene"), int)}

    for index, scene in enumerate(plan.scenes):
        if scene.asset and Path(scene.asset).exists() and scene.asset_kind != "blank":
            continue

        if scene.semantic_lock:
            compatible = [i for i in good if plan.scenes[i].semantic_lock and _lock_compatible(scene, plan.scenes[i])]
            if not compatible:
                compatible = [i for i in good if _context_compatible(scene, plan.scenes[i])]
        else:
            compatible = [
                i for i in good
                if not plan.scenes[i].semantic_lock
                and _tone_compatible(scene.tone, plan.scenes[i].tone)
            ]

        if not compatible and scene.semantic_lock:
            compatible = list(good)
        if not compatible:
            continue

        recent_assets = {
            plan.scenes[j].asset for j in range(max(0, index - 3), index)
            if plan.scenes[j].asset
        }
        diverse = [i for i in compatible if plan.scenes[i].asset not in recent_assets]
        candidates = diverse or compatible
        source_index = min(candidates, key=lambda other: (abs(other - index), 0 if other < index else 1, other))
        source = plan.scenes[source_index]
        scene.asset = source.asset
        scene.asset_kind = source.asset_kind
        scene.focus_x = source.focus_x
        scene.focus_y = source.focus_y
        scene.focus_source = f"fallback_recovery:{source_index}"
        scene.asset_score = source.asset_score
        scene.semantic_score = source.semantic_score
        scene.motion_preset = "micro_push" if source.asset_kind == "video" else "slow_push"
        filled.add(index)
        if index in by_scene:
            by_scene[index].update({
                "status": "fallback_recovery_no_blank",
                "fallback_from_scene": source_index,
                "path": scene.asset,
            })
    return filled


def _tone_compatible(a: str, b: str) -> bool:
    dark = _DARK_TONES
    light = {"positive", "funny", "victorious", "absurd"}
    if a in dark:
        return b in dark or b in {"emotional", "mysterious", "neutral"}
    if a in light:
        return b in light or b in {"neutral", "emotional"}
    return True


def _lock_compatible(target: Scene, source: Scene) -> bool:
    a = {x.lower() for x in target.required_entities}
    b = {x.lower() for x in source.required_entities}
    ca = {x.lower() for x in target.required_context}
    cb = {x.lower() for x in source.required_context}
    return (not a or bool(a & b)) and (not ca or bool(ca & cb))


def _context_compatible(target: Scene, source: Scene) -> bool:
    if not target.required_context:
        return True
    source_text = " ".join([source.caption or "", source.visual_description or "", *source.required_context]).lower()
    return any(ctx.lower() in source_text for ctx in target.required_context)


def _historical_metadata_conflict(scene: Scene, candidate: AssetCandidate) -> str | None:
    """Reject obvious factual conflicts before spending a Gemini judgement.

    This is intentionally conservative: only explicit named wars/conflicts,
    countries and century markers can trigger it. Generic archival titles pass
    through to Gemini.
    """
    if not scene.semantic_lock:
        return None

    target = " ".join([
        *(scene.required_entities or []),
        *(scene.required_context or []),
        scene.caption or "",
        scene.semantic_fallback or "",
    ]).lower()
    meta = " ".join([candidate.title or "", candidate.description or ""]).lower()

    target_events = _named_historical_events(target)
    candidate_events = _named_historical_events(meta)
    if target_events and candidate_events and not (target_events & candidate_events):
        return "wrong event/war in candidate metadata"

    target_countries = _named_countries(target)
    candidate_countries = _named_countries(meta)
    if target_countries and candidate_countries and not (target_countries & candidate_countries):
        return "wrong country in candidate metadata"

    target_century = _explicit_century(target)
    candidate_century = _explicit_century(meta)
    if target_century and candidate_century and target_century != candidate_century:
        return "wrong century in candidate metadata"

    return None


def _named_historical_events(value: str) -> set[str]:
    patterns = {
        "taiping rebellion": ("taiping rebellion", "taiping"),
        "world war i": ("world war i", "first world war", "wwi", "1914-1918", "1914–1918"),
        "world war ii": ("world war ii", "second world war", "wwii", "1939-1945", "1939–1945"),
        "american civil war": ("american civil war", "u.s. civil war", "us civil war"),
        "crimean war": ("crimean war",),
        "vietnam war": ("vietnam war",),
        "korean war": ("korean war",),
        "napoleonic wars": ("napoleonic war", "napoleonic wars"),
        "russian civil war": ("russian civil war",),
    }
    return {name for name, needles in patterns.items() if any(needle in value for needle in needles)}


def _named_countries(value: str) -> set[str]:
    patterns = {
        "china": ("china", "chinese", "qing", "китай", "цин"),
        "russia": ("russia", "russian", "росси", "русск"),
        "ukraine": ("ukraine", "ukrainian", "украин"),
        "germany": ("germany", "german", "герман", "немец"),
        "france": ("france", "french", "франц"),
        "britain": ("britain", "british", "england", "english", "британ", "англи"),
        "japan": ("japan", "japanese", "япон"),
        "united states": ("united states", "american", "u.s.", "usa", "сша"),
        "vietnam": ("vietnam", "vietnamese"),
        "korea": ("korea", "korean"),
    }
    return {name for name, needles in patterns.items() if any(needle in value for needle in needles)}


def _explicit_century(value: str) -> int | None:
    if re.search(r"\b18\d{2}\b|\b19th[- ]century\b|\bxix\b", value):
        return 19
    if re.search(r"\b19\d{2}\b|\b20th[- ]century\b|\bxx\b", value):
        return 20
    if re.search(r"\b20\d{2}\b|\b21st[- ]century\b|\bxxi\b", value):
        return 21
    if re.search(r"\b17\d{2}\b|\b18th[- ]century\b|\bxviii\b", value):
        return 18
    return None


def _source_allowed(scene: Scene, candidate: AssetCandidate) -> bool:
    if scene.semantic_lock:
        return candidate.source in {"commons", "openverse"} and candidate.kind == "image"
    if scene.source_mode == "historical_archive":
        return candidate.source in {"commons", "openverse"} and candidate.kind == "image"
    if scene.source_mode == "stock_video":
        return (
            (candidate.kind == "video" and candidate.source in {"pexels", "pixabay"})
            or (candidate.kind == "image" and candidate.source in {"commons", "openverse", "pexels", "pixabay"})
        )
    if scene.source_mode == "meme_library":
        return candidate.source == "local_meme"
    if scene.source_mode == "generic_image":
        return candidate.kind == "image"
    return True


def _query_variants(scene: Scene) -> Iterable[str]:
    seen: set[str] = set()
    locked = " ".join([*scene.required_entities, *scene.required_context]).strip()
    base: list[str] = []
    tone_hint = _tone_query_hint(scene.tone)
    if scene.semantic_lock and locked:
        base += [
            f"{locked} {scene.visual_description or ''}",
            f"{locked} historical illustration",
            f"{locked} archival portrait",
            scene.semantic_fallback or "",
        ]
    base += [f"{q} {tone_hint}".strip() for q in (scene.search_queries or [])]
    base += [f"{scene.visual_description or ''} {tone_hint}".strip(), scene.query]
    if scene.source_mode == "historical_archive":
        base += [
            f"{scene.visual_description or scene.query} archival engraving",
            f"{scene.visual_description or scene.query} historical painting",
        ]
    elif scene.visual_mode == "video":
        base += [
            f"{scene.visual_description or scene.query} {tone_hint} documentary footage",
            f"{scene.visual_description or scene.query} b roll",
        ]
    elif scene.visual_mode == "meme":
        base += [f"{scene.visual_description or scene.query} reaction"]
    else:
        base += [f"{scene.visual_description or scene.query} {tone_hint} photo illustration"]
    for value in base:
        value = re.sub(r"\s+", " ", value).strip()
        if value and value.lower() not in seen:
            seen.add(value.lower())
            yield value


def _recovery_queries(scene: Scene) -> list[str]:
    description = scene.visual_description or scene.caption or scene.query
    keywords = " ".join(list(_tokens(scene.caption or scene.query))[:5])
    kind = "real footage" if scene.visual_mode == "video" else "documentary photo"
    queries = [
        f"{description} {kind}",
        f"{keywords} {kind}",
        f"person {keywords} action {kind}",
    ]
    out: list[str] = []
    seen: set[str] = set()
    for q in queries:
        q = re.sub(r"\s+", " ", q).strip()
        if q and q.lower() not in seen:
            out.append(q)
            seen.add(q.lower())
    return out


def _tone_query_hint(tone: str) -> str:
    return {
        "tragic": "somber aftermath destruction mourning",
        "violent": "conflict destruction tense",
        "negative": "somber disappointed dark",
        "tense": "tense anxious dramatic",
        "shocking": "dramatic shocking serious",
        "mysterious": "mysterious surreal atmospheric",
        "religious": "religious sacred painting",
        "positive": "uplifting hopeful",
        "victorious": "victory triumphant",
        "funny": "funny reaction",
        "emotional": "emotional expressive",
    }.get(tone, "")


def _semantic_lock_bonus(scene: Scene, candidate: AssetCandidate) -> float:
    if not scene.semantic_lock:
        return 0.0
    hay = " ".join([candidate.title, candidate.description]).lower()
    bonus = 0.0
    for entity in scene.required_entities:
        if all(tok in hay for tok in _tokens(entity)):
            bonus += 14.0
        else:
            bonus -= 8.0
    for ctx in scene.required_context:
        toks = _tokens(ctx)
        if toks and any(tok in hay for tok in toks):
            bonus += 5.0
    return bonus


def _source_mode_bonus(scene: Scene, candidate: AssetCandidate) -> float:
    if scene.source_mode == "historical_archive":
        return 30.0 if candidate.source == "commons" else 16.0 if candidate.source == "openverse" else -100.0
    if scene.source_mode == "stock_video":
        return 22.0 if candidate.source in {"pexels", "pixabay"} and candidate.kind == "video" else 2.0
    if scene.source_mode == "meme_library":
        return 100.0 if candidate.source == "local_meme" else -100.0
    if scene.source_mode == "generic_image":
        return 8.0 if candidate.kind == "image" else -10.0
    return 0.0


def _visual_mode_bonus(scene: Scene, candidate: AssetCandidate) -> float:
    if scene.visual_mode == "video":
        return 18.0 if candidate.kind == "video" else -5.0
    if scene.visual_mode == "image":
        return 4.0 if candidate.kind == "image" else -1.0
    if scene.visual_mode == "meme":
        return 30.0 if candidate.source == "local_meme" else -10.0
    return 0.0


def _video_preview_rerank(scene: Scene, ranked: list[tuple[AssetCandidate, str]], *, top_k: int) -> None:
    # Deep retrieval: sample a broad, query-balanced set of videos instead of
    # visually checking only the metadata-ranked top few.
    video_pairs = [(candidate, query) for candidate, query in ranked if candidate.kind == "video"]
    if not video_pairs:
        return

    groups: dict[str, list[tuple[AssetCandidate, str]]] = {}
    order: list[str] = []
    for pair in video_pairs:
        key = pair[1].casefold().strip()
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(pair)

    pairs: list[tuple[AssetCandidate, str]] = []
    cursor = 0
    while len(pairs) < max(1, top_k) and order:
        progressed = False
        for key in order:
            bucket = groups[key]
            if cursor < len(bucket):
                pairs.append(bucket[cursor])
                progressed = True
                if len(pairs) >= max(1, top_k):
                    break
        if not progressed:
            break
        cursor += 1

    try:
        from .multimodal import ClipRanker
        ranker = ClipRanker()
    except Exception:
        return

    prompt = " ".join([
        scene.visual_description or scene.query or "documentary scene",
        scene.caption or "",
    ]).strip()

    cache_root = Path(tempfile.gettempdir()) / "video-ai-deepclip-cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="video-ai-deepclip-") as d:
        root = Path(d)
        candidate_frames: list[tuple[AssetCandidate, list[Path]]] = []
        for idx, (candidate, _) in enumerate(pairs):
            source_url = candidate.preview_url or candidate.download_url
            cache_key = hashlib.sha1(source_url.encode("utf-8", errors="ignore")).hexdigest()[:20]
            cached = [cache_root / f"{cache_key}_{frame_idx}.jpg" for frame_idx in range(3)]
            frame_paths = [p for p in cached if p.exists() and p.stat().st_size >= 1024]

            if len(frame_paths) < 2:
                video_path = root / f"candidate_{idx:03d}{_suffix(candidate)}"
                try:
                    if candidate.local_path:
                        shutil.copy2(candidate.local_path, video_path)
                    else:
                        _download(source_url, video_path)
                except Exception:
                    continue

                frame_paths = []
                # Three temporal samples catch actions that are absent from the
                # opening frame. Fixed offsets degrade gracefully for short clips.
                for frame_idx, sec in enumerate((0.35, 1.35, 2.75)):
                    frame_path = cached[frame_idx]
                    try:
                        completed = subprocess.run(
                            [
                                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                                "-ss", str(sec), "-i", str(video_path),
                                "-frames:v", "1",
                                "-vf", "scale=384:-2",
                                str(frame_path),
                            ],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            timeout=15,
                            check=False,
                        )
                        if completed.returncode == 0 and frame_path.exists() and frame_path.stat().st_size >= 1024:
                            frame_paths.append(frame_path)
                    except Exception:
                        continue
                video_path.unlink(missing_ok=True)

            if frame_paths:
                candidate_frames.append((candidate, frame_paths))

        if not candidate_frames:
            return

        # Score in small batches to avoid blowing RAM/VRAM on 150+ frames.
        flat: list[tuple[AssetCandidate, Path]] = [
            (candidate, path)
            for candidate, paths in candidate_frames
            for path in paths
        ]
        by_candidate: dict[int, list[float]] = {}
        batch_size = 24
        for offset in range(0, len(flat), batch_size):
            batch = flat[offset:offset + batch_size]
            try:
                scores = ranker.score_images(prompt, [path for _, path in batch])
            except Exception:
                continue
            for (candidate, _), score in zip(batch, scores):
                by_candidate.setdefault(id(candidate), []).append(float(score))

        scored_ids: set[int] = set()
        for candidate, _ in candidate_frames:
            scores = by_candidate.get(id(candidate), [])
            if not scores:
                continue
            best = max(scores)
            mean = sum(scores) / len(scores)
            visual = best * 0.72 + mean * 0.28
            candidate.semantic_score = visual
            # Visual similarity should dominate weak provider metadata.
            candidate.score = candidate.score * 0.20 + visual * 100.0
            scored_ids.add(id(candidate))

        # Never let an uninspected metadata-only video outrank a video that CLIP
        # actually looked at. It can be considered on a later query/repair pass.
        for candidate, _ in video_pairs:
            if id(candidate) not in scored_ids:
                candidate.score = min(candidate.score, -100.0)



def _semantic_rerank(scene: Scene, ranked: list[tuple[AssetCandidate, str]], *, top_k: int) -> None:
    pairs = [(c, q) for c, q in ranked if c.kind == "image"][:max(1, top_k)]
    if not pairs:
        return
    try:
        from .multimodal import ClipRanker
        ranker = ClipRanker()
    except Exception:
        return
    prompt = " ".join([
        *(scene.required_entities if scene.semantic_lock else []),
        *(scene.required_context if scene.semantic_lock else []),
        scene.visual_description or scene.query or "documentary scene",
        _tone_query_hint(scene.tone),
    ]).strip()
    with tempfile.TemporaryDirectory(prefix="video-ai-clip-") as d:
        paths: list[Path] = []
        valid: list[AssetCandidate] = []
        for idx, (c, _) in enumerate(pairs):
            p = Path(d) / f"candidate_{idx:02d}{_suffix(c)}"
            try:
                if c.local_path:
                    shutil.copy2(c.local_path, p)
                else:
                    _download(c.download_url, p)
            except Exception:
                continue
            paths.append(p)
            valid.append(c)
        if not paths:
            return
        try:
            scores = ranker.score_images(prompt, paths)
        except Exception:
            return
        for c, sim in zip(valid, scores):
            c.semantic_score = float(sim)
            c.score += float(sim) * 28.0


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
    desc = _tokens(candidate.description)
    score = len(wanted & title) * 7.0 + len(wanted & desc) * 2.5
    if candidate.width >= 1200 or candidate.height >= 1200:
        score += 2.5
    elif candidate.width >= 800 or candidate.height >= 800:
        score += 1.0
    if candidate.height and candidate.width:
        ratio = candidate.width / candidate.height
        if 0.45 <= ratio <= 0.8:
            score += 3.5
        elif ratio <= 1.1:
            score += 1.5
    if candidate.license:
        score += 0.5
    return score


def _title_similarity(title: str, previous_titles: list[str]) -> float:
    current = _tokens(title)
    if not current:
        return 0.0
    worst = 0.0
    for previous in previous_titles:
        other = _tokens(previous)
        if other:
            worst = max(worst, len(current & other) / max(1, len(current | other)))
    return worst


def _repeat_penalty(title: str, previous_titles: list[str]) -> float:
    return _title_similarity(title, previous_titles[-4:]) * 24.0


def _cleanup_scene_attempts(out_dir: Path, index: int, *, keep: Path | None = None) -> None:
    for path in out_dir.glob(f"_scene_{index:03d}_try_*.*"):
        if keep is None or path.resolve() != keep.resolve():
            path.unlink(missing_ok=True)


def _tokens(value: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(value) if len(t) > 1}


def _json_get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=25) as response:
        return json.load(response)


def _download(url: str, target: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as response, target.open("wb") as output:
        total = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > _MAX_DOWNLOAD_BYTES:
                raise RuntimeError("asset too large")
            output.write(chunk)


def _kind_from_mime(mime: str) -> str | None:
    if mime in {"image/jpeg", "image/png", "image/webp"}:
        return "image"
    if mime in {"video/webm", "video/mp4", "video/ogg"}:
        return "video"
    return None


def _suffix(candidate: AssetCandidate) -> str:
    suffix = Path(candidate.local_path).suffix.lower() if candidate.local_path else Path(urllib.parse.urlparse(candidate.download_url).path).suffix.lower()
    if candidate.kind == "image" and suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        return ".jpg"
    if candidate.kind == "video" and suffix not in {".webm", ".mp4", ".ogv", ".ogg", ".mov", ".mkv", ".avi"}:
        return ".mp4"
    return suffix or (".jpg" if candidate.kind == "image" else ".mp4")


def _meta(meta: dict, key: str) -> str:
    raw = meta.get(key) or {}
    return str(raw.get("value") or "") if isinstance(raw, dict) else str(raw or "")


def _clean_html(value: str) -> str:
    return html.unescape(_TAG_RE.sub(" ", value)).strip()
