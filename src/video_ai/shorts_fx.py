from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .assets import _download, search_commons
from .models import ShotPlan


def build_shorts_overlays(
    plan: ShotPlan,
    out_dir: str | Path,
    *,
    max_overlays: int = 4,
) -> list[dict[str, Any]]:
    """Plan a few concrete TikTok/Shorts-style PNG pop-ins.

    This never replaces the primary scene assets. Gemini only chooses explicit
    concrete nouns/people from the narration; Wikimedia Commons supplies the
    extra PNG layer. If Gemini, network access, or a suitable PNG is missing,
    the base render remains fully usable.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not plan.scenes or max_overlays <= 0:
        return []

    try:
        from .gemini_ai import get_gemini_client
        client = get_gemini_client()
    except Exception:
        client = None
    if client is None:
        print("[fx] Gemini unavailable; skipping PNG inserts", flush=True)
        return []

    scene_rows = []
    for index, scene in enumerate(plan.scenes):
        scene_rows.append({
            "scene": index,
            "start": round(float(scene.start), 3),
            "end": round(float(scene.end), 3),
            "caption": scene.caption or "",
        })

    prompt = f"""You are a TikTok/YouTube Shorts editor adding a SMALL number of animated PNG cutouts on top of an already edited video.

Scenes:
{json.dumps(scene_rows, ensure_ascii=False)}

Return JSON with key overlays. Choose at most {max_overlays} overlays total.

Rules:
- Only choose a concrete object, animal, food, person type, historical person, place symbol, or physical item EXPLICITLY mentioned in that exact scene.
- Good examples: milk -> "glass of milk"; grandfather -> "old man portrait"; phone -> "smartphone"; money -> "cash"; crown -> "crown".
- Never invent an object just because it is emotionally related.
- Do not choose abstract ideas such as stress, love, danger, religion, history, sadness.
- Prefer moments where a quick object/person pop-in would make the edit feel handmade, funny, explanatory, or emphatic.
- Spread inserts through the video; avoid adjacent scenes unless both are unusually strong.
- query MUST be a short ENGLISH Wikimedia image-search phrase, 1-5 words, describing the visible object/person only.
- anchor is the exact narrated word/short phrase that motivated the insert.
- position is left or right.
- duration is 0.75-1.25 seconds.
- offset is 0.05-0.55 seconds after the scene start.
- Return no overlay when there is no useful concrete insert.

Schema:
{{"overlays":[{{"scene":0,"query":"glass of milk","anchor":"молоко","position":"right","duration":1.0,"offset":0.15}}]}}
"""

    try:
        data = client._generate_json([{"text": prompt}], temperature=0.18)
    except Exception as exc:
        print(f"[fx] overlay planning unavailable: {exc}", flush=True)
        return []

    raw = data.get("overlays", []) if isinstance(data, dict) else []
    if not isinstance(raw, list):
        return []

    used_scenes: set[int] = set()
    overlays: list[dict[str, Any]] = []
    for item in raw:
        if len(overlays) >= max_overlays or not isinstance(item, dict):
            break
        try:
            scene_index = int(item.get("scene"))
        except (TypeError, ValueError):
            continue
        if scene_index < 0 or scene_index >= len(plan.scenes) or scene_index in used_scenes:
            continue
        if any(abs(scene_index - old) <= 1 for old in used_scenes) and len(plan.scenes) > 5:
            continue

        query = _clean_query(str(item.get("query") or ""))
        anchor = " ".join(str(item.get("anchor") or "").split())[:80]
        if not query or not anchor:
            continue
        scene = plan.scenes[scene_index]
        caption = (scene.caption or "").casefold()
        anchor_tokens = [t for t in _tokens(anchor) if len(t) > 2]
        if anchor_tokens and not any(t in caption for t in anchor_tokens):
            continue

        candidate = _find_png(query)
        if candidate is None:
            print(f"[fx] no PNG insert found for: {query}", flush=True)
            continue

        target = out / f"overlay_{len(overlays):02d}.png"
        try:
            _download(candidate.download_url, target)
        except Exception:
            target.unlink(missing_ok=True)
            continue

        duration = _clamp_float(item.get("duration"), 0.75, 1.25, 1.0)
        offset = _clamp_float(item.get("offset"), 0.05, 0.55, 0.15)
        display_end = (
            float(plan.scenes[scene_index + 1].start)
            if scene_index + 1 < len(plan.scenes)
            else float(scene.end)
        )
        start = max(0.0, float(scene.start) + offset)
        end = min(display_end, start + duration)
        if end - start < 0.55:
            target.unlink(missing_ok=True)
            continue

        position = "left" if str(item.get("position")).lower() == "left" else "right"
        overlay = {
            "scene": scene_index,
            "asset": str(target),
            "query": query,
            "anchor": anchor,
            "position": position,
            "start": round(start, 3),
            "end": round(end, 3),
            "source": candidate.source,
            "title": candidate.title,
            "page_url": candidate.page_url,
            "license": candidate.license,
        }
        overlays.append(overlay)
        used_scenes.add(scene_index)
        print(
            f"[fx] PNG insert {len(overlays)}/{max_overlays}: scene={scene_index + 1} "
            f"anchor={anchor!r} query={query!r}",
            flush=True,
        )

    (out / "overlays.json").write_text(
        json.dumps({"overlays": overlays}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return overlays


def _find_png(query: str):
    searches = [
        f"{query} transparent png",
        f"{query} png",
        query,
    ]
    for search in searches:
        try:
            candidates = search_commons(search, limit=18)
        except Exception:
            continue
        pngs = [
            c for c in candidates
            if c.kind == "image"
            and c.mime == "image/png"
            and c.download_url
            and c.width >= 180
            and c.height >= 180
        ]
        if not pngs:
            continue
        tokens = set(_tokens(query))
        pngs.sort(
            key=lambda c: (
                sum(1 for token in tokens if token in (c.title + " " + c.description).casefold()),
                min(c.width, 2200) * min(c.height, 2200),
            ),
            reverse=True,
        )
        return pngs[0]
    return None


def _clean_query(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9 '\-]", " ", value)
    return re.sub(r"\s+", " ", value).strip()[:100]


def _tokens(value: str) -> list[str]:
    return re.findall(r"[a-zа-яё0-9]+", value.casefold())


def _clamp_float(value: object, low: float, high: float, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))
