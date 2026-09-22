from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .assets import _download, search_commons
from .models import Scene, ShotPlan


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
- position is left, right, or center.
- animation is "fly" for a fast side fly-in with a smooth stop, or "pop" for an instant appearance.
- size is "large" or "hero". Prefer large. Use hero for a very important single object/person.
- duration is 0.85-1.45 seconds.
- offset is only a fallback. The renderer will align the insert to the exact spoken anchor when word timestamps exist.
- label is optional. Use it ONLY for a short value/quantity/name that is explicitly spoken in the SAME scene, preserving the narration language exactly. Example: narration says "930 мл молока" -> label "930 мл". Never invent a label.
- Return no overlay when there is no useful concrete insert.

Schema:
{{"overlays":[{{"scene":0,"query":"milk carton","anchor":"молоко","label":"930 мл","position":"right","animation":"fly","size":"hero","duration":1.1,"offset":0.12}}]}}
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

        target = out / f"overlay_{len(overlays):02d}.png"
        candidate = _find_png(query, target, client)
        if candidate is None:
            target.unlink(missing_ok=True)
            print(f"[fx] no verified PNG insert found for: {query}", flush=True)
            continue

        duration = _clamp_float(item.get("duration"), 0.85, 1.45, 1.08)
        offset = _clamp_float(item.get("offset"), 0.05, 0.55, 0.12)
        display_end = (
            float(plan.scenes[scene_index + 1].start)
            if scene_index + 1 < len(plan.scenes)
            else float(scene.end)
        )
        anchor_start = _anchor_start(scene, anchor)
        start = max(float(scene.start), (anchor_start - 0.035) if anchor_start is not None else float(scene.start) + offset)
        end = min(display_end, start + duration)
        if end - start < 0.55:
            target.unlink(missing_ok=True)
            continue

        raw_position = str(item.get("position") or "").lower()
        position = raw_position if raw_position in {"left", "right", "center"} else ("left" if len(overlays) % 2 else "right")
        animation = str(item.get("animation") or "").lower()
        if animation not in {"fly", "pop"}:
            animation = "fly" if len(overlays) % 2 == 0 else "pop"
        if overlays and animation == overlays[-1].get("animation"):
            animation = "pop" if animation == "fly" else "fly"
        if animation == "fly" and position == "center":
            position = "left" if len(overlays) % 2 else "right"

        size = str(item.get("size") or "").lower()
        if size not in {"large", "hero"}:
            size = "hero" if len(overlays) % 3 == 2 else "large"

        label = " ".join(str(item.get("label") or "").split())[:24]
        if label and not _label_is_spoken(scene.caption or "", label):
            label = ""
        if not label:
            label = _explicit_quantity(scene.caption or "") or ""

        overlay = {
            "scene": scene_index,
            "asset": str(target),
            "query": query,
            "anchor": anchor,
            "label": label,
            "position": position,
            "animation": animation,
            "size": size,
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


def _find_png(query: str, target: Path, client):
    """Download the first Gemini-verified PNG candidate.

    Commons search alone can return technically matching but visually useless
    diagrams. A small visual judge pass keeps inserts literal enough for Shorts.
    """
    searches = [
        f"{query} transparent png",
        f"{query} isolated png",
        f"{query} png",
        query,
    ]
    tested = 0
    for search in searches:
        try:
            candidates = search_commons(search, limit=20)
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
            key=lambda cand: (
                sum(1 for token in tokens if token in (cand.title + " " + cand.description).casefold()),
                1 if "transparent" in (cand.title + " " + cand.description).casefold() else 0,
                min(cand.width, 2200) * min(cand.height, 2200),
            ),
            reverse=True,
        )

        for candidate in pngs[:5]:
            tested += 1
            preview = target.with_name(target.stem + f"_candidate_{tested}.png")
            try:
                _download(candidate.download_url, preview)
            except Exception:
                preview.unlink(missing_ok=True)
                continue

            accepted = True
            try:
                probe_scene = Scene(
                    start=0.0,
                    end=1.0,
                    query=query,
                    caption=query,
                    visual_description=f"isolated clear cutout of {query}",
                    visual_mode="image",
                    source_mode="generic_image",
                    motion_preset="none",
                )
                judgement = client.judge_visual(
                    probe_scene,
                    preview,
                    candidate_title=candidate.title,
                    source="commons overlay",
                    match_level="exact",
                )
                if judgement is not None:
                    accepted = bool(judgement.accept and judgement.score >= 55 and judgement.quality_score >= 42)
            except Exception:
                accepted = True

            if accepted:
                preview.replace(target)
                for stale in target.parent.glob(target.stem + "_candidate_*.png"):
                    stale.unlink(missing_ok=True)
                return candidate
            preview.unlink(missing_ok=True)

    for stale in target.parent.glob(target.stem + "_candidate_*.png"):
        stale.unlink(missing_ok=True)
    return None


def _anchor_start(scene: Scene, anchor: str) -> float | None:
    wanted = _tokens(anchor)
    if not wanted or not scene.caption_words:
        return None
    words = [_tokens(word.text) for word in scene.caption_words]
    flattened = [tokens[0] if tokens else "" for tokens in words]
    for i in range(0, len(flattened) - len(wanted) + 1):
        if flattened[i:i + len(wanted)] == wanted:
            return float(scene.caption_words[i].start)
    # One explicit noun is enough for timing when punctuation/inflection differs.
    for i, token in enumerate(flattened):
        if any(len(w) > 3 and (w in token or token in w) for w in wanted):
            return float(scene.caption_words[i].start)
    return None


def _label_is_spoken(caption: str, label: str) -> bool:
    caption_tokens = _tokens(caption)
    label_tokens = _tokens(label)
    return bool(label_tokens) and all(token in caption_tokens for token in label_tokens)


def _explicit_quantity(caption: str) -> str | None:
    match = re.search(
        r"\b\d+(?:[.,]\d+)?\s*(?:мл|л|литр(?:а|ов)?|кг|г|гр|см|мм|км|м|%|процент(?:а|ов)?|грн|руб(?:ля|лей)?|доллар(?:а|ов)?|ml|kg|g|cm|mm|km)\b",
        caption,
        flags=re.IGNORECASE,
    )
    return " ".join(match.group(0).split()) if match else None

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
