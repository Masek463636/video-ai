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

    duration = max((float(scene.end) for scene in plan.scenes), default=0.0)
    auto_budget = 3 if duration <= 12 else 5 if duration <= 20 else 8 if duration <= 35 else 10
    budget = min(max_overlays, auto_budget) if max_overlays > 0 else auto_budget
    min_target = min(budget, 3 if duration <= 12 else 4 if duration <= 20 else 6 if duration <= 35 else 8)

    prompt = f"""You are the effects editor for a fast TikTok/YouTube Short.

The primary B-roll is already selected. DO NOT replace it. Your job is to add only a few high-value foreground accents that make the edit feel handmade.

Scenes:
{json.dumps(scene_rows, ensure_ascii=False)}

Return JSON with key effects. Aim for {min_target}-{budget} useful effects total when the narration gives enough concrete hooks. Do not stop at one or two effects unless the script truly has no more concrete or numeric moments.

ALLOWED EFFECT TYPES:
1. "png" — a concrete object/person cutout only.
2. "png_text" — concrete cutout + an explicit spoken value/name/quantity.
3. "text" — big text-only emphasis when the spoken number/value itself is the important visual.
4. "none" — do nothing. Prefer none over a weak effect.

SELECTION RULES:
- Every effect must be justified by something EXPLICITLY said in that exact scene.
- Strong PNG examples: milk -> milk carton/glass of milk; grandfather -> old man; phone -> smartphone; crown -> crown; money -> cash.
- Strong text examples: "930 мл", "$100", "50%", "3 года", "1998".
- If narration says a concrete object AND quantity together, prefer png_text. Example: "930 мл молока" -> milk carton + label "930 мл".
- Never visualize abstract ideas just because they sound dramatic.
- Do not add generic reaction PNGs unless the narration explicitly mentions that person/object.
- Avoid repeating the same object or same effect style.
- Spread effects across the timeline. Do not stack them every scene.
- A good 20–30 second Short should usually have 6–8 foreground accents if the narration gives enough concrete hooks.
- It is fine to use effects in neighboring scenes when they are triggered by different spoken ideas and use different visual styles.
- Prefer rhythm: roughly one meaningful accent every 2.5–4 seconds, with occasional quick clusters around strong factual phrases.
- Do NOT force an effect into a weak/abstract scene just to hit the target count.

TIMING:
- anchor = the exact spoken word/short phrase that should trigger the effect.
- The renderer aligns to the word timestamp when possible.
- duration 0.70–1.40 seconds.

VISUAL STYLE:
- animation: "fly", "pop", or "drop".
  - fly = very fast side entry, then smooth ease-out stop.
  - pop = appears instantly.
  - drop = comes quickly from above and eases into place.
- position: left, right, or center.
- size: "large" or "hero".
- For text effects use center unless there is a reason not to.
- label MUST preserve the exact narration language and units.
- query is required only for png/png_text and must be a short ENGLISH Wikimedia search phrase describing the visible object/person.

Return ONLY JSON:
{{"effects":[
  {{"scene":2,"type":"png_text","query":"milk carton","anchor":"молока","label":"930 мл","position":"right","animation":"fly","size":"hero","duration":1.05}},
  {{"scene":6,"type":"text","query":"","anchor":"50 процентов","label":"50%","position":"center","animation":"pop","size":"hero","duration":0.85}}
]}}
"""

    try:
        data = client._generate_json([{"text": prompt}], temperature=0.18)
    except Exception as exc:
        print(f"[fx] overlay planning unavailable: {exc}", flush=True)
        return []

    raw = (data.get("effects") or data.get("overlays") or []) if isinstance(data, dict) else []
    if not isinstance(raw, list):
        return []

    # Gemini sometimes under-edits and returns only 1-2 accents. Give it one
    # bounded second pass that can only fill unused scenes; this is much safer
    # than blindly forcing random effects.
    if len(raw) < min_target:
        used_scene_ids = {
            int(item.get("scene"))
            for item in raw
            if isinstance(item, dict) and str(item.get("scene", "")).lstrip("-").isdigit()
        }
        remaining = [row for row in scene_rows if int(row["scene"]) not in used_scene_ids]
        needed = max(0, min_target - len(raw))
        if remaining and needed:
            fill_prompt = f"""You are filling missing foreground accents for a TikTok/YouTube Short.

Already accepted proposals:
{json.dumps(raw, ensure_ascii=False)}

Unused scenes:
{json.dumps(remaining, ensure_ascii=False)}

Return ONLY JSON with key effects. Add up to {needed + 2} NEW effects from UNUSED scenes, aiming to fill at least {needed} when the words genuinely support it.

Use the same effect types: png, png_text, text.
- Prefer explicit concrete nouns, people, physical objects, food, money, devices, quantities, dates and percentages.
- text must quote a short value/word that is actually spoken.
- png/png_text query must be a short ENGLISH search phrase.
- Do not repeat an already proposed concept.
- Avoid abstract filler.

Schema:
{{"effects":[{{"scene":4,"type":"png","query":"shopping basket","anchor":"корзина","label":"","position":"left","animation":"drop","size":"large","duration":0.95}}]}}
"""
            try:
                extra = client._generate_json([{"text": fill_prompt}], temperature=0.12)
                extra_raw = (extra.get("effects") or []) if isinstance(extra, dict) else []
                if isinstance(extra_raw, list):
                    raw.extend(item for item in extra_raw if isinstance(item, dict))
            except Exception as exc:
                print(f"[fx] fill pass unavailable: {exc}", flush=True)

    # Gemini may nominate the same object several times (e.g. milk, 930 ml,
    # packaging). Keep one strong insert per concrete concept, preferring the
    # occurrence that carries an explicit spoken quantity/label.
    best_by_query: dict[str, dict[str, Any]] = {}
    passthrough: list[dict[str, Any]] = []
    for raw_item in raw:
        if not isinstance(raw_item, dict):
            continue
        key = _clean_query(str(raw_item.get("query") or "")).casefold()
        if not key:
            passthrough.append(raw_item)
            continue
        try:
            scene_index = int(raw_item.get("scene"))
        except (TypeError, ValueError):
            scene_index = -1
        scene_caption = plan.scenes[scene_index].caption if 0 <= scene_index < len(plan.scenes) else ""
        score = 0
        if str(raw_item.get("label") or "").strip():
            score += 4
        if _explicit_quantity(scene_caption or ""):
            score += 3
        anchor = str(raw_item.get("anchor") or "")
        if any(ch.isdigit() for ch in anchor):
            score += 2
        previous = best_by_query.get(key)
        if previous is None or score > int(previous.get("_dedupe_score", -1)):
            chosen = dict(raw_item)
            chosen["_dedupe_score"] = score
            best_by_query[key] = chosen

    raw = sorted(
        [*best_by_query.values(), *passthrough],
        key=lambda item: int(item.get("scene", 10**6)) if str(item.get("scene", "")).lstrip("-").isdigit() else 10**6,
    )

    used_scenes: set[int] = set()
    used_concepts: set[str] = set()
    used_times: list[float] = []
    overlays: list[dict[str, Any]] = []

    for item in raw:
        if len(overlays) >= budget or not isinstance(item, dict):
            break
        try:
            scene_index = int(item.get("scene"))
        except (TypeError, ValueError):
            continue
        if scene_index < 0 or scene_index >= len(plan.scenes) or scene_index in used_scenes:
            continue

        scene = plan.scenes[scene_index]
        effect_type = str(item.get("type") or "png").lower().strip()
        if effect_type not in {"png", "png_text", "text"}:
            continue

        anchor = " ".join(str(item.get("anchor") or "").split())[:80]
        if not anchor:
            continue
        caption = scene.caption or ""
        caption_cf = caption.casefold()
        anchor_tokens = [t for t in _tokens(anchor) if len(t) > 2]
        if anchor_tokens and not any(t in caption_cf for t in anchor_tokens):
            continue

        query = _clean_query(str(item.get("query") or ""))
        label = " ".join(str(item.get("label") or "").split())[:28]

        if effect_type in {"png_text", "text"}:
            if label and not _label_is_spoken(caption, label):
                label = ""
            if not label:
                label = _explicit_quantity(caption) or ""
            if not label:
                # Text-driven effect without explicit spoken text is too risky.
                if effect_type == "text":
                    continue
                effect_type = "png"

        if effect_type in {"png", "png_text"} and not query:
            continue

        concept = (query or label).casefold().strip()
        if concept and concept in used_concepts:
            continue

        duration = _clamp_float(item.get("duration"), 0.70, 1.40, 1.00)
        display_end = (
            float(plan.scenes[scene_index + 1].start)
            if scene_index + 1 < len(plan.scenes)
            else float(scene.end)
        )
        anchor_start = _anchor_start(scene, anchor)
        start = max(float(scene.start), (anchor_start - 0.025) if anchor_start is not None else float(scene.start) + 0.10)
        end = min(display_end, start + duration)
        if end - start < 0.45:
            continue

        # Keep density high enough for Shorts, but avoid unreadable effect spam.
        # Neighboring storyboard scenes are allowed; only near-simultaneous
        # accents are rejected.
        if any(abs(start - old_start) < 0.85 for old_start in used_times):
            continue

        raw_position = str(item.get("position") or "").lower()
        position = raw_position if raw_position in {"left", "right", "center"} else ("center" if effect_type == "text" else ("left" if len(overlays) % 2 else "right"))

        animation = str(item.get("animation") or "").lower()
        if animation not in {"fly", "pop", "drop"}:
            animation = ("pop" if effect_type == "text" else ("fly" if len(overlays) % 2 == 0 else "drop"))
        if overlays and animation == overlays[-1].get("animation"):
            animation = {"fly":"pop","pop":"drop","drop":"fly"}[animation]

        size = str(item.get("size") or "").lower()
        if size not in {"large", "hero"}:
            size = "hero" if effect_type in {"png_text", "text"} else "large"

        target: Path | None = None
        candidate = None
        if effect_type in {"png", "png_text"}:
            target = out / f"overlay_{len(overlays):02d}.png"
            candidate = _find_png(query, target, client)
            if candidate is None:
                target.unlink(missing_ok=True)
                # If we at least have a strong spoken value, preserve the idea
                # as a text-only effect instead of throwing the beat away.
                if label:
                    effect_type = "text"
                    query = ""
                    target = None
                    position = "center"
                    animation = "pop" if animation == "fly" else animation
                    size = "hero"
                else:
                    # Better than silently losing the beat: show the exact
                    # spoken anchor as a short callout. This is still grounded
                    # in narration and gives Shorts rhythm even when Commons
                    # has no usable transparent cutout.
                    fallback_label = _short_anchor_label(anchor)
                    if fallback_label:
                        print(f"[fx] PNG unavailable -> text fallback: {fallback_label!r}", flush=True)
                        effect_type = "text"
                        query = ""
                        label = fallback_label
                        target = None
                        position = "center"
                        animation = "pop" if animation == "fly" else animation
                        size = "large"
                    else:
                        print(f"[fx] no verified PNG insert found for: {query}", flush=True)
                        continue

        overlay = {
            "scene": scene_index,
            "type": effect_type,
            "asset": str(target) if target is not None else "",
            "query": query,
            "anchor": anchor,
            "label": label,
            "position": position,
            "animation": animation,
            "size": size,
            "start": round(start, 3),
            "end": round(end, 3),
            "source": candidate.source if candidate is not None else "generated_text",
            "title": candidate.title if candidate is not None else label,
            "page_url": candidate.page_url if candidate is not None else "",
            "license": candidate.license if candidate is not None else "",
        }
        overlays.append(overlay)
        used_scenes.add(scene_index)
        used_times.append(start)
        if concept:
            used_concepts.add(concept)
        print(
            f"[fx] {effect_type} {len(overlays)}/{budget}: scene={scene_index + 1} "
            f"anchor={anchor!r} query={query!r} label={label!r} animation={animation}",
            flush=True,
        )

    (out / "overlays.json").write_text(
        json.dumps({"effects": overlays, "overlays": overlays}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return overlays


def _find_png(query: str, target: Path, client):
    """Choose the best visually verified PNG candidate, not the first acceptable one."""
    searches = [
        f"{query} transparent png",
        f"{query} isolated png",
        f"{query} png",
        query,
    ]
    tested = 0
    scored: list[tuple[float, Any, Path]] = []

    for search in searches:
        try:
            candidates = search_commons(search, limit=24)
        except Exception:
            continue
        pngs = [
            cand for cand in candidates
            if cand.kind == "image"
            and cand.mime == "image/png"
            and cand.download_url
            and cand.width >= 220
            and cand.height >= 220
        ]
        if not pngs:
            continue

        tokens = set(_tokens(query))
        pngs.sort(
            key=lambda cand: (
                sum(1 for token in tokens if token in (cand.title + " " + cand.description).casefold()),
                1 if "transparent" in (cand.title + " " + cand.description).casefold() else 0,
                min(cand.width, 2400) * min(cand.height, 2400),
            ),
            reverse=True,
        )

        for candidate in pngs[:6]:
            tested += 1
            preview = target.with_name(target.stem + f"_candidate_{tested}.png")
            try:
                _download(candidate.download_url, preview)
            except Exception:
                preview.unlink(missing_ok=True)
                continue

            score = 45.0
            quality = 45.0
            accepted = True
            try:
                probe_scene = Scene(
                    start=0.0,
                    end=1.0,
                    query=query,
                    caption=query,
                    visual_description=f"single clear isolated cutout of {query}",
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
                    score = float(judgement.score)
                    quality = float(judgement.quality_score)
                    accepted = bool(judgement.accept and score >= 50 and quality >= 44)
            except Exception:
                accepted = True

            if accepted:
                # Prefer relevance first, then visual quality. Evaluate several
                # candidates so a mediocre first hit does not win by accident.
                rank = score * 0.68 + quality * 0.32
                scored.append((rank, candidate, preview))
            else:
                preview.unlink(missing_ok=True)

            if len(scored) >= 3:
                break
        if len(scored) >= 3:
            break

    if not scored:
        for stale in target.parent.glob(target.stem + "_candidate_*.png"):
            stale.unlink(missing_ok=True)
        return None

    scored.sort(key=lambda item: item[0], reverse=True)
    _, best_candidate, best_preview = scored[0]
    best_preview.replace(target)
    for _, _, preview in scored[1:]:
        preview.unlink(missing_ok=True)
    for stale in target.parent.glob(target.stem + "_candidate_*.png"):
        stale.unlink(missing_ok=True)
    return best_candidate

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

def _short_anchor_label(anchor: str) -> str:
    words = [w for w in anchor.strip().split() if w]
    if not words or len(words) > 3:
        return ""
    label = " ".join(words)
    if len(label) > 22:
        return ""
    # Uppercase Cyrillic/Latin naturally; preserve numbers and units.
    return label.upper()


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
