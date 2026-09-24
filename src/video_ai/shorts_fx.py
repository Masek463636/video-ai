from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from .assets import _download, search_commons
from .models import Scene, ShotPlan


def build_shorts_overlays(
    plan: ShotPlan,
    out_dir: str | Path,
    *,
    max_overlays: int = 0,
    use_gemini: bool = True,
    sticker_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Plan a few concrete TikTok/Shorts-style PNG pop-ins.

    This never replaces the primary scene assets. Gemini only chooses explicit
    concrete nouns/people from the narration; Wikimedia Commons supplies the
    extra PNG layer. If Gemini, network access, or a suitable PNG is missing,
    the base render remains fully usable.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not plan.scenes:
        return []

    client = None
    if use_gemini:
        try:
            from .gemini_ai import get_gemini_client
            client = get_gemini_client()
        except Exception:
            client = None
        if client is None:
            print("[fx] Gemini unavailable; using local Shorts FX fallback", flush=True)
    else:
        print("[fx] local-only Shorts FX mode: Gemini skipped", flush=True)

    scene_rows = []
    for index, scene in enumerate(plan.scenes):
        scene_rows.append({
            "scene": index,
            "start": round(float(scene.start), 3),
            "end": round(float(scene.end), 3),
            "caption": scene.caption or "",
        })

    duration = max((float(scene.end) for scene in plan.scenes), default=0.0)
    # No fixed "6 overlays" ceiling. By default use an adaptive Shorts rhythm:
    # roughly one foreground beat every 2.6-3.0 seconds. An explicit
    # --max-overlays still acts as a user override.
    auto_budget = max(3, min(14, int(round(duration / 2.7))))
    budget = max_overlays if max_overlays > 0 else auto_budget
    min_target = min(budget, max(3, int(round(duration / 3.5))))

    prompt = f"""You are the effects editor for a fast TikTok/YouTube Short.

The primary B-roll is already selected. DO NOT replace it. Your job is to add only a few high-value foreground accents that make the edit feel handmade.

Scenes:
{json.dumps(scene_rows, ensure_ascii=False)}

Return JSON with key effects. Aim for {min_target}-{budget} useful effects total when the narration gives enough concrete hooks. Do not stop at one or two effects unless the script truly has no more concrete or numeric moments.

LOCAL STICKER PACK AVAILABLE: {bool(sticker_dir)}

ALLOWED EFFECT TYPES:
1. "png" — a concrete object/person cutout only.
2. "png_text" — concrete cutout + an explicit spoken value/name/quantity.
3. "text" — big text-only emphasis when the spoken number/value itself is the important visual.
4. "sticker" — a reaction/emotion GIF from the user's LOCAL sticker pack. Use only when LOCAL STICKER PACK AVAILABLE is True.
5. "none" — do nothing. Prefer none over a weak effect.

SELECTION RULES:
- Every effect must be justified by something EXPLICITLY said in that exact scene.
- Strong PNG examples: milk -> milk carton/glass of milk; grandfather -> old man; phone -> smartphone; crown -> crown; money -> cash.
- Strong text examples: "930 мл", "$100", "50%", "3 года", "1998".
- If narration says a concrete object AND quantity together, prefer png_text. Example: "930 мл молока" -> milk carton + label "930 мл".
- Never visualize abstract ideas as literal PNG objects just because they sound dramatic.
- Reaction/emotion moments SHOULD use type "sticker" when the local sticker pack is available.
- For sticker effects, query must describe the REACTION in English, e.g. "skeptical suspicious reaction", "shocked money reaction", "confused thinking reaction", "laughing reaction".
- On a 20–30 second Short with a sticker pack, aim for roughly 2–4 sticker reactions when there are genuine emotional hooks; do not make every scene a sticker.
- Prefer stickers for surprise, skepticism, confusion, money/price shock, irony, frustration, excitement or humor.
- Avoid repeating the same reaction emotion twice unless the later beat is clearly stronger.
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
- query for png/png_text must be a short ENGLISH Wikimedia search phrase describing the visible object/person.
- query for sticker must be a short ENGLISH reaction description; the renderer will choose the actual GIF from the local sticker pack.

Return ONLY JSON:
{{"effects":[
  {{"scene":2,"type":"png_text","query":"milk carton","anchor":"молока","label":"930 мл","position":"right","animation":"fly","size":"hero","duration":1.75}},
  {{"scene":4,"type":"sticker","query":"skeptical money reaction","anchor":"цену","label":"","position":"left","animation":"fly","size":"hero","duration":1.85}},
  {{"scene":6,"type":"text","query":"","anchor":"50 процентов","label":"50%","position":"center","animation":"pop","size":"hero","duration":1.60}}
]}}
"""

    if client is None:
        data = {}
    else:
        try:
            data = client._generate_json([{"text": prompt}], temperature=0.18)
        except Exception as exc:
            print(f"[fx] overlay planning unavailable: {exc}", flush=True)
            print("[fx] disabling Gemini for the rest of this render; switching to local fallback", flush=True)
            data = {}
            client = None

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

Use the same effect types: png, png_text, text, sticker.
- If the local sticker pack is available, sticker is encouraged for strong reaction beats (surprise, skepticism, confusion, money shock, irony, humor).
- For sticker, query is a short ENGLISH reaction description such as "confused skeptical reaction".
- Prefer explicit concrete nouns, people, physical objects, food, money, devices, quantities, dates and percentages for png/png_text.
- text must quote a short value/word that is actually spoken.
- png/png_text query must be a short ENGLISH search phrase.
- Do not repeat an already proposed concept.
- Avoid abstract filler.

Schema:
{{"effects":[{{"scene":4,"type":"png","query":"shopping basket","anchor":"корзина","label":"","position":"left","animation":"drop","size":"large","duration":0.95}}]}}
"""
            try:
                if client is None:
                    raise RuntimeError("Gemini unavailable")
                extra = client._generate_json([{"text": fill_prompt}], temperature=0.12)
                extra_raw = (extra.get("effects") or []) if isinstance(extra, dict) else []
                if isinstance(extra_raw, list):
                    raw.extend(item for item in extra_raw if isinstance(item, dict))
            except Exception as exc:
                print(f"[fx] fill pass unavailable: {exc}", flush=True)

    # The user's local sticker pack is NOT merely a fallback. Always blend a
    # few context-aware reaction stickers into the plan, even when Gemini
    # successfully planned PNG/text accents.
    sticker_raw = _local_sticker_effect_candidates(
        plan,
        sticker_dir,
        out / "sticker_cache",
        budget=min(max(2, budget // 2), 5),
    )

    if not raw:
        object_raw = _local_effect_candidates(plan, budget)
        raw = [*sticker_raw, *object_raw]
        if raw:
            print(
                f"[fx] local fallback planned {len(raw)} candidate effects "
                f"({len(sticker_raw)} local stickers)",
                flush=True,
            )
    elif sticker_raw:
        # Put sticker reactions first so if Gemini and the sticker pack target
        # the exact same scene, the reaction wins that beat instead of being
        # silently discarded by the one-overlay-per-scene guard.
        raw = [*sticker_raw, *raw]
        print(
            f"[fx] Gemini plan blended with {len(sticker_raw)} local sticker reaction(s)",
            flush=True,
        )

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
        if effect_type not in {"png", "png_text", "text", "sticker"}:
            continue

        anchor = " ".join(str(item.get("anchor") or "").split())[:80]
        local_item = bool(item.get("_local"))
        if not anchor and not local_item:
            continue
        caption = scene.caption or ""
        caption_cf = caption.casefold()
        anchor_tokens = [t for t in _tokens(anchor) if len(t) > 2]
        if not local_item and anchor_tokens and not any(t in caption_cf for t in anchor_tokens):
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

        if effect_type in {"png", "png_text", "sticker"} and not query:
            continue

        concept = (query or label).casefold().strip()
        if concept and concept in used_concepts:
            continue

        duration = _clamp_float(item.get("duration"), 1.35, 2.20, 1.75)
        timeline_end = max(float(s.end) for s in plan.scenes)
        anchor_start = _anchor_start(scene, anchor) if anchor else None
        start = max(float(scene.start), (anchor_start - 0.025) if anchor_start is not None else float(scene.start) + 0.10)
        # Let foreground inserts survive across a storyboard cut. The old code
        # clipped them at the next scene boundary, which often left <1 second.
        end = min(timeline_end, start + duration)
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
        # Foreground visual inserts use one consistent meme motion language:
        # fast fly-in + slow cubic ease-out stop. Text-only emphasis may pop.
        if effect_type in {"png", "png_text", "sticker"}:
            animation = "fly"

        size = str(item.get("size") or "").lower()
        if size not in {"large", "hero"}:
            size = "hero" if effect_type in {"png_text", "text"} else "large"

        target: Path | None = None
        candidate = None
        if effect_type == "sticker":
            raw_asset = str(item.get("asset") or "")
            candidate_path = Path(raw_asset) if raw_asset else None
            if candidate_path is None or not candidate_path.exists():
                candidate_path = _find_local_sticker_by_prompt(
                    sticker_dir,
                    out / "sticker_cache",
                    query,
                )
                if candidate_path is not None:
                    print(
                        f"[fx] Gemini sticker resolved: scene={scene_index + 1} "
                        f"query={query!r} file={candidate_path.name!r}",
                        flush=True,
                    )
            if candidate_path is None or not candidate_path.exists():
                print(
                    f"[fx] Gemini requested sticker but no local match was found: {query}",
                    flush=True,
                )
                continue
            target = candidate_path
        elif effect_type in {"png", "png_text"}:
            target = out / f"overlay_{len(overlays):02d}.png"
            candidate = _find_png(query, target, client)
            if candidate is None:
                target.unlink(missing_ok=True)
                # Text fallback is allowed ONLY for explicit spoken values
                # such as "930 мл", "$100" or "50%". Never flash arbitrary
                # narration words just because PNG retrieval failed.
                if label:
                    effect_type = "text"
                    query = ""
                    target = None
                    position = "center"
                    animation = "pop" if animation == "fly" else animation
                    size = "hero"
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
            "source": (
                "local_sticker"
                if effect_type == "sticker"
                else candidate.source if candidate is not None
                else "generated_text"
            ),
            "title": (
                Path(str(target)).stem
                if effect_type == "sticker" and target is not None
                else candidate.title if candidate is not None
                else label
            ),
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

    # Post-validation density repair.
    #
    # The first planning pass can suggest 6-8 ideas but several may disappear
    # later because their anchor is invalid, the concept repeats, or Commons
    # has no usable PNG. Count what ACTUALLY survived and fill the missing
    # rhythm only after validation/material lookup.
    if len(overlays) < min_target:
        missing = min_target - len(overlays)
        accepted_summary = [
            {
                "scene": int(effect["scene"]),
                "type": effect.get("type"),
                "query": effect.get("query"),
                "anchor": effect.get("anchor"),
                "start": effect.get("start"),
            }
            for effect in overlays
        ]
        accepted_scene_ids = {int(effect["scene"]) for effect in overlays}
        remaining_rows = [
            row for row in scene_rows
            if int(row["scene"]) not in accepted_scene_ids
        ]

        if remaining_rows:
            repair_prompt = f"""You are repairing an under-edited TikTok/YouTube Short.

Only {len(overlays)} foreground accents survived validation, but this {duration:.1f}s video should have about {min_target} useful accents.

Already accepted effects:
{json.dumps(accepted_summary, ensure_ascii=False)}

Unused scenes:
{json.dumps(remaining_rows, ensure_ascii=False)}

Return ONLY JSON with key effects.
Give {missing + 2} candidates from UNUSED scenes, strongest first.

IMPORTANT:
- Prefer a concrete visible noun/person/object actually SPOKEN in the scene so it can become a PNG cutout.
- query must be a short ENGLISH image-search phrase for that literal object.
- anchor MUST be copied EXACTLY from the scene caption, 1-3 spoken words.
- Do not repeat concepts already accepted.
- If a scene has no concrete noun but has a strong explicit number/value/date, type may be text and label must be copied exactly.
- Do not invent objects.
- Vary animation between fly, drop and pop.
- position: left/right/center.
- size: large/hero.

Schema:
{{"effects":[
  {{"scene":1,"type":"png","query":"shopping cart","anchor":"корзину","label":"","position":"left","animation":"drop","size":"large","duration":0.95}},
  {{"scene":7,"type":"text","query":"","anchor":"50 процентов","label":"50 процентов","position":"center","animation":"pop","size":"hero","duration":0.8}}
]}}
"""
            try:
                if client is None:
                    raise RuntimeError("Gemini unavailable")
                repair_data = client._generate_json([{"text": repair_prompt}], temperature=0.10)
                repair_items = (repair_data.get("effects") or []) if isinstance(repair_data, dict) else []
            except Exception as exc:
                print(f"[fx] post-validation fill unavailable: {exc}", flush=True)
                repair_items = []

            for item in repair_items:
                if len(overlays) >= min_target or not isinstance(item, dict):
                    break
                try:
                    scene_index = int(item.get("scene"))
                except (TypeError, ValueError):
                    continue
                if scene_index < 0 or scene_index >= len(plan.scenes) or scene_index in used_scenes:
                    continue

                scene = plan.scenes[scene_index]
                caption = scene.caption or ""
                anchor = " ".join(str(item.get("anchor") or "").split())[:80]
                if not anchor or anchor.casefold() not in caption.casefold():
                    continue

                anchor_start = _anchor_start(scene, anchor)
                if anchor_start is None:
                    # Exact substring is still grounded; use scene-relative timing.
                    anchor_start = float(scene.start) + 0.12

                if any(abs(anchor_start - old_start) < 0.65 for old_start in used_times):
                    continue

                effect_type = str(item.get("type") or "png").lower()
                if effect_type not in {"png", "png_text", "text"}:
                    effect_type = "png"

                query = _clean_query(str(item.get("query") or ""))
                label = " ".join(str(item.get("label") or "").split())[:28]
                if effect_type in {"png_text", "text"}:
                    if label and not _label_is_spoken(caption, label):
                        label = ""
                    if not label:
                        label = _explicit_quantity(caption) or ""
                    if effect_type == "text" and not label:
                        # A grounded keyword card is acceptable as a last-resort
                        # rhythm repair, but it must be exact spoken text.
                        label = _short_anchor_label(anchor)
                    if effect_type == "text" and not label:
                        continue

                concept = (query or label or anchor).casefold().strip()
                if concept in used_concepts:
                    continue

                timeline_end = max(float(s.end) for s in plan.scenes)
                effect_duration = _clamp_float(item.get("duration"), 1.35, 2.10, 1.70)
                start = max(float(scene.start), anchor_start - 0.02)
                end = min(timeline_end, start + effect_duration)
                if end - start < 0.42:
                    continue

                raw_position = str(item.get("position") or "").lower()
                position = raw_position if raw_position in {"left", "right", "center"} else ("left" if len(overlays) % 2 else "right")
                animation = str(item.get("animation") or "").lower()
                if animation not in {"fly", "drop", "pop"}:
                    animation = ("fly", "drop", "pop")[len(overlays) % 3]
                if overlays and animation == overlays[-1].get("animation"):
                    animation = {"fly": "drop", "drop": "pop", "pop": "fly"}[animation]
                size = str(item.get("size") or "").lower()
                if size not in {"large", "hero"}:
                    size = "large"

                target: Path | None = None
                candidate = None
                if effect_type in {"png", "png_text"} and query:
                    target = out / f"overlay_{len(overlays):02d}.png"
                    candidate = _find_png(query, target, client)

                if effect_type in {"png", "png_text"} and candidate is None:
                    if target is not None:
                        target.unlink(missing_ok=True)
                    # Do not lose the beat again. Fall back to a grounded,
                    # short spoken keyword card only after PNG lookup failed.
                    fallback_label = label or _short_anchor_label(anchor)
                    if not fallback_label:
                        continue
                    effect_type = "text"
                    label = fallback_label
                    query = ""
                    target = None
                    position = "center"
                    animation = "pop" if animation == "fly" else animation
                    size = "large"

                effect = {
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
                overlays.append(effect)
                used_scenes.add(scene_index)
                used_times.append(start)
                used_concepts.add(concept)
                print(
                    f"[fx] density fill {len(overlays)}/{min_target}: scene={scene_index + 1} "
                    f"type={effect_type} anchor={anchor!r} query={query!r}",
                    flush=True,
                )

    # Absolute safety net: fill remaining rhythm locally from the storyboard.
    #
    # Never flash arbitrary verbs/adverbs just to hit a count. Prefer a concrete
    # PNG query already present in the scene plan. Text-only fallback is reserved
    # for explicit spoken quantities/values.
    if len(overlays) < min_target:
        for scene_index, scene in enumerate(plan.scenes):
            if len(overlays) >= min_target:
                break
            if scene_index in used_scenes:
                continue

            caption = scene.caption or ""
            quantity = _explicit_quantity(caption)
            query = _best_local_png_query(scene)

            start = max(float(scene.start) + 0.12, float(scene.start))
            if any(abs(start - old_start) < 0.65 for old_start in used_times):
                continue
            display_end = (
                float(plan.scenes[scene_index + 1].start)
                if scene_index + 1 < len(plan.scenes)
                else float(scene.end)
            )
            end = min(display_end, start + 0.90)
            if end - start < 0.42:
                continue

            # First choice: a concrete PNG from the existing storyboard query.
            if query:
                concept = query.casefold()
                if concept not in used_concepts:
                    target = out / f"overlay_{len(overlays):02d}.png"
                    candidate = _find_png(query, target, client)
                    if candidate is not None:
                        effect = {
                            "scene": scene_index,
                            "type": "png_text" if quantity else "png",
                            "asset": str(target),
                            "query": query,
                            "anchor": quantity or "",
                            "label": quantity or "",
                            "position": "left" if len(overlays) % 2 else "right",
                            "animation": ("fly", "drop", "pop")[len(overlays) % 3],
                            "size": "hero" if quantity else "large",
                            "start": round(start, 3),
                            "end": round(end, 3),
                            "source": candidate.source,
                            "title": candidate.title,
                            "page_url": candidate.page_url,
                            "license": candidate.license,
                        }
                        overlays.append(effect)
                        used_scenes.add(scene_index)
                        used_times.append(start)
                        used_concepts.add(concept)
                        print(
                            f"[fx] local visual fill {len(overlays)}/{min_target}: "
                            f"scene={scene_index + 1} query={query!r}"
                            + (f" label={quantity!r}" if quantity else ""),
                            flush=True,
                        )
                        continue
                    target.unlink(missing_ok=True)

            # Second choice: only a factual value that is literally spoken.
            if quantity:
                concept = ("qty:" + quantity.casefold())
                if concept not in used_concepts:
                    effect = {
                        "scene": scene_index,
                        "type": "text",
                        "asset": "",
                        "query": "",
                        "anchor": quantity,
                        "label": quantity,
                        "position": "center",
                        "animation": "pop",
                        "size": "hero",
                        "start": round(start, 3),
                        "end": round(end, 3),
                        "source": "spoken_quantity",
                        "title": quantity,
                        "page_url": "",
                        "license": "",
                    }
                    overlays.append(effect)
                    used_scenes.add(scene_index)
                    used_times.append(start)
                    used_concepts.add(concept)
                    print(
                        f"[fx] quantity fill {len(overlays)}/{min_target}: "
                        f"scene={scene_index + 1} label={quantity!r}",
                        flush=True,
                    )

    if len(overlays) < min_target:
        print(
            f"[fx] density stopped at {len(overlays)}/{min_target}: "
            "no more grounded visual/value inserts found",
            flush=True,
        )

    (out / "overlays.json").write_text(
        json.dumps({"effects": overlays, "overlays": overlays}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return overlays


def _find_png(query: str, target: Path, client):
    """Find a meme-style foreground visual with a guaranteed visual fallback.

    First prefer a real transparent cutout. If Commons has no good cutout,
    choose a relevant regular image with local CLIP and convert it into a
    rounded meme-card sticker. This keeps the edit dense instead of collapsing
    to zero inserts just because transparent PNG coverage is poor.
    """
    reject_words = {
        "logo", "seal", "emblem", "badge", "flag", "coat of arms",
        "newspaper", "article", "document", "screenshot", "diagram",
        "chart", "map", "poster", "banner", "infographic", "symbol",
    }
    tokens = set(_tokens(query))
    tested = 0

    # 1) Best case: real transparent meme/cutout.
    cutout_searches = [
        f"{query} funny meme sticker transparent png",
        f"{query} funny cartoon transparent png",
        f"{query} reaction sticker transparent png",
        f"{query} cartoon sticker png",
        f"{query} transparent cutout png",
        f"{query} isolated transparent png",
    ]
    cutouts: list[tuple[Any, Path, float]] = []
    for search in cutout_searches:
        try:
            candidates = search_commons(search, limit=30)
        except Exception:
            continue
        for candidate in candidates:
            if (
                candidate.kind != "image"
                or candidate.mime != "image/png"
                or not candidate.download_url
                or candidate.width < 240
                or candidate.height < 240
            ):
                continue
            meta = (candidate.title + " " + candidate.description).casefold()
            if any(word in meta for word in reject_words):
                continue

            tested += 1
            preview = target.with_name(target.stem + f"_cutout_{tested}.png")
            try:
                _download(candidate.download_url, preview)
            except Exception:
                preview.unlink(missing_ok=True)
                continue
            if not _is_real_cutout_png(preview):
                preview.unlink(missing_ok=True)
                continue

            overlap = sum(1 for token in tokens if token in meta)
            style_bonus = 3.0 if any(x in meta for x in ("sticker", "cartoon", "funny", "cutout")) else 0.0
            cutouts.append((candidate, preview, overlap * 4.0 + style_bonus))
            if len(cutouts) >= 8:
                break
        if len(cutouts) >= 8:
            break

    if cutouts:
        scored = _score_overlay_candidates(
            query,
            cutouts,
            client=client,
            prompt=f"funny meme sticker cutout of {query}",
        )
        if scored:
            _, best_candidate, best_preview = scored[0]
            best_preview.replace(target)
            _meme_stickerize(target)
            for _, _, preview in scored[1:]:
                preview.unlink(missing_ok=True)
            for stale in target.parent.glob(target.stem + "_cutout_*.png"):
                stale.unlink(missing_ok=True)
            return best_candidate

    # 2) Fallback: regular relevant image -> rounded meme-card sticker.
    # This is intentionally better than returning None: for Shorts, a clean
    # meme card is more useful than having no foreground beat at all.
    fallback_searches = [
        f"{query} funny",
        f"{query} reaction",
        f"{query} close up",
        query,
    ]
    regular: list[tuple[Any, Path, float]] = []
    seen_urls: set[str] = set()
    for search in fallback_searches:
        try:
            candidates = search_commons(search, limit=30)
        except Exception:
            continue
        for candidate in candidates:
            if (
                candidate.kind != "image"
                or not candidate.download_url
                or candidate.download_url in seen_urls
                or candidate.width < 420
                or candidate.height < 320
            ):
                continue
            seen_urls.add(candidate.download_url)
            meta = (candidate.title + " " + candidate.description).casefold()
            if any(word in meta for word in reject_words):
                continue

            suffix = ".png" if candidate.mime == "image/png" else ".jpg"
            tested += 1
            preview = target.with_name(target.stem + f"_card_{tested}{suffix}")
            try:
                _download(candidate.download_url, preview)
            except Exception:
                preview.unlink(missing_ok=True)
                continue

            overlap = sum(1 for token in tokens if token in meta)
            regular.append((candidate, preview, overlap * 4.0))
            if len(regular) >= 10:
                break
        if len(regular) >= 10:
            break

    if not regular:
        for stale in target.parent.glob(target.stem + "_*"):
            if stale != target:
                stale.unlink(missing_ok=True)
        return None

    scored = _score_overlay_candidates(
        query,
        regular,
        client=client,
        prompt=f"clear funny meme visual of {query}",
    )
    if not scored:
        scored = [
            (base_score, candidate, preview)
            for candidate, preview, base_score in regular
        ]
        scored.sort(key=lambda item: item[0], reverse=True)

    _, best_candidate, best_preview = scored[0]
    _meme_cardize(best_preview, target)

    for _, _, preview in scored:
        if preview.exists():
            preview.unlink(missing_ok=True)
    for stale in target.parent.glob(target.stem + "_card_*"):
        stale.unlink(missing_ok=True)

    return best_candidate if target.exists() else None


def _score_overlay_candidates(
    query: str,
    candidates: list[tuple[Any, Path, float]],
    *,
    client,
    prompt: str,
) -> list[tuple[float, Any, Path]]:
    """Rank overlay images visually; local CLIP is the default judge."""
    scored: list[tuple[float, Any, Path]] = []
    if client is not None:
        for candidate, preview, base_score in candidates:
            score = 50.0 + base_score
            quality = 50.0
            accepted = True
            try:
                probe_scene = Scene(
                    start=0.0,
                    end=1.0,
                    query=query,
                    caption=query,
                    visual_description=prompt,
                    visual_mode="image",
                    source_mode="generic_image",
                    motion_preset="none",
                )
                judgement = client.judge_visual(
                    probe_scene,
                    preview,
                    candidate_title=candidate.title,
                    source="commons meme overlay",
                    match_level="exact",
                )
                if judgement is not None:
                    score = float(judgement.score) + base_score
                    quality = float(judgement.quality_score)
                    accepted = bool(judgement.accept and score >= 45 and quality >= 40)
            except Exception:
                pass
            if accepted:
                scored.append((score * 0.68 + quality * 0.32, candidate, preview))
    else:
        try:
            from .multimodal import get_clip_ranker
            ranker = get_clip_ranker()
            sims = ranker.score_images(prompt, [item[1] for item in candidates])
            for (candidate, preview, base_score), sim in zip(candidates, sims):
                scored.append((float(sim) * 100.0 + base_score, candidate, preview))
        except Exception:
            scored = [
                (base_score, candidate, preview)
                for candidate, preview, base_score in candidates
            ]

    scored.sort(key=lambda item: item[0], reverse=True)
    return scored


def _meme_cardize(source: Path, target: Path) -> None:
    """Turn any regular image into a Shorts-style meme sticker card."""
    try:
        from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

        with Image.open(source) as opened:
            image = opened.convert("RGB")

        # Center square crop keeps the subject big on a phone.
        side = min(image.width, image.height)
        left = max(0, (image.width - side) // 2)
        top = max(0, (image.height - side) // 2)
        image = image.crop((left, top, left + side, top + side))
        image.thumbnail((780, 780), Image.Resampling.LANCZOS)
        image = ImageEnhance.Contrast(image).enhance(1.08)
        image = ImageEnhance.Color(image).enhance(1.10)

        radius = max(24, int(min(image.size) * 0.12))
        border = max(16, int(min(image.size) * 0.035))
        shadow_pad = border * 3

        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.rounded_rectangle(
            (0, 0, image.width - 1, image.height - 1),
            radius=radius,
            fill=255,
        )

        card = Image.new("RGBA", image.size, (0, 0, 0, 0))
        card.paste(image.convert("RGBA"), (0, 0), mask)

        canvas = Image.new(
            "RGBA",
            (
                image.width + shadow_pad * 2,
                image.height + shadow_pad * 2,
            ),
            (0, 0, 0, 0),
        )

        shadow_mask = mask.filter(ImageFilter.GaussianBlur(max(6, border * 0.75)))
        shadow_rgba = Image.new("RGBA", image.size, (0, 0, 0, 115))
        shadow_rgba.putalpha(shadow_mask.point(lambda a: int(a * 0.52)))
        canvas.alpha_composite(
            shadow_rgba,
            (shadow_pad + border // 2, shadow_pad + border // 2),
        )

        outline_mask = mask.filter(ImageFilter.MaxFilter(border * 2 + 1))
        outline = Image.new("RGBA", image.size, (255, 255, 255, 255))
        outline.putalpha(outline_mask)
        canvas.alpha_composite(outline, (shadow_pad, shadow_pad))
        canvas.alpha_composite(card, (shadow_pad, shadow_pad))
        canvas.save(target)
    except Exception:
        target.unlink(missing_ok=True)



def _is_real_cutout_png(path: Path) -> bool:
    """Require meaningful transparency and reject full rectangular screenshots."""
    try:
        from PIL import Image
        with Image.open(path) as image:
            rgba = image.convert("RGBA")
            alpha = rgba.getchannel("A")
            extrema = alpha.getextrema()
            if not extrema or extrema[0] >= 245:
                return False
            hist = alpha.histogram()
            total = max(1, rgba.width * rgba.height)
            transparent = sum(hist[:220]) / total
            opaque = sum(hist[245:]) / total
            bbox = alpha.getbbox()
            if bbox is None:
                return False
            bw = max(1, bbox[2] - bbox[0])
            bh = max(1, bbox[3] - bbox[1])
            occupancy = (bw * bh) / total

            # A meme sticker should have visible transparency around the object.
            # Rectangular photos/articles usually have ~0% transparent pixels.
            return (
                0.06 <= transparent <= 0.92
                and opaque >= 0.05
                and 0.08 <= occupancy <= 0.96
            )
    except Exception:
        return False


def _meme_stickerize(path: Path) -> None:
    """Add thick white outline + soft dark shadow around a transparent cutout."""
    try:
        from PIL import Image, ImageFilter
        with Image.open(path) as opened:
            image = opened.convert("RGBA")

        alpha = image.getchannel("A")
        bbox = alpha.getbbox()
        if bbox:
            image = image.crop(bbox)
            alpha = image.getchannel("A")

        min_side = max(1, min(image.width, image.height))
        outline_px = max(5, min(15, int(min_side * 0.035)))
        # MaxFilter requires an odd kernel size.
        kernel = outline_px * 2 + 1
        if kernel % 2 == 0:
            kernel += 1

        grown = alpha.filter(ImageFilter.MaxFilter(kernel))
        shadow = grown.filter(ImageFilter.GaussianBlur(max(2, outline_px * 0.65)))

        pad = outline_px * 3
        canvas = Image.new(
            "RGBA",
            (image.width + pad * 2, image.height + pad * 2),
            (0, 0, 0, 0),
        )

        shadow_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        shadow_rgba = Image.new("RGBA", image.size, (0, 0, 0, 115))
        shadow_rgba.putalpha(shadow.point(lambda a: int(a * 0.46)))
        shadow_layer.alpha_composite(
            shadow_rgba,
            (pad + outline_px // 2, pad + outline_px // 2),
        )
        canvas.alpha_composite(shadow_layer)

        outline_layer = Image.new("RGBA", image.size, (255, 255, 255, 255))
        outline_layer.putalpha(grown)
        canvas.alpha_composite(outline_layer, (pad, pad))
        canvas.alpha_composite(image, (pad, pad))
        canvas.save(path)
    except Exception:
        return



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

def _best_local_png_query(scene: Scene) -> str:
    """Choose a concrete object query for a foreground cutout.

    Material Brain queries often describe an action ("customer checking price
    tag"). For a foreground insert we want the actual prop ("price tag"), not
    the whole stock-video sentence.
    """
    candidates: list[str] = []

    hay = " ".join([
        str(getattr(scene, "caption", "") or ""),
        str(getattr(scene, "query", "") or ""),
        " ".join(getattr(scene, "search_queries", None) or []),
    ]).casefold()
    concrete_rules = [
        (("milk", "молок"), "milk carton"),
        (("price tag", "ценник", "цена"), "price tag"),
        (("shopping cart", "корзин", "тележ"), "shopping cart"),
        (("package", "packaging", "упаков"), "product package"),
        (("factory", "production line", "производ"), "cardboard box"),
        (("phone", "smartphone", "телефон"), "smartphone"),
        (("money", "cash", "деньг", "доллар", "грив"), "cash money"),
        (("car", "автомоб"), "car"),
        (("dog", "собак"), "dog"),
        (("cat", "кош"), "cat"),
    ]
    for needles, query in concrete_rules:
        if any(needle in hay for needle in needles):
            return query

    entities = getattr(scene, "required_entities", None)
    if isinstance(entities, list):
        candidates.extend(str(value).strip() for value in entities if str(value).strip())

    search_queries = getattr(scene, "search_queries", None)
    if isinstance(search_queries, list):
        candidates.extend(str(value).strip() for value in search_queries if str(value).strip())

    scene_query = str(getattr(scene, "query", "") or "").strip()
    if scene_query:
        candidates.append(scene_query)

    hard_bad_phrases = {
        "supermarket aisle", "grocery store aisle", "shopping scene",
        "historical archive", "street scene", "crowd scene",
        "wide shot", "close up", "closeup",
    }
    soft_bad_tokens = {
        "scene", "background", "interior", "exterior", "detail", "closeup",
        "close", "wide", "shot", "aisle", "store", "supermarket", "grocery",
        "shopping", "consumer", "products", "product", "price", "pricing",
        "concept", "illustration", "archive", "historical",
    }

    ranked: list[tuple[int, str]] = []
    seen: set[str] = set()
    for value in candidates:
        cleaned = _clean_query(value)
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        seen.add(key)
        words = cleaned.split()
        if not (1 <= len(words) <= 5):
            continue
        if key in hard_bad_phrases:
            continue

        bad_count = sum(1 for word in words if word.casefold() in soft_bad_tokens)
        good_count = len(words) - bad_count
        if good_count <= 0:
            continue

        score = 18 - len(words) * 2 + good_count * 4 - bad_count * 7
        if len(words) <= 3:
            score += 5
        ranked.append((score, cleaned))

    if not ranked:
        return ""
    ranked.sort(key=lambda item: item[0], reverse=True)
    best = ranked[0][1]
    # Foreground meme inserts must be object-like, not full stock-video actions.
    action_words = {
        "customer", "person", "shopper", "looking", "checking", "holding",
        "taking", "pushing", "comparing", "factory", "production", "line",
    }
    words = best.casefold().split()
    if len(words) > 3 or sum(1 for word in words if word in action_words) >= 2:
        return ""
    return best


def _local_effect_candidates(plan: ShotPlan, budget: int) -> list[dict[str, Any]]:
    """Local foreground accents with deliberate Shorts pacing.

    Prefer 3-6 concrete, non-repeating inserts distributed across the whole
    narration rather than filling the first few scenes. Quantities get hero
    emphasis; concrete props get PNG cutouts.
    """
    raw: list[dict[str, Any]] = []
    seen: set[str] = set()

    for index, scene in enumerate(plan.scenes):
        caption = (scene.caption or "").strip()
        if not caption:
            continue
        quantity = _explicit_quantity(caption)
        query = _best_local_png_query(scene)

        if quantity:
            key = "qty:" + quantity.casefold()
            if key not in seen:
                raw.append({
                    "scene": index,
                    "type": "png_text" if query else "text",
                    "query": query,
                    "anchor": quantity,
                    "label": quantity,
                    "position": "right" if index % 2 == 0 else "left",
                    "animation": "fly",
                    "size": "hero",
                    "duration": 1.75,
                    "_local": True,
                    "_strength": 3,
                })
                seen.add(key)
                continue

        if query:
            key = "q:" + query.casefold()
            if key not in seen:
                raw.append({
                    "scene": index,
                    "type": "png",
                    "query": query,
                    "anchor": "",
                    "label": "",
                    "position": "left" if index % 2 else "right",
                    "animation": "fly",
                    "size": "hero" if index % 4 == 0 else "large",
                    "duration": 1.70,
                    "_local": True,
                    "_strength": 2,
                })
                seen.add(key)

    if len(raw) <= budget:
        return raw

    # Spread accents over the full timeline. Pick the strongest candidate close
    # to evenly spaced target positions, then sort back into chronological order.
    total = max(1, len(plan.scenes) - 1)
    chosen: list[dict[str, Any]] = []
    remaining = raw[:]
    for slot in range(budget):
        target = (slot + 0.5) / budget * total
        best = min(
            remaining,
            key=lambda item: (
                abs(int(item["scene"]) - target) - 0.20 * int(item.get("_strength", 1)),
                int(item["scene"]),
            ),
        )
        chosen.append(best)
        remaining.remove(best)
        if not remaining:
            break

    chosen.sort(key=lambda item: int(item["scene"]))
    return chosen




_STICKER_EXTS = {".gif", ".png", ".webp", ".jpg", ".jpeg"}


def _find_local_sticker_by_prompt(
    sticker_dir: str | Path | None,
    cache_dir: Path,
    prompt: str,
) -> Path | None:
    """Resolve a Gemini reaction description to the best local sticker/GIF."""
    if not sticker_dir or not prompt.strip():
        return None
    root = Path(sticker_dir)
    if not root.exists() or not root.is_dir():
        return None

    assets = [
        path for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in _STICKER_EXTS
    ][:500]
    if not assets:
        return None

    cache_dir.mkdir(parents=True, exist_ok=True)
    pairs: list[tuple[Path, Path]] = []
    for asset in assets:
        preview = _sticker_preview(asset, cache_dir)
        if preview is not None:
            pairs.append((asset, preview))
    if not pairs:
        return None

    try:
        from .multimodal import get_clip_ranker
        ranker = get_clip_ranker()
        scores = ranker.score_images(
            f"{prompt.strip()} emoji sticker reaction",
            [preview for _, preview in pairs],
        )
    except Exception:
        return None

    if not scores:
        return None
    ranking = sorted(
        zip(scores, [asset for asset, _ in pairs]),
        key=lambda pair: float(pair[0]),
        reverse=True,
    )
    best_score, best_asset = ranking[0]
    return best_asset if float(best_score) > 0 else None


def _local_sticker_effect_candidates(
    plan: ShotPlan,
    sticker_dir: str | Path | None,
    cache_dir: Path,
    *,
    budget: int,
) -> list[dict[str, Any]]:
    """Pick reaction stickers by actually looking at the user's sticker pack.

    Filenames may be meaningless (e.g. AnimatedEmojis-512px-83.gif), so local
    CLIP scores preview frames against reaction prompts inferred from narration.
    """
    if budget <= 0:
        return []
    if not sticker_dir:
        print("[fx] no --sticker-dir supplied; local sticker pack disabled", flush=True)
        return []
    root = Path(sticker_dir)
    if not root.exists() or not root.is_dir():
        print(f"[fx] sticker pack not found: {root}", flush=True)
        return []

    assets = [
        path for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in _STICKER_EXTS
    ][:500]
    if not assets:
        print(f"[fx] sticker pack is empty: {root}", flush=True)
        return []

    cache_dir.mkdir(parents=True, exist_ok=True)
    preview_assets: list[tuple[Path, Path]] = []
    for asset in assets:
        preview = _sticker_preview(asset, cache_dir)
        if preview is not None:
            preview_assets.append((asset, preview))

    if not preview_assets:
        print("[fx] sticker pack previews unavailable", flush=True)
        return []

    try:
        from .multimodal import get_clip_ranker
        ranker = get_clip_ranker()
    except Exception as exc:
        print(f"[fx] sticker CLIP unavailable: {exc}", flush=True)
        return []

    scene_rows: list[tuple[int, int, str]] = []
    for index, scene in enumerate(plan.scenes):
        strength, prompt = _reaction_prompt(scene)
        if strength > 0:
            scene_rows.append((strength, index, prompt))

    if not scene_rows:
        return []

    # Strongest hooks first, but avoid packing all reactions into adjacent beats.
    scene_rows.sort(key=lambda item: (-item[0], item[1]))
    selected: list[dict[str, Any]] = []
    used_assets: set[Path] = set()
    used_scenes: list[int] = []

    for strength, scene_index, prompt in scene_rows:
        if len(selected) >= budget:
            break
        if any(abs(scene_index - other) < 2 for other in used_scenes):
            continue

        paths = [preview for asset, preview in preview_assets if asset not in used_assets]
        candidates = [asset for asset, preview in preview_assets if asset not in used_assets]
        if not paths:
            break
        try:
            scores = ranker.score_images(prompt, paths)
        except Exception:
            continue
        if not scores:
            continue

        ranking = sorted(
            zip(scores, candidates),
            key=lambda pair: float(pair[0]),
            reverse=True,
        )
        best_score, best_asset = ranking[0]
        if float(best_score) <= 0:
            continue

        scene = plan.scenes[scene_index]
        selected.append({
            "scene": scene_index,
            "type": "sticker",
            "asset": str(best_asset),
            "query": prompt,
            "anchor": "",
            "label": "",
            "position": "left" if len(selected) % 2 else "right",
            "animation": "fly",
            "size": "hero",
            "duration": 1.85,
            "_local": True,
            "_strength": strength + 2,
        })
        used_assets.add(best_asset)
        used_scenes.append(scene_index)
        print(
            f"[fx] sticker pick: scene={scene_index + 1} "
            f"prompt={prompt!r} file={best_asset.name!r} score={float(best_score):.3f}",
            flush=True,
        )

    return selected


def _reaction_prompt(scene: Scene) -> tuple[int, str]:
    text = " ".join([
        str(getattr(scene, "caption", "") or ""),
        str(getattr(scene, "query", "") or ""),
    ]).casefold()

    rules = [
        (4, ("обман", "хитр", "трюк", "скрыва", "не замеч", "secret", "trick", "deceiv"),
         "suspicious side eye skeptical reaction emoji sticker"),
        (4, ("шок", "неожидан", "оказалось", "вдруг", "wtf", "shock", "sudden"),
         "shocked surprised wide eyes reaction emoji sticker"),
        (4, ("цена", "дороже", "деньг", "стоим", "грн", "доллар", "price", "money", "expensive"),
         "shocked money skeptical reaction emoji sticker"),
        (3, ("меньше", "уменьш", "930", "объем", "объём", "упаков", "shrink", "smaller", "package"),
         "confused disappointed suspicious reaction emoji sticker"),
        (3, ("почему", "стран", "непонят", "сомне", "confus", "weird", "why"),
         "confused thinking skeptical reaction emoji sticker"),
        (3, ("смеш", "ахах", "лол", "прикол", "funny", "lol", "joke"),
         "laughing crying funny reaction emoji sticker"),
        (3, ("плохо", "груст", "обид", "потер", "sad", "bad", "loss"),
         "sad crying disappointed reaction emoji sticker"),
        (2, ("люб", "круто", "кайф", "рад", "heart", "love", "happy"),
         "happy heart eyes smiling reaction emoji sticker"),
    ]
    for strength, needles, prompt in rules:
        if any(needle in text for needle in needles):
            return strength, prompt

    if _explicit_quantity(str(getattr(scene, "caption", "") or "")):
        return 2, "surprised thinking reaction emoji sticker"
    return 0, ""


def _sticker_preview(asset: Path, cache_dir: Path) -> Path | None:
    digest = hashlib.sha1(str(asset.resolve()).encode("utf-8", errors="ignore")).hexdigest()[:16]
    target = cache_dir / f"{digest}.png"
    if target.exists() and target.stat().st_size > 1024:
        return target

    suffix = asset.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        try:
            from PIL import Image
            with Image.open(asset) as opened:
                image = opened.convert("RGBA")
                image.thumbnail((384, 384))
                image.save(target)
            return target
        except Exception:
            return None

    if suffix == ".gif":
        try:
            completed = subprocess.run(
                [
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-ss", "0.35", "-i", str(asset),
                    "-frames:v", "1",
                    "-vf", "scale=384:384:force_original_aspect_ratio=decrease",
                    str(target),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=20,
                check=False,
            )
            if completed.returncode == 0 and target.exists() and target.stat().st_size > 1024:
                return target
        except Exception:
            pass
    target.unlink(missing_ok=True)
    return None


def _fallback_scene_keyword(caption: str) -> str:
    """Pick one grounded 1-2 word callout from narration as an emergency beat.

    This is intentionally conservative and only exists so a 25 second Short
    never collapses to one or two foreground accents because image retrieval
    failed.
    """
    words = re.findall(r"[A-Za-zА-Яа-яЁё0-9%$₴€]+", caption)
    if not words:
        return ""

    stop = {
        "это", "этот", "эта", "эти", "как", "что", "чтобы", "когда", "где",
        "вот", "там", "тут", "уже", "ещё", "еще", "просто", "очень", "даже",
        "если", "или", "для", "его", "она", "они", "оно", "тебя", "тебе",
        "меня", "мне", "мы", "вы", "не", "ни", "на", "в", "во", "и", "а",
        "но", "по", "из", "за", "до", "от", "с", "со", "к", "ко", "же",
        "бы", "быть", "был", "была", "были", "есть", "the", "a", "an", "and",
        "or", "to", "of", "in", "on", "for", "with", "this", "that", "it",
    }

    content = [
        word for word in words
        if word.casefold() not in stop and (len(word) >= 4 or any(ch.isdigit() for ch in word))
    ]
    if not content:
        return ""

    # Prefer explicit numbers/values, otherwise a longer content word.
    numeric = [word for word in content if any(ch.isdigit() for ch in word)]
    chosen = numeric[0] if numeric else max(content, key=len)
    return chosen[:22]


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
