"""Viral / Darwin foreground attention engine."""

from __future__ import annotations

import json
import math
from pathlib import Path


def build_viral_overlays(
    plan,
    out_dir,
    *,
    max_overlays=0,
    use_gemini=True,
    sticker_dir=None,
    base_video=None,
):

    from .dynamic_reactions import build_reactions
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    duration = max((float(s.end) for s in plan.scenes), default=0.)
    if duration <= 0:
        return []
    budget = max_overlays if max_overlays > 0 else max(4, min(14, round(duration / 2.5)))
    # Preserve the existing asset retrieval and visual verification. The renderer
    # routes memes to the primary video track and emojis to the overlay track.
    effects = build_reactions(plan, out / "reactions", max_overlays=budget,
        use_gemini=use_gemini, sticker_dir=sticker_dir, montage=True)
    report = {"style":"viral", "budget":budget, "overlays":effects,
              "effects":effects, "mode":"meme_shots_and_emoji_overlays"}
    for name in ("viral_overlays.json", "overlays.json"):
        (out / name).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return effects


def _attention_accents(
    plan,
    out,
    *,
    base_video,
    budget,
    use_gemini,
    avoid_scenes,
):

    if (
        budget <= 0
        or not use_gemini
        or not base_video
    ):
        return []

    base = Path(base_video)

    if not base.is_file():
        return []

    try:
        from .gemini_ai import (
            get_gemini_client,
        )

        client = get_gemini_client()

    except Exception:
        client = None

    if client is None:
        return []

    from .shorts_fx import _anchor_start
    from .composition_review import (
        boxes,
        frames,
    )

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    scene_rows = [
        {
            "scene": index,
            "start": round(
                float(scene.start),
                3,
            ),
            "end": round(
                float(scene.end),
                3,
            ),
            "caption": scene.caption or "",
        }
        for index, scene in enumerate(plan.scenes)
        if index not in avoid_scenes
    ]

    if not scene_rows:
        return []

    prompt = f"""
You are the attention editor for an extremely fast
Viral/Darwin-style vertical YouTube Short.

UNUSED SCENES:
{json.dumps(scene_rows, ensure_ascii=False)}

Choose up to {budget} moments where a RED ARROW or
RED CIRCLE would genuinely help the viewer notice
a concrete visible detail in the underlying B-roll.

Return ONLY JSON:

{{
  "effects": [
    {{
      "scene": 2,
      "type": "arrow",
      "anchor": "exact spoken words",
      "target": "phone in the person's hand",
      "reason": "the narration calls out the phone"
    }}
  ]
}}

RULES:
- type is ONLY arrow or circle.
- anchor MUST be copied EXACTLY from that caption.
- target describes ONE concrete visible detail.
- arrow = reveal / person / object / surprising detail.
- circle = small easy-to-miss detail.
- Never point at subtitles, logos or empty space.
- Never invent a target just because narration mentions it.
- Fewer strong accents are better than fake accents.
- Never use two attention graphics in the same scene.
""".strip()

    try:
        data = client._generate_json(
            [{"text": prompt}],
            temperature=.06,
        )

    except Exception as exc:

        print(
            "[viral] attention planner failed: "
            + type(exc).__name__,
            flush=True,
        )

        return []

    if (
        isinstance(data, list)
        and len(data) == 1
        and isinstance(data[0], dict)
    ):
        data = data[0]

    rows = (
        data.get("effects", [])
        if isinstance(data, dict)
        else []
    )

    if not isinstance(rows, list):
        return []

    accepted = []
    decisions = []
    used = set(avoid_scenes)

    timeline_end = max(
        float(scene.end)
        for scene in plan.scenes
    )

    for raw in rows:

        if len(accepted) >= budget:
            break

        if not isinstance(raw, dict):
            continue

        scene_index = raw.get("scene")

        if (
            type(scene_index) is not int
            or not 0 <= scene_index < len(plan.scenes)
            or scene_index in used
        ):
            continue

        kind = str(
            raw.get("type") or ""
        ).strip().lower()

        if kind not in {
            "arrow",
            "circle",
        }:
            continue

        scene = plan.scenes[scene_index]

        anchor = " ".join(
            str(
                raw.get("anchor") or ""
            ).split()
        )[:80]

        target = " ".join(
            str(
                raw.get("target") or ""
            ).split()
        )[:180]

        import re
        normalize = lambda text: " ".join(re.findall(r"\w+", text.casefold().replace("ё", "е")))
        if not target or not normalize(anchor) or (" " + normalize(anchor) + " ") not in (" " + normalize(scene.caption or "") + " "):
            continue

        start = _anchor_start(
            scene,
            anchor,
        )

        if (
            start is None
            or not math.isfinite(start)
        ):
            decisions.append({
                "proposal": raw,
                "status": "anchor_not_found",
            })

            continue

        end = min(float(scene.end), timeline_end, start + (.82 if kind == "arrow" else .90))
        if end - start < .35:
            continue
        locator_prompt = (
            "Inspect these THREE actual rendered frames in chronological order. "
            f"Narration beat: {scene.caption or ''}. "
            f"Requested target: {target}. "
            "Find it ONLY if genuinely visible in ALL three frames. Return one box for each frame. "
            "Do not infer it from narration. "
            "Return exactly one JSON object "
            '{"found":true/false,'
            '"target_boxes":[[x,y,width,height],[x,y,width,height],[x,y,width,height]],'
            '"confidence":0.0,'
            '"reason":"visible evidence"}. '
            "Coordinates are normalized 0..1. "
            "If absent, ambiguous, mostly hidden, "
            "or only subtitle text: found=false."
        )

        try:

            frame_dir = out / "frames"
            frame_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            image_parts = frames(
                base,
                [start + (end-start)*f for f in (.08, .5, .92)],
                frame_dir,
                f"attention_{scene_index}",
            )

            verdict = client._generate_json(
                [
                    {
                        "text":
                            locator_prompt
                    }
                ]
                + image_parts,
                temperature=.01,
            )

            if (
                isinstance(verdict, list)
                and len(verdict) == 1
                and isinstance(
                    verdict[0],
                    dict,
                )
            ):
                verdict = verdict[0]

            if (
                not isinstance(
                    verdict,
                    dict,
                )
                or verdict.get("found")
                is not True
            ):
                decisions.append({
                    "proposal": raw,
                    "status":
                        "target_not_found",
                    "verdict": verdict,
                })

                continue

            confidence = float(
                verdict.get(
                    "confidence",
                    0.0,
                )
            )

            if (
                not math.isfinite(
                    confidence
                )
                or confidence < .60
            ):
                decisions.append({
                    "proposal": raw,
                    "status":
                        "low_confidence",
                    "verdict": verdict,
                })

                continue

            tracked_boxes = boxes(verdict.get("target_boxes"))
            if len(tracked_boxes) != 3:
                raise ValueError("Three target boxes required")
            target_box = tracked_boxes[1]
            cx, cy = target_box[0]+target_box[2]/2, target_box[1]+target_box[3]/2
            if any(abs(b[0]+b[2]/2-cx) > .04 or abs(b[1]+b[3]/2-cy) > .04 for b in tracked_boxes):
                decisions.append({"proposal": raw, "status": "moving_target"})
                continue

            # Caption-heavy lower area.
            if (
                target_box[1] + target_box[3] / 2 > .78
                or (target_box[1] < .66 and target_box[1] + target_box[3] > .50)
            ):
                decisions.append({
                    "proposal": raw,
                    "status":
                        "subtitle_zone",
                })

                continue

        except Exception as exc:

            decisions.append({
                "proposal": raw,
                "status":
                    "review_error",
                "error":
                    type(exc).__name__,
            })

            continue

        effect = {
            "scene": scene_index,
            "type": kind,
            "asset": "",
            "query": target,
            "anchor": anchor,
            "label": "",
            "position": "center",
            "animation": "pop",
            "size": "hero",
            "start": round(
                max(0.0, start),
                3,
            ),
            "end": round(
                end,
                3,
            ),
            "target_box": target_box,
            "source":
                "gemini_attention",
            "attention_verified":
                True,
            "reason": str(
                raw.get("reason")
                or ""
            )[:300],
        }

        accepted.append(effect)
        used.add(scene_index)

        decisions.append({
            "proposal": raw,
            "status": "selected",
            "target_box":
                target_box,
        })

    (out / "attention.json").write_text(
        json.dumps(
            {
                "planner": data,
                "decisions": decisions,
                "effects": accepted,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return accepted


def pace_viral_accents(overlays, spans):
    """Let each new shot read before an accent; never carry one over a cut.

    At most 0.32s of delay keeps the spoken anchor recognizable. Targets already
    located in preview frames are never retimed; omit them if too close to a cut.
    """
    result = []
    for item in overlays:
        if not isinstance(item, dict):
            continue
        try:
            start, end = float(item['start']), float(item['end'])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            continue
        span = next(((a, b) for a, b in spans if a <= start < b), None)
        if span is None:
            continue
        a, b = span
        kind = item.get('type')
        if kind in ('arrow', 'circle'):
            if start < a + .32 or end > b - .10:
                continue
            new_start, new_end = start, end
        else:
            new_start = max(start, a + .32)
            new_end = min(new_start + end - start, b - .10)
        minimum = .65 if kind == 'sticker' else .35
        if new_end - new_start < minimum - 1e-6:
            continue
        result.append(dict(item, start=round(new_start, 3), end=round(new_end, 3)))
    return result


def place_viral_overlays(
    overlays,
    duration,
    *,
    spans=None,
):
    if spans is not None:
        overlays = pace_viral_accents(overlays, spans)


    from .dynamic_reactions import (
        place_reactions,
    )

    stickers = [
        item
        for item in overlays
        if (
            isinstance(item, dict)
            and item.get("type")
            == "sticker"
        )
    ]

    placed = place_reactions(
        stickers,
        duration,
    )

    for item in overlays:

        if not isinstance(item, dict):
            continue

        kind = str(
            item.get("type") or ""
        )

        if kind == "sticker":
            continue

        try:
            start = float(
                item.get("start", 0)
            )

            end = float(
                item.get("end", start)
            )

        except (TypeError, ValueError):
            continue

        if not (
            math.isfinite(start)
            and math.isfinite(end)
        ):
            continue

        start = max(
            0.0,
            start,
        )

        end = min(
            float(duration),
            end,
        )

        if end - start < .30:
            continue

        if kind in {
            "png",
            "png_text",
        }:

            if not Path(
                str(
                    item.get("asset")
                    or ""
                )
            ).is_file():
                continue

        elif kind == "text":

            if not str(
                item.get("label")
                or ""
            ).strip():
                continue

        elif kind in {
            "arrow",
            "circle",
        }:

            if (
                item.get("source")
                != "gemini_attention"
                or item.get(
                    "attention_verified"
                )
                is not True
            ):
                continue

            box = item.get(
                "target_box"
            )

            if (
                not isinstance(box, list)
                or len(box) != 4
            ):
                continue

            try:
                x, y, w, h = [
                    float(v)
                    for v in box
                ]

            except (
                TypeError,
                ValueError,
            ):
                continue

            if not all(
                math.isfinite(v)
                for v
                in (x, y, w, h)
            ):
                continue

            if (
                x < 0
                or y < 0
                or w <= 0
                or h <= 0
                or x + w > 1.001
                or y + h > 1.001
            ):
                continue

        else:
            continue

        # Reactions take priority; do not stack decorative accents over them.
        if any(start < other["end"] + .15 and end + .15 > other["start"] for other in placed):
            continue
        placed.append(
            dict(
                item,
                start=start,
                end=end,
            )
        )

    placed.sort(
        key=lambda item:
            float(
                item.get(
                    "start",
                    0.0,
                )
            )
    )

    return placed
