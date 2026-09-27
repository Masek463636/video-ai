from __future__ import annotations

import re
from dataclasses import replace

from .director import _join_words
from .models import Scene, ShotPlan, Word


_REVEAL_RE = re.compile(
    r"\b(?:но|и тут|оказалось|вдруг|однако|только вот|на самом деле|"
    r"but|then|turns out|suddenly|however|actually)\b",
    flags=re.IGNORECASE,
)


def classify_pace(
    words: list[Word],
    *,
    caption: str = "",
    gap_after: float = 0.0,
) -> str:
    text = re.sub(r"\s+", " ", caption or _join_words(words)).strip()
    duration = (
        max(0.0, words[-1].end - words[0].start)
        if words
        else 0.0
    )
    if gap_after >= 0.32 or _REVEAL_RE.search(text):
        return "reveal"
    compact_len = len(re.sub(r"\s+", "", text))
    if duration <= 0.90 or compact_len <= 16:
        return "fast"
    return "normal"


def _split_index(scene: Scene) -> int | None:
    words = scene.caption_words
    if len(words) < 5:
        return None
    target = scene.start + min(0.78, scene.duration * 0.56)
    best: tuple[float, int] | None = None
    for index in range(2, len(words) - 2):
        boundary = words[index].start
        left = boundary - scene.start
        right = scene.end - boundary
        if left < 0.42 or right < 0.42:
            continue
        score = abs(boundary - target)
        if best is None or score < best[0]:
            best = (score, index)
    return best[1] if best else None


def _rotated_queries(scene: Scene) -> tuple[str, list[str]]:
    queries = [q for q in scene.search_queries if q]
    if len(queries) <= 1:
        return scene.query, queries
    rotated = [*queries[1:], queries[0]]
    return rotated[0], rotated


def apply_premium_pacing(plan: ShotPlan) -> list[int]:
    """Classify semantic pace and split only clearly overlong fast beats.

    This runs before asset retrieval, so any new beat gets its own material.
    Reveal beats are deliberately left longer instead of being chopped by a
    fixed timer.
    """
    original = list(plan.scenes)
    output: list[Scene] = []
    changed: list[int] = []

    for index, scene in enumerate(original):
        next_scene = original[index + 1] if index + 1 < len(original) else None
        if scene.caption_words and next_scene and next_scene.caption_words:
            gap_after = max(
                0.0,
                next_scene.caption_words[0].start
                - scene.caption_words[-1].end,
            )
        else:
            gap_after = 0.0

        pace = classify_pace(
            scene.caption_words,
            caption=scene.caption or "",
            gap_after=gap_after,
        )

        should_split = (
            pace == "fast"
            and not scene.semantic_lock
            and scene.duration > 1.05
            and len(scene.caption_words) >= 5
        )
        split_at = _split_index(scene) if should_split else None

        if split_at is None:
            output.append(replace(scene, pace_class=pace))
            continue

        left_words = list(scene.caption_words[:split_at])
        right_words = list(scene.caption_words[split_at:])
        if not left_words or not right_words:
            output.append(replace(scene, pace_class=pace))
            continue

        left_caption = _join_words(left_words)
        right_caption = _join_words(right_words)
        right_query, right_queries = _rotated_queries(scene)

        left = replace(
            scene,
            end=round(left_words[-1].end, 3),
            caption=left_caption,
            caption_words=left_words,
            pace_class=classify_pace(
                left_words,
                caption=left_caption,
                gap_after=max(
                    0.0,
                    right_words[0].start - left_words[-1].end,
                ),
            ),
        )
        right = replace(
            scene,
            start=round(right_words[0].start, 3),
            caption=right_caption,
            caption_words=right_words,
            query=right_query,
            search_queries=right_queries,
            pace_class=classify_pace(
                right_words,
                caption=right_caption,
                gap_after=gap_after,
            ),
        )
        output.extend([left, right])
        changed.append(len(output) - 2)

    plan.scenes = output

    print(
        f"[premium-v2] semantic pacing: {len(original)} -> "
        f"{len(output)} beats; split={len(changed)}",
        flush=True,
    )
    return changed
