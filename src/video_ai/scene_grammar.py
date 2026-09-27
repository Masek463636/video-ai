from __future__ import annotations

import re
import zlib

from .models import ShotPlan


_VALID = {
    "clean",
    "reaction",
    "focus_zoom",
    "arrow",
    "circle",
    "big_number",
    "png_cutout",
    "text_behind",
    "parallax",
    "split_screen",
    "freeze_frame",
    "spotlight",
    "blur_background",
    "before_after",
    "stacked_cards",
}

_HERO = {
    "reaction",
    "arrow",
    "circle",
    "big_number",
    "png_cutout",
    "freeze_frame",
    "split_screen",
    "before_after",
    "stacked_cards",
}

_LOCK_SAFE = {"clean", "focus_zoom", "big_number"}


def _seed(scene, index: int) -> int:
    text = f"{index}:{scene.start:.3f}:{scene.caption or ''}:{scene.query or ''}"
    return zlib.crc32(text.encode("utf-8", errors="replace")) & 0xFFFFFFFF


def _has_number(text: str) -> bool:
    value = (text or "").casefold()
    return bool(re.search(r"\d", value)) or any(
        token in value
        for token in (
            "миллион",
            "тысяч",
            "процент",
            "million",
            "thousand",
            "percent",
        )
    )


def resolve_scene_grammar(plan: ShotPlan) -> list[int]:
    """Arbitrate premium decisions so separate AI systems cannot create a mess."""

    changed: set[int] = set()
    last_hero_index = -99
    split_per_bucket: dict[int, int] = {}
    text_per_bucket: dict[int, int] = {}
    parallax_per_bucket: dict[int, int] = {}

    for index, scene in enumerate(plan.scenes):
        layout = str(scene.premium_layout or "clean")
        if layout not in _VALID:
            layout = "clean"

        if scene.semantic_lock and layout not in _LOCK_SAFE:
            layout = "clean"

        if layout == "big_number" and not _has_number(scene.caption or ""):
            layout = "clean"

        bucket = int(max(0.0, float(scene.start)) // 30.0)

        if layout in {"split_screen", "before_after"}:
            used = split_per_bucket.get(bucket, 0)
            if used >= 2:
                layout = "clean"
            else:
                split_per_bucket[bucket] = used + 1

        if layout == "text_behind":
            used = text_per_bucket.get(bucket, 0)
            if used >= 2:
                layout = "parallax" if scene.premium_foreground else "clean"
            else:
                text_per_bucket[bucket] = used + 1

        if layout == "parallax":
            used = parallax_per_bucket.get(bucket, 0)
            if used >= 3:
                layout = "clean"
            else:
                parallax_per_bucket[bucket] = used + 1

        if layout in _HERO and index - last_hero_index < 3:
            layout = "clean"
        elif layout in _HERO:
            last_hero_index = index

        if (
            index > 0
            and getattr(plan.scenes[index - 1], "pace_class", "normal") == "reveal"
            and not scene.semantic_lock
        ):
            if layout in _HERO or layout in {
                "text_behind",
                "parallax",
                "spotlight",
                "blur_background",
            }:
                layout = "clean"

        if scene.premium_layout != layout:
            scene.premium_layout = layout
            changed.add(index)

        pace = getattr(scene, "pace_class", "normal") or "normal"
        seed = _seed(scene, index)

        if layout == "focus_zoom":
            desired_motion = "dramatic_push"
        elif pace == "reveal" and layout not in {"freeze_frame", "split_screen", "before_after"}:
            desired_motion = "dramatic_push"
        elif pace == "fast" and layout == "clean":
            desired_motion = "snap_zoom" if seed % 3 == 0 else "micro_push"
        elif layout in {"spotlight", "blur_background", "text_behind", "parallax"}:
            desired_motion = "slow_push"
        else:
            desired_motion = scene.motion_preset

        if scene.motion_preset != desired_motion:
            scene.motion_preset = desired_motion
            changed.add(index)

    return sorted(changed)
