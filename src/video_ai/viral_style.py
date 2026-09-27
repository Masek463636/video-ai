"""Isolated Viral / Darwin camera grammar."""

from __future__ import annotations

from .models import ShotPlan


def apply_viral_motion(plan: ShotPlan) -> list[int]:
    """Fast beats with deliberate visual rests."""

    changed: list[int] = []

    for index, scene in enumerate(plan.scenes):

        # Historical / factual exact scenes stay readable.
        if scene.semantic_lock:
            wanted = "dramatic_push"

        # Memes already contain motion/visual information.
        elif (
            scene.visual_mode == "meme"
            or scene.source_mode == "meme_library"
        ):
            wanted = "none"

        # Every fourth scene gives the viewer a tiny rest.
        elif index % 4 == 3:
            wanted = "micro_push"

        else:
            wanted = "snap_zoom"

        if scene.motion_preset != wanted:
            scene.motion_preset = wanted
            changed.append(index)

    return changed
