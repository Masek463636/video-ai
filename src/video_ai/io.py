from __future__ import annotations

import json
from pathlib import Path

from .models import Scene, ShotPlan


def load_shot_plan(path: str | Path) -> ShotPlan:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    base = path.parent

    audio = Path(data["audio"])
    if not audio.is_absolute():
        audio = (base / audio).resolve()

    scenes = []
    for raw in data["scenes"]:
        asset = raw.get("asset")
        if asset:
            asset_path = Path(asset)
            if not asset_path.is_absolute():
                asset_path = (base / asset_path).resolve()
            asset = str(asset_path)
        scenes.append(
            Scene(
                start=float(raw["start"]),
                end=float(raw["end"]),
                query=str(raw.get("query", "")),
                asset=asset,
                asset_kind=raw.get("asset_kind", "blank"),
                motion=raw.get("motion", "none"),
                caption=raw.get("caption"),
            )
        )

    plan = ShotPlan(
        audio=audio,
        scenes=scenes,
        width=int(data.get("width", 1080)),
        height=int(data.get("height", 1920)),
        fps=int(data.get("fps", 30)),
    )
    validate_shot_plan(plan)
    return plan


def validate_shot_plan(plan: ShotPlan) -> None:
    if not plan.scenes:
        raise ValueError("Shot plan has no scenes")
    if plan.width <= 0 or plan.height <= 0 or plan.fps <= 0:
        raise ValueError("width, height and fps must be positive")

    previous_end = 0.0
    for index, scene in enumerate(plan.scenes):
        if scene.start < 0 or scene.end <= scene.start:
            raise ValueError(f"scene {index}: invalid time range")
        if index and scene.start < previous_end - 1e-6:
            raise ValueError(f"scene {index}: overlaps previous scene")
        previous_end = scene.end
        if scene.asset_kind != "blank" and not scene.asset:
            raise ValueError(f"scene {index}: asset_kind={scene.asset_kind!r} requires asset")
