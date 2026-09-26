from __future__ import annotations

import json
import math
from pathlib import Path

from .models import Scene, ShotPlan, Word


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
        scenes.append(Scene(
            start=float(raw["start"]),
            end=float(raw["end"]),
            query=str(raw.get("query", "")),
            asset=asset,
            asset_kind=raw.get("asset_kind", "blank"),
            motion=raw.get("motion", "none"),
            caption=raw.get("caption"),
            source_start=float(raw.get("source_start", 0.0)),
            caption_words=[Word(float(w["start"]), float(w["end"]), str(w["text"])) for w in (raw.get("caption_words") or [])],
            visual_description=raw.get("visual_description"),
            search_queries=list(raw.get("search_queries") or []),
            visual_mode=raw.get("visual_mode", "auto"),
            source_mode=raw.get("source_mode", "auto"),
            motion_preset=raw.get("motion_preset", "slow_push"),
            meme_filename=raw.get("meme_filename"),
            semantic_lock=bool(raw.get("semantic_lock", False)),
            required_entities=[str(x) for x in (raw.get("required_entities") or []) if str(x).strip()],
            required_context=[str(x) for x in (raw.get("required_context") or []) if str(x).strip()],
            semantic_fallback=(str(raw["semantic_fallback"]) if raw.get("semantic_fallback") else None),
            tone=raw.get("tone", "neutral"),
            focus_x=(float(raw["focus_x"]) if raw.get("focus_x") is not None else None),
            focus_y=(float(raw["focus_y"]) if raw.get("focus_y") is not None else None),
            focus_source=raw.get("focus_source"),
            asset_score=(float(raw["asset_score"]) if raw.get("asset_score") is not None else None),
            semantic_score=(float(raw["semantic_score"]) if raw.get("semantic_score") is not None else None),
        ))

    plan = ShotPlan(
        audio=audio,
        scenes=scenes,
        width=int(data.get("width", 1080)),
        height=int(data.get("height", 1920)),
        fps=int(data.get("fps", 30)),
        director_source=str(data.get("director_source", "rules")),
        director_model=(str(data["director_model"]) if data.get("director_model") else None),
    )
    validate_shot_plan(plan)
    return plan


def save_shot_plan(plan: ShotPlan, path: str | Path) -> Path:
    validate_shot_plan(plan)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    base = path.parent.resolve()

    def portable(value: str | Path | None) -> str | None:
        if value is None:
            return None
        resolved = Path(value).resolve()
        try:
            return str(resolved.relative_to(base))
        except (ValueError, OSError):
            return str(resolved)

    payload = {
        "audio": portable(plan.audio),
        "width": plan.width,
        "height": plan.height,
        "fps": plan.fps,
        "director_source": plan.director_source,
        "director_model": plan.director_model,
        "scenes": [
            {
                "start": scene.start,
                "end": scene.end,
                "query": scene.query,
                "asset": portable(scene.asset),
                "asset_kind": scene.asset_kind,
                "motion": scene.motion,
                "caption": scene.caption,
                "source_start": scene.source_start,
                "caption_words": [
                    {"start": w.start, "end": w.end, "text": w.text}
                    for w in scene.caption_words
                ],
                "visual_description": scene.visual_description,
                "search_queries": scene.search_queries,
                "visual_mode": scene.visual_mode,
                "source_mode": scene.source_mode,
                "motion_preset": scene.motion_preset,
                "meme_filename": scene.meme_filename,
                "semantic_lock": scene.semantic_lock,
                "required_entities": scene.required_entities,
                "required_context": scene.required_context,
                "semantic_fallback": scene.semantic_fallback,
                "tone": scene.tone,
                "focus_x": scene.focus_x,
                "focus_y": scene.focus_y,
                "focus_source": scene.focus_source,
                "asset_score": scene.asset_score,
                "semantic_score": scene.semantic_score,
            }
            for scene in plan.scenes
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def validate_shot_plan(plan: ShotPlan) -> None:
    if not plan.scenes:
        raise ValueError("Shot plan has no scenes")
    if plan.width <= 0 or plan.height <= 0 or plan.fps <= 0:
        raise ValueError("width, height and fps must be positive")

    valid_tones = {"neutral","informational","positive","negative","tragic","tense","shocking","absurd","funny","victorious","mysterious","religious","violent","emotional"}
    previous_end = 0.0
    for index, scene in enumerate(plan.scenes):
        if not math.isfinite(scene.source_start) or scene.source_start < 0:
            raise ValueError(f"scene {index}: invalid source_start")
        if scene.start < 0 or scene.end <= scene.start:
            raise ValueError(f"scene {index}: invalid time range")
        if index and scene.start < previous_end - 1e-6:
            raise ValueError(f"scene {index}: overlaps previous scene")
        previous_end = scene.end
        if scene.asset_kind != "blank" and not scene.asset:
            raise ValueError(f"scene {index}: asset_kind={scene.asset_kind!r} requires asset")
        if scene.visual_mode not in {"auto", "image", "video", "meme"}:
            raise ValueError(f"scene {index}: invalid visual_mode={scene.visual_mode!r}")
        if scene.source_mode not in {"auto", "historical_archive", "stock_video", "meme_library", "generic_image"}:
            raise ValueError(f"scene {index}: invalid source_mode={scene.source_mode!r}")
        if scene.motion_preset not in {"none", "micro_push", "slow_push", "dramatic_push", "pull_back", "reveal_left", "reveal_right"}:
            raise ValueError(f"scene {index}: invalid motion_preset={scene.motion_preset!r}")
        if scene.tone not in valid_tones:
            raise ValueError(f"scene {index}: invalid tone={scene.tone!r}")
        if scene.semantic_lock and not (scene.required_entities or scene.required_context or scene.semantic_fallback):
            raise ValueError(f"scene {index}: semantic_lock requires entities, context or fallback")
        for name, value in (("focus_x", scene.focus_x), ("focus_y", scene.focus_y)):
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"scene {index}: {name} must be between 0 and 1")
