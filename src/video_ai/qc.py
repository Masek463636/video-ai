from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from .models import ShotPlan


@dataclass(slots=True)
class SceneQC:
    scene: int
    ok: bool
    score: float
    reasons: list[str]


def inspect_plan(plan: ShotPlan) -> list[SceneQC]:
    """QC pass for obvious visual failures and weak semantic matches."""
    results: list[SceneQC] = []
    previous_asset: str | None = None
    for index, scene in enumerate(plan.scenes):
        score = 1.0
        reasons: list[str] = []
        if not scene.asset or scene.asset_kind == "blank":
            score -= 0.75
            reasons.append("missing visual asset")
        else:
            path = Path(scene.asset)
            if not path.exists():
                score -= 0.75
                reasons.append("asset path does not exist")
            if previous_asset and str(path) == previous_asset:
                score -= 0.35
                reasons.append("same asset as previous scene")
            previous_asset = str(path)

        if scene.duration < 0.8:
            score -= 0.15
            reasons.append("scene too short")
        if scene.duration > 4.0:
            score -= 0.15
            reasons.append("scene too long")
        if scene.asset_kind == "image" and scene.focus_source is None:
            score -= 0.1
            reasons.append("no visual focus metadata")
        if not scene.caption:
            score -= 0.08
            reasons.append("no caption text")

        if scene.semantic_score is not None:
            if scene.semantic_score < 0.16:
                score -= 0.35
                reasons.append("very weak text-image semantic match")
            elif scene.semantic_score < 0.20:
                score -= 0.18
                reasons.append("weak text-image semantic match")
            elif scene.semantic_score >= 0.28:
                score += 0.05

        if scene.asset_score is not None and scene.asset_score < 1.0:
            score -= 0.12
            reasons.append("weak retrieval score")

        score = max(0.0, min(1.0, score))
        results.append(SceneQC(scene=index, ok=score >= 0.55, score=round(score, 3), reasons=reasons))
    return results


def failed_scene_indexes(results: list[SceneQC]) -> set[int]:
    return {item.scene for item in results if not item.ok}


def save_qc(results: list[SceneQC], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ok": all(item.ok for item in results),
        "average_score": round(sum(item.score for item in results) / max(1, len(results)), 3),
        "scenes": [asdict(item) for item in results],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
