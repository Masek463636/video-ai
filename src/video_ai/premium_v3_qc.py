from __future__ import annotations

import json
from pathlib import Path

from .models import ShotPlan
from .visual_diversity import assign_fingerprints


def inspect_premium_v3(plan: ShotPlan) -> dict:
    fingerprints = assign_fingerprints(plan)
    issues: list[dict] = []
    hero_count = 0

    for index, scene in enumerate(plan.scenes):
        if scene.asset_kind == "blank" or not scene.asset:
            issues.append({"scene": index, "type": "missing_asset"})
        if scene.duration < 0.45:
            issues.append({"scene": index, "type": "too_short", "duration": round(scene.duration, 3)})
        if scene.duration > 4.8:
            issues.append({"scene": index, "type": "too_long", "duration": round(scene.duration, 3)})
        if scene.premium_layout != "clean":
            hero_count += 1
        if index >= 2 and fingerprints[index]:
            if fingerprints[index] == fingerprints[index - 1] == fingerprints[index - 2]:
                issues.append({"scene": index, "type": "repetitive_sequence"})

    density = hero_count / max(1, len(plan.scenes))
    if density > 0.45:
        issues.append({"scene": None, "type": "effect_density", "value": round(density, 3)})

    return {
        "style": "premium-v3",
        "scenes": len(plan.scenes),
        "hero_effects": hero_count,
        "hero_density": round(density, 3),
        "issues": issues,
        "passed": not any(item["type"] in {"missing_asset", "effect_density"} for item in issues),
    }


def save_premium_v3_qc(plan: ShotPlan, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    report = inspect_premium_v3(plan)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[premium-v3] QC: {len(report['issues'])} issue(s), "
        f"hero density={report['hero_density']:.2f}",
        flush=True,
    )
    return target
