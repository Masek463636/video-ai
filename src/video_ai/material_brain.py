from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .models import Scene, ShotPlan


_IMAGE_SIMILARITY = 0.94
_VIDEO_SIMILARITY = 0.975


@dataclass(slots=True)
class DuplicateMatch:
    scene: int
    original_scene: int
    similarity: float
    reason: str


def find_duplicate_scenes(
    plan: ShotPlan,
    *,
    image_similarity: float = _IMAGE_SIMILARITY,
    video_similarity: float = _VIDEO_SIMILARITY,
) -> tuple[list[int], list[DuplicateMatch]]:
    """Find repeated/near-repeated visuals after real assets have been chosen.

    This deliberately operates on materialized files rather than provider URLs,
    so it catches the same image returned by multiple APIs, resized mirrors and
    fallback copies from another scene.
    """
    repairs: list[int] = []
    matches: list[DuplicateMatch] = []
    seen_paths: dict[str, int] = {}
    seen_hashes: list[tuple[int, str, int]] = []

    for index, scene in enumerate(plan.scenes):
        if not scene.asset or scene.asset_kind == "blank":
            continue
        path = Path(scene.asset)
        key = _path_key(path)

        if key in seen_paths:
            repairs.append(index)
            matches.append(DuplicateMatch(index, seen_paths[key], 1.0, "same_asset"))
            continue
        seen_paths[key] = index

        if not path.exists():
            continue
        fingerprint = media_ahash(path)
        if fingerprint is None:
            continue

        threshold = video_similarity if scene.asset_kind == "video" else image_similarity
        best: tuple[int, float] | None = None
        for previous_index, previous_kind, previous_hash in seen_hashes:
            # Comparing stills to videos creates too many false positives on
            # generic dark frames, so only compare within the same media kind.
            if previous_kind != scene.asset_kind:
                continue
            similarity = hash_similarity(fingerprint, previous_hash)
            if similarity >= threshold and (best is None or similarity > best[1]):
                best = (previous_index, similarity)

        if best is not None:
            repairs.append(index)
            matches.append(DuplicateMatch(index, best[0], best[1], "near_duplicate"))
        else:
            seen_hashes.append((index, scene.asset_kind, fingerprint))

    return sorted(set(repairs)), matches


def prepare_diversity_repair(plan: ShotPlan, indexes: list[int], *, pass_index: int) -> None:
    """Bias repeated scenes toward a genuinely different visual concept.

    The original meaning stays first. We only add alternate framing/context
    queries and, on later passes, allow unlocked generic stills to become stock
    video when that is semantically safe.
    """
    for index in indexes:
        if index < 0 or index >= len(plan.scenes):
            continue
        scene = plan.scenes[index]
        base = _base_query(scene)
        alternates = _variation_queries(scene, base, pass_index)
        scene.search_queries = _merge_queries(alternates, scene.search_queries or [scene.query])[:7]
        if scene.search_queries:
            scene.query = scene.search_queries[0]

        if not scene.semantic_lock:
            descriptor = (scene.visual_description or scene.caption or base).rstrip(" .")
            scene.visual_description = (
                f"{descriptor}. Show a visually distinct angle, subject or context from adjacent shots; "
                "do not reuse the same composition."
            )
            if pass_index >= 1 and scene.source_mode == "generic_image":
                scene.visual_mode = "video"
                scene.source_mode = "stock_video"


def diversity_summary(plan: ShotPlan) -> dict:
    duplicates, matches = find_duplicate_scenes(plan)
    materialized = sum(1 for s in plan.scenes if s.asset and s.asset_kind != "blank")
    unique = max(0, materialized - len(duplicates))
    return {
        "materialized_scenes": materialized,
        "unique_visuals": unique,
        "duplicate_scenes": duplicates,
        "duplicate_matches": [
            {
                "scene": item.scene,
                "original_scene": item.original_scene,
                "similarity": round(item.similarity, 4),
                "reason": item.reason,
            }
            for item in matches
        ],
    }


def media_ahash(path: str | Path) -> int | None:
    """Return a dependency-free perceptual aHash using one decoded frame.

    FFmpeg normalizes every image/video frame to a 16x16 grayscale byte grid.
    That makes the hash resilient to resize/re-encode differences while keeping
    this feature available even when Pillow/OpenCV extras are not installed.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return None
    try:
        completed = subprocess.run(
            [
                ffmpeg,
                "-v", "error",
                "-xerror",
                "-err_detect", "explode",
                "-i", str(path),
                "-frames:v", "1",
                "-vf", "scale=16:16:flags=area,format=gray",
                "-f", "rawvideo",
                "-pix_fmt", "gray",
                "pipe:1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=15,
            check=False,
        )
    except Exception:
        return None
    pixels = completed.stdout
    if completed.returncode != 0 or completed.stderr.strip() or len(pixels) < 256:
        return None
    pixels = pixels[:256]
    mean = sum(pixels) / len(pixels)
    value = 0
    for pixel in pixels:
        value = (value << 1) | int(pixel >= mean)
    return value


def hash_similarity(a: int, b: int, *, bits: int = 256) -> float:
    distance = (a ^ b).bit_count()
    return max(0.0, min(1.0, 1.0 - distance / float(bits)))


def _path_key(path: Path) -> str:
    try:
        return str(path.resolve()).casefold()
    except OSError:
        return str(path).casefold()


def _base_query(scene: Scene) -> str:
    return re.sub(r"\s+", " ", (scene.query or scene.caption or scene.visual_description or "documentary scene")).strip()


def _variation_queries(scene: Scene, base: str, pass_index: int) -> list[str]:
    caption = (scene.caption or "").lower()
    historical = scene.source_mode == "historical_archive" or scene.semantic_lock

    if scene.semantic_lock:
        if pass_index == 0:
            return [f"{base} historical engraving", f"{base} archival illustration", f"{base} full scene"]
        if pass_index == 1:
            return [f"{base} different engraving", f"{base} full body historical", f"{base} period illustration"]
        return [f"{base} alternate historical depiction", f"{base} archival print"]

    if any(token in caption for token in ("арм", "солдат", "войск", "восстан", "army", "soldier", "troops", "rebellion")):
        if historical:
            return [f"{base} wide army scene", f"{base} soldiers marching engraving", f"{base} battle crowd detail"]
        return [f"{base} soldiers wide shot", f"{base} marching crowd", f"{base} military detail"]

    if any(token in caption for token in ("сон", "проснул", "видение", "dream", "wake", "vision")):
        return [f"{base} human reaction", f"{base} different cinematic angle", f"{base} environment detail"]

    if any(token in caption for token in ("провал", "стресс", "паник", "груст", "failed", "stress", "panic", "sad")):
        return [f"{base} close reaction", f"{base} hands face detail", f"{base} different person reaction"]

    if pass_index == 0:
        return [f"{base} wide shot", f"{base} different angle", f"{base} environmental context"]
    if pass_index == 1:
        return [f"{base} close detail", f"{base} action b roll", f"{base} alternate subject"]
    return [f"{base} visual metaphor", f"{base} context shot", f"{base} different composition"]


def _merge_queries(*groups: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for raw in group:
            query = re.sub(r"\s+", " ", str(raw or "")).strip()
            key = query.casefold()
            if query and key not in seen:
                seen.add(key)
                result.append(query)
    return result
