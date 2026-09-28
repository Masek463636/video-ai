from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .models import Scene, ShotPlan


_IMAGE_SIMILARITY = 0.90
_VIDEO_SIMILARITY = 0.92


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

    Detection deliberately ignores provider URLs. First we compare decoded file
    bytes, then perceptual fingerprints. This catches the same Pexels/Commons
    asset downloaded again under another scene filename and visually equivalent
    mirrors/crops from different providers.
    """
    repairs: list[int] = []
    matches: list[DuplicateMatch] = []
    seen_paths: dict[str, int] = {}
    seen_digests: dict[str, int] = {}
    seen_hashes: list[tuple[int, str, int]] = []

    for index, scene in enumerate(plan.scenes):
        if not scene.asset or scene.asset_kind == "blank":
            continue
        path = Path(scene.asset)
        key = _path_key(path)

        if key in seen_paths:
            repairs.append(index)
            matches.append(DuplicateMatch(index, seen_paths[key], 1.0, "same_path"))
            continue
        seen_paths[key] = index

        if not path.exists():
            continue

        digest = media_digest(path)
        if digest:
            if digest in seen_digests:
                repairs.append(index)
                matches.append(DuplicateMatch(index, seen_digests[digest], 1.0, "same_bytes"))
                continue
            seen_digests[digest] = index

        fingerprint = media_ahash(path)
        if fingerprint is None:
            continue

        threshold = video_similarity if scene.asset_kind == "video" else image_similarity
        best: tuple[int, float] | None = None
        for previous_index, previous_kind, previous_hash in seen_hashes:
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
    """Bias repeated scenes toward a genuinely different visual concept."""
    for index in indexes:
        if index < 0 or index >= len(plan.scenes):
            continue
        scene = plan.scenes[index]
        base = _base_query(scene)
        alternates = _variation_queries(scene, base, pass_index)
        scene.search_queries = _merge_queries(alternates, scene.search_queries or [scene.query])[:8]
        if scene.search_queries:
            scene.query = scene.search_queries[0]

        if not scene.semantic_lock:
            descriptor = (scene.visual_description or scene.caption or base).rstrip(" .")
            scene.visual_description = (
                f"{descriptor}. Use a new subject and a clearly different composition from every previous shot. "
                "Do not reuse the same person, location, map, monument, bed, portrait or camera setup unless the narration explicitly requires continuity."
            )

            # A repeated generic still is better repaired by changing medium,
            # not by asking an image search for the same noun again.
            if pass_index >= 1 and scene.source_mode in {"generic_image", "auto"}:
                scene.visual_mode = "video"
                scene.source_mode = "stock_video"
            elif pass_index >= 2 and scene.visual_mode == "video" and scene.source_mode == "stock_video":
                scene.visual_mode = "image"
                scene.source_mode = "generic_image"


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


def media_digest(path: str | Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def media_ahash(path: str | Path) -> int | None:
    """Dependency-free perceptual hash from a representative decoded frame."""
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
        return [f"{base} alternate historical depiction", f"{base} archival print", f"{base} different composition"]

    if any(token in caption for token in ("арм", "солдат", "войск", "восстан", "army", "soldier", "troops", "rebellion")):
        if historical:
            return [f"{base} wide army scene", f"{base} soldiers marching engraving", f"{base} battle crowd detail", f"{base} weapons uniforms detail"]
        return [f"{base} soldiers wide shot", f"{base} marching crowd", f"{base} military detail", f"{base} battlefield environment"]

    if any(token in caption for token in ("сон", "проснул", "видение", "dream", "wake", "vision")):
        if pass_index == 0:
            return [f"{base} waking reaction close up", f"{base} bedroom environment", f"{base} dream atmosphere"]
        return [f"{base} eyes opening close up", f"{base} hands face reaction", f"{base} surreal vision detail", f"{base} room detail no sleeping person"]

    if any(token in caption for token in ("провал", "стресс", "паник", "груст", "failed", "stress", "panic", "sad")):
        return [f"{base} close reaction", f"{base} hands face detail", f"{base} different person reaction", f"{base} symbolic failure visual"]

    if any(token in caption for token in ("карта", "china", "китай", "территор", "map")):
        return [f"{base} archival illustration no map", f"{base} historical crowd", f"{base} location detail", f"{base} historical document"]

    if any(token in caption for token in ("жертв", "миллион", "погиб", "смерт", "casualt", "million", "death")):
        return [f"{base} mourning people", f"{base} historical devastation", f"{base} destroyed settlement", f"{base} somber memorial detail"]

    if pass_index == 0:
        return [f"{base} wide shot", f"{base} different angle", f"{base} environmental context", f"{base} detail shot"]
    if pass_index == 1:
        return [f"{base} close detail", f"{base} action b roll", f"{base} alternate subject", f"{base} contextual cutaway"]
    return [f"{base} visual metaphor", f"{base} context shot", f"{base} different composition", f"{base} different subject"]


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
