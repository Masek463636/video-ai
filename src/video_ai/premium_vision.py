from __future__ import annotations

import base64
import subprocess
from pathlib import Path

from .cache import DiskCache, cache_key, file_signature
from .composition_review import boxes
from .vision import detect_focus


def _preview(scene, root: Path) -> Path | None:
    if not scene.asset:
        return None
    asset = Path(scene.asset)
    if not asset.is_file():
        return None

    root.mkdir(parents=True, exist_ok=True)
    key = cache_key(file_signature(asset), scene.source_start, "preview")
    target = root / f"{key}.jpg"
    if target.is_file() and target.stat().st_size > 1024:
        return target

    try:
        if scene.asset_kind == "video":
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-v",
                    "error",
                    "-ss",
                    f"{max(0.0, float(scene.source_start) + 0.28):.3f}",
                    "-i",
                    str(asset),
                    "-frames:v",
                    "1",
                    "-vf",
                    "scale=768:-2:force_original_aspect_ratio=decrease",
                    str(target),
                ],
                check=True,
                capture_output=True,
                timeout=25,
            )
        else:
            from PIL import Image

            image = Image.open(asset).convert("RGB")
            image.thumbnail((768, 1366))
            image.save(target, "JPEG", quality=88)

        return target if target.is_file() and target.stat().st_size > 1024 else None
    except Exception:
        target.unlink(missing_ok=True)
        return None


def detect_subject_gemini(
    scene,
    root: str | Path,
    cache: DiskCache,
    client,
) -> tuple[float, float, str, list[float] | None]:
    """Return subject-aware focus. Gemini is best-effort; OpenCV stays fallback."""

    root = Path(root)
    preview = _preview(scene, root / "frames")
    if preview is None:
        return 0.5, 0.5, "center", None

    asset_sig = file_signature(scene.asset or preview)
    key = cache_key(
        "subject",
        asset_sig,
        round(float(scene.source_start), 2),
        scene.caption or "",
    )
    cached = cache.get_json(key)
    if isinstance(cached, dict):
        try:
            confidence = float(cached.get("confidence", 0.0))
            box = boxes([cached.get("box")])[0]
            if confidence >= 0.55:
                x, y, w, h = box
                return x + w / 2, y + h / 2, "gemini_subject", box
        except Exception:
            pass

    if client is not None:
        try:
            encoded = base64.b64encode(preview.read_bytes()).decode("ascii")
            prompt = (
                "Inspect this actual frame for a vertical Short. "
                f"Narration beat: {scene.caption or ''}. "
                "Find ONE main visible subject/object that the viewer should look at first "
                "and that honestly supports the narration. Do not infer hidden things. "
                "Return only JSON "
                '{"box":[x,y,width,height],"confidence":0.0,"reason":"visible evidence"}. '
                "Coordinates normalized 0..1. If no clear subject, confidence=0."
            )
            data = client._generate_json(
                [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": encoded,
                        }
                    },
                ],
                temperature=0.01,
            )
            if isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict):
                data = data[0]
            if isinstance(data, dict):
                confidence = float(data.get("confidence", 0.0))
                box = boxes([data.get("box")])[0]
                cache.set_json(
                    key,
                    {
                        "box": box,
                        "confidence": confidence,
                        "reason": str(data.get("reason") or "")[:300],
                    },
                )
                if confidence >= 0.55:
                    x, y, w, h = box
                    return x + w / 2, y + h / 2, "gemini_subject", box
        except Exception as exc:
            print(
                f"[premium-v2] subject vision fallback: {type(exc).__name__}",
                flush=True,
            )

    fx, fy, source = detect_focus(preview)
    return fx, fy, f"fallback_{source}", None
