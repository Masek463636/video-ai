from __future__ import annotations

import base64
import json
import os
import re
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import Scene

_API_ROOT = "https://generativelanguage.googleapis.com/v1beta"
_JSON_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", flags=re.IGNORECASE)


@dataclass(slots=True)
class VisualJudgement:
    accept: bool
    score: int
    reason: str = ""
    mismatch: str = ""


class GeminiClient:
    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.api_key = (api_key or os.getenv("GEMINI_API_KEY", "")).strip()
        preferred = (model or os.getenv("GEMINI_MODEL", "")).strip()
        candidates = [preferred] if preferred else ["gemini-3.5-flash", "gemini-3.1-flash-lite"]
        self.models = [m for m in candidates if m]
        self.last_model: str | None = None
        self.last_error: str | None = None

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def direct(self, transcript: str, scenes: list[Scene], *, meme_names: list[str] | None = None) -> list[dict[str, Any]]:
        if not self.available or not scenes:
            return []
        scene_rows = [
            {"index": i, "start": round(s.start, 3), "end": round(s.end, 3), "caption": s.caption or ""}
            for i, s in enumerate(scenes)
        ]
        meme_names = meme_names or []
        prompt = f"""
You are the Editing Brain for a fast-paced vertical YouTube Short.
Read the ENTIRE narration first, then plan every visual beat like a human editor.

NARRATION:
{transcript}

TIMED SCENES:
{json.dumps(scene_rows, ensure_ascii=False)}

AVAILABLE LOCAL MEMES (exact filenames; use only if genuinely useful):
{json.dumps(meme_names, ensure_ascii=False)}

Return ONLY valid JSON:
{{
  "scenes": [
    {{
      "index": 0,
      "visual_mode": "image|video|meme",
      "source_mode": "historical_archive|stock_video|meme_library|generic_image",
      "motion_preset": "none|micro_push|slow_push|dramatic_push|pull_back|reveal_left|reveal_right",
      "visual_description": "exact English description of what should be visible",
      "search_queries": ["3 to 5 concise English searches"],
      "meme_tags": ["optional English reaction tags"],
      "meme_filename": "exact available filename or empty string"
    }}
  ]
}}

Editorial rules:
- Keep scene indexes unchanged.
- Historical accuracy beats generic motion. For a named historical person/event/era, choose historical_archive + image unless authentic footage is plausibly available.
- Do NOT use stock_video for specific historical people, named wars, dynasties or events if that would create a misleading modern substitute.
- Use stock_video only for generic concepts/actions that can honestly be represented: crowd, fire, road, phone, money, city, typing, walking, etc.
- Use memes sparingly: usually 0-2 per ~20 seconds. If meme is chosen and one of the available files clearly fits, return its exact filename.
- Prefer a strong still image with intentional camera motion over a semantically wrong video.
- Motion should support meaning: slow_push for calm emphasis, dramatic_push for shocks/revelations, pull_back for consequences/endings, reveal_* for spatial discovery, micro_push for video/very subtle movement, none for memes.
- Search queries must preserve country, era, person and event when known.
- Never ask for text overlays/logos/subtitles inside the visual.
""".strip()
        data = self._generate_json([{"text": prompt}], temperature=0.12)
        raw = data.get("scenes", []) if isinstance(data, dict) else []
        return [item for item in raw if isinstance(item, dict)]

    def judge_visual(self, scene: Scene, media_path: str | Path, *, candidate_title: str = "", source: str = "") -> VisualJudgement | None:
        if not self.available:
            return None
        previews = _make_previews(Path(media_path), count=3)
        if not previews:
            return None
        try:
            parts: list[dict[str, Any]] = []
            for preview in previews:
                encoded = base64.b64encode(preview.read_bytes()).decode("ascii")
                parts.append({"inline_data": {"mime_type": "image/jpeg", "data": encoded}})
            prompt = f"""
You are a strict visual relevance judge for an automatically edited YouTube Short.
The supplied images are representative frames from ONE candidate asset.

Narration in this scene: {scene.caption or ''}
Director wants to show: {scene.visual_description or scene.query}
Preferred visual type: {scene.visual_mode}
Required source type: {scene.source_mode}
Candidate title/source: {candidate_title} / {source}

Return ONLY JSON:
{{
  "accept": true,
  "score": 0,
  "reason": "short explanation",
  "mismatch": "none OR exact mismatch"
}}

Scoring:
90-100 directly depicts intended meaning and context.
75-89 strongly relevant with small compromises.
60-74 usable but generic/indirect.
40-59 weak or wrong context.
0-39 unrelated/misleading.

Be VERY strict about era, country, identity, event and subject. If source_mode is historical_archive, reject obviously modern stock. If the narration names a specific historical person/event, generic modern substitutes should normally score below 60.
""".strip()
            parts.append({"text": prompt})
            data = self._generate_json(parts, temperature=0.03)
            if not isinstance(data, dict):
                return None
            score = max(0, min(100, int(float(data.get("score", 0)))))
            accept = bool(data.get("accept", False)) and score >= 60
            return VisualJudgement(
                accept=accept,
                score=score,
                reason=str(data.get("reason", ""))[:300],
                mismatch=str(data.get("mismatch", ""))[:200],
            )
        except Exception as exc:
            self.last_error = str(exc)
            return None
        finally:
            for preview in previews:
                try:
                    preview.unlink(missing_ok=True)
                    preview.parent.rmdir()
                except OSError:
                    pass

    def _generate_json(self, parts: list[dict[str, Any]], *, temperature: float) -> Any:
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured")
        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": temperature,
                "maxOutputTokens": 8192,
            },
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        errors: list[str] = []
        for model in self.models:
            url = f"{_API_ROOT}/models/{urllib.parse.quote(model, safe='')}:generateContent"
            req = urllib.request.Request(url, data=body, method="POST", headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key,
                "User-Agent": "video-ai/1.0",
            })
            try:
                with urllib.request.urlopen(req, timeout=90) as response:
                    response_data = json.load(response)
                parsed = _parse_json_text(_extract_text(response_data))
                self.last_model = model
                self.last_error = None
                return parsed
            except urllib.error.HTTPError as exc:
                detail = ""
                try:
                    detail = exc.read().decode("utf-8", errors="ignore")[:500]
                except Exception:
                    pass
                errors.append(f"{model}: HTTP {exc.code} {detail}")
            except Exception as exc:
                errors.append(f"{model}: {exc}")
        self.last_error = " | ".join(errors)
        raise RuntimeError("Gemini request failed: " + self.last_error)


def get_gemini_client() -> GeminiClient | None:
    client = GeminiClient()
    return client if client.available else None


def _extract_text(payload: dict[str, Any]) -> str:
    candidates = payload.get("candidates") or []
    if not candidates:
        raise RuntimeError("Gemini returned no candidates")
    parts = candidates[0].get("content", {}).get("parts", [])
    text = "\n".join(str(part.get("text", "")) for part in parts if part.get("text"))
    if not text.strip():
        raise RuntimeError("Gemini returned no text")
    return text.strip()


def _parse_json_text(text: str) -> Any:
    cleaned = _JSON_FENCE_RE.sub("", text.strip()).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            return json.loads(cleaned[start:end + 1])
        raise


def _make_previews(path: Path, *, count: int = 3) -> list[Path]:
    if not path.exists() or shutil.which("ffmpeg") is None:
        return []
    suffix = path.suffix.lower()
    is_video = suffix in {".mp4", ".mov", ".mkv", ".webm", ".avi", ".ogv", ".ogg"}
    temp_dir = Path(tempfile.mkdtemp(prefix="video-ai-gemini-"))
    if not is_video:
        target = temp_dir / "preview_0.jpg"
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(path), "-frames:v", "1", "-vf", "scale='min(768,iw)':-2", "-q:v", "4", str(target)]
        try:
            subprocess.run(cmd, check=True, timeout=35)
            return [target] if target.exists() and target.stat().st_size > 512 else []
        except Exception:
            shutil.rmtree(temp_dir, ignore_errors=True)
            return []

    duration = _probe_duration(path)
    times = [0.15, 0.5, 0.82][:max(1, count)]
    previews: list[Path] = []
    for i, fraction in enumerate(times):
        target = temp_dir / f"preview_{i}.jpg"
        seek = max(0.0, duration * fraction if duration > 0 else 0.5 + i * 0.7)
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{seek:.3f}", "-i", str(path), "-frames:v", "1", "-vf", "scale='min(768,iw)':-2", "-q:v", "4", str(target)]
        try:
            subprocess.run(cmd, check=True, timeout=35)
            if target.exists() and target.stat().st_size > 512:
                previews.append(target)
        except Exception:
            continue
    if not previews:
        shutil.rmtree(temp_dir, ignore_errors=True)
    return previews


def _probe_duration(path: Path) -> float:
    try:
        completed = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)], check=True, capture_output=True, text=True, timeout=20)
        return float(json.loads(completed.stdout).get("format", {}).get("duration") or 0.0)
    except Exception:
        return 0.0
