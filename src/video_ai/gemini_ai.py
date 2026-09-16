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
    """Small dependency-free Gemini REST client used by the v0.9 director/judge.

    GEMINI_API_KEY enables it. GEMINI_MODEL can override the preferred model.
    The default uses a stable Flash model to keep the local app predictable.
    """

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.api_key = (api_key or os.getenv("GEMINI_API_KEY", "")).strip()
        preferred = (model or os.getenv("GEMINI_MODEL", "")).strip()
        candidates = [preferred] if preferred else ["gemini-2.5-flash"]
        self.models = [m for m in candidates if m]
        self.last_model: str | None = None

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def direct(self, transcript: str, scenes: list[Scene]) -> list[dict[str, Any]]:
        if not self.available or not scenes:
            return []

        scene_rows = [
            {
                "index": index,
                "start": round(scene.start, 3),
                "end": round(scene.end, 3),
                "caption": scene.caption or "",
            }
            for index, scene in enumerate(scenes)
        ]
        prompt = f"""
You are the visual director for a fast-paced vertical YouTube Short.
Read the ENTIRE narration first, then plan the visual for every timed scene.

NARRATION:
{transcript}

TIMED SCENES:
{json.dumps(scene_rows, ensure_ascii=False)}

Return ONLY valid JSON with this shape:
{{
  "scenes": [
    {{
      "index": 0,
      "visual_mode": "image|video|meme",
      "visual_description": "concrete English description of exactly what should be visible",
      "search_queries": ["3 to 5 concise English searches"],
      "meme_tags": ["optional", "English tags"]
    }}
  ]
}}

Rules:
- Keep the same scene indexes. Do not merge or invent scenes.
- Historical accuracy and story continuity matter more than generic stock-video motion.
- Preserve person, country, era, clothing, architecture and event when they are known.
- Never replace a specific historical person/event with a random modern person just because keywords overlap.
- Use video only when realistic generic motion/B-roll can represent the meaning accurately.
- Prefer an archival image/painting/engraving over misleading modern footage for historical specifics.
- Use memes sparingly, normally 0-2 in a ~20 second short, only when a reaction genuinely improves the joke/emotion.
- For meme mode, meme_tags must describe the reaction, e.g. shocked disbelief, facepalm, money flex, panic.
- Search queries must be concrete, visual, English, and ordered specific -> broader fallback.
- Do not put narration text, subtitles, logos, watermarks, or explanatory text inside the desired visual.
""".strip()
        data = self._generate_json([{"text": prompt}], temperature=0.15)
        raw = data.get("scenes", []) if isinstance(data, dict) else []
        return [item for item in raw if isinstance(item, dict)]

    def judge_visual(
        self,
        scene: Scene,
        media_path: str | Path,
        *,
        candidate_title: str = "",
        source: str = "",
    ) -> VisualJudgement | None:
        """Judge one downloaded candidate using a normalized JPEG preview.

        Video candidates are represented by a frame near the beginning. The goal
        is not cinematic scoring; it is to reject obvious semantic/era/person
        mismatches before they enter the final timeline.
        """
        if not self.available:
            return None
        preview = _make_preview(Path(media_path))
        if preview is None:
            return None
        try:
            encoded = base64.b64encode(preview.read_bytes()).decode("ascii")
            prompt = f"""
You are a strict visual relevance judge for an automatically edited YouTube Short.

Narration in this scene: {scene.caption or ''}
Director wants to show: {scene.visual_description or scene.query}
Preferred visual type: {scene.visual_mode}
Candidate title/source: {candidate_title} / {source}

Inspect the supplied frame. Return ONLY JSON:
{{
  "accept": true,
  "score": 0,
  "reason": "short explanation",
  "mismatch": "none OR specific mismatch such as wrong era, wrong country, wrong subject, generic unrelated stock"
}}

Scoring:
90-100 = directly depicts the intended scene and context.
75-89 = strongly relevant, small compromises only.
60-74 = usable but generic/indirect.
40-59 = weak or noticeably wrong context.
0-39 = unrelated or misleading.
Be especially strict about historical era, country and subject identity. A modern priest is NOT a valid substitute for a specific 19th-century Chinese historical figure/event.
""".strip()
            data = self._generate_json(
                [
                    {"inline_data": {"mime_type": "image/jpeg", "data": encoded}},
                    {"text": prompt},
                ],
                temperature=0.05,
            )
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
        except Exception:
            return None
        finally:
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
            request = urllib.request.Request(
                url,
                data=body,
                method="POST",
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": self.api_key,
                    "User-Agent": "video-ai/0.9",
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=90) as response:
                    response_data = json.load(response)
                text = _extract_text(response_data)
                parsed = _parse_json_text(text)
                self.last_model = model
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
        raise RuntimeError("Gemini request failed: " + " | ".join(errors))


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
        start_obj = cleaned.find("{")
        end_obj = cleaned.rfind("}")
        if start_obj >= 0 and end_obj > start_obj:
            return json.loads(cleaned[start_obj : end_obj + 1])
        raise


def _make_preview(path: Path) -> Path | None:
    if not path.exists() or shutil.which("ffmpeg") is None:
        return None
    temp_dir = Path(tempfile.mkdtemp(prefix="video-ai-gemini-"))
    target = temp_dir / "preview.jpg"
    suffix = path.suffix.lower()
    video = suffix in {".mp4", ".mov", ".mkv", ".webm", ".avi", ".ogv", ".ogg"}
    command = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    if video:
        command += ["-ss", "0.6"]
    command += [
        "-i", str(path),
        "-frames:v", "1",
        "-vf", "scale='min(768,iw)':-2",
        "-q:v", "4",
        str(target),
    ]
    try:
        subprocess.run(command, check=True, timeout=35)
    except Exception:
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass
        return None
    return target if target.exists() and target.stat().st_size > 512 else None
