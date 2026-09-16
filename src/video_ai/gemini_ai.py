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
    tone_match: int = 0
    quality_score: int = 0
    quality_issues: list[str] | None = None


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
            {"index": i, "start": round(s.start, 3), "end": round(s.end, 3), "caption": s.caption or "", "tone": s.tone}
            for i, s in enumerate(scenes)
        ]
        meme_names = meme_names or []
        prompt = f"""
You are the Editing Brain for a fast-paced vertical YouTube Short.
Read the entire narration for story context, but EDIT EACH TIMED SCENE according to the words actually spoken in that scene.

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
      "tone": "neutral|informational|positive|negative|tragic|tense|shocking|absurd|funny|victorious|mysterious|religious|violent|emotional",
      "visual_description": "exact English description of what should be visible in THIS beat",
      "search_queries": ["3 to 5 concise English searches"],
      "meme_tags": ["optional English reaction tags"],
      "meme_filename": "exact available filename or empty string",
      "semantic_lock": false,
      "required_entities": ["ONLY entities literally named in this scene that must match"],
      "required_context": ["era/country/culture that is mandatory only when semantic_lock=true"],
      "semantic_fallback": "safe exact fallback visual, only for locked scenes"
    }}
  ]
}}

BALANCED EDITING RULES:
- Keep scene indexes unchanged.
- Tone describes how the visual should FEEL, not just the nouns it contains.
- Mass death, casualties, destruction and tragedy must be tragic/negative/violent, never cheerful, vacation-like, luxurious or relaxing.
- semantic_lock is a HARD FACT LOCK, not general story context.
- Set semantic_lock=true ONLY when the CURRENT scene caption explicitly names a specific person, named event/war/rebellion, named historical institution, dynasty, or similarly concrete factual entity that the visual must depict accurately.
- DO NOT hard-lock a person/event merely because it appeared in the previous/next scene or elsewhere in the narration.
- Pronouns do NOT automatically justify repeating the person's portrait. Use the action, emotion or consequence being spoken instead.
- Never use the same visual idea on consecutive scenes if another honest visual exists.
- A named person's portrait should usually appear once when introduced, not on every later reference.
- A map should normally appear at most once in a ~20 second Short unless geography actually changes.
- Historical accuracy still matters for contextual visuals.
- Use historical_archive for exact historical facts and genuinely historical action beats.
- Use stock_video only for generic actions/concepts that can honestly be represented.
- Use generic_image for illustrations, dreams, emotions and conceptual beats when a still is stronger.
- Use memes sparingly: usually 0-2 per ~20 seconds. Never use a meme for a hard-locked factual beat.
- Search queries for unlocked scenes should describe the current ACTION/EMOTION first.
- Never ask for text overlays/logos/subtitles inside the visual.
""".strip()
        data = self._generate_json([{"text": prompt}], temperature=0.11)
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
You are a strict visual relevance + tone + quality judge for an automatically edited YouTube Short.
The supplied images are representative frames from ONE candidate asset.

Narration in this scene: {scene.caption or ''}
Scene tone: {scene.tone}
Director wants to show: {scene.visual_description or scene.query}
Preferred visual type: {scene.visual_mode}
Required source type: {scene.source_mode}
Semantic lock: {scene.semantic_lock}
Required entities: {json.dumps(scene.required_entities, ensure_ascii=False)}
Required context: {json.dumps(scene.required_context, ensure_ascii=False)}
Safe fallback: {scene.semantic_fallback or ''}
Candidate title/source: {candidate_title} / {source}

Return ONLY JSON:
{{
  "accept": true,
  "score": 0,
  "tone_match": 0,
  "quality_score": 0,
  "quality_issues": ["short issue labels"],
  "reason": "short explanation",
  "mismatch": "none OR exact mismatch"
}}

Overall score measures semantic relevance.
Tone_match 0-100 measures emotional compatibility with the narration.
Quality_score 0-100 measures whether this looks like usable Shorts footage/image.

AUTO-REJECT conditions:
- tone is tragic/negative/violent and visual looks cheerful, vacation-like, luxurious, playful, celebratory or relaxing;
- obvious website screenshot, UI screenshot, text-heavy plaque/sign, infographic, poster, watermark, logo or unusable text image unless explicitly requested;
- subject is tiny or the frame is mostly empty/useless;
- visibly low-quality, badly compressed, extremely blurry or poor scan when a cleaner alternative should exist;
- wrong historical era/country/person/event;
- modern object substitutes for a historical fact when context matters;
- candidate is visually generic to the point that it does not communicate the scene.

If Semantic lock is true:
- candidate must be compatible with EVERY required entity/context;
- wrong war, century, country, named person or modern substitute is an automatic reject.

If Semantic lock is false:
- judge CURRENT action/emotion/idea first;
- do not demand the story's main character if another visual honestly represents the beat.

For 9:16 Shorts, prefer a clear main subject and composition that can survive a vertical crop.
""".strip()
            parts.append({"text": prompt})
            data = self._generate_json(parts, temperature=0.02)
            if not isinstance(data, dict):
                return None
            score = max(0, min(100, int(float(data.get("score", 0)))))
            tone_match = max(0, min(100, int(float(data.get("tone_match", 0)))))
            quality_score = max(0, min(100, int(float(data.get("quality_score", 0)))))
            issues_raw = data.get("quality_issues") or []
            issues = [str(x)[:80] for x in issues_raw[:8]] if isinstance(issues_raw, list) else []
            threshold = 76 if scene.semantic_lock else 60
            accept = bool(data.get("accept", False)) and score >= threshold and tone_match >= 58 and quality_score >= 55
            return VisualJudgement(
                accept=accept,
                score=score,
                reason=str(data.get("reason", ""))[:300],
                mismatch=str(data.get("mismatch", ""))[:200],
                tone_match=tone_match,
                quality_score=quality_score,
                quality_issues=issues,
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
                "User-Agent": "video-ai/1.2",
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
