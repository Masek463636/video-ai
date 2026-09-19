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
_SOFT_CONTEXT_RE = re.compile(r"(?:\.|\s)*Soft historical setting:\s*[^.]+\.?", flags=re.IGNORECASE)


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
- The CURRENT caption is the authority for the visible beat.
- Global story setting is background knowledge, NOT a mandatory property of every unlocked visual.
- For generic stress, shock, sleep, emotion, money, fire or other universal concepts, modern/generic B-roll is allowed unless the caption itself makes period identity essential.
- Tone describes how the visual should FEEL, not just the nouns it contains.
- Mass death, casualties, destruction and tragedy must be tragic/negative/violent, never cheerful, vacation-like, luxurious or relaxing.
- semantic_lock is a HARD FACT LOCK, not general story context.
- Set semantic_lock=true ONLY when the CURRENT scene caption explicitly names a specific person, named event/war/rebellion, named historical institution, dynasty, or similarly concrete factual entity that the visual must depict accurately.
- DO NOT hard-lock a person/event merely because it appeared in another scene.
- Pronouns do NOT automatically justify repeating the person's portrait. Use the action, emotion or consequence being spoken instead.
- Never use the same visual idea on consecutive scenes if another honest visual exists.
- A named person's portrait should usually appear once when introduced, not on every later reference.
- A map should normally appear at most once in a ~20 second Short unless geography actually changes.
- Use historical_archive for exact historical facts and genuinely historical action beats.
- Use stock_video for generic actions/concepts that can honestly be represented.
- Use generic_image for illustrations, dreams, emotions and conceptual beats when a still is stronger.
- Use memes sparingly: usually 0-2 per ~20 seconds. Never use a meme for a hard-locked factual beat.
- Search queries for unlocked scenes should describe the current ACTION/EMOTION first.
- Never ask for text overlays/logos/subtitles inside the visual.
""".strip()
        data = self._generate_json([{"text": prompt}], temperature=0.11)
        raw = data.get("scenes", []) if isinstance(data, dict) else []
        return [item for item in raw if isinstance(item, dict)]

    def rewrite_search_queries(
        self,
        scene: Scene,
        *,
        rejected: list[dict[str, Any]] | None = None,
        previous_queries: list[str] | None = None,
    ) -> list[str]:
        """Rewrite a failed stock search into concrete, searchable visual actions."""
        if not self.available:
            return []
        rejected = rejected or []
        previous_queries = previous_queries or []
        feedback = [
            {
                "title": str(item.get("title", ""))[:120],
                "reason": str(item.get("reason") or item.get("mismatch") or "")[:220],
            }
            for item in rejected[-8:]
        ]
        prompt = f"""
You repair failed stock-footage searches for a vertical YouTube Short.

NARRATION BEAT:
{scene.caption or ""}

DIRECTOR INTENT:
{scene.visual_description or scene.query or ""}

MEDIA TYPE:
{scene.visual_mode} / {scene.source_mode}

PREVIOUS SEARCHES THAT FAILED:
{json.dumps(previous_queries[-8:], ensure_ascii=False)}

REJECTED RESULTS AND WHY:
{json.dumps(feedback, ensure_ascii=False)}

Return ONLY JSON:
{{
  "queries": [
    "concrete English stock search",
    "different concrete English stock search",
    "different concrete English stock search",
    "different concrete English stock search"
  ]
}}

STRICT RULES:
- Every query must describe something a camera can literally see.
- Use 3-7 simple English words.
- Preferred structure: SUBJECT + VISIBLE ACTION + OBJECT/PLACE.
- Prefer common stock-footage vocabulary that Pexels/Pixabay can actually match.
- Search for the underlying action, not an abstract metaphor.
- If narration is abstract (inflation, betrayal, value loss), translate it into
  a simple visible human action (checking wallet, shocked shopper, disappointed person).
- Queries must be meaningfully different from one another.
- Preserve critical concrete nouns from narration when useful (milk, supermarket,
  shopping cart, package, conveyor, wallet, price).
- DO NOT use: abstract, concept, metaphor, animation, background, VJ, earth,
  planet, revolution, energy, aesthetic, symbolic, cinematic concept.
- DO NOT include camera jargon unless essential.
- DO NOT request text overlays, logos, screenshots, UI, posters or infographics.
- For video scenes, prefer real human/action footage.
""".strip()
        try:
            data = self._generate_json([{"text": prompt}], temperature=0.06)
        except Exception:
            return []
        raw = data.get("queries", []) if isinstance(data, dict) else []
        out: list[str] = []
        seen: set[str] = set()
        banned = {
            "abstract", "concept", "metaphor", "animation", "background", "vj",
            "earth", "planet", "revolution", "aesthetic", "symbolic",
        }
        for value in raw:
            q = re.sub(r"\s+", " ", str(value)).strip(" ,.;:-")
            words = re.findall(r"[A-Za-z0-9'-]+", q)
            if not (2 <= len(words) <= 9):
                continue
            lowered = q.casefold()
            if any(re.search(rf"\b{re.escape(term)}\b", lowered) for term in banned):
                continue
            if lowered not in seen:
                seen.add(lowered)
                out.append(q[:140])
            if len(out) >= 4:
                break
        return out

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
            director_intent = _judge_description(scene)
            prompt = f"""
You are a visual relevance + tone + quality judge for an automatically edited YouTube Short.
The supplied images are representative frames from ONE candidate asset.

Narration in this scene: {scene.caption or ''}
Scene tone: {scene.tone}
Director wants to show: {director_intent}
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

Overall score measures semantic relevance to the CURRENT narration beat.
Tone_match 0-100 measures emotional compatibility.
Quality_score 0-100 measures whether this looks like usable Shorts footage/image.

For SEMANTIC LOCK scenes be strict: wrong person/event/war/country/century is a reject.
For UNLOCKED scenes:
- judge the current action/emotion/concept first;
- global historical setting is advisory only unless the current narration explicitly makes era/country identity important;
- do NOT reject generic emotional/action B-roll merely because it is modern;
- do reject a visually unrelated asset even if its mood is correct.
Strongly penalize obvious website screenshots, watermarks, posters, tiny subjects, broken images, extremely poor scans, or a severe tone contradiction.
For 9:16 Shorts prefer a clear main subject and composition that survives a vertical crop.
""".strip()
            parts.append({"text": prompt})
            data = self._generate_json(parts, temperature=0.02)
            if not isinstance(data, dict):
                return None
            raw_score = max(0, min(100, int(float(data.get("score", 0)))))
            tone_match = max(0, min(100, int(float(data.get("tone_match", 0)))))
            quality_score = max(0, min(100, int(float(data.get("quality_score", 0)))))
            issues_raw = data.get("quality_issues") or []
            issues = [str(x)[:80] for x in issues_raw[:8]] if isinstance(issues_raw, list) else []
            mismatch = str(data.get("mismatch", ""))[:200]
            reason = str(data.get("reason", ""))[:300]

            score = raw_score
            if not scene.semantic_lock and raw_score < 45:
                score = 29
                mismatch = mismatch if mismatch and mismatch.lower() != "none" else "semantic relevance below minimum"
                reason = f"Low semantic relevance ({raw_score}/100). {reason}".strip()

            if scene.semantic_lock:
                accept = bool(data.get("accept", False)) and score >= 72 and quality_score >= 45
            else:
                accept = bool(data.get("accept", False)) and score >= 48 and tone_match >= 42 and quality_score >= 42
            return VisualJudgement(
                accept=accept,
                score=score,
                reason=reason,
                mismatch=mismatch,
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
                "User-Agent": "video-ai/1.5.1",
            })
            try:
                with urllib.request.urlopen(req, timeout=60) as response:
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


def _judge_description(scene: Scene) -> str:
    value = scene.visual_description or scene.query or "documentary scene"
    if scene.semantic_lock:
        return value
    value = _SOFT_CONTEXT_RE.sub("", value)
    return re.sub(r"\s+", " ", value).strip(" .,-") or (scene.caption or scene.query or "documentary scene")


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
    ffmpeg = shutil.which("ffmpeg")
    if not path.exists() or ffmpeg is None:
        return []
    suffix = path.suffix.lower()
    is_video = suffix in {".mp4", ".mov", ".mkv", ".webm", ".avi", ".ogv", ".ogg"}

    if not is_video and not _silent_decode_probe(path):
        return []

    temp_dir = Path(tempfile.mkdtemp(prefix="video-ai-gemini-"))
    if not is_video:
        target = temp_dir / "preview_0.jpg"
        cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(path), "-frames:v", "1", "-vf", "scale='min(768,iw)':-2", "-q:v", "4", str(target)]
        try:
            subprocess.run(cmd, check=True, timeout=35, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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
        cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{seek:.3f}", "-i", str(path), "-frames:v", "1", "-vf", "scale='min(768,iw)':-2", "-q:v", "4", str(target)]
        try:
            subprocess.run(cmd, check=True, timeout=35, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if target.exists() and target.stat().st_size > 512:
                previews.append(target)
        except Exception:
            continue
    if not previews:
        shutil.rmtree(temp_dir, ignore_errors=True)
    return previews


def _silent_decode_probe(path: Path) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return True
    try:
        completed = subprocess.run(
            [
                ffmpeg,
                "-v", "error",
                "-xerror",
                "-err_detect", "explode",
                "-i", str(path),
                "-frames:v", "1",
                "-f", "null", "-",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=12,
            check=False,
        )
        return completed.returncode == 0 and not completed.stderr.strip()
    except Exception:
        return False


def _probe_duration(path: Path) -> float:
    try:
        completed = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)], check=True, capture_output=True, text=True, timeout=20)
        return float(json.loads(completed.stdout).get("format", {}).get("duration") or 0.0)
    except Exception:
        return 0.0
