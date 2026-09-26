from __future__ import annotations

import base64
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
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

    def stock_search_queries(self, scene: Scene, *, mode: str = "exact") -> list[str]:
        """Create short provider-friendly searches for the v2 Visual Director."""
        if not self.available:
            return []
        mode = mode if mode in {"exact", "broad"} else "exact"
        mode_rule = (
            "Keep the core visible subject/object and the real-world place. Use a common action only when it is easy to find in stock footage."
            if mode == "exact"
            else
            "Broaden to the same topic/place while preserving the core subject. Do not require the exact gesture or micro-action."
        )
        prompt = f"""
You write search queries for Pexels and Pixabay stock VIDEO.

NARRATION BEAT:
{scene.caption or ""}

DIRECTOR INTENT:
{_judge_description(scene)}

MODE: {mode.upper()}
{mode_rule}

Return ONLY JSON:
{{"queries": ["query one", "query two", "query three"]}}

RULES:
- Exactly 3 English queries.
- Each query is 2-5 simple words.
- Put the main visible noun first when possible.
- Use words that stock sites commonly tag: supermarket, shopper, milk, carton, receipt, shelf, price, package, factory, face, cart.
- No cinematic jargon, no abstract concepts, no metaphor, no adjectives like beautiful/aesthetic.
- Do not request text overlays, split screens, exact numeric labels, logos, or impossible micro-actions.
- Queries must be meaningfully different while staying on the SAME topic.
""".strip()
        try:
            data = self._generate_json([{"text": prompt}], temperature=0.03)
        except Exception:
            return []
        raw = data.get("queries", []) if isinstance(data, dict) else []
        out: list[str] = []
        seen: set[str] = set()
        for value in raw:
            q = re.sub(r"\s+", " ", str(value)).strip(" ,.;:-")
            words = re.findall(r"[A-Za-z0-9'-]+", q)
            if not (2 <= len(words) <= 6):
                continue
            key = q.casefold()
            if key in seen:
                continue
            seen.add(key)
            out.append(q[:100])
            if len(out) >= 3:
                break
        return out

    def choose_visual_candidates(
        self,
        scene: Scene,
        candidates: list[dict[str, Any]],
        *,
        mode: str = "exact",
    ) -> list[dict[str, Any]]:
        """See all candidate preview frames together and rank the best B-roll."""
        if not self.available or not candidates:
            return []
        mode = mode if mode in {"exact", "broad"} else "exact"
        mode_rule = (
            "Prefer the requested subject/object and action, but a close natural stock-footage action is acceptable."
            if mode == "exact"
            else
            "Exact action is not required. Preserve the main topic/object/place and choose honest contextual B-roll."
        )
        parts: list[dict[str, Any]] = [{
            "text": f"""
You are the Visual Director for a fast-paced vertical YouTube Short.
You will see MANY candidate frames at once. Compare them against each other and choose the best footage for THIS narration beat.

NARRATION:
{scene.caption or ""}

DIRECTOR INTENT:
{_judge_description(scene)}

TONE:
{scene.tone}

SELECTION MODE: {mode.upper()}
{mode_rule}

IMPORTANT:
- Pick visuals that actually contain the core subject/topic. Do not reward a pretty unrelated frame.
- Matching only the noun is not an action match. A walking dog does not show
  collapse; a shopping cart does not show reduced package volume. Score such
  contextual footage below a candidate showing the requested event/comparison.
- Reject obvious topic substitutions: milk is not beer/coffee; supermarket is not library; receipt is not landscape; measuring cup is not ocean.
- Prefer a clear human/object subject and footage usable in a 9:16 Short.
- Candidate URLs are already unique and unused; do not worry about repetition.
- Return up to 5 choices, best first.
""".strip()
        }]

        for row in candidates:
            path = Path(str(row.get("preview_path") or ""))
            if not path.exists():
                continue
            label = int(row.get("index", 0))
            parts.append({
                "text": (
                    f"CANDIDATE {label} | source={row.get('source','')} | "
                    f"search={row.get('search','')} | title={str(row.get('title',''))[:100]}"
                )
            })
            try:
                encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            except OSError:
                continue
            parts.append({"inline_data": {"mime_type": "image/jpeg", "data": encoded}})

        parts.append({"text": """
Return ONLY JSON:
{
  "choices": [
    {"index": 7, "fit": 95, "reason": "short reason"},
    {"index": 12, "fit": 84, "reason": "short reason"}
  ]
}

Rules for fit:
90-100 = visible evidence of the requested action/comparison and right subject
70-89 = close/contextual but honestly supports narration
50-69 = weak fallback
below 50 = weak/emergency-only
Always return the 5 best candidates available, even if some are weak. Score them honestly.
""".strip()})
        try:
            data = self._generate_json(parts, temperature=0.01)
        except Exception:
            return []
        raw = data.get("choices", []) if isinstance(data, dict) else []
        valid_indexes = {int(row.get("index", -1)) for row in candidates}
        out: list[dict[str, Any]] = []
        seen: set[int] = set()
        for item in raw:
            if not isinstance(item, dict):
                continue
            try:
                idx = int(item.get("index"))
                fit = max(0, min(100, int(float(item.get("fit", 0)))))
            except (TypeError, ValueError):
                continue
            if idx not in valid_indexes or idx in seen:
                continue
            seen.add(idx)
            out.append({
                "index": idx,
                "fit": fit,
                "reason": str(item.get("reason") or "")[:220],
            })
            if len(out) >= 5:
                break
        return out

    def choose_roll_visuals(
        self,
        scenes: list[Scene],
        groups: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Choose B-roll for the whole Short in one multimodal Gemini request.

        Each group contains one scene index plus a few locally searched preview
        frames. Gemini sees the entire sequence at once, so one request can
        optimize semantic fit and visual variety without spending one request
        per scene.
        """
        if not self.available or not groups:
            return []

        scene_map = {i: scene for i, scene in enumerate(scenes)}
        timeline = [
            {
                "scene": i,
                "caption": scene.caption or "",
                "intent": _judge_description(scene),
                "tone": scene.tone,
            }
            for i, scene in enumerate(scenes)
        ]

        parts: list[dict[str, Any]] = [{
            "text": f"""
You are the Visual Director for an ENTIRE fast-paced vertical YouTube Short.

You will see the full narration timeline and a small candidate set for several
scenes. Choose the best B-roll candidate for EACH scene while also making the
whole sequence visually varied and easy to understand.

FULL TIMELINE:
{json.dumps(timeline, ensure_ascii=False)}

GLOBAL RULES:
- Distinguish caller from recipient, observing from acting, and cause from reaction.
  Reading/scrolling a phone is not evidence of calling or dismissing a call.
- Use the observable requirements in visual_description. A shared noun without
  the required action is contextual footage: score below 65, never as exact.
- Favor continuity of setting, visible object and participant role across related
  beats when relevance is comparable; do not assume different people are one person.
- Judge each scene against its own caption and director intent.
- Prefer a clear human/animal/object ACTION over a static object when possible.
- Keep the core subject honest: milk is not coffee, receipt is not landscape,
  shopping cart is not random street footage.
- Prefer shots that remain readable in a vertical Short.
- Preserve the subject across related beats; vary action/framing when useful.
  Do not substitute a different subject merely to create visual variety.
- A matching noun without the requested event/comparison is contextual fallback,
  not a strong action match. Judge what the frames visibly demonstrate.
- It is better to select a close contextual shot than leave a generic unlocked
  scene empty.
- Return ONE choice per supplied scene whenever at least one candidate is
  reasonably relevant. Use candidate 0 only when every option is clearly wrong.
- Candidate numbers are LOCAL to each scene.
""".strip()
        }]

        valid: dict[int, set[int]] = {}
        for group in groups:
            try:
                scene_index = int(group.get("scene"))
            except (TypeError, ValueError):
                continue
            scene = scene_map.get(scene_index)
            if scene is None:
                continue
            rows = group.get("candidates") or []
            valid[scene_index] = set()
            parts.append({
                "text": (
                    f"\nSCENE {scene_index}\n"
                    f"NARRATION: {scene.caption or ''}\n"
                    f"INTENT: {_judge_description(scene)}\n"
                    f"TONE: {scene.tone}\n"
                    "CANDIDATES:"
                )
            })
            for row in rows:
                try:
                    candidate_index = int(row.get("candidate"))
                except (TypeError, ValueError):
                    continue
                path = Path(str(row.get("preview_path") or ""))
                if not path.exists():
                    continue
                valid[scene_index].add(candidate_index)
                parts.append({
                    "text": (
                        f"SCENE {scene_index} CANDIDATE {candidate_index} | "
                        f"source={row.get('source','')} | search={row.get('search','')} | "
                        f"title={str(row.get('title',''))[:100]}"
                    )
                })
                try:
                    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
                except OSError:
                    continue
                parts.append({"inline_data": {"mime_type": "image/jpeg", "data": encoded}})

        parts.append({"text": """
Return ONLY JSON:
{
  "choices": [
    {"scene": 0, "candidate": 2, "fit": 91, "reason": "short reason"},
    {"scene": 1, "candidate": 1, "fit": 78, "reason": "short reason"}
  ]
}

FIT GUIDE:
90-100 = strong semantic/action match
70-89 = good honest B-roll
50-69 = usable contextual fallback
30-49 = weak emergency fallback
0 = none of the supplied candidates should be used

Return at most one row per scene. Cover every supplied scene.
""".strip()})

        try:
            data = self._generate_json(parts, temperature=0.01)
        except Exception as exc:
            print(f"[v2-roll] Gemini batch request failed: {exc}", flush=True)
            return []

        if not isinstance(data, dict):
            print("[v2-roll] Gemini batch returned non-object JSON", flush=True)
            return []
        raw = data.get("choices") or data.get("selections") or data.get("scene_choices") or []
        out: list[dict[str, Any]] = []
        seen_scenes: set[int] = set()
        for item in raw:
            if not isinstance(item, dict):
                continue
            try:
                scene_index = int(item.get("scene"))
                candidate_index = int(item.get("candidate"))
                fit = max(0, min(100, int(float(item.get("fit", 0)))))
            except (TypeError, ValueError):
                continue
            if scene_index in seen_scenes or scene_index not in valid:
                continue
            if candidate_index != 0 and candidate_index not in valid[scene_index]:
                continue
            seen_scenes.add(scene_index)
            out.append({
                "scene": scene_index,
                "candidate": candidate_index,
                "fit": fit,
                "reason": str(item.get("reason") or "")[:220],
            })
        print(f"[v2-roll] Gemini parsed {len(out)}/{len(valid)} scene choice(s)", flush=True)
        return out


    def rewrite_search_queries(
        self,
        scene: Scene,
        *,
        rejected: list[dict[str, Any]] | None = None,
        previous_queries: list[str] | None = None,
        mode: str = "exact",
    ) -> list[str]:
        """Rewrite a failed stock search into concrete, searchable visual actions."""
        if not self.available:
            return []
        rejected = rejected or []
        previous_queries = previous_queries or []
        mode = mode if mode in {"exact", "close", "context"} else "exact"
        mode_rule = {
            "exact": "Keep the same concrete object and visible action, but phrase it in simpler stock-footage language.",
            "close": "Broaden the action while keeping the same subject/object/topic. A related usable B-roll action is preferred over an exact reenactment.",
            "context": "Search for the broader real-world setting/topic. Exact action is NOT required; return honest contextual B-roll that supports the narration.",
        }[mode]
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

FALLBACK LEVEL:
{mode.upper()}
{mode_rule}

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
- Obey the FALLBACK LEVEL above. CLOSE and CONTEXT must be visibly broader than EXACT.
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

    def judge_visual(self, scene: Scene, media_path: str | Path, *, candidate_title: str = "", source: str = "", match_level: str = "exact") -> VisualJudgement | None:
        if not self.available:
            return None
        match_level = match_level if match_level in {"exact", "close", "context"} else "exact"
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
B-roll match level: {match_level.upper()}

MATCH LEVEL RULES:
- EXACT: prefer the requested object/person AND visible action. Reject meaningful action/object mismatches.
- CLOSE: the exact gesture/action is NOT required. Accept the same subject/object/topic in a closely related usable situation. Example: narration says "grab milk from shelf"; pouring milk, holding a milk carton, or choosing dairy can be acceptable CLOSE B-roll.
- CONTEXT: exact object/action is NOT required. Accept honest contextual footage of the broader setting/topic if it supports the narration and does not contradict it. Example: supermarket aisle/shoppers can support a price-shopping narration.
- For CLOSE or CONTEXT, do NOT reject merely because the candidate is not a literal reenactment.

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
            semantic_floor = {"exact": 45, "close": 34, "context": 25}[match_level]
            if not scene.semantic_lock and raw_score < semantic_floor:
                score = min(raw_score, {"exact": 29, "close": 23, "context": 19}[match_level])
                mismatch = mismatch if mismatch and mismatch.lower() != "none" else "semantic relevance below minimum"
                reason = f"Low semantic relevance ({raw_score}/100). {reason}".strip()

            if scene.semantic_lock:
                accept = bool(data.get("accept", False)) and score >= 72 and quality_score >= 45
            elif match_level == "exact":
                accept = bool(data.get("accept", False)) and score >= 48 and tone_match >= 42 and quality_score >= 42
            elif match_level == "close":
                accept = score >= 40 and tone_match >= 35 and quality_score >= 40
            else:
                accept = score >= 32 and tone_match >= 30 and quality_score >= 38
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
            for attempt in range(2):
                req = urllib.request.Request(url, data=body, method="POST", headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": self.api_key,
                    "User-Agent": "video-ai/2.0.0",
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
                        detail = exc.read().decode("utf-8", errors="ignore")[:2000]
                    except Exception:
                        pass

                    # Free-tier 429s commonly include a short "Please retry in
                    # 16.8s" hint. Waiting once is much better than immediately
                    # falling through to another model and then hammering the
                    # next scene/batch.
                    if exc.code == 429 and attempt == 0:
                        match = re.search(
                            r"Please retry in\s+([0-9.]+)\s*(ms|s)",
                            detail,
                            flags=re.IGNORECASE,
                        )
                        if match:
                            value = float(match.group(1))
                            delay = value / 1000.0 if match.group(2).lower() == "ms" else value
                            delay = max(0.25, min(delay + 0.35, 30.0))
                            print(
                                f"[gemini] rate limited on {model}; waiting {delay:.2f}s and retrying once",
                                flush=True,
                            )
                            time.sleep(delay)
                            continue

                    errors.append(f"{model}: HTTP {exc.code} {detail[:500]}")
                    break
                except Exception as exc:
                    errors.append(f"{model}: {exc}")
                    break
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
