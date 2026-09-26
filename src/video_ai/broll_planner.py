from __future__ import annotations

import re
from pathlib import Path

from .director import _global_visual_context, _join_words, _rule_semantic_lock
from .models import Scene, ShotPlan, Transcript
from .quality_guard import infer_tone


def build_donor_shot_plan(
    transcript: Transcript,
    audio: str | Path,
    *,
    meme_dir: str | Path | None = None,
) -> ShotPlan:
    """Whole-transcript storyboard planner inspired by AutoBroll/MoneyPrinterTurbo.

    Unlike the legacy per-scene planner, Gemini sees every indexed word before
    deciding visual cut points. The result covers the complete narration because
    video-ai has no talking-head base layer underneath the B-roll.
    """
    from .gemini_ai import get_gemini_client

    client = get_gemini_client()
    if client is None:
        raise RuntimeError("Gemini is required for donor storyboard planning")

    words = transcript.words
    if not words:
        raise ValueError("Transcript contains no words")

    indexed = " ".join(f"{i}:{word.text}" for i, word in enumerate(words))
    meme_names = _meme_names(meme_dir)
    duration = transcript.duration
    target_beats = max(5, min(18, round(duration / 2.15)))

    prompt = f"""
You are a senior short-form video editor planning the COMPLETE visual timeline
for a vertical YouTube Short. There is only voiceover underneath: every moment
must have a useful visual.

FULL TRANSCRIPT WITH WORD INDEXES:
{indexed}

AVAILABLE LOCAL MEMES (exact filenames; optional):
{meme_names if meme_names else "(none)"}

Target roughly {target_beats} visual beats across {duration:.1f}s.

Return ONLY JSON:
{{
  "beats": [
    {{
      "start_idx": 0,
      "end_idx": 8,
      "kind": "video|image|meme",
      "query": "2-5 concrete English stock-search words",
      "alternatives": ["different concrete query", "another different query"],
      "visual_description": "what the viewer should literally see",
      "reason": "why this visual fits this exact narration beat",
      "meme_filename": ""
    }}
  ]
}}

EDITING RULES:
- Plan the WHOLE story before choosing any beat.
- Cover the transcript from first word to last word in chronological order.
- Typical beat length: 1.4-3.0 seconds. Avoid >3.4s unless continuity truly helps.
- Prefer VIDEO for actions, reactions, environments and modern generic concepts.
- Prefer IMAGE only for exact historical portraits/maps/documents, still artwork,
  or when motion footage would be dishonest.
- Use MEME rarely: maximum 1 meme per ~15-20s, never repeat the same meme.
- Search queries must describe VISIBLE ACTION/SUBJECT, not abstract narration.
- Show what is HAPPENING, not merely the noun that was spoken.
- Build cause -> action -> consequence across connected beats. Resolve who acts
  and who reacts before writing queries; do not exchange their roles.
- In visual_description specify the observable action and a misleading near-match
  to avoid. Example: ignoring an incoming call requires noticing/rejecting it;
  scrolling a phone only illustrates phone use, not ignoring a call.
- Prefer feasible stock actions with short literal queries. Do not prepend
  "person interacting with" to every object or requested action.
- Preserve recurring roles via compatible setting/clothing or object close-ups;
  do not imply unrelated stock actors are the same identifiable person.

- Preserve the same subject/object/location when consecutive beats explain it.
  A product comparison needs the same product; a reaction needs its cause.
  Vary the action or framing, not the subject merely for variety.
- Avoid duplicate footage and redundant shots. Reusing the subject with a new
  informative action is useful continuity, not repetition.
- Plan visible evidence: what does this shot teach beyond the subtitle?
  For a comparison show both states; for a quantity show its label/measurement;
  for an action show the event, not just an unrelated shot of the same animal.
- Specify this observable evidence in visual_description and carry the concrete
  subject into queries even when the current words only say 'it' or 'the same'.
- Do not invent a different subject or a visual metaphor to fill a difficult beat.
- Never use abstract sky/clouds/light/glow in two adjacent beats. After one atmospheric abstract shot, the next beat must use a grounded human action, object, place, document, or event.
- Spiritual/religious narration does NOT automatically mean clouds or light rays; prefer concrete visible actions such as praying hands, a church/temple interior, candles, a historical religious image, or a person reacting when context allows.
- For a price/money beat, do not repeatedly use a rich-man reaction meme.
- For phone/computer/store examples, vary subject, angle and action.
- Historical named people/events must stay historically accurate.
- No text overlays, watermarks, logos or screenshots as the visual itself.
- start_idx/end_idx are word indexes from the transcript above.
""".strip()

    data = client._generate_json([{"text": prompt}], temperature=0.08)
    raw_beats = data.get("beats", []) if isinstance(data, dict) else []
    if not isinstance(raw_beats, list) or len(raw_beats) < 2:
        raise RuntimeError("Gemini donor planner returned too few beats")

    planned = _sanitize_beats(raw_beats, len(words))
    starts = _normalize_start_indices(planned, transcript, target_seconds=2.15)
    if len(starts) < 2:
        raise RuntimeError("Donor planner could not build a useful timeline")

    by_start = {int(item["start_idx"]): item for item in planned}
    global_context = _global_visual_context(transcript.text)
    scenes: list[Scene] = []

    for pos, start_idx in enumerate(starts):
        next_start = starts[pos + 1] if pos + 1 < len(starts) else len(words)
        end_idx = max(start_idx, next_start - 1)
        beat = _closest_beat(start_idx, planned)
        slice_words = words[start_idx : end_idx + 1]
        caption = _join_words(slice_words)

        lock, entities, context, fallback = _rule_semantic_lock(caption, global_context)
        kind = str(beat.get("kind", "video")).lower().strip()
        if kind not in {"video", "image", "meme"}:
            kind = "video"

        meme_filename = str(beat.get("meme_filename", "") or "").strip() or None
        if kind == "meme" and (not meme_filename or meme_filename not in meme_names):
            kind = "video"
            meme_filename = None

        if lock:
            visual_mode = "image"
            source_mode = "historical_archive"
            motion_preset = "micro_push"
            meme_filename = None
        elif kind == "meme":
            visual_mode = "meme"
            source_mode = "meme_library"
            motion_preset = "none"
        elif kind == "image":
            visual_mode = "image"
            source_mode = "generic_image"
            motion_preset = "slow_push"
        else:
            visual_mode = "video"
            source_mode = "stock_video"
            motion_preset = "none"

        query = _clean_query(str(beat.get("query", "")))
        alternatives = [
            _clean_query(str(value))
            for value in (beat.get("alternatives") or [])
            if _clean_query(str(value))
        ]
        if not query:
            query = _fallback_query(caption, visual_mode)
        queries = _unique([query, *alternatives, _fallback_query(caption, visual_mode)])[:6]

        description = re.sub(r"\s+", " ", str(beat.get("visual_description", "") or "")).strip()
        if not description:
            description = caption

        scenes.append(
            Scene(
                start=round(slice_words[0].start, 3),
                end=round(slice_words[-1].end, 3),
                query=queries[0],
                search_queries=queries,
                visual_description=description[:500],
                asset_kind="blank",
                motion="none",
                caption=caption,
                caption_words=list(slice_words),
                visual_mode=visual_mode,  # type: ignore[arg-type]
                source_mode=source_mode,  # type: ignore[arg-type]
                motion_preset=motion_preset,  # type: ignore[arg-type]
                meme_filename=meme_filename,
                semantic_lock=lock,
                required_entities=entities,
                required_context=context,
                semantic_fallback=fallback,
                tone=infer_tone(caption),
            )
        )

    # Retrieval deduplicates actual footage. Repeated subject queries can be
    # intentional continuity; do not rotate them into generic alternatives.

    return ShotPlan(
        audio=Path(audio),
        scenes=scenes,
        director_source="donor_gemini",
        director_model=client.last_model,
    )


def _sanitize_beats(raw_beats: list[object], word_count: int) -> list[dict]:
    out: list[dict] = []
    seen_starts: set[int] = set()
    for raw in raw_beats:
        if not isinstance(raw, dict):
            continue
        try:
            start = int(raw.get("start_idx", -1))
            end = int(raw.get("end_idx", start))
        except (TypeError, ValueError):
            continue
        start = max(0, min(word_count - 1, start))
        end = max(start, min(word_count - 1, end))
        if start in seen_starts:
            continue
        seen_starts.add(start)
        item = dict(raw)
        item["start_idx"] = start
        item["end_idx"] = end
        out.append(item)
    out.sort(key=lambda item: int(item["start_idx"]))
    return out


def _normalize_start_indices(planned: list[dict], transcript: Transcript, *, target_seconds: float) -> list[int]:
    words = transcript.words
    starts = sorted({int(item["start_idx"]) for item in planned if 0 <= int(item["start_idx"]) < len(words)})
    if not starts or starts[0] != 0:
        starts.insert(0, 0)

    # Drop cut points that would create twitchy sub-second shots.
    filtered = [starts[0]]
    for idx in starts[1:]:
        if words[idx].start - words[filtered[-1]].start >= 0.85:
            filtered.append(idx)

    # Fill suspiciously long planner gaps with a neutral midpoint cut.
    result: list[int] = []
    for pos, idx in enumerate(filtered):
        result.append(idx)
        next_idx = filtered[pos + 1] if pos + 1 < len(filtered) else len(words)
        start_time = words[idx].start
        end_time = words[next_idx].start if next_idx < len(words) else words[-1].end
        while end_time - start_time > 3.5:
            target = start_time + min(target_seconds, (end_time - start_time) / 2)
            split = _nearest_word_index(words, target, low=idx + 1, high=next_idx - 1)
            if split is None or split <= result[-1]:
                break
            result.append(split)
            idx = split
            start_time = words[idx].start

    return sorted(set(result))


def _nearest_word_index(words, target: float, *, low: int, high: int) -> int | None:
    if low > high:
        return None
    best = None
    best_distance = float("inf")
    for index in range(max(0, low), min(len(words) - 1, high) + 1):
        distance = abs(words[index].start - target)
        if distance < best_distance:
            best = index
            best_distance = distance
    return best


def _closest_beat(start_idx: int, planned: list[dict]) -> dict:
    previous = planned[0]
    for item in planned:
        if int(item["start_idx"]) > start_idx:
            break
        previous = item
    return previous


def _clean_query(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip(" ,.;:-")
    return value[:160]


def _fallback_query(caption: str, visual_mode: str) -> str:
    words = [
        re.sub(r"[^\w-]+", "", token, flags=re.UNICODE)
        for token in caption.split()
    ]
    useful = [word for word in words if len(word) >= 4][:4]
    base = " ".join(useful) or "human reaction"
    suffix = "real footage" if visual_mode == "video" else "documentary image"
    return f"{base} {suffix}".strip()


def _unique(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        q = _clean_query(value)
        if q and q.casefold() not in seen:
            seen.add(q.casefold())
            out.append(q)
    return out


def _dedupe_neighbor_queries(scenes: list[Scene]) -> None:
    recent: list[str] = []
    for scene in scenes:
        key = scene.query.casefold().strip()
        if key in recent:
            alternatives = [q for q in scene.search_queries if q.casefold().strip() not in recent]
            if alternatives:
                scene.search_queries = _unique([*alternatives, *scene.search_queries])
                scene.query = scene.search_queries[0]
        recent.append(scene.query.casefold().strip())
        recent = recent[-4:]


def _meme_names(meme_dir: str | Path | None) -> list[str]:
    if not meme_dir:
        return []
    root = Path(meme_dir)
    if not root.exists():
        return []
    allowed = {".mp4", ".mov", ".webm", ".gif", ".png", ".jpg", ".jpeg"}
    return sorted(path.name for path in root.iterdir() if path.is_file() and path.suffix.lower() in allowed)
