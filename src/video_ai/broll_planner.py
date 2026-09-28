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
    viral_style: bool = False,
    style5: bool = False,
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
    # Viral reactions are a separate timed foreground layer, not random base shots.
    meme_names = [] if (viral_style or style5) else _meme_names(meme_dir)
    meme_catalog_prompt = (
        ""
        if style5
        else (
            "AVAILABLE LOCAL MEMES (exact filenames; optional):\n"
            + str(meme_names if meme_names else "(none)")
        )
    )
    duration = transcript.duration

    if viral_style:
        target_seconds = 1.15
        target_beats = max(
            8,
            min(35, round(duration / target_seconds)),
        )

        beat_rule = (
            "- Typical beat length: 0.8-1.5 seconds. "
            "KEEP IT FAST. Avoid >2.0s unless continuity truly helps."
        )

        min_shot_seconds = 0.65
        max_gap_seconds = 1.90

    else:
        target_seconds = 2.15
        target_beats = max(
            5,
            min(18, round(duration / target_seconds)),
        )

        beat_rule = (
            "- Typical beat length: 1.4-3.0 seconds. "
            "Avoid >3.4s unless continuity truly helps."
        )

        min_shot_seconds = 0.85
        max_gap_seconds = 3.50

    meme_rule = (
        "- For Style 5, use MEME whenever it is a better semantic match than stock. "
        "There is no meme quota; do not force memes, but do not artificially limit them either. "
        "Never repeat the same meme."
        if style5
        else "- Use MEME rarely: maximum 1 meme per ~15-20s, never repeat the same meme."
    )
    scene_count_rule = (
        "Choose the number of visual scenes yourself from the story's meaning and rhythm. "
        "There is NO target scene count. Cut only when the visual idea, action, reaction, setup, "
        "or punchline genuinely changes; do not create filler cuts just to increase pace."
        if style5
        else f"Target roughly {target_beats} visual beats across {duration:.1f}s."
    )
    style5_beat_rule = (
        "- You control the scene boundaries. Avoid useless micro-scenes, but keep a scene longer "
        "when one continuous visual idea genuinely covers the narration."
        if style5
        else beat_rule
    )

    schema_meme_field = (
        ""
        if style5
        else ',\n      "meme_filename": ""'
    )
    query_schema = (
        "2-6 English search words describing the visible action/reaction"
        if style5
        else "2-5 concrete English stock-search words"
    )
    media_preference_rule = (
        "- Choose VIDEO for literal physical actions/environments that stock can show specifically. "
        "Choose MEME for ironic/social/internal reactions or punchlines when stock would be generic or misleading."
        if style5
        else "- Prefer VIDEO for actions, reactions, environments and modern generic concepts."
    )

    prompt = f"""
You are a senior short-form video editor planning the COMPLETE visual timeline
for a vertical YouTube Short. There is only voiceover underneath: every moment
must have a useful visual.

FULL TRANSCRIPT WITH WORD INDEXES:
{indexed}

{meme_catalog_prompt}

{scene_count_rule}

Return ONLY JSON:
{{
  "beats": [
    {{
      "start_idx": 0,
      "end_idx": 8,
      "kind": "video|image|meme",
      "query": "{query_schema}",
      "alternatives": ["different concrete query", "another different query"],
      "visual_description": "what the viewer should literally see",
      "reason": "why this visual fits this exact narration beat"{schema_meme_field}
    }}
  ]
}}

EDITING RULES:
- Plan the WHOLE story before choosing any beat.
- Cover the transcript from first word to last word in chronological order.
{style5_beat_rule}
{media_preference_rule}
- Prefer IMAGE only for exact historical portraits/maps/documents, still artwork,
  or when motion footage would be dishonest.
{meme_rule}
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

    if viral_style:
        prompt += ("\nVIRAL BASE FOOTAGE: use grounded story footage; memes are added separately. "
                   "Do not choose meme beats or generic screaming/grimacing stock actors. "
                   "Keep everyday situations emotionally proportionate. For choosing a film show "
                   "browsing titles, discussing a choice or using a remote, not an unrelated comic face. "
                   "Fast cuts must still explain the action. A short fragment such as 'not yet' "
                   "inherits its meaning from the surrounding sentence.")
    elif style5:
        prompt += (
            "\nSTYLE 5 BASE-TRACK RULES:\n"
            "- MEMES ARE NOT OVERLAYS. A meme is a PRIMARY visual beat that replaces stock footage "
            "for that beat on the main timeline.\n"
            "- First ask: can stock footage show this phrase LITERALLY and SPECIFICALLY? "
            "If yes, use VIDEO.\n"
            "- If the phrase is abstract, social, ironic, exaggerated, internal/emotional, a punchline, "
            "or would force a generic/weak stock metaphor, prefer MEME instead.\n"
            "- Examples that often deserve MEME: awkward internal thoughts, 'me pretending everything is fine', "
            "sarcastic reactions, absurd comparisons, embarrassment, disbelief, 'I am done', social anxiety, "
            "or a joke whose exact action is not realistically searchable.\n"
            "- Examples that should stay VIDEO: opening a fridge, checking a phone, walking into a room, "
            "drinking water, driving, shopping, cooking, opening a package, or any other literal visible action.\n"
            "- Do NOT use a meme just because it is funny. Use it when it is a BETTER semantic match than stock.\n"
            "- There is NO meme quota. Clean stock-only stretches are good. Several meme beats are allowed "
            "when several consecutive phrases genuinely cannot be represented precisely with stock.\n"
            "- THIS PASS DECIDES ONLY THE EDIT: scene boundaries, media type, and what should be visible. "
            "Do not choose or name any media file.\n"
            "- After this plan is returned, the material engine will search real candidates and ask you to visually rank them.\n"
            "- For MEME beats, query must be a concise English reaction/search phrase suitable for meme search, and "
            "visual_description must describe the exact emotion/joke that should be visible.\n"
            "- Emoji/sticker reactions are handled later and are NOT part of this base-track decision."
        )
    data = client._generate_json([{"text": prompt}], temperature=0.08)
    raw_beats = data.get("beats", []) if isinstance(data, dict) else []
    minimum_beats = 1 if style5 else 2
    if not isinstance(raw_beats, list) or len(raw_beats) < minimum_beats:
        raise RuntimeError("Gemini donor planner returned too few beats")

    planned = _sanitize_beats(raw_beats, len(words))
    starts = (
        _style5_start_indices(planned, len(words))
        if style5
        else _normalize_start_indices(
            planned,
            transcript,
            target_seconds=target_seconds,
            min_shot_seconds=min_shot_seconds,
            max_gap_seconds=max_gap_seconds,
        )
    )
    if len(starts) < minimum_beats:
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
        if kind == "meme":
            if style5:
                # Style 5 is two-stage: planner decides MEME vs stock only.
                # Asset selection happens later from local memes + internet candidates.
                meme_filename = None
            elif not meme_filename or meme_filename not in meme_names:
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


def _style5_start_indices(planned: list[dict], word_count: int) -> list[int]:
    """Preserve Gemini's Style 5 cut count; only repair invalid coverage at word 0."""
    starts = sorted({
        int(item["start_idx"])
        for item in planned
        if 0 <= int(item["start_idx"]) < word_count
    })
    if not starts:
        return []
    if starts[0] != 0:
        # Repair coverage without inventing an extra scene/cut.
        starts[0] = 0
    return sorted(set(starts))


def _normalize_start_indices(
    planned: list[dict],
    transcript: Transcript,
    *,
    target_seconds: float,
    min_shot_seconds: float = 0.85,
    max_gap_seconds: float = 3.50,
) -> list[int]:
    words = transcript.words
    starts = sorted({int(item["start_idx"]) for item in planned if 0 <= int(item["start_idx"]) < len(words)})
    if not starts or starts[0] != 0:
        starts.insert(0, 0)

    # Drop cut points that would create twitchy sub-second shots.
    filtered = [starts[0]]
    for idx in starts[1:]:
        if words[idx].start - words[filtered[-1]].start >= min_shot_seconds:
            filtered.append(idx)

    # Fill suspiciously long planner gaps with a neutral midpoint cut.
    result: list[int] = []
    for pos, idx in enumerate(filtered):
        result.append(idx)
        next_idx = filtered[pos + 1] if pos + 1 < len(filtered) else len(words)
        start_time = words[idx].start
        end_time = words[next_idx].start if next_idx < len(words) else words[-1].end
        while end_time - start_time > max_gap_seconds:
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
