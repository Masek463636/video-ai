from __future__ import annotations

import re
from pathlib import Path

from .models import MotionKind, Scene, ShotPlan, Transcript, Word


_SENTENCE_END = re.compile(r"[.!?…]+$")
_STRIP = re.compile(r"[^\w\-]+", flags=re.UNICODE)

# Tiny language-agnostic-ish stopword set. This is deliberately conservative:
# the heuristic director is only V0.1 and will later be replaceable by an LLM.
_STOPWORDS = {
    "а", "и", "но", "или", "в", "во", "на", "по", "к", "ко", "у", "из", "за",
    "с", "со", "от", "до", "для", "что", "это", "как", "же", "бы", "не", "ну",
    "я", "ты", "он", "она", "они", "мы", "вы", "мой", "моя", "его", "ее", "её",
    "the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "for", "with",
    "this", "that", "it", "is", "are", "was", "were", "i", "you", "he", "she", "we",
}

_MOTIONS: tuple[MotionKind, ...] = (
    "zoom_in",
    "pan_right",
    "zoom_out",
    "pan_left",
)


def build_shot_plan(
    transcript: Transcript,
    audio: str | Path,
    *,
    target_scene_seconds: float = 2.1,
    min_scene_seconds: float = 1.25,
    max_scene_seconds: float = 3.2,
) -> ShotPlan:
    """Turn word-level timestamps into a first-pass short-form shot plan.

    V0.1 is deterministic on purpose. It groups speech into fast semantic beats,
    favors sentence boundaries, creates a stock-search query for every scene and
    rotates simple motion so still-image edits already feel less static.
    """
    if min_scene_seconds <= 0 or target_scene_seconds < min_scene_seconds:
        raise ValueError("invalid scene duration settings")
    if max_scene_seconds < target_scene_seconds:
        raise ValueError("max_scene_seconds must be >= target_scene_seconds")

    chunks = _chunk_words(
        transcript.words,
        target=target_scene_seconds,
        minimum=min_scene_seconds,
        maximum=max_scene_seconds,
    )

    scenes: list[Scene] = []
    for index, words in enumerate(chunks):
        caption = _join_words(words)
        scenes.append(
            Scene(
                start=round(words[0].start, 3),
                end=round(words[-1].end, 3),
                query=_make_search_query(caption),
                asset_kind="blank",
                motion=_MOTIONS[index % len(_MOTIONS)],
                caption=caption,
            )
        )

    return ShotPlan(audio=Path(audio), scenes=scenes)


def _chunk_words(
    words: list[Word],
    *,
    target: float,
    minimum: float,
    maximum: float,
) -> list[list[Word]]:
    chunks: list[list[Word]] = []
    current: list[Word] = []

    for word in words:
        current.append(word)
        duration = current[-1].end - current[0].start
        sentence_end = bool(_SENTENCE_END.search(word.text))

        should_cut = False
        if duration >= maximum:
            should_cut = True
        elif duration >= target and sentence_end:
            should_cut = True
        elif duration >= target + 0.45:
            should_cut = True

        if should_cut and duration >= minimum:
            chunks.append(current)
            current = []

    if current:
        if chunks and current[-1].end - current[0].start < minimum * 0.72:
            chunks[-1].extend(current)
        else:
            chunks.append(current)

    return chunks


def _join_words(words: list[Word]) -> str:
    text = " ".join(word.text for word in words)
    # Whisper-like word streams often put punctuation in separate-ish tokens.
    text = re.sub(r"\s+([,.!?;:…])", r"\1", text)
    return text.strip()


def _make_search_query(text: str, *, max_terms: int = 7) -> str:
    raw_tokens = text.split()
    useful: list[str] = []
    seen: set[str] = set()

    for raw in raw_tokens:
        token = _STRIP.sub("", raw).strip("_-").lower()
        if len(token) < 2 or token in _STOPWORDS or token in seen:
            continue
        seen.add(token)
        useful.append(token)
        if len(useful) >= max_terms:
            break

    if useful:
        return " ".join(useful)

    fallback = " ".join(raw_tokens[:max_terms]).strip()
    return fallback or "abstract background"
