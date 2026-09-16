from __future__ import annotations

import re
from pathlib import Path

from .models import MotionKind, Scene, ShotPlan, Transcript, VisualMode, Word


_SENTENCE_END = re.compile(r"[.!?…]+$")
_STRIP = re.compile(r"[^\w\-]+", flags=re.UNICODE)

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

_GLOBAL_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("китай", "китайск"), "China Chinese"),
    (("тайпин",), "Taiping Rebellion"),
    (("сюцюан", "хун сю", "hong xiu"), "Hong Xiuquan"),
    (("ссср", "советск"), "Soviet Union Soviet"),
    (("украин",), "Ukraine Ukrainian"),
    (("росси", "русск"), "Russia Russian"),
    (("америк", "сша"), "United States American"),
    (("герман", "немец"), "Germany German"),
    (("япон",), "Japan Japanese"),
    (("франц",), "France French"),
    (("британ", "англи"), "Britain British"),
    (("римск", "римская импер"), "Roman Empire"),
    (("егип",), "Egypt Egyptian"),
)

_SCENE_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("экзам", "госэкзам", "тест"), "student civil service examination"),
    (("стресс", "нерв", "расстро", "груст", "провал"), "stressed disappointed person"),
    (("чиновник", "служб"), "government official civil service"),
    (("сон", "спал", "снилось"), "sleeping person dream"),
    (("галлюцин", "видение", "озарен"), "surreal vision revelation"),
    (("иисус", "христ", "религи", "бог"), "Jesus Christian religious painting"),
    (("арм", "солдат", "войск"), "historical soldiers army"),
    (("восстан", "бунт", "битв", "револю"), "historical rebellion battle"),
    (("деньг", "цена", "миллион", "тысяч"), "money finance"),
    (("телефон", "сообщен", "звон"), "person using smartphone"),
    (("машин", "авто"), "car driving road"),
    (("компьют", "ютуб", "видео"), "video creator computer camera"),
    (("дом", "квартир"), "home apartment"),
    (("друг",), "friends talking"),
    (("парень", "мужчин"), "young man portrait"),
    (("девуш", "женщин"), "young woman portrait"),
    (("отнош", "пара"), "young couple relationship"),
)

_HISTORY_STEMS = (
    "истор", "восстан", "импер", "династ", "корол", "войн", "арм", "солдат",
    "битв", "револю", "xix", "xviii", "xx век",
)

_VIDEO_STEMS = (
    "идет", "идут", "беж", "летит", "летел", "едет", "едут", "горит", "взрыв",
    "арм", "солдат", "войск", "битв", "восстан", "толп", "марш", "атак", "стрел",
    "поех", "плыв", "движ", "танцу", "дерет", "сраж",
)

_MEME_STEMS = (
    "вдруг", "и тут", "прикол", "шок", "жесть", "серьезно", "неожидан", "пиздец",
    "лол", "смеш", "офиг", "охрен", "что за", "ну и",
)


def build_shot_plan(
    transcript: Transcript,
    audio: str | Path,
    *,
    target_scene_seconds: float = 2.1,
    min_scene_seconds: float = 1.25,
    max_scene_seconds: float = 3.2,
) -> ShotPlan:
    """Turn word timestamps into context-aware short-form scenes.

    V0.7 adds visual intent. Each scene gets an image/video/meme preference so
    the renderer is no longer fed a monotonous sequence of still photos.
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
    full_text = _join_words(transcript.words)
    global_context = _global_visual_context(full_text)

    scenes: list[Scene] = []
    captions = [_join_words(words) for words in chunks]
    for index, (words, caption) in enumerate(zip(chunks, captions)):
        previous_caption = captions[index - 1] if index > 0 else ""
        next_caption = captions[index + 1] if index + 1 < len(captions) else ""
        neighborhood = " ".join(x for x in (previous_caption, caption, next_caption) if x)
        visual_mode = _choose_visual_mode(caption, index=index, duration=words[-1].end - words[0].start)
        scenes.append(
            Scene(
                start=round(words[0].start, 3),
                end=round(words[-1].end, 3),
                query=_make_search_query(caption, global_context=global_context, neighborhood=neighborhood),
                asset_kind="blank",
                motion=_MOTIONS[index % len(_MOTIONS)],
                caption=caption,
                visual_mode=visual_mode,
            )
        )

    return ShotPlan(audio=Path(audio), scenes=scenes)


def _choose_visual_mode(text: str, *, index: int, duration: float) -> VisualMode:
    lowered = text.lower()
    if any(stem in lowered for stem in _MEME_STEMS):
        return "meme"
    if any(stem in lowered for stem in _VIDEO_STEMS):
        return "video"
    # Force some motion variety in otherwise static narration. The source search
    # can still fall back to an image if no suitable open video exists.
    if index > 0 and index % 3 == 0 and duration >= 1.6:
        return "video"
    return "image"


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
    text = re.sub(r"\s+([,.!?;:…])", r"\1", text)
    return text.strip()


def _make_search_query(
    text: str,
    *,
    global_context: str = "",
    neighborhood: str = "",
    max_terms: int = 5,
) -> str:
    local_keywords = _useful_keywords(text, max_terms=max_terms)
    visual_hints = _scene_visual_hints(neighborhood or text)

    pieces: list[str] = []
    if global_context:
        pieces.append(global_context)
    if visual_hints:
        pieces.extend(visual_hints[:2])
    if local_keywords:
        pieces.append(" ".join(local_keywords))

    query = " ".join(pieces).strip()
    return query or "people documentary photo"


def _useful_keywords(text: str, *, max_terms: int) -> list[str]:
    useful: list[str] = []
    seen: set[str] = set()
    for raw in text.split():
        token = _STRIP.sub("", raw).strip("_-").lower()
        if len(token) < 2 or token in _STOPWORDS or token in seen:
            continue
        seen.add(token)
        useful.append(token)
        if len(useful) >= max_terms:
            break
    return useful


def _scene_visual_hints(text: str) -> list[str]:
    lowered = text.lower()
    return [phrase for stems, phrase in _SCENE_RULES if any(stem in lowered for stem in stems)]


def _global_visual_context(text: str) -> str:
    lowered = text.lower()
    parts: list[str] = []

    for stems, phrase in _GLOBAL_RULES:
        if any(stem in lowered for stem in stems):
            parts.append(phrase)

    era = _detect_era(lowered)
    if era:
        parts.append(era)

    historical = bool(era) or any(stem in lowered for stem in _HISTORY_STEMS)
    if historical:
        parts.append("historical archival illustration")

    joined = " ".join(parts).lower()
    if "china" in joined and "19th century" in joined:
        parts.append("Qing dynasty")
    if "taiping rebellion" in joined:
        parts.append("1850s China")

    return " ".join(dict.fromkeys(parts))


def _detect_era(text: str) -> str:
    if re.search(r"\b18\d{2}\b|\b19\s*(?:-|‑)?\s*(?:й|ый)?\s*век|\bxix\b", text):
        return "19th century"
    if re.search(r"\b17\d{2}\b|\b18\s*(?:-|‑)?\s*(?:й|ый)?\s*век|\bxviii\b", text):
        return "18th century"
    if re.search(r"\b19\d{2}\b|\b20\s*(?:-|‑)?\s*(?:й|ый)?\s*век|\bxx\b", text):
        return "20th century"
    if re.search(r"\b20\d{2}\b|\b21\s*(?:-|‑)?\s*(?:й|ый)?\s*век|\bxxi\b", text):
        return "21st century modern"
    return ""
