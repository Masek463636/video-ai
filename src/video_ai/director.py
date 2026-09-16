from __future__ import annotations

import re
from pathlib import Path

from .models import MotionPreset, Scene, ShotPlan, SourceMode, Transcript, VisualMode, Word

_SENTENCE_END = re.compile(r"[.!?…]+$")
_STRIP = re.compile(r"[^\w\-]+", flags=re.UNICODE)

_STOPWORDS = {
    "а", "и", "но", "или", "в", "во", "на", "по", "к", "ко", "у", "из", "за",
    "с", "со", "от", "до", "для", "что", "это", "как", "же", "бы", "не", "ну",
    "я", "ты", "он", "она", "они", "мы", "вы", "мой", "моя", "его", "ее", "её",
    "the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "for", "with",
    "this", "that", "it", "is", "are", "was", "were", "i", "you", "he", "she", "we",
}

_GLOBAL_RULES = (
    (("китай", "китайск"), "China Chinese"), (("тайпин",), "Taiping Rebellion"),
    (("сюцюан", "хун сю", "hong xiu"), "Hong Xiuquan"), (("ссср", "советск"), "Soviet Union Soviet"),
    (("украин",), "Ukraine Ukrainian"), (("росси", "русск"), "Russia Russian"),
    (("америк", "сша"), "United States American"), (("герман", "немец"), "Germany German"),
    (("япон",), "Japan Japanese"), (("франц",), "France French"), (("британ", "англи"), "Britain British"),
)

_SCENE_RULES = (
    (("экзам", "госэкзам", "тест"), "civil service examination student"),
    (("стресс", "нерв", "расстро", "груст", "провал"), "stressed disappointed person"),
    (("чиновник", "служб"), "government official civil service"),
    (("сон", "спал", "снилось"), "sleeping person dream"),
    (("галлюцин", "видение", "озарен"), "surreal vision revelation"),
    (("иисус", "христ", "религи", "бог"), "Jesus Christian religious painting"),
    (("арм", "солдат", "войск"), "historical soldiers army marching"),
    (("восстан", "бунт", "битв", "револю"), "historical rebellion battle crowd"),
    (("деньг", "цена", "миллион", "тысяч"), "money finance counting cash"),
    (("телефон", "сообщен", "звон"), "person using smartphone"),
    (("машин", "авто"), "car driving road"),
    (("компьют", "ютуб", "видео"), "video creator computer camera"),
)

_HISTORY_STEMS = ("истор", "восстан", "импер", "династ", "войн", "арм", "солдат", "битв", "револю", "xix", "xviii", "тайпин", "сюцюан")
_VIDEO_STEMS = ("идет", "идут", "беж", "летит", "едет", "горит", "взрыв", "толп", "марш", "атак", "движ", "сраж")
_MEME_STEMS = ("вдруг", "и тут", "прикол", "шок", "жесть", "серьезно", "неожидан", "пиздец", "лол", "смеш", "офиг", "охрен", "что за")
_DRAMATIC_STEMS = ("вдруг", "шок", "галлюцин", "видение", "убит", "погиб", "битв", "войн", "восстан", "огонь", "взрыв")
_PULLBACK_STEMS = ("итог", "в итоге", "миллион", "погиб", "закончил", "после", "последств")


def build_shot_plan(
    transcript: Transcript,
    audio: str | Path,
    *,
    target_scene_seconds: float = 1.7,
    min_scene_seconds: float = 0.95,
    max_scene_seconds: float = 2.7,
    meme_dir: str | Path | None = None,
) -> ShotPlan:
    if min_scene_seconds <= 0 or target_scene_seconds < min_scene_seconds:
        raise ValueError("invalid scene duration settings")
    if max_scene_seconds < target_scene_seconds:
        raise ValueError("max_scene_seconds must be >= target_scene_seconds")

    chunks = _chunk_words(transcript.words, target=target_scene_seconds, minimum=min_scene_seconds, maximum=max_scene_seconds)
    full_text = _join_words(transcript.words)
    global_context = _global_visual_context(full_text)
    captions = [_join_words(words) for words in chunks]
    scenes: list[Scene] = []

    for index, (words, caption) in enumerate(zip(chunks, captions)):
        previous_caption = captions[index - 1] if index > 0 else ""
        next_caption = captions[index + 1] if index + 1 < len(captions) else ""
        neighborhood = " ".join(x for x in (previous_caption, caption, next_caption) if x)
        duration = words[-1].end - words[0].start
        mode = _choose_visual_mode(caption, index=index, duration=duration)
        source_mode = _choose_source_mode(caption, neighborhood, global_context, mode)
        motion_preset = _choose_motion_preset(caption, index=index, mode=mode)
        description = _visual_description(caption, global_context, neighborhood, mode)
        queries = _search_queries(caption, global_context, neighborhood, mode, description)
        scenes.append(Scene(
            start=round(words[0].start, 3),
            end=round(words[-1].end, 3),
            query=queries[0],
            search_queries=queries,
            visual_description=description,
            asset_kind="blank",
            motion="none",
            caption=caption,
            visual_mode=mode,
            source_mode=source_mode,
            motion_preset=motion_preset,
        ))

    meme_names = _meme_names(meme_dir)
    source, model = _apply_gemini_direction(full_text, scenes, meme_names=meme_names)
    return ShotPlan(audio=Path(audio), scenes=scenes, director_source=source, director_model=model)


def _apply_gemini_direction(full_text: str, scenes: list[Scene], *, meme_names: list[str]) -> tuple[str, str | None]:
    try:
        from .gemini_ai import get_gemini_client
        client = get_gemini_client()
        if client is None:
            return "rules", None
        directed = client.direct(full_text, scenes, meme_names=meme_names)
    except Exception:
        return "rules", None

    by_index = {int(item.get("index")): item for item in directed if str(item.get("index", "")).isdigit()}
    applied = 0
    for index, scene in enumerate(scenes):
        item = by_index.get(index)
        if not item:
            continue

        mode = str(item.get("visual_mode", scene.visual_mode)).lower().strip()
        if mode in {"image", "video", "meme", "auto"}:
            scene.visual_mode = mode  # type: ignore[assignment]

        source_mode = str(item.get("source_mode", scene.source_mode)).lower().strip()
        if source_mode in {"auto", "historical_archive", "stock_video", "meme_library", "generic_image"}:
            scene.source_mode = source_mode  # type: ignore[assignment]

        motion_preset = str(item.get("motion_preset", scene.motion_preset)).lower().strip()
        if motion_preset in {"none", "micro_push", "slow_push", "dramatic_push", "pull_back", "reveal_left", "reveal_right"}:
            scene.motion_preset = motion_preset  # type: ignore[assignment]

        description = str(item.get("visual_description", "")).strip()
        if description:
            scene.visual_description = description[:500]

        queries = item.get("search_queries") or []
        cleaned: list[str] = []
        seen: set[str] = set()
        for value in queries:
            q = re.sub(r"\s+", " ", str(value)).strip()
            if q and q.lower() not in seen:
                seen.add(q.lower())
                cleaned.append(q[:180])
            if len(cleaned) >= 5:
                break
        if cleaned:
            scene.search_queries = cleaned
            scene.query = cleaned[0]

        if scene.visual_mode == "meme":
            scene.source_mode = "meme_library"
            chosen_name = str(item.get("meme_filename", "")).strip()
            if chosen_name and chosen_name in meme_names:
                scene.meme_filename = chosen_name
            tags = item.get("meme_tags") or []
            meme_query = " ".join(str(tag).strip() for tag in tags if str(tag).strip())
            if meme_query:
                scene.visual_description = f"{scene.visual_description or ''} {meme_query} reaction meme".strip()

        # Safety rule: a specific historical/archive scene must not accidentally
        # become generic modern stock merely because Gemini selected video.
        if scene.source_mode == "historical_archive" and scene.visual_mode == "video":
            scene.visual_mode = "image"

        applied += 1

    if applied == 0:
        return "rules", None
    return "gemini", client.last_model


def _visual_description(text: str, global_context: str, neighborhood: str, mode: VisualMode) -> str:
    hints = _scene_visual_hints(neighborhood or text)
    local = hints[0] if hints else "documentary scene related to narration"
    style = "moving documentary B-roll" if mode == "video" else "reaction meme insert" if mode == "meme" else "documentary archival visual"
    return " ".join(x for x in (global_context, local, style) if x).strip()


def _search_queries(text: str, global_context: str, neighborhood: str, mode: VisualMode, description: str) -> list[str]:
    hints = _scene_visual_hints(neighborhood or text)
    local_keywords = " ".join(_useful_keywords(text, max_terms=4))
    suffix = "video b roll" if mode == "video" else "reaction meme" if mode == "meme" else "photo illustration"
    candidates = [
        f"{global_context} {hints[0] if hints else ''} {suffix}",
        description,
        f"{global_context} {local_keywords} {suffix}",
        f"{hints[0] if hints else local_keywords} {suffix}",
    ]
    out: list[str] = []
    seen: set[str] = set()
    for query in candidates:
        query = re.sub(r"\s+", " ", query).strip()
        if query and query.lower() not in seen:
            out.append(query)
            seen.add(query.lower())
    return out or ["documentary people video b roll" if mode == "video" else "documentary photo"]


def _choose_visual_mode(text: str, *, index: int, duration: float) -> VisualMode:
    lowered = text.lower()
    if any(stem in lowered for stem in _MEME_STEMS):
        return "meme"
    if any(stem in lowered for stem in _VIDEO_STEMS):
        return "video"
    if index > 0 and index % 4 == 3 and duration >= 1.4:
        return "video"
    return "image"


def _choose_source_mode(text: str, neighborhood: str, global_context: str, mode: VisualMode) -> SourceMode:
    lowered = f"{text} {neighborhood}".lower()
    if mode == "meme":
        return "meme_library"
    if global_context and ("historical archival" in global_context.lower() or any(stem in lowered for stem in _HISTORY_STEMS)):
        return "historical_archive"
    if mode == "video":
        return "stock_video"
    return "generic_image"


def _choose_motion_preset(text: str, *, index: int, mode: VisualMode) -> MotionPreset:
    lowered = text.lower()
    if mode == "meme":
        return "none"
    if any(stem in lowered for stem in _PULLBACK_STEMS):
        return "pull_back"
    if any(stem in lowered for stem in _DRAMATIC_STEMS):
        return "dramatic_push"
    if index % 5 == 2:
        return "reveal_right"
    if index % 5 == 4:
        return "reveal_left"
    return "slow_push" if mode == "image" else "micro_push"


def _chunk_words(words: list[Word], *, target: float, minimum: float, maximum: float) -> list[list[Word]]:
    chunks: list[list[Word]] = []
    current: list[Word] = []
    for word in words:
        current.append(word)
        duration = current[-1].end - current[0].start
        if duration >= maximum or (duration >= target and _SENTENCE_END.search(word.text)) or duration >= target + 0.32:
            if duration >= minimum:
                chunks.append(current)
                current = []
    if current:
        if chunks and current[-1].end - current[0].start < minimum * 0.68:
            chunks[-1].extend(current)
        else:
            chunks.append(current)
    return chunks


def _join_words(words: list[Word]) -> str:
    return re.sub(r"\s+([,.!?;:…])", r"\1", " ".join(word.text for word in words)).strip()


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
    if era or any(stem in lowered for stem in _HISTORY_STEMS):
        parts.append("historical archival")
    joined = " ".join(parts).lower()
    if "china" in joined and "19th century" in joined:
        parts.append("Qing dynasty")
    if "taiping rebellion" in joined:
        parts.append("1850s China")
    return " ".join(dict.fromkeys(parts))


def _detect_era(text: str) -> str:
    if re.search(r"\b18\d{2}\b|\b19\s*(?:-|‑)?\s*(?:й|ый)?\s*век|\bxix\b", text): return "19th century"
    if re.search(r"\b17\d{2}\b|\b18\s*(?:-|‑)?\s*(?:й|ый)?\s*век|\bxviii\b", text): return "18th century"
    if re.search(r"\b19\d{2}\b|\b20\s*(?:-|‑)?\s*(?:й|ый)?\s*век|\bxx\b", text): return "20th century"
    if re.search(r"\b20\d{2}\b|\b21\s*(?:-|‑)?\s*(?:й|ый)?\s*век|\bxxi\b", text): return "21st century modern"
    return ""


def _meme_names(directory: str | Path | None) -> list[str]:
    if not directory:
        return []
    root = Path(directory)
    if not root.exists() or not root.is_dir():
        return []
    allowed = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".jpg", ".jpeg", ".png", ".webp"}
    return sorted(path.name for path in root.rglob("*") if path.is_file() and path.suffix.lower() in allowed)[:250]
