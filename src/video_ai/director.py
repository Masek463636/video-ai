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
    (("иисус", "христ", "религи", "бог"), "Christian religious imagery"),
    (("арм", "солдат", "войск", "крестьян"), "historical soldiers peasants army marching"),
    (("восстан", "бунт", "битв", "револю"), "historical rebellion battle crowd"),
    (("деньг", "цена", "миллион", "тысяч"), "money finance counting cash"),
    (("телефон", "сообщен", "звон"), "person using smartphone"),
    (("машин", "авто"), "car driving road"),
    (("компьют", "ютуб", "видео"), "video creator computer camera"),
)

_HISTORY_STEMS = ("истор", "восстан", "импер", "династ", "войн", "арм", "солдат", "битв", "револю", "xix", "xviii", "тайпин", "сюцюан", "крестьян")
_VIDEO_STEMS = ("идет", "идут", "беж", "летит", "едет", "горит", "взрыв", "толп", "марш", "атак", "движ", "сраж")
_MEME_STEMS = ("вдруг", "и тут", "прикол", "шок", "жесть", "серьезно", "неожидан", "пиздец", "лол", "смеш", "офиг", "охрен", "что за")
_DRAMATIC_STEMS = ("вдруг", "шок", "галлюцин", "видение", "убит", "погиб", "битв", "войн", "восстан", "огонь", "взрыв")
_PULLBACK_STEMS = ("итог", "в итоге", "миллион", "погиб", "закончил", "после", "последств")

_QUERY_TAGS: dict[str, tuple[str, ...]] = {
    "china": ("china", "chinese", "qing", "taiping"),
    "religion": ("jesus", "christ", "christian", "biblical", "religious", "church", "sacred"),
    "army": ("army", "soldier", "soldiers", "military", "rebel", "rebellion", "peasants", "battle", "troops"),
    "emotion": ("stress", "stressed", "anxious", "sad", "upset", "disappointed", "panic", "crying"),
    "vision": ("vision", "dream", "surreal", "hallucination", "revelation", "mystical"),
    "exam": ("exam", "examination", "student", "civil service"),
    "war": ("war", "wwi", "world war", "trench", "battlefield"),
    "money": ("money", "cash", "finance", "currency"),
}


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
        source_mode = _choose_source_mode(caption, global_context, mode)
        motion_preset = _choose_motion_preset(caption, index=index, mode=mode)
        description = _visual_description(caption, global_context, mode)
        queries = _search_queries(caption, global_context, neighborhood, mode, description)
        lock, entities, context, fallback = _rule_semantic_lock(caption, global_context)
        if lock:
            mode = "image"
            source_mode = "historical_archive"
        scenes.append(Scene(
            start=round(words[0].start, 3),
            end=round(words[-1].end, 3),
            query=queries[0],
            search_queries=queries,
            visual_description=description,
            asset_kind="blank",
            motion="none",
            caption=caption,
            caption_words=list(words),
            visual_mode=mode,
            source_mode=source_mode,
            motion_preset=motion_preset,
            semantic_lock=lock,
            required_entities=entities,
            required_context=context,
            semantic_fallback=fallback,
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

        deterministic_lock = bool(scene.semantic_lock)
        deterministic_entities = list(scene.required_entities)
        deterministic_context = list(scene.required_context)
        deterministic_fallback = scene.semantic_fallback
        local_historical = _locally_historical(scene.caption or "")
        rule_queries = list(scene.search_queries)

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
            if not deterministic_lock and not local_historical:
                description = _strip_historical_constraints(description)
            scene.visual_description = description[:500]

        queries = item.get("search_queries") or []
        cleaned: list[str] = []
        seen: set[str] = set()
        for value in queries:
            q = re.sub(r"\s+", " ", str(value)).strip()
            if not deterministic_lock and not local_historical:
                q = _strip_historical_constraints(q)
            if not q:
                continue
            if not deterministic_lock and rule_queries and not _query_matches_intent(q, rule_queries):
                continue
            if q.lower() not in seen:
                seen.add(q.lower())
                cleaned.append(q[:180])
            if len(cleaned) >= 4:
                break

        # Gemini may enrich search intent, but it may no longer replace the
        # deterministic current-beat intent wholesale. Rule queries stay first.
        merged_queries = _merge_queries(rule_queries[:2], cleaned, rule_queries[2:])
        if merged_queries:
            scene.search_queries = merged_queries[:5]
            scene.query = scene.search_queries[0]

        if deterministic_lock:
            scene.semantic_lock = True
            scene.required_entities = deterministic_entities
            scene.required_context = deterministic_context
            scene.semantic_fallback = deterministic_fallback or scene.visual_description
            scene.visual_mode = "image"
            scene.source_mode = "historical_archive"
            scene.meme_filename = None
        else:
            scene.semantic_lock = False
            scene.required_entities = []
            scene.required_context = []
            scene.semantic_fallback = None
            if scene.source_mode == "historical_archive" and not local_historical:
                scene.source_mode = "stock_video" if scene.visual_mode == "video" else "generic_image"

        if _explicit_christian_reference(scene.caption or "") and not scene.semantic_lock:
            scene.visual_mode = "image"
            scene.source_mode = "generic_image"
            scene.visual_description = "Christian religious imagery associated with Jesus Christ, spiritual revelation or biblical symbolism; no unrelated religion"
            scene.search_queries = _merge_queries(
                [
                    "Jesus Christ painting",
                    "Christian religious painting",
                    "biblical vision Jesus",
                    "Jesus Christ religious art",
                ],
                scene.search_queries,
            )[:5]
            scene.query = scene.search_queries[0]

        if _explicit_army_action(scene.caption or "") and not scene.semantic_lock:
            scene.source_mode = "historical_archive"
            scene.visual_mode = "image"
            scene.visual_description = "historical peasant rebel army or soldiers marching; documentary archival illustration"
            if _taiping_story(full_text):
                army_queries = [
                    "Taiping Rebellion soldiers",
                    "19th century Chinese rebel army",
                    "Chinese peasant rebel army engraving",
                    "Qing dynasty soldiers historical illustration",
                ]
            else:
                army_queries = [
                    "historical rebel army engraving",
                    "peasant soldiers historical illustration",
                    "historical troops marching engraving",
                ]
            scene.search_queries = _merge_queries(army_queries, scene.search_queries)[:5]
            scene.query = scene.search_queries[0]

        if scene.visual_mode == "meme" and not scene.semantic_lock:
            scene.source_mode = "meme_library"
            chosen_name = str(item.get("meme_filename", "")).strip()
            if chosen_name and chosen_name in meme_names:
                scene.meme_filename = chosen_name
            tags = item.get("meme_tags") or []
            meme_query = " ".join(str(tag).strip() for tag in tags if str(tag).strip())
            if meme_query:
                scene.visual_description = f"{scene.visual_description or ''} {meme_query} reaction meme".strip()

        if scene.source_mode == "historical_archive" and scene.visual_mode == "video":
            scene.visual_mode = "image"

        applied += 1

    if applied == 0:
        return "rules", None
    return "gemini", client.last_model


def _explicit_entities(text: str) -> list[str]:
    """Entities safe to turn into HARD factual visual locks.

    Prefer one primary concrete subject. A beat saying "Hong Xiuquan failed the
    imperial examination" should hard-lock Hong Xiuquan and use the examination
    as action/search context, rather than requiring one asset to depict both.
    """
    lowered = text.lower()
    entities: list[str] = []
    has_hong = "сюцюан" in lowered or "хун сю" in lowered or "hong xiu" in lowered
    if has_hong:
        entities.append("Hong Xiuquan")
    if "тайпин" in lowered:
        entities.append("Taiping Rebellion")
    if not has_hong and "экзам" in lowered and ("импер" in lowered or "чиновник" in lowered or "гос" in lowered):
        entities.append("Imperial examination")
    if "первая миров" in lowered or "первой миров" in lowered or "world war i" in lowered or "first world war" in lowered:
        entities.append("World War I")
    return list(dict.fromkeys(entities))


def _rule_semantic_lock(text: str, global_context: str) -> tuple[bool, list[str], list[str], str | None]:
    """Hard-lock only facts explicitly named in THIS beat."""
    entities = _explicit_entities(text)
    if not entities:
        return False, [], [], None

    lowered = text.lower()
    context: list[str] = []
    if any(token in lowered for token in ("китай", "китайск", "china", "chinese")):
        context.append("China")
    if any(token in lowered for token in ("династия цин", "династии цин", "qing dynasty")):
        context.append("Qing dynasty")
    era = _detect_era(lowered)
    if era and era != "21st century modern":
        context.append(era)

    fallback = " ".join([*entities, *context, "archival illustration"])
    return True, entities, list(dict.fromkeys(context)), fallback


def _clean_string_list(value: object, *, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = re.sub(r"\s+", " ", str(item)).strip()
        if text and text.lower() not in {x.lower() for x in out}:
            out.append(text[:120])
        if len(out) >= limit:
            break
    return out


def _setting_context(global_context: str) -> str:
    """Broad story setting used for SEARCH only, never as a hard requirement."""
    value = global_context
    for phrase in ("Hong Xiuquan", "Taiping Rebellion"):
        value = re.sub(re.escape(phrase), " ", value, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", value).strip()


def _visual_description(text: str, global_context: str, mode: VisualMode) -> str:
    hints = _scene_visual_hints(text)
    local = hints[0] if hints else "documentary scene related to narration"
    explicit = " ".join(_explicit_entities(text))
    style = "moving documentary B-roll" if mode == "video" else "reaction meme insert" if mode == "meme" else "documentary visual"
    return " ".join(x for x in (explicit, local, style) if x).strip()


def _search_queries(text: str, global_context: str, neighborhood: str, mode: VisualMode, description: str) -> list[str]:
    hints = _scene_visual_hints(text)
    setting = _setting_context(global_context)
    explicit = " ".join(_explicit_entities(text))
    local_keywords = " ".join(_useful_keywords(text, max_terms=5))
    neighborhood_keywords = " ".join(_useful_keywords(neighborhood, max_terms=3))
    suffix = "video b roll" if mode == "video" else "reaction meme" if mode == "meme" else "photo illustration"
    candidates = [
        f"{explicit} {setting} {hints[0] if hints else local_keywords} {suffix}",
        f"{setting} {description}",
        f"{setting} {local_keywords} {suffix}",
        f"{local_keywords or neighborhood_keywords} {suffix}",
    ]
    out: list[str] = []
    seen: set[str] = set()
    for query in candidates:
        query = re.sub(r"\s+", " ", query).strip()
        if query and query.lower() not in seen:
            out.append(query)
            seen.add(query.lower())
    return out or ["documentary people video b roll" if mode == "video" else "documentary photo"]


def _query_matches_intent(query: str, baseline_queries: list[str]) -> bool:
    """Reject obvious Gemini query drift without blocking reasonable synonyms.

    We compare both literal English tokens and coarse semantic tags. If the
    deterministic query contains a strong intent tag (China, religion, army,
    emotion, vision, exam, war...), a Gemini query must share at least one tag
    or a meaningful literal token. Generic beats without a known tag stay open.
    """
    baseline = " ".join(baseline_queries[:3]).lower()
    q = query.lower()
    baseline_tags = _semantic_query_tags(baseline)
    query_tags = _semantic_query_tags(q)
    if baseline_tags and baseline_tags & query_tags:
        return True

    baseline_tokens = _english_tokens(baseline)
    query_tokens = _english_tokens(q)
    overlap = baseline_tokens & query_tokens
    if len(overlap) >= 2:
        return True
    if len(overlap) == 1 and len(query_tokens) <= 7:
        return True

    # If we have no meaningful English intent signal, do not over-filter.
    return not baseline_tags


def _semantic_query_tags(text: str) -> set[str]:
    lowered = text.lower()
    return {
        tag
        for tag, needles in _QUERY_TAGS.items()
        if any(needle in lowered for needle in needles)
    }


def _english_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z][a-z-]{2,}", text.lower())
        if token not in {"the", "and", "with", "from", "into", "photo", "image", "video", "documentary", "historical", "illustration", "footage", "scene", "visual", "broll", "roll"}
    }


def _merge_queries(*groups: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for value in group:
            q = re.sub(r"\s+", " ", str(value)).strip()
            if q and q.lower() not in seen:
                seen.add(q.lower())
                out.append(q)
    return out


def _taiping_story(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in ("тайпин", "сюцюан", "taiping", "hong xiu"))


def _strip_historical_constraints(value: str) -> str:
    cleaned = value
    patterns = (
        r"\b19th[- ]century\b", r"\b1800s\b", r"\bQing dynasty\b",
        r"\bTaiping Rebellion era\b", r"\bhistorical setting\b",
        r"\bperiod[- ]appropriate\b", r"\btraditional Chinese (?:clothing|costume|setting)\b",
        r"\bin Qing China\b", r"\bQing China\b",
    )
    for pattern in patterns:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", cleaned).strip(" ,.;:-")


def _explicit_christian_reference(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in ("иисус", "христ", "jesus", "christ"))


def _explicit_army_action(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in ("арм", "войск", "солдат", "крестьян"))


def _choose_visual_mode(text: str, *, index: int, duration: float) -> VisualMode:
    lowered = text.lower()
    if any(stem in lowered for stem in _MEME_STEMS):
        return "meme"
    if any(stem in lowered for stem in _VIDEO_STEMS):
        return "video"
    if index > 0 and index % 4 == 3 and duration >= 1.4:
        return "video"
    return "image"


def _locally_historical(text: str) -> bool:
    lowered = text.lower()
    return any(stem in lowered for stem in _HISTORY_STEMS) or bool(_explicit_entities(text))


def _choose_source_mode(text: str, global_context: str, mode: VisualMode) -> SourceMode:
    lowered = text.lower()
    if mode == "meme":
        return "meme_library"
    historical_story = "historical archival" in global_context.lower()
    locally_historical = any(stem in lowered for stem in _HISTORY_STEMS)
    if historical_story and locally_historical:
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
    if re.search(r"\b18\d{2}\b|\b19\s*(?:-|‑)?\s*(?:й|ый)?\s*век|\bxix\b", text):
        return "19th century"
    if re.search(r"\b17\d{2}\b|\b18\s*(?:-|‑)?\s*(?:й|ый)?\s*век|\bxviii\b", text):
        return "18th century"
    if re.search(r"\b19\d{2}\b|\b20\s*(?:-|‑)?\s*(?:й|ый)?\s*век|\bxx\b", text):
        return "20th century"
    if re.search(r"\b20\d{2}\b|\b21\s*(?:-|‑)?\s*(?:й|ый)?\s*век|\bxxi\b", text):
        return "21st century modern"
    return ""


def _meme_names(directory: str | Path | None) -> list[str]:
    if not directory:
        return []
    root = Path(directory)
    if not root.exists() or not root.is_dir():
        return []
    allowed = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".jpg", ".jpeg", ".png", ".webp"}
    return sorted(path.name for path in root.rglob("*") if path.is_file() and path.suffix.lower() in allowed)[:250]
