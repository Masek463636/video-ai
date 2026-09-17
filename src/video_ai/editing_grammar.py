from __future__ import annotations

import re
from pathlib import Path

from .models import Scene, ShotPlan


_ACTION_ARMY = ("арм", "войск", "солдат", "крестьян", "повстан", "восстан", "army", "troops", "soldier", "rebel", "rebellion", "peasant")
_REACTION = ("стресс", "паник", "провал", "разочар", "груст", "нерв", "stress", "panic", "failed", "failure", "disappoint", "anxious")
_AWAKEN = ("проснул", "просып", "разбуд", "очнул", "woke", "wake up", "awoke")
_DREAM = ("сон", "снилось", "видение", "галлюцин", "dream", "vision", "hallucin")
_RELIGIOUS = ("иисус", "христ", "бог", "jesus", "christ", "god")
_BELIEF = ("брат", "уверен", "считал", "поверил", "решил", "brother", "believed", "convinced", "certain")
_CASUALTY = ("погиб", "жертв", "смерт", "миллион", "унесл", "casualt", "killed", "dead", "death", "million lives")
_BATTLE = ("битв", "атак", "сраж", "бой", "взрыв", "battle", "attack", "fight", "explosion")


def apply_pre_asset_grammar(plan: ShotPlan) -> list[int]:
    """Turn literal scene nouns into edit-friendly visual beats.

    Gemini remains the story director, but this deterministic layer prevents a
    single noun (Jesus, a named person, etc.) from dominating several adjacent
    shots. It rewrites only unlocked beats; exact factual locks remain intact.
    """
    story = " ".join((scene.caption or "") for scene in plan.scenes).lower()
    taiping_story = any(token in story for token in ("тайпин", "сюцюан", "taiping", "hong xiu"))
    rewritten: list[int] = []

    for index, scene in enumerate(plan.scenes):
        if scene.semantic_lock:
            continue
        role = _beat_role(scene.caption or "")
        preferred: list[str] = []
        description: str | None = None

        if role == "reaction":
            preferred = [
                "stressed disappointed man reaction",
                "anxious frustrated man close up",
                "person reacting to failure documentary b roll",
            ]
            description = "a human emotional reaction to failure or stress; clear face, no text"
            scene.visual_mode = "video"
            scene.source_mode = "stock_video"
        elif role == "awakening":
            preferred = [
                "man waking suddenly from sleep",
                "person waking from vivid dream",
                "surprised man sitting up in bed",
            ]
            description = "a person waking abruptly after an intense dream; human reaction first"
            scene.visual_mode = "video"
            scene.source_mode = "stock_video"
        elif role == "revelation":
            preferred = [
                "surreal religious vision heavenly light person",
                "mystical spiritual revelation human figure",
                "dreamlike biblical vision person seeing light",
            ]
            description = "a human experiencing a surreal religious revelation; dreamlike sacred imagery, not a static portrait"
            scene.visual_mode = "image"
            scene.source_mode = "generic_image"
        elif role == "vision":
            preferred = [
                "surreal dream vision person",
                "mystical hallucination human silhouette",
                "dreamlike revelation cinematic image",
            ]
            description = "a surreal dream or vision experienced by a person; atmospheric and readable in vertical crop"
            scene.visual_mode = "image"
            scene.source_mode = "generic_image"
        elif role == "army":
            preferred = (
                [
                    "Taiping Rebellion soldiers",
                    "19th century Chinese rebel army",
                    "Chinese peasant rebel army engraving",
                    "Qing dynasty soldiers historical illustration",
                ]
                if taiping_story
                else [
                    "historical rebel army engraving",
                    "peasant soldiers historical illustration",
                    "historical troops marching engraving",
                ]
            )
            description = "historical rebel army or soldiers in action; people and movement, not religious iconography"
            scene.visual_mode = "image"
            scene.source_mode = "historical_archive"
        elif role == "aftermath":
            preferred = (
                [
                    "Taiping Rebellion destruction aftermath",
                    "19th century China war devastation",
                    "Taiping Rebellion ruins historical illustration",
                    "war casualties mourning historical China",
                ]
                if taiping_story
                else [
                    "war devastation aftermath historical",
                    "battlefield aftermath mourning",
                    "ruined city historical conflict",
                ]
            )
            description = "somber aftermath of conflict, destruction or mourning; historically compatible and never cheerful"
            scene.visual_mode = "image"
            if taiping_story:
                scene.source_mode = "historical_archive"
        elif role == "battle":
            preferred = (
                ["Taiping Rebellion battle", "19th century Chinese battle engraving", "Qing dynasty battle illustration"]
                if taiping_story
                else ["historical battle engraving", "soldiers fighting historical illustration"]
            )
            description = "historical battle or violent action with clear human activity"
            scene.visual_mode = "image"
            scene.source_mode = "historical_archive"

        if not preferred:
            continue

        old_queries = list(scene.search_queries or [])
        scene.search_queries = _merge_queries(preferred, old_queries)[:5]
        scene.query = scene.search_queries[0]
        scene.visual_description = description
        rewritten.append(index)

    # If two adjacent unlocked beats still request almost the same thing, force
    # the later one toward a detail/reaction angle instead of another portrait.
    for index in range(1, len(plan.scenes)):
        current = plan.scenes[index]
        previous = plan.scenes[index - 1]
        if current.semantic_lock or previous.semantic_lock:
            continue
        if _query_signature(current) and _query_signature(current) == _query_signature(previous):
            detail = _variation_query(current.caption or "")
            if detail:
                current.search_queries = _merge_queries([detail], current.search_queries)[:5]
                current.query = current.search_queries[0]
                if index not in rewritten:
                    rewritten.append(index)

    return sorted(set(rewritten))


def diversity_repair_indexes(
    plan: ShotPlan,
    *,
    max_consecutive_seconds: float = 2.6,
    recent_window: int = 3,
) -> list[int]:
    """Return scenes whose chosen asset makes the edit feel repetitive.

    We keep the first appearance and ask the selector to replace only later
    repetitions. This is deliberately conservative: exact hard-locked facts may
    stay a little longer when no honest alternative exists.
    """
    repairs: set[int] = set()
    run_key: str | None = None
    run_seconds = 0.0

    for index, scene in enumerate(plan.scenes):
        key = _asset_key(scene)
        duration = _display_duration(plan, index)
        if key and key == run_key:
            run_seconds += duration
            threshold = max_consecutive_seconds * (1.35 if scene.semantic_lock else 1.0)
            if run_seconds > threshold:
                repairs.add(index)
        else:
            run_key = key
            run_seconds = duration

        if not key or scene.semantic_lock:
            continue
        for previous_index in range(max(0, index - recent_window), index):
            previous = plan.scenes[previous_index]
            if _asset_key(previous) == key:
                # Consecutive reuse is already governed by duration above. For
                # a returning shot after another visual, request a fresh option.
                if previous_index != index - 1:
                    repairs.add(index)
                break

    return sorted(repairs)


def _beat_role(text: str) -> str:
    value = text.lower()
    if _contains(value, _CASUALTY):
        return "aftermath"
    if _contains(value, _BATTLE):
        return "battle"
    if _contains(value, _ACTION_ARMY):
        return "army"
    if _contains(value, _AWAKEN):
        return "awakening"
    if _contains(value, _RELIGIOUS) and _contains(value, _BELIEF):
        return "revelation"
    if _contains(value, _DREAM) and _contains(value, _RELIGIOUS):
        return "revelation"
    if _contains(value, _DREAM):
        return "vision"
    if _contains(value, _REACTION):
        return "reaction"
    return ""


def _contains(value: str, needles: tuple[str, ...]) -> bool:
    return any(needle in value for needle in needles)


def _merge_queries(*groups: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for value in group:
            query = re.sub(r"\s+", " ", str(value)).strip()
            if query and query.lower() not in seen:
                seen.add(query.lower())
                out.append(query)
    return out


def _query_signature(scene: Scene) -> str:
    value = (scene.query or scene.visual_description or "").lower()
    tokens = [
        token for token in re.findall(r"[a-zа-яё][a-zа-яё-]{2,}", value)
        if token not in {"photo", "image", "video", "historical", "documentary", "illustration", "scene", "visual", "the", "and", "with"}
    ]
    return " ".join(tokens[:4])


def _variation_query(text: str) -> str:
    role = _beat_role(text)
    if role == "reaction":
        return "close emotional reaction face disappointment"
    if role == "awakening":
        return "close up person waking from dream"
    if role in {"vision", "revelation"}:
        return "human silhouette experiencing mystical vision"
    if role == "army":
        return "historical soldiers marching close detail"
    if role == "aftermath":
        return "close detail destruction mourning aftermath"
    if role == "battle":
        return "historical battle action close detail"
    return ""


def _asset_key(scene: Scene) -> str | None:
    if not scene.asset or scene.asset_kind == "blank":
        return None
    try:
        return str(Path(scene.asset).resolve()).casefold()
    except OSError:
        return str(scene.asset).casefold()


def _display_duration(plan: ShotPlan, index: int) -> float:
    scene = plan.scenes[index]
    start = 0.0 if index == 0 else max(0.0, scene.start)
    if index + 1 < len(plan.scenes):
        end = max(start + 0.05, plan.scenes[index + 1].start)
    else:
        end = max(start + 0.05, scene.end)
    return end - start
