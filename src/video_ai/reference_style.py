from __future__ import annotations

import re

from .models import Scene, ShotPlan


_EXAM = ("экзам", "госэкзам", "test", "exam")
_FAIL = ("провал", "не сдал", "стресс", "нерв", "разочар", "failed", "failure", "stress")
_SLEEP = ("спал", "сон", "проснул", "просып", "sleep", "dream", "woke", "wake")
_RELIGION = ("иисус", "христ", "бог", "религи", "видение", "jesus", "christ", "relig", "vision")
_ARMY = ("арм", "войск", "солдат", "крестьян", "повстан", "восстан", "army", "troops", "soldier", "rebel")
_CHINA = ("китай", "китайск", "china", "chinese")
_DEATH = ("погиб", "жертв", "смерт", "миллион", "унесл", "death", "killed", "casualt", "million")
_ACTION = ("идет", "идут", "беж", "едет", "говор", "пиш", "крич", "смотр", "берет", "doing", "walking", "running")


def apply_reference_style(plan: ShotPlan) -> list[int]:
    """Make the shot plan behave like a modern viral B-roll Short.

    Reference style is video-first: unlocked narrative beats become short real
    footage cutaways. Exact factual/historical locks remain archive images.
    Motion on footage is disabled; the source motion should carry the shot.
    """
    changed: list[int] = []
    for index, scene in enumerate(plan.scenes):
        if scene.semantic_lock:
            scene.visual_mode = "image"
            scene.source_mode = "historical_archive"
            scene.motion_preset = "micro_push"
            continue

        if scene.visual_mode == "meme" and scene.source_mode == "meme_library":
            scene.motion_preset = "none"
            continue

        scene.visual_mode = "video"
        scene.source_mode = "stock_video"
        scene.motion_preset = "none"
        scene.motion = "none"

        queries = _action_queries(scene)
        scene.search_queries = _merge_queries(queries, scene.search_queries or [scene.query])[:7]
        if scene.search_queries:
            scene.query = scene.search_queries[0]

        base = (scene.visual_description or scene.caption or scene.query or "real human action").rstrip(" .")
        scene.visual_description = (
            f"{base}. Prefer real moving footage with a clear human/animal/action subject, "
            "natural motion, no text, no static poster, no map unless explicitly required."
        )
        changed.append(index)
    return changed


def _action_queries(scene: Scene) -> list[str]:
    text = " ".join((scene.caption or "", scene.visual_description or "", scene.query or "")).lower()

    if _contains(text, _EXAM):
        return [
            "student taking exam writing paper",
            "man stressed during exam",
            "student writing test close up",
        ]
    if _contains(text, _FAIL):
        return [
            "frustrated man disappointed reaction",
            "stressed man sitting alone",
            "anxious man emotional reaction",
        ]
    if _contains(text, _SLEEP):
        if any(x in text for x in ("проснул", "просып", "woke", "wake")):
            return [
                "man waking up suddenly in bed",
                "person waking from dream surprised",
                "sleeping man wakes up reaction",
            ]
        return [
            "man sleeping in bed",
            "person having vivid dream sleeping",
            "sleeping person close up",
        ]
    if _contains(text, _RELIGION):
        return [
            "person experiencing spiritual revelation bright light",
            "man praying dramatic church light",
            "mystical religious vision person",
        ]
    if _contains(text, _ARMY):
        return [
            "soldiers marching crowd cinematic footage",
            "army marching people action",
            "historical reenactment soldiers marching",
        ]
    if _contains(text, _DEATH):
        return [
            "war aftermath destroyed buildings people",
            "somber ruins aftermath conflict",
            "mourning people tragedy documentary",
        ]
    if _contains(text, _CHINA):
        return [
            "China people street documentary footage",
            "Chinese crowd walking cinematic footage",
            "China city people b roll",
        ]
    if _contains(text, _ACTION):
        return [
            _clean(scene.visual_description or scene.query) + " real footage",
            _clean(scene.visual_description or scene.query) + " action b roll",
        ]

    subject = _clean(scene.visual_description or scene.caption or scene.query)
    return [
        f"{subject} real footage",
        f"{subject} vertical b roll",
        f"{subject} people action",
    ]


def _contains(value: str, needles: tuple[str, ...]) -> bool:
    return any(needle in value for needle in needles)


def _clean(value: str) -> str:
    value = re.sub(
        r"\b(painting|illustration|engraving|portrait|historical archive|archival|photo illustration)\b",
        " ",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"\s+", " ", value).strip(" ,.;:-")
    return value or "real human action"


def _merge_queries(*groups: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for raw in group:
            q = re.sub(r"\s+", " ", str(raw or "")).strip()
            if q and q.casefold() not in seen:
                seen.add(q.casefold())
                out.append(q)
    return out
