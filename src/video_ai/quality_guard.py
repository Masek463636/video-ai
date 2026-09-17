from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .models import Tone


@dataclass(slots=True)
class LocalQualityResult:
    accept: bool
    score: int
    issues: list[str]


_TONE_RULES: list[tuple[Tone, tuple[str, ...]]] = [
    ("tragic", ("погиб", "погибл", "смерт", "умер", "жертв", "casualt", "death", "killed", "dead", "traged", "died", "lost their lives", "lost lives")),
    ("violent", ("битв", "войн", "убил", "атак", "взрыв", "резн", "battle", "war ", "attack", "explosion", "massacre")),
    ("shocking", ("шок", "вдруг", "неожидан", "оказалось", "ужас", "shock", "suddenly", "unbelievable")),
    ("tense", ("стресс", "паник", "опас", "угроз", "страх", "напряж", "stress", "panic", "danger", "fear")),
    ("mysterious", ("сон", "видение", "галлюцин", "тайн", "мист", "dream", "vision", "myster", "hallucin")),
    ("religious", ("иисус", "христ", "бог", "религи", "церк", "jesus", "christ", "god", "relig")),
    ("funny", ("смеш", "лол", "прикол", "мем", "ахах", "funny", "lol", "meme", "joke")),
    ("absurd", ("абсурд", "безум", "дико", "crazy", "absurd", "ridiculous")),
    ("victorious", ("побед", "триумф", "выиграл", "victor", "triumph", "won")),
    ("positive", ("счаст", "радост", "успех", "любов", "happy", "joy", "success", "love")),
    ("negative", ("провал", "проиграл", "разочар", "груст", "failed", "failure", "sad", "disappoint")),
    ("emotional", ("эмоц", "плак", "слез", "cry", "emotional", "tears")),
    ("informational", ("согласно", "статист", "факт", "данн", "according", "statistics", "fact", "data")),
]

_BAD_TITLE_TERMS = (
    "screenshot", "screen shot", "website", "webpage", "logo", "watermark", "poster", "banner",
    "sign", "signboard", "plaque", "label", "caption", "text only", "diagram", "infographic",
)


def infer_tone(text: str | None) -> Tone:
    value = (text or "").lower()

    # Mass-casualty wording often contains no literal word "death". Catch
    # constructions such as "унесло до 30 миллионов жизней" explicitly.
    russian_life_loss = any(x in value for x in ("жизн", "жертв", "погиб", "смерт"))
    large_count = any(x in value for x in ("миллион", "млн", "тысяч")) or bool(re.search(r"\b\d{3,}\b", value))
    loss_verb = any(x in value for x in ("унесл", "лишил", "потер", "погиб", "умер"))
    if russian_life_loss and (large_count or loss_verb):
        return "tragic"

    english_life_loss = any(x in value for x in ("lives", "casualties", "deaths", "died", "killed"))
    english_mass = any(x in value for x in ("million", "thousand", "mass", "hundreds", "tens of"))
    if english_life_loss and (english_mass or any(x in value for x in ("lost", "claimed", "killed"))):
        return "tragic"

    # v1.2.6: current ACTION beats theme. A sentence can mention Jesus/religion
    # while actually describing an army, rebellion or soldiers. In that case
    # the visual tone must support the action instead of steering CLIP toward
    # icons/paintings merely because a religious word is present.
    violent_action = any(x in value for x in (
        "битв", "сраж", "атак", "штурм", "убил", "взрыв", "резн",
        "battle", "fight", "attack", "assault", "explosion", "massacre",
    ))
    if violent_action:
        return "violent"

    military_action = any(x in value for x in (
        "арм", "войск", "солдат", "крестьянск", "повстан", "мятеж", "восстан",
        "army", "troops", "soldier", "soldiers", "military", "rebel", "rebellion", "peasant army",
    ))
    if military_action:
        return "tense"

    for tone, needles in _TONE_RULES:
        if any(needle in value for needle in needles):
            return tone
    return "neutral"


def local_quality_guard(
    path: str | Path,
    *,
    title: str = "",
    description: str = "",
    width: int = 0,
    height: int = 0,
    kind: str = "image",
) -> LocalQualityResult:
    """Cheap local pre-filter before spending a Gemini vision request."""
    issues: list[str] = []
    score = 100
    hay = f"{title} {description}".lower()

    for bad in _BAD_TITLE_TERMS:
        pattern = r"(?<!\w)" + re.escape(bad) + r"(?!\w)"
        if re.search(pattern, hay, flags=re.IGNORECASE):
            issues.append(f"metadata:{bad}")
            score -= 24
            break

    if kind == "image":
        if width and height:
            short = min(width, height)
            long = max(width, height)
            if long < 700:
                issues.append("low_resolution")
                score -= 35
            elif long < 1000:
                issues.append("medium_resolution")
                score -= 10
            if short < 320:
                issues.append("extreme_aspect_or_tiny_short_edge")
                score -= 18

        if (not width or not height) and Path(path).exists():
            probed = _probe_video_stream(path)
            pw, ph = int(probed.get("width") or 0), int(probed.get("height") or 0)
            if pw and ph and max(pw, ph) < 700:
                issues.append("low_resolution")
                score -= 35

    if kind == "video" and Path(path).exists():
        probed = _probe_video_stream(path)
        pw, ph = int(probed.get("width") or 0), int(probed.get("height") or 0)
        if pw and ph and max(pw, ph) < 720:
            issues.append("low_video_resolution")
            score -= 20
        duration = _probe_duration(path)
        if 0 < duration < 0.65:
            issues.append("too_short")
            score -= 25

    return LocalQualityResult(accept=score >= 55, score=max(0, score), issues=issues)


def _probe_video_stream(path: str | Path) -> dict:
    if shutil.which("ffprobe") is None:
        return {}
    try:
        completed = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "json", str(path)],
            check=True, capture_output=True, text=True, timeout=20,
        )
        streams = json.loads(completed.stdout).get("streams") or []
        return streams[0] if streams else {}
    except Exception:
        return {}


def _probe_duration(path: str | Path) -> float:
    if shutil.which("ffprobe") is None:
        return 0.0
    try:
        completed = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
            check=True, capture_output=True, text=True, timeout=20,
        )
        return float(json.loads(completed.stdout).get("format", {}).get("duration") or 0.0)
    except Exception:
        return 0.0
