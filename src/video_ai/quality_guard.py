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
    russian_life_loss = any(x in value for x in ("жизн", "жертв", "погиб", "смерт"))
    large_count = any(x in value for x in ("миллион", "млн", "тысяч")) or bool(re.search(r"\b\d{3,}\b", value))
    loss_verb = any(x in value for x in ("унесл", "лишил", "потер", "погиб", "умер"))
    if russian_life_loss and (large_count or loss_verb):
        return "tragic"

    english_life_loss = any(x in value for x in ("lives", "casualties", "deaths", "died", "killed"))
    english_mass = any(x in value for x in ("million", "thousand", "mass", "hundreds", "tens of"))
    if english_life_loss and (english_mass or any(x in value for x in ("lost", "claimed", "killed"))):
        return "tragic"

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
    path = Path(path)
    issues: list[str] = []
    score = 100
    hay = f"{title} {description}".lower()

    # Critical v1.3 guard: upstream proxies sometimes save HTML/error bytes
    # under a .png/.jpg-looking URL. Metadata width/height may still look valid,
    # so verify the actual payload before CLIP/Gemini/ffmpeg ever sees it.
    if kind == "image":
        if not path.exists() or path.stat().st_size < 512 or not _looks_like_supported_image(path):
            return LocalQualityResult(accept=False, score=0, issues=["invalid_image_payload"])

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

        if not width or not height:
            probed = _probe_video_stream(path)
            pw, ph = int(probed.get("width") or 0), int(probed.get("height") or 0)
            if not pw or not ph:
                return LocalQualityResult(accept=False, score=0, issues=["undecodable_image"])
            if max(pw, ph) < 700:
                issues.append("low_resolution")
                score -= 35

    if kind == "video" and path.exists():
        probed = _probe_video_stream(path)
        pw, ph = int(probed.get("width") or 0), int(probed.get("height") or 0)
        if not pw or not ph:
            return LocalQualityResult(accept=False, score=0, issues=["undecodable_video"])
        if max(pw, ph) < 720:
            issues.append("low_video_resolution")
            score -= 20
        duration = _probe_duration(path)
        if 0 < duration < 0.65:
            issues.append("too_short")
            score -= 25

    return LocalQualityResult(accept=score >= 55, score=max(0, score), issues=issues)


def _looks_like_supported_image(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            header = handle.read(16)
    except OSError:
        return False
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return True
    if header.startswith(b"\xff\xd8\xff"):
        return True
    if len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return True
    return False


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
