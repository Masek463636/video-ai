from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re

from .models import ShotPlan


@dataclass(slots=True)
class AudioCue:
    time: float
    kind: str
    name: str
    gain_db: float = -8.0


@dataclass(slots=True)
class AudioPlan:
    mood: str
    music_gain_db: float
    cues: list[AudioCue]


def build_audio_plan(plan: ShotPlan) -> AudioPlan:
    """Create cheap deterministic music/SFX cues from captions and cut positions."""
    full = " ".join((scene.caption or "") for scene in plan.scenes).lower()
    mood = _mood(full)
    cues: list[AudioCue] = []

    for index, scene in enumerate(plan.scenes):
        text = (scene.caption or "").lower()
        if index > 0:
            cues.append(AudioCue(time=scene.start, kind="transition", name="whoosh", gain_db=-12.0))
        if re.search(r"\b(но|вдруг|резко|однако|и тут|но тут)\b", text):
            cues.append(AudioCue(time=scene.start, kind="impact", name="impact", gain_db=-7.0))
        if re.search(r"(телефон|сообщени|уведомлен|звонок)", text):
            cues.append(AudioCue(time=scene.start + 0.08, kind="ui", name="notification", gain_db=-10.0))
        if re.search(r"(страш|шок|ужас|неожидан|пиздец)", text):
            cues.append(AudioCue(time=scene.start, kind="impact", name="bass_hit", gain_db=-6.0))
        if re.search(r"(деньг|цена|дорог|дешев|миллион|тысяч)", text):
            cues.append(AudioCue(time=scene.start + 0.1, kind="accent", name="cash_pop", gain_db=-12.0))

    return AudioPlan(mood=mood, music_gain_db=-22.0, cues=_dedupe(cues))


def save_audio_plan(audio_plan: AudioPlan, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "mood": audio_plan.mood,
        "music_gain_db": audio_plan.music_gain_db,
        "cues": [asdict(cue) for cue in audio_plan.cues],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _mood(text: str) -> str:
    buckets = {
        "dark": ("страш", "ужас", "убий", "ноч", "тайн", "шок"),
        "sad": ("груст", "расстро", "провал", "потер", "расстал"),
        "energetic": ("деньг", "успех", "выигр", "миллион", "топ", "быстро"),
    }
    for mood, stems in buckets.items():
        if any(stem in text for stem in stems):
            return mood
    return "neutral"


def _dedupe(cues: list[AudioCue]) -> list[AudioCue]:
    cues.sort(key=lambda cue: cue.time)
    out: list[AudioCue] = []
    for cue in cues:
        if out and cue.name == out[-1].name and cue.time - out[-1].time < 0.45:
            continue
        out.append(cue)
    return out
