from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from .audio_mix import mix_audio
from .audio_plan import AudioCue, AudioPlan
from .models import Scene, ShotPlan


def _first_word_time(scene: Scene, stems: tuple[str, ...]) -> float | None:
    for word in scene.caption_words:
        token = re.sub(r"[^\w]+", "", word.text.casefold())
        if any(stem in token for stem in stems):
            return float(word.start)
    return None


def _speech_gap(previous: Scene | None, current: Scene) -> float:
    if previous and previous.caption_words and current.caption_words:
        return max(
            0.0,
            float(current.caption_words[0].start)
            - float(previous.caption_words[-1].end),
        )
    if previous:
        return max(0.0, float(current.start) - float(previous.end))
    return 1.0


def build_premium_audio_plan(plan: ShotPlan) -> AudioPlan:
    cues: list[AudioCue] = []
    drops: list[float] = []
    full = " ".join((scene.caption or "") for scene in plan.scenes).casefold()

    mood = "neutral"
    if any(stem in full for stem in ("страш", "ужас", "смерт", "dark", "death", "war")):
        mood = "dark"
    elif any(stem in full for stem in ("деньг", "успех", "миллион", "money", "success")):
        mood = "energetic"

    for index, scene in enumerate(plan.scenes):
        previous = plan.scenes[index - 1] if index else None
        gap = _speech_gap(previous, scene)
        pace = getattr(scene, "pace_class", "normal") or "normal"

        if index > 0:
            if pace == "reveal":
                transition = "bass_hit"
                gain = -8.0
            elif scene.motion_preset == "snap_zoom" and gap < 0.08:
                transition = "whip"
                gain = -13.0
            elif gap < 0.10:
                transition = "impact"
                gain = -11.0
            elif gap >= 0.20:
                transition = "whoosh"
                gain = -15.0
            else:
                transition = "whoosh"
                gain = -17.0
            cues.append(
                AudioCue(
                    time=max(0.0, float(scene.start)),
                    kind="transition",
                    name=transition,
                    gain_db=gain,
                )
            )

        phone_time = _first_word_time(
            scene,
            ("телефон", "звон", "сообщ", "уведом", "phone", "call", "message"),
        )
        if phone_time is not None:
            cues.append(
                AudioCue(
                    time=phone_time,
                    kind="ui",
                    name="notification",
                    gain_db=-12.0,
                )
            )

        money_time = _first_word_time(
            scene,
            ("деньг", "цен", "миллион", "тысяч", "money", "price", "million", "thousand"),
        )
        if money_time is not None:
            cues.append(
                AudioCue(
                    time=money_time,
                    kind="accent",
                    name="cash_pop",
                    gain_db=-13.0,
                )
            )

        if pace == "reveal":
            first_word = (
                float(scene.caption_words[0].start)
                if scene.caption_words
                else float(scene.start)
            )
            lower_bound = (
                float(previous.start) + 0.25
                if previous is not None
                else 0.0
            )
            riser_start = max(lower_bound, first_word - 0.62)
            if first_word - riser_start >= 0.22:
                cues.append(
                    AudioCue(
                        time=riser_start,
                        kind="riser",
                        name="riser_up",
                        gain_db=-15.0,
                    )
                )

        if scene.premium_music_drop:
            drops.append(float(scene.start))

    cues.sort(key=lambda cue: cue.time)
    deduped: list[AudioCue] = []
    for cue in cues:
        if (
            deduped
            and cue.name == deduped[-1].name
            and cue.time - deduped[-1].time < 0.35
        ):
            continue
        deduped.append(cue)

    return AudioPlan(
        mood=mood,
        music_gain_db=-24.0,
        cues=deduped,
        music_drops=drops,
    )


def _generate_special_sfx(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)

    riser = root / "riser_up.wav"
    if not riser.is_file():
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "anoisesrc=color=white:duration=0.62:sample_rate=48000",
                "-af",
                (
                    "highpass=f=700,lowpass=f=7200,"
                    "afade=t=in:st=0:d=0.50,"
                    "afade=t=out:st=0.54:d=0.08,"
                    "volume=0.18"
                ),
                "-ar",
                "48000",
                "-ac",
                "2",
                str(riser),
            ],
            check=True,
        )

    whip = root / "whip.wav"
    if not whip.is_file():
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "anoisesrc=color=white:duration=0.16:sample_rate=48000",
                "-af",
                "highpass=f=1200,lowpass=f=9000,afade=t=out:st=0.035:d=0.125,volume=0.12",
                "-ar",
                "48000",
                "-ac",
                "2",
                str(whip),
            ],
            check=True,
        )

    return root


def _combined_sfx_dir(
    work_dir: Path,
    user_sfx_dir: str | Path | None,
) -> Path:
    root = work_dir / "premium_sfx"
    root.mkdir(parents=True, exist_ok=True)

    if user_sfx_dir:
        source = Path(user_sfx_dir)
        if source.is_dir():
            for item in source.iterdir():
                if item.suffix.lower() in {".wav", ".mp3", ".ogg", ".m4a"}:
                    target = root / item.name
                    if not target.exists():
                        shutil.copy2(item, target)

    return _generate_special_sfx(root)


def mix_premium_audio(
    voiceover: str | Path,
    plan: ShotPlan,
    output: str | Path,
    *,
    work_dir: str | Path,
    music: str | Path | None = None,
    sfx_dir: str | Path | None = None,
) -> Path:
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    plan_audio = build_premium_audio_plan(plan)
    combined = _combined_sfx_dir(work, sfx_dir)
    return mix_audio(
        voiceover,
        plan_audio,
        output,
        work_dir=work / "mix",
        music=music,
        sfx_dir=combined,
        premium=True,
    )
