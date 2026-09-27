from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path

from .audio_plan import AudioPlan, AudioCue


def mix_audio(
    voiceover: str | Path,
    audio_plan: AudioPlan,
    output: str | Path,
    *,
    work_dir: str | Path,
    music: str | Path | None = None,
    sfx_dir: str | Path | None = None,
    premium: bool = False,
) -> Path:
    """Mix voiceover with planned SFX and optional background music using FFmpeg.

    Premium mode uses real sidechain ducking and planned music drops.
    Legacy mode below remains unchanged.
    """
    if premium:
        return _mix_audio_premium(
            voiceover,
            audio_plan,
            output,
            work_dir=work_dir,
            music=music,
            sfx_dir=sfx_dir,
        )

    _require("ffmpeg")
    voiceover = Path(voiceover)
    output = Path(output)
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    sfx_root = Path(sfx_dir) if sfx_dir else None

    inputs: list[Path] = [voiceover]
    delays: list[int] = [0]
    gains: list[float] = [0.0]

    for index, cue in enumerate(audio_plan.cues):
        asset = _resolve_sfx(cue, sfx_root, work / f"sfx_{index:03d}_{cue.name}.wav")
        inputs.append(asset)
        delays.append(max(0, int(round(cue.time * 1000))))
        gains.append(cue.gain_db)

    music_index: int | None = None
    if music:
        music_path = Path(music)
        if music_path.exists():
            music_index = len(inputs)
            inputs.append(music_path)
            delays.append(0)
            gains.append(audio_plan.music_gain_db)

    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    for idx, path in enumerate(inputs):
        if music_index is not None and idx == music_index:
            cmd += ["-stream_loop", "-1", "-i", str(path)]
        else:
            cmd += ["-i", str(path)]

    filters: list[str] = []
    mix_labels: list[str] = []
    for idx, (delay_ms, gain_db) in enumerate(zip(delays, gains)):
        label = f"a{idx}"
        chain = f"[{idx}:a]aresample=48000"
        if idx == 0:
            chain += ",highpass=f=70,alimiter=limit=0.95"
        if delay_ms:
            chain += f",adelay={delay_ms}|{delay_ms}"
        if abs(gain_db) > 1e-6:
            volume = math.pow(10.0, gain_db / 20.0)
            chain += f",volume={volume:.6f}"
        if music_index is not None and idx == music_index:
            chain += ",atrim=0:3600"
        filters.append(chain + f"[{label}]")
        mix_labels.append(f"[{label}]")

    # duration=first keeps the final mix exactly aligned to the voiceover.
    filters.append("".join(mix_labels) + f"amix=inputs={len(mix_labels)}:duration=first:normalize=0,alimiter=limit=0.96[mix]")
    cmd += [
        "-filter_complex", ";".join(filters),
        "-map", "[mix]",
        "-c:a", "aac", "-b:a", "192k",
        str(output),
    ]
    subprocess.run(cmd, check=True)
    return output



def _mix_audio_premium(
    voiceover: str | Path,
    audio_plan: AudioPlan,
    output: str | Path,
    *,
    work_dir: str | Path,
    music: str | Path | None = None,
    sfx_dir: str | Path | None = None,
) -> Path:
    """Voice-first premium mix with SFX, sidechain music and reveal drops."""

    _require("ffmpeg")

    voiceover = Path(voiceover)
    output = Path(output)
    work = Path(work_dir)

    work.mkdir(
        parents=True,
        exist_ok=True,
    )
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sfx_root = (
        Path(sfx_dir)
        if sfx_dir
        else None
    )

    sfx_assets = []

    for index, cue in enumerate(audio_plan.cues):
        asset = _resolve_sfx(
            cue,
            sfx_root,
            work / f"premium_sfx_{index:03d}_{cue.name}.wav",
        )

        sfx_assets.append(
            (
                asset,
                max(
                    0,
                    int(
                        round(
                            cue.time * 1000
                        )
                    ),
                ),
                cue.gain_db,
            )
        )

    music_path = (
        Path(music)
        if music
        else None
    )

    if (
        music_path is not None
        and not music_path.is_file()
    ):
        music_path = None

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(voiceover),
    ]

    for asset, _, _ in sfx_assets:
        cmd += [
            "-i",
            str(asset),
        ]

    music_index = None

    if music_path is not None:
        music_index = (
            1 + len(sfx_assets)
        )

        cmd += [
            "-stream_loop",
            "-1",
            "-i",
            str(music_path),
        ]

    filters = []

    if music_index is not None:
        filters.append(
            "[0:a]"
            "aresample=48000,"
            "highpass=f=70,"
            "alimiter=limit=0.95,"
            "asplit=2"
            "[voice][voice_sc]"
        )
    else:
        filters.append(
            "[0:a]"
            "aresample=48000,"
            "highpass=f=70,"
            "alimiter=limit=0.95"
            "[voice]"
        )

    sfx_labels = []

    for offset, (
        _,
        delay_ms,
        gain_db,
    ) in enumerate(
        sfx_assets,
        start=1,
    ):
        volume = math.pow(
            10.0,
            gain_db / 20.0,
        )

        label = f"sfx{offset}"

        filters.append(
            f"[{offset}:a]"
            f"aresample=48000,"
            f"adelay={delay_ms}|{delay_ms},"
            f"volume={volume:.6f}"
            f"[{label}]"
        )

        sfx_labels.append(
            f"[{label}]"
        )

    music_label = ""

    if music_index is not None:
        music_gain = math.pow(
            10.0,
            audio_plan.music_gain_db / 20.0,
        )

        chain = (
            f"[{music_index}:a]"
            f"aresample=48000,"
            f"volume={music_gain:.6f},"
            f"atrim=0:3600"
        )

        # Create a short vacuum immediately before the selected reveal.
        for cue_time in audio_plan.music_drops:
            start = max(
                0.0,
                float(cue_time) - 0.72,
            )

            end = max(
                start + .10,
                float(cue_time) + .10,
            )

            chain += (
                f",volume=0:"
                f"enable='between(t,{start:.3f},{end:.3f})'"
            )

        chain += "[music_raw]"

        filters.append(chain)

        filters.append(
            "[music_raw][voice_sc]"
            "sidechaincompress="
            "threshold=0.030:"
            "ratio=5:"
            "attack=5:"
            "release=140"
            "[music_ducked]"
        )

        music_label = "[music_ducked]"

    mix_inputs = (
        ["[voice]"]
        + sfx_labels
        + (
            [music_label]
            if music_label
            else []
        )
    )

    filters.append(
        "".join(mix_inputs)
        + f"amix=inputs={len(mix_inputs)}:"
        "duration=first:"
        "normalize=0,"
        "alimiter=limit=0.96"
        "[mix]"
    )

    cmd += [
        "-filter_complex",
        ";".join(filters),
        "-map",
        "[mix]",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        str(output),
    ]

    subprocess.run(
        cmd,
        check=True,
    )

    return output

def _resolve_sfx(cue: AudioCue, sfx_dir: Path | None, fallback: Path) -> Path:
    if sfx_dir:
        for suffix in (".wav", ".mp3", ".ogg", ".m4a"):
            candidate = sfx_dir / f"{cue.name}{suffix}"
            if candidate.exists():
                return candidate
    _synthesize_sfx(cue.name, fallback)
    return fallback


def _synthesize_sfx(name: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if name == "notification":
        src = "sine=frequency=880:duration=0.11"
        af = "volume=0.8,afade=t=out:st=0.07:d=0.04"
    elif name == "bass_hit":
        src = "sine=frequency=65:duration=0.34"
        af = "volume=1.0,afade=t=out:st=0.08:d=0.26"
    elif name == "impact":
        src = "anoisesrc=color=brown:duration=0.22"
        af = "lowpass=f=700,volume=0.75,afade=t=out:st=0.03:d=0.19"
    elif name == "cash_pop":
        src = "sine=frequency=1320:duration=0.08"
        af = "volume=0.7,afade=t=out:st=0.04:d=0.04"
    else:  # whoosh and unknown transition names
        src = "anoisesrc=color=white:duration=0.24"
        af = "highpass=f=700,lowpass=f=7000,volume=0.22,afade=t=in:st=0:d=0.06,afade=t=out:st=0.10:d=0.14"
    subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", src,
        "-af", af,
        "-ar", "48000", "-ac", "2", str(output),
    ], check=True)


def _require(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"{name} is required and was not found on PATH")
