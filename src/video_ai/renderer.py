from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

from .models import Scene, ShotPlan


def render_plan(
    plan: ShotPlan,
    output: str | Path,
    *,
    work_dir: str | Path | None = None,
    captions: bool = True,
    crf: int = 20,
) -> Path:
    """Render a local ShotPlan to a vertical MP4 with the original voiceover."""
    _require("ffmpeg")
    _require("ffprobe")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    audio_duration = _probe_duration(plan.audio)
    if audio_duration <= 0:
        audio_duration = max(scene.end for scene in plan.scenes)

    if work_dir is None:
        temp = tempfile.TemporaryDirectory(prefix="video-ai-")
        root = Path(temp.name)
    else:
        temp = None
        root = Path(work_dir)
        root.mkdir(parents=True, exist_ok=True)

    try:
        clips_dir = root / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)
        clips: list[Path] = []
        ranges = _display_ranges(plan, audio_duration)
        for index, (scene, start, end) in enumerate(ranges):
            clip = clips_dir / f"scene_{index:03d}.mp4"
            _render_scene(scene, end - start, plan, clip, crf=crf)
            clips.append(clip)

        concat_file = root / "concat.txt"
        concat_file.write_text(
            "\n".join("file '" + str(p.resolve()).replace("'", "'\\''") + "'" for p in clips) + "\n",
            encoding="utf-8",
        )
        base = root / "base.mp4"
        _run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", str(concat_file),
            "-c", "copy", str(base),
        ])

        final_filter: list[str] = []
        if captions:
            ass = root / "captions.ass"
            _write_ass(plan, ass)
            final_filter = ["-vf", f"ass='{_filter_path(ass)}'"]

        _run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(base), "-i", str(plan.audio),
            *final_filter,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264" if captions else "copy",
            *([] if not captions else ["-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p"]),
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart", "-shortest", str(output),
        ])
        return output
    finally:
        if temp is not None:
            temp.cleanup()


def _display_ranges(plan: ShotPlan, audio_duration: float) -> list[tuple[Scene, float, float]]:
    ranges: list[tuple[Scene, float, float]] = []
    for index, scene in enumerate(plan.scenes):
        start = 0.0 if index == 0 else max(0.0, scene.start)
        if index + 1 < len(plan.scenes):
            end = max(start + 0.05, plan.scenes[index + 1].start)
        else:
            end = max(start + 0.05, audio_duration)
        ranges.append((scene, start, end))
    return ranges


def _render_scene(scene: Scene, duration: float, plan: ShotPlan, output: Path, *, crf: int) -> None:
    duration = max(0.05, duration)
    common_out = [
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-r", str(plan.fps), "-t", f"{duration:.3f}", str(output),
    ]
    if scene.asset_kind == "image" and scene.asset and Path(scene.asset).exists():
        vf = _image_filter(scene, plan, duration)
        _run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-loop", "1", "-framerate", str(plan.fps), "-i", str(scene.asset),
            "-vf", vf, *common_out,
        ])
        return

    if scene.asset_kind == "video" and scene.asset and Path(scene.asset).exists():
        vf = (
            f"scale={plan.width}:{plan.height}:force_original_aspect_ratio=increase,"
            f"crop={plan.width}:{plan.height},fps={plan.fps}"
        )
        _run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-stream_loop", "-1", "-i", str(scene.asset), "-vf", vf, *common_out,
        ])
        return

    _run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", f"color=c=0x101014:s={plan.width}x{plan.height}:r={plan.fps}:d={duration:.3f}",
        *common_out,
    ])


def _image_filter(scene: Scene, plan: ShotPlan, duration: float) -> str:
    w, h, fps = plan.width, plan.height, plan.fps
    frames = max(2, int(math.ceil(duration * fps)))
    big_w = int(math.ceil(w * 1.16 / 2) * 2)
    big_h = int(math.ceil(h * 1.16 / 2) * 2)
    base = f"scale={big_w}:{big_h}:force_original_aspect_ratio=increase,crop={big_w}:{big_h}"

    if scene.motion == "zoom_in":
        return (
            base + "," +
            f"zoompan=z='min(1.0+0.12*on/{frames},1.12)':"
            f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s={w}x{h}:fps={fps}"
        )
    if scene.motion == "zoom_out":
        return (
            base + "," +
            f"zoompan=z='max(1.12-0.12*on/{frames},1.0)':"
            f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s={w}x{h}:fps={fps}"
        )
    if scene.motion == "pan_right":
        return base + f",crop={w}:{h}:x='(iw-ow)*min(n/{frames},1)':y='(ih-oh)/2',fps={fps}"
    if scene.motion == "pan_left":
        return base + f",crop={w}:{h}:x='(iw-ow)*(1-min(n/{frames},1))':y='(ih-oh)/2',fps={fps}"
    return base + f",crop={w}:{h},fps={fps}"


def _write_ass(plan: ShotPlan, path: Path) -> None:
    header = f"""[Script Info]\nScriptType: v4.00+\nPlayResX: {plan.width}\nPlayResY: {plan.height}\nScaledBorderAndShadow: yes\n\n[V4+ Styles]\nFormat: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding\nStyle: Default,Arial,{max(48, int(plan.width*0.066))},&H00FFFFFF,&H00FFFFFF,&H00101010,&H50000000,-1,0,0,0,100,100,0,0,1,5,0,2,80,80,{int(plan.height*0.26)},1\n\n[Events]\nFormat: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n"""
    events: list[str] = []
    for scene in plan.scenes:
        if not scene.caption:
            continue
        text = _ass_text(scene.caption)
        events.append(
            f"Dialogue: 0,{_ass_time(scene.start)},{_ass_time(scene.end)},Default,,0,0,0,,{text}"
        )
    path.write_text(header + "\n".join(events) + "\n", encoding="utf-8-sig")


def _ass_text(text: str) -> str:
    words = text.replace("{", "(").replace("}", ")").split()
    if len(words) <= 5:
        return " ".join(words)
    mid = min(5, max(2, math.ceil(len(words) / 2)))
    return " ".join(words[:mid]) + r"\N" + " ".join(words[mid:])


def _ass_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    cs = int(round(seconds * 100))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, cent = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{cent:02d}"


def _filter_path(path: Path) -> str:
    text = path.resolve().as_posix().replace("'", r"\'")
    if len(text) >= 2 and text[1] == ":":
        text = text[0] + r"\:" + text[2:]
    return text


def _probe_duration(path: str | Path) -> float:
    completed = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    )
    data = json.loads(completed.stdout)
    return float(data.get("format", {}).get("duration") or 0.0)


def _require(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"{name} is required and was not found on PATH")


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)
