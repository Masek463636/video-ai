from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

from .models import Scene, ShotPlan, Word


_MAX_CONTINUOUS_VISUAL_SECONDS = 2.85


def render_plan(
    plan: ShotPlan,
    output: str | Path,
    *,
    work_dir: str | Path | None = None,
    captions: bool = True,
    crf: int = 20,
    editing_polish: bool = False,
    overlays: list[dict] | None = None,
) -> Path:
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
        spans = _visual_spans(plan, audio_duration)
        print(f"[render] {len(plan.scenes)} subtitle scenes -> {len(spans)} continuous visual spans", flush=True)
        if editing_polish:
            print("[render] editing polish enabled: same assets and cut points, smoother motion + caption pop", flush=True)
        clips: list[Path] = []
        for index, (scene, start, end) in enumerate(spans):
            clip = clips_dir / f"span_{index:03d}.mp4"
            _render_scene(scene, end - start, plan, clip, crf=crf, editing_polish=editing_polish)
            clips.append(clip)

        concat_file = root / "concat.txt"
        concat_file.write_text(
            "\n".join("file '" + str(p.resolve()).replace("'", "'\\''") + "'" for p in clips) + "\n",
            encoding="utf-8",
        )
        base = root / "base.mp4"
        _run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(concat_file), "-c", "copy", str(base)])

        picture = base
        if overlays:
            overlayed = root / "overlayed.mp4"
            _apply_overlays(base, overlays, plan, overlayed, crf=crf)
            picture = overlayed

        final_filter: list[str] = []
        if captions:
            ass = root / "captions.ass"
            _write_ass(plan, ass, editing_polish=editing_polish)
            final_filter = ["-vf", f"ass='{_filter_path(ass)}'"]
        final_cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(picture), "-i", str(plan.audio), *final_filter,
        ]
        if overlays:
            # overlayed.mp4 carries only generated SFX; mix them under the real voiceover.
            final_cmd += [
                "-filter_complex", "[0:a:0][1:a:0]amix=inputs=2:normalize=0:dropout_transition=0[aout]",
                "-map", "0:v:0", "-map", "[aout]",
            ]
        else:
            final_cmd += ["-map", "0:v:0", "-map", "1:a:0"]
        final_cmd += [
            "-c:v", "libx264" if captions else "copy",
            *([] if not captions else ["-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p"]),
            "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest", str(output),
        ]
        _run(final_cmd)
        return output
    finally:
        if temp is not None:
            temp.cleanup()


def _display_ranges(plan: ShotPlan, audio_duration: float) -> list[tuple[Scene, float, float]]:
    ranges: list[tuple[Scene, float, float]] = []
    for index, scene in enumerate(plan.scenes):
        start = 0.0 if index == 0 else max(0.0, scene.start)
        end = max(start + 0.05, plan.scenes[index + 1].start) if index + 1 < len(plan.scenes) else max(start + 0.05, audio_duration)
        ranges.append((scene, start, end))
    return ranges


def _visual_spans(plan: ShotPlan, audio_duration: float) -> list[tuple[Scene, float, float]]:
    ranges = _display_ranges(plan, audio_duration)
    if not ranges:
        return []
    spans: list[tuple[Scene, float, float]] = []
    current_scene, current_start, current_end = ranges[0]
    for scene, start, end in ranges[1:]:
        can_merge = (
            _same_visual(current_scene, scene)
            and abs(start - current_end) <= 0.08
            and (end - current_start) <= _MAX_CONTINUOUS_VISUAL_SECONDS
        )
        if can_merge:
            current_end = end
            continue
        spans.append((current_scene, current_start, current_end))
        current_scene, current_start, current_end = scene, start, end
    spans.append((current_scene, current_start, current_end))
    return spans


def _same_visual(a: Scene, b: Scene) -> bool:
    if a.asset_kind == "blank" or b.asset_kind == "blank":
        return a.asset_kind == b.asset_kind == "blank"
    if a.asset_kind != b.asset_kind or not a.asset or not b.asset:
        return False
    try:
        same_asset = Path(a.asset).resolve() == Path(b.asset).resolve()
    except OSError:
        same_asset = str(a.asset) == str(b.asset)
    if not same_asset:
        return False
    if _resolved_preset(a) != _resolved_preset(b):
        return False
    if abs(_clamp_focus(a.focus_x) - _clamp_focus(b.focus_x)) > 0.03:
        return False
    if abs(_clamp_focus(a.focus_y) - _clamp_focus(b.focus_y)) > 0.03:
        return False
    return True


def _render_scene(scene: Scene, duration: float, plan: ShotPlan, output: Path, *, crf: int, editing_polish: bool = False) -> None:
    duration = max(0.05, duration)
    common = ["-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p", "-r", str(plan.fps), "-t", f"{duration:.3f}", str(output)]

    if scene.asset_kind == "image" and scene.asset and Path(scene.asset).exists():
        asset = Path(scene.asset)
        if not _silent_decode_probe(asset):
            print(f"[render] skipped corrupt image: {asset.name}", flush=True)
            _render_safe_background(duration, plan, output, crf=crf)
            return
        _run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-loop", "1", "-framerate", str(plan.fps), "-i", str(asset), "-vf", _image_filter(scene, plan, duration, editing_polish=editing_polish), *common])
        return

    if scene.asset_kind == "video" and scene.asset and Path(scene.asset).exists():
        asset = Path(scene.asset)
        if not _silent_decode_probe(asset):
            print(f"[render] skipped corrupt video: {asset.name}", flush=True)
            _render_safe_background(duration, plan, output, crf=crf)
            return
        if scene.source_mode == "meme_library":
            vf = _video_filter(scene, plan, duration)
            asset_duration = _safe_probe_duration(asset)
            if 0 < asset_duration < duration:
                stretch = duration / asset_duration
                if stretch <= 1.35:
                    vf = f"{vf},setpts={stretch:.6f}*PTS"
                else:
                    vf = f"{vf},tpad=stop_mode=clone:stop_duration={duration:.3f}"
            _run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(asset), "-vf", vf, *common])
            return
        _render_reference_video(asset, scene, duration, plan, output, crf=crf, editing_polish=editing_polish)
        return

    _render_safe_background(duration, plan, output, crf=crf)


def _render_reference_video(asset: Path, scene: Scene, duration: float, plan: ShotPlan, output: Path, *, crf: int, editing_polish: bool = False) -> None:
    """Render stock B-roll full-bleed.

    Editing polish deliberately changes only framing/motion. The selected asset
    and storyboard cut points stay untouched so the same materialized plan can
    be compared A/B.
    """
    w, h, fps = plan.width, plan.height, plan.fps
    if editing_polish:
        vf = _editorial_video_filter(scene, plan, duration)
    else:
        vf = (
            f"scale={w}:{h}:force_original_aspect_ratio=increase,"
            f"crop={w}:{h},fps={fps}"
        )
    _run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-stream_loop", "-1", "-i", str(asset),
        "-vf", vf,
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-r", str(fps), "-t", f"{duration:.3f}", str(output),
    ])


def _apply_overlays(base: Path, overlays: list[dict], plan: ShotPlan, output: Path, *, crf: int) -> None:
    """Composite larger, varied Shorts inserts plus generated whoosh/pop SFX."""
    valid = [
        item for item in overlays
        if item.get("asset") and Path(str(item["asset"])).exists()
        and float(item.get("end", 0)) > float(item.get("start", 0))
    ]
    if not valid:
        shutil.copyfile(base, output)
        return

    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(base)]
    for item in valid:
        cmd += ["-loop", "1", "-framerate", str(plan.fps), "-i", str(item["asset"])]

    filters: list[str] = []
    current = "[0:v]"
    for index, item in enumerate(valid):
        start = float(item["start"])
        end = float(item["end"])
        side = str(item.get("position") or "right")
        animation = str(item.get("animation") or "fly")
        size = str(item.get("size") or "large")
        scale_ratio = 0.56 if size == "hero" else 0.44
        max_h_ratio = 0.46 if size == "hero" else 0.36
        width = int(plan.width * scale_ratio)
        max_h = int(plan.height * max_h_ratio)
        target_y = int(plan.height * (0.20 if size == "hero" else (0.24 if index % 2 == 0 else 0.37)))

        ov = f"[ov{index}]"
        nxt = f"[v{index}]"
        filters.append(
            f"[{index + 1}:v]"
            f"scale=w='min({width},iw)':h='min({max_h},ih)':force_original_aspect_ratio=decrease,"
            f"format=rgba{ov}"
        )

        if side == "left":
            settled_x = "70"
        elif side == "center":
            settled_x = "(main_w-overlay_w)/2"
        else:
            settled_x = "main_w-overlay_w-70"

        if animation == "fly":
            settle = start + 0.16
            if side == "left":
                x = (
                    f"if(lt(t,{settle:.3f}),"
                    f"-overlay_w+(t-{start:.3f})/0.16*(overlay_w+70),70)"
                )
            elif side == "right":
                x = (
                    f"if(lt(t,{settle:.3f}),"
                    f"main_w-(t-{start:.3f})/0.16*(overlay_w+70),"
                    f"main_w-overlay_w-70)"
                )
            else:
                x = settled_x
        else:
            x = settled_x

        filters.append(
            f"{current}{ov}overlay=x='{x}':y={target_y}:"
            f"enable='between(t,{start:.3f},{end:.3f})':shortest=1{nxt}"
        )

        label = str(item.get("label") or "").strip()
        if label:
            labelled = f"[vl{index}]"
            font_size = max(54, int(plan.width * (0.073 if size == "hero" else 0.062)))
            if side == "left":
                text_x = 80
            elif side == "right":
                text_x = f"w-tw-80"
            else:
                text_x = "(w-tw)/2"
            text_y = int(plan.height * (0.56 if size == "hero" else 0.52))
            safe_label = label.replace("\\", r"\\").replace(":", r"\:").replace("'", r"\'")
            filters.append(
                f"{nxt}drawtext=text='{safe_label}':"
                f"font='Arial Black':fontsize={font_size}:fontcolor=white:"
                f"borderw=7:bordercolor=black:"
                f"x='{text_x}':y={text_y}:"
                f"enable='between(t,{start:.3f},{end:.3f})'{labelled}"
            )
            current = labelled
        else:
            current = nxt

    # Add simple generated sound accents so fly-ins and instant pops are audible
    # without requiring an external SFX library.
    sound_labels: list[str] = []
    for index, item in enumerate(valid):
        start = float(item["start"])
        animation = str(item.get("animation") or "fly")
        delay = int(round(start * 1000))
        label = f"[s{index}]"
        if animation == "fly":
            # Short filtered-noise whoosh with a fast decay.
            filters.append(
                f"anoisesrc=color=white:duration=0.18:sample_rate=48000,"
                f"highpass=f=650,lowpass=f=5200,"
                f"afade=t=out:st=0.06:d=0.12,volume=0.11,"
                f"adelay={delay}|{delay}{label}"
            )
        else:
            # Tight pop/click for instant appearance.
            filters.append(
                f"sine=frequency=1550:duration=0.075:sample_rate=48000,"
                f"afade=t=out:st=0.02:d=0.055,volume=0.085,"
                f"adelay={delay}|{delay}{label}"
            )
        sound_labels.append(label)

    if sound_labels:
        filters.append("".join(sound_labels) + f"amix=inputs={len(sound_labels)}:normalize=0[sfx]")
        audio_map = ["-map", "[sfx]"]
    else:
        audio_map = []

    cmd += [
        "-filter_complex", ";".join(filters),
        "-map", current, *audio_map,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-r", str(plan.fps),
        *([] if not sound_labels else ["-c:a", "aac", "-b:a", "128k"]),
        str(output),
    ]
    _run(cmd)


def _render_safe_background(duration: float, plan: ShotPlan, output: Path, *, crf: int) -> None:
    _run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", f"color=c=0x101014:s={plan.width}x{plan.height}:r={plan.fps}:d={duration:.3f}",
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-r", str(plan.fps), "-t", f"{duration:.3f}", str(output),
    ])


def _silent_decode_probe(path: Path) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return True
    try:
        completed = subprocess.run(
            [
                ffmpeg,
                "-v", "error",
                "-xerror",
                "-err_detect", "explode",
                "-i", str(path),
                "-frames:v", "1",
                "-f", "null", "-",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=12,
            check=False,
        )
        return completed.returncode == 0 and not completed.stderr.strip()
    except Exception:
        return False


def _image_filter(scene: Scene, plan: ShotPlan, duration: float, *, editing_polish: bool = False) -> str:
    w, h, fps = plan.width, plan.height, plan.fps
    render_scale = 1.5
    rw = int(math.ceil(w * render_scale / 2) * 2)
    rh = int(math.ceil(h * render_scale / 2) * 2)
    frames = max(2, int(math.ceil(duration * fps)))
    denominator = max(1, frames - 1)
    fx = _clamp_focus(scene.focus_x)
    fy = _clamp_focus(scene.focus_y)
    preset = _composition_preset(scene, fx)

    overscan = 1.16 if preset in {"reveal_left", "reveal_right"} else 1.13
    big_w = int(math.ceil(rw * overscan / 2) * 2)
    big_h = int(math.ceil(rh * overscan / 2) * 2)
    base = f"scale={big_w}:{big_h}:force_original_aspect_ratio=increase,crop={big_w}:{big_h}:x='{_focus_expr('iw','ow',fx)}':y='{_focus_expr('ih','oh',fy)}'"

    t_on = f"min(max(on/{denominator},0),1)"
    t_n = f"min(max(n/{denominator},0),1)"
    ease_on = _ae_ease_expr(t_on, 2.1)
    ease_n = _ae_ease_expr(t_n, 2.1)
    finish = f",scale={w}:{h}:flags=lanczos,fps={fps}"

    if preset == "none":
        return base + f",crop={rw}:{rh}:x='{_focus_expr('iw','ow',fx)}':y='{_focus_expr('ih','oh',fy)}'" + finish

    if preset in {"micro_push", "slow_push", "dramatic_push", "pull_back"}:
        if preset == "micro_push":
            start, end = 1.000, 1.018 if editing_polish else 1.020
        elif preset == "slow_push":
            if editing_polish:
                end = 1.026 if duration < 1.45 else 1.040 if duration < 2.45 else 1.052
                start = 1.000
            else:
                start, end = 1.000, 1.050
        elif preset == "dramatic_push":
            start, end = 1.000, 1.065 if editing_polish else 1.080
        else:
            start, end = (1.050, 1.000) if editing_polish else (1.065, 1.000)
        delta = end - start
        zoom = f"{start:.5f}+({delta:.5f})*{ease_on}"
        return base + "," + f"zoompan=z='{zoom}':x='max(0,min(iw-iw/zoom,{fx:.5f}*iw-iw/zoom/2))':y='max(0,min(ih-ih/zoom,{fy:.5f}*ih-ih/zoom/2))':d=1:s={rw}x{rh}:fps={fps}" + finish

    if preset in {"reveal_left", "reveal_right"}:
        direction = "1" if preset == "reveal_left" else "-1"
        travel = 0.18 if editing_polish else 0.28
        offset = f"({direction})*(iw-ow)*{travel:.2f}*(1-{ease_n})"
        x_pan = f"max(0,min(iw-ow,{fx:.5f}*iw-ow/2+{offset}))"
        return base + f",crop={rw}:{rh}:x='{x_pan}':y='{_focus_expr('ih','oh',fy)}'" + finish

    return base + f",crop={rw}:{rh}:x='{_focus_expr('iw','ow',fx)}':y='{_focus_expr('ih','oh',fy)}'" + finish


def _editorial_video_filter(scene: Scene, plan: ShotPlan, duration: float) -> str:
    """Subtle editor-style motion for already selected stock footage.

    Motion stays intentionally small because the source video already moves.
    High-energy tones get a short smooth settle; ordinary footage receives a
    restrained push/pull so static full-bleed crops do not feel machine-made.
    """
    w, h, fps = plan.width, plan.height, plan.fps
    fx = _clamp_focus(scene.focus_x)
    fy = _clamp_focus(scene.focus_y)
    base = (
        f"scale={w}:{h}:force_original_aspect_ratio=increase,"
        f"crop={w}:{h}:x='{_focus_expr('iw','ow',fx)}':y='{_focus_expr('ih','oh',fy)}'"
    )
    frames = max(2, int(math.ceil(duration * fps)))
    denominator = max(1, frames - 1)
    t = f"min(max(n/{denominator},0),1)"
    ease = _ae_ease_expr(t, 2.2)
    preset = _resolved_preset(scene)

    if preset == "dramatic_push":
        start, end = 1.000, 1.045
    elif preset == "pull_back":
        start, end = 1.028, 1.000
    elif preset == "slow_push":
        start, end = 1.000, 1.026
    elif preset == "micro_push":
        start, end = 1.000, 1.016
    elif scene.tone in {"shocking", "violent", "tense", "absurd", "funny"}:
        start, end = 1.030, 1.006
    else:
        seed = sum(ord(ch) for ch in (scene.caption or scene.query or ""))
        start, end = ((1.000, 1.016) if seed % 2 == 0 else (1.016, 1.000))

    delta = end - start
    scale = f"{start:.5f}+({delta:.5f})*{ease}"
    return (
        base
        + f",scale='trunc(iw*({scale})/2)*2':'trunc(ih*({scale})/2)*2':eval=frame"
        + f",crop={w}:{h}:x='(iw-ow)/2':y='(ih-oh)/2',fps={fps}"
    )


def _video_filter(scene: Scene, plan: ShotPlan, duration: float) -> str:
    fx = _clamp_focus(scene.focus_x)
    fy = _clamp_focus(scene.focus_y)
    preset = _resolved_preset(scene)
    base = f"scale={plan.width}:{plan.height}:force_original_aspect_ratio=increase,crop={plan.width}:{plan.height}:x='{_focus_expr('iw','ow',fx)}':y='{_focus_expr('ih','oh',fy)}'"
    if preset not in {"micro_push", "slow_push"}:
        return base + f",fps={plan.fps}"

    frames = max(2, int(math.ceil(duration * plan.fps)))
    denominator = max(1, frames - 1)
    t = f"min(max(n/{denominator},0),1)"
    ease = _ae_ease_expr(t, 2.0)
    strength = 0.008 if preset == "micro_push" else 0.014
    scale = f"1+({strength:.5f})*{ease}"
    return base + f",scale='iw*({scale})':'ih*({scale})':eval=frame,crop={plan.width}:{plan.height},fps={plan.fps}"


def _composition_preset(scene: Scene, focus_x: float) -> str:
    preset = _resolved_preset(scene)
    if preset not in {"reveal_left", "reveal_right"}:
        return preset
    if 0.40 <= focus_x <= 0.60:
        return "slow_push"
    return "reveal_left" if focus_x < 0.40 else "reveal_right"


def _resolved_preset(scene: Scene) -> str:
    preset = getattr(scene, "motion_preset", "none") or "none"
    if preset != "none":
        return preset
    mapping = {"zoom_in": "slow_push", "zoom_out": "pull_back", "pan_left": "reveal_left", "pan_right": "reveal_right"}
    return mapping.get(scene.motion, "none")


def _ae_ease_expr(t: str, power: float = 2.0) -> str:
    p = f"{power:.3f}"
    a = f"pow({t},{p})"
    b = f"pow(1-({t}),{p})"
    return f"(({a})/max(0.000001,({a})+({b})))"


def _clamp_focus(value: float | None) -> float:
    return 0.5 if value is None else max(0.0, min(1.0, float(value)))


def _focus_expr(inner: str, outer: str, focus: float) -> str:
    return f"max(0,min({inner}-{outer},{focus:.5f}*{inner}-{outer}/2))"


def _write_ass(plan: ShotPlan, path: Path, *, editing_polish: bool = False) -> None:
    # Reference-style Shorts subtitles: very large Impact text around the
    # lower-middle of the frame, with a heavy black stroke.
    font_size = max(96, int(plan.width * 0.118))
    outline = max(9, int(plan.width * 0.010))
    margin_v = int(plan.height * 0.40)
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {plan.width}
PlayResY: {plan.height}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
Style: Default,Impact,{font_size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H20000000,-1,0,0,0,96,100,-2,0,1,{outline},0,2,50,50,{margin_v},1

[Events]
Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
"""
    events: list[str] = []
    for scene in plan.scenes:
        if not scene.caption:
            continue
        pages = _caption_pages(scene.caption, scene.start, scene.end, timed_words=scene.caption_words)
        for page_start, page_end, text in pages:
            events.append(
                f"Dialogue: 0,{_ass_time(page_start)},{_ass_time(page_end)},Default,,0,0,0,,{_ass_text(text, editing_polish=editing_polish)}"
            )
    path.write_text(header + "\n".join(events) + "\n", encoding="utf-8-sig")


def _caption_pages(text: str, start: float, end: float, *, timed_words: list[Word] | None = None) -> list[tuple[float, float, str]]:
    """Split captions into punchy 1-2 word chunks.

    Short connector words ("и", "в", "на", "to", "of", etc.) are attached to
    a neighbour when possible, while long content words usually get their own
    card. This keeps every subtitle to at most two words.
    """
    words = text.replace("{", "(").replace("}", ")").split()
    if not words:
        return []

    # Use speech timestamps only when they describe this exact caption. Older
    # plans (or edited captions) retain the stable-test45 character weighting.
    aligned = _aligned_caption_words(words, timed_words, start, end)

    groups: list[list[str]] = []
    i = 0
    while i < len(words):
        current = [words[i]]
        clean_current = _caption_token(words[i])
        if i + 1 < len(words):
            next_word = words[i + 1]
            clean_next = _caption_token(next_word)
            combined_len = len(clean_current) + len(clean_next)
            current_is_connector = len(clean_current) <= 3
            next_is_connector = len(clean_next) <= 3
            current_has_break = words[i].endswith((",", ";", ":", ".", "!", "?"))

            # Prefer two-word cards only when they read naturally and stay compact.
            if (
                not current_has_break
                and (not aligned or aligned[i + 1].start - aligned[i].end <= 0.18)
                and (
                    current_is_connector
                    or next_is_connector
                )
            ):
                current.append(next_word)
                i += 1
        groups.append(current)
        i += 1

    if aligned:
        out: list[tuple[float, float, str]] = []
        offset = 0
        for group in groups:
            first = offset
            offset += len(group)
            page_start = max(start, aligned[first].start)
            page_end = min(end, aligned[offset - 1].end)
            if offset < len(aligned):
                page_end = min(page_end, aligned[offset].start)
            out.append((page_start, page_end, " ".join(group)))
        return out

    total_weight = sum(max(1, _caption_group_weight(group)) for group in groups)
    duration = max(0.06, end - start)
    cursor = start
    out: list[tuple[float, float, str]] = []
    for i, group in enumerate(groups):
        weight = max(1, _caption_group_weight(group))
        share = duration * (weight / max(1, total_weight))
        page_end = end if i == len(groups) - 1 else min(end, cursor + share)
        out.append((cursor, max(cursor + 0.05, page_end), " ".join(group)))
        cursor = page_end
    return out


def _aligned_caption_words(tokens: list[str], words: list[Word] | None, start: float, end: float) -> list[Word] | None:
    if not words or len(words) != len(tokens):
        return None
    previous_start = -1.0
    for token, word in zip(tokens, words):
        safe_text = word.text.replace("{", "(").replace("}", ")").strip()
        if safe_text != token:
            return None
        if not math.isfinite(word.start) or not math.isfinite(word.end):
            return None
        if (word.start < start - 0.001 or word.end > end + 0.001
                or word.end <= word.start or word.start <= previous_start
                or word.start >= end):
            return None
        previous_start = word.start
    return words


def _caption_token(word: str) -> str:
    return word.strip(" \\t\\r\\n,.;:!?—–-()[]«»\"'").casefold()


def _caption_group_weight(group: list[str]) -> int:
    # Spoken duration tracks characters reasonably well when exact word timings
    # are not stored in the ShotPlan. Give each word a tiny base weight so short
    # connectors do not flash too quickly.
    return sum(max(2, len(_caption_token(word))) for word in group)


def _ass_text(text: str, *, editing_polish: bool = False) -> str:
    safe = text.replace("{", "(").replace("}", ")")
    if editing_polish:
        # Small 70ms scale-up gives the caption a manual-edit punch without
        # bouncing or lingering after the spoken word.
        start_scale = 92 if len(text.split()) == 1 else 95
        return (
            "{\\\\fscx" + str(start_scale)
            + "\\\\fscy" + str(start_scale)
            + "\\\\t(0,70,\\\\fscx100\\\\fscy100)\\\\fad(4,8)}"
            + safe
        )
    # Stable/test45 behaviour remains the default.
    return r"{\\fad(8,12)}" + safe

def _ass_time(seconds: float) -> str:
    cs = int(round(max(0.0, seconds) * 100))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, cent = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{cent:02d}"


def _filter_path(path: Path) -> str:
    text = path.resolve().as_posix().replace("'", r"\'")
    if len(text) >= 2 and text[1] == ":":
        text = text[0] + r"\:" + text[2:]
    return text


def _safe_probe_duration(path: str | Path) -> float:
    try:
        return _probe_duration(path)
    except Exception:
        return 0.0


def _probe_duration(path: str | Path) -> float:
    completed = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)], check=True, capture_output=True, text=True)
    return float(json.loads(completed.stdout).get("format", {}).get("duration") or 0.0)


def _require(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"{name} is required and was not found on PATH")


def _run(cmd: list[str]) -> None:
    completed = subprocess.run(
        cmd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode == 0:
        return
    lines = [line.strip() for line in completed.stderr.splitlines() if line.strip()]
    detail = " | ".join(lines[-3:])[:800] if lines else f"exit code {completed.returncode}"
    raise RuntimeError(f"ffmpeg failed: {detail}")
