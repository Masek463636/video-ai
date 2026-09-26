from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from dataclasses import replace

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
    reference_framing: bool = False,
    composition_review: bool = False,
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
            print("[render] editing polish enabled: smoother motion + caption pop", flush=True)
        reviewer = None
        if composition_review:
            from .composition_review import CompositionReviewer
            reviewer = CompositionReviewer(root / "composition")
        clips: list[Path] = []
        for index, (scene, start, end) in enumerate(spans):
            clip = clips_dir / f"span_{index:03d}.mp4"
            _render_scene(
                scene,
                end - start,
                plan,
                clip,
                crf=crf,
                editing_polish=editing_polish,
                reference_framing=reference_framing,
            )
            if reviewer is not None:
                reviewer.scene(scene, plan, end-start, clip, index,
                               reference_framing=reference_framing, editing_polish=editing_polish)
            clips.append(clip)

        concat_file = root / "concat.txt"
        concat_file.write_text(
            "\n".join("file '" + str(p.resolve()).replace("'", "'\\''") + "'" for p in clips) + "\n",
            encoding="utf-8",
        )
        base = root / "base.mp4"
        _run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(concat_file), "-c", "copy", str(base)])

        if reviewer is not None:
            overlays = reviewer.overlays(base, overlays or [], plan)
            counts = {}
            for row in reviewer.report['scenes']:
                counts[row['status']] = counts.get(row['status'], 0) + 1
            print('[composition] result: '+json.dumps(counts)+f'; visible reactions={len(overlays)}',flush=True)
        picture = base
        if overlays:
            overlayed = root / "overlayed.mp4"
            _apply_overlays(base, overlays, plan, overlayed, crf=crf)
            picture = overlayed

        final_filter: list[str] = []
        if captions:
            ass = root / "captions.ass"
            _write_ass(plan, ass, editing_polish=editing_polish, overlays=overlays)
            final_filter = ["-vf", f"ass='{_filter_path(ass)}'"]
        sfx_track: Path | None = None
        if overlays:
            sfx_track = root / "shorts_fx.wav"
            _build_overlay_sfx(overlays, audio_duration, sfx_track)

        final_cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-reinit_filter", "0", "-i", str(picture), "-i", str(plan.audio),
        ]
        if sfx_track is not None and sfx_track.exists():
            final_cmd += ["-i", str(sfx_track)]

        if captions:
            final_cmd += final_filter

        if sfx_track is not None and sfx_track.exists():
            final_cmd += [
                "-filter_complex", "[1:a:0][2:a:0]amix=inputs=2:duration=first:normalize=0[aout]",
                "-map", "0:v:0", "-map", "[aout]",
            ]
        else:
            final_cmd += ["-map", "0:v:0", "-map", "1:a:0"]

        final_cmd += [
            "-c:v", "libx264" if captions else "copy",
            *([] if not captions else ["-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p"]),
            "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
            "-t", f"{audio_duration:.3f}", str(output),
        ]
        _run(final_cmd)
        _assert_duration(output, audio_duration, label="final render")
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
    if a.asset_kind == "video" and (a.source_start or b.source_start):
        return False
    if _resolved_preset(a) != _resolved_preset(b):
        return False
    if abs(_clamp_focus(a.focus_x) - _clamp_focus(b.focus_x)) > 0.03:
        return False
    if abs(_clamp_focus(a.focus_y) - _clamp_focus(b.focus_y)) > 0.03:
        return False
    return True


def _render_scene(
    scene: Scene,
    duration: float,
    plan: ShotPlan,
    output: Path,
    *,
    crf: int,
    editing_polish: bool = False,
    reference_framing: bool = False,
) -> None:
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
        _render_reference_video(
            asset,
            scene,
            duration,
            plan,
            output,
            crf=crf,
            editing_polish=editing_polish,
            reference_framing=reference_framing,
        )
        return

    _render_safe_background(duration, plan, output, crf=crf)


def _render_reference_video(
    asset: Path,
    scene: Scene,
    duration: float,
    plan: ShotPlan,
    output: Path,
    *,
    crf: int,
    editing_polish: bool = False,
    reference_framing: bool = False,
) -> None:
    """Fill portraits; crop wide sources into a large top panel over blur."""
    w, h, fps = plan.width, plan.height, plan.fps
    size = _probe_video_size(asset)
    ratio = (size[0] / size[1]) if size and size[1] else None
    use_blur_fit = bool(reference_framing and ratio is not None and ratio >= 1.0)
    source_duration = _safe_probe_duration(asset)
    source_start = min(max(0.0, scene.source_start), max(0.0, source_duration - duration))
    seek = ["-ss", f"{source_start:.3f}"] if source_start else []

    if use_blur_fit:
        # A fixed large panel is intentional: ultrawide sources must not become
        # a thin strip. Crop edges around the supplied focus (center by default).
        fg_h = max(2, int(h * 0.52) // 2 * 2)
        panel_plan = replace(plan, height=fg_h)
        if editing_polish:
            foreground = _editorial_video_filter(scene, panel_plan, duration)
        else:
            foreground = (
                f"scale={w}:{fg_h}:force_original_aspect_ratio=increase,"
                f"crop={w}:{fg_h}:x='{_focus_expr('iw','ow',_clamp_focus(scene.focus_x))}':"
                f"y='{_focus_expr('ih','oh',_clamp_focus(scene.focus_y))}'"
            )
        filter_complex = (
            f"[0:v]scale=trunc(iw*sar/2)*2:ih,setsar=1,split=2[bg][fg];"
            f"[bg]scale={w}:{h}:force_original_aspect_ratio=increase,"
            f"crop={w}:{h},gblur=sigma=28:steps=2[bg2];"
            f"[fg]{foreground}[fg2];"
            f"[bg2][fg2]overlay=0:0:shortest=1,setsar=1,fps={fps}[v]"
        )
        _run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-stream_loop", "-1", *seek, "-i", str(asset),
            "-filter_complex", filter_complex,
            "-map", "[v]",
            "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
            "-pix_fmt", "yuv420p", "-r", str(fps), "-t", f"{duration:.3f}", str(output),
        ])
        return

    if editing_polish:
        vf = _editorial_video_filter(scene, plan, duration)
    else:
        vf = (
            f"scale={w}:{h}:force_original_aspect_ratio=increase,"
            f"crop={w}:{h},fps={fps}"
        )
    vf = "scale=trunc(iw*sar/2)*2:ih,setsar=1," + vf
    _run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-stream_loop", "-1", *seek, "-i", str(asset),
        "-vf", vf,
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-r", str(fps), "-t", f"{duration:.3f}", str(output),
    ])


def _probe_video_size(path: str | Path) -> tuple[int, int] | None:
    try:
        completed = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height,sample_aspect_ratio:stream_side_data=rotation",
                "-of", "json",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        streams = json.loads(completed.stdout).get("streams") or []
        if not streams:
            return None
        width = int(streams[0].get("width") or 0)
        height = int(streams[0].get("height") or 0)
        sar = str(streams[0].get("sample_aspect_ratio") or "1:1").split(":")
        if len(sar) == 2 and all(part.isdigit() for part in sar) and int(sar[1]) > 0:
            width = round(width * int(sar[0]) / int(sar[1]))
        rotation = next((item.get("rotation", 0) for item in streams[0].get("side_data_list", []) if "rotation" in item), 0)
        if abs(round(float(rotation))) % 180 == 90:
            width, height = height, width
        return (width, height) if width > 0 and height > 0 else None
    except Exception:
        return None

def _apply_overlays(base: Path, overlays: list[dict], plan: ShotPlan, output: Path, *, crf: int) -> None:
    """Composite PNG inserts over the existing edit without touching audio.

    PNG inputs are explicitly limited to the base duration and overlay never
    uses shortest=1. The base video therefore owns the timeline and cannot be
    truncated to a few seconds or extended by a looped PNG stream.
    """
    base_duration = max(0.05, _probe_duration(base))
    # Foreground sticker motion is rendered at 60 fps even when the base edit
    # is 30 fps. The B-roll simply duplicates frames, while the overlay
    # position/easing is evaluated twice as often and therefore looks much
    # smoother on fast fly-ins.
    fx_fps = max(60, int(plan.fps))
    valid = [
        item for item in overlays
        if float(item.get("end", 0)) > float(item.get("start", 0))
        and (
            str(item.get("type") or "png") == "text"
            or (item.get("asset") and Path(str(item["asset"])).exists())
        )
    ]
    if not valid:
        shutil.copyfile(base, output)
        return

    # All rendered spans already have identical dimensions/pixel format. Stock
    # clips may still carry different color metadata. Recent FFmpeg versions
    # reinitialize the filter graph at those boundaries, resetting setpts/fps
    # and making the remaining footage play early followed by cloned frames.
    # Preserve the graph's timeline across metadata-only changes.
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-reinit_filter", "0", "-i", str(base)]
    input_index_by_effect: dict[int, int] = {}
    next_input = 1
    for effect_index, item in enumerate(valid):
        if str(item.get("type") or "png") == "text":
            continue
        input_index_by_effect[effect_index] = next_input
        asset_path = Path(str(item["asset"]))
        if asset_path.suffix.lower() in {".gif", ".mp4", ".webm", ".mov"}:
            cmd += [
                "-stream_loop", "-1",
                "-i", str(asset_path),
            ]
        else:
            cmd += [
                "-loop", "1",
                "-framerate", str(plan.fps),
                "-t", f"{base_duration:.3f}",
                "-i", str(asset_path),
            ]
        next_input += 1

    filters: list[str] = [
        # Concat-copy MP4s can report the correct container duration while their
        # decoded video frames end a little earlier because of source time-base
        # gaps. Overlay would then stop at the last decoded frame even with
        # shortest=0. Pad with the final frame first, then trim to the intended
        # base duration so the FX pass can never shorten the edit.
        f"[0:v]setpts=PTS-STARTPTS,"
        f"tpad=stop_mode=clone:stop_duration={base_duration:.3f},"
        f"trim=duration={base_duration:.3f},fps={fx_fps}[basev]"
    ]
    current = "[basev]"

    for index, item in enumerate(valid):
        effect_type = str(item.get("type") or "png")
        if effect_type == "text":
            continue

        start = max(0.0, float(item["start"]))
        end = min(base_duration, float(item["end"]))
        side = str(item.get("position") or "right")
        animation = str(item.get("animation") or "fly")
        size = str(item.get("size") or "large")
        # Foreground inserts should read immediately on a phone. Keep them
        # large and inside the upper third so they never compete with captions.
        # Foreground inserts should be unmistakably readable on a phone.
        # The previous 60-72% width looked tiny against a 9:16 canvas.
        scale_ratio = 0.90 if size == "hero" else 0.80
        max_h_ratio = 0.48 if size == "hero" else 0.42
        width = int(plan.width * scale_ratio)
        max_h = int(plan.height * max_h_ratio)
        target_y = int(plan.height * (0.10 if size == "hero" else 0.12))

        layout = item.get("layout_box")
        if layout is not None:
            from .composition_review import boxes
            layout = boxes([layout])[0]
            width = max(2, int(plan.width * layout[2]))
            max_h = max(2, int(plan.height * layout[3]))
            target_y = round(plan.height * layout[1])
            animation = "pop"

        ov = f"[ov{index}]"
        nxt = f"[v{index}]"
        input_index = input_index_by_effect[index]
        # Give foreground cutouts a loose meme-sticker feel instead of a
        # perfectly upright catalogue-PNG look.
        tilt = 0.0 if layout is not None else (-0.055 if index % 2 == 0 else 0.045)
        rotation_filter = "" if layout is not None else f"rotate={tilt}:fillcolor=none:ow=rotw(iw):oh=roth(ih),"
        fade_start = max(start + 0.20, end - 0.10)
        filters.append(
            f"[{input_index}:v]"
            f"trim=duration={base_duration:.3f},setpts=PTS-STARTPTS,fps={fx_fps},"
            # Deliberately allow upscaling. The old min(iw/ih) clamp left many
            # Commons PNGs tiny even when the editor requested a hero insert.
            f"scale=w={width}:h={max_h}:force_original_aspect_ratio=decrease,"
            f"format=rgba,"
            f"{rotation_filter}"
            # Hold fully visible, then disappear quickly instead of lingering.
            f"fade=t=out:st={fade_start:.3f}:d=0.10:alpha=1{ov}"
        )

        if side == "left":
            settled_x = "46"
        elif side == "center":
            settled_x = "(main_w-overlay_w)/2"
        else:
            settled_x = "main_w-overlay_w-46"

        if layout is not None:
            settled_x = str(int(plan.width * layout[0]))
        y_expr = str(target_y)
        if animation == "fly":
            # New Shorts rhythm: enter FAST, finish the movement quickly, then
            # stay completely still for 1-2 seconds. No long floating/coasting.
            travel = 0.22
            p = f"max(0,min(1,(t-{start:.3f})/{travel:.3f}))"
            ease = f"(1-pow(1-({p}),3))"
            if side == "left":
                settled = 70
                x = (
                    f"-overlay_w+({ease})*"
                    f"(overlay_w+{settled})"
                )
            elif side == "right":
                settled = 70
                x = (
                    f"main_w+({ease})*"
                    f"(-overlay_w-{settled})"
                )
            else:
                # Center inserts rise in quickly and decelerate near the target.
                x = settled_x
                y_expr = (
                    f"main_h+({ease})*"
                    f"({target_y}-main_h)"
                )
        elif animation == "drop":
            x = settled_x
            settle = start + 0.15
            y_expr = (
                f"if(lt(t,{settle:.3f}),"
                f"-overlay_h+(t-{start:.3f})/0.15*({target_y}+overlay_h),"
                f"{target_y})"
            )
        else:
            x = settled_x

        filters.append(
            f"{current}{ov}overlay="
            f"x='{x}':y='{y_expr}':"
            f"enable='between(t,{start:.3f},{end:.3f})':"
            f"eof_action=pass:repeatlast=1:shortest=0{nxt}"
        )
        current = nxt

    filters.append(f"{current}trim=duration={base_duration:.3f},setpts=PTS-STARTPTS[vout]")
    cmd += [
        "-filter_complex", ";".join(filters),
        "-map", "[vout]", "-an",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-r", str(fx_fps),
        "-t", f"{base_duration:.3f}",
        str(output),
    ]
    _run(cmd)
    _assert_duration(output, base_duration, label="overlay video")


def _build_overlay_sfx(overlays: list[dict], duration: float, output: Path) -> None:
    """Create one fixed-duration SFX bed for all pop-ins.

    A silent full-length bed is always input #0, so amix duration=first makes
    the result exactly match the narration duration regardless of delayed SFX.
    """
    duration = max(0.05, float(duration))
    valid = [
        item for item in overlays
        if 0.0 <= float(item.get("start", 0.0)) < duration
    ]
    if not valid:
        _run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
            "-t", f"{duration:.3f}",
            "-c:a", "pcm_s16le", str(output),
        ])
        return

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
    ]
    filters: list[str] = []
    labels: list[str] = ["[0:a]"]

    for index, item in enumerate(valid, start=1):
        start = max(0.0, min(duration - 0.01, float(item.get("start", 0.0))))
        animation = str(item.get("animation") or "fly")
        delay = int(round(start * 1000))
        label = f"[s{index}]"

        if animation == "fly":
            cmd += [
                "-f", "lavfi", "-i",
                "anoisesrc=color=white:duration=0.18:sample_rate=48000",
            ]
            filters.append(
                f"[{index}:a]highpass=f=650,lowpass=f=5200,"
                f"afade=t=out:st=0.06:d=0.12,volume=0.11,"
                f"adelay={delay}|{delay}{label}"
            )
        else:
            cmd += [
                "-f", "lavfi", "-i",
                "sine=frequency=1550:duration=0.075:sample_rate=48000",
            ]
            filters.append(
                f"[{index}:a]afade=t=out:st=0.02:d=0.055,volume=0.085,"
                f"adelay={delay}|{delay}{label}"
            )
        labels.append(label)

    filters.append(
        "".join(labels)
        + f"amix=inputs={len(labels)}:duration=first:normalize=0,"
        + f"atrim=duration={duration:.3f},asetpts=PTS-STARTPTS[aout]"
    )

    cmd += [
        "-filter_complex", ";".join(filters),
        "-map", "[aout]",
        "-t", f"{duration:.3f}",
        "-c:a", "pcm_s16le",
        str(output),
    ]
    _run(cmd)
    _assert_duration(output, duration, label="SFX bed")


def _assert_duration(path: str | Path, expected: float, *, label: str) -> None:
    actual = _safe_probe_duration(path)
    tolerance = max(0.20, 2.0 / 30.0)
    if actual <= 0 or abs(actual - expected) > tolerance:
        raise RuntimeError(
            f"{label} duration mismatch: expected {expected:.3f}s, got {actual:.3f}s"
        )

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


def _write_ass(
    plan: ShotPlan,
    path: Path,
    *,
    editing_polish: bool = False,
    overlays: list[dict] | None = None,
) -> None:
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
    # Overlay value labels (e.g. "930 мл") are rendered with libass instead
    # of FFmpeg drawtext. This avoids Windows fontconfig failures while keeping
    # the same bold Shorts look used by the normal captions.
    for index, item in enumerate(overlays or []):
        effect_type = str(item.get("type") or "png")
        label = str(item.get("label") or "").strip()
        if not label:
            continue
        start = float(item.get("start", 0.0))
        end = float(item.get("end", start))
        if end <= start:
            continue

        side = str(item.get("position") or "right")
        size = str(item.get("size") or "large")
        animation = str(item.get("animation") or "pop")

        # Text-only effects should read like deliberate Shorts callouts.
        if effect_type == "text":
            x = int(plan.width * 0.50)
            y = int(plan.height * 0.16)
            fs = max(92, int(plan.width * (0.125 if size == "hero" else 0.105)))
            start_scale = 76 if animation == "pop" else 88
            safe = _ass_text_plain(label)
            events.append(
                f"Dialogue: 3,{_ass_time(start)},{_ass_time(end)},Default,,0,0,0,,"
                + r"{\an5\pos(" + str(x) + "," + str(y) + r")\fs" + str(fs)
                + r"\bord10\shad0\fscx" + str(start_scale) + r"\fscy" + str(start_scale)
                + r"\t(0,90,\fscx100\fscy100)\fad(2,8)}" + safe
            )
            continue

        x = int(plan.width * (0.25 if side == "left" else 0.75 if side == "right" else 0.50))
        y = int(plan.height * (0.30 if size == "hero" else 0.28))
        fs = max(68, int(plan.width * (0.088 if size == "hero" else 0.074)))
        safe = _ass_text_plain(label)
        events.append(
            f"Dialogue: 2,{_ass_time(start)},{_ass_time(end)},Default,,0,0,0,,"
            + r"{\an5\pos(" + str(x) + "," + str(y) + r")\fs" + str(fs)
            + r"\bord8\shad0}" + safe
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


def _ass_text_plain(text: str) -> str:
    return text.replace("{", "(").replace("}", ")").replace("\\", r"\\")


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
