from __future__ import annotations

import math
import re
import zlib
from dataclasses import replace
from pathlib import Path

from .models import Scene, ShotPlan


_NUMBER_WORDS = {
    "миллион","миллиона","миллионов","тысяча","тысяч","процент","процентов",
    "million","thousand","percent",
}


def _token(word: str) -> str:
    return word.strip(" \t\r\n,.;:!?—–-()[]«»\"'").casefold()


def _is_number_word(text: str) -> bool:
    clean = _token(text)
    return bool(re.search(r"\d", clean)) or clean in _NUMBER_WORDS


def _caption_anchor(scene: Scene, plan: ShotPlan) -> tuple[int, int]:
    seed = zlib.crc32(
        f"{scene.start:.3f}:{scene.caption or ''}:{scene.query or ''}".encode(
            "utf-8", errors="replace"
        )
    )
    x_shift = (-0.035, 0.0, 0.035)[seed % 3]
    x = int(plan.width * max(0.36, min(0.64, 0.5 + x_shift)))
    y = int(plan.height * 0.25) if scene.focus_y is not None and scene.focus_y > 0.58 else int(plan.height * 0.72)
    return x, y


def premium_caption_events_v2(scene: Scene, plan: ShotPlan) -> list[str]:
    from .renderer import _ass_time

    words = [
        word for word in scene.caption_words
        if math.isfinite(word.start) and math.isfinite(word.end) and word.end > word.start
    ]
    if not words:
        return []

    highlights = {_token(value) for value in scene.premium_highlights if _token(value)}
    pace = getattr(scene, "pace_class", "normal") or "normal"
    window = 2 if pace in {"fast", "reveal"} else 3
    x, y = _caption_anchor(scene, plan)
    events: list[str] = []

    for index, word in enumerate(words):
        start = max(float(scene.start), float(word.start))
        end = min(float(scene.end), float(word.end))
        if end <= start:
            continue
        raw = word.text.replace("{", "(").replace("}", ")")
        if _is_number_word(raw):
            text = (
                r"{\c&H00D7FF&\3c&H000000&\bord11\fscx78\fscy78"
                r"\t(0,55,\fscx150\fscy150)\t(55,135,\fscx108\fscy108)}"
                + raw
            )
            events.append(
                f"Dialogue: 2,{_ass_time(start)},{_ass_time(end)},Default,,0,0,0,,"
                + r"{\an5\pos(" + f"{x},{int(plan.height * .46)}" + r")\fad(0,20)}" + text
            )
            continue

        left = max(0, index - window // 2)
        right = min(len(words), left + window)
        if right - left < window:
            left = max(0, right - window)

        parts: list[str] = []
        for pos in range(left, right):
            token_text = words[pos].text.replace("{", "(").replace("}", ")")
            if pos == index:
                if _token(token_text) in highlights:
                    tags = (
                        r"{\c&H0040FF&\3c&H000000&\bord10\fscx80\fscy80"
                        r"\t(0,50,\fscx132\fscy132)\t(50,115,\fscx100\fscy100)}"
                    )
                else:
                    tags = (
                        r"{\c&H00FFFF&\3c&H000000&\bord9\fscx86\fscy86"
                        r"\t(0,48,\fscx121\fscy121)\t(48,108,\fscx100\fscy100)}"
                    )
            else:
                tags = r"{\c&HFFFFFF&\3c&H000000&\bord8\fscx91\fscy91}"
            parts.append(tags + token_text)

        events.append(
            f"Dialogue: 1,{_ass_time(start)},{_ass_time(end)},Default,,0,0,0,,"
            + r"{\an5\pos(" + f"{x},{y}" + r")\fad(0,18)}"
            + " ".join(parts)
        )
    return events


def _font_path() -> str | None:
    candidates = [
        Path(r"C:\Windows\Fonts\impact.ttf"),
        Path(r"C:\Windows\Fonts\arialbd.ttf"),
        Path(r"C:\Windows\Fonts\seguisb.ttf"),
    ]
    return next((str(path) for path in candidates if path.is_file()), None)


def _subject_alpha_box(cutout: Path) -> tuple[float, float, float, float] | None:
    try:
        from PIL import Image
        image = Image.open(cutout).convert("RGBA")
        box = image.getchannel("A").getbbox()
        if not box:
            return None
        left, top, right, bottom = box
        width, height = image.size
        return left / width, top / height, (right-left) / width, (bottom-top) / height
    except Exception:
        return None


def _make_text_layer(text: str, plan: ShotPlan, target: Path, *, cutout: Path | None = None, seed_text: str = "") -> Path | None:
    if not text.strip():
        return None
    try:
        from PIL import Image, ImageDraw, ImageFont
        image = Image.new("RGBA", (plan.width, plan.height), (0,0,0,0))
        draw = ImageDraw.Draw(image)
        font_path = _font_path()
        font_size = max(105, int(plan.width * .16))
        font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
        clean = text.upper()
        bbox = draw.textbbox((0,0), clean, font=font, stroke_width=8)
        text_w, text_h = bbox[2]-bbox[0], bbox[3]-bbox[1]
        while text_w > plan.width * .90 and font_path and font_size > 58:
            font_size -= 6
            font = ImageFont.truetype(font_path, font_size)
            bbox = draw.textbbox((0,0), clean, font=font, stroke_width=8)
            text_w, text_h = bbox[2]-bbox[0], bbox[3]-bbox[1]

        subject = _subject_alpha_box(cutout) if cutout else None
        seed = zlib.crc32(seed_text.encode("utf-8", errors="replace"))
        jitter = (-0.04, 0.0, 0.04)[seed % 3]
        if subject:
            sx, sy, sw, sh = subject
            center_x = (sx + sw/2) * plan.width
            x = int(center_x - text_w/2 + plan.width*jitter)
            y = int(sy*plan.height - text_h*.64)
        else:
            x = int((plan.width-text_w)/2 + plan.width*jitter)
            y = int(plan.height*.27)
        x = max(30, min(plan.width-text_w-30, x))
        y = max(int(plan.height*.05), min(int(plan.height*.72), y))
        draw.text(
            (x,y), clean, font=font, fill=(255,226,35,255),
            stroke_width=max(5,int(font_size*.055)), stroke_fill=(0,0,0,255)
        )
        image.save(target)
        return target
    except Exception as exc:
        print(f"[premium-v2] text-behind fallback: {type(exc).__name__}", flush=True)
        return None


def _overlay_full_frame(base: Path, overlay: Path, output: Path, duration: float, plan: ShotPlan, *, crf: int) -> None:
    from . import renderer as base_renderer
    base_renderer._run([
        "ffmpeg","-y","-hide_banner","-loglevel","error","-i",str(base),
        "-loop","1","-framerate",str(plan.fps),"-t",f"{duration:.3f}","-i",str(overlay),
        "-filter_complex",f"[1:v]scale={plan.width}:{plan.height},format=rgba[ov];[0:v][ov]overlay=0:0:shortest=1[v]",
        "-map","[v]","-an","-c:v","libx264","-preset","veryfast","-crf",str(crf),
        "-pix_fmt","yuv420p","-r",str(plan.fps),"-t",f"{duration:.3f}",str(output)
    ])


def _hero_number(scene: Scene) -> str | None:
    for word in scene.caption_words:
        if _is_number_word(word.text):
            return word.text
    match = re.search(r"\b\d[\d\s.,%$€₴]*\b", scene.caption or "")
    return match.group(0).strip() if match else None


def _make_number_layer(text: str, plan: ShotPlan, target: Path) -> Path | None:
    try:
        from PIL import Image, ImageDraw, ImageFont
        image = Image.new("RGBA",(plan.width,plan.height),(0,0,0,0))
        draw = ImageDraw.Draw(image)
        font_path = _font_path()
        size = max(150,int(plan.width*.25))
        font = ImageFont.truetype(font_path,size) if font_path else ImageFont.load_default()
        clean = text.upper()
        bbox = draw.textbbox((0,0),clean,font=font,stroke_width=10)
        width = bbox[2]-bbox[0]
        while width > plan.width*.90 and font_path and size > 80:
            size -= 8
            font = ImageFont.truetype(font_path,size)
            bbox = draw.textbbox((0,0),clean,font=font,stroke_width=10)
            width = bbox[2]-bbox[0]
        height = bbox[3]-bbox[1]
        draw.text(
            (int((plan.width-width)/2), int(plan.height*.36-height/2)),
            clean,font=font,fill=(255,215,25,255),stroke_width=10,stroke_fill=(0,0,0,255)
        )
        image.save(target)
        return target
    except Exception:
        return None


def _make_cards_layer(scene: Scene, plan: ShotPlan, target: Path) -> Path | None:
    try:
        from PIL import Image, ImageDraw, ImageFont
        words = [word.text for word in scene.caption_words] or (scene.caption or "").split()
        if len(words) < 6:
            return None
        cut1=max(2,len(words)//3); cut2=max(4,2*len(words)//3)
        chunks=[" ".join(words[:cut1])," ".join(words[cut1:cut2])," ".join(words[cut2:])]
        image=Image.new("RGBA",(plan.width,plan.height),(0,0,0,0))
        draw=ImageDraw.Draw(image)
        font_path=_font_path()
        font=ImageFont.truetype(font_path,max(46,int(plan.width*.052))) if font_path else ImageFont.load_default()
        card_w=int(plan.width*.78); card_h=int(plan.height*.10); start_y=int(plan.height*.18); gap=int(plan.height*.035)
        for index,chunk in enumerate([c for c in chunks if c.strip()][:3]):
            x=int((plan.width-card_w)/2 + ((index%2)*2-1)*20); y=start_y+index*(card_h+gap)
            draw.rounded_rectangle((x,y,x+card_w,y+card_h),radius=28,fill=(10,10,12,215),outline=(255,255,255,220),width=4)
            draw.text((x+26,y+20),chunk[:42],font=font,fill=(255,255,255,255))
        image.save(target)
        return target
    except Exception:
        return None


def _render_split(scene: Scene, duration: float, plan: ShotPlan, output: Path, *, crf: int) -> None:
    from . import renderer as base_renderer
    primary=Path(scene.asset or ""); secondary=Path(scene.secondary_asset or "")
    layout=getattr(scene,"split_layout","50_50") or "50_50"
    cmd=["ffmpeg","-y","-hide_banner","-loglevel","error"]
    if scene.asset_kind=="image":
        cmd += ["-loop","1","-framerate",str(plan.fps),"-i",str(primary)]
    else:
        cmd += ["-stream_loop","-1"]
        if scene.source_start: cmd += ["-ss",f"{scene.source_start:.3f}"]
        cmd += ["-i",str(primary)]
    if scene.secondary_asset_kind=="video":
        cmd += ["-stream_loop","-1","-i",str(secondary)]
    else:
        cmd += ["-loop","1","-framerate",str(plan.fps),"-i",str(secondary)]
    w,h,fps=plan.width,plan.height,plan.fps
    if layout=="left_right":
        lw=max(2,(w//2)//2*2); rw=w-lw
        filt=f"[0:v]scale={lw}:{h}:force_original_aspect_ratio=increase,crop={lw}:{h},fps={fps}[a];[1:v]scale={rw}:{h}:force_original_aspect_ratio=increase,crop={rw}:{h},fps={fps}[b];[a][b]hstack=inputs=2,drawbox=x={lw-3}:y=0:w=6:h={h}:color=white@0.95:t=fill[v]"
    elif layout=="60_40":
        th=max(2,int(h*.60)//2*2); bh=h-th
        filt=f"[0:v]scale={w}:{th}:force_original_aspect_ratio=increase,crop={w}:{th},fps={fps}[a];[1:v]scale={w}:{bh}:force_original_aspect_ratio=increase,crop={w}:{bh},fps={fps}[b];[a][b]vstack=inputs=2,drawbox=x=0:y={th-3}:w={w}:h=6:color=white@0.95:t=fill[v]"
    elif layout=="pip":
        pw=max(2,int(w*.44)//2*2); ph=max(2,int(h*.28)//2*2)
        filt=f"[0:v]scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},fps={fps}[base];[1:v]scale={pw}:{ph}:force_original_aspect_ratio=increase,crop={pw}:{ph},pad=iw+8:ih+8:4:4:white[pip];[base][pip]overlay=W-w-42:70:shortest=1[v]"
    elif layout=="diagonal":
        filt=f"[0:v]scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},fps={fps}[a];[1:v]scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},fps={fps}[b];[a][b]blend=all_expr='if(gte(X/W+Y/H,1),A,B)'[v]"
    else:
        hh=max(2,(h//2)//2*2); bh=h-hh
        filt=f"[0:v]scale={w}:{hh}:force_original_aspect_ratio=increase,crop={w}:{hh},fps={fps}[a];[1:v]scale={w}:{bh}:force_original_aspect_ratio=increase,crop={w}:{bh},fps={fps}[b];[a][b]vstack=inputs=2,drawbox=x=0:y={hh-3}:w={w}:h=6:color=white@0.95:t=fill[v]"
    cmd += ["-filter_complex",filt,"-map","[v]","-an","-c:v","libx264","-preset","veryfast","-crf",str(crf),"-pix_fmt","yuv420p","-r",str(fps),"-t",f"{duration:.3f}",str(output)]
    base_renderer._run(cmd)


def _render_layered(scene: Scene, duration: float, plan: ShotPlan, output: Path, *, crf: int) -> None:
    from . import renderer as base_renderer
    asset=Path(scene.asset or ""); foreground=Path(scene.premium_foreground or "")
    text_layer=None
    if scene.premium_layout=="text_behind" and scene.premium_text:
        text_layer=_make_text_layer(scene.premium_text,plan,output.with_suffix(".premium_v2_text.png"),cutout=foreground,seed_text=scene.caption or "")
    bg_scene=replace(scene,premium_layout="clean",motion_preset="micro_push")
    bg_filter=base_renderer._image_filter(bg_scene,plan,duration,editing_polish=True)
    if scene.premium_layout=="blur_background":
        bg_filter += ",gblur=sigma=18:steps=2"
    fx=base_renderer._clamp_focus(scene.focus_x); fy=base_renderer._clamp_focus(scene.focus_y)
    fw=max(2,int(plan.width*1.09)//2*2); fh=max(2,int(plan.height*1.09)//2*2)
    fg_filter=f"scale={plan.width}:{plan.height}:force_original_aspect_ratio=increase,crop={plan.width}:{plan.height}:x='{base_renderer._focus_expr('iw','ow',fx)}':y='{base_renderer._focus_expr('ih','oh',fy)}',format=rgba,scale={fw}:{fh},crop={plan.width}:{plan.height}:x='(iw-ow)/2':y='(ih-oh)/2',fps={plan.fps}"
    cmd=["ffmpeg","-y","-hide_banner","-loglevel","error","-loop","1","-framerate",str(plan.fps),"-i",str(asset),"-loop","1","-framerate",str(plan.fps),"-i",str(foreground)]
    filters=[f"[0:v]{bg_filter}[bg]",f"[1:v]{fg_filter}[fg0]","[fg0]split=2[fg][shadow0]","[shadow0]colorchannelmixer=rr=0:gg=0:bb=0:aa=.32,gblur=sigma=10[shadow]"]
    current="[bg]"
    if text_layer is not None:
        cmd += ["-loop","1","-framerate",str(plan.fps),"-i",str(text_layer)]
        filters += [f"[2:v]scale={plan.width}:{plan.height},format=rgba[txt]","[bg][txt]overlay=0:0:shortest=1[mid0]"]
        current="[mid0]"
    filters += [f"{current}[shadow]overlay=8:16:shortest=1[mid1]","[mid1][fg]overlay=0:0:shortest=1[v]"]
    cmd += ["-filter_complex",";".join(filters),"-map","[v]","-an","-c:v","libx264","-preset","veryfast","-crf",str(crf),"-pix_fmt","yuv420p","-r",str(plan.fps),"-t",f"{duration:.3f}",str(output)]
    base_renderer._run(cmd)


def _render_scene_v2(original, scene: Scene, duration: float, plan: ShotPlan, output: Path, *, crf: int, editing_polish: bool=False, reference_framing: bool=False, editing_style: str="premium") -> None:
    from . import renderer as base_renderer
    layout=str(scene.premium_layout or "clean")
    if layout in {"split_screen","before_after"} and scene.asset and scene.secondary_asset and Path(scene.asset).is_file() and Path(scene.secondary_asset).is_file():
        _render_split(scene,duration,plan,output,crf=crf); return
    if layout in {"parallax","text_behind","blur_background"} and scene.asset_kind=="image" and scene.asset and scene.premium_foreground and Path(scene.asset).is_file() and Path(scene.premium_foreground).is_file():
        try:
            _render_layered(scene,duration,plan,output,crf=crf); return
        except Exception as exc:
            print(f"[premium-v2] layered render fallback: {type(exc).__name__}",flush=True)

    clean=replace(scene,premium_layout="clean")
    if layout=="focus_zoom":
        clean=replace(clean,motion_preset="dramatic_push")
    if layout in {"clean","reaction","arrow","circle","png_cutout","focus_zoom"}:
        original(clean,duration,plan,output,crf=crf,editing_polish=True,reference_framing=reference_framing,editing_style="premium"); return

    temp=output.with_suffix(".premium_v2_base.mp4")
    original(clean,duration,plan,temp,crf=crf,editing_polish=True,reference_framing=reference_framing,editing_style="premium")
    try:
        if layout=="freeze_frame" and scene.asset_kind=="video":
            moving=max(.18,duration*.66); hold=max(.05,duration-moving)
            base_renderer._run(["ffmpeg","-y","-hide_banner","-loglevel","error","-i",str(temp),"-vf",f"trim=duration={moving:.3f},setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop_duration={hold:.3f}","-an","-c:v","libx264","-preset","veryfast","-crf",str(crf),"-pix_fmt","yuv420p","-r",str(plan.fps),"-t",f"{duration:.3f}",str(output)]); return
        if layout=="spotlight":
            base_renderer._run(["ffmpeg","-y","-hide_banner","-loglevel","error","-i",str(temp),"-vf","vignette=PI/5:eval=frame","-an","-c:v","libx264","-preset","veryfast","-crf",str(crf),"-pix_fmt","yuv420p","-r",str(plan.fps),"-t",f"{duration:.3f}",str(output)]); return
        if layout=="big_number":
            number=_hero_number(scene); layer=_make_number_layer(number,plan,output.with_suffix(".premium_v2_number.png")) if number else None
            if layer: _overlay_full_frame(temp,layer,output,duration,plan,crf=crf); return
        if layout=="stacked_cards":
            layer=_make_cards_layer(scene,plan,output.with_suffix(".premium_v2_cards.png"))
            if layer: _overlay_full_frame(temp,layer,output,duration,plan,crf=crf); return
        temp.replace(output)
    finally:
        temp.unlink(missing_ok=True)


def render_premium_v2(plan: ShotPlan, output: str | Path, **kwargs) -> Path:
    from . import renderer as base_renderer
    original_scene=base_renderer._render_scene
    original_captions=base_renderer._premium_caption_events

    def patched(scene,duration,render_plan,target,*,crf,editing_polish=False,reference_framing=False,editing_style="classic"):
        return _render_scene_v2(original_scene,scene,duration,render_plan,target,crf=crf,editing_polish=editing_polish,reference_framing=reference_framing,editing_style=editing_style)

    base_renderer._render_scene=patched
    base_renderer._premium_caption_events=premium_caption_events_v2
    try:
        kwargs["editing_style"]="premium"
        return base_renderer.render_plan(plan,output,**kwargs)
    finally:
        base_renderer._render_scene=original_scene
        base_renderer._premium_caption_events=original_captions
