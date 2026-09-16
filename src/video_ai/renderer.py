from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

from .models import Scene, ShotPlan


def render_plan(plan: ShotPlan, output: str | Path, *, work_dir: str | Path | None = None, captions: bool = True, crf: int = 20) -> Path:
    _require("ffmpeg"); _require("ffprobe")
    output=Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    audio_duration=_probe_duration(plan.audio)
    if audio_duration<=0: audio_duration=max(scene.end for scene in plan.scenes)
    if work_dir is None:
        temp=tempfile.TemporaryDirectory(prefix="video-ai-"); root=Path(temp.name)
    else:
        temp=None; root=Path(work_dir); root.mkdir(parents=True,exist_ok=True)
    try:
        clips_dir=root/"clips"; clips_dir.mkdir(parents=True,exist_ok=True); clips=[]
        for index,(scene,start,end) in enumerate(_display_ranges(plan,audio_duration)):
            clip=clips_dir/f"scene_{index:03d}.mp4"; _render_scene(scene,end-start,plan,clip,crf=crf); clips.append(clip)
        concat_file=root/"concat.txt"
        concat_file.write_text("\n".join("file '"+str(p.resolve()).replace("'","'\\''")+"'" for p in clips)+"\n",encoding="utf-8")
        base=root/"base.mp4"
        _run(["ffmpeg","-y","-hide_banner","-loglevel","error","-f","concat","-safe","0","-i",str(concat_file),"-c","copy",str(base)])
        final_filter=[]
        if captions:
            ass=root/"captions.ass"; _write_ass(plan,ass); final_filter=["-vf",f"ass='{_filter_path(ass)}'"]
        _run(["ffmpeg","-y","-hide_banner","-loglevel","error","-i",str(base),"-i",str(plan.audio),*final_filter,"-map","0:v:0","-map","1:a:0","-c:v","libx264" if captions else "copy",*([] if not captions else ["-preset","medium","-crf",str(crf),"-pix_fmt","yuv420p"]),"-c:a","aac","-b:a","192k","-movflags","+faststart","-shortest",str(output)])
        return output
    finally:
        if temp is not None: temp.cleanup()


def _display_ranges(plan: ShotPlan, audio_duration: float) -> list[tuple[Scene,float,float]]:
    ranges=[]
    for index,scene in enumerate(plan.scenes):
        start=0.0 if index==0 else max(0.0,scene.start)
        end=max(start+0.05,plan.scenes[index+1].start) if index+1<len(plan.scenes) else max(start+0.05,audio_duration)
        ranges.append((scene,start,end))
    return ranges


def _render_scene(scene: Scene, duration: float, plan: ShotPlan, output: Path, *, crf: int) -> None:
    duration=max(0.05,duration)
    common=["-an","-c:v","libx264","-preset","veryfast","-crf",str(crf),"-pix_fmt","yuv420p","-r",str(plan.fps),"-t",f"{duration:.3f}",str(output)]
    if scene.asset_kind=="image" and scene.asset and Path(scene.asset).exists():
        _run(["ffmpeg","-y","-hide_banner","-loglevel","error","-loop","1","-framerate",str(plan.fps),"-i",str(scene.asset),"-vf",_image_filter(scene,plan,duration),*common]); return
    if scene.asset_kind=="video" and scene.asset and Path(scene.asset).exists():
        _run(["ffmpeg","-y","-hide_banner","-loglevel","error","-stream_loop","-1","-i",str(scene.asset),"-vf",_video_filter(scene,plan),*common]); return
    _run(["ffmpeg","-y","-hide_banner","-loglevel","error","-f","lavfi","-i",f"color=c=0x101014:s={plan.width}x{plan.height}:r={plan.fps}:d={duration:.3f}",*common])


def _image_filter(scene: Scene, plan: ShotPlan, duration: float) -> str:
    """Strong AE-like ease: nearly still at both ends, fast through the middle."""
    w,h,fps=plan.width,plan.height,plan.fps
    frames=max(2,int(math.ceil(duration*fps))); denominator=max(1,frames-1)
    overscan=1.14; big_w=int(math.ceil(w*overscan/2)*2); big_h=int(math.ceil(h*overscan/2)*2)
    fx=_clamp_focus(scene.focus_x); fy=_clamp_focus(scene.focus_y)
    base=f"scale={big_w}:{big_h}:force_original_aspect_ratio=increase,crop={big_w}:{big_h}:x='{_focus_expr('iw','ow',fx)}':y='{_focus_expr('ih','oh',fy)}'"

    # smootherstep(t)=6t^5-15t^4+10t^3. Stronger than cosine/smoothstep near endpoints.
    t_on=f"min(max(on/{denominator},0),1)"
    ease_on=f"((6*pow({t_on},5))-(15*pow({t_on},4))+(10*pow({t_on},3)))"
    t_n=f"min(max(n/{denominator},0),1)"
    ease_n=f"((6*pow({t_n},5))-(15*pow({t_n},4))+(10*pow({t_n},3)))"

    if scene.motion=="zoom_in":
        return base+","+f"zoompan=z='1.005+0.105*{ease_on}':x='max(0,min(iw-iw/zoom,{fx:.5f}*iw-iw/zoom/2))':y='max(0,min(ih-ih/zoom,{fy:.5f}*ih-ih/zoom/2))':d=1:s={w}x{h}:fps={fps}"
    if scene.motion=="zoom_out":
        return base+","+f"zoompan=z='1.110-0.105*{ease_on}':x='max(0,min(iw-iw/zoom,{fx:.5f}*iw-iw/zoom/2))':y='max(0,min(ih-ih/zoom,{fy:.5f}*ih-ih/zoom/2))':d=1:s={w}x{h}:fps={fps}"
    if scene.motion in {"pan_right","pan_left"}:
        direction="1" if scene.motion=="pan_right" else "-1"
        x_pan=f"max(0,min(iw-ow,{fx:.5f}*iw-ow/2+({direction})*({ease_n}-0.5)*(iw-ow)*1.15))"
        return base+f",crop={w}:{h}:x='{x_pan}':y='{_focus_expr('ih','oh',fy)}',fps={fps}"
    return base+f",crop={w}:{h}:x='{_focus_expr('iw','ow',fx)}':y='{_focus_expr('ih','oh',fy)}',fps={fps}"


def _video_filter(scene: Scene, plan: ShotPlan) -> str:
    fx=_clamp_focus(scene.focus_x); fy=_clamp_focus(scene.focus_y)
    return f"scale={plan.width}:{plan.height}:force_original_aspect_ratio=increase,crop={plan.width}:{plan.height}:x='{_focus_expr('iw','ow',fx)}':y='{_focus_expr('ih','oh',fy)}',fps={plan.fps}"


def _clamp_focus(value: float | None) -> float:
    return 0.5 if value is None else max(0.0,min(1.0,float(value)))


def _focus_expr(inner: str, outer: str, focus: float) -> str:
    return f"max(0,min({inner}-{outer},{focus:.5f}*{inner}-{outer}/2))"


def _write_ass(plan: ShotPlan, path: Path) -> None:
    header=f"""[Script Info]\nScriptType: v4.00+\nPlayResX: {plan.width}\nPlayResY: {plan.height}\nScaledBorderAndShadow: yes\n\n[V4+ Styles]\nFormat: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding\nStyle: Default,Arial,{max(48,int(plan.width*0.066))},&H00FFFFFF,&H00FFFFFF,&H00101010,&H50000000,-1,0,0,0,100,100,0,0,1,5,0,2,80,80,{int(plan.height*0.26)},1\n\n[Events]\nFormat: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n"""
    events=[]
    for scene in plan.scenes:
        if scene.caption: events.append(f"Dialogue: 0,{_ass_time(scene.start)},{_ass_time(scene.end)},Default,,0,0,0,,{_ass_text(scene.caption)}")
    path.write_text(header+"\n".join(events)+"\n",encoding="utf-8-sig")


def _ass_text(text: str) -> str:
    words=text.replace("{","(").replace("}",")").split()
    if len(words)<=5: return " ".join(words)
    mid=min(5,max(2,math.ceil(len(words)/2))); return " ".join(words[:mid])+r"\N"+" ".join(words[mid:])


def _ass_time(seconds: float) -> str:
    cs=int(round(max(0.0,seconds)*100)); h,rem=divmod(cs,360000); m,rem=divmod(rem,6000); s,cent=divmod(rem,100); return f"{h}:{m:02d}:{s:02d}.{cent:02d}"


def _filter_path(path: Path) -> str:
    text=path.resolve().as_posix().replace("'",r"\'")
    if len(text)>=2 and text[1]==":": text=text[0]+r"\:"+text[2:]
    return text


def _probe_duration(path: str | Path) -> float:
    completed=subprocess.run(["ffprobe","-v","error","-show_entries","format=duration","-of","json",str(path)],check=True,capture_output=True,text=True)
    return float(json.loads(completed.stdout).get("format",{}).get("duration") or 0.0)


def _require(name: str) -> None:
    if shutil.which(name) is None: raise RuntimeError(f"{name} is required and was not found on PATH")


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd,check=True)
