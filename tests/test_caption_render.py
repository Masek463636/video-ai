"""Offline FFmpeg integration check, not a stock/Gemini end-to-end benchmark."""
import hashlib
import json
import shutil
import subprocess

import pytest

from video_ai.io import load_shot_plan, save_shot_plan
from video_ai.material_brain import find_duplicate_scenes
from video_ai.models import Scene, ShotPlan, Transcript, Word
from video_ai.qc import failed_scene_indexes, inspect_plan
from video_ai.renderer import render_plan
from video_ai.transcript import attach_caption_timings


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg integration")
def test_caption_timing_preserves_visual_render_and_qc(tmp_path):
    def ffmpeg(*args):
        subprocess.run(["ffmpeg", "-y", "-v", "error", *map(str, args)], check=True, capture_output=True)

    audio = tmp_path / "audio.wav"
    video = tmp_path / "video.mp4"
    still = tmp_path / "still.ppm"
    ffmpeg("-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono", "-t", "4", audio)
    ffmpeg("-f", "lavfi", "-i", "testsrc2=s=720x1280:r=30:d=2", "-c:v", "libx264", "-preset", "ultrafast", video)
    still.write_bytes(b"P6\n180 320\n255\n" + bytes(
        channel for y in range(320) for x in range(180)
        for channel in (60 + x, 80 + y % 150, 180 if x < 90 else 70)))
    plan = ShotPlan(audio, [
        Scene(0, 2, "test motion", asset=str(video), asset_kind="video", source_mode="stock_video", caption="Собака внезапно уснула."),
        Scene(2, 4, "test still", asset=str(still), asset_kind="image", caption="Проснулась утром.", focus_source="test", motion_preset="slow_push"),
    ], width=720, height=1280, director_source="donor_gemini")
    plan = load_shot_plan(save_shot_plan(plan, tmp_path / "plan.json"))
    render_plan(plan, tmp_path / "before.mp4", work_dir=tmp_path / "before")
    words = [Word(0, .2, "Собака"), Word(.2, 1.5, "внезапно"), Word(1.5, 2, "уснула."),
             Word(2, 2.4, "Проснулась"), Word(2.8, 4, "утром.")]
    attach_caption_timings(plan, Transcript(words))
    plan = load_shot_plan(save_shot_plan(plan, tmp_path / "timed-plan.json"))
    render_plan(plan, tmp_path / "after.mp4", work_dir=tmp_path / "after")
    assert hashlib.sha256((tmp_path / "before/base.mp4").read_bytes()).digest() == hashlib.sha256((tmp_path / "after/base.mp4").read_bytes()).digest()
    assert not failed_scene_indexes(inspect_plan(plan))
    assert find_duplicate_scenes(plan)[0] == []
    assert all(s.asset and s.asset_kind != "blank" for s in plan.scenes)
    result = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(tmp_path / "after.mp4"),
                             "-vf", "blackdetect=d=0.01:pix_th=0.10", "-an", "-f", "null", "-"], capture_output=True, text=True, check=True)
    assert "black_start:" not in result.stderr
    probe = json.loads(subprocess.check_output(["ffprobe", "-v", "quiet", "-show_streams", "-of", "json", str(tmp_path / "after.mp4")]))
    stream = next(s for s in probe["streams"] if s["codec_type"] == "video")
    assert (stream["width"], stream["height"], stream["r_frame_rate"]) == (720, 1280, "30/1")
    assert abs(float(stream["duration"]) - 4) < 1/30
