"""Mixed stock metadata must not reset the effects timeline."""
import shutil
import subprocess
from pathlib import Path
import pytest
from video_ai.models import ShotPlan
from video_ai.renderer import _apply_overlays


def test_overlay_preserves_mixed_metadata_scene_order(tmp_path):
    if not shutil.which('ffmpeg'):
        pytest.skip('FFmpeg unavailable')
    clips = []
    for i, (color, space) in enumerate([('red', 'bt709'), ('green', 'smpte170m'), ('blue', 'bt709')]):
        clip = tmp_path / f'{i}.mp4'
        subprocess.run(['ffmpeg', '-y', '-v', 'error', '-f', 'lavfi', '-i',
                        f'color={color}:s=96x160:r=30:d=1', '-c:v', 'libx264',
                        '-colorspace', space, str(clip)], check=True)
        clips.append(clip)
    listing = tmp_path / 'concat.txt'
    listing.write_text(''.join(f"file '{p.as_posix()}'\n" for p in clips))
    base = tmp_path / 'base.mp4'
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-f', 'concat', '-safe', '0',
                    '-i', str(listing), '-c', 'copy', str(base)], check=True)
    output = tmp_path / 'fx.mp4'
    plan = ShotPlan(Path('unused.wav'), [], width=96, height=160, fps=30)
    # Text effects exercise base timeline filters without hiding sample pixels.
    _apply_overlays(base, [{'type': 'text', 'start': 0, 'end': .5}], plan, output, crf=18)
    for timestamp, channel in [(.5, 0), (1.5, 1), (2.5, 2)]:
        pixel = subprocess.check_output(['ffmpeg', '-v', 'error', '-ss', str(timestamp),
            '-i', str(output), '-frames:v', '1', '-vf', 'scale=1:1',
            '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'])
        assert len(pixel) == 3
        assert pixel[channel] > max(pixel[j] for j in range(3) if j != channel) + 60
