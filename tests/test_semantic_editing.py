from pathlib import Path
from unittest.mock import patch
import subprocess

import pytest

from video_ai.models import Scene, ShotPlan
from video_ai.material_v2 import apply_material_brain_v2
from video_ai.io import save_shot_plan, load_shot_plan, validate_shot_plan
from video_ai.moments import select_moments, window_starts
from video_ai.renderer import _render_scene, _same_visual
from video_ai.shorts_fx import build_shorts_overlays


def test_director_comparison_survives_keyword_rules_and_repair():
    scene = Scene(0, 2, 'hands comparing two milk carton labels',
                  caption='Берёшь молоко, а объём уже меньше',
                  visual_description='Compare quantity labels on two milk cartons')
    plan = ShotPlan(Path('voice.wav'), [scene])
    apply_material_brain_v2(plan)
    assert scene.query == 'hands comparing two milk carton labels'
    first = scene.search_queries[:]
    apply_material_brain_v2(plan)
    assert scene.search_queries == first
    assert scene.query == 'hands comparing two milk carton labels'


def test_empty_effect_plan_is_respected(tmp_path):
    class Client:
        def _generate_json(self, *args, **kwargs):
            return {'effects': []}
    plan = ShotPlan(Path('voice.wav'), [Scene(0, 24, 'milk', caption='930 мл молока')])
    with patch('video_ai.gemini_ai.get_gemini_client', return_value=Client()), \
         patch('video_ai.shorts_fx._local_effect_candidates', side_effect=AssertionError('unwanted fill')), \
         patch('video_ai.shorts_fx._local_sticker_effect_candidates', side_effect=AssertionError('unwanted sticker')):
        assert build_shorts_overlays(plan, tmp_path) == []


def test_offset_roundtrip_and_old_plan(tmp_path):
    plan = ShotPlan(Path('voice.wav'), [Scene(0, 2, 'dog', source_start=3.5)])
    file = save_shot_plan(plan, tmp_path / 'plan.json')
    assert load_shot_plan(file).scenes[0].source_start == 3.5
    import json
    raw = json.loads(file.read_text())
    del raw['scenes'][0]['source_start']
    file.write_text(json.dumps(raw))
    assert load_shot_plan(file).scenes[0].source_start == 0
    for invalid in [-1, float('nan'), float('inf')]:
        plan.scenes[0].source_start = invalid
        with pytest.raises(ValueError, match='source_start'):
            validate_shot_plan(plan)


def test_moment_windows_bounded():
    assert window_starts(1, 2) == [0]
    starts = window_starts(90, 3)
    assert len(starts) == 8
    assert starts[0] == 0 and starts[-1] == 87


@pytest.fixture
def colored_source(tmp_path):
    import shutil
    if not shutil.which('ffmpeg'):
        pytest.skip('FFmpeg not available')
    source = tmp_path / 'source.mp4'
    subprocess.run(['ffmpeg', '-y', '-v', 'error',
        '-f', 'lavfi', '-i', 'color=red:s=160x90:r=10:d=2',
        '-f', 'lavfi', '-i', 'color=blue:s=160x90:r=10:d=2',
        '-filter_complex', '[0:v][1:v]concat=n=2:v=1:a=0[v]', '-map', '[v]',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(source)], check=True)
    return source


@pytest.mark.parametrize('framing', [False, True])
def test_real_render_uses_selected_window(colored_source, tmp_path, framing):
    scene = Scene(0, 1, 'blue', asset=str(colored_source), asset_kind='video', source_start=2.5)
    plan = ShotPlan(Path('voice.wav'), [scene], width=90, height=160, fps=10)
    output = tmp_path / 'render.mp4'
    _render_scene(scene, 1, plan, output, crf=20, reference_framing=framing)
    raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(output),
        '-frames:v', '1', '-vf', 'scale=1:1', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'])
    assert raw[2] > 150 and raw[0] < 60, list(raw)  # blue, not opening red
    other = Scene(1, 2, 'red', asset=str(colored_source), asset_kind='video')
    assert not _same_visual(scene, other)


def test_selector_accepts_only_confident_valid_window(colored_source, tmp_path):
    class Client:
        def __init__(self, window, fit):
            self.window, self.fit = window, fit
        def _generate_json(self, parts, **kwargs):
            assert len([p for p in parts if 'inline_data' in p]) == 12
            return {'window': self.window, 'fit': self.fit, 'reason': 'blue visible'}
    scene = Scene(0, 1, 'blue', asset=str(colored_source), asset_kind='video')
    plan = ShotPlan(tmp_path / 'absent.wav', [scene])
    report = select_moments(plan, tmp_path / 'moments', client=Client(3, 95))
    assert report[0]['status'] == 'selected'
    assert scene.source_start == 3
    for index, fit in [(99, 90), (None, 95), (0, 40), (True, 90)]:
        report = select_moments(plan, tmp_path / 'moments', client=Client(index, fit))
        assert report[0]['status'] == 'no_confident_match'
        assert scene.source_start == 3
