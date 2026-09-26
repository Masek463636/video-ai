from pathlib import Path
import subprocess
from unittest.mock import patch
import pytest
from video_ai.models import Scene, ShotPlan
from video_ai.renderer import _render_scene, _probe_video_size

@pytest.mark.parametrize('polish', [False, True])
@pytest.mark.parametrize('width,height', [(400,100),(100,120)])
def test_large_panel_or_full_portrait(tmp_path,width,height,polish):
    source=tmp_path/'source.mp4'
    subprocess.run(['ffmpeg','-y','-v','error','-f','lavfi','-i',f'color=red:s={width}x{height}:r=10:d=1','-vf',f'drawbox=x=0:y=0:w={width//2}:h=ih:color=blue:t=fill','-c:v','libx264',str(source)],check=True)
    output=tmp_path/'out.mp4'
    plan=ShotPlan(Path('unused.wav'),[],width=90,height=160,fps=10)
    scene=Scene(0,1,'red',asset=str(source),asset_kind='video',focus_x=1 if width>height else 0.5)
    _render_scene(scene,1,plan,output,crf=18,editing_polish=polish,reference_framing=True)
    raw=subprocess.check_output(['ffmpeg','-v','error','-ss','0.5','-i',str(output),'-frames:v','1','-f','rawvideo','-pix_fmt','rgb24','-'])
    assert len(raw)==90*160*3
    # Ultrawide panel remains sharp near its bottom, covering > half the canvas.
    # Portraits remain sharp near the bottom of the entire canvas.
    y=75 if width>height else 145
    x=8 if width>height else 52
    pixel=raw[(y*90+x)*3:(y*90+x)*3+3]
    assert pixel[0]>180 and pixel[2]<70, list(pixel)


def test_display_size_honors_rotation_and_anamorphic_pixels():
    import json
    result=type('Result',(),{'stdout':json.dumps({'streams':[{'width':720,'height':480,'sample_aspect_ratio':'4:3','side_data_list':[{'rotation':90}]}]})})()
    with patch('video_ai.renderer.subprocess.run',return_value=result):
        assert _probe_video_size('unused')==(480,960)


def test_cli_reuses_overlays_without_regenerating(tmp_path):
    import json
    import sys
    from video_ai.cli import main
    from video_ai.io import save_shot_plan
    saved=tmp_path/'overlays.json'
    effects=[{'type':'text','start':0,'end':1,'label':'20 минут'}]
    saved.write_text(json.dumps({'overlays':effects}))
    plan=save_shot_plan(ShotPlan(tmp_path/'voice.wav',[Scene(0,1,'phone')]),tmp_path/'plan.json')
    output=tmp_path/'output.mp4'
    with patch.object(sys,'argv',['video-ai','render',str(plan),'-o',str(output),'--overlays-file',str(saved)]), patch('video_ai.cli.build_shorts_overlays',side_effect=AssertionError('API regeneration')), patch('video_ai.cli.render_plan',return_value=output) as render:
        main()
    assert render.call_args.kwargs['overlays']==effects
