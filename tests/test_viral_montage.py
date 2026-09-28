from pathlib import Path
import subprocess
import json
import numpy as np
from video_ai.models import Scene, ShotPlan
from video_ai.renderer import render_plan
from video_ai.viral_montage import emoji_overlays, select_meme_shots


def test_primary_meme_replaces_every_pixel_and_preserves_audio_timeline(tmp_path):
    def ff(*args): subprocess.run(['ffmpeg','-y','-v','error',*args],check=True)
    base=tmp_path/'base.mp4';meme=tmp_path/'meme.mp4';voice=tmp_path/'voice.wav'
    ff('-f','lavfi','-i','color=blue:s=180x320:r=24:d=3',str(base))
    ff('-f','lavfi','-i','color=red:s=320x180:r=24:d=0.5',str(meme))
    ff('-f','lavfi','-i','sine=frequency=440:duration=3',str(voice))
    plan=ShotPlan(voice,[Scene(0,3,'stock',caption='Озвучка остаётся',motion_preset='snap_zoom')],width=180,height=320,fps=24)
    effects=[dict(type='sticker',asset=str(meme),source='local_reaction',reaction_kind='meme',reaction_verified=True,start=1,end=2)]
    out=render_plan(plan,tmp_path/'final.mp4',base_video=base,work_dir=tmp_path/'work',editing_style='viral',captions=False,overlays=effects)
    def rgb(t):
        b=subprocess.check_output(['ffmpeg','-v','error','-ss',str(t),'-i',str(out),'-frames:v','1','-f','rawvideo','-pix_fmt','rgb24','-'])
        return np.frombuffer(b,dtype=np.uint8).reshape(320,180,3).astype(int)
    for t in (.5,2.5):
        a=rgb(t);assert (a[:,:,2]>a[:,:,0]+100).mean()>.99
    for t in (1.05,1.8):
        a=rgb(t);assert (a[:,:,0]>a[:,:,2]+100).mean()>.99
    assert plan.scenes[0].motion_preset=='none'
    placed=json.loads((tmp_path/'work/overlays.placed.json').read_text())['overlays']
    assert placed==[]
    info=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(out)]))
    assert {s['codec_type'] for s in info['streams']}=={'audio','video'}
    assert abs(float(info['format']['duration'])-3)<.1


def test_only_emoji_overlays_and_none_over_meme(tmp_path):
    asset=tmp_path/'reaction.gif';asset.write_bytes(b'placeholder')
    common=dict(type='sticker',asset=str(asset),source='local_reaction',reaction_verified=True)
    effects=[dict(common,reaction_kind='meme',start=1,end=2),
             dict(common,reaction_kind='emoji',start=1.1,end=1.9),
             dict(common,reaction_kind='emoji',start=3,end=4)]
    shots=select_meme_shots(effects,5,24)
    assert len(shots)==1
    result=emoji_overlays(effects,5,shots,[(0,5)])
    assert len(result)==1 and result[0]['start']==3
    assert result[0]['layout_box'][2]==.32
