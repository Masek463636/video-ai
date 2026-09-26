import json
import os
from pathlib import Path
import subprocess
import sys

from PIL import Image, ImageDraw
import pytest

from video_ai.models import Transcript, Word
from video_ai.story_render import render_composition, rerender_story, write_story_captions, composition_layout
from video_ai.transcript import save_transcript


def frame(path):
    raw=subprocess.check_output(['ffmpeg','-v','error','-ss','0.6','-i',str(path),'-frames:v','1',
                                 '-f','rawvideo','-pix_fmt','rgb24','-'])
    return Image.frombytes('RGB',(180,320),raw)


def beat(assets,kind='photo'):
    return {'kind':kind,'start':0,'end':1,'start_word':0,'end_word':1,
            'assets':assets,'element':None,'point_to':None}


def asset(path):
    return {'kind':'video' if path.suffix=='.mp4' else 'image','path':str(path),'label':'Пример'}


@pytest.mark.parametrize('video',[False,True])
def test_solo_vertical_media_keeps_all_corners_and_full_height(tmp_path,video):
    path=tmp_path/'portrait.png'
    im=Image.new('RGB',(180,320),'gray');draw=ImageDraw.Draw(im)
    draw.rectangle((0,0,35,35),fill='red')
    draw.rectangle((144,284,179,319),fill='blue');im.save(path)
    if video:
        target=tmp_path/'portrait.mp4'
        subprocess.run(['ffmpeg','-y','-v','error','-loop','1','-i',str(path),'-t','1',
                        '-pix_fmt','yuv420p',str(target)],check=True)
        path=target
    output=tmp_path/'result.mp4'
    render_composition(beat([asset(path)]),output,tmp_path,width=180,height=320,fps=15)
    result=frame(output)
    top=result.getpixel((10,10));bottom=result.getpixel((170,310))
    assert top[0]>200 and top[2]<40
    assert bottom[2]>200 and bottom[0]<40


@pytest.mark.parametrize('dimensions,axis', [((160,90),'stacked'),((90,160),'side')])
def test_comparison_objects_each_occupy_over_quarter_screen(tmp_path,dimensions,axis):
    assets=[]
    for color in ('red','blue'):
        path=tmp_path/(color+'.png');Image.new('RGB',dimensions,color).save(path);assets.append(asset(path))
    scene=beat(assets,'comparison')
    assert composition_layout(scene,180,320)['axis']==axis
    output=tmp_path/'comparison.mp4'
    render_composition(scene,output,tmp_path,width=180,height=320,fps=15)
    im=frame(output)
    for channel in (0,2):
        colored=[(x,y) for y in range(im.height) for x in range(im.width)
                 if im.getpixel((x,y))[channel]>180 and im.getpixel((x,y))[2-channel]<60]
        assert len(colored)>180*320*.25
        if axis=='stacked':
            assert max(x for x,y in colored)-min(x for x,y in colored)>160
        else:
            assert max(y for x,y in colored)-min(y for x,y in colored)>170
    captions=tmp_path/'captions.ass'
    write_story_captions(Transcript([Word(0,1,'Сравни.')],'ru'),[scene],captions,width=180,height=320)
    text=captions.read_text(encoding='utf-8-sig')
    assert '\\pos(90,272)' in text


def saved_story(tmp_path):
    work=tmp_path/'saved';work.mkdir()
    source=tmp_path/'portrait.png';Image.new('RGB',(90,160),'red').save(source)
    audio=tmp_path/'voice.wav'
    subprocess.run(['ffmpeg','-y','-v','error','-f','lavfi','-i','sine=duration=1',str(audio)],check=True)
    scene=beat([asset(source)])
    data={'audio':str(audio),'effects':False,'beats':[scene]}
    (work/'story.materialized.json').write_text(json.dumps(data))
    save_transcript(Transcript([Word(0,1,'Пример.')],'ru'),work/'transcript.json')
    return work,data


def test_rerender_cli_works_without_provider_keys(tmp_path):
    work,data=saved_story(tmp_path)
    output=tmp_path/'rerendered.mp4'
    env=dict(os.environ)
    for key in ('GEMINI_API_KEY','PEXELS_API_KEY','PIXABAY_API_KEY'):
        env.pop(key,None)
    result=subprocess.run([sys.executable,'-m','video_ai.cli','story-render',str(work),'-o',str(output)],
                          env=env,capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stderr
    assert '"reused_scenes": 1' in result.stdout
    info=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries',
                                          'format=duration:stream=codec_type','-of','json',str(output)]))
    assert abs(float(info['format']['duration'])-1)<.1
    assert {s['codec_type'] for s in info['streams']}=={'audio','video'}
    assert json.loads((work/'story.materialized.json').read_text())==data


def test_rerender_rejects_partial_checkpoint_before_render(tmp_path,monkeypatch):
    work,data=saved_story(tmp_path)
    save_transcript(Transcript([Word(0,.5,'Первое'),Word(.5,1,'второе.')],'ru'),work/'transcript.json')
    data['beats'][0]['end']=.5
    (work/'story.materialized.json').write_text(json.dumps(data))
    monkeypatch.setattr('video_ai.story_render.render_story',lambda *a,**kw:pytest.fail('Partial edit must not render'))
    with pytest.raises(ValueError,match='фрагмент'):
        rerender_story(work,tmp_path/'out.mp4')
