import copy
import json
import subprocess
from pathlib import Path
import pytest
from PIL import Image
from video_ai.models import Transcript, Word
from video_ai.fifth_render import render_fifth, materialize, _emoji_for
from video_ai.fifth_catalog import fingerprint


def fixture(tmp):
    audio=tmp/'voice.wav'
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','sine=frequency=440:duration=2',str(audio)],check=True)
    stock=tmp/'stock.mp4'
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','color=c=red:s=160x90:r=15:d=3','-c:v','mpeg4',str(stock)],check=True)
    meme=tmp/'meme.png';Image.new('RGB',(90,160),'blue').save(meme)
    transcript=Transcript([Word(0,.9,'Начало'),Word(1,2,'Конец')],'ru')
    beats=[]
    for i,source in enumerate(('stock','meme')):
        beats.append(dict(first_word=i,last_word=i,role='setup' if i==0 else 'punchline',source=source,
                          visual_goal='test',continuity='test',reason='test',queries=['test'],
                          camera='gentle_push',emoji=None,sound='soft_hit' if i==1 else 'none'))
    plan=dict(premise='test',payoff='test',beats=beats)
    selections={f'beat-{i:03}':{'selected':dict(id=f'c{i}',path=str(path),source=source,start=0,full_url='')} for i,(path,source) in enumerate(((stock,'stock'),(meme,'meme')))}
    return plan,dict(schema='fifth-selection-v1',selections=selections,unresolved=[]),transcript,audio


def pixel(path,seconds):
    data=subprocess.check_output(['ffmpeg','-v','error','-ss',str(seconds),'-i',str(path),'-frames:v','1','-vf','crop=2:2:30:30','-f','rawvideo','-pix_fmt','rgb24','-'])
    return tuple(data[:3])


def test_end_to_end_standalone_meme_and_cached_rerender(tmp_path):
    plan,selection,transcript,audio=fixture(tmp_path)
    result=render_fifth(plan,selection,transcript,audio,tmp_path/'out.mp4',tmp_path/'work',cache=tmp_path/'cache',width=180,height=320,fps=15)
    assert result['ok'] and result['gemini_calls']==0 and result['reused_scenes']==0
    red=pixel(tmp_path/'out.mp4',.3);blue=pixel(tmp_path/'out.mp4',1.4)
    assert red[0]>180 and red[2]<60
    assert blue[2]>180 and blue[0]<60
    data=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration:stream=codec_type','-of','json',str(tmp_path/'out.mp4')]))
    assert abs(float(data['format']['duration'])-2)<.08
    assert {s['codec_type'] for s in data['streams']}=={'audio','video'}
    result=render_fifth(plan,selection,transcript,audio,tmp_path/'out.mp4',tmp_path/'work',cache=tmp_path/'cache',width=180,height=320,fps=15)
    assert result['reused_scenes']==2
    selection['selections']['beat-000']['selected']['start']=.2
    result=render_fifth(plan,selection,transcript,audio,tmp_path/'out.mp4',tmp_path/'work',cache=tmp_path/'cache',width=180,height=320,fps=15)
    assert result['reused_scenes']==1
    captions=(tmp_path/'work/captions.ass').read_text(encoding='utf-8-sig')
    assert 'Начало' in captions and 'Конец' in captions


def test_partial_selection_rejected_before_render(tmp_path):
    plan,selection,transcript,audio=fixture(tmp_path)
    selection['selections'].pop('beat-001')
    with pytest.raises(ValueError,match='всех сцен'):
        materialize(plan,selection,transcript,audio,tmp_path/'cache')


def test_short_stock_refused_instead_of_freeze_or_loop(tmp_path):
    plan,selection,transcript,audio=fixture(tmp_path)
    selection['selections']['beat-000']['selected']['start']=2.8
    with pytest.raises(ValueError,match='короче'):
        materialize(plan,selection,transcript,audio,tmp_path/'cache')


def test_emoji_requires_verified_exact_tag_transparency_and_current_file(tmp_path):
    path=tmp_path/'emoji.png';Image.new('RGBA',(20,20),(0,0,0,0)).save(path)
    im=Image.open(path);im.putpixel((10,10),(255,0,0,255));im.save(path)
    entry=dict(path=str(path),id=fingerprint(path),description=dict(kind='emoji',readable=True,tags=['surprised']))
    beat=dict(source='stock',emoji=dict(query='surprised'))
    assert _emoji_for(beat,{'assets':[entry]}) is not None
    assert _emoji_for(dict(beat,source='meme'),{'assets':[entry]}) is None
    assert _emoji_for(dict(beat,emoji={'query':'angry'}),{'assets':[entry]}) is None
    Image.new('RGBA',(20,20),'red').save(path)
    assert _emoji_for(beat,{'assets':[entry]}) is None


def test_short_meme_loops_and_emoji_overlay_renders(tmp_path):
    plan,selection,transcript,audio=fixture(tmp_path)
    short=tmp_path/'short.mp4'
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','testsrc2=s=90x160:r=15:d=0.4','-c:v','mpeg4',str(short)],check=True)
    selection['selections']['beat-001']['selected']['path']=str(short)
    path=tmp_path/'emoji.png';im=Image.new('RGBA',(40,40),(0,0,0,0))
    from PIL import ImageDraw
    ImageDraw.Draw(im).ellipse((2,2,38,38),fill=(0,255,0,255));im.save(path)
    entry=dict(path=str(path),id=fingerprint(path),description=dict(kind='emoji',readable=True,tags=['surprised']))
    plan['beats'][0]['emoji']=dict(word=0,query='surprised',reason='test')
    result=render_fifth(plan,selection,transcript,audio,tmp_path/'out.mp4',tmp_path/'work',cache=tmp_path/'cache',catalog={'assets':[entry]},width=180,height=320,fps=15)
    assert result['warnings']==[]
    # Green emoji is visible during its anchor window, below the large stock panel.
    data=subprocess.check_output(['ffmpeg','-v','error','-ss','0.5','-i',str(tmp_path/'out.mp4'),'-frames:v','1','-vf','crop=2:2:90:218','-f','rawvideo','-pix_fmt','rgb24','-'])
    assert data[1]>160 and data[0]<80
    # The late meme frame continues animation instead of cloning its last frame.
    def frame(t):
        return subprocess.check_output(['ffmpeg','-v','error','-ss',str(t),'-i',str(tmp_path/'out.mp4'),'-frames:v','1','-vf','crop=180:180:0:0','-f','rawvideo','-pix_fmt','rgb24','-'])
    assert frame(1.5)!=frame(1.8)
