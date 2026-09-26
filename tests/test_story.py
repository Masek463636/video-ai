import json
from pathlib import Path
import subprocess

import pytest
from PIL import Image

from video_ai.models import Transcript, Word
from video_ai.story import create_story, validate_story
from video_ai.story_media import download_complete, identity_supported, index_pack, resolve_media
from video_ai.story_render import render_composition, render_story
from video_ai.transcript import save_transcript


def narration():
    return Transcript([Word(i*.5,(i+1)*.5,text) for i,text in enumerate(['Один','дом','богатый.','Другой','дом','бедный.'])], 'ru')


def subject():
    return {'intent':'house front view','queries':['house front view'],'entity':'','aliases':[],'label':'Дом'}


def test_plan_rejects_missing_words_and_invented_people():
    t=narration();packs={'memes':[],'elements':[]}
    raw={'beats':[{'start_word':0,'end_word':6,'kind':'comparison','subjects':[subject(),subject()]}]}
    assert validate_story(raw,t,3,packs)['beats'][0]['end']==3
    raw['beats'][0]['end_word']=5
    with pytest.raises(ValueError,match='конец'):
        validate_story(raw,t,3,packs)
    raw['beats'][0]['end_word']=6
    raw['beats'][0]['subjects'][0]['entity']='Никола Тесла'
    with pytest.raises(ValueError,match='Имя'):
        validate_story(raw,t,3,packs)


def test_identity_requires_source_description():
    assert identity_supported(['Тэцу Накамура','Tetsu Nakamura'],'Photograph of Tetsu Nakamura in Afghanistan')
    assert not identity_supported(['Tetsu Nakamura'],'Photograph of a Japanese doctor')
    assert not identity_supported(['Tetsu Nakamura'],'Tetsu NakamuraX')


def test_pack_cache_excludes_its_own_previews(tmp_path):
    pack=tmp_path/'memes';pack.mkdir();Image.new('RGB',(80,80),'red').save(pack/'ожидание.png')
    class Client:
        calls=0
        def _generate_json(self,parts,**kwargs):
            self.calls+=1
            row=json.loads(parts[1]['text'])
            return {'assets':[{'id':row['id'],'description':'red test image','safe':True}]}
    client=Client()
    first=index_pack(pack,pack/'.video-ai-index',client)
    second=index_pack(pack,pack/'.video-ai-index',client)
    assert len(first)==len(second)==1
    assert client.calls==1


def test_wrong_person_is_rejected_before_visual_judgement(tmp_path,monkeypatch):
    monkeypatch.setattr('video_ai.story_media.candidates',lambda *a,**k:[{'download_url':'https://example.org/photo.jpg','title':'A doctor','description':'An unknown person'}])
    wanted=dict(subject(),entity='Тэцу Накамура',aliases=['Тэцу Накамура','Tetsu Nakamura'])
    class Client:
        def _generate_json(self,*a,**k):
            raise AssertionError('Wrong identity must not reach visual scoring')
    with pytest.raises(RuntimeError,match='Не найден'):
        resolve_media(wanted,tmp_path,Client())


def test_interrupted_download_is_not_reused(tmp_path,monkeypatch):
    target=tmp_path/'media.mp4'
    def interrupted(url,path):
        path.write_bytes(b'incomplete')
        raise OSError('connection interrupted')
    monkeypatch.setattr('video_ai.story_media._download',interrupted)
    with pytest.raises(OSError):
        download_complete('https://example.org/media.mp4',target)
    assert not target.exists()
    assert not list(tmp_path.glob('*.part'))
    monkeypatch.setattr('video_ai.story_media._download',lambda url,path:path.write_bytes(b'complete'))
    download_complete('https://example.org/media.mp4',target)
    assert target.read_bytes()==b'complete'


def test_full_story_with_controlled_providers_and_real_render(tmp_path,monkeypatch):
    audio=tmp_path/'voice.wav'
    subprocess.run(['ffmpeg','-y','-v','error','-f','lavfi','-i','sine=frequency=220:duration=3',str(audio)],check=True)
    paths=[]
    for color in ['red','blue','green']:
        p=tmp_path/(color+'.png');Image.new('RGB',(90,120),color).save(p);paths.append(p)
    def item(color):return dict(subject(),queries=[color],label=color)
    plan={'beats':[{'start_word':0,'end_word':3,'kind':'comparison','subjects':[item('red'),item('blue')],'point_to':'left'},
                   {'start_word':3,'end_word':6,'kind':'photo','subjects':[item('green')]}]}
    class Client:
        def _generate_json(self,parts,**kwargs):
            return plan if 'You edit a vertical' in parts[0]['text'] else {'choice':0,'fit':90,'reason':'Fixture colour matches'}
    def candidates(query,**kwargs):
        return [{'kind':'image','download_url':f'https://example.org/{query}.png','title':query,'description':query,'page_url':f'https://example.org/{query}','source':'fixture'}]
    monkeypatch.setattr('video_ai.story_media.candidates',candidates)
    monkeypatch.setattr('video_ai.story_media._download',lambda url,path:path.write_bytes((tmp_path/Path(url).name).read_bytes()))
    monkeypatch.setattr('video_ai.story_render.render_story',lambda *a,**kw:render_story(*a,**kw,width=180,height=320,fps=15))
    transcript=tmp_path/'transcript.json';save_transcript(narration(),transcript)
    output=tmp_path/'result.mp4'
    work=tmp_path/'work'
    create_story(audio,output,work,transcript_path=transcript,client=Client())
    assert len(json.loads((work/'sources.json').read_text()))==3
    # Fractional cut at 1.5s/15fps must total 45 frames, not 23+23.
    frame_counts=[int(json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=nb_frames','-of','json',str(p)]))['streams'][0]['nb_frames']) for p in sorted((work/'render').glob('scene-*.mp4'))]
    assert frame_counts==[23,22]
    data=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration:stream=codec_type','-of','json',str(output)]))
    assert abs(float(data['format']['duration'])-3)<.15
    assert {s['codec_type'] for s in data['streams']}=={'audio','video'}
    # Both halves of the comparison survive compositing; the last scene isn't a frozen old frame.
    for timestamp,x,channel in [(.8,40,0),(.8,140,2),(2.7,90,1)]:
        raw=subprocess.check_output(['ffmpeg','-v','error','-ss',str(timestamp),'-i',str(output),'-frames:v','1','-vf',f'crop=10:10:{x}:60,scale=1:1','-f','rawvideo','-pix_fmt','rgb24','-'])
        assert raw[channel]>max(raw[c] for c in range(3) if c!=channel)+40


def test_animated_meme_loops_and_transparent_element(tmp_path):
    meme=tmp_path/'reaction.gif'
    Image.new('RGB',(90,120),'red').save(meme,save_all=True,append_images=[Image.new('RGB',(90,120),'blue')],duration=200,loop=0)
    element=tmp_path/'element.png'
    im=Image.new('RGBA',(50,50),(0,0,0,0))
    im.paste((255,255,255,255),(10,10,40,40));im.save(element)
    output=tmp_path/'animated.mp4'
    beat={'kind':'meme','start':0,'end':1.2,'assets':[{'path':str(meme),'kind':'video'}],
          'element':{'path':str(element),'kind':'image'},'point_to':None}
    render_composition(beat,output,tmp_path,width=180,height=320,fps=15)
    def pixel(t,x,y):
        return subprocess.check_output(['ffmpeg','-v','error','-ss',str(t),'-i',str(output),'-frames:v','1','-vf',f'crop=6:6:{x}:{y},scale=1:1','-f','rawvideo','-pix_fmt','rgb24','-'])
    first=pixel(.1,30,40);later=pixel(.7,30,40)
    assert first[0]>first[2]+80 and later[2]>later[0]+80
    assert min(pixel(.7,130,125))>210
