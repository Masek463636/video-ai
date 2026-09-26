import json
from pathlib import Path
import subprocess

import pytest
from PIL import Image

from video_ai.models import Transcript, Word
from video_ai.story import create_story, validate_story
from video_ai.story_media import download_complete, identity_supported, index_pack, resolve_media
from video_ai.story_render import render_composition, render_story
from video_ai.story_render import select_window
from video_ai.story_json import StoryResponseError, generate_validated, selection_response
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


@pytest.mark.parametrize('mode', ['object_reply', 'array_reply', 'video_recovery'])
def test_full_story_with_controlled_providers_and_real_render(tmp_path,monkeypatch,mode):
    array_reply = mode == 'array_reply'
    audio=tmp_path/'voice.wav'
    subprocess.run(['ffmpeg','-y','-v','error','-f','lavfi','-i','sine=frequency=220:duration=3',str(audio)],check=True)
    paths=[]
    for color in ['red','blue','green']:
        p=tmp_path/(color+'.png');Image.new('RGB',(90,120),color).save(p);paths.append(p)
    if mode == 'video_recovery':
        subprocess.run(['ffmpeg','-y','-v','error','-f','lavfi','-i','color=blue:s=90x120:r=30:d=1',
                        '-f','lavfi','-i','color=cyan:s=90x120:r=30:d=0.5',
                        '-filter_complex','[0:v][1:v]concat=n=2:v=1:a=0[v]','-map','[v]',str(tmp_path/'blue.mp4')],check=True)
    def item(color):return dict(subject(),queries=[color],label=color)
    plan={'beats':[{'start_word':0,'end_word':3,'kind':'comparison','subjects':[item('red'),item('blue')],'point_to':'left'},
                   {'start_word':3,'end_word':6,'kind':'photo','subjects':[item('green')]}]}
    class Client:
        def _generate_json(self,parts,**kwargs):
            if 'You edit a vertical' in parts[0]['text']:
                return plan['beats'] if array_reply else plan
            result = {'choice':0,'fit':90,'reason':'Fixture colour matches'}
            return [result] if array_reply else result
    def candidates(query,**kwargs):
        is_video = mode == 'video_recovery' and query == 'blue'
        if is_video and not kwargs['video']:
            return []
        ext = 'mp4' if is_video else 'png'
        return [{'kind':'video' if is_video else 'image','download_url':f'https://example.org/{query}.{ext}','title':query,'description':query,'page_url':f'https://example.org/{query}','source':'fixture'}]
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
    if mode == 'video_recovery':
        # The recovered video moves inside the comparison; it isn't a frozen preview.
        raw=subprocess.check_output(['ffmpeg','-v','error','-ss','1.3','-i',str(output),'-frames:v','1',
                                     '-vf','crop=10:10:140:60,scale=1:1','-f','rawvideo','-pix_fmt','rgb24','-'])
        assert raw[1]>180 and raw[2]>180 and raw[0]<60
        beats=json.loads((work/'story.materialized.json').read_text())['beats']
        assert [a['kind'] for a in beats[0]['assets']]==['image','video']


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


def test_pack_bare_array_keeps_cache_and_respects_new_file_budget(tmp_path,monkeypatch):
    pack=tmp_path/'memes';pack.mkdir()
    for i in range(14):
        (pack/f'meme-{i:02}.png').write_bytes(b'fixture')
    monkeypatch.setattr('video_ai.story_media.preview_parts',lambda *a,**kw:([{'text':'fixture preview'}],0))
    class Client:
        calls=0
        def _generate_json(self,parts,**kwargs):
            self.calls+=1
            files=[json.loads(p['text']) for p in parts if p.get('text','').startswith('{"id":')]
            rows=[{'id':p['id'],'description':'verified test reaction','safe':True} for p in files]
            return {'assets':rows} if self.calls==1 else rows
    client=Client()
    first=index_pack(pack,pack/'.video-ai-index',client,new_file_limit=6)
    second=index_pack(pack,pack/'.video-ai-index',client,new_file_limit=6)
    cached=index_pack(pack,pack/'.video-ai-index',client,new_file_limit=0)
    assert [len(first),len(second),len(cached)]==[6,12,12]
    assert client.calls==2
    assert {a['id'] for a in first}<={a['id'] for a in second}


def test_ambiguous_selection_is_corrected_once_not_silently_chosen():
    class Client:
        calls=0
        def _generate_json(self,*args,**kwargs):
            self.calls+=1
            return [{'choice':0,'fit':95},{'choice':1,'fit':98}] if self.calls==1 else {'choice':1,'fit':90}
    client=Client()
    result=generate_validated(client,[{'text':'Choose'}],temperature=0,
                              validate=lambda raw:selection_response(raw,'choice',range(2)),stage='test')
    assert result['choice']==1 and client.calls==2


@pytest.mark.parametrize('reply', [None, 'not an object', [], {'choice':True,'fit':99},
                                  {'choice':5,'fit':99}, {'choice':0,'fit':float('nan')},
                                  {'choice':0,'fit':101}, {'choice':0}, {'fit':99}])
def test_malformed_selection_has_bounded_repair(reply):
    class Client:
        calls=0
        def _generate_json(self,*a,**kw):
            self.calls+=1
            return reply
    client=Client()
    with pytest.raises(StoryResponseError,match='дважды'):
        generate_validated(client,[],temperature=0,validate=lambda raw:selection_response(raw,'choice',range(2)),stage='test')
    assert client.calls==2


def test_format_repair_does_not_repeat_api_quota_failure():
    class Client:
        calls=0
        def _generate_json(self,*a,**kw):
            self.calls+=1
            raise RuntimeError('Gemini request failed: HTTP 429')
    client=Client()
    with pytest.raises(RuntimeError,match='429'):
        generate_validated(client,[],temperature=0,validate=lambda raw:selection_response(raw,'choice',range(2)),stage='test')
    assert client.calls==1


def test_window_selection_accepts_array_reply(tmp_path,monkeypatch):
    monkeypatch.setattr('video_ai.story_render._safe_probe_duration',lambda path:8)
    def preview(command,**kw):
        Path(command[-1]).write_bytes(b'fixture')
        return subprocess.CompletedProcess(command,0)
    monkeypatch.setattr('video_ai.story_render.subprocess.run',preview)
    class Client:
        def _generate_json(self,*a,**kw):
            return [{'window':2,'fit':90}]
    assert select_window({'path':'fixture.mp4'},2,'reaction',tmp_path,Client())==4
