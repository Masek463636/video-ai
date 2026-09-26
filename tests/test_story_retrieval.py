import hashlib
import json

import pytest

from video_ai import story_media as media
from video_ai.assets import AssetCandidate
from video_ai.openverse import OpenverseImage
from video_ai.story import validate_story
from video_ai.models import Transcript, Word


def subject():
    return {'intent':'Show someone putting on pants in a rush.',
            'queries':['person putting on jeans in a hurry','person dressing quickly in bedroom'],
            'entity':'','aliases':[],'label':'Ещё дома'}


def candidate(url='https://example.org/dressing.mp4', *, kind='video'):
    return {'download_url':url,'kind':kind,'source':'fixture','title':'getting dressed',
            'description':'a person getting dressed at home','page_url':'https://example.org/dressing'}


@pytest.fixture
def previews(monkeypatch):
    downloaded=[]
    def download(url,path):
        downloaded.append(url)
        path.write_bytes(b'fixture')
    monkeypatch.setattr(media,'download_complete',download)
    monkeypatch.setattr(media,'preview_parts',lambda *a,**kw:([{'text':'fixture frame'}],3))
    return downloaded


def test_photo_shortlist_includes_each_provider_and_names_skip_stock(monkeypatch):
    stock_calls=[]
    def stock(*args,**kw):
        stock_calls.append(args)
        return [dict(candidate(f'https://example.org/{source}-{i}.jpg',kind='image'),source=source)
                for source in ('pexels','pixabay') for i in range(6)]
    monkeypatch.setattr(media,'search_stock_images',stock)
    monkeypatch.setattr(media,'search_commons',lambda *a,**kw:[
        AssetCandidate('House','page','https://example.org/commons.jpg','image/jpeg',100,100,500,'image')])
    monkeypatch.setattr(media,'search_openverse',lambda *a,**kw:[
        OpenverseImage('House','page','https://example.org/openverse.jpg',100,100)])
    rows=media.candidates('house',video=False)
    assert [r['source'] for r in rows[:4]]==['pexels','pixabay','commons','openverse']
    rows=media.candidates('named building',video=False,named=True)
    assert {r['source'] for r in rows}=={'commons','openverse'}
    assert len(stock_calls)==1


def test_rewrite_recovers_video_without_changing_intent_or_reusing_comparison_side(tmp_path,monkeypatch,previews):
    wanted=subject()
    narration='Ты ждёшь его, а он ещё дома одевается.'
    left='https://example.org/left-side.mp4'
    searches=[]
    def candidates(query,**kw):
        searches.append((query,kw['video']))
        return [candidate(left),candidate()] if query=='getting dressed' and kw['video'] else []
    monkeypatch.setattr(media,'candidates',candidates)
    class Client:
        rewrites=0
        def _generate_json(self,parts,**kw):
            if parts[0]['text'].startswith('Rewrite failed'):
                self.rewrites+=1
                return [{'queries':['getting dressed','dressing'],'media_type':'video'}]
            context=parts[0]['text']
            assert wanted['intent'] in context and narration in context
            assert not any(left in p.get('text','') for p in parts)
            return {'choice':0,'fit':86,'reason':'The frames show the dressing action'}
    client=Client();used={left}
    chosen=media.resolve_media(wanted,tmp_path,client,used=used,exclude_urls={left},narration=narration)
    assert chosen['kind']=='video' and chosen['query']=='getting dressed'
    assert chosen['intent']==wanted['intent']
    assert len(searches)==5 and client.rewrites==1
    assert left not in previews and chosen['download_url'] in used
    report=json.loads(next(tmp_path.glob('retrieval-*.json')).read_text())
    assert report['subject']==wanted
    assert report['attempts'][-1]['phase']=='rewrite'
    assert report['attempts'][-1]['outcome']=='selected'


def test_all_bad_matches_still_fail_with_bounded_search_and_diagnostics(tmp_path,monkeypatch,previews):
    wanted=dict(subject(),queries=['first','second','third'])
    searches=[]
    def candidates(query,**kw):
        searches.append((query,kw['video']))
        key=hashlib.sha256(repr(searches[-1]).encode()).hexdigest()
        return [candidate(f'https://example.org/{key}.mp4')]
    monkeypatch.setattr(media,'candidates',candidates)
    class Client:
        calls=0
        def _generate_json(self,parts,**kw):
            self.calls+=1
            if parts[0]['text'].startswith('Rewrite failed'):
                return {'queries':['fourth','fifth'],'media_type':'video'}
            return {'choice':0,'fit':64.9,'reason':'Main action is not visible'}
    client=Client()
    with pytest.raises(RuntimeError,match='Автопоиск исчерпан'):
        media.resolve_media(wanted,tmp_path,client)
    assert len(searches)==len(set(searches))==10
    assert client.calls==11  # Ten visual judgements, one rewrite, no recursion.
    report=json.loads(next(tmp_path.glob('retrieval-*.json')).read_text())
    assert len(report['attempts'])==10
    assert all(a['outcome']=='rejected' for a in report['attempts'])


def test_named_subject_never_switches_to_generic_video_or_query_rewrite(tmp_path,monkeypatch):
    searches=[]
    def candidates(query,**kw):
        searches.append(kw)
        return [candidate()]  # Missing the requested identity in source metadata.
    monkeypatch.setattr(media,'candidates',candidates)
    class Client:
        def _generate_json(self,*a,**kw):
            pytest.fail('Unidentified subject must not reach model selection or generic rewrite')
    wanted=dict(subject(),entity='Тэцу Накамура',aliases=['Тэцу Накамура','Tetsu Nakamura'],media_type='video')
    with pytest.raises(RuntimeError,match='Не найден'):
        media.resolve_media(wanted,tmp_path,Client(),video=True)
    assert len(searches)==2 and all(q['named'] and not q['video'] for q in searches)


def test_query_rewrite_does_not_restart_exhausted_api_retries(tmp_path,monkeypatch):
    monkeypatch.setattr(media,'candidates',lambda *a,**kw:[])
    class Client:
        calls=0
        def _generate_json(self,*a,**kw):
            self.calls+=1
            raise RuntimeError('API quota exhausted')
    client=Client()
    with pytest.raises(RuntimeError,match='quota'):
        media.resolve_media(subject(),tmp_path,client)
    assert client.calls==1


def test_legacy_plans_default_to_auto_and_action_comparisons_can_prefer_video():
    transcript=Transcript([Word(0,1,'Одевается.')],'ru')
    raw={'beats':[{'start_word':0,'end_word':1,'kind':'comparison',
                   'subjects':[subject(),dict(subject(),media_type='video')]}]}
    planned=validate_story(raw,transcript,1,{'memes':[],'elements':[]})
    assert [s['media_type'] for s in planned['beats'][0]['subjects']]==['auto','video']
    raw['beats'][0]['subjects'][1]['media_type']='unsupported'
    with pytest.raises(ValueError,match='Тип материала'):
        validate_story(raw,transcript,1,{'memes':[],'elements':[]})
