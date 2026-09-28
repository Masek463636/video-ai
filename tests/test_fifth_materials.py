import json
import shutil
import subprocess
from unittest.mock import Mock
import pytest
from video_ai import fifth_catalog as catalog
from video_ai.fifth_session import FifthSession
from video_ai.fifth_materials import choose_batches, validate_choices, window_preview


def description(key):
    return dict(id=key,kind='meme',readable=True,description='Person sleeping',emotion='tired',tags=['sleep'],avoid='active running')


def session(tmp, responses):
    client=Mock(models=['test'],last_model='test')
    client._generate_json.side_effect=responses
    return FifthSession(client,tmp/'cache',tmp/'metrics.json'),client


def test_catalog_rename_change_and_delete(tmp_path,monkeypatch):
    root=tmp_path/'memes';root.mkdir();path=root/'one.jpg';path.write_bytes(b'first')
    monkeypatch.setattr(catalog,'preview_parts',lambda *a,**k:([{'text':'preview'}],1.))
    first=catalog.fingerprint(path)
    sess,client=session(tmp_path,[{'assets':[description(first)]}])
    result=catalog.describe_pack([root],tmp_path/'assets',sess)
    assert result['ready']==1
    moved=root/'renamed.jpg';path.rename(moved)
    assert catalog.describe_pack([root],tmp_path/'assets',sess)['ready']==1
    assert client._generate_json.call_count==1
    moved.write_bytes(b'changed')
    result=catalog.describe_pack([root],tmp_path/'assets',sess,max_new=0)
    assert result['ready']==0 and result['pending']==1
    moved.unlink()
    assert catalog.describe_pack([root],tmp_path/'assets',sess)['assets']==[]


def test_catalog_new_file_budget(tmp_path,monkeypatch):
    root=tmp_path/'memes';root.mkdir()
    for i in range(3):(root/f'{i}.jpg').write_bytes(str(i).encode())
    monkeypatch.setattr(catalog,'preview_parts',lambda *a,**k:([{'text':'preview'}],1.))
    sess,c=session(tmp_path,[{'assets':[description(catalog.fingerprint(root/'0.jpg'))]}])
    report=catalog.describe_pack([root],tmp_path/'assets',sess,max_new=1)
    assert (report['ready'],report['pending'])==(1,2)
    assert c._generate_json.call_count==1


def fixtures():
    beats=[dict(id=f'b{i}') for i in range(4)]
    candidates={f'b{i}':[dict(id=f'c{i}',source='meme',path=f'{i}.mp4',preview={'parts':[{'text':f'frame{i}'}]})] for i in range(4)}
    choices=[dict(beat=f'b{i}',candidate=f'c{i}',alternatives=[],reason='Visible action') for i in range(4)]
    return dict(premise='test',payoff='end',beats=beats),candidates,choices


def test_neighbor_batches_and_cached_resume(tmp_path):
    plan,candidates,rows=fixtures()
    sess,c=session(tmp_path,[{'choices':rows[:2]},{'choices':rows[2:]}])
    first=choose_batches(plan,candidates,sess,tmp_path/'one.json')
    second=choose_batches(plan,candidates,sess,tmp_path/'two.json')
    assert first==second and len(first['selections'])==4
    assert c._generate_json.call_count==2 and sess.stats['cache_hits']==2
    assert 'preview' not in first['selections']['b0']['selected']


def test_failure_preserves_completed_batch_and_stops_requests(tmp_path):
    plan,candidates,rows=fixtures()
    sess,c=session(tmp_path,[{'choices':rows[:1]},RuntimeError('provider unavailable')])
    report=choose_batches(plan,candidates,sess,tmp_path/'selection.json',batch_size=1)
    assert list(report['selections'])==['b0']
    assert report['unresolved']==['b1','b2','b3']
    assert c._generate_json.call_count==2
    assert json.loads((tmp_path/'selection.json').read_text())==report


@pytest.mark.parametrize('mode',['foreign','invented','previous','bad_type'])
def test_invalid_candidates_rejected(mode):
    plan,candidates,rows=fixtures();rows=rows[:1];used=()
    if mode=='foreign':rows[0]['candidate']='c1'
    if mode=='invented':rows[0]['candidate']='made-up'
    if mode=='previous':used=('c0',)
    if mode=='bad_type':rows[0]['beat']=[]
    with pytest.raises(ValueError):validate_choices({'choices':rows},plan['beats'][:1],candidates,used=used)


def test_missing_sources_stay_unresolved_without_api(tmp_path):
    plan,_,_=fixtures();sess,c=session(tmp_path,[])
    report=choose_batches(plan,{},sess,tmp_path/'out.json')
    assert len(report['unresolved'])==4 and c._generate_json.call_count==0


@pytest.mark.skipif(not shutil.which('ffmpeg'),reason='FFmpeg unavailable')
def test_real_window_preview_and_cache(tmp_path,monkeypatch):
    video=tmp_path/'clip.mp4'
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','testsrc2=size=320x240:rate=10','-t','2','-c:v','mpeg4',str(video)],check=True)
    key,data=window_preview(video,.5,1.,tmp_path/'media')
    assert len(data['parts'])==3
    def fail(*a,**k):raise AssertionError('cached preview must not rerun FFmpeg')
    monkeypatch.setattr('video_ai.composition_review.frames',fail)
    assert window_preview(video,.5,1.,tmp_path/'media')==(key,data)


def test_stock_preparation_shortlists_locally_without_gemini(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from video_ai import fifth_materials as material
    items=[SimpleNamespace(page_url=f'page{i}',duration=8,width=640,height=360,
                           preview_url=f'preview{i}',download_url=f'full{i}') for i in range(4)]
    monkeypatch.setattr(material,'search_stock_videos',lambda *a,**k:items)
    def download(url,path):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(url)
    monkeypatch.setattr(material,'download_complete',download)
    monkeypatch.setattr(material,'window_preview',lambda path,start,duration,cache:(path.stem+str(start),dict(parts=[{'text':'actual window'}],image=str(path))))
    ranker=Mock();ranker.score_images.side_effect=lambda prompt,paths:list(range(len(paths)))
    beat=dict(id='b0',source='stock',start=0,end=2,queries=['walk'],visual_goal='walking')
    result=material.prepare_candidates({'beats':[beat]}, {}, tmp_path/'media',ranker=ranker)
    assert len(result['b0'])==3
    assert result['b0'][0]['full_url']=='full3'
    assert result['b0'][0]['start']==3
    ranker.score_images.assert_called_once()
