import json
from unittest.mock import Mock
import pytest
from video_ai import fifth_pipeline as pipeline
from video_ai.models import Transcript, Word
from video_ai.fifth_plan import validate_plan
from video_ai.fifth_session import atomic_json
from video_ai.transcript import save_transcript


def setup(tmp):
    audio=tmp/'voice.mp3';audio.write_bytes(b'audio');work=tmp/'work';work.mkdir()
    transcript=Transcript([Word(0,1,'One'),Word(1,2,'Two')])
    save_transcript(transcript,work/'transcript.json')
    raw=dict(premise='test',payoff='test',beats=[dict(first_word=i,last_word=i,source='stock',role='setup',visual_goal='walk',continuity='same person',reason='test',queries=['walk','walking'],camera='none',emoji=None,sound='none') for i in range(2)])
    plan=validate_plan(raw,transcript)
    atomic_json(work/'director-plan.json',plan);atomic_json(work/'catalog.json',{'assets':[]})
    return audio,work,plan


def selected(bid):return dict(selected=dict(id='c'+bid,path='video.mp4',source='stock',start=0),alternatives=[],reason='visible action')


def test_resume_only_retrieves_unresolved_scene(tmp_path,monkeypatch):
    audio,work,plan=setup(tmp_path)
    atomic_json(work/'selection.json',dict(schema='fifth-selection-v1',selections={'beat-000':selected('0')},unresolved=['beat-001']))
    client=Mock(models=['fake'],last_model='fake')
    client._generate_json.return_value={'choices':[dict(beat='beat-001',candidate='c1',alternatives=[],reason='walking') ]}
    prepared=[]
    def prepare(partial,*a,**k):
        prepared.extend(b['id'] for b in partial['beats'])
        return {'beat-001':[dict(id='c1',source='stock',path='video.mp4',preview={'parts':[{'text':'frame'}]})]}
    monkeypatch.setattr(pipeline,'prepare_candidates',prepare)
    monkeypatch.setattr(pipeline,'render_fifth',lambda *a,**k:{'ok':True})
    monkeypatch.setattr(pipeline,'transcribe_local',lambda *a,**k:pytest.fail('Must reuse transcript'))
    assert pipeline.create_fifth(audio,tmp_path/'out.mp4',work,cache=tmp_path/'cache',client=client,semantic=False)['ok']
    assert prepared==['beat-001'] and client._generate_json.call_count==1
    result=json.loads((work/'selection.json').read_text())
    assert len(result['selections'])==2
    pipeline.create_fifth(audio,tmp_path/'out.mp4',work,cache=tmp_path/'cache',client=client,semantic=False)
    assert prepared==['beat-001'] and client._generate_json.call_count==1


def test_new_audio_cannot_reuse_old_job(tmp_path,monkeypatch):
    audio,work,_=setup(tmp_path)
    atomic_json(work/'input.json',{'audio':'another audio','effects':True})
    with pytest.raises(ValueError,match='другой озвучки'):
        pipeline.create_fifth(audio,tmp_path/'out.mp4',work,cache=tmp_path/'cache')


def test_local_edit_swaps_approved_alternative_only(tmp_path,monkeypatch):
    audio,work,_=setup(tmp_path)
    row=selected('0');row['alternatives']=[dict(id='second',source='stock',path='second.mp4',start=0)]
    atomic_json(work/'selection.json',{'selections':{'beat-000':row}})
    atomic_json(work/'render/timeline.json',{'audio':str(audio)})
    atomic_json(work/'input.json',{'effects':True})
    calls=[]
    monkeypatch.setattr(pipeline,'render_fifth',lambda *a,**k:calls.append(a) or {'ok':True})
    pipeline.edit_fifth(work,tmp_path/'out.mp4',cache=tmp_path/'cache',beat_id='beat-000',action='alternative')
    assert calls[0][1]['selections']['beat-000']['selected']['id']=='second'
    assert calls[0][1]['selections']['beat-000']['alternatives'][0]['id']=='c0'
