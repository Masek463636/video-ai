import copy
import io
import json
import urllib.error
from unittest.mock import Mock, patch
import pytest
from video_ai.models import Transcript, Word
from video_ai.fifth_plan import plan_story, validate_plan
from video_ai.fifth_session import FifthSession
from video_ai.gemini_ai import GeminiClient


def transcript():
    return Transcript([Word(0,.4,'Выбирали'),Word(.5,.9,'фильм'),Word(1,1.4,'и'),Word(1.5,2,'уснули')])


def answer():
    return dict(premise='Выбор фильма',payoff='Уснули вместо просмотра',beats=[
        dict(first_word=0,last_word=1,role='setup',source='stock',visual_goal='Browsing a TV menu',
             continuity='Living room throughout',reason='Show the choice',queries=['person television remote'],camera='gentle_push',emoji=None,sound='none'),
        dict(first_word=2,last_word=3,role='punchline',source='meme',visual_goal='Sleeping reaction',
             continuity='Consequence of waiting',reason='Reveal the outcome',queries=['sleeping tired reaction'],camera='none',emoji=None,sound='none')])


def client(*responses):
    c=Mock();c.models=['test-model'];c.last_model='test-model';c._generate_json.side_effect=list(responses)
    return c


def test_story_plan_preserves_words_and_reuses_cache(tmp_path):
    c=client(answer()); cache=tmp_path/'cache'
    first=plan_story(transcript(),tmp_path/'one',cache_dir=cache,client=c)
    second=plan_story(transcript(),tmp_path/'two',cache_dir=cache,client=c)
    assert c._generate_json.call_count==1 and first==second
    assert [(b['start'],b['end']) for b in first['beats']]==[(0,1),(1,2)]
    assert [b['source'] for b in first['beats']]==['stock','meme']
    stats=json.loads((tmp_path/'two/gemini-metrics.json').read_text())
    assert stats['cache_hits']==1 and stats['logical_calls']==0


def test_invalid_plan_repaired_once_and_repaired_result_cached(tmp_path):
    bad=answer();bad['beats'][0]['last_word']=2
    c=client(bad,answer())
    plan_story(transcript(),tmp_path/'one',cache_dir=tmp_path/'cache',client=c)
    plan_story(transcript(),tmp_path/'two',cache_dir=tmp_path/'cache',client=c)
    assert c._generate_json.call_count==2
    metrics=json.loads((tmp_path/'one/gemini-metrics.json').read_text())
    assert metrics['invalid_responses']==1


@pytest.mark.parametrize('mutation', ['gap','overlap','missing_end','outside_emoji','meme_emoji','unknown_camera'])
def test_invalid_editing_contract_rejected(mutation):
    data=answer()
    if mutation=='gap':data['beats'][1]['first_word']=3
    if mutation=='overlap':data['beats'][1]['first_word']=1
    if mutation=='missing_end':data['beats'].pop()
    if mutation=='outside_emoji':data['beats'][0]['emoji']=dict(word=3,query='happy',reason='test')
    if mutation=='meme_emoji':data['beats'][1]['emoji']=dict(word=3,query='happy',reason='test')
    if mutation=='unknown_camera':data['beats'][0]['camera']='random_zoom'
    with pytest.raises(ValueError):validate_plan(data,transcript())


def test_provider_failure_stops_cascade_and_does_not_log_secret(tmp_path):
    c=client(RuntimeError('SECRET_API_KEY leaked by provider'))
    session=FifthSession(c,tmp_path/'cache',tmp_path/'metrics.json')
    for stage in ('one','two'):
        with pytest.raises(RuntimeError):session.ask(stage,[{'text':stage}],lambda v:v,version='v1')
    assert c._generate_json.call_count==1
    assert 'SECRET_API_KEY' not in (tmp_path/'metrics.json').read_text()
    assert not list((tmp_path/'cache').glob('*.json'))


def test_cache_version_and_request_content_are_part_of_key(tmp_path):
    c=client({'ok':True},{'ok':True},{'ok':True})
    session=FifthSession(c,tmp_path/'cache',tmp_path/'metrics.json')
    for text,version in [('one','v1'),('one','v2'),('two','v2')]:
        session.ask('test',[{'text':text}],lambda v:v,version=version)
    assert c._generate_json.call_count==3


def test_actual_http_attempts_waits_and_usage_are_measured(tmp_path):
    c=GeminiClient('fake-secret',model='test-model')
    session=FifthSession(c,tmp_path/'cache',tmp_path/'metrics.json')
    error=urllib.error.HTTPError('https://example.invalid',429,'limited',{},io.BytesIO(b'Please retry in 1s'))
    reply={'candidates':[{'content':{'parts':[{'text':'{"ok":true}'}]}}],
           'usageMetadata':{'promptTokenCount':100,'candidatesTokenCount':5}}
    with patch('urllib.request.urlopen',side_effect=[error,io.StringIO(json.dumps(reply))]),patch('time.sleep') as sleep:
        assert session.ask('test',[{'text':'hello'}],lambda v:v,version='v1')=={'ok':True}
    assert session.stats['network_attempts']==2
    assert session.stats['rate_limits']==1
    assert session.stats['input_tokens']==100 and session.stats['output_tokens']==5
    assert session.stats['retry_wait_seconds']==pytest.approx(1.35)
    sleep.assert_called_once()
