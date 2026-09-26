import json
from pathlib import Path
from unittest.mock import patch
import subprocess
import pytest
from video_ai.composition_review import CompositionReviewer, boxes, choose_slot
from video_ai.models import ShotPlan, Scene
from video_ai.renderer import _apply_overlays

@pytest.mark.parametrize('value', [None, [[0,0,2,1]], [[0,0,float('nan'),1]], [[True,0,.2,.2]]])
def test_invalid_boxes_rejected(value):
    with pytest.raises(ValueError): boxes(value)

def test_free_slot_respects_all_frames_and_captions():
    assert choose_slot([[0,0,1,1]]) is None
    slot=choose_slot([[0,0,1,.5]])
    assert slot[1]>.64

class Client:
    def __init__(self, replies): self.replies=iter(replies); self.calls=0
    def _generate_json(self,*args,**kwargs):
        self.calls+=1
        reply=next(self.replies)
        if isinstance(reply,Exception): raise reply
        return reply

def test_irrelevant_insert_removed_and_relevant_placed(tmp_path):
    client=Client([{'relevant':False,'reason':'unrelated restaurant'}, {'relevant':True,'reason':'surprised reaction'}, {'protected_boxes':[[0,0,1,.5]]}])
    review=CompositionReviewer(tmp_path,client)
    effects=[{'type':'png','asset':'card.png','query':'phone','start':0,'end':1}, {'type':'sticker','asset':'emoji.gif','start':1,'end':2}]
    plan=ShotPlan(Path('audio'),[Scene(0,2,'phone',caption='Ты звонишь')])
    with patch('video_ai.composition_review.frames',return_value=[]):
        result=review.overlays('base',effects,plan)
    assert len(result)==1 and result[0]['animation']=='pop'
    assert result[0]['layout_box'][1]>.64
    assert review.report['overlays'][0]['reason']=='unrelated restaurant'
    assert json.loads((tmp_path/'overlays.reviewed.json').read_text())['overlays']==result

def test_failure_stops_further_calls_and_preserves_text(tmp_path):
    client=Client([RuntimeError('quota')]); review=CompositionReviewer(tmp_path,client)
    effects=[{'type':'png','asset':'x','start':0,'end':1}]*3+[{'type':'text','label':'20','start':0,'end':1}]
    with patch('video_ai.composition_review.frames',return_value=[]):
        result=review.overlays('base',effects,ShotPlan(Path('audio'),[]))
    assert len(result)==1 and result[0]['type']=='text'
    assert client.calls==1

def test_repair_applies_only_selected_valid_alternative(tmp_path):
    approved={'usable':True,'choice':2,'preserves_visible_subjects':True,'checks':[{'requirement':'face visible','visible':True,'evidence':'face unobscured'}]}
    client=Client([{'usable':False,'reason':'face hidden','requirements':['face visible']},approved,approved])
    review=CompositionReviewer(tmp_path,client)
    scene=Scene(0,1,'face',asset='source.mp4',asset_kind='video')
    plan=ShotPlan(Path('audio'),[scene])
    with patch('video_ai.composition_review.frames',return_value=[]), patch('video_ai.renderer._safe_probe_duration',return_value=10), patch('video_ai.renderer._render_scene',side_effect=lambda *a,**k: Path(a[3]).write_bytes(b'clip')) as render:
        review.scene(scene,plan,1,tmp_path/'clip.mp4',0,reference_framing=True,editing_polish=True)
    assert review.report['scenes'][0]['status']=='repaired'
    assert render.call_args.args[0].source_start==4.5
    assert scene.source_start==0  # input plan unchanged

def test_real_overlay_is_inside_reviewed_box(tmp_path):
    from PIL import Image
    base=tmp_path/'base.mp4'; sticker=tmp_path/'sticker.png'; out=tmp_path/'out.mp4'
    subprocess.run(['ffmpeg','-y','-v','error','-f','lavfi','-i','color=black:s=100x180:r=10:d=1','-c:v','libx264',str(base)],check=True)
    Image.new('RGB',(100,100),'red').save(sticker)
    _apply_overlays(base,[{'type':'png','asset':str(sticker),'start':0,'end':1,'animation':'fly','layout_box':[.1,.7,.3,.2]}],ShotPlan(Path('audio'),[],width=100,height=180,fps=10),out,crf=18)
    raw=subprocess.check_output(['ffmpeg','-v','error','-ss','0.5','-i',str(out),'-frames:v','1','-f','rawvideo','-pix_fmt','rgb24','-'])
    red=[(i%100,i//100) for i in range(18000) if raw[i*3]>150]
    assert red
    assert min(x for x,y in red)>=9 and max(x for x,y in red)<=41
    assert min(y for x,y in red)>=125 and max(y for x,y in red)<=163


def test_bad_response_gets_one_format_retry(tmp_path):
    client=Client([[{'choice':0},{'choice':1}], {'choice':1,'usable':True}])
    review=CompositionReviewer(tmp_path,client)
    assert review.ask([{'text':'choose'}])['choice']==1
    assert not review.disabled and client.calls==2


def test_persistent_bad_format_does_not_disable_following_scene(tmp_path):
    client=Client([[],ValueError('invalid JSON'),{'usable':True}])
    review=CompositionReviewer(tmp_path,client)
    with pytest.raises(ValueError): review.ask([])
    assert not review.disabled
    assert review.ask([])=={'usable':True}


def test_programming_error_is_not_provider_outage(tmp_path):
    client=Client([TypeError('unexpected value'),{'usable':True}])
    review=CompositionReviewer(tmp_path,client)
    with pytest.raises(TypeError): review.ask([])
    assert review.ask([])=={'usable':True}


def test_changed_criterion_and_missing_object_rejected():
    from video_ai.composition_review import repair_approved
    data={'usable':True,'preserves_visible_subjects':True,
          'checks':[{'requirement':'face visible','visible':True,'evidence':'smile'}]}
    assert not repair_approved(data,['phone visible'])
    data['checks'][0]['requirement']='phone visible'
    data['preserves_visible_subjects']=False
    assert not repair_approved(data,['phone visible'])


def test_full_render_failure_keeps_original(tmp_path):
    approved={'usable':True,'choice':0,'preserves_visible_subjects':True,
              'checks':[{'requirement':'phone','visible':True,'evidence':'phone held'}]}
    review=CompositionReviewer(tmp_path,Client([{'usable':False,'requirements':['phone']},approved]))
    clip=tmp_path/'clip.mp4'; clip.write_bytes(b'original')
    scene=Scene(0,1,'phone',asset='source.mp4',asset_kind='video')
    def render(*args,**kwargs):
        Path(args[3]).write_bytes(b'partial')
        if 'pending' in str(args[3]):
            raise subprocess.CalledProcessError(1,'ffmpeg',stderr=b'encoder failed')
    with patch('video_ai.composition_review.frames',return_value=[]), patch('video_ai.renderer._safe_probe_duration',return_value=10), patch('video_ai.renderer._render_scene',side_effect=render):
        review.scene(scene,ShotPlan(Path('audio'),[scene]),1,clip,0,reference_framing=True,editing_polish=True)
    assert clip.read_bytes()==b'original'
    assert review.report['scenes'][0]['ffmpeg_error']=='encoder failed'
    assert not list(tmp_path.glob('*pending.mp4'))


def test_insert_is_judged_without_background_and_coordinates_retried(tmp_path):
    client=Client([{'relevant':True}, {'protected_boxes':[[10,20,30,40]]}, {'protected_boxes':[[0,0,1,.5]]}])
    review=CompositionReviewer(tmp_path,client)
    requests=[]
    original=client._generate_json
    def generate(parts,**kwargs):
        requests.append(parts)
        return original(parts,**kwargs)
    client._generate_json=generate
    def sample(path,*args): return [{'text':'IMAGE:'+str(path)}]
    with patch('video_ai.composition_review.frames',side_effect=sample):
        result=review.overlays('background',[{'type':'png','asset':'insert','start':0,'end':1}],ShotPlan(Path('audio'),[]))
    assert len(result)==1
    assert {'text':'IMAGE:background'} not in requests[0]
    assert {'text':'IMAGE:insert'} in requests[0]
    assert {'text':'IMAGE:insert'} not in requests[1]
    assert len(review.report['overlays'][0]['placement_reviews'])==2


def test_final_verification_can_veto_repair(tmp_path):
    approved={'usable':True,'choice':0,'preserves_visible_subjects':True,
              'checks':[{'requirement':'phone','visible':True,'evidence':'phone held'}]}
    review=CompositionReviewer(tmp_path,Client([{'usable':False,'requirements':['phone']},approved,
        dict(approved,preserves_visible_subjects=False)]))
    clip=tmp_path/'clip.mp4'; clip.write_bytes(b'original')
    scene=Scene(0,1,'phone',asset='source.mp4',asset_kind='video')
    with patch('video_ai.composition_review.frames',return_value=[]), patch('video_ai.renderer._safe_probe_duration',return_value=10), patch('video_ai.renderer._render_scene',side_effect=lambda *a,**k: Path(a[3]).write_bytes(b'candidate')):
        review.scene(scene,ShotPlan(Path('audio'),[scene]),1,clip,0,reference_framing=True,editing_polish=True)
    assert clip.read_bytes()==b'original'
    assert review.report['scenes'][0]['status']=='unresolved'
