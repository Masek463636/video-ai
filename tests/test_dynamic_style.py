import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from video_ai.dynamic_reactions import build_reactions, place_reactions
from video_ai.models import Scene, ShotPlan, Word
from video_ai.renderer import render_plan, _ass_text


class Client:
    def __init__(self, replies): self.replies = iter(replies)
    def _generate_json(self, *args, **kwargs): return next(self.replies)


def test_actual_display_intervals_cannot_overlap(tmp_path):
    pack=tmp_path/'stickers'; pack.mkdir()
    assets=[pack/f'{i}.gif' for i in range(3)]
    for p in assets: p.touch()
    plan=ShotPlan(Path('voice.wav'), [Scene(0,5,'',caption='Первый второй третий',
        caption_words=[Word(0,.5,'Первый'), Word(.8,1.3,'второй'), Word(3,3.5,'третий')])])
    proposals={'effects':[{'scene':0,'anchor':anchor,'query':'surprise'} for anchor in ('Первый','второй','третий')]}
    client=Client([proposals]+[{'readable':True,'relevant':True,'kind':'emoji'}]*2)
    with patch('video_ai.gemini_ai.get_gemini_client',return_value=client), patch('video_ai.shorts_fx._find_local_sticker_by_prompt',side_effect=assets), patch('video_ai.story_media.preview_parts',return_value=([{'text':'preview'}],1)):
        effects=build_reactions(plan,tmp_path/'out',sticker_dir=pack)
    assert [(e['start'],e['end']) for e in effects]==[(0,1.25),(3,4.25)]
    report=json.loads((tmp_path/'out/overlays.json').read_text())
    assert report['decisions'][1]['status']=='too_short_or_too_close'


def test_saved_reactions_are_clamped_and_never_text_cards(tmp_path):
    asset=tmp_path/'meme.gif';asset.touch()
    item={'type':'sticker','asset':str(asset),'reaction_kind':'meme','source':'giphy','start':0,'end':1.25,'label':'unwanted'}
    effects=place_reactions([item, dict(item,start=.5,end=1.5),dict(item,start=2,end=20),dict(item,type='text',start=3,end=4)],3)
    assert [(e['start'],e['end']) for e in effects]==[(0,1.25),(2,3)]
    assert all(e['layout_box']==[.02,.05,.96,.5] and not e['label'] for e in effects)
    assert place_reactions([dict(item,source='unverified')],3)==[]
    assert place_reactions([dict(item,end=float('nan'))],3)==[]


def test_bounce_only_in_third_style():
    old=_ass_text('Привет',editing_polish=True)
    new=_ass_text('Привет',editing_polish=True,editing_style='dynamic')
    assert r'\fscx40' not in old and r'\fscx40' in new
    assert r'\t(60,120,\fscx100\fscy100)' in new


@pytest.mark.parametrize('extension',['mp4','gif'])
def test_real_render_keeps_large_centered_reaction_and_audio(tmp_path,extension):
    def ff(*args): subprocess.run(['ffmpeg','-y','-v','error',*args],check=True)
    base=tmp_path/'base.mp4';audio=tmp_path/'voice.wav';asset=tmp_path/f'reaction.{extension}'
    ff('-f','lavfi','-i','color=blue:s=360x640:r=24:d=3',str(base))
    ff('-f','lavfi','-i','sine=frequency=440:duration=3',str(audio))
    ff('-f','lavfi','-i','color=lime:s=80x120:r=12:d=1',str(asset))
    plan=ShotPlan(audio,[Scene(0,3,'',caption='Вот это да!')],width=360,height=640,fps=24)
    item={'type':'sticker','asset':str(asset),'reaction_kind':'meme','source':'local_reaction','start':.5,'end':1.75}
    with patch('video_ai.gemini_ai.get_gemini_client',return_value=None),patch('video_ai.composition_review.CompositionReviewer.overlays',side_effect=AssertionError('second review erased reaction')):
        result=render_plan(plan,tmp_path/'final.mp4',work_dir=tmp_path/'work',base_video=base,
                           composition_review=True,editing_style='dynamic',overlays=[item])
    info=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(result)]))
    assert {s['codec_type'] for s in info['streams']}=={'video','audio'}
    assert abs(float(info['format']['duration'])-3)<.15
    def pixel(t,x,y):
        return subprocess.check_output(['ffmpeg','-v','error','-ss',str(t),'-i',str(result),'-frames:v','1','-vf',f'crop=10:10:{x}:{y},scale=1:1','-f','rawvideo','-pix_fmt','rgb24','-'])
    # Tall insert fills panel height and is centred (x~73..286), not pinned left.
    for x,y in [(80,50),(270,330)]:
        p=pixel(1,x,y);assert p[1]>p[2]+100
    for t,x,y in [(1,15,100),(.2,180,100),(2,180,100)]:
        p=pixel(t,x,y);assert p[2]>p[1]+100
    assert len(json.loads((tmp_path/'work/overlays.placed.json').read_text())['overlays'])==1
