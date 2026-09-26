from pathlib import Path
from unittest.mock import patch
import json
from video_ai.models import Scene,ShotPlan
from video_ai.classic_reactions import build_reactions

class Client:
    def __init__(self, replies): self.replies=iter(replies)
    def _generate_json(self,*a,**k): return next(self.replies)

def test_reactions_require_readability_and_no_text_fallback(tmp_path):
    pack=tmp_path/'stickers'; pack.mkdir()
    asset=pack/'smile.gif'; asset.touch()
    plan=ShotPlan(Path('a'),[Scene(0,3,'phone',caption='Ты звонил?')])
    client=Client([{'effects':[{'scene':0,'anchor':'Ты звонил','query':'smug grin'}]},
                   {'readable':False,'relevant':True,'kind':'meme'}])
    with patch('video_ai.gemini_ai.get_gemini_client',return_value=client), patch('video_ai.shorts_fx._find_local_sticker_by_prompt',return_value=asset), patch('video_ai.story_media.preview_parts',return_value=([{'text':'preview'}],1)):
        assert build_reactions(plan,tmp_path/'out',sticker_dir=pack)==[]
    assert json.loads((tmp_path/'out/overlays.json').read_text())['overlays']==[]

def test_verified_meme_uses_meme_pack_and_exact_anchor(tmp_path):
    from video_ai.models import Word
    pack=tmp_path/'stickers'; pack.mkdir(); memes=tmp_path/'memes'; memes.mkdir()
    asset=memes/'reaction.mp4'; asset.touch()
    plan=ShotPlan(Path('a'),[Scene(0,4,'phone',caption='Ты звонил?',caption_words=[Word(1,2,'Ты'),Word(2,3,'звонил?')])])
    client=Client([{'effects':[{'scene':0,'anchor':'Ты звонил','query':'smug grin','pack':'memes'}]},
                   {'readable':True,'relevant':True,'kind':'meme'}])
    with patch('video_ai.gemini_ai.get_gemini_client',return_value=client), patch('video_ai.shorts_fx._find_local_sticker_by_prompt',return_value=asset) as find, patch('video_ai.story_media.preview_parts',return_value=([{'text':'preview'}],1)):
        effects=build_reactions(plan,tmp_path/'out',sticker_dir=pack)
    assert find.call_args.args[0]==memes
    assert len(effects)==1 and effects[0]['type']=='sticker'
    assert effects[0]['start']==1 and effects[0]['label']==''
    assert effects[0]['reaction_kind']=='meme'


def test_full_render_replacement_and_video_reaction_preserve_audio(tmp_path):
    import subprocess
    from video_ai.renderer import render_plan
    def ffmpeg(*args):
        subprocess.run(['ffmpeg','-y','-v','error',*args],check=True)
    audio=tmp_path/'voice.wav'; old=tmp_path/'old.mp4'; new=tmp_path/'new.mp4'; meme=tmp_path/'meme.mp4'
    ffmpeg('-f','lavfi','-i','sine=frequency=440:duration=2',str(audio))
    for path,color in [(old,'red'),(new,'blue'),(meme,'green')]:
        ffmpeg('-f','lavfi','-i',f'color={color}:s=160x90:r=12:d=2','-c:v','libx264',str(path))
    plan=ShotPlan(audio,[Scene(0,2,'call',asset=str(old),asset_kind='video',caption='Ты звонил?')],width=180,height=320,fps=12)
    client=Client([{'usable':False,'failure_kind':'action','requirements':['call'],'replacement_queries':['phone call']},
        {'usable':True}, {'relevant':True,'readable':True,'kind':'meme'}, {'protected_boxes':[[0,0,1,.5]]}])
    output=tmp_path/'output.mp4'
    with patch('video_ai.gemini_ai.get_gemini_client',return_value=client), patch('video_ai.story_media._resolve_query',return_value={'path':str(new),'download_url':'https://media/new'}):
        render_plan(plan,output,work_dir=tmp_path/'work',reference_framing=True,composition_review=True,
            overlays=[{'type':'text','label':'unwanted','start':0,'end':1},
                      {'type':'sticker','asset':str(meme),'query':'reaction','start':.2,'end':1.8}])
    info=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(output)]))
    assert {s['codec_type'] for s in info['streams']}=={'video','audio'}
    assert abs(float(info['format']['duration'])-2)<.2
    report=json.loads((tmp_path/'work/composition/review.json').read_text())
    assert report['scenes'][0]['status']=='replaced'
    assert report['overlays'][0]['status']=='unsupported_accent'
    assert report['overlays'][1]['status']=='placed'
    # New blue source actually appears, not only a changed report.
    pixel=subprocess.check_output(['ffmpeg','-v','error','-ss','1','-i',str(output),'-frames:v','1','-vf','crop=20:20:0:0,scale=1:1','-f','rawvideo','-pix_fmt','rgb24','-'])
    assert pixel[2]>pixel[0]+80
