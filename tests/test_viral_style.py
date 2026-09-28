from pathlib import Path

from video_ai.io import (
    load_shot_plan,
    save_shot_plan,
)
from video_ai.models import (
    Scene,
    ShotPlan,
)
from video_ai.renderer import _ass_text
from video_ai.viral_fx import (
    place_viral_overlays,
)
from video_ai.viral_style import (
    apply_viral_motion,
)


def test_old_style_default_is_untouched():
    scene = Scene(
        0,
        1,
        "normal",
    )

    assert scene.motion_preset == "slow_push"


def test_viral_disables_all_camera_motion():
    scenes = [
        Scene(
            0, 1, "a",
            visual_mode="video",
            source_mode="stock_video",
            motion_preset="none",
        ),
        Scene(
            1, 2, "b",
            visual_mode="video",
            source_mode="stock_video",
            motion_preset="none",
        ),
        Scene(
            2, 3, "c",
            visual_mode="image",
            source_mode="generic_image",
            motion_preset="slow_push",
        ),
        Scene(
            3, 4, "d",
            visual_mode="video",
            source_mode="stock_video",
            motion_preset="none",
        ),
    ]

    plan = ShotPlan(
        Path("voice.wav"),
        scenes,
    )

    apply_viral_motion(plan)

    assert [
        scene.motion_preset
        for scene in scenes
    ] == [
        "none",
        "none",
        "none",
        "none",
    ]


def test_viral_keeps_meme_without_snap():
    scene = Scene(
        0,
        1,
        "meme",
        visual_mode="meme",
        source_mode="meme_library",
        motion_preset="slow_push",
    )

    plan = ShotPlan(
        Path("voice.wav"),
        [scene],
    )

    apply_viral_motion(plan)

    assert scene.motion_preset == "none"


def test_viral_locked_scene_is_static():
    scene = Scene(
        0,
        1,
        "history",
        visual_mode="image",
        source_mode="historical_archive",
        motion_preset="micro_push",
        semantic_lock=True,
        required_entities=["person"],
    )

    plan = ShotPlan(
        Path("voice.wav"),
        [scene],
    )

    apply_viral_motion(plan)

    assert scene.motion_preset == "none"


def test_snap_zoom_round_trip(tmp_path):
    plan = ShotPlan(
        Path("voice.wav"),
        [
            Scene(
                0,
                1,
                "viral",
                motion_preset="snap_zoom",
            )
        ],
    )

    target = save_shot_plan(
        plan,
        tmp_path / "plan.json",
    )

    loaded = load_shot_plan(
        target
    )

    assert (
        loaded.scenes[0].motion_preset
        == "snap_zoom"
    )


def test_viral_caption_is_independent():
    classic = _ass_text(
        "??????",
        editing_polish=True,
        editing_style="classic",
    )

    dynamic = _ass_text(
        "??????",
        editing_polish=True,
        editing_style="dynamic",
    )

    viral = _ass_text(
        "??????",
        editing_polish=True,
        editing_style="viral",
    )

    assert r"\fscx35" not in classic
    assert r"\fscx35" not in dynamic

    assert r"\fscx35" in viral
    assert r"\fscx120" in viral


def test_only_verified_attention_survives(tmp_path):
    reaction = tmp_path / "reaction.gif"
    reaction.touch()

    overlays = [
        {
            "type": "sticker",
            "asset": str(reaction),
            "reaction_kind": "meme",
            "source": "local_reaction",
            "start": 0,
            "end": 1.25,
        },
        {
            "type": "arrow",
            "source": "gemini_attention",
            "attention_verified": True,
            "target_box": [
                .60,
                .20,
                .15,
                .18,
            ],
            "start": 1.5,
            "end": 2.3,
        },
        {
            "type": "circle",
            "source": "gemini_attention",
            "attention_verified": False,
            "target_box": [
                .10,
                .10,
                .20,
                .20,
            ],
            "start": 2.5,
            "end": 3.2,
        },
    ]

    result = place_viral_overlays(
        overlays,
        5,
    )

    assert [
        effect["type"]
        for effect in result
    ] == [
        "sticker",
        "arrow",
    ]


def test_attention_stays_inside_shot_and_rejects_moving_target(tmp_path):
    from unittest.mock import patch
    from video_ai.models import Word
    from video_ai.viral_fx import _attention_accents
    base=tmp_path/'base.mp4';base.touch()
    plan=ShotPlan(Path('voice.wav'),[Scene(0,1,'',caption='Вот телефон',caption_words=[Word(.1,.3,'Вот'),Word(.6,.9,'телефон')]),Scene(1,2,'',caption='Другое')])
    proposal={'effects':[{'scene':0,'type':'arrow','anchor':'телефон','target':'phone'}]}
    for moving in (False,True):
        verdict={'found':True,'confidence':.9,'target_boxes':[[.2,.2,.1,.1],[.2,.2,.1,.1],[.4 if moving else .2,.2,.1,.1]]}
        client=type('Client',(),{})()
        replies=iter([proposal,verdict]);client._generate_json=lambda *a,**k: next(replies)
        with patch('video_ai.gemini_ai.get_gemini_client',return_value=client),patch('video_ai.composition_review.frames',return_value=[{'text':'frame'}]*3) as frames:
            effects=_attention_accents(plan,tmp_path/str(moving),base_video=base,budget=1,use_gemini=True,avoid_scenes=set())
        if moving:
            assert not effects
        else:
            assert effects[0]['end']==1 and effects[0]['start']==.6
            assert len(frames.call_args.args[1])==3
            assert max(frames.call_args.args[1])<1


def test_accents_do_not_cover_reactions(tmp_path):
    meme=tmp_path/'meme.gif';meme.touch()
    effects=place_viral_overlays([
        {'type':'sticker','asset':str(meme),'source':'local_reaction','reaction_kind':'meme','start':0,'end':1.25},
        {'type':'arrow','source':'gemini_attention','attention_verified':True,'target_box':[.2,.2,.1,.1],'start':.8,'end':1.6},
        {'type':'text','label':'5','start':1.5,'end':2.5},
    ],3)
    assert [e['type'] for e in effects]==['sticker','text']


def test_viral_ignores_legacy_decorative_overlays(tmp_path):
    import subprocess,json
    import numpy as np
    from video_ai.renderer import render_plan
    def ff(*args):subprocess.run(['ffmpeg','-y','-v','error',*args],check=True)
    base=tmp_path/'base.mp4';audio=tmp_path/'voice.wav'
    ff('-f','lavfi','-i','color=blue:s=360x640:r=24:d=3',str(base))
    ff('-f','lavfi','-i','sine=frequency=440:duration=3',str(audio))
    plan=ShotPlan(audio,[Scene(0,3,'',caption='Не показывать')],width=360,height=640,fps=24)
    effects=[{'type':kind,'start':start,'end':end,'source':'gemini_attention','attention_verified':True,'target_box':[.2,.2,.2,.2]} for kind,start,end in [('arrow',.4,1),('circle',1.3,2.3)]]
    output=render_plan(plan,tmp_path/'final.mp4',base_video=base,work_dir=tmp_path/'work',editing_style='viral',captions=False,overlays=effects)
    def red_count(t):
        raw=subprocess.check_output(['ffmpeg','-v','error','-ss',str(t),'-i',str(output),'-frames:v','1','-f','rawvideo','-pix_fmt','rgb24','-'])
        pixels=np.frombuffer(raw,dtype=np.uint8).reshape(640,360,3).astype(int)
        return ((pixels[:,:,0]>pixels[:,:,1]+80)&(pixels[:,:,0]>pixels[:,:,2]+80)).sum()
    assert red_count(.6)==0
    assert red_count(1.7)==0
    assert red_count(2.7)==0
    assert not (tmp_path/'work/captions.ass').exists()
    info=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(output)]))
    assert {s['codec_type'] for s in info['streams']}=={'audio','video'}
    assert abs(float(info['format']['duration'])-3)<.15


def test_hard_cap_one_does_not_request_more_objects(tmp_path):
    from unittest.mock import patch
    from video_ai.viral_fx import build_viral_overlays
    plan=ShotPlan(Path('voice.wav'),[Scene(0,4,'')])
    with patch('video_ai.dynamic_reactions.build_reactions',return_value=[{'scene':0,'start':0,'end':1,'type':'sticker'}]) as reactions,patch('video_ai.shorts_fx._build_legacy_shorts_overlays') as objects:
        result=build_viral_overlays(plan,tmp_path,max_overlays=1,use_gemini=False)
    assert reactions.call_args.kwargs['max_overlays']==1
    objects.assert_not_called()
    assert len(result)==1


def test_viral_keeps_wide_video_static(tmp_path):
    import subprocess
    from PIL import Image,ImageDraw
    from video_ai.renderer import render_plan
    image=Image.new('RGB',(320,180),'white');draw=ImageDraw.Draw(image)
    for x in range(0,320,20): draw.rectangle((x,0,x+8,180),fill='black')
    still=tmp_path/'stripes.png';source=tmp_path/'source.mp4';output=tmp_path/'snap.mp4';image.save(still)
    subprocess.run(['ffmpeg','-y','-v','error','-loop','1','-i',str(still),'-t','1.5','-pix_fmt','yuv420p',str(source)],check=True)
    scene=Scene(0,1.5,'',asset=str(source),asset_kind='video',motion_preset='snap_zoom')
    audio=tmp_path/'voice.wav'
    subprocess.run(['ffmpeg','-y','-v','error','-f','lavfi','-i','sine=duration=1.5',str(audio)],check=True)
    plan=ShotPlan(audio,[scene],width=180,height=320,fps=24)
    apply_viral_motion(plan)
    render_plan(plan,output,work_dir=tmp_path/"work",editing_style="viral",captions=False,editing_polish=True,reference_framing=True)
    def pixels(t):return subprocess.check_output(['ffmpeg','-v','error','-ss',str(t),'-i',str(output),'-frames:v','1','-vf','crop=180:100:0:0','-f','rawvideo','-pix_fmt','gray','-'])
    first,last=pixels(.05),pixels(1.4)
    assert len(first)==len(last)==18000
    assert sum(abs(a-b) for a,b in zip(first,last))/len(first)<2


def test_viral_reaction_waits_for_cut_and_leaves_before_next():
    from video_ai.viral_fx import pace_viral_accents
    dog = {'type': 'sticker', 'start': 2., 'end': 3.25, 'anchor': 'пиццы'}
    result = pace_viral_accents([dog], [(0, 2), (2, 3.2), (3.2, 5)])
    assert result[0]['start'] == 2.32
    assert result[0]['end'] == 3.1
    assert result[0]['anchor'] == 'пиццы'
    assert dog['start'] == 2.  # Do not corrupt cached planner timing.
    assert pace_viral_accents(result, [(0, 2), (2, 3.2), (3.2, 5)]) == result
    # Tiny shots cannot fit both the new background and a readable meme.
    assert pace_viral_accents([dog], [(2, 2.8)]) == []
    # An accent on the final word must not spill into a different shot.
    assert pace_viral_accents([dict(dog, start=3.)], [(2, 3.2), (3.2, 5)]) == []


def test_located_attention_is_omitted_instead_of_retimed():
    from video_ai.viral_fx import pace_viral_accents
    arrow = {'type':'arrow', 'start':2., 'end':2.9}
    assert pace_viral_accents([arrow], [(2, 4)]) == []
    arrow['start'] = 2.4
    assert pace_viral_accents([arrow], [(2, 4)]) == [arrow]


def test_context_review_rejects_wrong_emotion_and_requests_replacement(tmp_path):
    from unittest.mock import patch, Mock
    from video_ai.composition_review import CompositionReviewer
    client = Mock()
    client._generate_json.return_value = {'usable': False, 'failure_kind': 'action',
        'requirements': ['choosing a movie on a screen'],
        'replacement_queries': ['person browsing streaming movies'], 'reason':'unrelated grimace'}
    review = CompositionReviewer(tmp_path, client, contextual=True)
    shot = Scene(0, 2, 'screaming man', caption='ещё не', asset_kind='video')
    plan = ShotPlan(Path('voice'), [shot, Scene(2,4,'remote',caption='выбрали фильм')])
    with patch('video_ai.composition_review.frames', return_value=[]), patch.object(review,'replace_source') as repair:
        review.scene(shot,plan,2,Path('clip'),0,reference_framing=True,editing_polish=True)
    repair.assert_called_once()
    prompt = client._generate_json.call_args.args[0][0]['text']
    assert 'выбрали фильм' in prompt and 'narration and story context outrank' in prompt
    assert review.report['scenes'][0]['status'] == 'unresolved'
