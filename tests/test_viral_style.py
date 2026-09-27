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


def test_viral_motion_has_fast_beats_and_rest():
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
        "snap_zoom",
        "snap_zoom",
        "snap_zoom",
        "micro_push",
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


def test_viral_locked_scene_uses_safe_push():
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

    assert scene.motion_preset == "dramatic_push"


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
