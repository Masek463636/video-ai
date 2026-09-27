from pathlib import Path

from video_ai.audio_plan import (
    build_audio_plan,
)
from video_ai.io import (
    load_shot_plan,
    save_shot_plan,
)
from video_ai.models import (
    Scene,
    ShotPlan,
    Word,
)
from video_ai.renderer import (
    _premium_caption_events,
)


def test_premium_defaults_do_not_change_old_scene():
    scene = Scene(
        0,
        1,
        "normal",
    )

    assert scene.premium_layout == "clean"
    assert scene.premium_highlights == []
    assert scene.secondary_asset is None
    assert scene.premium_music_drop is False


def test_premium_metadata_round_trip(tmp_path):
    secondary = tmp_path / "other.jpg"
    secondary.write_bytes(b"x")

    cutout = tmp_path / "cutout.png"
    cutout.write_bytes(b"x")

    scene = Scene(
        0,
        1,
        "test",
        premium_layout="split_screen",
        premium_highlights=["???????"],
        premium_text="???????",
        premium_music_drop=True,
        secondary_asset=str(secondary),
        secondary_asset_kind="image",
        secondary_query="other subject",
        premium_foreground=str(cutout),
    )

    plan = ShotPlan(
        tmp_path / "voice.wav",
        [scene],
    )

    path = save_shot_plan(
        plan,
        tmp_path / "plan.json",
    )

    loaded = load_shot_plan(path)

    result = loaded.scenes[0]

    assert result.premium_layout == "split_screen"
    assert result.premium_highlights == ["???????"]
    assert result.premium_music_drop is True
    assert result.secondary_asset_kind == "image"
    assert Path(result.secondary_asset).resolve() == secondary.resolve()


def test_premium_caption_uses_exact_word_timing():
    scene = Scene(
        0,
        2,
        "",
        caption="this million rubles",
        caption_words=[
            Word(0.10, 0.40, "this"),
            Word(0.42, 1.00, "million"),
            Word(1.02, 1.50, "rubles"),
        ],
        premium_highlights=["million"],
    )

    plan = ShotPlan(
        Path("voice.wav"),
        [scene],
    )

    events = _premium_caption_events(
        scene,
        plan,
    )

    assert len(events) == 3
    assert "0:00:00.42" in events[1]

    # Strong semantic highlight = larger red/orange punch.
    assert r"\fscx128" in events[1]
    assert "million" in events[1]


def test_premium_audio_adds_drop_but_legacy_does_not():
    scene = Scene(
        0,
        1,
        "",
        caption="? ??? ?????????",
        premium_music_drop=True,
    )

    plan = ShotPlan(
        Path("voice.wav"),
        [scene],
    )

    normal = build_audio_plan(
        plan,
    )

    premium = build_audio_plan(
        plan,
        premium=True,
    )

    assert normal.music_drops == []
    assert premium.music_drops == [0.0]
    assert premium.music_gain_db < normal.music_gain_db
