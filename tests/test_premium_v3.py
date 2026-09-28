from pathlib import Path

from video_ai.models import Scene, ShotPlan, Transcript, Word
from video_ai.premium_v3 import _normalize_beats, _scene_from_beat, apply_callback_reuse
from video_ai.premium_v3_sequence import apply_sequence_repairs
from video_ai.premium_v3_qc import inspect_premium_v3


def _transcript():
    return Transcript(
        words=[
            Word(0.0, 0.3, "Мы"),
            Word(0.3, 0.7, "час"),
            Word(0.7, 1.1, "выбирали"),
            Word(1.1, 1.5, "фильм"),
            Word(1.5, 1.9, "а"),
            Word(1.9, 2.3, "потом"),
            Word(2.3, 2.7, "уснули"),
        ],
        language="ru",
    )


def test_v3_normalizes_word_coverage():
    t = _transcript()
    data = {"beats": [{"end_word": 3, "role": "setup"}, {"end_word": 999, "role": "ending"}]}
    beats = _normalize_beats(t, data)
    assert beats[0]["start_word"] == 0
    assert beats[0]["end_word"] == 3
    assert beats[-1]["end_word"] == len(t.words)


def test_v3_action_recipe_sets_punch_zoom():
    t = _transcript()
    scene, recipe = _scene_from_beat(t.words, {
        "start_word": 0,
        "end_word": 4,
        "role": "reveal",
        "visual_intent": "person exhausted after choosing a movie",
        "media": "stock_video",
        "queries": ["tired person couch television"],
        "technique": "punch_zoom",
        "tone": "funny",
    })
    assert scene.premium_layout == "focus_zoom"
    assert scene.motion_preset == "dramatic_push"
    assert recipe["role"] == "reveal"


def test_v3_sequence_repair_changes_queries_only_for_requested_scene():
    plan = ShotPlan(
        audio=Path("voice.mp3"),
        scenes=[
            Scene(0, 1, "old one", caption="a"),
            Scene(1, 2, "old two", caption="b"),
        ],
    )
    changed = apply_sequence_repairs(plan, [{
        "scene": 1,
        "action": "replace",
        "new_queries": ["person choosing movie on television"],
    }])
    assert changed == [1]
    assert plan.scenes[0].query == "old one"
    assert plan.scenes[1].query == "person choosing movie on television"


def test_v3_callback_reuses_non_adjacent_known_visual():
    scenes = [Scene(i, i + 1, f"q{i}", caption=f"c{i}") for i in range(5)]
    scenes[0].asset = "known.mp4"
    scenes[0].asset_kind = "video"
    plan = ShotPlan(audio=Path("voice.mp3"), scenes=scenes)
    blueprint = {"recipes": [{"scene": 4, "return_to": 0}]}
    changed = apply_callback_reuse(plan, blueprint)
    assert changed == [4]
    assert plan.scenes[4].asset == "known.mp4"
    assert plan.scenes[4].focus_source == "premium-v3-callback"


def test_v3_qc_flags_excessive_effect_density():
    plan = ShotPlan(
        audio=Path("voice.mp3"),
        scenes=[
            Scene(i, i + 1, f"q{i}", asset=f"a{i}.mp4", asset_kind="video", premium_layout="focus_zoom")
            for i in range(5)
        ],
    )
    report = inspect_premium_v3(plan)
    assert any(item["type"] == "effect_density" for item in report["issues"])
