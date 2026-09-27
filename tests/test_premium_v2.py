from pathlib import Path

from video_ai.cache import DiskCache, cache_key
from video_ai.models import Scene, ShotPlan, Word
from video_ai.premium_audio_v2 import build_premium_audio_plan
from video_ai.premium_pacing import apply_premium_pacing, classify_pace
from video_ai.premium_v2_renderer import premium_caption_events_v2
from video_ai.scene_grammar import resolve_scene_grammar


def _words(texts, start=0.0, step=0.22):
    out = []
    cursor = start
    for text in texts:
        out.append(Word(cursor, cursor + step * 0.72, text))
        cursor += step
    return out


def test_cache_round_trip(tmp_path):
    cache = DiskCache(tmp_path / "cache")
    key = cache_key("scene", 1)
    assert cache.get_json(key) is None
    cache.set_json(key, {"ok": True})
    assert cache.get_json(key) == {"ok": True}


def test_semantic_pacing_marks_reveal():
    words = _words(["и", "тут", "оказалось"])
    assert classify_pace(words, caption="и тут оказалось", gap_after=0.05) == "reveal"


def test_premium_pacing_splits_long_fast_beat():
    words = [
        Word(0.00, 0.12, "a"),
        Word(0.20, 0.32, "b"),
        Word(0.40, 0.52, "c"),
        Word(0.60, 0.72, "d"),
        Word(0.80, 0.92, "e"),
        Word(1.00, 1.12, "f"),
    ]
    scene = Scene(
        0.0,
        1.12,
        "fast action",
        caption="a b c d e f",
        caption_words=words,
    )
    plan = ShotPlan(Path("voice.wav"), [scene])
    changed = apply_premium_pacing(plan)
    assert changed
    assert len(plan.scenes) == 2
    assert plan.scenes[0].end <= plan.scenes[1].start


def test_scene_grammar_prevents_adjacent_hero_effects():
    plan = ShotPlan(
        Path("voice.wav"),
        [
            Scene(0, 1, "a", caption="10", premium_layout="big_number"),
            Scene(1, 2, "b", caption="wow", premium_layout="freeze_frame"),
            Scene(2, 3, "c", caption="normal", premium_layout="clean"),
        ],
    )
    resolve_scene_grammar(plan)
    assert plan.scenes[0].premium_layout == "big_number"
    assert plan.scenes[1].premium_layout == "clean"


def test_semantic_lock_allows_safe_focus_but_not_split():
    scene = Scene(
        0,
        1,
        "history",
        caption="named event",
        semantic_lock=True,
        required_entities=["Named Event"],
        premium_layout="split_screen",
    )
    plan = ShotPlan(Path("voice.wav"), [scene])
    resolve_scene_grammar(plan)
    assert scene.premium_layout == "clean"

    scene.premium_layout = "focus_zoom"
    resolve_scene_grammar(plan)
    assert scene.premium_layout == "focus_zoom"


def test_number_caption_gets_hero_typography():
    scene = Scene(
        0,
        1,
        "money",
        caption="он получил 5000 долларов",
        caption_words=[
            Word(0.05, 0.18, "он"),
            Word(0.20, 0.38, "получил"),
            Word(0.40, 0.62, "5000"),
            Word(0.64, 0.90, "долларов"),
        ],
        premium_highlights=["5000"],
    )
    plan = ShotPlan(Path("voice.wav"), [scene])
    events = premium_caption_events_v2(scene, plan)
    numeric = next(event for event in events if "5000" in event)
    assert r"\fscx150" in numeric
    assert r"\pos(" in numeric


def test_caption_moves_up_when_subject_is_low():
    scene = Scene(
        0,
        1,
        "person",
        caption="важное слово",
        caption_words=[
            Word(0.05, 0.35, "важное"),
            Word(0.40, 0.75, "слово"),
        ],
        focus_y=0.75,
    )
    plan = ShotPlan(Path("voice.wav"), [scene])
    events = premium_caption_events_v2(scene, plan)
    expected_y = int(plan.height * 0.25)
    assert f",{expected_y})" in events[0]


def test_audio_plan_uses_word_timing_and_riser():
    scene = Scene(
        1.0,
        2.0,
        "phone",
        caption="и тут телефон",
        caption_words=[
            Word(1.00, 1.12, "и"),
            Word(1.16, 1.30, "тут"),
            Word(1.44, 1.70, "телефон"),
        ],
        pace_class="reveal",
        premium_music_drop=True,
    )
    plan = ShotPlan(Path("voice.wav"), [scene])
    audio = build_premium_audio_plan(plan)
    assert any(cue.name == "riser_up" for cue in audio.cues)
    notification = next(cue for cue in audio.cues if cue.name == "notification")
    assert abs(notification.time - 1.44) < 0.001
    assert audio.music_drops == [1.0]
