import json
import math
from pathlib import Path
from unittest.mock import patch

import pytest

from video_ai.broll_planner import build_donor_shot_plan
from video_ai.director import build_shot_plan
from video_ai.io import load_shot_plan, save_shot_plan
from video_ai.models import Scene, ShotPlan, Transcript, Word
from video_ai.renderer import _caption_pages, _write_ass
from video_ai.transcript import attach_caption_timings


def test_uneven_speech_uses_timestamps_not_character_count():
    words = [Word(0, .2, "Собака"), Word(.2, 1.5, "внезапно"), Word(1.5, 2, "уснула.")]
    actual = _caption_pages("Собака внезапно уснула.", 0, 2, timed_words=words)
    assert actual == [(0, .2, "Собака"), (.2, 1.5, "внезапно"), (1.5, 2, "уснула.")]
    assert actual != _caption_pages("Собака внезапно уснула.", 0, 2)


def test_connector_pairs_keep_two_word_limit_and_pauses():
    words = [Word(0, .1, "И"), Word(.1, .5, "собака"), Word(1, 1.1, "в"), Word(1.5, 2, "комнате")]
    assert _caption_pages("И собака в комнате", 0, 2, timed_words=words) == [
        (0, .5, "И собака"), (1, 1.1, "в"), (1.5, 2, "комнате")]


def test_overlapping_whisper_words_do_not_overlap_caption_cards():
    words = [Word(0, .6, "Собака"), Word(.5, 1, "уснула")]
    assert _caption_pages("Собака уснула", 0, 1, timed_words=words) == [
        (0, .5, "Собака"), (.5, 1, "уснула")]


@pytest.mark.parametrize("words", [
    [], [Word(0, 1, "Другое")],
    [Word(0, .5, "Собака"), Word(.5, 1, "играет")],
    [Word(0, .5, "Собака"), Word(.5, math.nan, "уснула")],
    [Word(0, .5, "Собака"), Word(.5, 2, "уснула")],
    [Word(0, .5, "Собака"), Word(0, 1, "уснула")],
    [Word(0, .5, "Собака"), Word(.5, .5, "уснула")],
])
def test_missing_stale_or_invalid_timings_preserve_legacy_fallback(words):
    assert _caption_pages("Собака уснула", 0, 1, timed_words=words) == _caption_pages("Собака уснула", 0, 1)


def test_roundtrip_and_legacy_json(tmp_path):
    words = [Word(0, .2, "Собака"), Word(.2, 1, "уснула")]
    plan = ShotPlan(Path("voice.wav"), [Scene(0, 1, "dog", caption="Собака уснула", caption_words=words)])
    path = save_shot_plan(plan, tmp_path / "plan.json")
    assert load_shot_plan(path).scenes[0].caption_words == words
    data = json.loads(path.read_text())
    del data["scenes"][0]["caption_words"]
    path.write_text(json.dumps(data))
    assert load_shot_plan(path).scenes[0].caption_words == []


def test_ass_uses_speech_boundaries_and_preserves_style(tmp_path):
    scene = Scene(0, 1, "dog", caption="Собака уснула", caption_words=[Word(0, .2, "Собака"), Word(.2, 1, "уснула")])
    path = tmp_path / "captions.ass"
    _write_ass(ShotPlan(Path("voice.wav"), [scene]), path)
    ass = path.read_text(encoding="utf-8-sig")
    assert "Dialogue: 0,0:00:00.00,0:00:00.20," in ass
    assert "Dialogue: 0,0:00:00.20,0:00:01.00," in ass
    assert "Style: Default,Impact,127," in ass
    assert ",2,50,50,768,1" in ass


def test_cached_transcript_attachment_preserves_visuals_and_is_atomic():
    plan = ShotPlan(Path("voice.wav"), [Scene(0, 1, "dog", asset="dog.mp4", asset_kind="video", caption="Собака уснула")])
    words = [Word(0, .2, "Собака"), Word(.2, 1, "уснула")]
    attach_caption_timings(plan, Transcript(words))
    assert plan.scenes[0].caption_words == words
    assert (plan.scenes[0].asset, plan.scenes[0].start, plan.scenes[0].end) == ("dog.mp4", 0, 1)
    with pytest.raises(ValueError, match="does not match"):
        attach_caption_timings(plan, Transcript([Word(0, 1, "Другой")]))
    assert plan.scenes[0].caption_words == words


def test_both_planners_preserve_original_timestamps():
    transcript = Transcript([Word(i * .5, (i + 1) * .5, text) for i, text in enumerate(
        "Собака гуляет спокойно. Потом собака внезапно уснула дома.".split())])
    with patch("video_ai.director._apply_gemini_direction", return_value=("rules", None)):
        legacy = build_shot_plan(transcript, "voice.wav")
    with patch("video_ai.gemini_ai.get_gemini_client") as get_client:
        get_client.return_value.last_model = "test"
        get_client.return_value._generate_json.return_value = {"beats": [
            {"start_idx": 0, "end_idx": 3, "kind": "video", "query": "dog walking"},
            {"start_idx": 4, "end_idx": 7, "kind": "video", "query": "dog sleeping"}]}
        donor = build_donor_shot_plan(transcript, "voice.wav")
    for plan in [legacy, donor]:
        assert [w for s in plan.scenes for w in s.caption_words] == transcript.words
        for scene in plan.scenes:
            pages = _caption_pages(scene.caption, scene.start, scene.end, timed_words=scene.caption_words)
            assert all(1 <= len(text.split()) <= 2 for _, _, text in pages)
            assert pages[0][0] == scene.caption_words[0].start


def test_attach_caption_timings_keeps_embedded_words_for_same_transcript():
    words = [Word(0, .2, "Собака"), Word(.2, 1.05, "уснула")]
    plan = ShotPlan(
        Path("voice.wav"),
        [Scene(0, 1, "dog", caption="Собака уснула", caption_words=list(words))],
    )
    attach_caption_timings(plan, Transcript(list(words)))
    assert plan.scenes[0].caption_words == words


def test_attach_caption_timings_still_rejects_different_transcript_with_embedded_words():
    saved = [Word(0, .2, "Собака"), Word(.2, 1.05, "уснула")]
    plan = ShotPlan(
        Path("voice.wav"),
        [Scene(0, 1, "dog", caption="Собака уснула", caption_words=list(saved))],
    )
    with pytest.raises(ValueError, match="does not match"):
        attach_caption_timings(
            plan,
            Transcript([Word(0, .2, "Собака"), Word(.2, 1.05, "проснулась")]),
        )
    assert plan.scenes[0].caption_words == saved
