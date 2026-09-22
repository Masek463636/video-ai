from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest

from video_ai.broll_planner import build_donor_shot_plan, _validate_semantic_beats
from video_ai.cli import main
from video_ai.models import Transcript, Word
from video_ai.transcript import save_transcript


def sample():
    transcript = Transcript([Word(i * .5, (i + 1) * .5, f"word{i}") for i in range(12)])
    beats = [
        {"start_idx": 0, "end_idx": 7, "kind": "video", "query": "dog falling asleep",
         "visual_description": "A dog gradually falls asleep", "reason": "Hold to see the complete action"},
        {"start_idx": 8, "end_idx": 11, "kind": "image", "query": "historical document",
         "visual_description": "An archival document"},
    ]
    return transcript, beats


def test_semantic_hold_keeps_own_intent_and_all_words_without_midpoint():
    transcript, beats = sample()
    with patch("video_ai.gemini_ai.get_gemini_client") as get_client:
        client = get_client.return_value
        client.last_model = "mock"
        client._generate_json.return_value = {"beats": beats}
        semantic = build_donor_shot_plan(transcript, "voice.wav", semantic_rhythm=True)
        prompt = client._generate_json.call_args.args[0][0]["text"]
        stable = build_donor_shot_plan(transcript, "voice.wav")
    assert len(semantic.scenes) == 2
    assert semantic.scenes[0].end == 4
    assert semantic.scenes[1].visual_description == "An archival document"
    assert len(stable.scenes) == 3  # Existing timer-split behavior remains the default.
    assert [w for s in semantic.scenes for w in s.caption_words] == transcript.words
    assert semantic.director_source == "donor_gemini"  # Retains strict donor QC/rescue.
    assert "0 [0.000-0.500s]: word0" in prompt
    assert "no shot-count quota" in prompt
    assert "complete personal names" in prompt


@pytest.mark.parametrize("mutation", [
    lambda b: b[0].update(start_idx=1),
    lambda b: b[1].update(start_idx=7),
    lambda b: b[1].update(start_idx=9),
    lambda b: b[1].update(end_idx=10),
    lambda b: b[1].update(end_idx=12),
    lambda b: b[1].update(start_idx="8"),
    lambda b: b[0].update(reason=""),
    lambda b: b[1].update(visual_description=" A DOG gradually falls asleep. "),
    lambda b: b[1].update(query=""),
    lambda b: b[1].update(kind="blank"),
])
def test_invalid_plan_is_rejected_without_invented_cuts(mutation):
    transcript, beats = sample()
    mutation(beats)
    with pytest.raises(ValueError):
        _validate_semantic_beats(beats, transcript)


@pytest.mark.parametrize("boundary", [1, 11])
def test_short_or_overlong_beats_rejected(boundary):
    transcript, beats = sample()
    beats[0]["end_idx"] = boundary - 1
    beats[1]["start_idx"] = boundary
    with pytest.raises(ValueError, match="outside"):
        _validate_semantic_beats(beats, transcript)


def test_valid_semantic_plan_does_not_mutate_model_response():
    transcript, beats = sample()
    original = deepcopy(beats)
    result = _validate_semantic_beats(beats, transcript)
    assert beats == original
    assert result == beats
    assert result[0] is not beats[0]


def test_cli_experiment_reuses_transcript_and_stops_before_search(tmp_path):
    transcript, _ = sample()
    path = save_transcript(transcript, tmp_path / "input.json")
    argv = ["video-ai", "create", "voice.mp3", "--transcript", str(path),
            "--editing-rhythm", "semantic", "-o", str(tmp_path / "out.mp4"),
            "--work-dir", str(tmp_path / "work")]
    with patch("sys.argv", argv), patch("video_ai.cli.transcribe_local") as asr, \
            patch("video_ai.cli.build_donor_shot_plan", side_effect=ValueError("invalid beat")) as planner, \
            patch("video_ai.cli.build_shot_plan") as legacy, \
            patch("video_ai.cli._resolve_and_repair") as search:
        with pytest.raises(RuntimeError, match="stopped before asset search"):
            main()
    asr.assert_not_called()
    legacy.assert_not_called()
    search.assert_not_called()
    assert planner.call_args.kwargs["semantic_rhythm"] is True
    assert planner.call_args.args[0].words == transcript.words
    assert not (tmp_path / "out.mp4").exists()
