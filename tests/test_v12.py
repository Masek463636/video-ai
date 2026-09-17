from pathlib import Path

from video_ai.io import load_shot_plan, save_shot_plan
from video_ai.models import Scene, ShotPlan
from video_ai.quality_guard import infer_tone, local_quality_guard


def test_tragic_tone_detection():
    assert infer_tone("восстание унесло до 30 миллионов жизней") == "tragic"


def test_stress_is_tense():
    assert infer_tone("от стресса он начал паниковать") == "tense"


def test_design_word_does_not_trigger_sign_filter(tmp_path: Path):
    fake = tmp_path / "fake.jpg"
    fake.write_bytes(b"x" * 2048)
    result = local_quality_guard(fake, title="Historic costume design", width=1200, height=900, kind="image")
    assert result.accept
    assert not any("metadata:sign" == issue for issue in result.issues)


def test_low_resolution_is_penalized(tmp_path: Path):
    fake = tmp_path / "fake.jpg"
    fake.write_bytes(b"x" * 2048)
    result = local_quality_guard(fake, title="clean portrait", width=320, height=240, kind="image")
    assert not result.accept
    assert "low_resolution" in result.issues


def test_tone_survives_shotplan_roundtrip(tmp_path: Path):
    audio = tmp_path / "voice.mp3"
    audio.write_bytes(b"fake")
    plan = ShotPlan(audio=audio, scenes=[Scene(start=0.0, end=1.5, query="aftermath", caption="millions died", tone="tragic")])
    path = save_shot_plan(plan, tmp_path / "plan.json")
    loaded = load_shot_plan(path)
    assert loaded.scenes[0].tone == "tragic"
