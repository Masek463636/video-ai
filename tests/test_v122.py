from pathlib import Path

from video_ai.assets import ensure_visual_coverage
from video_ai.models import Scene, ShotPlan


def test_recovery_does_not_leave_unlocked_blank_when_neighbor_exists(tmp_path: Path):
    audio = tmp_path / "voice.mp3"
    audio.write_bytes(b"fake")
    asset = tmp_path / "safe.jpg"
    asset.write_bytes(b"not-real-image-but-existing")

    scenes = [
        Scene(start=0.0, end=1.5, query="safe", caption="neutral beat", asset=str(asset), asset_kind="image", tone="neutral"),
        Scene(start=1.5, end=3.0, query="missing", caption="another neutral beat", asset=None, asset_kind="blank", tone="neutral"),
    ]
    plan = ShotPlan(audio=audio, scenes=scenes)

    filled = ensure_visual_coverage(plan, [])

    assert 1 in filled
    assert plan.scenes[1].asset == str(asset)
    assert plan.scenes[1].asset_kind == "image"


def test_recovery_prefers_nonrecent_asset_when_possible(tmp_path: Path):
    audio = tmp_path / "voice.mp3"
    audio.write_bytes(b"fake")
    a = tmp_path / "a.jpg"
    b = tmp_path / "b.jpg"
    a.write_bytes(b"a")
    b.write_bytes(b"b")

    scenes = [
        Scene(start=0.0, end=1.0, query="a", caption="one", asset=str(a), asset_kind="image", tone="neutral"),
        Scene(start=1.0, end=2.0, query="b", caption="two", asset=str(b), asset_kind="image", tone="neutral"),
        Scene(start=2.0, end=3.0, query="a2", caption="three", asset=str(a), asset_kind="image", tone="neutral"),
        Scene(start=3.0, end=4.0, query="missing", caption="four", asset=None, asset_kind="blank", tone="neutral"),
    ]
    plan = ShotPlan(audio=audio, scenes=scenes)

    ensure_visual_coverage(plan, [])

    # The recovery path should try not to immediately reuse the most recent asset.
    assert plan.scenes[3].asset == str(b)


def test_locked_scene_prefers_contextual_period_recovery_over_black(tmp_path: Path):
    audio = tmp_path / "voice.mp3"
    audio.write_bytes(b"fake")
    china = tmp_path / "china.jpg"
    china.write_bytes(b"x")

    scenes = [
        Scene(
            start=0.0, end=1.0, query="China",
            caption="China nineteenth century",
            asset=str(china), asset_kind="image", tone="neutral",
            required_context=["China", "19th century"],
            visual_description="19th century China",
        ),
        Scene(
            start=1.0, end=2.0, query="Hong Xiuquan",
            caption="Hong Xiuquan",
            asset=None, asset_kind="blank", tone="neutral",
            semantic_lock=True,
            required_entities=["Hong Xiuquan"],
            required_context=["China", "19th century"],
        ),
    ]
    plan = ShotPlan(audio=audio, scenes=scenes)

    filled = ensure_visual_coverage(plan, [])

    assert 1 in filled
    assert plan.scenes[1].asset == str(china)
