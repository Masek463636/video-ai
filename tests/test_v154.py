from pathlib import Path

from video_ai.assets import ensure_visual_coverage
from video_ai.models import Scene, ShotPlan


def test_never_black_reuses_existing_visual(tmp_path: Path) -> None:
    asset = tmp_path / "good.mp4"
    asset.write_bytes(b"x" * 2048)
    plan = ShotPlan(
        audio=tmp_path / "voice.mp3",
        scenes=[
            Scene(0.0, 1.0, "milk", asset=str(asset), asset_kind="video"),
            Scene(1.0, 2.0, "supermarket"),
            Scene(2.0, 3.0, "shopping"),
        ],
    )
    filled = ensure_visual_coverage(plan, [])
    assert filled == {1, 2}
    assert all(scene.asset_kind != "blank" for scene in plan.scenes)
    assert all(scene.asset for scene in plan.scenes[1:])


def test_never_black_keeps_existing_asset(tmp_path: Path) -> None:
    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"
    first.write_bytes(b"a" * 2048)
    second.write_bytes(b"b" * 2048)
    plan = ShotPlan(
        audio=tmp_path / "voice.mp3",
        scenes=[
            Scene(0.0, 1.0, "one", asset=str(first), asset_kind="video"),
            Scene(1.0, 2.0, "two", asset=str(second), asset_kind="video"),
        ],
    )
    filled = ensure_visual_coverage(plan, [])
    assert filled == set()
    assert plan.scenes[1].asset == str(second)
