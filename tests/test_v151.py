from pathlib import Path

from video_ai.assets import ensure_visual_coverage
from video_ai.models import Scene, ShotPlan


def test_coverage_reuse_function_still_works_for_legacy_mode() -> None:
    plan = ShotPlan(
        audio=Path("voice.mp3"),
        scenes=[
            Scene(start=0.0, end=1.0, query="a", asset="same.mp4", asset_kind="video"),
            Scene(start=1.0, end=2.0, query="b"),
        ],
    )
    # The function itself remains available for legacy callers; donor mode
    # disables calling it at the materialization layer.
    assert isinstance(ensure_visual_coverage(plan), set)
