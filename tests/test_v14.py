from pathlib import Path

from video_ai.models import Scene, ShotPlan
from video_ai.reference_style import apply_reference_style


def test_reference_style_prefers_video_for_unlocked_scene() -> None:
    plan = ShotPlan(
        audio=Path("voice.mp3"),
        scenes=[
            Scene(
                start=0.0,
                end=2.0,
                query="man waking",
                caption="Он резко проснулся после странного сна",
                visual_mode="image",
                source_mode="generic_image",
                motion_preset="slow_push",
            )
        ],
    )
    changed = apply_reference_style(plan)
    scene = plan.scenes[0]
    assert changed == [0]
    assert scene.visual_mode == "video"
    assert scene.source_mode == "stock_video"
    assert scene.motion_preset == "none"
    assert scene.search_queries[0].startswith("man waking up suddenly")


def test_reference_style_preserves_hard_historical_lock() -> None:
    plan = ShotPlan(
        audio=Path("voice.mp3"),
        scenes=[
            Scene(
                start=0.0,
                end=2.0,
                query="Hong Xiuquan",
                caption="Хун Сюцюань",
                visual_mode="video",
                source_mode="stock_video",
                semantic_lock=True,
                required_entities=["Hong Xiuquan"],
            )
        ],
    )
    changed = apply_reference_style(plan)
    scene = plan.scenes[0]
    assert changed == []
    assert scene.visual_mode == "image"
    assert scene.source_mode == "historical_archive"
