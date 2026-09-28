from pathlib import Path
from unittest.mock import patch

from video_ai.models import Scene, ShotPlan
from video_ai.shorts_fx import build_shorts_overlays
from video_ai.renderer import _ass_text


def test_style5_uses_dynamic_reaction_planner_in_montage_mode(tmp_path):
    plan = ShotPlan(Path("voice.wav"), [Scene(0, 2, "", caption="Вот это да")])
    with patch("video_ai.dynamic_reactions.build_reactions", return_value=[]) as build:
        result = build_shorts_overlays(
            plan,
            tmp_path,
            editing_style="style5",
            sticker_dir=tmp_path / "stickers",
        )
    assert result == []
    assert build.call_args.kwargs["montage"] is True


def test_style5_keeps_dynamic_caption_bounce():
    dynamic = _ass_text("Привет", editing_polish=True, editing_style="dynamic")
    style5 = _ass_text("Привет", editing_polish=True, editing_style="style5")
    assert style5 == dynamic
    assert r"\fscx40" in style5
