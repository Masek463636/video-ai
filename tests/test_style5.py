from pathlib import Path
from unittest.mock import patch

from video_ai.dynamic_reactions import place_style5_emojis
from video_ai.models import Scene, ShotPlan
from video_ai.shorts_fx import build_shorts_overlays
from video_ai.renderer import _ass_text


def test_style5_uses_dynamic_reaction_planner_in_emoji_only_mode(tmp_path):
    plan = ShotPlan(Path("voice.wav"), [Scene(0, 2, "", caption="Вот это да")])
    with patch("video_ai.dynamic_reactions.build_reactions", return_value=[]) as build:
        result = build_shorts_overlays(
            plan,
            tmp_path,
            editing_style="style5",
            sticker_dir=tmp_path / "stickers",
        )
    assert result == []
    assert build.call_args.kwargs["emoji_only"] is True
    assert build.call_args.kwargs["montage"] is False


def test_style5_keeps_dynamic_caption_bounce():
    dynamic = _ass_text("Привет", editing_polish=True, editing_style="dynamic")
    style5 = _ass_text("Привет", editing_polish=True, editing_style="style5")
    assert style5 == dynamic
    assert r"\fscx40" in style5


def test_style5_emoji_placement_preserves_directional_entry(tmp_path):
    asset = tmp_path / "emoji.gif"
    asset.write_bytes(b"x")
    items = [{
        "type": "sticker",
        "asset": str(asset),
        "reaction_kind": "emoji",
        "source": "local_reaction",
        "reaction_verified": True,
        "start": 1.0,
        "end": 2.0,
        "animation": "fly",
        "position": "center",
    }]
    placed = place_style5_emojis(items, 3.0)
    assert len(placed) == 1
    assert placed[0]["animation"] == "fly"
    assert placed[0]["position"] == "center"
    assert "layout_box" not in placed[0]


def test_style5_internet_meme_candidate_enters_primary_pool(tmp_path):
    from video_ai.assets import _build_ranked_pool
    from video_ai.meme_library import GiphyMemeAsset

    scene = Scene(
        0,
        2,
        "awkward excuse reaction",
        caption="Не, я недавно ел",
        visual_mode="meme",
        source_mode="meme_library",
        visual_description="awkward lying excuse reaction",
    )
    fake = GiphyMemeAsset(
        title="awkward reaction",
        download_url="https://example.invalid/reaction.mp4",
        page_url="https://giphy.com/gifs/example",
        preview_url="https://example.invalid/reaction.gif",
        score=10.0,
    )
    with patch("video_ai.assets.search_giphy_memes", return_value=[fake]):
        ranked, _ = _build_ranked_pool(
            scene,
            limit=20,
            meme_dir=tmp_path / "missing-memes",
            used_urls=set(),
            used_titles=[],
            semantic=False,
            semantic_top_k=6,
        )

    assert ranked
    assert ranked[0][0].source == "giphy_meme"
    assert ranked[0][0].kind == "video"
