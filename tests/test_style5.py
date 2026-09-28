from pathlib import Path
from unittest.mock import patch

from video_ai.dynamic_reactions import place_style5_emojis
from video_ai.models import Scene, ShotPlan, Transcript, Word
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


def test_style5_preserves_gemini_scene_count_without_midpoint_injection():
    from video_ai.broll_planner import _style5_start_indices

    planned = [
        {"start_idx": 0, "end_idx": 4},
        {"start_idx": 5, "end_idx": 19},
        {"start_idx": 20, "end_idx": 29},
    ]
    assert _style5_start_indices(planned, 30) == [0, 5, 20]


def test_style5_director_prompt_has_no_target_count_or_meme_catalog(tmp_path):
    from video_ai.broll_planner import build_donor_shot_plan

    class FakeClient:
        last_model = "fake-gemini"

        def __init__(self):
            self.prompt = ""

        def _generate_json(self, parts, *, temperature):
            self.prompt = parts[0]["text"]
            return {
                "beats": [{
                    "start_idx": 0,
                    "end_idx": 3,
                    "kind": "meme",
                    "query": "awkward nervous reaction",
                    "alternatives": ["embarrassed reaction"],
                    "visual_description": "A visibly awkward nervous reaction.",
                    "reason": "Punchline is better shown as a reaction meme.",
                }]
            }

    client = FakeClient()
    transcript = Transcript([
        Word(0.0, 0.4, "Мне"),
        Word(0.4, 0.8, "очень"),
        Word(0.8, 1.2, "неловко"),
        Word(1.2, 1.6, "сейчас"),
    ])
    meme_dir = tmp_path / "memes"
    meme_dir.mkdir()
    (meme_dir / "DO_NOT_SEND_THIS_FILENAME.mp4").write_bytes(b"x")

    with patch("video_ai.gemini_ai.get_gemini_client", return_value=client):
        plan = build_donor_shot_plan(
            transcript,
            tmp_path / "voice.mp3",
            meme_dir=meme_dir,
            style5=True,
        )

    assert len(plan.scenes) == 1
    assert plan.scenes[0].visual_mode == "meme"
    assert plan.scenes[0].meme_filename is None
    assert "There is NO target scene count" in client.prompt
    assert "Target roughly" not in client.prompt
    assert "AVAILABLE LOCAL MEMES" not in client.prompt
    assert "DO_NOT_SEND_THIS_FILENAME" not in client.prompt
    assert "meme_filename" not in client.prompt


def test_local_meme_search_uses_verified_index_descriptions(tmp_path):
    import json
    from video_ai.meme_library import search_memes

    root = tmp_path / "memes"
    root.mkdir()
    target = root / "VID_20230731_023329_878.mp4"
    target.write_bytes(b"video")
    distractor = root / "awkward_name_only.mp4"
    distractor.write_bytes(b"video")

    cache = root / ".video-ai-index"
    cache.mkdir()
    (cache / "record.json").write_text(json.dumps({
        "verified": True,
        "name": target.name,
        "path": r"C:\\old-copy\\memes\\VID_20230731_023329_878.mp4",
        "description": "nervous embarrassed man sweating, awkward lying excuse reaction",
    }), encoding="utf-8")

    found = search_memes(root, "awkward lying excuse", limit=2)
    assert found
    assert found[0].path == target
    assert "awkward lying excuse" in found[0].description


def test_style5_meme_shortlist_balances_local_and_giphy(tmp_path):
    from video_ai.assets import AssetCandidate, _rank_meme_shortlist_with_gemini

    local = tmp_path / "local.mp4"
    web = tmp_path / "web.mp4"
    local.write_bytes(b"x" * 2048)
    web.write_bytes(b"x" * 2048)

    scene = Scene(
        0, 2, "awkward reaction",
        caption="Неловкая отмазка",
        visual_mode="meme",
        source_mode="meme_library",
        visual_description="awkward nervous reaction",
    )
    ranked = [
        (
            AssetCandidate(
                "local reaction", "", f"local:{local}", "video/mp4",
                0, 0, local.stat().st_size, "video",
                source="local_meme", local_path=str(local), score=100,
            ),
            "local meme library",
        ),
        (
            AssetCandidate(
                "giphy reaction", "", "https://example.invalid/giphy.mp4", "video/mp4",
                0, 0, 0, "video",
                source="giphy_meme", score=90,
            ),
            "Giphy: awkward reaction",
        ),
    ]

    class FakeGemini:
        def choose_visual_candidates(self, scene, rows, mode="exact"):
            sources = {row["source"] for row in rows}
            assert sources == {"local_meme", "giphy_meme"}
            return [{"index": 1, "fit": 92, "reason": "best visible reaction"}]

    def fake_run(command, **kwargs):
        # The helper only needs a decodable-preview result in this unit test.
        out = Path(command[-1])
        out.write_bytes(b"jpg" * 300)
        class Result:
            returncode = 0
            stdout = b""
            stderr = b""
        return Result()

    with patch("video_ai.assets.subprocess.run", side_effect=fake_run), \
         patch("video_ai.assets._download", side_effect=lambda url, target: target.write_bytes(b"x" * 2048)):
        ordered = _rank_meme_shortlist_with_gemini(
            scene,
            ranked,
            tmp_path / "work",
            index=0,
            gemini=FakeGemini(),
            prefix="[test]",
        )

    assert ordered
    assert ordered[0][0].source == "local_meme"
    assert ordered[0][0].score == 92
