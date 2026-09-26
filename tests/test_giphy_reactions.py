import io
import json
from pathlib import Path
from unittest.mock import patch

from video_ai.classic_reactions import build_reactions
from video_ai.models import Scene, ShotPlan
from video_ai.shorts_fx import _find_giphy_candidates


class Client:
    def __init__(self, replies):
        self.replies = iter(replies)

    def _generate_json(self, *args, **kwargs):
        reply = next(self.replies)
        if isinstance(reply, BaseException):
            raise reply
        return reply


def plan():
    return ShotPlan(
        Path("dummy.wav"),
        [
            Scene(
                0,
                3,
                "test",
                caption="This is a funny moment",
            )
        ],
    )


def planner():
    return {
        "effects": [{
            "scene": 0,
            "anchor": "funny moment",
            "query": "laughing reaction",
            "pack": "stickers",
            "duration": .60,
        }]
    }


def accepted(kind="emoji"):
    return {
        "readable": True,
        "relevant": True,
        "kind": kind,
        "reason": "clear reaction",
    }


def rejected():
    return {
        "readable": False,
        "relevant": False,
        "kind": "emoji",
        "reason": "wrong reaction",
    }


def test_local_accepted_never_calls_giphy(tmp_path):

    pack = tmp_path / "stickers"
    pack.mkdir()

    local = pack / "local.mp4"
    local.write_bytes(b"x" * 2048)

    client = Client([
        planner(),
        accepted(),
    ])

    with (
        patch(
            "video_ai.gemini_ai.get_gemini_client",
            return_value=client,
        ),
        patch(
            "video_ai.shorts_fx._find_local_sticker_by_prompt",
            return_value=local,
        ),
        patch(
            "video_ai.story_media.preview_parts",
            return_value=([{"text": "preview"}], 1),
        ),
        patch(
            "video_ai.shorts_fx._find_giphy_candidates",
        ) as giphy,
    ):

        effects = build_reactions(
            plan(),
            tmp_path / "out",
            sticker_dir=pack,
        )

    giphy.assert_not_called()

    assert len(effects) == 1
    assert effects[0]["source"] == "local_reaction"


def test_local_rejected_giphy_accepted(tmp_path):

    pack = tmp_path / "stickers"
    pack.mkdir()

    local = pack / "local.mp4"
    local.write_bytes(b"x" * 2048)

    giphy = tmp_path / "giphy_test.mp4"
    giphy.write_bytes(b"x" * 2048)

    client = Client([
        planner(),
        rejected(),
        accepted("meme"),
    ])

    with (
        patch(
            "video_ai.gemini_ai.get_gemini_client",
            return_value=client,
        ),
        patch(
            "video_ai.shorts_fx._find_local_sticker_by_prompt",
            side_effect=[local, None],
        ),
        patch(
            "video_ai.shorts_fx._find_giphy_candidates",
            return_value=[{
                "path": giphy,
                "id": "abc",
                "url": "https://giphy.com/gifs/abc",
                "mp4_url": "https://media.giphy.com/abc.mp4",
            }],
        ),
        patch(
            "video_ai.story_media.preview_parts",
            return_value=([{"text": "preview"}], 1),
        ),
    ):

        effects = build_reactions(
            plan(),
            tmp_path / "out",
            sticker_dir=pack,
        )

    assert len(effects) == 1
    assert effects[0]["source"] == "giphy"
    assert effects[0]["giphy_id"] == "abc"


def test_rejected_first_giphy_uses_second(tmp_path):

    pack = tmp_path / "stickers"
    pack.mkdir()

    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"

    first.write_bytes(b"x" * 2048)
    second.write_bytes(b"x" * 2048)

    client = Client([
        planner(),
        rejected(),
        accepted("meme"),
    ])

    with (
        patch(
            "video_ai.gemini_ai.get_gemini_client",
            return_value=client,
        ),
        patch(
            "video_ai.shorts_fx._find_local_sticker_by_prompt",
            return_value=None,
        ),
        patch(
            "video_ai.shorts_fx._find_giphy_candidates",
            return_value=[
                {
                    "path": first,
                    "id": "first",
                    "url": "",
                    "mp4_url": "",
                },
                {
                    "path": second,
                    "id": "second",
                    "url": "",
                    "mp4_url": "",
                },
            ],
        ),
        patch(
            "video_ai.story_media.preview_parts",
            return_value=([{"text": "preview"}], 1),
        ),
    ):

        effects = build_reactions(
            plan(),
            tmp_path / "out",
            sticker_dir=pack,
        )

    assert len(effects) == 1
    assert effects[0]["giphy_id"] == "second"


def test_all_giphy_rejected(tmp_path):

    pack = tmp_path / "stickers"
    pack.mkdir()

    first = tmp_path / "first.mp4"
    first.write_bytes(b"x" * 2048)

    client = Client([
        planner(),
        rejected(),
    ])

    with (
        patch(
            "video_ai.gemini_ai.get_gemini_client",
            return_value=client,
        ),
        patch(
            "video_ai.shorts_fx._find_local_sticker_by_prompt",
            return_value=None,
        ),
        patch(
            "video_ai.shorts_fx._find_giphy_candidates",
            return_value=[{
                "path": first,
                "id": "first",
                "url": "",
                "mp4_url": "",
            }],
        ),
        patch(
            "video_ai.story_media.preview_parts",
            return_value=([{"text": "preview"}], 1),
        ),
    ):

        effects = build_reactions(
            plan(),
            tmp_path / "out",
            sticker_dir=pack,
        )

    assert effects == []


def test_no_api_key_returns_empty(tmp_path, monkeypatch):

    monkeypatch.delenv(
        "GIPHY_API_KEY",
        raising=False,
    )

    assert _find_giphy_candidates(
        "laughing reaction",
        tmp_path,
        set(),
    ) == []


def test_network_error_returns_empty(tmp_path, monkeypatch):

    monkeypatch.setenv(
        "GIPHY_API_KEY",
        "TEST",
    )

    with patch(
        "urllib.request.urlopen",
        side_effect=OSError("network down"),
    ):

        assert _find_giphy_candidates(
            "laughing reaction",
            tmp_path,
            set(),
        ) == []


def test_giphy_helper_downloads_mp4(tmp_path, monkeypatch):

    monkeypatch.setenv(
        "GIPHY_API_KEY",
        "TEST",
    )

    payload = {
        "data": [{
            "id": "giphy123",
            "url": "https://giphy.com/gifs/giphy123",
            "images": {
                "fixed_height": {
                    "mp4":
                    "https://media.giphy.com/giphy123.mp4"
                }
            },
        }]
    }

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(
            self,
            exc_type,
            exc,
            tb,
        ):
            self.close()


    def fake_download(url, target):
        Path(target).write_bytes(
            b"x" * 2048
        )


    with (
        patch(
            "urllib.request.urlopen",
            return_value=Response(
                json.dumps(payload).encode("utf-8")
            ),
        ),
        patch(
            "video_ai.shorts_fx._download",
            side_effect=fake_download,
        ),
    ):

        candidates = _find_giphy_candidates(
            "funny laughing reaction",
            tmp_path,
            set(),
            limit=3,
        )

    assert len(candidates) == 1

    candidate = candidates[0]

    assert candidate["id"] == "giphy123"
    assert candidate["path"].name == "giphy_giphy123.mp4"
    assert candidate["path"].stat().st_size == 2048


def test_one_review_error_does_not_kill_reaction_pipeline(tmp_path):
    pack = tmp_path / "stickers"
    pack.mkdir()

    first = pack / "broken.mp4"
    second = pack / "good.mp4"

    first.write_bytes(b"x" * 2048)
    second.write_bytes(b"x" * 2048)

    client = Client([
        planner(),
        RuntimeError("temporary Gemini failure"),
        accepted("meme"),
    ])

    with (
        patch(
            "video_ai.gemini_ai.get_gemini_client",
            return_value=client,
        ),
        patch(
            "video_ai.shorts_fx._find_local_sticker_by_prompt",
            side_effect=[first, second],
        ),
        patch(
            "video_ai.story_media.preview_parts",
            return_value=([{"text": "preview"}], 1),
        ),
    ):
        effects = build_reactions(
            plan(),
            tmp_path / "out",
            sticker_dir=pack,
        )

    assert len(effects) == 1
    assert effects[0]["asset"] == str(second.resolve())

    report = json.loads(
        (tmp_path / "out" / "overlays.json").read_text(encoding="utf-8")
    )

    assert report["status"] == "planned"
    assert report["decisions"][0]["candidates"][0]["status"] == "review_error"
