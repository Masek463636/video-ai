from pathlib import Path

from video_ai.gemini_session import GeminiEditorSession


def test_session_persists_turns_and_memory(tmp_path: Path):
    path = tmp_path / "gemini-session.json"
    session = GeminiEditorSession(path, max_turns=4)

    session.record(
        [{"text": "Plan the whole Short."}],
        {
            "scenes": [
                {
                    "index": 0,
                    "visual_mode": "video",
                    "visual_description": "close-up of a ringing phone",
                }
            ]
        },
        model="gemini-test",
    )

    reopened = GeminiEditorSession(path, max_turns=4)
    contents = reopened.prior_contents()
    joined = "\n".join(
        str(part.get("text") or "")
        for row in contents
        for part in row.get("parts", [])
        if isinstance(part, dict)
    )
    assert "ringing phone" in joined
    assert reopened.stats()["turns"] == 1
    assert reopened.stats()["memory"] == 1


def test_session_does_not_persist_inline_image_bytes(tmp_path: Path):
    path = tmp_path / "gemini-session.json"
    session = GeminiEditorSession(path)

    secret_blob = "A" * 5000
    session.record(
        [
            {"text": "Choose the best frame."},
            {"inline_data": {"mime_type": "image/jpeg", "data": secret_blob}},
        ],
        {"choices": [{"scene": 1, "index": 2, "fit": 91, "reason": "best action"}]},
        model="gemini-test",
    )

    raw = path.read_text(encoding="utf-8")
    assert secret_blob not in raw
    assert "1 image preview" in raw
