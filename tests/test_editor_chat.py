from pathlib import Path

from video_ai.editor_chat import EditorChat


def test_editor_chat_persists_conversations_and_memory(tmp_path: Path):
    db = tmp_path / "chat.sqlite3"
    chat = EditorChat(db, api_key="fake")
    conversation = chat.create_conversation(job_id="job-1")

    assert conversation["job_id"] == "job-1"
    assert chat.list_conversations()[0]["id"] == conversation["id"]

    chat._remember_if_useful("Запомни: мне не нравятся случайные мемы без смысла.")
    assert chat.memories() == ["Запомни: мне не нравятся случайные мемы без смысла."]

    reopened = EditorChat(db, api_key="fake")
    assert reopened.list_conversations()[0]["id"] == conversation["id"]
    assert reopened.memories() == ["Запомни: мне не нравятся случайные мемы без смысла."]


def test_editor_chat_does_not_store_every_message_as_memory(tmp_path: Path):
    chat = EditorChat(tmp_path / "chat.sqlite3", api_key="fake")
    chat._remember_if_useful("Просто привет")
    assert chat.memories() == []
