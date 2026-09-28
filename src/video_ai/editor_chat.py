from __future__ import annotations

import json
import os
import re
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

_API_ROOT = "https://generativelanguage.googleapis.com/v1beta"
_MEMORY_HINTS = (
    "запомни",
    "мне нравится",
    "мне не нравится",
    "не используй",
    "всегда",
    "никогда",
    "предпочитаю",
    "remember",
    "i like",
    "i don't like",
    "never use",
    "always",
)


class EditorChat:
    """Persistent local chat for the AI editor.

    Conversation history and durable preferences live in SQLite on the user's
    computer. Gemini only receives a bounded recent slice plus compact memories,
    so the chat survives browser/app restarts without depending on provider-side
    retention.
    """

    def __init__(
        self,
        db_path: Path,
        *,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.api_key = (api_key or os.getenv("GEMINI_API_KEY", "")).strip()
        preferred = (model or os.getenv("GEMINI_CHAT_MODEL", "")).strip()
        self.models = []
        for candidate in (
            preferred or "gemini-3.5-flash-lite",
            "gemini-3.5-flash",
        ):
            if candidate and candidate not in self.models:
                self.models.append(candidate)
        self._init_db()

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path, timeout=20)
        con.row_factory = sqlite3.Row
        return con

    def _init_db(self) -> None:
        with self._connect() as con:
            con.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    job_id TEXT,
                    created REAL NOT NULL,
                    updated REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user','assistant')),
                    content TEXT NOT NULL,
                    created REAL NOT NULL,
                    FOREIGN KEY(conversation_id) REFERENCES conversations(id)
                );
                CREATE INDEX IF NOT EXISTS idx_messages_conversation
                    ON messages(conversation_id, id);
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL UNIQUE,
                    created REAL NOT NULL,
                    updated REAL NOT NULL
                );
                """
            )

    def create_conversation(self, *, job_id: str | None = None) -> dict[str, Any]:
        import secrets

        now = time.time()
        conversation_id = secrets.token_hex(8)
        with self._connect() as con:
            con.execute(
                "INSERT INTO conversations(id,title,job_id,created,updated) VALUES(?,?,?,?,?)",
                (conversation_id, "Новый разговор", job_id or None, now, now),
            )
        return {
            "id": conversation_id,
            "title": "Новый разговор",
            "job_id": job_id,
            "created": now,
            "updated": now,
        }

    def list_conversations(self, *, limit: int = 30) -> list[dict[str, Any]]:
        with self._connect() as con:
            rows = con.execute(
                """
                SELECT c.id,c.title,c.job_id,c.created,c.updated,
                       (SELECT COUNT(*) FROM messages m WHERE m.conversation_id=c.id) AS message_count
                FROM conversations c
                ORDER BY c.updated DESC
                LIMIT ?
                """,
                (max(1, min(int(limit), 100)),),
            ).fetchall()
        return [dict(row) for row in rows]

    def messages(self, conversation_id: str, *, limit: int = 80) -> list[dict[str, Any]]:
        with self._connect() as con:
            rows = con.execute(
                """
                SELECT role,content,created
                FROM (
                    SELECT id,role,content,created
                    FROM messages
                    WHERE conversation_id=?
                    ORDER BY id DESC
                    LIMIT ?
                )
                ORDER BY created ASC
                """,
                (conversation_id, max(1, min(int(limit), 200))),
            ).fetchall()
        return [dict(row) for row in rows]

    def memories(self, *, limit: int = 24) -> list[str]:
        with self._connect() as con:
            rows = con.execute(
                "SELECT content FROM memories ORDER BY updated DESC LIMIT ?",
                (max(1, min(int(limit), 100)),),
            ).fetchall()
        return [str(row["content"]) for row in rows]

    def send(
        self,
        conversation_id: str,
        message: str,
        *,
        job_context: str = "",
    ) -> dict[str, Any]:
        if not self.available:
            raise RuntimeError("GEMINI_API_KEY не настроен")
        message = str(message or "").strip()
        if not message:
            raise ValueError("Напиши сообщение")
        if len(message) > 6000:
            raise ValueError("Сообщение слишком длинное")

        with self._connect() as con:
            row = con.execute(
                "SELECT id,title,job_id FROM conversations WHERE id=?",
                (conversation_id,),
            ).fetchone()
            if row is None:
                raise ValueError("Разговор не найден")
            now = time.time()
            con.execute(
                "INSERT INTO messages(conversation_id,role,content,created) VALUES(?,?,?,?)",
                (conversation_id, "user", message, now),
            )
            if row["title"] == "Новый разговор":
                title = re.sub(r"\s+", " ", message).strip()[:52] or "Разговор"
                con.execute(
                    "UPDATE conversations SET title=?,updated=? WHERE id=?",
                    (title, now, conversation_id),
                )
            else:
                con.execute(
                    "UPDATE conversations SET updated=? WHERE id=?",
                    (now, conversation_id),
                )

        self._remember_if_useful(message)
        recent = self.messages(conversation_id, limit=24)
        memory = self.memories(limit=18)
        answer, model = self._call_gemini(recent, memory, job_context=job_context)

        with self._connect() as con:
            now = time.time()
            con.execute(
                "INSERT INTO messages(conversation_id,role,content,created) VALUES(?,?,?,?)",
                (conversation_id, "assistant", answer, now),
            )
            con.execute(
                "UPDATE conversations SET updated=? WHERE id=?",
                (now, conversation_id),
            )

        return {
            "conversation_id": conversation_id,
            "message": answer,
            "model": model,
            "memories": self.memories(limit=18),
        }

    def _remember_if_useful(self, message: str) -> None:
        lowered = message.lower()
        if not any(hint in lowered for hint in _MEMORY_HINTS):
            return
        compact = re.sub(r"\s+", " ", message).strip()[:700]
        if len(compact) < 5:
            return
        now = time.time()
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO memories(content,created,updated)
                VALUES(?,?,?)
                ON CONFLICT(content) DO UPDATE SET updated=excluded.updated
                """,
                (compact, now, now),
            )

    def _call_gemini(
        self,
        recent: list[dict[str, Any]],
        memory: list[str],
        *,
        job_context: str,
    ) -> tuple[str, str]:
        memory_text = "\n".join(f"- {item}" for item in memory) or "- пока нет"
        context_text = job_context.strip()[:10000] or "Сейчас конкретный ролик не выбран."
        instruction = f"""
Ты постоянный AI-монтажёр внутри локальной программы Video AI Studio.
Общайся по-человечески, коротко и по делу. Ты помогаешь монтировать Shorts:
понимаешь историю, предлагаешь сцены, B-roll, мемы, зумы, SFX, субтитры и правки.
Учитывай прошлые решения и не повторяй уже отвергнутые идеи.

ВАЖНО:
- Не утверждай, что реально изменил видео/код, если у тебя нет подтверждения выполнения действия.
- Если пользователь просит изменить монтаж, чётко опиши, что именно надо поменять.
- Контекст текущего ролика ниже может содержать логи/транскрипт; опирайся на него.
- Постоянные предпочтения пользователя тоже ниже.
- Отвечай на языке пользователя.

ПОСТОЯННАЯ ПАМЯТЬ:
{memory_text}

ТЕКУЩИЙ РОЛИК:
{context_text}
""".strip()

        contents: list[dict[str, Any]] = [
            {"role": "user", "parts": [{"text": instruction}]},
            {
                "role": "model",
                "parts": [{
                    "text": "Понял. Я монтажёр этого проекта и буду учитывать контекст ролика и сохранённые предпочтения."
                }],
            },
        ]
        for item in recent:
            role = "model" if item.get("role") == "assistant" else "user"
            text = str(item.get("content") or "").strip()
            if text:
                contents.append({"role": role, "parts": [{"text": text[:6000]}]})

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": 0.45,
                "maxOutputTokens": 1400,
            },
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        errors: list[str] = []

        for model in self.models:
            url = f"{_API_ROOT}/models/{urllib.parse.quote(model, safe='')}:generateContent"
            for attempt in range(2):
                req = urllib.request.Request(
                    url,
                    data=body,
                    method="POST",
                    headers={
                        "Content-Type": "application/json",
                        "x-goog-api-key": self.api_key,
                        "User-Agent": "video-ai/2.0.0",
                    },
                )
                try:
                    with urllib.request.urlopen(req, timeout=75) as response:
                        data = json.load(response)
                    candidates = data.get("candidates") or []
                    if not candidates:
                        raise RuntimeError("Gemini не вернул ответ")
                    parts = candidates[0].get("content", {}).get("parts", [])
                    text = "\n".join(
                        str(part.get("text", ""))
                        for part in parts
                        if isinstance(part, dict) and part.get("text")
                    ).strip()
                    if not text:
                        raise RuntimeError("Gemini вернул пустой ответ")
                    return text, model
                except urllib.error.HTTPError as exc:
                    detail = ""
                    try:
                        detail = exc.read().decode("utf-8", errors="ignore")[:1800]
                    except Exception:
                        pass
                    if exc.code == 429 and attempt == 0:
                        match = re.search(
                            r"(?:retry in|retry after)\s+([0-9.]+)\s*(ms|s|sec|seconds?)",
                            detail,
                            flags=re.IGNORECASE,
                        )
                        value = float(match.group(1)) if match else 3.0
                        unit = match.group(2).lower() if match else "s"
                        delay = value / 1000.0 if unit == "ms" else value
                        delay = max(0.5, min(delay + 0.35, 30.0))
                        time.sleep(delay)
                        continue
                    errors.append(f"{model}: HTTP {exc.code} {detail[:500]}")
                    break
                except Exception as exc:
                    errors.append(f"{model}: {type(exc).__name__}: {exc}")
                    break

        raise RuntimeError("Gemini chat failed: " + " | ".join(errors))
