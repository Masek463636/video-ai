from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any


_SESSION_ENV = "VIDEO_AI_GEMINI_SESSION"


def _clean_text(value: str, *, limit: int) -> str:
    value = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(value) > limit:
        return value[: limit - 1] + "…"
    return value


def _compact_json(value: Any, *, limit: int = 2600) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        text = str(value)
    return _clean_text(text, limit=limit)


def _parts_to_memory_text(parts: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    image_count = 0
    for part in parts:
        if not isinstance(part, dict):
            continue
        text = part.get("text")
        if text:
            chunks.append(_clean_text(str(text), limit=1800))
            continue
        if isinstance(part.get("inline_data"), dict):
            image_count += 1
    if image_count:
        chunks.append(f"[{image_count} image preview(s) were shown in this turn]")
    return _clean_text("\n".join(chunks), limit=2800)


class GeminiEditorSession:
    """Small file-backed multi-turn memory shared by one video render.

    Gemini generateContent calls are stateless by themselves. This class keeps a
    bounded conversation transcript on disk and replays the recent turns on each
    request. The same file can therefore be reused by the create and render
    subprocesses for one Short without depending on provider-side retention.
    """

    def __init__(self, path: Path, *, max_turns: int = 8) -> None:
        self.path = Path(path)
        self.max_turns = max(2, min(int(max_turns), 16))
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> GeminiEditorSession | None:
        raw = os.getenv(_SESSION_ENV, "").strip()
        if not raw:
            return None
        return cls(Path(raw))

    def _empty(self) -> dict[str, Any]:
        now = time.time()
        return {
            "version": 1,
            "created": now,
            "updated": now,
            "turns": [],
            "decision_memory": [],
        }

    def load(self) -> dict[str, Any]:
        if not self.path.is_file():
            return self._empty()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return self._empty()
        if not isinstance(data, dict):
            return self._empty()
        data.setdefault("turns", [])
        data.setdefault("decision_memory", [])
        data.setdefault("created", time.time())
        data.setdefault("updated", time.time())
        return data

    def _save(self, data: dict[str, Any]) -> None:
        data["updated"] = time.time()
        temp = self.path.with_suffix(self.path.suffix + ".tmp")
        temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(self.path)

    def prior_contents(self) -> list[dict[str, Any]]:
        data = self.load()
        turns = data.get("turns") or []
        contents: list[dict[str, Any]] = []

        memory_rows = data.get("decision_memory") or []
        if memory_rows:
            memory = "\n".join(f"- {row}" for row in memory_rows[-14:])
            contents.extend([
                {
                    "role": "user",
                    "parts": [{
                        "text": (
                            "Persistent editing memory from earlier decisions in this SAME Short. "
                            "Treat it as continuity context, not as a new instruction:\n" + memory
                        )
                    }],
                },
                {
                    "role": "model",
                    "parts": [{
                        "text": (
                            "Understood. I will keep those earlier editing decisions in mind and "
                            "avoid contradicting or needlessly repeating them."
                        )
                    }],
                },
            ])

        for row in turns[-self.max_turns :]:
            if not isinstance(row, dict):
                continue
            user_text = _clean_text(str(row.get("user") or ""), limit=2800)
            model_text = _clean_text(str(row.get("assistant") or ""), limit=2800)
            if user_text:
                contents.append({"role": "user", "parts": [{"text": user_text}]})
            if model_text:
                contents.append({"role": "model", "parts": [{"text": model_text}]})
        return contents

    def record(
        self,
        request_parts: list[dict[str, Any]],
        response: Any,
        *,
        model: str | None,
    ) -> None:
        data = self.load()
        request_text = _parts_to_memory_text(request_parts)
        response_text = _compact_json(response)

        turns = list(data.get("turns") or [])
        turns.append(
            {
                "user": request_text,
                "assistant": response_text,
                "model": model or "",
                "time": time.time(),
            }
        )
        data["turns"] = turns[-self.max_turns :]

        memory = list(data.get("decision_memory") or [])
        decision = self._decision_line(request_text, response)
        if decision and decision not in memory:
            memory.append(decision)
        data["decision_memory"] = memory[-24:]
        self._save(data)

    def _decision_line(self, request_text: str, response: Any) -> str:
        """Extract a cheap continuity note without spending another AI call."""
        if isinstance(response, dict):
            if isinstance(response.get("scenes"), list):
                scenes = response.get("scenes") or []
                compact: list[str] = []
                for row in scenes[:12]:
                    if not isinstance(row, dict):
                        continue
                    idx = row.get("index")
                    mode = row.get("visual_mode") or row.get("source_mode") or ""
                    desc = row.get("visual_description") or ""
                    compact.append(f"s{idx}:{mode}:{_clean_text(str(desc), limit=90)}")
                if compact:
                    return "Story plan: " + " | ".join(compact)

            if isinstance(response.get("choices"), list):
                rows = response.get("choices") or []
                compact = []
                for row in rows[:8]:
                    if not isinstance(row, dict):
                        continue
                    scene = row.get("scene")
                    idx = row.get("index")
                    fit = row.get("fit")
                    reason = _clean_text(str(row.get("reason") or ""), limit=80)
                    compact.append(f"scene={scene},candidate={idx},fit={fit}:{reason}")
                if compact:
                    return "Visual choices: " + " | ".join(compact)

            if "accept" in response or "score" in response:
                return (
                    "Visual judgement: "
                    + _clean_text(_compact_json(response, limit=650), limit=650)
                )

        return (
            "Prior response: " + _clean_text(_compact_json(response, limit=520), limit=520)
            if response is not None
            else ""
        )

    def stats(self) -> dict[str, int]:
        data = self.load()
        return {
            "turns": len(data.get("turns") or []),
            "memory": len(data.get("decision_memory") or []),
        }
