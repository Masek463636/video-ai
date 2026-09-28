from __future__ import annotations

import json
import os
import re
from typing import Any

from .gemini_ai import GeminiClient, _parse_json_text
from .gemini_browser import test_chat, _read_state, _connect_existing_chrome, send_message, DEFAULT_CDP_ENDPOINT

try:
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover
    sync_playwright = None


class GeminiBrowserClient(GeminiClient):
    """Use the user's normal Gemini web chat for text and image reasoning."""

    provider_name = "gemini_browser"

    def __init__(self) -> None:
        super().__init__()
        self.last_model = "gemini-web"

    @property
    def browser_ready(self) -> bool:
        state = _read_state()
        url = str(state.get("chat_url") or "").strip()
        return bool(url and "gemini.google.com/app/" in url)

    @property
    def available(self) -> bool:
        return self.browser_ready or super().available

    def _browser_json(self, parts: list[dict[str, Any]]) -> Any:
        if sync_playwright is None:
            raise RuntimeError("Playwright is not installed")

        prompt_chunks: list[str] = []
        image_paths: list[str] = []
        temp_files: list[str] = []

        import base64
        import tempfile
        from pathlib import Path

        try:
            for index, part in enumerate(parts):
                if not isinstance(part, dict):
                    continue
                if part.get("text"):
                    prompt_chunks.append(str(part["text"]))
                inline = part.get("inline_data")
                if isinstance(inline, dict) and inline.get("data"):
                    mime = str(inline.get("mime_type") or "image/jpeg").lower()
                    suffix = ".png" if "png" in mime else ".webp" if "webp" in mime else ".jpg"
                    target = Path(tempfile.gettempdir()) / f"video-ai-gemini-{os.getpid()}-{index}{suffix}"
                    target.write_bytes(base64.b64decode(str(inline["data"])))
                    image_paths.append(str(target))
                    temp_files.append(str(target))

            prompt = "\n\n".join(prompt_chunks).strip()
            if not prompt:
                raise RuntimeError("browser bridge request contains no text")

            state = _read_state()
            chat_url = str(state.get("chat_url") or "").strip()
            endpoint = str(state.get("cdp_endpoint") or DEFAULT_CDP_ENDPOINT)
            if not chat_url or "gemini.google.com/app/" not in chat_url:
                raise RuntimeError(
                    "Gemini browser chat is not pinned. Run: "
                    "python -m video_ai.gemini_browser setup"
                )

            with sync_playwright() as p:
                browser, context, page = _connect_existing_chrome(p, endpoint)
                if page.url != chat_url:
                    page.goto(chat_url, wait_until="domcontentloaded")
                answer = send_message(page, prompt, file_paths=image_paths)

            parsed = _parse_json_text(answer)
            self.last_model = "gemini-web"
            self.last_error = None
            return parsed
        finally:
            for value in temp_files:
                try:
                    Path(value).unlink(missing_ok=True)
                except OSError:
                    pass

    def _generate_json(self, parts: list[dict[str, Any]], *, temperature: float) -> Any:
        try:
            return self._browser_json(parts)
        except Exception as exc:
            self.last_error = str(exc)
            allow_api_fallback = os.getenv("VIDEO_AI_BROWSER_ALLOW_API_FALLBACK", "0") == "1"
            if not allow_api_fallback:
                raise RuntimeError(f"Gemini browser request failed: {exc}") from exc
            print(f"[gemini-browser] browser request fallback to API: {exc}", flush=True)
            return super()._generate_json(parts, temperature=temperature)


def get_gemini_browser_client() -> GeminiBrowserClient | None:
    client = GeminiBrowserClient()
    if client.browser_ready:
        return client
    return None
