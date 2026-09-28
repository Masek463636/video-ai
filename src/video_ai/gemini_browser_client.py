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
    """Use the user's normal Gemini web chat for text-only reasoning.

    Multimodal requests still fall back to the regular Gemini API because the
    browser bridge currently does not attach image files yet.
    """

    provider_name = "gemini_browser"

    def __init__(self) -> None:
        super().__init__()
        self.last_model = "gemini-web"

    @property
    def browser_ready(self) -> bool:
        state = _read_state()
        url = str(state.get("chat_url") or "").strip()
        return bool(url and "gemini.google.com/app/" in url)

    def _browser_json(self, parts: list[dict[str, Any]]) -> Any:
        if sync_playwright is None:
            raise RuntimeError("Playwright is not installed")

        if any(isinstance(p, dict) and p.get("inline_data") for p in parts):
            raise RuntimeError("browser bridge image upload is not enabled yet")

        prompt_chunks: list[str] = []
        for part in parts:
            if isinstance(part, dict) and part.get("text"):
                prompt_chunks.append(str(part["text"]))
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
            answer = send_message(page, prompt)

        parsed = _parse_json_text(answer)
        self.last_model = "gemini-web"
        self.last_error = None
        return parsed

    def _generate_json(self, parts: list[dict[str, Any]], *, temperature: float) -> Any:
        # Text-only planning/search/reasoning goes through the persistent browser
        # chat. Visual judging remains on the API until browser file attachments
        # are implemented.
        if not any(isinstance(p, dict) and p.get("inline_data") for p in parts):
            try:
                return self._browser_json(parts)
            except Exception as exc:
                self.last_error = str(exc)
                if os.getenv("VIDEO_AI_BROWSER_STRICT", "0") == "1":
                    raise
                print(f"[gemini-browser] text request fallback to API: {exc}", flush=True)

        return super()._generate_json(parts, temperature=temperature)


def get_gemini_browser_client() -> GeminiBrowserClient | None:
    client = GeminiBrowserClient()
    if client.browser_ready:
        return client
    return None
