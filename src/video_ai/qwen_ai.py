from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any

from .gemini_ai import GeminiClient, _parse_json_text


_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


def _parts_to_openai_content(parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for part in parts:
        text = part.get("text")
        if text is not None:
            content.append({"type": "text", "text": str(text)})
            continue

        inline = part.get("inline_data")
        if not isinstance(inline, dict):
            continue
        data = str(inline.get("data") or "").strip()
        if not data:
            continue
        mime = str(inline.get("mime_type") or "image/jpeg").strip() or "image/jpeg"
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{data}"},
            }
        )
    return content


def _extract_qwen_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError("Qwen returned no choices")
    message = choices[0].get("message") or {}
    content = message.get("content")

    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict):
                value = item.get("text") or item.get("content")
                if value:
                    chunks.append(str(value))
        text = "\n".join(chunks)
    else:
        text = str(content or "")

    if not text.strip():
        raise RuntimeError("Qwen returned no text")
    return text.strip()


def _retry_after_seconds(exc: urllib.error.HTTPError, detail: str) -> float | None:
    try:
        header = exc.headers.get("Retry-After") if exc.headers else None
        if header:
            return max(0.0, float(header))
    except (TypeError, ValueError):
        pass

    patterns = (
        r"retry\s+in\s+([0-9.]+)\s*(ms|s|sec|seconds?)",
        r"retry\s+after\s+([0-9.]+)\s*(ms|s|sec|seconds?)",
    )
    for pattern in patterns:
        match = re.search(pattern, detail, flags=re.IGNORECASE)
        if not match:
            continue
        value = float(match.group(1))
        unit = match.group(2).lower()
        return value / 1000.0 if unit == "ms" else value
    return None


class QwenClient(GeminiClient):
    """Drop-in replacement for GeminiClient used by Premium v2.

    The high-level editing helpers live on GeminiClient; this subclass keeps the
    same interface but sends every JSON/vision request to Qwen through
    OpenRouter's OpenAI-compatible endpoint. The exact Premium v2 editing
    pipeline stays unchanged; only the AI provider/model is swapped.
    """

    provider_name = "qwen"

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.api_key = (
            api_key
            or os.getenv("OPENROUTER_API_KEY", "")
            or os.getenv("QWEN_API_KEY", "")
        ).strip()

        preferred = (model or os.getenv("QWEN_MODEL", "")).strip()
        fallback = os.getenv(
            "QWEN_FALLBACK_MODEL",
            "qwen/qwen3.6-plus:free",
        ).strip()
        candidates = [preferred or "qwen/qwen3.8-27b:free", fallback]
        self.models = []
        for name in candidates:
            if name and name not in self.models:
                self.models.append(name)

        self.base_url = (
            os.getenv("QWEN_BASE_URL", "").strip() or _DEFAULT_BASE_URL
        ).rstrip("/")
        self.last_model: str | None = None
        self.last_error: str | None = None
        self.request_count = 0

    def _generate_json(self, parts: list[dict[str, Any]], *, temperature: float) -> Any:
        if not self.api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not configured"
            )

        content = _parts_to_openai_content(parts)
        if not content:
            raise RuntimeError("Qwen request contains no usable content")

        payload_base = {
            "messages": [{"role": "user", "content": content}],
            "temperature": temperature,
            "max_tokens": 8192,
            "stream": False,
            "response_format": {"type": "json_object"},
        }

        errors: list[str] = []
        try:
            max_wait = float(os.getenv("QWEN_MAX_RETRY_WAIT", "45"))
        except ValueError:
            max_wait = 45.0
        max_wait = max(1.0, min(max_wait, 180.0))

        for model in self.models:
            url = self.base_url + "/chat/completions"
            for attempt in range(2):
                payload = dict(payload_base)
                payload["model"] = model
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                request = urllib.request.Request(
                    url,
                    data=body,
                    method="POST",
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}",
                        "User-Agent": "video-ai/2.0.0",
                        "HTTP-Referer": "http://127.0.0.1",
                        "X-OpenRouter-Title": "Video AI Studio",
                    },
                )
                self.request_count += 1
                print(
                    f"[qwen] request {self.request_count} -> {model}",
                    flush=True,
                )

                try:
                    with urllib.request.urlopen(request, timeout=90) as response:
                        response_data = json.load(response)
                    parsed = _parse_json_text(_extract_qwen_text(response_data))
                    self.last_model = model
                    self.last_error = None
                    return parsed
                except urllib.error.HTTPError as exc:
                    detail = ""
                    try:
                        detail = exc.read().decode("utf-8", errors="ignore")[:3000]
                    except Exception:
                        pass

                    if exc.code == 429 and attempt == 0:
                        requested = _retry_after_seconds(exc, detail)
                        delay = min(
                            max_wait,
                            max(1.0, (requested if requested is not None else 4.0) + 0.5),
                        )
                        print(
                            f"[qwen] rate limited on {model}; waiting {delay:.2f}s "
                            "and retrying once",
                            flush=True,
                        )
                        time.sleep(delay)
                        continue

                    errors.append(f"{model}: HTTP {exc.code} {detail[:700]}")
                    break
                except Exception as exc:
                    errors.append(f"{model}: {type(exc).__name__}: {exc}")
                    break

        self.last_error = " | ".join(errors)
        raise RuntimeError("Qwen request failed: " + self.last_error)


def get_qwen_client() -> QwenClient | None:
    client = QwenClient()
    return client if client.available else None
