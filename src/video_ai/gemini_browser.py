from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

try:
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
    from playwright.sync_api import sync_playwright
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Playwright is not installed. Run: python -m pip install playwright"
    ) from exc


GEMINI_HOME = "https://gemini.google.com/app"
STATE_DIR = Path.home() / ".video-ai"
PROFILE_DIR = STATE_DIR / "gemini-browser-profile"
STATE_FILE = STATE_DIR / "gemini-browser-state.json"
DEFAULT_CDP_ENDPOINT = "http://127.0.0.1:9222"

PROMPT_SELECTORS = [
    'div[contenteditable="true"][role="textbox"]',
    'rich-textarea [contenteditable="true"]',
    'textarea[aria-label*="prompt" i]',
    '[contenteditable="true"][aria-label*="prompt" i]',
]

RESPONSE_SELECTORS = [
    'model-response',
    '.model-response-text',
    '.response-content',
    'message-content',
]


def _read_state() -> dict:
    if not STATE_FILE.is_file():
        return {}
    try:
        value = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _write_state(value: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    temp = STATE_FILE.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(STATE_FILE)


def _first_visible(page, selectors: list[str], timeout_ms: int = 15000):
    deadline = time.monotonic() + timeout_ms / 1000
    last_error = None
    while time.monotonic() < deadline:
        for selector in selectors:
            try:
                locator = page.locator(selector)
                count = locator.count()
                for i in range(count - 1, -1, -1):
                    item = locator.nth(i)
                    if item.is_visible():
                        return item
            except Exception as exc:
                last_error = exc
        page.wait_for_timeout(250)
    if last_error:
        raise RuntimeError(f"Gemini UI element not found: {last_error}")
    raise RuntimeError("Gemini UI element not found")


def _last_response_text(page) -> str:
    best = ""
    for selector in RESPONSE_SELECTORS:
        try:
            locator = page.locator(selector)
            count = locator.count()
            for i in range(count):
                item = locator.nth(i)
                if not item.is_visible():
                    continue
                text = (item.inner_text(timeout=3000) or "").strip()
                if len(text) > len(best):
                    best = text
        except Exception:
            continue
    return best


def wait_for_answer(page, before: str, timeout_s: float = 120.0) -> str:
    deadline = time.monotonic() + timeout_s
    stable_text = ""
    stable_since = time.monotonic()
    while time.monotonic() < deadline:
        current = _last_response_text(page)
        if current and current != before:
            if current != stable_text:
                stable_text = current
                stable_since = time.monotonic()
            elif time.monotonic() - stable_since >= 2.2:
                return current
        page.wait_for_timeout(450)
    if stable_text:
        return stable_text
    raise RuntimeError("Timed out waiting for Gemini response")


def send_message(page, message: str) -> str:
    box = _first_visible(page, PROMPT_SELECTORS, timeout_ms=20000)
    before = _last_response_text(page)
    box.click()
    try:
        box.fill(message)
    except Exception:
        page.keyboard.insert_text(message)
    page.keyboard.press("Enter")
    return wait_for_answer(page, before)


def _connect_existing_chrome(p, endpoint: str):
    try:
        browser = p.chromium.connect_over_cdp(endpoint)
    except Exception as exc:
        raise RuntimeError(
            "Не удалось подключиться к обычному Chrome. "
            "Сначала запусти Chrome с --remote-debugging-port=9222 "
            "и отдельным --user-data-dir. "
            f"Endpoint: {endpoint}. Ошибка: {exc}"
        ) from exc
    if not browser.contexts:
        raise RuntimeError("Chrome подключён, но browser context не найден")
    context = browser.contexts[0]
    pages = context.pages
    page = pages[-1] if pages else context.new_page()
    return browser, context, page


def setup_chat(endpoint: str = DEFAULT_CDP_ENDPOINT) -> int:
    with sync_playwright() as p:
        try:
            browser, context, page = _connect_existing_chrome(p, endpoint)
        except RuntimeError as exc:
            print(exc)
            return 3

        if "gemini.google.com" not in page.url:
            page.goto(GEMINI_HOME, wait_until="domcontentloaded")

        print("\nПодключился к уже открытому обычному Chrome.")
        print("Открой нужный постоянный чат Gemini в этом окне.")
        print("Когда нужный чат открыт — вернись сюда и нажми Enter.\n")
        input()
        url = page.url
        if "gemini.google.com" not in url:
            print("Ошибка: сейчас открыт не Gemini.")
            browser.close()
            return 2

        if "gemini.google.com/app/" not in url:
            print("У этого окна пока нет ID конкретного чата. Отправляю служебное сообщение...")
            try:
                answer = send_message(
                    page,
                    "Это постоянный рабочий чат VIDEO AI EDITOR. Ответь только: VIDEO_AI_CHAT_READY",
                )
                print("Gemini:", answer.splitlines()[-1] if answer else "")
                deadline = time.monotonic() + 15
                while time.monotonic() < deadline and "gemini.google.com/app/" not in page.url:
                    page.wait_for_timeout(250)
                url = page.url
            except Exception as exc:
                print(f"Не удалось закрепить чат: {exc}")
                browser.close()
                return 4

        if "gemini.google.com/app/" not in url:
            print(f"Не удалось получить URL конкретного чата: {url}")
            browser.close()
            return 5

        state = _read_state()
        state["chat_url"] = url
        state["cdp_endpoint"] = endpoint
        _write_state(state)
        print(f"Чат сохранён: {url}")
        browser.close()
    return 0


def test_chat(message: str, endpoint: str | None = None) -> int:
    state = _read_state()
    chat_url = str(state.get("chat_url") or "").strip()
    if not chat_url:
        print("Сначала запусти: python -m video_ai.gemini_browser setup")
        return 2

    endpoint = endpoint or str(state.get("cdp_endpoint") or DEFAULT_CDP_ENDPOINT)
    with sync_playwright() as p:
        try:
            browser, context, page = _connect_existing_chrome(p, endpoint)
        except RuntimeError as exc:
            print(exc)
            return 3

        page.goto(chat_url, wait_until="domcontentloaded")
        try:
            answer = send_message(page, message)
        except PlaywrightTimeoutError as exc:
            print(f"Playwright timeout: {exc}")
            browser.close()
            return 3
        except Exception as exc:
            print(f"Gemini browser bridge error: {exc}")
            print("Интерфейс Gemini мог измениться. Chrome оставлен открытым для проверки.")
            return 4

        print("\n===== GEMINI ANSWER =====")
        print(answer)
        print("=========================\n")
        browser.close()
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Gemini website bridge for VIDEO AI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_setup = sub.add_parser("setup", help="Attach to an already authenticated normal Chrome and save the Gemini chat URL")
    p_setup.add_argument("--cdp-endpoint", default=DEFAULT_CDP_ENDPOINT)

    p_test = sub.add_parser("test", help="Send one message to the saved Gemini chat")
    p_test.add_argument("--cdp-endpoint", default=None)
    p_test.add_argument(
        "message",
        nargs="?",
        default="Ответь только одной строкой: VIDEO_AI_OK",
    )

    args = parser.parse_args()
    if args.command == "setup":
        raise SystemExit(setup_chat(args.cdp_endpoint))
    raise SystemExit(test_chat(args.message, endpoint=args.cdp_endpoint))


if __name__ == "__main__":
    main()
