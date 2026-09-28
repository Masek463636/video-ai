from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
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

FILE_INPUT_SELECTORS = [
    'input[type="file"]',
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


def browser_chat_ready() -> bool:
    state = _read_state()
    url = str(state.get("chat_url") or "").strip()
    return bool(url and "gemini.google.com/app/" in url)


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
    """Return the newest visible Gemini answer, not the longest old answer."""
    newest = ""
    newest_y = -1.0

    for selector in RESPONSE_SELECTORS:
        try:
            locator = page.locator(selector)
            count = locator.count()
            for i in range(count):
                item = locator.nth(i)
                try:
                    if not item.is_visible():
                        continue
                    text = (item.inner_text(timeout=3000) or "").strip()
                    if not text:
                        continue
                    box = item.bounding_box()
                    y = float(box["y"]) if box else float(i)
                    if y >= newest_y:
                        newest_y = y
                        newest = text
                except Exception:
                    continue
        except Exception:
            continue

    return newest
def wait_for_answer(page, before: str, timeout_s: float = 120.0) -> str:
    deadline = time.monotonic() + timeout_s
    stable_text = ""
    stable_since = time.monotonic()
    while time.monotonic() < deadline:
        current = _last_response_text(page)
        if current and current.strip() != before.strip():
            if current != stable_text:
                stable_text = current
                stable_since = time.monotonic()
            elif time.monotonic() - stable_since >= 2.5:
                return current
        page.wait_for_timeout(450)
    if stable_text and stable_text.strip() != before.strip():
        return stable_text
    raise RuntimeError("Timed out waiting for a NEW Gemini response")


def _try_set_file_input(page, paths: list[str]) -> bool:
    try:
        locator = page.locator('input[type="file"]')
        count = locator.count()
        for i in range(count - 1, -1, -1):
            try:
                locator.nth(i).set_input_files(paths)
                page.wait_for_timeout(1400)
                return True
            except Exception:
                continue
    except Exception:
        pass
    return False


def _click_and_choose_files(page, locator, paths: list[str], *, timeout: int = 5000) -> bool:
    try:
        with page.expect_file_chooser(timeout=timeout) as chooser_info:
            locator.click(force=True)
        chooser_info.value.set_files(paths)
        page.wait_for_timeout(1400)
        return True
    except Exception:
        return False


def _attach_files(page, file_paths: list[str] | None) -> None:
    paths = [str(Path(p).resolve()) for p in (file_paths or []) if Path(p).is_file()]
    if not paths:
        return

    # Fast path: sometimes Gemini keeps a hidden input in the composer.
    if _try_set_file_input(page, paths):
        return

    # Current Gemini UI (2026) commonly calls the plus button "Upload & tools".
    menu_selectors = [
        'button[aria-label="Upload & tools"]',
        'button[aria-label="Open upload file menu"]',
        'button[aria-label*="Upload" i]',
        'button[aria-label*="Attach" i]',
        'button[aria-label*="Add file" i]',
        'button[aria-label*="Add files" i]',
        'button[aria-label*="Загруз" i]',
        'button[aria-label*="Прикреп" i]',
        'button[aria-label*="Добав" i]',
        'button[aria-label*="Файл" i]',
    ]
    for selector in menu_selectors:
        try:
            button = page.locator(selector)
            if not button.count():
                continue
            button.last.click(force=True)
            page.wait_for_timeout(500)
            if _try_set_file_input(page, paths):
                return
            break
        except Exception:
            continue

    # Russian Gemini UI often shows an exact visible menu item "Файлы".
    try:
        text_node = page.get_by_text("Файлы", exact=True)
        if text_node.count():
            candidate = text_node.last
            # Walk up to a clickable menu/button container when possible.
            clickable = candidate.locator(
                "xpath=ancestor-or-self::*[self::button or @role='button' or @role='menuitem'][1]"
            )
            if clickable.count():
                candidate = clickable
            if _click_and_choose_files(page, candidate, paths):
                return
            try:
                candidate.click(force=True)
                page.wait_for_timeout(350)
            except Exception:
                pass
            if _try_set_file_input(page, paths):
                return
    except Exception:
        pass

    # The current UI may render the upload action in this dedicated wrapper.
    direct_upload_selectors = [
        'images-files-uploader[data-test-id="uploader-images-files-button-advanced"]',
        '[data-test-id="local-images-files-uploader-icon"]',
        '[data-test-id*="uploader-images-files"]',
        '[role="menuitem"]:has-text("Upload files")',
        '[role="menuitem"]:has-text("Загрузить файлы")',
        '[role="menuitem"]:has-text("Файлы")',
        'button:has-text("Upload files")',
        'button:has-text("Загрузить файлы")',
        'button:has-text("Файлы")',
        'div[role="button"]:has-text("Файлы")',
        'span:has-text("Файлы")',
    ]
    for selector in direct_upload_selectors:
        try:
            item = page.locator(selector)
            if not item.count():
                continue
            if _click_and_choose_files(page, item.last, paths):
                return
            # Some builds create the input only after this click instead of
            # emitting the native file chooser event.
            try:
                item.last.click(force=True)
                page.wait_for_timeout(350)
            except Exception:
                pass
            if _try_set_file_input(page, paths):
                return
        except Exception:
            continue

    # Last resort: inspect visible menu/buttons by accessible text.
    for pattern in (
        r"upload files",
        r"upload from computer",
        r"attach files",
        r"загрузить файлы",
        r"загрузить с компьютера",
        r"прикрепить файлы",
        r"^файлы$",
        r"файлы",
    ):
        for role in ("menuitem", "button"):
            try:
                item = page.get_by_role(role, name=re.compile(pattern, re.I))
                if not item.count():
                    continue
                if _click_and_choose_files(page, item.last, paths):
                    return
                if _try_set_file_input(page, paths):
                    return
            except Exception:
                continue

    raise RuntimeError(
        "Не удалось найти загрузку файлов в интерфейсе Gemini "
        "(искал Upload & tools / Upload files / uploader-images-files)"
    )
def send_message(
    page,
    message: str,
    *,
    file_paths: list[str] | None = None,
) -> str:
    box = _first_visible(page, PROMPT_SELECTORS, timeout_ms=20000)
    before = _last_response_text(page)
    _attach_files(page, file_paths)
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
                return 4

        if "gemini.google.com/app/" not in url:
            print(f"Не удалось получить URL конкретного чата: {url}")
            return 5

        state = _read_state()
        state["chat_url"] = url
        state["cdp_endpoint"] = endpoint
        _write_state(state)
        print(f"Чат сохранён: {url}")
    return 0


def test_chat(message: str, endpoint: str | None = None, file_paths: list[str] | None = None) -> int:
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
            answer = send_message(page, message, file_paths=file_paths)
        except PlaywrightTimeoutError as exc:
            print(f"Playwright timeout: {exc}")
            return 3
        except Exception as exc:
            print(f"Gemini browser bridge error: {exc}")
            print("Интерфейс Gemini мог измениться. Chrome оставлен открытым для проверки.")
            return 4

        print("\n===== GEMINI ANSWER =====")
        print(answer)
        print("=========================\n")
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

    p_upload = sub.add_parser("test-upload", help="Attach a tiny test image in the saved Gemini chat")
    p_upload.add_argument("--cdp-endpoint", default=None)

    args = parser.parse_args()
    if args.command == "setup":
        raise SystemExit(setup_chat(args.cdp_endpoint))
    if args.command == "test-upload":
        # 1x1 PNG, only to verify that Gemini's file uploader is being driven.
        test_png = Path(tempfile.gettempdir()) / "video-ai-upload-test.png"
        test_png.write_bytes(
            bytes.fromhex(
                "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
                "0000000d49444154789c6360f8cfc000000301010018dd8db10000000049454e44ae426082"
            )
        )
        try:
            raise SystemExit(test_chat(
                "Если ты видишь прикреплённое изображение, ответь только: IMAGE_UPLOAD_OK",
                endpoint=args.cdp_endpoint,
                file_paths=[str(test_png)],
            ))
        finally:
            try:
                test_png.unlink(missing_ok=True)
            except OSError:
                pass
    raise SystemExit(test_chat(args.message, endpoint=args.cdp_endpoint))


if __name__ == "__main__":
    main()
