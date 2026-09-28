import json
import urllib.request

from video_ai.gemini_ai import get_gemini_client
from video_ai.qwen_ai import QwenClient, _parts_to_openai_content


def test_qwen_converts_inline_image_to_openai_data_url():
    parts = [
        {"text": "inspect"},
        {"inline_data": {"mime_type": "image/jpeg", "data": "YWJj"}},
    ]
    content = _parts_to_openai_content(parts)
    assert content[0] == {"type": "text", "text": "inspect"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"] == "data:image/jpeg;base64,YWJj"


def test_provider_switch_returns_qwen(monkeypatch):
    monkeypatch.setenv("VIDEO_AI_AI_PROVIDER", "qwen")
    monkeypatch.setenv("QWEN_API_KEY", "test-qwen")
    client = get_gemini_client()
    assert isinstance(client, QwenClient)


def test_qwen_json_request_uses_fast_non_thinking_mode(monkeypatch):
    monkeypatch.setenv("QWEN_API_KEY", "test-qwen")
    monkeypatch.setenv("QWEN_MODEL", "qwen3.5-flash")
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b""

    def fake_urlopen(request, timeout=0):
        captured["url"] = request.full_url
        captured["authorization"] = request.headers.get("Authorization")
        captured["body"] = json.loads(request.data.decode("utf-8"))

        class JsonResponse(Response):
            pass

        response = JsonResponse()
        response_payload = {
            "choices": [
                {"message": {"content": '{"ok": true}'}}
            ]
        }

        # json.load only needs a .read() method returning JSON bytes.
        response.read = lambda: json.dumps(response_payload).encode("utf-8")
        return response

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    client = QwenClient()
    result = client._generate_json([{"text": "return json"}], temperature=0.01)

    assert result == {"ok": True}
    assert captured["body"]["model"] == "qwen3.5-flash"
    assert captured["body"]["enable_thinking"] is False
    assert captured["body"]["stream"] is False
    assert captured["authorization"] == "Bearer test-qwen"
