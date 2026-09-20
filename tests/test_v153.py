from video_ai.stock_video import _smallest_pexels_preview, _smallest_pixabay_variant


def test_smallest_pexels_preview_prefers_lightweight_variant() -> None:
    files = [
        {"link": "hd", "width": 1080, "height": 1920},
        {"link": "small", "width": 360, "height": 640},
        {"link": "medium", "width": 720, "height": 1280},
    ]
    assert _smallest_pexels_preview(files)["link"] == "small"


def test_smallest_pixabay_preview_prefers_lightweight_variant() -> None:
    videos = {
        "large": {"url": "large", "width": 1920, "height": 1080},
        "medium": {"url": "medium", "width": 1280, "height": 720},
        "small": {"url": "small", "width": 640, "height": 360},
        "tiny": {"url": "tiny", "width": 320, "height": 180},
    }
    assert _smallest_pixabay_variant(videos)["url"] == "tiny"
