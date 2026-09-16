from __future__ import annotations

from pathlib import Path


def detect_focus(path: str | Path) -> tuple[float, float, str]:
    """Return normalized focus coordinates for an image.

    V0.3 keeps this optional. If OpenCV is installed we first look for faces,
    because portraits are the easiest thing to ruin when converting landscape
    media into 9:16. If no face is found we fall back to a simple saliency-like
    edge-density center. Without OpenCV we safely return frame center.
    """
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except ImportError:
        return 0.5, 0.5, "center"

    image = cv2.imread(str(path))
    if image is None:
        return 0.5, 0.5, "center"
    height, width = image.shape[:2]
    if width <= 0 or height <= 0:
        return 0.5, 0.5, "center"

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    try:
        cascade_path = str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
        cascade = cv2.CascadeClassifier(cascade_path)
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(max(30, width // 18), max(30, height // 18)),
        )
    except Exception:
        faces = ()

    if len(faces):
        # Prefer the largest face, which usually represents the main subject.
        x, y, w, h = max(faces, key=lambda box: int(box[2]) * int(box[3]))
        return _clamp((x + w / 2) / width), _clamp((y + h / 2) / height), "face"

    # Cheap local fallback: edge density approximates where the interesting
    # visual detail is, without bringing a heavy ML dependency into the MVP.
    edges = cv2.Canny(gray, 80, 180)
    small = cv2.resize(edges, (12, 12), interpolation=cv2.INTER_AREA).astype("float32")
    total = float(small.sum())
    if total <= 1e-6:
        return 0.5, 0.5, "center"

    ys, xs = np.mgrid[0:12, 0:12]
    focus_x = float((small * (xs + 0.5)).sum() / total / 12.0)
    focus_y = float((small * (ys + 0.5)).sum() / total / 12.0)
    return _clamp(focus_x), _clamp(focus_y), "saliency"


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
