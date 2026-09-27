from __future__ import annotations

import re
from pathlib import Path

from .models import Scene, ShotPlan


_ACTION_GROUPS = {
    "phone": ("phone", "smartphone", "call", "message", "screen", "телефон"),
    "money": ("money", "cash", "price", "receipt", "wallet", "деньг", "цена"),
    "food": ("food", "pizza", "milk", "restaurant", "eat", "drink", "еда", "молок"),
    "car": ("car", "vehicle", "drive", "road", "truck", "машин", "дорог"),
    "animal": ("dog", "cat", "rabbit", "animal", "pet", "собак", "кошк", "крол"),
    "person": ("person", "man", "woman", "people", "face", "human", "человек"),
    "document": ("document", "paper", "map", "letter", "book", "receipt", "документ", "карт"),
    "war": ("war", "soldier", "battle", "army", "rebel", "войн", "солдат", "битв"),
    "building": ("building", "house", "city", "street", "store", "shop", "дом", "город", "магаз"),
}

_INDOOR = ("indoor", "room", "office", "store", "shop", "home", "kitchen", "room", "комнат", "магаз", "офис")
_OUTDOOR = ("outdoor", "street", "road", "park", "field", "city", "forest", "улиц", "дорог", "поле", "лес")


def _semantic_hint(scene: Scene) -> tuple[str, str]:
    text = " ".join(
        [
            scene.query or "",
            scene.visual_description or "",
            scene.caption or "",
        ]
    ).casefold()

    action = "other"
    for name, needles in _ACTION_GROUPS.items():
        if any(needle in text for needle in needles):
            action = name
            break

    indoor = any(needle in text for needle in _INDOOR)
    outdoor = any(needle in text for needle in _OUTDOOR)
    environment = "indoor" if indoor and not outdoor else "outdoor" if outdoor and not indoor else "mixed"
    return action, environment


def _frame(scene: Scene):
    if not scene.asset:
        return None
    path = Path(scene.asset)
    if not path.is_file():
        return None
    try:
        import cv2
    except ImportError:
        return None

    if scene.asset_kind == "image":
        return cv2.imread(str(path))

    if scene.asset_kind == "video":
        cap = cv2.VideoCapture(str(path))
        try:
            cap.set(
                cv2.CAP_PROP_POS_MSEC,
                max(0.0, float(scene.source_start) + 0.25) * 1000.0,
            )
            ok, image = cap.read()
            return image if ok else None
        finally:
            cap.release()

    return None


def _shot_type(image) -> str:
    try:
        import cv2
    except ImportError:
        return "unknown"

    if image is None:
        return "unknown"
    height, width = image.shape[:2]
    if width <= 0 or height <= 0:
        return "unknown"

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    try:
        cascade = cv2.CascadeClassifier(
            str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
        )
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(max(24, width // 20), max(24, height // 20)),
        )
    except Exception:
        faces = ()

    if not len(faces):
        return "wide"

    _, _, fw, fh = max(
        faces,
        key=lambda box: int(box[2]) * int(box[3]),
    )
    ratio = float(fw * fh) / float(width * height)
    if ratio >= 0.18:
        return "close"
    if ratio >= 0.055:
        return "medium"
    return "wide"


def _palette(image) -> str:
    try:
        import cv2
        import numpy as np
    except ImportError:
        return "unknown"

    if image is None:
        return "unknown"

    small = cv2.resize(image, (8, 8), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV).astype("float32")
    hue = float(np.mean(hsv[:, :, 0]))
    sat = float(np.mean(hsv[:, :, 1]))
    value = float(np.mean(hsv[:, :, 2]))

    if sat < 35:
        tone = "neutral"
    else:
        bucket = int(hue // 30) % 6
        tone = ("red", "yellow", "green", "cyan", "blue", "magenta")[bucket]

    light = "dark" if value < 85 else "bright" if value > 175 else "mid"
    return f"{tone}-{light}"


def fingerprint_scene(scene: Scene) -> str | None:
    if scene.asset_kind == "blank" or not scene.asset:
        scene.shot_fingerprint = None
        return None

    image = _frame(scene)
    action, environment = _semantic_hint(scene)
    fingerprint = "|".join(
        [
            _shot_type(image),
            _palette(image),
            action,
            environment,
        ]
    )
    scene.shot_fingerprint = fingerprint
    return fingerprint


def assign_fingerprints(plan: ShotPlan) -> list[str | None]:
    return [fingerprint_scene(scene) for scene in plan.scenes]


def soft_diversity_repair_indexes(plan: ShotPlan) -> list[int]:
    fingerprints = assign_fingerprints(plan)
    repairs: set[int] = set()

    for index in range(len(plan.scenes)):
        scene = plan.scenes[index]
        if scene.semantic_lock or not fingerprints[index]:
            continue

        current = fingerprints[index].split("|")
        shot = current[0]
        action = current[2]

        if index >= 2 and action != "other":
            recent = [
                (fingerprints[j] or "").split("|")
                for j in range(index - 2, index + 1)
            ]
            if all(
                len(parts) >= 3
                and parts[0] == shot
                and parts[2] == action
                for parts in recent
            ):
                repairs.add(index)
                continue

        if index >= 3:
            window = [
                fingerprints[j]
                for j in range(index - 3, index + 1)
                if fingerprints[j]
            ]
            if fingerprints[index] and window.count(fingerprints[index]) >= 3:
                repairs.add(index)

    return sorted(repairs)


def diversify_queries(plan: ShotPlan, indexes: list[int]) -> None:
    for index in indexes:
        if not 0 <= index < len(plan.scenes):
            continue
        scene = plan.scenes[index]
        if scene.semantic_lock:
            continue

        seed = sum(ord(ch) for ch in (scene.caption or scene.query or "")) + index
        modifiers = (
            "wide environmental shot",
            "close detail shot",
            "side angle action",
            "hands object close up",
        )
        modifier = modifiers[seed % len(modifiers)]
        base = re.sub(r"\s+", " ", scene.query or "").strip()
        variant = f"{base} {modifier}".strip()
        existing = [q for q in scene.search_queries if q]
        scene.search_queries = [variant, *[q for q in existing if q.casefold() != variant.casefold()]][:6]
        scene.query = scene.search_queries[0] if scene.search_queries else scene.query
