from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


AssetKind = Literal["image", "video", "blank"]
MotionKind = Literal["none", "zoom_in", "zoom_out", "pan_left", "pan_right"]


@dataclass(slots=True)
class Word:
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(slots=True)
class Transcript:
    words: list[Word]
    language: str | None = None

    @property
    def duration(self) -> float:
        return self.words[-1].end if self.words else 0.0

    @property
    def text(self) -> str:
        return " ".join(word.text for word in self.words).strip()


@dataclass(slots=True)
class Scene:
    start: float
    end: float
    query: str
    asset: str | None = None
    asset_kind: AssetKind = "blank"
    motion: MotionKind = "none"
    caption: str | None = None
    # Normalized 0..1 coordinates of the visual subject. V0.3 uses these when
    # cropping landscape source material into a vertical frame.
    focus_x: float | None = None
    focus_y: float | None = None
    focus_source: str | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(slots=True)
class ShotPlan:
    audio: Path
    scenes: list[Scene]
    width: int = 1080
    height: int = 1920
    fps: int = 30
