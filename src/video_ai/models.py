from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


AssetKind = Literal["image", "video", "blank"]
MotionKind = Literal["none", "zoom_in", "zoom_out", "pan_left", "pan_right"]
VisualMode = Literal["auto", "image", "video", "meme"]
SourceMode = Literal["auto", "historical_archive", "stock_video", "meme_library", "generic_image"]
MotionPreset = Literal[
    "none",
    "micro_push",
    "slow_push",
    "dramatic_push",
    "pull_back",
    "reveal_left",
    "reveal_right",
]


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
    # Legacy motion stays for loading older plans. v1.0 renders motion_preset first.
    motion: MotionKind = "none"
    caption: str | None = None
    visual_description: str | None = None
    search_queries: list[str] = field(default_factory=list)
    visual_mode: VisualMode = "auto"
    source_mode: SourceMode = "auto"
    motion_preset: MotionPreset = "slow_push"
    # When Gemini selects a known local meme it can name the exact file.
    meme_filename: str | None = None
    focus_x: float | None = None
    focus_y: float | None = None
    focus_source: str | None = None
    asset_score: float | None = None
    semantic_score: float | None = None

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
    director_source: str = "rules"
    director_model: str | None = None
