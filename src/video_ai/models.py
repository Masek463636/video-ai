from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


AssetKind = Literal["image", "video", "blank"]
MotionKind = Literal["none", "zoom_in", "zoom_out", "pan_left", "pan_right"]
VisualMode = Literal["auto", "image", "video", "meme"]
SourceMode = Literal["auto", "historical_archive", "stock_video", "meme_library", "generic_image"]
Tone = Literal[
    "neutral",
    "informational",
    "positive",
    "negative",
    "tragic",
    "tense",
    "shocking",
    "absurd",
    "funny",
    "victorious",
    "mysterious",
    "religious",
    "violent",
    "emotional",
]
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
    motion: MotionKind = "none"
    caption: str | None = None
    visual_description: str | None = None
    search_queries: list[str] = field(default_factory=list)
    visual_mode: VisualMode = "auto"
    source_mode: SourceMode = "auto"
    motion_preset: MotionPreset = "slow_push"
    meme_filename: str | None = None

    # v1.1 Semantic Lock. Hard factual requirements only; Gemini cannot invent
    # new hard entities that are absent from the current caption.
    semantic_lock: bool = False
    required_entities: list[str] = field(default_factory=list)
    required_context: list[str] = field(default_factory=list)
    semantic_fallback: str | None = None

    # v1.2 Semantic Tone Guard. This describes how a visual should FEEL, not
    # merely what nouns it contains. It prevents e.g. a sunny beach from being
    # accepted for a mass-casualty sentence.
    tone: Tone = "neutral"

    focus_x: float | None = None
    focus_y: float | None = None
    focus_source: str | None = None
    asset_score: float | None = None
    semantic_score: float | None = None
    # Original absolute speech timings; optional for older saved ShotPlans.
    caption_words: list[Word] = field(default_factory=list)

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
