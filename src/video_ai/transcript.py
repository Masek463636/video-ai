from __future__ import annotations

import json
import re
from pathlib import Path

from .models import ShotPlan, Transcript, Word


def _caption_text(words: list[Word]) -> str:
    """Rebuild caption text exactly like the planners do."""
    return re.sub(r"\s+([,.!?;:…])", r"\1", " ".join(word.text for word in words)).strip()


def _plan_already_has_transcript_timings(plan: ShotPlan, transcript: Transcript) -> bool:
    """Return True when the saved plan already carries this exact transcript timing.

    New materialized plans persist the original caption words. Reusing them avoids
    reconstructing scene membership from cut boundaries, where a word can legally
    extend a few milliseconds past the scene end. A different transcript still
    falls through to the strict A/B re-attachment path below.
    """
    embedded: list[Word] = []
    for scene in plan.scenes:
        if not scene.caption:
            continue
        if not scene.caption_words:
            return False
        if _caption_text(scene.caption_words) != scene.caption.strip():
            return False
        embedded.extend(scene.caption_words)

    if len(embedded) != len(transcript.words):
        return False

    return all(
        saved.text == current.text
        and abs(saved.start - current.start) <= 0.001
        and abs(saved.end - current.end) <= 0.001
        for saved, current in zip(embedded, transcript.words)
    )


def attach_caption_timings(plan: ShotPlan, transcript: Transcript) -> None:
    """Attach cached speech timing without changing visuals or scene boundaries.

    Explicit A/B input must match every caption; fail before mutating the plan
    if the transcript belongs to a different narration or edit.
    """
    # Current materialized plans already contain the original word timestamps.
    # Keep those authoritative timings when --transcript points to that same
    # transcript instead of re-deriving scene membership from cut boundaries.
    if _plan_already_has_transcript_timings(plan, transcript):
        return

    matched: list[list[Word]] = []
    for index, scene in enumerate(plan.scenes):
        if not scene.caption:
            matched.append([])
            continue
        words = [w for w in transcript.words
                 if w.start >= scene.start - 0.001 and w.end <= scene.end + 0.001]
        if _caption_text(words) != scene.caption.strip():
            raise ValueError(f"scene {index}: transcript does not match caption; use the original transcript.json")
        matched.append(words)
    for scene, words in zip(plan.scenes, matched):
        scene.caption_words = list(words)


def load_transcript(path: str | Path) -> Transcript:
    """Load a provider-neutral timed transcript.

    Expected JSON:
    {
      "language": "ru",
      "words": [{"start": 0.0, "end": 0.5, "text": "Привет"}]
    }
    """
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    words = [
        Word(
            start=float(item["start"]),
            end=float(item["end"]),
            text=str(item["text"]).strip(),
        )
        for item in data.get("words", [])
        if str(item.get("text", "")).strip()
    ]
    _validate_words(words)
    return Transcript(words=words, language=data.get("language"))


def save_transcript(transcript: Transcript, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "language": transcript.language,
        "duration": transcript.duration,
        "text": transcript.text,
        "words": [
            {"start": word.start, "end": word.end, "text": word.text}
            for word in transcript.words
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def transcribe_local(
    media: str | Path,
    *,
    model_size: str = "small",
    language: str | None = None,
    device: str = "cpu",
) -> Transcript:
    """Transcribe locally with faster-whisper.

    CPU/int8 is the default for the MVP because it works on ordinary Windows
    machines without requiring a CUDA/cuBLAS/cuDNN installation. GPU support can
    be exposed later as an explicit opt-in once the required NVIDIA runtime is
    present and verified.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "Local transcription requires the optional extra: "
            "pip install 'video-ai[transcribe]'"
        ) from exc

    selected_device = (device or "cpu").lower()
    compute_type = "int8" if selected_device == "cpu" else "float16"
    model = WhisperModel(model_size, device=selected_device, compute_type=compute_type)
    segments, info = model.transcribe(
        str(media),
        language=language,
        word_timestamps=True,
        vad_filter=True,
    )

    words: list[Word] = []
    for segment in segments:
        for raw in segment.words or []:
            text = str(raw.word).strip()
            if not text:
                continue
            words.append(Word(start=float(raw.start), end=float(raw.end), text=text))

    _validate_words(words)
    detected = language or getattr(info, "language", None)
    return Transcript(words=words, language=detected)


def _validate_words(words: list[Word]) -> None:
    if not words:
        raise ValueError("Transcript contains no timed words")
    previous_end = 0.0
    for index, word in enumerate(words):
        if word.start < 0 or word.end <= word.start:
            raise ValueError(f"word {index}: invalid time range")
        if word.start + 0.25 < previous_end:
            raise ValueError(f"word {index}: timestamps are not monotonic")
        previous_end = max(previous_end, word.end)
