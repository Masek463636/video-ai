from __future__ import annotations

import shutil
import subprocess
from functools import lru_cache
from pathlib import Path


class ClipRanker:
    """Optional local image/text similarity ranker.

    The heavy ML stack is intentionally optional. Install `video-ai[semantic]` and
    enable semantic ranking from the CLI. One model instance is reused for all
    scenes in a process so a 30-scene Short does not reload CLIP 30 times.
    """

    _instances: dict[str, "ClipRanker"] = {}

    def __new__(cls, model_name: str = "openai/clip-vit-base-patch32"):
        if model_name not in cls._instances:
            cls._instances[model_name] = super().__new__(cls)
        return cls._instances[model_name]

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32") -> None:
        if getattr(self, "_initialized", False):
            return
        try:
            import torch  # type: ignore
            from transformers import CLIPModel, CLIPProcessor  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Semantic visual ranking requires: pip install -e \".[semantic]\""
            ) from exc

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self._initialized = True

    def score_images(self, prompt: str, paths: list[str | Path]) -> list[float]:
        if not paths:
            return []
        try:
            from PIL import Image  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Semantic ranking requires Pillow") from exc

        images = []
        valid_indexes: list[int] = []
        scores = [0.0] * len(paths)
        for index, path in enumerate(paths):
            p = Path(path)
            try:
                if not _has_known_image_signature(p) or not _silent_decode_probe(p):
                    continue
                with Image.open(p) as opened:
                    opened.verify()
                with Image.open(p) as opened:
                    image = opened.convert("RGB")
                images.append(image)
                valid_indexes.append(index)
            except Exception:
                continue

        if not images:
            return scores

        # Use a single CLIP forward pass. Newer transformers versions can
        # return BaseModelOutputWithPooling from get_text_features/get_image_features,
        # so calling .norm() on those helper results crashes. CLIPModel.forward()
        # exposes projection-space embeddings directly.
        inputs = self.processor(
            text=[prompt],
            images=images,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with self.torch.no_grad():
            outputs = self.model(**inputs)
            image_features = outputs.image_embeds
            text_features = outputs.text_embeds
            similarities = (
                image_features @ text_features.T
            ).squeeze(-1).detach().cpu().tolist()

        if isinstance(similarities, float):
            similarities = [similarities]
        for original_index, value in zip(valid_indexes, similarities):
            scores[original_index] = float(value)
        return scores


def _has_known_image_signature(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            header = handle.read(16)
    except OSError:
        return False
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return True
    if header.startswith(b"\xff\xd8\xff"):
        return True
    if len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return True
    return False


def _silent_decode_probe(path: Path) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return True
    try:
        completed = subprocess.run(
            [
                ffmpeg,
                "-v", "error",
                "-xerror",
                "-err_detect", "explode",
                "-i", str(path),
                "-frames:v", "1",
                "-f", "null", "-",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=12,
            check=False,
        )
        return completed.returncode == 0 and not completed.stderr.strip()
    except Exception:
        return False


@lru_cache(maxsize=2)
def get_clip_ranker(model_name: str = "openai/clip-vit-base-patch32") -> ClipRanker:
    return ClipRanker(model_name)
