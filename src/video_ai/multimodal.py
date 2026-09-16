from __future__ import annotations

from pathlib import Path


class ClipRanker:
    """Optional local image/text similarity ranker.

    The heavy ML stack is intentionally optional. Install `video-ai[semantic]` and
    enable semantic ranking from the CLI. The model is downloaded once and then
    cached by Hugging Face locally.
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32") -> None:
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
            try:
                image = Image.open(path).convert("RGB")
            except Exception:
                continue
            images.append(image)
            valid_indexes.append(index)

        if not images:
            return scores

        text_inputs = self.processor(text=[prompt], return_tensors="pt", padding=True)
        image_inputs = self.processor(images=images, return_tensors="pt")
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        pixel_values = image_inputs["pixel_values"].to(self.device)

        with self.torch.no_grad():
            text_features = self.model.get_text_features(**text_inputs)
            image_features = self.model.get_image_features(pixel_values=pixel_values)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            similarities = (image_features @ text_features.T).squeeze(-1).detach().cpu().tolist()

        if isinstance(similarities, float):
            similarities = [similarities]
        for original_index, value in zip(valid_indexes, similarities):
            scores[original_index] = float(value)
        return scores
