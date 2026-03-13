"""Competition evaluator classes and initialization helpers."""

from __future__ import annotations

from typing import Optional, Tuple

import clip
import kagglehub
import torch
import torch.nn as nn
from PIL import Image

from .config import default_device


class VQAEvaluator:
    """Evaluates images based on text-image alignment."""

    def __init__(self) -> None:
        pass

    def score(self, image: Image.Image, description: str) -> float:
        # TODO: Preserve notebook behavior (placeholder returns 0).
        return 0.0


class AestheticPredictor(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)


class AestheticEvaluator:
    def __init__(self, device: Optional[str] = None):
        self.device = device or default_device()
        self.model_path = (
            kagglehub.notebook_output_download("metric/sac-logos-ava1-l14-linearmse")
            + "/sac+logos+ava1-l14-linearMSE.pth"
        )
        self.clip_model_path = (
            kagglehub.notebook_output_download("metric/openai-clip-vit-large-patch14")
            + "/ViT-L-14.pt"
        )
        self.predictor, self.clip_model, self.preprocessor = self.load()

    def load(self):
        state_dict = torch.load(
            self.model_path,
            weights_only=True,
            map_location=self.device,
        )
        predictor = AestheticPredictor(768)
        predictor.load_state_dict(state_dict)
        predictor.to(self.device)
        predictor.eval()
        clip_model, preprocessor = clip.load(self.clip_model_path, device=self.device)
        return predictor, clip_model, preprocessor

    def score(self, image: Image.Image) -> float:
        image_tensor = self.preprocessor(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            image_features = image_features.cpu().detach().numpy()
        score = self.predictor(torch.from_numpy(image_features).to(self.device).float())
        return score.item() / 10.0


_global_vqa_evaluator: Optional[VQAEvaluator] = None
_global_aesthetic_evaluator: Optional[AestheticEvaluator] = None


def initialize_evaluators(
    device: Optional[str] = None,
    force_reload: bool = False,
) -> Tuple[VQAEvaluator, AestheticEvaluator]:
    global _global_vqa_evaluator, _global_aesthetic_evaluator

    if force_reload:
        _global_vqa_evaluator = None
        _global_aesthetic_evaluator = None

    if _global_vqa_evaluator is None:
        print("Initializing VQA Evaluator...")
        _global_vqa_evaluator = VQAEvaluator()

    if _global_aesthetic_evaluator is None:
        print("Initializing Aesthetic Evaluator...")
        _global_aesthetic_evaluator = AestheticEvaluator(device=device)

    return _global_vqa_evaluator, _global_aesthetic_evaluator
