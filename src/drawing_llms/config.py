"""Configuration values shared across the project."""

from dataclasses import dataclass

import torch


def default_device(prefer_second_gpu: bool = True) -> str:
    if not torch.cuda.is_available():
        return "cpu"
    if prefer_second_gpu and torch.cuda.device_count() > 1:
        return "cuda:1"
    return "cuda:0"


@dataclass
class GenerationConfig:
    num_attempts_per_prompt: int = 4
    num_inference_steps: int = 9
    guidance_scale: float = 3.0


@dataclass
class PromptConfig:
    prompt_prefix: str = "Simple, vector, color drawing,"
    prompt_suffix: str = (
        "cartoon style, simple details, vivid colors, complementary colors,"
        "saturated solors,limited color palette,clear, uncluttered,expressive,dynamic"
    )
    negative_prompt: str = "deformed,ugly"


DEFAULT_MAX_SVG_BYTES = 9800
DEFAULT_TARGET_SIZE = (384, 384)
