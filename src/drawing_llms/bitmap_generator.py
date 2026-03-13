"""Bitmap image generation from prompts."""

from __future__ import annotations

from PIL import Image

from .model_loader import get_generation_pipeline


def generate_bitmap(
    prompt: str,
    negative_prompt: str = "",
    num_inference_steps: int = 20,
    guidance_scale: float = 15,
    pipe=None,
) -> Image.Image:
    active_pipe = pipe or get_generation_pipeline()
    image = active_pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    ).images[0]
    return image
