"""Kaggle submission Model class."""

from __future__ import annotations

from .config import GenerationConfig, PromptConfig
from .pipeline import generate_and_convert
from .postprocess import modify_svg


class Model:
    def __init__(
        self,
        generation_config: GenerationConfig | None = None,
        prompt_config: PromptConfig | None = None,
    ):
        generation_config = generation_config or GenerationConfig()
        prompt_config = prompt_config or PromptConfig()

        self.num_attempts_per_prompt = generation_config.num_attempts_per_prompt
        self.num_inference_steps = generation_config.num_inference_steps
        self.guidance_scale = generation_config.guidance_scale

        self.prompt_prefix = prompt_config.prompt_prefix
        self.prompt_suffix = prompt_config.prompt_suffix
        self.negative_prompt = prompt_config.negative_prompt
        self.last_score = None

    def modify_svg(self, full_svg_str: str) -> str:
        return modify_svg(full_svg_str)

    def predict(self, prompt: str) -> str:
        best_svg, best_score = generate_and_convert(
            prompt,
            prompt_prefix=self.prompt_prefix,
            prompt_suffix=self.prompt_suffix,
            negative_prompt=self.negative_prompt,
            num_attempts=self.num_attempts_per_prompt,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            verbose=False,
        )
        self.last_score = best_score
        if best_svg is None:
            # TODO: Keep fallback explicit if generation fails unexpectedly.
            return '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 384 384"></svg>'
        return self.modify_svg(best_svg)
