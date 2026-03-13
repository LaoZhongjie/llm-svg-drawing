"""End-to-end generation pipeline orchestration."""

from __future__ import annotations

import time

import matplotlib.pyplot as plt

from .bitmap_generator import generate_bitmap
from .metrics import evaluate_with_competition_metric, svg_to_png
from .svg_converter import bitmap_to_svg_layered


def generate_and_convert(
    prompt: str,
    prompt_prefix: str = "",
    prompt_suffix: str = "",
    negative_prompt: str = "",
    num_attempts: int = 3,
    num_inference_steps: int = 20,
    guidance_scale: float = 15,
    verbose: bool = True,
):
    if num_attempts <= 0:
        raise ValueError("num_attempts must be >= 1")

    best_svg = None
    best_bitmap = None
    best_similarity = -1.0

    total_start_time = time.time()
    generation_times = []
    conversion_times = []
    evaluation_times = []
    attempt_times = []

    combined_prompt = prompt_prefix + " " + prompt + " " + prompt_suffix

    for i in range(num_attempts):
        attempt_start_time = time.time()
        if verbose:
            print(f"\n=== Attempt {i + 1}/{num_attempts} ===")

        generation_start = time.time()
        bitmap = generate_bitmap(
            combined_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
        generation_end = time.time()
        generation_time = generation_end - generation_start
        generation_times.append(generation_time)

        if verbose:
            print("Converting to SVG... ", end="")
        conversion_start = time.time()
        svg_content = bitmap_to_svg_layered(bitmap, max_size_bytes=9800)
        conversion_end = time.time()
        conversion_time = conversion_end - conversion_start
        conversion_times.append(conversion_time)

        rendered_svg = svg_to_png(svg_content)
        svg_size = len(svg_content.encode("utf-8"))
        if verbose:
            print(f"SVG size: {svg_size} bytes")

        if verbose:
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(bitmap)
            plt.title(f"Original Image {i + 1}")
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.imshow(rendered_svg)
            plt.title(f"SVG Conversion {i + 1}")
            plt.axis("off")

            plt.tight_layout()
            plt.show()

        evaluation_start = time.time()
        svg_scores = evaluate_with_competition_metric(svg_content, prompt)
        evaluation_end = time.time()
        evaluation_time = evaluation_end - evaluation_start
        evaluation_times.append(evaluation_time)

        if verbose:
            print(f"SVG VQA Score: {svg_scores['vqa_score']:.4f}")
            print(f"SVG Aesthetic Score: {svg_scores['aesthetic_score']:.4f}")
            print(f"SVG Competition Score: {svg_scores['combined_score']:.4f}")

        if svg_scores["combined_score"] > best_similarity:
            best_similarity = svg_scores["combined_score"]
            best_svg = svg_content
            best_bitmap = bitmap
            if verbose:
                print(f"✅ New best result: {svg_scores['combined_score']:.4f}")
        else:
            if verbose:
                print(f"❌ Not better than current best: {best_similarity:.4f}")

        attempt_end_time = time.time()
        attempt_time = attempt_end_time - attempt_start_time
        attempt_times.append(attempt_time)

        if verbose:
            print(f"Image generation time: {generation_time:.2f}s")
            print(f"SVG conversion time: {conversion_time:.2f}s")
            print(f"Image evaluation time: {evaluation_time:.2f}s")
            print(f"Total time for attempt {i + 1}: {attempt_time:.2f}s")

    total_end_time = time.time()
    total_time = total_end_time - total_start_time

    if verbose:
        print("\n=== Timing Summary ===")
        print(
            f"Average image generation time: {sum(generation_times) / len(generation_times):.2f}s"
        )
        print(
            f"Average SVG conversion time: {sum(conversion_times) / len(conversion_times):.2f}s"
        )
        print(
            f"Average image evaluation time: {sum(evaluation_times) / len(evaluation_times):.2f}s"
        )
        print(f"Average time per attempt: {sum(attempt_times) / len(attempt_times):.2f}s")
        print(f"Total processing time ({num_attempts} attempts): {total_time:.2f}s")
        print(f"Best score achieved: {best_similarity:.4f}")

    if best_bitmap is None and verbose:
        print("No bitmap was generated.")
    return best_svg, best_similarity
