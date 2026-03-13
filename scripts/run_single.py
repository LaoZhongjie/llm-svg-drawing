#!/usr/bin/env python3
"""Run one prompt through the full Drawing-with-LLMs pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from drawing_llms.kaggle_model import Model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True, help="Text prompt to draw.")
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "outputs" / "single.svg"),
        help="Path to save the generated SVG.",
    )
    parser.add_argument("--num-attempts", type=int, default=4)
    parser.add_argument("--num-inference-steps", type=int, default=9)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument(
        "--prompt-prefix",
        default="Simple, vector, color drawing,",
    )
    parser.add_argument(
        "--prompt-suffix",
        default=(
            "cartoon style, simple details, vivid colors, complementary colors,"
            "saturated solors,limited color palette,clear, uncluttered,expressive,dynamic"
        ),
    )
    parser.add_argument("--negative-prompt", default="deformed,ugly")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = Model()
    model.num_attempts_per_prompt = args.num_attempts
    model.num_inference_steps = args.num_inference_steps
    model.guidance_scale = args.guidance_scale
    model.prompt_prefix = args.prompt_prefix
    model.prompt_suffix = args.prompt_suffix
    model.negative_prompt = args.negative_prompt

    svg = model.predict(args.prompt)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg, encoding="utf-8")

    print(f"Saved SVG to: {output_path}")
    if model.last_score is not None:
        print(f"Best score: {model.last_score:.4f}")


if __name__ == "__main__":
    main()
