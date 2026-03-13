#!/usr/bin/env python3
"""Batch evaluation on Kaggle train.csv or a local CSV file."""

from __future__ import annotations

import argparse
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from drawing_llms.kaggle_model import Model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        default="/kaggle/input/drawing-with-llms/train.csv",
        help="CSV containing prompt descriptions.",
    )
    parser.add_argument(
        "--description-col",
        default="description",
        help="Prompt text column.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of rows.",
    )
    parser.add_argument(
        "--results-csv",
        default=str(PROJECT_ROOT / "outputs" / "train_eval_results.csv"),
        help="Where to save per-prompt scores and timings.",
    )
    parser.add_argument(
        "--save-svgs-dir",
        default=str(PROJECT_ROOT / "outputs" / "eval_svgs"),
        help="Directory to store generated SVGs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)
    if args.limit is not None:
        df = df.head(args.limit)

    if args.description_col not in df.columns:
        raise ValueError(f"Column '{args.description_col}' not found in {args.csv}")

    model = Model()
    scores = []
    generation_times = []
    records = []

    save_svgs_dir = Path(args.save_svgs_dir)
    save_svgs_dir.mkdir(parents=True, exist_ok=True)

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        description = row[args.description_col]
        start_time = time.time()
        svg = model.predict(description)
        generation_time = time.time() - start_time

        score = model.last_score
        scores.append(score)
        generation_times.append(generation_time)

        svg_path = save_svgs_dir / f"sample_{i:04d}.svg"
        svg_path.write_text(svg, encoding="utf-8")

        records.append(
            {
                "index": i,
                "description": description,
                "score": score,
                "generation_time_sec": generation_time,
                "svg_path": str(svg_path),
            }
        )

        current_avg_score = np.mean(scores) if scores else 0
        current_avg_time = np.mean(generation_times) if generation_times else 0
        print(f"Processed {i}/{len(df)} prompts")
        print(f"Current average score: {current_avg_score:.2f}")
        print(f"Time for this prompt: {generation_time:.2f}s")
        print(f"Current average generation time: {current_avg_time:.2f}s")

    avg_score = np.mean(scores) if scores else 0
    avg_generation_time = np.mean(generation_times) if generation_times else 0
    total_time_taken = sum(generation_times)
    projected_time_500_images = 500 * avg_generation_time if avg_generation_time > 0 else 0
    projected_hours = projected_time_500_images / 3600

    print("\n=== SUMMARY ===")
    print(f"Prompts processed: {len(df)}")
    print(f"Final average score: {avg_score:.2f}")
    print(f"Average generation time per prompt: {avg_generation_time:.2f} seconds")
    print(f"Total time elapsed: {timedelta(seconds=total_time_taken)}")
    print(
        "Projected time for 500 prompts: "
        f"{projected_hours:.2f} hours ({timedelta(seconds=projected_time_500_images)})"
    )

    results_path = Path(args.results_csv)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(results_path, index=False)
    print(f"Saved evaluation results to: {results_path}")


if __name__ == "__main__":
    main()
