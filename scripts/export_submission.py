#!/usr/bin/env python3
"""Generate a Kaggle submission CSV from a prompt CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from drawing_llms.kaggle_model import Model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        default="/kaggle/input/drawing-with-llms/test.csv",
        help="Input CSV with prompt column.",
    )
    parser.add_argument(
        "--output-csv",
        default=str(PROJECT_ROOT / "outputs" / "submission.csv"),
        help="Output submission path.",
    )
    parser.add_argument("--id-col", default="id", help="ID column name.")
    parser.add_argument("--prompt-col", default="description", help="Prompt column name.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of rows for quick dry-runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    if args.limit is not None:
        df = df.head(args.limit)

    for required_col in (args.id_col, args.prompt_col):
        if required_col not in df.columns:
            raise ValueError(f"Column '{required_col}' not found in {args.input_csv}")

    model = Model()
    outputs = []
    total = len(df)

    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        prompt = row[args.prompt_col]
        svg = model.predict(prompt)
        outputs.append({args.id_col: row[args.id_col], "svg": svg})
        print(f"Generated {idx}/{total}")

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(outputs).to_csv(output_path, index=False)
    print(f"Saved submission CSV to: {output_path}")


if __name__ == "__main__":
    main()
