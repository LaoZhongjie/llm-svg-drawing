# Drawing with LLMs

Refactored Kaggle project for the **Drawing with LLMs** competition.  
This repository keeps the original notebook and extracts reusable logic into a clean Python package for local experimentation and GitHub presentation.

## Pipeline Overview

1. Prompt engineering (`prefix + prompt + suffix`)
2. Bitmap generation with SDXL + Lightning UNet + LoRA
3. Bitmap-to-SVG layered conversion under size constraints
4. Scoring with competition metric wrappers (aesthetic + placeholder VQA)
5. Final SVG post-processing for Kaggle submission output

## Project Structure

```text
drawing-with-llms/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── sdxl-lora-original.ipynb
├── src/
│   └── drawing_llms/
│       ├── __init__.py
│       ├── config.py
│       ├── metrics.py
│       ├── evaluators.py
│       ├── model_loader.py
│       ├── bitmap_generator.py
│       ├── svg_converter.py
│       ├── pipeline.py
│       ├── postprocess.py
│       └── kaggle_model.py
├── scripts/
│   ├── run_single.py
│   ├── evaluate_train.py
│   └── export_submission.py
├── examples/
│   └── README.md
└── outputs/
    └── .gitkeep
```

## Quick Start

```bash
pip install -r requirements.txt
```

Run a single prompt:

```bash
python scripts/run_single.py --prompt "a lighthouse overlooking the ocean"
```

Evaluate on train CSV:

```bash
python scripts/evaluate_train.py --csv /path/to/train.csv --limit 10
```

Export a submission CSV:

```bash
python scripts/export_submission.py --input-csv /path/to/test.csv --output-csv outputs/submission.csv
```

## Key Techniques

- SDXL base + Lightning UNet weights + LoRA adaptation
- Color quantization + contour extraction + polygon simplification for SVG
- Size-aware adaptive SVG filling (under byte constraints)
- Multi-attempt generation with metric-based best-candidate selection
- Lightweight post-processing for final submission formatting

## Notes

- Original notebook logic is preserved and modularized (no major algorithm rewrite).
- Some evaluator pieces remain placeholders by design (same as notebook behavior).
