#!/usr/bin/env python3
"""Command-line interface for the OCR model.

Usage
-----
    python main.py path/to/report.jpg
    python main.py path/to/report.pdf --backend easyocr
    python main.py path/to/report.png --no-embeddings --output result.json

Options
-------
    --backend       OCR backend: tesseract (default) or easyocr.
    --lang          Comma-separated language codes (default: en).
    --no-embeddings Disable transformer embedding (faster, offline).
    --denoise       NLM denoising filter strength (default: 10.0, 0 = off).
    --output        Write JSON output to this file instead of stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract structured medical parameters from an image or PDF."
    )
    parser.add_argument("source", help="Path to the input file (JPEG / PNG / PDF).")
    parser.add_argument(
        "--backend",
        default="tesseract",
        choices=["tesseract", "easyocr"],
        help="OCR backend (default: tesseract).",
    )
    parser.add_argument(
        "--lang",
        default="en",
        help="Comma-separated OCR language codes (default: en).",
    )
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Disable transformer embeddings (faster, no GPU/transformers needed).",
    )
    parser.add_argument(
        "--denoise",
        type=float,
        default=10.0,
        help="NLM denoising filter strength (default: 10.0; 0 to disable).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write JSON result to this file (default: print to stdout).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    source = Path(args.source)

    if not source.exists():
        print(f"Error: file not found: {source}", file=sys.stderr)
        sys.exit(1)

    from ocr_model import OCRModel

    model = OCRModel(
        ocr_backend=args.backend,
        ocr_languages=args.lang.split(","),
        use_embeddings=not args.no_embeddings,
        denoise_h=args.denoise,
    )

    result = model.process(source)
    output_str = json.dumps(result, indent=2)

    if args.output:
        Path(args.output).write_text(output_str)
        print(f"Results written to {args.output}")
    else:
        print(output_str)


if __name__ == "__main__":
    main()
