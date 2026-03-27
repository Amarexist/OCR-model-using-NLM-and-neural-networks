"""OCR Model using NLM and Neural Networks.

This package provides a comprehensive OCR pipeline that:
- Extracts text from JPEG, PNG, and PDF documents.
- Applies Non-Local Means (NLM) image denoising for better OCR accuracy.
- Uses a transformer-based self-attention encoder for contextual embedding.
- Performs spelling correction and medical-term relation extraction.
- Employs a neural-network parameter extractor to produce structured JSON output.
- Classifies numerical values against medical reference ranges (normal / low / high).
"""

from ocr_model.model import OCRModel  # noqa: F401

__all__ = ["OCRModel"]
