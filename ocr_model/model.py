"""Main OCRModel orchestrator.

Usage
-----
>>> from ocr_model import OCRModel
>>> model = OCRModel()
>>> result = model.process("report.jpg")
>>> import json; print(json.dumps(result, indent=2))

The returned dictionary has the shape::

    {
        "raw_text": "...",
        "corrected_text": "...",
        "parameters": {
            "hemoglobin": {"value": 14.0, "unit": "g/dL", "status": "normal"},
            "wbc":        {"value": 15000.0, "unit": "cells/µL", "status": "high"},
            "platelets":  {"value": 250000.0, "unit": "cells/µL", "status": "normal"}
        },
        "diseases": ["Leukocytosis (Infection / Inflammation)"],
        "pages": 1
    }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ocr_model.preprocessor import load_pages
from ocr_model.ocr_engine import OCREngine, Backend
from ocr_model.nlm_processor import NLMProcessor
from ocr_model.neural_extractor import NeuralExtractor
from ocr_model.medical_classifier import classify_value, infer_diseases


class OCRModel:
    """End-to-end OCR pipeline combining NLM processing and neural extraction.

    Parameters
    ----------
    ocr_backend:
        OCR backend to use: ``"tesseract"`` (default) or ``"easyocr"``.
    ocr_languages:
        List of language codes for the OCR engine.
    nlm_model:
        HuggingFace model identifier for the self-attention embedder.
        Defaults to ``"distilbert-base-uncased"``.
    use_embeddings:
        Whether to generate transformer embeddings (requires ``transformers``
        and ``torch``).  Set to ``False`` to skip the embedding step and use
        only the rule-based extractor — useful for fast / offline inference.
    denoise_h:
        NLM filter strength (0 disables denoising).
    min_confidence:
        Minimum MLP confidence score for a candidate to be included.
    """

    def __init__(
        self,
        ocr_backend: Backend = "tesseract",
        ocr_languages: Optional[List[str]] = None,
        nlm_model: str = "distilbert-base-uncased",
        use_embeddings: bool = True,
        denoise_h: float = 10.0,
        min_confidence: float = 0.0,
    ) -> None:
        self._engine = OCREngine(
            backend=ocr_backend, languages=ocr_languages or ["en"]
        )
        self._nlm = NLMProcessor(model_name=nlm_model, denoise_h=denoise_h)
        self._extractor = NeuralExtractor(min_score=min_confidence)
        self._use_embeddings = use_embeddings

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, source: "str | Path") -> Dict[str, Any]:
        """Process a document and return structured medical parameters.

        Parameters
        ----------
        source:
            Path to a JPEG, PNG, PDF, or other supported image file.

        Returns
        -------
        dict
            Structured result as described in the module docstring.
        """
        pages = load_pages(source)
        all_raw_text: List[str] = []

        for page_img in pages:
            # 1. NLM denoising (improves OCR on noisy scans)
            if self._nlm.denoiser_h > 0:
                page_img = self._nlm.denoise(page_img)

            # 2. OCR
            raw_text = self._engine.extract_text(page_img)
            all_raw_text.append(raw_text)

        raw_text_combined = "\n".join(all_raw_text)

        return self._analyse(raw_text_combined, num_pages=len(pages))

    def process_text(self, raw_text: str) -> Dict[str, Any]:
        """Analyse already-extracted OCR text (skip image loading / OCR).

        Useful for testing or when text is obtained from an external OCR
        service.

        Parameters
        ----------
        raw_text:
            Raw OCR text string.

        Returns
        -------
        dict
            Structured result.
        """
        return self._analyse(raw_text, num_pages=1)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _analyse(self, raw_text: str, num_pages: int) -> Dict[str, Any]:
        # 3. Spelling correction
        corrected = self._nlm.correct_spelling(raw_text)

        # 4. Self-attention embedding (optional)
        embedding = None
        if self._use_embeddings:
            try:
                embedding = self._nlm.embed(corrected)
            except Exception:
                # Non-fatal — fall back to rule-based extraction only.
                embedding = None

        # 5. Neural parameter extraction
        candidates = self._extractor.extract(corrected, embedding)

        # 6. Medical classification
        parameters: Dict[str, Any] = {}
        statuses: Dict[str, str] = {}

        for cand in candidates:
            param = cand["param"]
            value = cand["value"]
            unit = cand["unit"]
            status, ref = classify_value(param, value)

            # Use reference unit if OCR unit is missing
            if unit is None and ref is not None:
                unit = ref.unit

            parameters[param] = {
                "value": value,
                "unit": unit,
                "status": status,
            }
            statuses[param] = status

        # 7. Disease inference
        diseases = infer_diseases(statuses)

        return {
            "raw_text": raw_text,
            "corrected_text": corrected,
            "parameters": parameters,
            "diseases": diseases,
            "pages": num_pages,
        }

    # ------------------------------------------------------------------
    # Convenience serialisation
    # ------------------------------------------------------------------

    def process_to_json(self, source: "str | Path", indent: int = 2) -> str:
        """Process *source* and return the result as a JSON string."""
        result = self.process(source)
        return json.dumps(result, indent=indent)
