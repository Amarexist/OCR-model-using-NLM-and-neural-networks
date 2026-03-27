"""OCR engine wrapper.

Supports two backends selectable at runtime:
- ``"tesseract"``  — pytesseract (default; must have Tesseract installed)
- ``"easyocr"``   — EasyOCR (pure Python, GPU-capable)

The engine returns raw text extracted from a PIL image.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from PIL import Image


Backend = Literal["tesseract", "easyocr"]


class OCREngine:
    """Wraps Tesseract and EasyOCR behind a uniform interface.

    Parameters
    ----------
    backend:
        Which OCR library to use.
    languages:
        List of language codes to pass to the engine
        (e.g. ``["en"]`` or ``["en", "fr"]``).
    """

    def __init__(
        self,
        backend: Backend = "tesseract",
        languages: Optional[List[str]] = None,
    ) -> None:
        self.backend = backend
        self.languages = languages or ["en"]
        self._easyocr_reader = None  # lazy-loaded

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_text(self, image: Image.Image) -> str:
        """Extract raw text from a pre-processed PIL image.

        Parameters
        ----------
        image:
            A PIL image (typically already pre-processed).

        Returns
        -------
        str
            Raw OCR text.
        """
        if self.backend == "tesseract":
            return self._tesseract(image)
        if self.backend == "easyocr":
            return self._easyocr(image)
        raise ValueError(f"Unknown OCR backend: {self.backend!r}")

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------

    def _tesseract(self, image: Image.Image) -> str:
        try:
            import pytesseract  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "pytesseract is required for the Tesseract backend. "
                "Install it with: pip install pytesseract"
            ) from exc

        lang_str = "+".join(self.languages)
        config = "--oem 3 --psm 6"
        return pytesseract.image_to_string(image, lang=lang_str, config=config)

    def _easyocr(self, image: Image.Image) -> str:
        try:
            import easyocr  # type: ignore
            import numpy as np
        except ImportError as exc:
            raise ImportError(
                "easyocr is required for the EasyOCR backend. "
                "Install it with: pip install easyocr"
            ) from exc

        if self._easyocr_reader is None:
            self._easyocr_reader = easyocr.Reader(
                self.languages, gpu=False, verbose=False
            )

        arr = np.array(image.convert("RGB"))
        results = self._easyocr_reader.readtext(arr, detail=0, paragraph=True)
        return "\n".join(results)
