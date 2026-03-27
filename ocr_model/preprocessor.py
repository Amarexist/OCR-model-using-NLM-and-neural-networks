"""Image and PDF preprocessing utilities.

Steps performed:
1. Load the source (JPEG / PNG / PDF page).
2. Convert to grayscale.
3. Apply CLAHE (Contrast Limited Adaptive Histogram Equalisation).
4. Deskew using the Hough line-transform.
5. Optionally upscale low-resolution scans.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import List

import cv2
import numpy as np
from PIL import Image


def _pil_to_cv(img: Image.Image) -> np.ndarray:
    """Convert a PIL image (RGB) to an OpenCV array (BGR)."""
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def _cv_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert an OpenCV BGR array to a PIL RGB image."""
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def _to_grayscale(arr: np.ndarray) -> np.ndarray:
    if len(arr.shape) == 2:
        return arr
    return cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)


def _clahe_enhance(gray: np.ndarray) -> np.ndarray:
    """Improve local contrast with CLAHE."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _deskew(gray: np.ndarray) -> np.ndarray:
    """Rotate the image to correct skew detected via Hough lines."""
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=200)
    if lines is None:
        return gray

    angles = []
    for rho, theta in lines[:, 0]:
        angle = math.degrees(theta) - 90
        if abs(angle) < 45:  # ignore near-vertical lines
            angles.append(angle)

    if not angles:
        return gray

    median_angle = float(np.median(angles))
    if abs(median_angle) < 0.5:
        return gray  # negligible skew

    h, w = gray.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(
        gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


def _upscale_if_needed(gray: np.ndarray, min_height: int = 1000) -> np.ndarray:
    """Upscale images that are too small for reliable OCR."""
    h, w = gray.shape
    if h < min_height:
        scale = min_height / h
        new_h, new_w = int(h * scale), int(w * scale)
        return cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return gray


def _binarize(gray: np.ndarray) -> np.ndarray:
    """Apply adaptive thresholding to produce a clean binary image."""
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )


def preprocess_image(source: "str | Path | Image.Image") -> Image.Image:
    """Full preprocessing pipeline for a single image.

    Parameters
    ----------
    source:
        A file path (str / Path) or an already-loaded PIL image.

    Returns
    -------
    Image.Image
        Pre-processed PIL image ready for OCR.
    """
    if isinstance(source, (str, Path)):
        pil = Image.open(source).convert("RGB")
    else:
        pil = source.convert("RGB")

    arr = _pil_to_cv(pil)
    gray = _to_grayscale(arr)
    gray = _upscale_if_needed(gray)
    gray = _clahe_enhance(gray)
    gray = _deskew(gray)
    binary = _binarize(gray)
    return Image.fromarray(binary)


def load_pages(source: "str | Path") -> List[Image.Image]:
    """Load a document (image or PDF) and return a list of PIL images.

    For PDFs each page is returned as a separate image at 300 DPI.
    For images the list contains exactly one element.

    Parameters
    ----------
    source:
        Path to a JPEG, PNG, or PDF file.
    """
    path = Path(source)
    ext = path.suffix.lower()

    if ext == ".pdf":
        try:
            from pdf2image import convert_from_path  # type: ignore

            pages = convert_from_path(str(path), dpi=300)
            return [preprocess_image(p) for p in pages]
        except ImportError as exc:
            raise ImportError(
                "pdf2image is required to process PDF files. "
                "Install it with: pip install pdf2image"
            ) from exc

    if ext in {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}:
        return [preprocess_image(path)]

    raise ValueError(f"Unsupported file extension: {ext!r}")
