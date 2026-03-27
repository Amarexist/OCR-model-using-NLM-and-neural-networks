"""NLM (Natural Language Model) processor.

This module provides three independent services that together form the NLP
layer of the OCR pipeline:

1. **NLM image denoiser** — OpenCV fastNlMeansDenoising applied to raw
   scanned images before Tesseract sees them (improves recognition on noisy
   scans).

2. **Self-attention embedding** — a lightweight transformer encoder
   (DistilBERT by default) that converts the OCR text into contextual token
   embeddings.  These embeddings are used downstream by the neural extractor.

3. **Spelling corrector** — pyspellchecker is used to fix common OCR typos.
   Medical terms that appear in the built-in vocabulary are never "corrected".

4. **Relation extractor** — a rule-based / regex system that surfaces
   (entity, value, unit) triples from free-form text and maps them to a
   canonical parameter name.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# 1. NLM image denoiser
# ---------------------------------------------------------------------------

def nlm_denoise(image: Image.Image, h: float = 10.0) -> Image.Image:
    """Apply OpenCV Non-Local Means (NLM) denoising to a PIL image.

    Parameters
    ----------
    image:
        Input PIL image.
    h:
        Filter strength — higher removes more noise but also more detail.

    Returns
    -------
    Image.Image
        Denoised image.
    """
    arr = np.array(image.convert("L"))  # convert to grayscale
    denoised = cv2.fastNlMeansDenoising(arr, h=h, templateWindowSize=7, searchWindowSize=21)
    return Image.fromarray(denoised)


# ---------------------------------------------------------------------------
# 2. Self-attention embedding
# ---------------------------------------------------------------------------

class SelfAttentionEmbedder:
    """Wraps a HuggingFace transformer to produce contextual embeddings.

    The model is loaded lazily the first time ``embed`` is called.

    Parameters
    ----------
    model_name:
        Any HuggingFace model identifier.  Defaults to
        ``"distilbert-base-uncased"`` — small, fast, and CPU-friendly.
    max_length:
        Maximum token length passed to the tokeniser.
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self._tokenizer = None
        self._model = None

    def _load(self) -> None:
        try:
            from transformers import AutoModel, AutoTokenizer  # type: ignore
            import torch  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "transformers and torch are required for self-attention embedding. "
                "Install with: pip install transformers torch"
            ) from exc

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        self._model.eval()

    def embed(self, text: str) -> np.ndarray:
        """Return a (seq_len, hidden_size) embedding matrix for *text*.

        The [CLS] token embedding (index 0) represents the whole sequence.

        Parameters
        ----------
        text:
            Raw or corrected OCR text.

        Returns
        -------
        np.ndarray
            Shape ``(seq_len, hidden_size)``.
        """
        if self._model is None:
            self._load()

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        )
        with self._torch.no_grad():
            outputs = self._model(**inputs)

        # last_hidden_state shape: (1, seq_len, hidden_size)
        return outputs.last_hidden_state.squeeze(0).cpu().numpy()

    def sentence_embedding(self, text: str) -> np.ndarray:
        """Return a single (hidden_size,) mean-pooled sentence embedding."""
        token_embeddings = self.embed(text)
        return token_embeddings.mean(axis=0)


# ---------------------------------------------------------------------------
# 3. Spelling corrector
# ---------------------------------------------------------------------------

# Medical / domain terms that should NEVER be "corrected".
_MEDICAL_VOCAB = {
    "hemoglobin", "haemoglobin", "hematocrit", "haematocrit",
    "leukocyte", "erythrocyte", "thrombocyte", "platelet", "platelets",
    "wbc", "rbc", "mcv", "mch", "mchc", "rdw", "mpv",
    "neutrophil", "lymphocyte", "monocyte", "eosinophil", "basophil",
    "creatinine", "bilirubin", "albumin", "globulin", "glucose",
    "cholesterol", "triglyceride", "hdl", "ldl", "vldl",
    "sodium", "potassium", "chloride", "calcium", "phosphorus",
    "magnesium", "urea", "uric", "ast", "alt", "alp", "ggt",
    "tsh", "t3", "t4", "hba1c", "psa", "crp", "esr",
    "ferritin", "transferrin", "fibrinogen", "prothrombin",
    "inr", "aptt", "pt", "d-dimer",
    "infection", "anemia", "anaemia", "leukemia", "thrombocytopenia",
    "polycythemia", "neutropenia", "lymphopenia", "eosinophilia",
    "sepsis", "bacteremia", "viremia",
}


class SpellingCorrector:
    """Correct OCR spelling errors while preserving medical terminology.

    Parameters
    ----------
    distance:
        Edit distance used by pyspellchecker (1 or 2).
    """

    def __init__(self, distance: int = 1) -> None:
        self.distance = distance
        self._spell = None

    def _load(self) -> None:
        try:
            from spellchecker import SpellChecker  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "pyspellchecker is required for spelling correction. "
                "Install with: pip install pyspellchecker"
            ) from exc

        self._spell = SpellChecker(distance=self.distance)
        self._spell.word_frequency.load_words(_MEDICAL_VOCAB)

    def correct(self, text: str) -> str:
        """Return a spelling-corrected version of *text*.

        Only words that are clearly mis-spelled (not in the dictionary or the
        medical vocabulary) are replaced.

        Parameters
        ----------
        text:
            Raw OCR text.

        Returns
        -------
        str
            Text with corrected spelling.
        """
        if self._spell is None:
            self._load()

        tokens = re.findall(r"\S+|\s+", text)
        corrected_tokens = []
        for token in tokens:
            # Preserve whitespace tokens verbatim.
            if token.strip() == "":
                corrected_tokens.append(token)
                continue

            # Preserve tokens that contain non-ASCII characters (e.g. µ, °),
            # unit-like tokens (contain / ^ % digits), or single characters.
            if (
                not token.isascii()
                or re.search(r"[/\\^%\d]", token)
            ):
                corrected_tokens.append(token)
                continue

            # Preserve numbers, punctuation, and special tokens.
            word_only = re.sub(r"[^a-zA-Z]", "", token)
            if not word_only:
                corrected_tokens.append(token)
                continue

            lower = word_only.lower()
            if lower in _MEDICAL_VOCAB:
                corrected_tokens.append(token)
                continue

            correction = self._spell.correction(lower)
            if correction and correction != lower:
                # Preserve original capitalisation.
                if word_only[0].isupper():
                    correction = correction.capitalize()
                corrected_tokens.append(token.replace(word_only, correction))
            else:
                corrected_tokens.append(token)

        return "".join(corrected_tokens)


# ---------------------------------------------------------------------------
# 4. Relation extractor
# ---------------------------------------------------------------------------

# Mapping of possible surface forms → canonical parameter name.
_PARAM_ALIASES: Dict[str, str] = {
    # Haematology
    "hemoglobin": "hemoglobin", "haemoglobin": "hemoglobin",
    "hgb": "hemoglobin", "hb": "hemoglobin",
    "hematocrit": "hematocrit", "haematocrit": "hematocrit", "hct": "hematocrit",
    "rbc": "rbc", "red blood cell": "rbc", "red blood cells": "rbc",
    "wbc": "wbc", "white blood cell": "wbc", "white blood cells": "wbc",
    "white blood count": "wbc", "leukocyte": "wbc", "leukocytes": "wbc",
    "platelet": "platelets", "platelets": "platelets",
    "thrombocyte": "platelets", "thrombocytes": "platelets", "plt": "platelets",
    "mcv": "mcv", "mch": "mch", "mchc": "mchc",
    "rdw": "rdw", "mpv": "mpv",
    "neutrophil": "neutrophils", "neutrophils": "neutrophils",
    "lymphocyte": "lymphocytes", "lymphocytes": "lymphocytes",
    "monocyte": "monocytes", "monocytes": "monocytes",
    "eosinophil": "eosinophils", "eosinophils": "eosinophils",
    "basophil": "basophils", "basophils": "basophils",
    # Chemistry
    "glucose": "glucose", "blood glucose": "glucose", "fbs": "glucose",
    "creatinine": "creatinine", "urea": "urea",
    "sodium": "sodium", "na": "sodium",
    "potassium": "potassium", "k": "potassium",
    "chloride": "chloride", "cl": "chloride",
    "calcium": "calcium", "ca": "calcium",
    "cholesterol": "cholesterol", "total cholesterol": "cholesterol",
    "triglyceride": "triglycerides", "triglycerides": "triglycerides",
    "hdl": "hdl", "ldl": "ldl",
    "ast": "ast", "alt": "alt", "alp": "alp", "ggt": "ggt",
    "bilirubin": "bilirubin", "total bilirubin": "bilirubin",
    "albumin": "albumin", "globulin": "globulin",
    "uric acid": "uric_acid", "uric": "uric_acid",
    # Hormones
    "tsh": "tsh", "t3": "t3", "t4": "t4", "hba1c": "hba1c",
    # Inflammatory
    "crp": "crp", "c-reactive protein": "crp",
    "esr": "esr", "ferritin": "ferritin",
    # Coagulation
    "pt": "pt", "inr": "inr", "aptt": "aptt", "d-dimer": "d_dimer",
}

# Common unit patterns.
_UNIT_PATTERN = (
    r"(?:g/dl|g/l|mg/dl|mg/l|mmol/l|µmol/l|umol/l|iu/l|u/l|"
    r"10\^3/µl|10\^3/ul|10\^6/µl|10\^6/ul|cells/µl|cells/ul|"
    r"%|fl|pg|ng/ml|ng/dl|µg/dl|ug/dl|miu/ml|mu/l|mm/hr|"
    r"mmhg|g|mg|mcg|meq/l)"
)

# Pattern: PARAM_NAME  VALUE  UNIT?
_RELATION_PATTERN = re.compile(
    r"((?:[a-z][a-z0-9\- ]{0,30}))"         # group 1: param name (loose)
    r"[\s:=\|\-]+?"                          # separator
    r"([<>]?\d{1,7}(?:[.,]\d{1,4})?)"        # group 2: numeric value
    r"\s*"
    r"(" + _UNIT_PATTERN + r")?",            # group 3: optional unit
    re.IGNORECASE,
)


def extract_relations(text: str) -> List[Tuple[str, float, Optional[str]]]:
    """Extract (canonical_param, numeric_value, unit) triples from text.

    Parameters
    ----------
    text:
        OCR text (ideally spelling-corrected).

    Returns
    -------
    list of (canonical_param, value, unit)
        ``unit`` may be ``None`` when no unit is present in the source text.
    """
    results: List[Tuple[str, float, Optional[str]]] = []
    seen_params: set = set()

    # Normalise whitespace.
    text = re.sub(r"[ \t]+", " ", text)

    for m in _RELATION_PATTERN.finditer(text):
        raw_param = m.group(1).strip().lower()
        raw_value = m.group(2).replace(",", "")
        raw_unit = m.group(3)

        # Match against aliases (longest match first).
        canonical = None
        for alias, canon in sorted(_PARAM_ALIASES.items(), key=lambda x: -len(x[0])):
            if alias in raw_param or raw_param in alias:
                canonical = canon
                break

        if canonical is None:
            continue

        try:
            value = float(raw_value.lstrip("<>"))
        except ValueError:
            continue

        if canonical not in seen_params:
            results.append((canonical, value, raw_unit))
            seen_params.add(canonical)

    return results


# ---------------------------------------------------------------------------
# Facade
# ---------------------------------------------------------------------------

class NLMProcessor:
    """High-level facade that bundles image denoising, embedding, spelling
    correction, and relation extraction.

    Parameters
    ----------
    model_name:
        HuggingFace model to use for self-attention embedding.
    denoise_h:
        NLM filter strength (passed to :func:`nlm_denoise`).
    spell_distance:
        Edit distance for the spelling corrector.
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        denoise_h: float = 10.0,
        spell_distance: int = 1,
    ) -> None:
        self.denoiser_h = denoise_h
        self.embedder = SelfAttentionEmbedder(model_name=model_name)
        self.corrector = SpellingCorrector(distance=spell_distance)

    def denoise(self, image: Image.Image) -> Image.Image:
        """Apply NLM denoising to an image."""
        return nlm_denoise(image, h=self.denoiser_h)

    def correct_spelling(self, text: str) -> str:
        """Apply spelling correction to OCR text."""
        return self.corrector.correct(text)

    def embed(self, text: str) -> np.ndarray:
        """Return self-attention sentence embedding for *text*."""
        return self.embedder.sentence_embedding(text)

    def extract_relations(
        self, text: str
    ) -> List[Tuple[str, float, Optional[str]]]:
        """Extract (param, value, unit) triples from corrected text."""
        return extract_relations(text)
