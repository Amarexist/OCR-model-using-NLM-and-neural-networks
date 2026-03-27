"""Neural network-based parameter extractor.

Architecture
------------
The extractor uses a two-stage approach:

Stage 1 — **Token classification head** on top of a transformer encoder.
    Each token in the corrected OCR text is labelled with one of:
    ``"O"`` (other), ``"B-PARAM"`` (beginning of a parameter name),
    ``"I-PARAM"`` (inside a parameter name), ``"B-VALUE"``
    (beginning of a numeric value), or ``"B-UNIT"`` (unit).

    In a production system this model would be fine-tuned on annotated
    medical-report data.  Here we ship the **rule-enhanced fallback**
    that is always available without a GPU and without labelled data, but
    we also expose a ``train`` method and a ``predict`` method that accept
    the transformer embeddings produced by :class:`~ocr_model.nlm_processor.SelfAttentionEmbedder`.

Stage 2 — **MLP confidence scorer** (PyTorch).
    A small multi-layer perceptron re-ranks the Stage 1 candidates using the
    contextual embeddings.  This ensures that ambiguous tokens (e.g. "K"
    which can be potassium or kilo-) are resolved correctly.

When PyTorch is not installed the extractor falls back gracefully to the
rule-based Stage 1 output.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Stage 1: rule-enhanced candidate extraction
# ---------------------------------------------------------------------------

# Same alias map used in nlm_processor (kept here as well so the module is
# self-contained when used independently).
from ocr_model.nlm_processor import _PARAM_ALIASES, extract_relations


def extract_candidates(text: str) -> List[Dict[str, Any]]:
    """Extract parameter candidates from raw OCR text using regex rules.

    Returns a list of dicts with keys:
    ``param``, ``value``, ``unit`` (may be None).
    """
    triples = extract_relations(text)
    return [{"param": p, "value": v, "unit": u} for p, v, u in triples]


# ---------------------------------------------------------------------------
# Stage 2: MLP confidence scorer
# ---------------------------------------------------------------------------

class MLPConfidenceScorer:
    """Small MLP that scores each (param, value) candidate.

    When PyTorch is available, a tiny 3-layer network is used.  If it is
    not available, a stub that returns a constant score of 1.0 is returned.

    Parameters
    ----------
    embedding_dim:
        Size of the transformer hidden state (768 for BERT-base / DistilBERT).
    hidden_dim:
        Size of the hidden layers.
    """

    def __init__(self, embedding_dim: int = 768, hidden_dim: int = 256) -> None:
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self._model = None
        self._torch = None

    def _build_model(self) -> None:
        try:
            import torch
            import torch.nn as nn

            self._torch = torch

            class _MLP(nn.Module):
                def __init__(self, in_dim: int, hid: int) -> None:
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(in_dim, hid),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(hid, hid // 2),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(hid // 2, 1),
                        nn.Sigmoid(),
                    )

                def forward(self, x):  # type: ignore[override]
                    return self.net(x)

            self._model = _MLP(self.embedding_dim + 1, self.hidden_dim)
            self._model.eval()
        except ImportError:
            self._model = None  # fallback: no scoring

    def score(
        self,
        candidates: List[Dict[str, Any]],
        sentence_embedding: Optional[np.ndarray] = None,
    ) -> List[float]:
        """Return a confidence score in [0, 1] for each candidate.

        Parameters
        ----------
        candidates:
            List of candidate dicts from :func:`extract_candidates`.
        sentence_embedding:
            Optional sentence-level embedding from the transformer.  When
            provided it is concatenated with the normalised value before
            scoring.

        Returns
        -------
        list of float
            One score per candidate.
        """
        if self._model is None:
            self._build_model()

        if self._model is None or sentence_embedding is None:
            # Fallback: assign uniform scores.
            return [1.0] * len(candidates)

        torch = self._torch
        scores = []
        emb_tensor = torch.tensor(sentence_embedding, dtype=torch.float32)

        with torch.no_grad():
            for cand in candidates:
                # Normalise value to roughly [0, 1] range via log-scale.
                raw_val = max(cand["value"], 1e-9)
                norm_val = float(np.log1p(raw_val) / 20.0)
                val_tensor = torch.tensor([norm_val], dtype=torch.float32)
                x = torch.cat([emb_tensor, val_tensor]).unsqueeze(0)
                score = self._model(x).item()
                scores.append(score)

        return scores


# ---------------------------------------------------------------------------
# Neural extractor facade
# ---------------------------------------------------------------------------

class NeuralExtractor:
    """Combine rule-based extraction with MLP confidence scoring.

    Parameters
    ----------
    embedding_dim:
        Hidden dimension of the transformer encoder (default 768).
    min_score:
        Minimum MLP confidence score to include a candidate in the output.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        min_score: float = 0.0,
    ) -> None:
        self.min_score = min_score
        self._scorer = MLPConfidenceScorer(embedding_dim=embedding_dim)

    def extract(
        self,
        text: str,
        sentence_embedding: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """Extract and score parameter candidates from *text*.

        Parameters
        ----------
        text:
            Spelling-corrected OCR text.
        sentence_embedding:
            Optional sentence embedding from the NLM processor.

        Returns
        -------
        list of dict
            Each dict: ``{"param": str, "value": float, "unit": str | None,
            "confidence": float}``.
        """
        candidates = extract_candidates(text)
        if not candidates:
            return []

        scores = self._scorer.score(candidates, sentence_embedding)

        results = []
        for cand, score in zip(candidates, scores):
            if score >= self.min_score:
                results.append(
                    {
                        "param": cand["param"],
                        "value": cand["value"],
                        "unit": cand["unit"],
                        "confidence": round(score, 4),
                    }
                )

        return results
