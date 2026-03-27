"""Unit tests for the NLM processor."""
import numpy as np
import pytest
from PIL import Image

from ocr_model.nlm_processor import (
    nlm_denoise,
    extract_relations,
    SpellingCorrector,
)


def _make_noisy_gray(w: int = 100, h: int = 100) -> Image.Image:
    rng = np.random.default_rng(42)
    arr = rng.integers(0, 256, (h, w), dtype=np.uint8)
    return Image.fromarray(arr, mode="L")


class TestNLMDenoise:
    def test_returns_pil_image(self):
        img = _make_noisy_gray()
        out = nlm_denoise(img)
        assert isinstance(out, Image.Image)

    def test_same_size(self):
        img = _make_noisy_gray(80, 60)
        out = nlm_denoise(img)
        assert out.size == (80, 60)

    def test_different_h_changes_output(self):
        img = _make_noisy_gray()
        out1 = np.array(nlm_denoise(img, h=1.0))
        out2 = np.array(nlm_denoise(img, h=30.0))
        assert not np.array_equal(out1, out2)


class TestExtractRelations:
    def test_hemoglobin_extraction(self):
        text = "Hemoglobin 14 g/dL"
        triples = extract_relations(text)
        params = {t[0]: t for t in triples}
        assert "hemoglobin" in params
        assert params["hemoglobin"][1] == 14.0

    def test_wbc_extraction(self):
        text = "WBC 15000 cells/µL"
        triples = extract_relations(text)
        params = {t[0] for t in triples}
        assert "wbc" in params

    def test_platelets_extraction(self):
        text = "Platelets: 250000"
        triples = extract_relations(text)
        params = {t[0] for t in triples}
        assert "platelets" in params

    def test_unit_captured(self):
        text = "glucose 95 mg/dl"
        triples = extract_relations(text)
        params = {t[0]: t for t in triples}
        assert "glucose" in params
        assert params["glucose"][2] is not None

    def test_unknown_param_ignored(self):
        text = "foobar 999"
        triples = extract_relations(text)
        assert triples == []

    def test_no_duplicate_params(self):
        text = "hemoglobin 14 g/dL\nhemoglobin 14 g/dL"
        triples = extract_relations(text)
        param_list = [t[0] for t in triples]
        assert param_list.count("hemoglobin") == 1

    def test_multiline_report(self):
        text = (
            "Patient: John Doe\n"
            "Hemoglobin: 14 g/dL\n"
            "WBC: 15000 cells/uL\n"
            "Platelets: 250000\n"
            "Glucose: 95 mg/dL\n"
        )
        triples = extract_relations(text)
        params = {t[0] for t in triples}
        assert {"hemoglobin", "wbc", "platelets", "glucose"}.issubset(params)


class TestSpellingCorrector:
    def test_basic_correction(self):
        corrector = SpellingCorrector()
        corrected = corrector.correct("teh quick brwon fox")
        # 'teh' → 'the', 'brwon' → 'brown'
        assert "teh" not in corrected.lower()

    def test_medical_terms_preserved(self):
        corrector = SpellingCorrector()
        text = "hemoglobin wbc platelets hematocrit"
        corrected = corrector.correct(text)
        for term in ["hemoglobin", "wbc", "platelets", "hematocrit"]:
            assert term in corrected.lower()

    def test_numbers_preserved(self):
        corrector = SpellingCorrector()
        text = "value: 14.5 g/dL"
        corrected = corrector.correct(text)
        assert "14.5" in corrected

    def test_empty_string(self):
        corrector = SpellingCorrector()
        assert corrector.correct("") == ""
