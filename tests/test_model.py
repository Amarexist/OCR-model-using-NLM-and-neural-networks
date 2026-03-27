"""Integration-style tests for OCRModel.process_text (no external OCR needed)."""
import pytest
from ocr_model.model import OCRModel


# A synthetic medical report text that mimics OCR output.
SAMPLE_REPORT = """
Patient: Jane Doe
Date: 2024-01-15

COMPLETE BLOOD COUNT (CBC)
Hemoglobin        : 14  g/dL
WBC               : 15000  cells/uL
Platelets         : 250000
RBC               : 5.1  10^6/uL
Hematocrit        : 45  %
MCV               : 88  fL
MCH               : 29  pg
MCHC              : 34  g/dL

CHEMISTRY PANEL
Glucose           : 95  mg/dL
Creatinine        : 1.0  mg/dL
Sodium            : 140  mEq/L
Potassium         : 4.2  mEq/L
Cholesterol       : 195  mg/dL
"""


@pytest.fixture(scope="module")
def model():
    """Create an OCRModel with embeddings disabled for fast tests."""
    return OCRModel(use_embeddings=False, denoise_h=0)


class TestProcessText:
    def test_returns_dict(self, model):
        result = model.process_text(SAMPLE_REPORT)
        assert isinstance(result, dict)

    def test_keys_present(self, model):
        result = model.process_text(SAMPLE_REPORT)
        assert "raw_text" in result
        assert "corrected_text" in result
        assert "parameters" in result
        assert "diseases" in result
        assert "pages" in result

    def test_hemoglobin_extracted(self, model):
        result = model.process_text(SAMPLE_REPORT)
        assert "hemoglobin" in result["parameters"]
        hgb = result["parameters"]["hemoglobin"]
        assert hgb["value"] == 14.0
        assert hgb["status"] == "normal"

    def test_wbc_extracted_and_high(self, model):
        result = model.process_text(SAMPLE_REPORT)
        assert "wbc" in result["parameters"]
        wbc = result["parameters"]["wbc"]
        assert wbc["value"] == 15000.0
        assert wbc["status"] == "high"

    def test_platelets_extracted(self, model):
        result = model.process_text(SAMPLE_REPORT)
        assert "platelets" in result["parameters"]
        plt = result["parameters"]["platelets"]
        assert plt["value"] == 250000.0
        assert plt["status"] == "normal"

    def test_diseases_list(self, model):
        result = model.process_text(SAMPLE_REPORT)
        assert isinstance(result["diseases"], list)

    def test_leukocytosis_detected(self, model):
        result = model.process_text(SAMPLE_REPORT)
        diseases_str = " ".join(result["diseases"])
        assert "Leukocytosis" in diseases_str

    def test_pages_count(self, model):
        result = model.process_text(SAMPLE_REPORT)
        assert result["pages"] == 1

    def test_raw_text_preserved(self, model):
        result = model.process_text(SAMPLE_REPORT)
        assert "Hemoglobin" in result["raw_text"]

    def test_normal_report_no_diseases(self, model):
        normal_text = (
            "Hemoglobin: 15 g/dL\n"
            "WBC: 7000 cells/uL\n"
            "Platelets: 250000\n"
            "Glucose: 90 mg/dL\n"
        )
        result = model.process_text(normal_text)
        assert result["diseases"] == []


class TestProcessToJson:
    def test_returns_string(self, model):
        import json
        result = model.process_text(SAMPLE_REPORT)
        json_str = json.dumps(result, indent=2)
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert "parameters" in parsed
