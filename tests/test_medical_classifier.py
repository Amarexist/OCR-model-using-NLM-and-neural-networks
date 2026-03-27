"""Unit tests for medical classifier."""
import pytest
from ocr_model.medical_classifier import (
    classify_value,
    infer_diseases,
    get_reference_range,
    ReferenceRange,
)


class TestClassifyValue:
    def test_normal_hemoglobin(self):
        status, ref = classify_value("hemoglobin", 15.0)
        assert status == "normal"
        assert ref is not None

    def test_low_hemoglobin(self):
        status, _ = classify_value("hemoglobin", 8.0)
        assert status == "low"

    def test_high_wbc(self):
        status, _ = classify_value("wbc", 15000)
        assert status == "high"

    def test_normal_wbc(self):
        status, _ = classify_value("wbc", 7000)
        assert status == "normal"

    def test_normal_platelets(self):
        status, _ = classify_value("platelets", 250000)
        assert status == "normal"

    def test_low_platelets(self):
        status, _ = classify_value("platelets", 100000)
        assert status == "low"

    def test_high_platelets(self):
        status, _ = classify_value("platelets", 500000)
        assert status == "high"

    def test_unknown_param(self):
        status, ref = classify_value("unknown_param_xyz", 42.0)
        assert status == "unknown"
        assert ref is None

    def test_high_glucose(self):
        status, _ = classify_value("glucose", 250.0)
        assert status == "high"

    def test_normal_glucose(self):
        status, _ = classify_value("glucose", 90.0)
        assert status == "normal"

    def test_no_lower_bound(self):
        # cholesterol has no lower bound — any value above 200 is high
        status, _ = classify_value("cholesterol", 250.0)
        assert status == "high"

    def test_no_lower_bound_normal(self):
        status, _ = classify_value("cholesterol", 150.0)
        assert status == "normal"


class TestInferDiseases:
    def test_high_wbc_infection(self):
        statuses = {"wbc": "high"}
        diseases = infer_diseases(statuses)
        assert any("Leukocytosis" in d for d in diseases)

    def test_low_hemoglobin_anemia(self):
        diseases = infer_diseases({"hemoglobin": "low"})
        assert "Anemia" in diseases

    def test_low_platelets_thrombocytopenia(self):
        diseases = infer_diseases({"platelets": "low"})
        assert "Thrombocytopenia" in diseases

    def test_multiple_conditions(self):
        statuses = {"hemoglobin": "low", "wbc": "high"}
        diseases = infer_diseases(statuses)
        assert "Anemia" in diseases
        assert any("Leukocytosis" in d for d in diseases)

    def test_no_diseases_for_normal(self):
        statuses = {"hemoglobin": "normal", "wbc": "normal", "platelets": "normal"}
        diseases = infer_diseases(statuses)
        assert diseases == []

    def test_empty_statuses(self):
        assert infer_diseases({}) == []


class TestGetReferenceRange:
    def test_known_param(self):
        ref = get_reference_range("hemoglobin")
        assert isinstance(ref, ReferenceRange)
        assert ref.unit == "g/dL"

    def test_unknown_param(self):
        assert get_reference_range("unicorn") is None
