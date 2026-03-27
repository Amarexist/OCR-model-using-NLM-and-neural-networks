"""Medical reference ranges and status classification.

Each entry stores:
- ``low``   — lower bound of the normal range (or ``None``).
- ``high``  — upper bound of the normal range (or ``None``).
- ``unit``  — canonical SI / conventional unit.

Status rules:
- ``"low"``    — value < low
- ``"normal"`` — low <= value <= high
- ``"high"``   — value > high
- ``"unknown"``— parameter not in the reference table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class ReferenceRange:
    low: Optional[float]
    high: Optional[float]
    unit: str


# Reference ranges for adult males (general guidelines).
# These values are approximate; actual ranges vary by laboratory.
_REFERENCE_RANGES: Dict[str, ReferenceRange] = {
    # Haematology
    "hemoglobin":   ReferenceRange(13.5, 17.5, "g/dL"),
    "hematocrit":   ReferenceRange(41.0, 53.0, "%"),
    "rbc":          ReferenceRange(4.5,  5.9,  "10^6/µL"),
    "wbc":          ReferenceRange(4000, 11000, "cells/µL"),
    "platelets":    ReferenceRange(150000, 400000, "cells/µL"),
    "mcv":          ReferenceRange(80.0, 100.0, "fL"),
    "mch":          ReferenceRange(27.0, 33.0,  "pg"),
    "mchc":         ReferenceRange(32.0, 36.0,  "g/dL"),
    "rdw":          ReferenceRange(11.5, 14.5,  "%"),
    "mpv":          ReferenceRange(7.5,  12.5,  "fL"),
    "neutrophils":  ReferenceRange(1800, 7700,  "cells/µL"),
    "lymphocytes":  ReferenceRange(1000, 4800,  "cells/µL"),
    "monocytes":    ReferenceRange(200,  1000,  "cells/µL"),
    "eosinophils":  ReferenceRange(100,  500,   "cells/µL"),
    "basophils":    ReferenceRange(0,    100,   "cells/µL"),
    # Chemistry
    "glucose":      ReferenceRange(70.0, 100.0, "mg/dL"),
    "creatinine":   ReferenceRange(0.7,  1.3,   "mg/dL"),
    "urea":         ReferenceRange(7.0,  20.0,  "mg/dL"),
    "sodium":       ReferenceRange(136,  145,   "mEq/L"),
    "potassium":    ReferenceRange(3.5,  5.0,   "mEq/L"),
    "chloride":     ReferenceRange(98,   106,   "mEq/L"),
    "calcium":      ReferenceRange(8.5,  10.5,  "mg/dL"),
    "cholesterol":  ReferenceRange(None, 200.0, "mg/dL"),
    "triglycerides":ReferenceRange(None, 150.0, "mg/dL"),
    "hdl":          ReferenceRange(40.0, None,  "mg/dL"),
    "ldl":          ReferenceRange(None, 100.0, "mg/dL"),
    "ast":          ReferenceRange(10,   40,    "U/L"),
    "alt":          ReferenceRange(7,    56,    "U/L"),
    "alp":          ReferenceRange(44,   147,   "U/L"),
    "ggt":          ReferenceRange(8,    61,    "U/L"),
    "bilirubin":    ReferenceRange(0.2,  1.2,   "mg/dL"),
    "albumin":      ReferenceRange(3.5,  5.0,   "g/dL"),
    "uric_acid":    ReferenceRange(3.5,  7.2,   "mg/dL"),
    # Hormones / other
    "tsh":          ReferenceRange(0.4,  4.0,   "mIU/mL"),
    "t3":           ReferenceRange(80,   200,   "ng/dL"),
    "t4":           ReferenceRange(5.0,  12.0,  "µg/dL"),
    "hba1c":        ReferenceRange(None, 5.7,   "%"),
    "crp":          ReferenceRange(None, 1.0,   "mg/L"),
    "esr":          ReferenceRange(0,    20,    "mm/hr"),
    "ferritin":     ReferenceRange(12,   300,   "ng/mL"),
    # Coagulation
    "pt":           ReferenceRange(11.0, 13.5,  "seconds"),
    "inr":          ReferenceRange(0.8,  1.1,   "ratio"),
    "aptt":         ReferenceRange(25.0, 35.0,  "seconds"),
    "d_dimer":      ReferenceRange(None, 0.5,   "µg/mL"),
}


def classify_value(
    param: str, value: float
) -> Tuple[str, Optional[ReferenceRange]]:
    """Classify a numeric *value* for *param* as ``"low"`` / ``"normal"`` / ``"high"`` / ``"unknown"``.

    Parameters
    ----------
    param:
        Canonical parameter name (must match a key in ``_REFERENCE_RANGES``).
    value:
        Numeric value to classify.

    Returns
    -------
    (status, reference_range)
        ``reference_range`` is ``None`` when the parameter is not in the table.
    """
    ref = _REFERENCE_RANGES.get(param)
    if ref is None:
        return "unknown", None

    if ref.low is not None and value < ref.low:
        return "low", ref
    if ref.high is not None and value > ref.high:
        return "high", ref
    return "normal", ref


def get_reference_range(param: str) -> Optional[ReferenceRange]:
    """Return the :class:`ReferenceRange` for *param*, or ``None``."""
    return _REFERENCE_RANGES.get(param)


# ---------------------------------------------------------------------------
# Disease inference
# ---------------------------------------------------------------------------

# Simple rule-based disease inference from CBC / chemistry abnormalities.
_DISEASE_RULES: Dict[str, Dict[str, str]] = {
    "Anemia":                {"hemoglobin": "low"},
    "Polycythemia":          {"hemoglobin": "high", "hematocrit": "high"},
    "Leukocytosis (Infection / Inflammation)": {"wbc": "high"},
    "Leukopenia":            {"wbc": "low"},
    "Thrombocytopenia":      {"platelets": "low"},
    "Thrombocytosis":        {"platelets": "high"},
    "Neutropenia":           {"neutrophils": "low"},
    "Eosinophilia (Allergy / Parasitic Infection)": {"eosinophils": "high"},
    "Hyperglycemia / Diabetes risk": {"glucose": "high"},
    "Hypoglycemia":          {"glucose": "low"},
    "Renal Impairment":      {"creatinine": "high", "urea": "high"},
    "Hypercholesterolemia":  {"cholesterol": "high"},
    "Hypertriglyceridemia":  {"triglycerides": "high"},
    "Liver Disease":         {"ast": "high", "alt": "high"},
    "Hypothyroidism":        {"tsh": "high"},
    "Hyperthyroidism":       {"tsh": "low"},
    "Iron Deficiency":       {"ferritin": "low"},
    "Hyperkalemia":          {"potassium": "high"},
    "Hypokalemia":           {"potassium": "low"},
    "Hypernatremia":         {"sodium": "high"},
    "Hyponatremia":          {"sodium": "low"},
}


def infer_diseases(statuses: Dict[str, str]) -> list:
    """Return a list of possible diseases inferred from abnormal *statuses*.

    Parameters
    ----------
    statuses:
        Mapping of canonical_param → status string
        (``"low"`` / ``"normal"`` / ``"high"``).

    Returns
    -------
    list of str
        Disease names whose diagnostic criteria are fully satisfied.
    """
    diseases = []
    for disease, criteria in _DISEASE_RULES.items():
        if all(statuses.get(p) == s for p, s in criteria.items()):
            diseases.append(disease)
    return diseases
