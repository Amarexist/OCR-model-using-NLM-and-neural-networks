# OCR Model using NLM and Neural Networks

A powerful end-to-end OCR pipeline that extracts structured parameters from
medical reports (JPEG, PNG, PDF) using:

- **Non-Local Means (NLM) image denoising** — cleans noisy scans before OCR.
- **Self-attention embedding** — DistilBERT contextualises the extracted text.
- **Spelling correction** — pyspellchecker repairs OCR typos while preserving
  medical terminology.
- **Neural parameter extractor** — regex + MLP confidence scorer surfaces
  `(parameter, value, unit)` triples.
- **Medical classifier** — maps each extracted value to a clinical status
  (`normal` / `low` / `high`) using built-in reference ranges.
- **Disease inference** — rule-based engine flags potential conditions
  (Anemia, Leukocytosis, Thrombocytopenia, …) from abnormal values.

## Example output

```json
{
  "parameters": {
    "hemoglobin": { "value": 14.0, "unit": "g/dL",      "status": "normal" },
    "wbc":        { "value": 15000.0, "unit": "cells/µL","status": "high"   },
    "platelets":  { "value": 250000.0,"unit": "cells/µL","status": "normal" }
  },
  "diseases": ["Leukocytosis (Infection / Inflammation)"]
}
```

## Project structure

```
ocr_model/
├── __init__.py           # Public API — exposes OCRModel
├── preprocessor.py       # Image / PDF loading, CLAHE, deskew, binarise
├── ocr_engine.py         # Tesseract / EasyOCR wrapper
├── nlm_processor.py      # NLM denoising, self-attention embedding,
│                         # spelling correction, relation extraction
├── neural_extractor.py   # MLP confidence scorer + candidate extractor
├── medical_classifier.py # Reference ranges, status classification,
│                         # disease inference
└── model.py              # OCRModel orchestrator
main.py                   # CLI entry point
tests/                    # pytest test suite (58 tests)
requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
# Tesseract must also be installed at the OS level:
# Ubuntu/Debian: sudo apt install tesseract-ocr
# macOS:         brew install tesseract
```

## Usage

### Python API

```python
from ocr_model import OCRModel
import json

model = OCRModel()                        # uses Tesseract + DistilBERT
result = model.process("report.jpg")     # or .pdf / .png
print(json.dumps(result, indent=2))
```

Pass `use_embeddings=False` for fast, offline inference (no GPU / transformers
required):

```python
model = OCRModel(use_embeddings=False)
result = model.process_text(raw_ocr_string)   # skip image loading
```

### Command-line

```bash
python main.py report.jpg
python main.py report.pdf --backend easyocr --output result.json
python main.py report.png --no-embeddings
```

## Running tests

```bash
pytest tests/ -v
```

## Architecture

```
Input (JPEG / PNG / PDF)
        │
        ▼
┌─────────────────┐
│  Preprocessor   │  load_pages → CLAHE → deskew → binarise
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  NLM Denoiser   │  fastNlMeansDenoising (OpenCV)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   OCR Engine    │  Tesseract (psm 6) or EasyOCR
└────────┬────────┘
         │  raw text
         ▼
┌─────────────────────────────────────┐
│          NLM Processor              │
│  ┌──────────────────────────────┐   │
│  │  Spelling Corrector          │   │  pyspellchecker
│  └──────────────────────────────┘   │
│  ┌──────────────────────────────┐   │
│  │  Self-Attention Embedder     │   │  DistilBERT (HuggingFace)
│  └──────────────────────────────┘   │
│  ┌──────────────────────────────┐   │
│  │  Relation Extractor          │   │  regex → (param, value, unit)
│  └──────────────────────────────┘   │
└────────────────┬────────────────────┘
                 │  candidates + embeddings
                 ▼
┌─────────────────────────────────────┐
│        Neural Extractor             │
│   MLP Confidence Scorer (PyTorch)   │
└────────────────┬────────────────────┘
                 │  scored candidates
                 ▼
┌─────────────────────────────────────┐
│       Medical Classifier            │
│  Reference ranges → normal/low/high │
│  Disease inference rules            │
└────────────────┬────────────────────┘
                 │
                 ▼
        Structured JSON output
```
