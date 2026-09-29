# AI Document Intelligence

AI Document Intelligence is an experimental project focused on extracting structured information from documents, starting with retail receipts.

## 1) Project Objective

Build a scalable document understanding pipeline that evolves from a simple OCR + rule-based baseline into NLP-driven and layout-aware Transformer approaches for robust information extraction.

## 2) Current Dataset

The project currently targets the **SROIE (Scanned Receipts OCR and Information Extraction)** dataset as the first benchmark for experimentation and iteration.

The raw SROIE dataset is not included in this repository. Place the downloaded files locally under `data/raw/SROIE/`, with one `.jpg` image and matching JSON `.txt` annotation per receipt:

```text
data/raw/SROIE/
├── X00016469612.jpg
├── X00016469612.txt
├── X00016469619.jpg
└── X00016469619.txt
```

The dataset loader validates image/annotation pairs and exposes the annotations as a pandas DataFrame. A Tesseract OCR baseline is available; field extraction and model training are not implemented yet.

## 3) OCR Baseline

The OCR baseline uses Tesseract through `pytesseract`. Install the Python
dependencies with `pip install -r requirements.txt`, then install the Tesseract
executable separately:

- Windows: install Tesseract and set `pytesseract.pytesseract.tesseract_cmd`
	to the executable path when it is not on `PATH`.
- macOS: `brew install tesseract`
- Ubuntu/Debian: `sudo apt-get install tesseract-ocr`

Run OCR on one image:

```python
from src.ocr import TesseractOCREngine

result = TesseractOCREngine().recognize_path("path/to/receipt.jpg")
print(result.status)
print(result.text)
```

Run OCR over the valid canonical SROIE records:

```python
from src.ocr import run_sroie_ocr

summary = run_sroie_ocr(
		"data/raw/0325updated.task2train(626p)-20260928T174749Z-1-001",
		"data/processed/sroie_ocr.jsonl",
)
print(summary)
```

Results are stored as one JSON object per line under `data/processed/`. Each
record contains the canonical `image_id`, recognized text, words, bounding
boxes, confidence values, and an observable `success`, `empty`, or `failure`
status. The baseline does not extract receipt fields, correct OCR output, or
guarantee reliable recognition on poor-quality images.

## 4) Planned Development Phases

1. **Foundation setup**: project scaffolding, data organization, reproducible environment.
2. **Baseline**: OCR + rule-based field extraction.
3. **Error analysis**: systematic failure categorization and metric tracking.
4. **Modeling**: NLP and layout-aware Transformer models for document understanding.
5. **Serving & UX**: API and Streamlit interface for inference workflows.

## 5) Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or install via project metadata:

```bash
pip install .
```

## 6) Project Structure

```text
ai-document-intelligence/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_error_analysis.ipynb
├── src/
│   ├── data/
│   ├── preprocessing/
│   ├── ocr/
│   ├── models/
│   ├── evaluation/
│   └── inference/
├── api/
│   └── main.py
├── app/
│   └── app.py
├── configs/
├── tests/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
└── Dockerfile
```

## 7) Disclaimer

This repository is **experimental** and intended for **research/portfolio purposes**. It is not production-ready at this stage.

---

> Note: Dataset download, model training, OCR implementation, API implementation, and Streamlit app implementation are intentionally out of scope for this initial scaffold.
