# AI Document Intelligence

AI Document Intelligence is an experimental project focused on extracting structured information from documents, starting with retail receipts.

## 1) Project Objective

Build a scalable document understanding pipeline that evolves from a simple OCR + rule-based baseline into NLP-driven and layout-aware Transformer approaches for robust information extraction.

## 2) Current Dataset

The project currently targets the **SROIE (Scanned Receipts OCR and Information Extraction)** dataset as the first benchmark for experimentation and iteration.

## 3) Planned Development Phases

1. **Foundation setup**: project scaffolding, data organization, reproducible environment.
2. **Baseline**: OCR + rule-based field extraction.
3. **Error analysis**: systematic failure categorization and metric tracking.
4. **Modeling**: NLP and layout-aware Transformer models for document understanding.
5. **Serving & UX**: API and Streamlit interface for inference workflows.

## 4) Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or install via project metadata:

```bash
pip install .
```

## 5) Project Structure

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

## 6) Disclaimer

This repository is **experimental** and intended for **research/portfolio purposes**. It is not production-ready at this stage.

---

> Note: Dataset download, model training, OCR implementation, API implementation, and Streamlit app implementation are intentionally out of scope for this initial scaffold.
