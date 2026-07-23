# PipeSight AI

Automated instrument tag and symbol extraction for P&ID (Piping & Instrumentation Diagram) engineering drawings — turning unstructured PDF drawings into a structured, auditable, human-verified database.

Developed as part of a Schneider Electric internship project.

---

## What this is

Engineering drawings encode critical plant information — instrument tags, equipment identifiers, symbol relationships — that today is transcribed by hand. PipeSight AI reads P&ID PDFs directly, detects symbols and reads instrument tags off the page, and produces a structured database with full audit traceability, so every extracted fact can be traced back to the exact page, coordinates, and confidence it came from.

The pipeline is verification-first by design: nothing is silently accepted. Every extracted tag is either auto-confirmed against a calibrated confidence threshold or routed to a human reviewer, and every review action — accept, correct, dismiss, promote — is logged.

## Architecture

The pipeline runs in five stages:

1. **Input processing** — PDF pages are rendered to high-resolution images (300 DPI, via PyMuPDF).
2. **Symbol detection** — a YOLO-family object detector locates instrument and equipment symbols. Because P&ID pages run far larger than a standard model's input size, detection runs as a dual-scale tiled strategy (small tiles for individual symbols, large tiles for equipment context), merged and de-duplicated across both passes.
3. **OCR and tag extraction** — instrument-class detections are read via OCR in a layered, cost-proportional strategy (cheap zone-level reads first, escalating to whole-page OCR and a rescue pass only where needed), then normalized into ISA tag format. Every tag receives a calibrated confidence score.
4. **Database and human review** — extracted data is written to a relational database with full audit history. A live review interface lets an engineer accept, correct, dismiss, or promote any tag or candidate; every action re-generates the affected page's structured export immediately.
5. **Structured output** — an Excel workbook and an annotated overlay PDF, both generated from the same canonical per-page JSON export that the dashboard and review interface also read from — a single source of truth, not three independent exports.

## Repository structure

```
tag_extractor_v2.py       Core OCR + tag-normalization pipeline (Phase 3)
matching_engine.py        Spatial tag-symbol matching, instrument-type inference, noise filtering
rescue_to_db.py           Rescue pass — recovers symbols missed at production confidence
rescue_low_conf.py        Low-confidence review-queue rescue logic
db_builder.py             Database schema and connection/insert helpers (SQLite)
page_json.py              Canonical per-page JSON export (single source of truth)
page_summary.py           Human-readable companion export, grouped by ISA loop
review_server.py          Live review API/backend (Flask)
templates/review.html     Review interface frontend
human_review_export.py    Parallel spreadsheet-based review workflow (writes to the same DB)
dashboard.py              Operational dashboard — status, metrics, and embedded review (Streamlit)
pdf_excel_export.py       Excel + annotated-PDF export
pdf_to_image.py           Single-PDF to image conversion (Phase 1)
batch_pdf_to_image.py     Batch wrapper — converts every PDF in a folder
config.toml               Streamlit configuration
```

Note: the symbol-detection script (Phase 2, YOLO inference) is maintained separately from this repository and is not included here.

## Requirements

- Python 3.10+
- A YOLO-family model checkpoint trained on the project's instrument/equipment symbol classes (not included in this repository — see below)

Python packages, confirmed from source:

```
flask
streamlit
pandas
plotly
Pillow
openpyxl
paddleocr
```

Additionally required to run symbol detection (used by the detection script, maintained separately):

```
ultralytics
opencv-python
PyYAML
PyMuPDF
```

## What you need to provide

This repository ships code only — no data, no trained weights, no PDFs. To run the pipeline, you'll need to supply:

1. **Source P&ID PDFs**, placed under a local `data/` folder (git-ignored; organize as one subfolder per drawing set — `batch_pdf_to_image.py` expects this layout).
2. **A trained YOLO model checkpoint** for symbol detection, placed under `models/`. This repo doesn't include one.
3. **A class-name file** (`dataset.yaml`) listing the model's symbol classes in order, matching your checkpoint. Point the detection script at your copy via the `PIPESIGHT_DATASET_YAML` environment variable, or edit its path directly.
4. **PaddleOCR's model weights**, which PaddleOCR downloads automatically on first run.

## Running the pipeline

```bash
# 1. Convert PDFs to page images
python batch_pdf_to_image.py --input_dir data\your_pdfs --output_dir data\your_images

# 2. Run symbol detection (maintained separately — produces a detections JSON per page)

# 3. Extract and normalize tags, write to the database
python tag_extractor_v2.py

# 4. Generate structured exports
python page_json.py
python page_summary.py

# 5. Review — either interface reads/writes the same database
streamlit run dashboard.py          # opens the dashboard; review launches inline per page
# or, for offline/bulk review:
python human_review_export.py

# 6. Export deliverables
python pdf_excel_export.py
```

## Known limitations

- Symbol detection has a documented recall gap on dense equipment clusters — some correctly-OCR'd tags currently have no detected symbol nearby. Actively being investigated, not treated as a fixed ceiling.
- The current confidence-scoring formula's stated weighting (`0.4 × detection confidence + 0.6 × OCR confidence`) is accurate on paper but collapses in practice to being driven almost entirely by detection confidence, since OCR confidence saturates near its maximum on this document type regardless of read quality. The pipeline's actual acceptance threshold accounts for this; documentation describing the formula in isolation can be misleading without that context.
- ISA tag normalization is currently rule-based (regex + pattern matching). A move to a verified, LLM-assisted normalization step is planned — see Roadmap.

## Roadmap

- **Detector upgrade**: migrating symbol detection from the current YOLO generation to a newer YOLO release, targeting the dense-cluster recall gap above alongside general accuracy and CPU inference speed improvements.
- **LLM-assisted tag normalization**: introducing a local LLM to run alongside the existing regex normalization (not replacing it outright), to catch confidently-wrong regex matches that currently pass through unflagged — with any LLM-proposed correction required to be anchored to the actual OCR'd text before acceptance, never accepted on its own.

---

*Internship project — Schneider Electric.*
