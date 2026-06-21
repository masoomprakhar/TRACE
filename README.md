# TRACE

**Traffic Regulation & Analytics for Continuous Enforcement**

> Every day, traffic cameras capture thousands of frames. Manual review is slow, inconsistent, and expensive. **TRACE** turns that photographic evidence into searchable, annotated violation records — automatically.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-green)]()

**Repository:** [github.com/masoomprakhar/TRACE](https://github.com/masoomprakhar/TRACE)

---

## The problem

Cities deploy cameras faster than they can review footage. Officers spend hours scrolling through images to catch helmet violations, red-light jumps, and plate numbers — often reaching different conclusions on the same frame. Enforcement scales with headcount, not with data.

## What TRACE does

TRACE is an end-to-end computer vision system that **ingests a traffic image** (or live frame), **understands the scene**, and **returns court-ready output**:

1. **Preprocesses** for low light, blur, haze, and shadow  
2. **Detects** vehicles, riders, plates, and signals  
3. **Flags seven violation types** with confidence scores  
4. **Reads Indian license plates** (OCR + format correction)  
5. **Produces annotated evidence** with timestamps and searchable metadata  
6. **Surfaces analytics** in a control-room dashboard and REST API  

No violation is fabricated when a model is missing — the pipeline degrades honestly.

```
upload / CCTV frame  →  preprocess  →  detect + track  →  7 violation modules
                    →  plate OCR  →  evidence image  →  SQLite  →  dashboard
```

---

## Why it matters

| Challenge | TRACE response |
|-----------|----------------|
| Variable image quality | Adaptive preprocessing router (CLAHE, dehaze, deblur) |
| Sequential violations (parking, red-light) | Multi-frame tracking + confirm-frames |
| Indian plate formats | Domain OCR corrector (`0↔O`, `1↔I`, state codes) |
| Model gaps in the field | Roboflow-hosted inference + local fine-tuning pipelines |
| Judge / auditor trust | Confidence scores, evidence images, CSV export, eval harness |

---

## Measured results

Evaluated on **63 labeled traffic frames** (`scripts/run_full_eval.py`):

| Metric | Score |
|--------|------:|
| Detection mAP@0.5 | **0.83** |
| Motorcycle AP@0.5 | **1.00** |
| No-helmet F1 | **1.00** |
| Violation micro-F1 | **0.89** |
| Plate detection mAP (Roboflow) | **0.86** |
| End-to-end latency (CPU) | ~2.4 s/frame |

Full report: `data/eval/REPORT-quick-train.txt` · Dashboard: **Settings → Performance**

---

## Live demo (5 minutes)

```bash
git clone https://github.com/masoomprakhar/TRACE.git
cd TRACE
pip install -r requirements.txt -r requirements-ml.txt
pip install -e .

cp .env.example .env          # add ROBOFLOW_API_KEY
export TRACE_CONFIG=config/roboflow.yaml

./scripts/judge_demo.sh         # seeds DB + starts server
```

Open **http://127.0.0.1:8000/#overview**

| Step | Where |
|------|--------|
| KPIs & live feed | Overview |
| Upload & analyze | Evidence Center |
| Violation queue + proof | Violations |
| Plate lookup | ANPR Search |
| mAP / F1 metrics | Settings → Performance |
| Export | Reports → CSV |

---

## Violation coverage

All seven types from the problem statement:

- Helmet non-compliance  
- Seatbelt non-compliance  
- Triple riding  
- Wrong-side driving  
- Stop-line violation  
- Red-light violation  
- Illegal parking  

Geometry-based rules use per-camera calibration (stop line, lane divider, signal ROI, no-parking zones) in `config/roboflow.yaml`.

---

## Stack

| Layer | Technology |
|-------|------------|
| Detection | YOLO11 (VioVision fine-tune) |
| Helmet / riders | Multi-label CNN + Roboflow workflows |
| Seatbelt | 3-class classifier (belt / no_belt / occluded) |
| Plates & OCR | Roboflow `general-segmentation-api-4` + `ocr-character-cgtzm/4`, TrOCR |
| API | FastAPI + SQLite |
| UI | Vanilla JS dashboard (Chart.js) |
| Training | `scripts/prepare_all_datasets.py` → `scripts/train_all.py` |

---

## Training & retraining

```bash
export ROBOFLOW_API_KEY=...
python scripts/prepare_all_datasets.py    # Roboflow downloads + OCR labels
python scripts/train_all.py --device cuda --epochs 30
python scripts/run_full_eval.py --config config/roboflow.yaml
```

Registry: `config/roboflow_models.yaml` · Quick iteration: `scripts/run_quick_train.py`

---

## API (excerpt)

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/health` | Model status |
| POST | `/api/analyze` | Analyze uploaded image |
| GET | `/api/violations` | Searchable violation list |
| GET | `/api/analytics/summary` | Trends, top plates, FPS |
| GET | `/api/eval/summary` | Offline mAP / F1 for dashboard |
| GET | `/api/violations.csv` | Export report |

Interactive docs: `/docs`

---

## Project layout

```
trace_cv/          pipeline, violations, OCR, evidence, evaluation
dashboard/         control-room UI
training/          YOLO, rider CNN, seatbelt, TrOCR trainers
scripts/           datasets, eval, judge_demo.sh, train_all.py
config/            default.yaml, roboflow.yaml, roboflow_models.yaml
data/eval/         manifest, reports, eval-summary.json
```

---

## Docker

```bash
docker compose up --build    # http://localhost:8000
```

For production demos with GPU weights, mount `models/weights/` or use Roboflow-only config.

---

## Testing

```bash
PYTHONPATH=$PWD pytest -q     # 41 tests
```

---

## Team & submission

Built for **Flipkart Grid** — *Automated Photo Identification and Classification for Traffic Violations Using Computer Vision*.

TRACE reduces manual review from hours to seconds per frame, standardizes enforcement, and scales from a single upload to multi-camera analytics — **see the violation, trust the evidence, enforce at scale.**

---

## License

MIT · Model weights and `.env` are not committed; see `.env.example`.
