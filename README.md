# TRACE

**Traffic Regulation & Analytics for Continuous Enforcement**

> A traffic camera sees everything. A human reviewer sees a fraction. **TRACE** closes that gap — turning raw frames into searchable, annotated, court-ready violation records in seconds, not hours.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)]()
[![Tests](https://img.shields.io/badge/tests-60%20passing-brightgreen)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-green)]()

**Repository:** [github.com/masoomprakhar/TRACE](https://github.com/masoomprakhar/TRACE)

---

## At a glance

Imagine a control room where every uploaded frame is preprocessed, scanned for seven violation types, linked to a license plate, and stored with annotated evidence — before an officer finishes their coffee. That is the workflow TRACE is built for.

| | |
|---|---|
| **Input** | Single image, batch upload, or CCTV frame |
| **Output** | Violations + confidence, plate text, evidence image, analytics |
| **Latency** | ~2.4 s/frame on CPU (full pipeline) |
| **Trust model** | No fabricated violations when a model is unavailable |

```
upload / CCTV frame  →  preprocess  →  detect + track  →  7 violation modules
                    →  plate OCR  →  evidence image  →  SQLite  →  dashboard
```

---

## The problem we set out to solve

Indian cities are adding cameras faster than they can staff review desks. Officers scroll through thousands of frames to catch helmet violations, red-light jumps, and plate numbers — often reaching different conclusions on the same image. Enforcement scales with headcount, not with data.

**Flipkart Grid** asked for a system that could automate photo identification and classification for traffic violations using computer vision. TRACE is our answer: an end-to-end pipeline that does not just detect objects, but **reasons about violations** and **produces evidence an auditor can trust**.

---

## What TRACE does

TRACE ingests a traffic image, understands the scene, and returns actionable output:

1. **Preprocesses** for low light, blur, haze, and shadow  
2. **Detects** vehicles, riders, plates, and signals  
3. **Flags seven violation types** with per-violation confidence  
4. **Reads Indian license plates** (Roboflow char OCR, TrOCR, or EasyOCR + format correction)  
5. **Produces annotated evidence** with timestamps and searchable metadata  
6. **Surfaces analytics** in a control-room dashboard and REST API  

When a model weight is missing in the field, the pipeline **degrades honestly** — it reports what it can verify instead of guessing.

---

## Why the design choices matter

| Real-world friction | How TRACE handles it |
|---------------------|----------------------|
| Variable image quality | Adaptive preprocessing router (CLAHE, dehaze, deblur) |
| Violations that need time (parking, red-light) | Multi-frame tracking with confirm-frames |
| Indian plate formats | Domain OCR corrector (`0↔O`, `1↔I`, state codes) |
| Models that drift or fail in production | Roboflow-hosted inference + local retraining pipelines |
| Judges and auditors who need proof | Confidence scores, evidence images, CSV export, eval harness |

---

## Numbers that back the story

We report **only what we can defend.** Evaluated with `scripts/run_full_eval.py`
on a labeled holdout; full provenance and methodology in `data/eval/README.md`.

| Metric | Score | Basis |
|--------|------:|-------|
| License-plate detection mAP@0.5 | **0.86** | 50-image plate holdout (GT boxes) |
| License-plate detection mAP@0.5:0.95 | **0.64** | same holdout |
| Helmet (no-helmet) F1 | **1.00** | small set — n=4 |
| Violation rule logic | **17/17 tests** | `tests/test_violation_rules.py` |
| End-to-end latency (CPU) | ~2.4 s/frame | hosted-API full pipeline |

**Honesty note.** Vehicle/road-user mAP, cross-violation F1, and OCR accuracy
are *not* headlined: the current holdout lacks full per-image detection ground
truth and balanced labels for 6 of 7 violation types. We fixed a metric bug
that had inflated detection mAP to 0.83 (it scored classes *absent from both
GT and predictions* a free 1.0), collapsed 11 conflicting report files into one,
and documented the path to a complete eval set. The seven violation rules are
verified deterministically in tests instead.

Full report: `data/eval/REPORT.txt` · Live in dashboard: **Settings → Performance**

---

## See it in five minutes

```bash
git clone https://github.com/masoomprakhar/TRACE.git
cd TRACE
pip install -r requirements.txt -r requirements-ml.txt
pip install -e .

# Option A — zero-key local mode (no account needed):
#   YOLO weights auto-download; OCR uses local EasyOCR.
export TRACE_CONFIG=config/default.yaml

# Option B — hosted models (best accuracy):
#   cp .env.example .env && add ROBOFLOW_API_KEY
#   export TRACE_CONFIG=config/roboflow.yaml

./scripts/judge_demo.sh         # seeds DB + starts server
```

Open **http://127.0.0.1:8000/#overview**

| What to show | Where |
|--------------|--------|
| KPIs & live feed | Overview |
| Upload & analyze | Evidence Center |
| Violation queue + proof | Violations |
| Plate lookup | ANPR Search |
| mAP / F1 metrics | Settings → Performance |
| Export for review | Reports → CSV |

---

## Full violation coverage

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

## Under the hood

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

## Retrain on your data

```bash
export ROBOFLOW_API_KEY=...
python scripts/prepare_all_datasets.py    # Roboflow downloads + OCR labels
python scripts/train_all.py --device cuda --epochs 30
python scripts/run_full_eval.py --config config/roboflow.yaml
```

Registry: `config/roboflow_models.yaml` · Quick iteration: `scripts/run_quick_train.py`

---

## API surface

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

## Deploy

```bash
docker compose up --build    # http://localhost:8000
```

For production demos with GPU weights, mount `models/weights/` or use Roboflow-only config.

---

## Quality bar

```bash
pip install -r requirements.txt -r requirements-dev.txt
pip install -e .
pytest -q     # 60 tests, no GPU / API key / model weights required
```

The suite runs on the lightweight stack alone — metrics, geometry, all seven
violation rules, OCR correction, storage, and API — and is enforced in CI
(`.github/workflows/ci.yml`) on Python 3.10–3.12.

---

## Closing thought

TRACE was built for **Flipkart Grid** — *Automated Photo Identification and Classification for Traffic Violations Using Computer Vision*.

The goal is not to replace human judgment, but to **give enforcement teams their time back**: standardize decisions, scale from one upload to multi-camera analytics, and leave behind evidence that holds up under scrutiny.

**See the violation. Trust the evidence. Enforce at scale.**

---

## License

MIT · Model weights and `.env` are not committed; see `.env.example`.
