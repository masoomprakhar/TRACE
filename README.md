# TRACE — Traffic Rule Analysis & Compliance Engine

Automated photo identification and classification of traffic violations using
computer vision. TRACE ingests traffic images (or video frames), enhances them
adaptively, detects road users, identifies and classifies **seven** violation
types with calibrated confidence, reads license plates, and produces
annotated, searchable, court-ready evidence — exposed through a REST API and a
control-room dashboard.

> Built for the *Automated Photo Identification and Classification for Traffic
> Violations* challenge. See the concept note in [`docs/`](docs/).

---

## Why TRACE stands out

1. **Quality-adaptive preprocessing** — each frame is *diagnosed* (blur,
   low-light, haze, contrast) and only the needed correction is applied, so the
   average frame stays fast while hard frames get rescued.
2. **Temporal reasoning** — a tracker gives persistent IDs, so inherently
   sequential violations (triple riding, parking duration, red-light + stop-line)
   are judged across frames, not guessed from one.
3. **Honest, pluggable models** — helmet/seatbelt classifiers are optional; when
   a model isn't loaded the module is skipped, never faked. Geometry-driven
   violations work out of the box on COCO weights.
4. **Domain-adapted Indian-plate OCR** — format + confusion-aware correction
   (`0↔O`, `1↔I`, `8↔B`, …) recovers plates generic OCR returns malformed.

---

## Architecture

```
image/frame
   → adaptive preprocessing  (trace_cv/preprocessing)
   → detection + tracking    (trace_cv/detection)      YOLO + IOU/ByteTrack
   → violation engine        (trace_cv/violation)      7 modules
   → license-plate OCR        (trace_cv/ocr)            EasyOCR + corrector
   → evidence + metadata     (trace_cv/evidence)       annotated image + JSON
   → storage + analytics      (trace_cv/storage)        SQLite/Postgres
   → REST API + dashboard     (trace_cv/api, dashboard)
```

The seven violations: **helmet non-compliance, seatbelt non-compliance, triple
riding, wrong-side driving, stop-line, red-light, illegal parking**.

---

## Quickstart (no ML models required)

```bash
pip install -r requirements.txt          # core, CPU-friendly
export PYTHONPATH=$PWD                    # or: pip install -e .

python -m trace_cv.cli seed-demo -n 40   # populate DB + evidence images
python -m trace_cv.cli serve             # http://localhost:8000  (dashboard + API)
python -m trace_cv.cli eval              # evaluation showcase (mAP, P/R/F1, CER)
```

Open <http://localhost:8000> for the dashboard.

### Enable real detection / OCR

```bash
pip install -r requirements-ml.txt       # ultralytics, torch, easyocr, ...
python -m trace_cv.cli detect path/to/traffic.jpg
```

YOLO COCO weights auto-download on first use and already cover person, bicycle,
car, motorcycle, bus, truck and traffic light — enough to drive triple-riding,
red-light, stop-line, parking and wrong-side detection. Helmet/seatbelt need a
checkpoint configured in `config/default.yaml`.

---

## REST API

| Method | Path | Purpose |
|---|---|---|
| GET  | `/api/health` | status + which models are loaded |
| POST | `/api/analyze` | analyze an uploaded image (`file`) |
| GET  | `/api/violations` | list (`?type=&plate=&limit=&offset=`) |
| GET  | `/api/violations/{id}` | one record |
| GET  | `/api/violations/{id}/evidence` | annotated evidence image |
| GET  | `/api/events/{event_id}/evidence` | evidence by event |
| GET  | `/api/analytics/summary` | counts by type/hour/vehicle, top plates, FPS |
| GET  | `/api/plates/search?q=` | fuzzy plate search |

Interactive docs at `/docs`.

---

## Configuration

`config/default.yaml` holds thresholds, model paths, and **per-camera scene
geometry** — the stop-line row, lane divider + legal direction, signal ROI, and
no-parking polygons. Scene geometry is what turns geometric detections into
red-light / stop-line / wrong-side / parking violations. Point at a custom file
with `--config` or the `TRACE_CONFIG` env var.

---

## Project layout

```
trace_cv/
  core/         types, geometry, config, logging
  preprocessing/ adaptive quality pipeline (OpenCV)
  detection/    YOLO detector, IOU tracker, ROI helpers
  violation/    base + 7 modules + engine
  ocr/          Indian-plate corrector + EasyOCR wrapper
  evidence/     annotator + evidence builder
  storage/      SQLAlchemy models + repository
  evaluation/   P/R/F1, mAP, OCR CER/exact-match
  api/          FastAPI app + routes
  pipeline.py   end-to-end orchestrator
  cli.py        command-line interface
dashboard/      control-room web UI (served at /)
scripts/        seed_demo, run_eval
tests/          pytest suite
docs/           concept note (md/html/pdf)
```

---

## Testing & evaluation

```bash
PYTHONPATH=$PWD pytest -q          # 34 tests, no ML stack required
python -m trace_cv.cli eval        # metric harness on synthetic data
```

`trace_cv/evaluation/metrics.py` provides Accuracy, Precision, Recall, F1,
multi-label reports, detection **mAP@0.5** and **mAP@0.5:0.95**, and OCR CER /
exact-match. Feed predictions + ground truth from your annotated test split
directly into these functions.

---

## Docker

```bash
docker compose up --build          # API + dashboard on :8000 (SQLite)
```

---

## Notes

- The system degrades gracefully: it imports and runs before the ML stack is
  installed, and never emits a violation for a model that isn't loaded.
- Runtime artifacts (`data/output/`, `*.db`, model weights) are git-ignored.
