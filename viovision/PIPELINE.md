# VioVision — Pipeline Execution Guide

> **Multi-model traffic violation detection system** · YOLOv11 · ByteTrack · sklearn SVM · PaddleOCR · Claude VLM adjudication

---

## Architecture Overview

VioVision implements a **six-stage inference pipeline** with a strict adapter contract separating every component behind a Protocol interface. All components run behind a `USE_MOCKS` toggle in `configs/pipeline.yaml` — swapping any individual component from mock to production is a single config line change, no code modifications required.

```
Frame Input
    │
    ▼
[YOLOv11n Detector + ByteTrack]  ←── fine-tuned, 6 classes
    │
    ├──▶ Geometry Engine (no model) ──▶ triple_riding, stop_line,
    │                                    illegal_parking, wrong_side_driving
    │
    ├──▶ [Helmet SVM Classifier]   ──▶ no_helmet violation
    │
    ├──▶ [Seatbelt SVM Classifier] ──▶ no_seatbelt violation
    │
    ├──▶ [Signal-State Classifier] ──▶ feeds red_light geometry check
    │         (sklearn or HSV heuristic fallback)
    │
    ├──▶ [PaddleOCR Ensemble]      ──▶ plate text + Indian regex post-filter
    │
    └──▶ [Claude VLM Adjudicator]  ←── low-confidence detections only
              (prompt-only, zero fine-tuning)
```

---

## Execution Order

### Phase 0 — Environment Setup

```bash
# Core dependencies (sklearn stack — no GPU required)
pip install -r requirements.txt

# Verify end-to-end pipeline wiring before any training (all-mock mode)
python scripts/smoke_test_pipeline.py
```

Expected output: all 7 pipeline components wire and produce output. ByteTrack stateful tracking verified across a 20-frame simulated sequence. **If this fails, fix it before touching any training script.**

---

### Phase 1 — Dataset Acquisition & Preprocessing

Run in this exact order. Each script is idempotent — safe to re-run with `--skip-download` if the raw data is already on disk.

#### 1a. Helmet Dataset
**Source:** Khadatkar & Wasule · Roboflow Universe
```bash
python scripts/prepare_helmet_dataset.py --api-key <ROBOFLOW_API_KEY>
```
- Downloads ~1,533 images in YOLOv11 format
- Remaps `With Helmet` → `helmet`, `Without Helmet` → `no_helmet`
- Discards `License Plate` boxes (not needed for the classifier)
- Output: `data/annotations/helmet/{helmet,no_helmet}/`

#### 1b. Seatbelt Dataset — Primary
**Source:** seatbelttraining · Roboflow Universe
```bash
python scripts/prepare_seatbelt_dataset.py --api-key <ROBOFLOW_API_KEY>
```
- Downloads 3,489 images; single positive class (`seatbelt`)
- Synthesises `no_seatbelt` negatives via non-overlapping upper-body region sampling at 1:1 ratio
- Applies CLAHE preprocessing to all crops at save time (train/inference consistency enforced)
- Output: `data/annotations/seatbelt/{seatbelt,no_seatbelt}/`

#### 1c. Seatbelt + Windshield Dataset — Supplement
**Source:** aiactive · Roboflow Universe  
⚠️ **Must run AFTER 1b** — this script appends to the seatbelt annotation dirs
```bash
python scripts/prepare_seatbelt_and_windshield_dataset.py --api-key <ROBOFLOW_API_KEY>
```
- Produces **true-negative** `no_seatbelt` crops (windshield regions with no seatbelt annotation — higher label quality than synthesised negatives from 1b)
- Side-output: `data/splits/windshield_finetune/` — YOLO-format windshield boxes for the primary detector's `windshield` class (class id 4)

#### 1d. Primary Detector Dataset — UVH-26
**Source:** IIT Hyderabad · [Download from paper](https://arxiv.org/abs/2511.02563)  
⚠️ **Manual download required** — extract and then run:
```bash
# Dry-run first — inspect the class remap before writing anything
python scripts/prepare_uvh26_detector.py --uvh-dir /path/to/uvh26/ --dry-run

# If remap looks correct, execute
python scripts/prepare_uvh26_detector.py --uvh-dir /path/to/uvh26/
```
- Remaps 14 India-specific vehicle classes → 3 VioVision base classes (`car`, `two_wheeler`, `person`)
- Writes directly to `data/splits/{train,valid,test}/` with `uvh_` filename prefix to prevent merge collisions
- 66,986 images from Bengaluru CCTV, includes night/rain/fog/dense traffic

#### 1e. Merge & Validate
```bash
python scripts/merge_detector_datasets.py --extra data/splits/windshield_finetune
```
- Merges all detector data sources into the canonical `data/splits/` layout
- Prints per-class box count with ASCII balance chart
- Flags any class with < 200 boxes
- **Do not proceed to Phase 2 until `license_plate` and `signal_light` class counts are non-zero** — see [Manual Requirements](#manual-requirements) below

---

### Phase 2 — Model Training

#### 2a. Helmet Classifier
```bash
python scripts/train_crop_classifier.py \
    --model helmet \
    --data-dir data/annotations/helmet \
    --out models/weights/helmet_svm.pkl
```
- HOG + HSV color histogram features → RBF SVM with `class_weight='balanced'`
- Prints per-class precision/recall/F1 on held-out val split
- Expected accuracy: 90%+ on clean daytime crops; check hard-conditions subset separately

#### 2b. Seatbelt Classifier
```bash
python scripts/train_crop_classifier.py \
    --model seatbelt \
    --data-dir data/annotations/seatbelt \
    --out models/weights/seatbelt_svm.pkl
```
- CLAHE suppressed at training time (crops pre-processed in Phase 1); re-applied at inference on raw windshield crops
- Higher review threshold (0.65 vs 0.55) — low-confidence predictions route to VLM adjudication rather than auto-filing

#### 2c. Signal-State Classifier *(optional — HSV heuristic is the default fallback)*
```bash
# Only run this if you have sourced LISA / Bosch traffic-light crops
python scripts/train_crop_classifier.py \
    --model signal \
    --data-dir data/annotations/signal \
    --out models/weights/signal_svm.pkl
```
- If not trained, pipeline defaults to `mode: hsv_heuristic` in `pipeline.yaml` — robust for daylight, degrades gracefully to `unknown` at night (routes to VLM review)

#### 2d. YOLOv11n Fine-Tune
```bash
# Install GPU stack (not needed for Phases 1–2a/b/c)
pip install torch torchvision ultralytics

# Single-phase (recommended if dataset < 5k images)
python scripts/train_yolo.py --data configs/traffic.yaml --epochs 80 --imgsz 960

# Two-phase with backbone freeze warmup (recommended if dataset > 5k images)
python scripts/train_yolo.py --data configs/traffic.yaml --epochs 80 \
    --freeze-backbone-epochs 15 --imgsz 960
```
- Preflight check runs automatically: validates split dirs, label counts, and class schema against `traffic.yaml` before consuming GPU time
- Phase 1 saves to `runs/train/viovision_yolo11n_phase1/`, Phase 2 loads `best.pt` from it — no `resume=True` (broken in Ultralytics for cross-phase unfreezing)
- Drop `--imgsz` to 640 if GPU OOM on T4/Colab

---

### Phase 3 — Activate Real Models

Edit `configs/pipeline.yaml` — flip each component as its weights become available:

```yaml
use_mocks:
  detector: false      # after train_yolo.py completes
  helmet: false        # after train_crop_classifier.py --model helmet
  seatbelt: false      # after train_crop_classifier.py --model seatbelt
  signal: false        # after train_crop_classifier.py --model signal (or leave true for HSV)
  ocr: false           # after pip install paddlepaddle paddleocr
  vlm: false           # after setting ANTHROPIC_API_KEY

weights:
  detector: runs/train/viovision_yolo11n/weights/best.pt
  helmet:   models/weights/helmet_svm.pkl
  seatbelt: models/weights/seatbelt_svm.pkl
  signal:   models/weights/signal_svm.pkl
```

Re-run smoke test after each flip:
```bash
python scripts/smoke_test_pipeline.py
```

---

## What Is Complete ✅

| Component | Status | Notes |
|---|---|---|
| Adapter contract (`schema.py`) | ✅ Complete | Frozen Protocol definitions — `Detection`, `ClassifierResult`, `OCRResult`, `VLMVerdict` |
| Mock adapters | ✅ Complete | Full pipeline runs end-to-end without any trained model |
| YOLOv11 detector adapter | ✅ Complete | `predict()` + ByteTrack `track()`, SAHI-guarded, deferred torch import |
| ByteTrack integration | ✅ Complete | `persist=True`, per-camera instance semantics, incompatibility with SAHI guarded |
| HOG + color histogram feature extractor | ✅ Complete | Size-invariant 8,148-dim vector; CLAHE toggle; single source of truth for train and inference |
| Helmet SVM classifier | ✅ Complete | Trained, saved, loaded, and smoke-tested |
| Seatbelt SVM classifier | ✅ Complete | CLAHE double-application bug fixed; higher review threshold enforced |
| Signal-state classifier | ✅ Complete | sklearn path + HSV heuristic fallback behind one adapter interface |
| Geometry violation engine | ✅ Complete | Triple riding (IoU), stop-line (polygon), illegal parking (ParkingTracker), wrong-side (motion vector) — all unit-tested |
| OCR ensemble adapter | ✅ Complete | PaddleOCR + Indian plate regex + confidence-voted multi-engine |
| VLM adjudication adapter | ✅ Complete | Claude API, prompt-only, JSON parse with fail-safe fallback |
| Pipeline factory | ✅ Complete | Per-component mock/real toggle from `pipeline.yaml` |
| Dataset prep — helmet | ✅ Complete | Roboflow download + class remap + crop extraction |
| Dataset prep — seatbelt (seatbelttraining) | ✅ Complete | Positive crops + synthesised negatives + CLAHE baked in |
| Dataset prep — seatbelt + windshield (aiactive) | ✅ Complete | True-negative crops + windshield YOLO side-output |
| Dataset prep — UVH-26 detector | ✅ Complete | 14-class → 3-class remap + dry-run validation |
| Dataset merge + balance report | ✅ Complete | ASCII class distribution chart, hard-conditions counter |
| `train_crop_classifier.py` | ✅ Complete | Preflight check, imbalance warning, CLAHE override for seatbelt |
| `train_yolo.py` | ✅ Complete | Two-phase freeze fixed, preflight validation, wrong `resume=True` removed |

---

## Manual Requirements ⚠️

These cannot be automated — they require human input, external data, or per-deployment calibration.

### 1. `license_plate` and `signal_light` detector training data
The merge script will report 0 boxes for both classes. You need to source and prep these separately before YOLO training:
- **license_plate:** [License Plate Detection Dataset (10,125 images)](https://www.kaggle.com/datasets/barkataliarbab/license-plate-detection-dataset-10125-images) — download, remap class id to `3`, place in `data/splits/`
- **signal_light:** LISA Traffic Light Dataset or Bosch BSTLD — crop the light boxes, remap to class id `5`

### 2. Camera calibration polygons
Open `configs/pipeline.yaml` and replace the placeholder polygons with real pixel-space coordinates from your actual camera feeds:
```yaml
calibration:
  stop_line_zone:  [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]  # polygon beyond stop line
  no_parking_zone: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]  # restricted parking area
  allowed_direction: [dx, dy]   # unit vector of legal traffic flow direction
```
Without these, `stop_line`, `illegal_parking`, and `wrong_side_driving` violations will not fire correctly.

### 3. Anthropic API key (VLM adjudication)
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```
Set `use_mocks.vlm: false` in `pipeline.yaml` only after this is configured.

### 4. Signal-state training data (if not using HSV heuristic)
Source traffic-light crops from LISA or Bosch BSTLD, organize into `data/annotations/signal/{red,yellow,green}/`, then run `train_crop_classifier.py --model signal`. Note: yellow class will be heavily underrepresented (~2% of LISA+Bosch combined) — oversample or keep HSV heuristic for yellow.

### 5. Per-violation orchestrator
`pipeline_factory.py` provides all initialized adapters. The per-frame decision loop — which crops feed which classifier, when to invoke OCR, when to escalate to VLM, how violations get logged — is not implemented. This is intentional: it is where deployment-specific business logic lives (confidence thresholds, filing rules, alert routing).

---

## File Reference

```
viovision/
├── configs/
│   ├── pipeline.yaml          # USE_MOCKS toggles, weights paths, thresholds, calibration
│   └── traffic.yaml           # YOLO dataset config — class order is the contract
├── scripts/
│   ├── smoke_test_pipeline.py              # run first, run after every mock→real flip
│   ├── prepare_helmet_dataset.py           # Phase 1a
│   ├── prepare_seatbelt_dataset.py         # Phase 1b
│   ├── prepare_seatbelt_and_windshield_dataset.py  # Phase 1c
│   ├── prepare_uvh26_detector.py           # Phase 1d
│   ├── merge_detector_datasets.py          # Phase 1e
│   ├── train_crop_classifier.py            # Phase 2a/b/c
│   └── train_yolo.py                       # Phase 2d
├── src/
│   ├── adapters/
│   │   ├── schema.py          # Frozen adapter contract — do not modify class names
│   │   ├── mocks.py           # Drop-in mocks for every adapter
│   │   ├── ocr_adapter.py     # PaddleOCR ensemble + Indian plate regex
│   │   └── vlm_adapter.py     # Claude API adjudicator
│   ├── models/
│   │   ├── base_crop_classifier.py   # Shared SVM/RF train-predict-save-load
│   │   ├── helmet_classifier.py
│   │   ├── seatbelt_classifier.py
│   │   ├── signal_state_classifier.py
│   │   └── yolo_detector.py          # YOLOv11 + ByteTrack, torch isolated here
│   ├── utils/
│   │   ├── features.py               # HOG + color histogram, CLAHE
│   │   └── geometry_violations.py    # Zero-model violation logic
│   └── pipeline_factory.py           # Builds real/mock adapters from config
└── requirements.txt
```

---

*VioVision — built for 5-day deployment velocity without sacrificing architectural integrity.*
