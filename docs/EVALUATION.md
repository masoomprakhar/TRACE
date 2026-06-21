# How to evaluate TRACE (Accuracy, Precision, Recall, F1, mAP)

This guide matches the problem statement evaluation requirements.

## What gets measured

| Metric | What it tests | TRACE output key |
|---|---|---|
| **mAP@0.5** | Detection box quality (car, person, plate…) | `detection.map50` |
| **mAP@0.5:0.95** | Stricter detection average | `detection.map5095` |
| **Precision** | How many predicted violations were correct | `violation_classification.macro.precision` |
| **Recall** | How many real violations were caught | `violation_classification.macro.recall` |
| **F1-score** | Balance of precision and recall | `violation_classification.macro.f1` |
| **Accuracy** | Exact violation-set match per image | `exact_match_accuracy` |
| **OCR exact match / CER** | Plate text correctness | `ocr.exact_match`, `ocr.mean_cer` |
| **FPS / latency** | Computational efficiency | `efficiency.fps`, `efficiency.mean_ms_per_frame` |

## Step 1 — Install and configure

```bash
cd /path/to/Flipkart-grid-claude-tender-goldberg-3cf1j7
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-ml.txt
export PYTHONPATH=$PWD
export ROBOFLOW_API_KEY=your_key        # if using Roboflow helmet/plate
export TRACE_CONFIG=config/roboflow.yaml
```

## Step 2 — Build labeled eval set (CRITICAL)

Metrics are only as good as your labels. You need **real images** + `data/eval/manifest.json`.

### Option A — Roboflow datasets (fastest)

```bash
# Plates
python scripts/roboflow_download_eval.py \
  --project indian-license-plate-detection-computer-vision-dataset \
  --version 1 --split test --max-images 50

# Helmets (adjust slug from --list-projects)
python scripts/roboflow_download_eval.py \
  --project helmet-gj8do --version 2 --split test --max-images 50
```

Then edit `manifest.json`: set `"violations": ["no_helmet"]` or `[]` per image.

### Option B — IDD Lite (IIIT Hyderabad) — **now integrated**

If you have `idd-lite.tar.gz` from [IDD](https://idd.insaan.iiit.ac.in/):

```bash
python scripts/import_idd_lite.py --tar ~/Downloads/idd-lite.tar.gz --split val
# or: python -m trace_cv.cli import-idd
```

This imports **204 real Indian road images** (Hyderabad/Bangalore) with
auto-generated detection boxes from semantic masks.

### Option C — Mixed eval set (best for submission)

Combine in one `manifest.json`:

- 30 junction images (IDD or CCTV) → detection + triple riding + red light
- 20 helmet images → `no_helmet` labels
- 20 plate images → `license_plate` boxes + `detail.plate_text`
- 10 clean images → `"violations": []`

### Manifest entry template

```json
{
  "id": "sample_01",
  "image": "data/eval/images/sample_01.jpg",
  "width": 1280,
  "height": 720,
  "vehicle": "motorcycle",
  "violations": ["no_helmet"],
  "detections_gt": [
    {"cls": "motorcycle", "bbox": [400, 300, 600, 500], "confidence": 1.0},
    {"cls": "person", "bbox": [430, 250, 500, 350], "confidence": 1.0}
  ],
  "detail": {"plate_text": "MH01AB1234"}
}
```

## Step 3 — Run full evaluation

```bash
# Full end-to-end (detector + violations + OCR) — use for final report
python scripts/run_full_eval.py --config config/roboflow.yaml

# Violation rules only (GT boxes) — use to debug logic separate from YOLO
python scripts/run_full_eval.py --config config/roboflow.yaml --gt-detections
```

Outputs:

- `data/eval/results.json` — machine-readable
- `data/eval/REPORT.txt` — paste into concept note

## Step 4 — Read the report

Example:

```
DETECTION — mAP
  mAP@0.5      : 0.6200
  mAP@0.5:0.95 : 0.4100

VIOLATION CLASSIFICATION
  Macro F1     : 0.7800
  Per violation:
    no_helmet          P=0.85  R=0.80  F1=0.82
    triple_riding      P=0.90  R=0.88  F1=0.89

EFFICIENCY
  Mean latency : 3200 ms/frame
  Throughput   : 0.31 FPS
```

## Step 5 — Put numbers in your concept note

| Component | Metric | Your result |
|---|---|---|
| Object detection | mAP@0.5 | from report |
| Violation classification | Macro F1 | from report |
| Helmet | Per-class F1 | from `per_label.no_helmet` |
| License plate OCR | Exact match | from `ocr.exact_match` |
| System | Latency / FPS | from `efficiency` |

## Two eval modes — when to use which

| Mode | Command | Measures |
|---|---|---|
| **Full pipeline** | `run_full_eval.py` | Real-world: YOLO + rules + OCR together |
| **GT detections** | `run_full_eval.py --gt-detections` | Violation logic only (ignores detector errors) |

For submission: report **both**. Full pipeline = honest system score. GT mode = shows your rules work.

## Minimum credible eval set

| Category | Min images | Labels needed |
|---|---|---|
| Detection | 50 | Boxes: car, motorcycle, person |
| Helmet | 20 | `no_helmet` or clean |
| Plates | 30 | `license_plate` box + `plate_text` |
| Red light | 15 | `red_light` + signal visible |
| Clean (no violation) | 15 | `violations: []` |

**Total: ~80–100 labeled images** → solid, defensible metrics.

## Quick smoke test (synthetic — not for submission)

```bash
python -m trace_cv.cli eval          # tiny synthetic demo numbers
python scripts/run_full_eval.py --rebuild   # synthetic images only
```

Replace synthetic set before submitting.
