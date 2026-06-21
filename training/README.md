# TRACE — Training pipeline

Scripts to (re)train the models TRACE can use. The runtime works out of the
box on COCO YOLO weights for the geometry-driven violations (triple riding,
red-light, stop-line, parking, wrong-side). The models trained here are the
**optional** ones that unlock the rest:

| Model | Script | Output | Wired via |
|---|---|---|---|
| Vehicle/person detector (fine-tuned) | `train_detector.py` | `models/weights/vehicle_detector.pt` | `models.detector` |
| Helmet (cls or det) | `train_helmet.py` | `models/weights/helmet.pt` | `models.helmet` |
| Seatbelt (cls or det) | `train_seatbelt.py` | `models/weights/seatbelt.pt` | `models.seatbelt` |
| License-plate detector | `train_plate.py` | `models/weights/plate.pt` | `models.plate` |

All scripts are [Ultralytics](https://docs.ultralytics.com/)-based. They
import the ML stack lazily, so they stay importable for `--help` without it;
running an actual train requires the extras:

```bash
pip install -r requirements-ml.txt        # ultralytics, torch, albumentations, ...
export PYTHONPATH=$PWD                     # so trace_cv is importable
```

> The class **names** matter — the runtime interpreters key off them. Keep the
> names exactly as documented below or the violation modules won't recognise
> the predictions. (helmet: `helmet`/`no_helmet`; seatbelt:
> `belt`/`no_belt`/`occluded`; plate: `license_plate`.)

---

## Datasets

You bring the data. Recommended public sources:

### Vehicles + persons — IDD (India Driving Dataset)
- IDD Detection from <https://idd.insaan.iiit.ac.in/> (free registration).
  Indian road scenes with vehicles, riders, pedestrians, autorickshaws.
- Convert its annotations to **YOLO detection format** and **remap class ids**
  to the names in `datasets/vehicle.example.yaml`
  (`person, bicycle, car, motorcycle, bus, truck, traffic_light`). Keeping
  these COCO-style names lets the rest of the pipeline run unchanged.
- Fine-tuning is optional: COCO `yolov8n.pt` already detects these classes.
  Fine-tune when you need better recall on Indian traffic (autos, dense
  two-wheelers, night).

### Helmet — Roboflow Universe
- Search Roboflow Universe for "helmet detection" / "rider helmet" datasets
  (e.g. motorcyclist helmet-vs-no-helmet sets).
- For **classification** (default mode): export as **Folder** and arrange as
  `train/helmet`, `train/no_helmet`, `val/helmet`, `val/no_helmet`. Rename the
  exported class folders to exactly `helmet` and `no_helmet`.
- For **detection**: export as **YOLOv8** and edit the dataset YAML so
  `names: [helmet, no_helmet]`.

### Seatbelt — Roboflow Universe
- Search for "seatbelt detection" datasets (driver belted / not belted /
  occluded windshield).
- Use **three** classes: `belt`, `no_belt`, `occluded`. The `occluded` class
  is essential — it is how TRACE avoids false positives from window glare /
  A-pillars (it never flags an occluded driver). If your source has only two
  classes, add `occluded` crops (glare, reflections, blocked view) yourself.
- Crop to the **driver window** before training so inputs match the runtime ROI
  (`trace_cv/detection/roi.py` → `driver_roi`, front-left of a car box).

### License plate — Roboflow Universe
- Search for "license plate" / "number plate" / "Indian number plate"
  detection datasets. Export as **YOLOv8** with a single class.
- Class name doesn't matter to the runtime (the plate detector is built with
  `keep=set()` and accepts anything), but keep `license_plate` for clarity —
  see `datasets/plate.example.yaml`.

### Exporting in YOLO format — quick notes
- **Detection**: `images/{train,val}/*.jpg` + `labels/{train,val}/*.txt`, each
  label line `class_id x_center y_center w h` (normalised 0–1), plus a dataset
  YAML (`path`, `train`, `val`, `names`). See the two example YAMLs in
  `datasets/`.
- **Classification**: an ImageFolder tree `train/<class>/*.jpg` and
  `val/<class>/*.jpg`; no YAML needed — pass the folder as `--data`.
- Roboflow's "YOLOv8" (detection) and "Folder" (classification) export presets
  produce exactly these layouts.

---

## Augmentation & synthetic data

`augment.py` builds an [Albumentations](https://albumentations.ai/) pipeline
that simulates the hard conditions Indian footage is full of — **RandomRain,
RandomFog, RandomShadow, MotionBlur, RandomBrightnessContrast, GaussNoise**.

Use it as a library inside your Dataloader (recommended), or as a CLI to
materialise an augmented copy of a folder / preview a grid:

```bash
# preview a few augmentations stacked into one image
python training/augment.py --src data/raw/helmet --preview data/aug_preview.jpg

# write 3 augmented variants per source image into a new folder
python training/augment.py --src data/raw/helmet --dst data/aug/helmet -n 3
```

```python
from training.augment import build_transform
tf = build_transform(p=0.5)
img_aug = tf(image=img)["image"]
```

The CLI transforms **images only** (safe for classification folders). For
**detection** sets, run these transforms through Ultralytics' built-in
Albumentations hook, or wrap them in an `A.Compose(..., bbox_params=...)` so
boxes move with the pixels. (The `train_*.py` detection modes already enable
Ultralytics' standard mosaic/HSV/flip augmentation; `augment.py` is for the
heavier weather effects and for pre-baking an augmented copy.)

**Rare classes (e.g. night triple-riding, no-helmet at night).** These are
scarce in public data. Boost them by:
- over-augmenting the few real positives you have (run `augment.py` with a
  higher `-n` and lower brightness), and
- **synthetic compositing** — copy-paste helmet/no-helmet head crops onto
  rider boxes, paste extra riders to fabricate triple-riding, then apply the
  night/rain transforms so the composite blends in. Keep a held-out set of
  *real* hard frames for validation so you don't fool yourself with synthetics.

---

## Commands

Run from the repo root with `PYTHONPATH=$PWD` set.

### 1. Vehicle/person detector
```bash
python training/train_detector.py \
    --data training/datasets/vehicle.example.yaml \
    --weights yolov8n.pt --epochs 50 --imgsz 640 --batch 16 --device 0
```
- `--weights yolov10n.pt` to fine-tune a YOLOv10 backbone instead.
- Best weights → `models/weights/vehicle_detector.pt`.

### 2. Helmet
```bash
# classification (default) — folder dataset with helmet/ and no_helmet/
python training/train_helmet.py --mode cls \
    --data data/datasets/helmet_cls --epochs 30 --imgsz 224 --device 0

# detection — dataset YAML with names: [helmet, no_helmet]
python training/train_helmet.py --mode det \
    --data training/datasets/helmet.yaml --epochs 50 --imgsz 640 --device 0
```
- Best weights → `models/weights/helmet.pt`.

> **No dataset handy?** `make_synthetic_helmet.py` generates a synthetic
> helmet/no_helmet set so you can exercise the whole pipeline end-to-end and
> produce a loadable `helmet.pt`. It is a **plumbing smoke test only** — a model
> trained on synthetic heads will not generalise to real photos; use a real
> Roboflow set for production.
> ```bash
> python training/make_synthetic_helmet.py --out data/datasets/helmet_cls
> python training/train_helmet.py --mode cls --data data/datasets/helmet_cls \
>     --epochs 12 --imgsz 96 --device cpu
> ```
> (Verified: trains in ~1 min on CPU to 100% top-1 on the held-out synthetic
> val split, and `model_status()` then reports `helmet: True`.)

### 3. Seatbelt
```bash
# classification (default) — folder dataset belt/ no_belt/ occluded/
python training/train_seatbelt.py --mode cls \
    --data data/datasets/seatbelt_cls --epochs 30 --imgsz 224 --device 0

# detection — dataset YAML with names: [belt, no_belt, occluded]
python training/train_seatbelt.py --mode det \
    --data training/datasets/seatbelt.yaml --epochs 50 --imgsz 640 --device 0
```
- Best weights → `models/weights/seatbelt.pt`.

> **No dataset handy?** `make_synthetic_seatbelt.py` generates a synthetic
> belt/no_belt/occluded set (same plumbing-smoke-test caveat as helmet — not
> for production).
> ```bash
> python training/make_synthetic_seatbelt.py --out data/datasets/seatbelt_cls
> python training/train_seatbelt.py --mode cls --data data/datasets/seatbelt_cls \
>     --epochs 12 --imgsz 96 --device cpu
> ```
> (Verified: ~1 min on CPU, 100% top-1 on the held-out synthetic val split,
> and `model_status()` then reports `seatbelt: True`.)

### 4. License plate
```bash
python training/train_plate.py \
    --data training/datasets/plate.example.yaml \
    --weights yolov8n.pt --epochs 50 --imgsz 640 --batch 16 --device 0
```
- Best weights → `models/weights/plate.pt`.

> **No dataset handy?** `make_synthetic_plate.py` generates a synthetic
> license-plate detection set (YOLO detection format) so you can exercise the
> full pipeline end-to-end. Same plumbing-smoke-test caveat as helmet/seatbelt —
> not for production.
> ```bash
> python training/make_synthetic_plate.py --out data/datasets/plate_det
> python training/train_plate.py --data data/datasets/plate_det/dataset.yaml \
>     --weights yolov8n.pt --epochs 12 --imgsz 320 --batch 16 --device cpu
> ```
> (Verified: ~3 min on CPU to mAP50=0.995 on the held-out synthetic val split,
> and `model_status()` then reports `plate: True`.)

Every script prints the exact config line to set when it finishes, and copies
`best.pt` into `models/weights/` for you. Run any with `--help` for the full
flag list. Use `--device cpu` to force CPU, `--device 0` (or `0,1`) for GPU(s).

---

## Wire the weights into TRACE

Trained weights land in `models/weights/` (git-ignored — only `.gitkeep` is
tracked). Point the runtime at them in `config/default.yaml`:

```yaml
models:
  detector: models/weights/vehicle_detector.pt   # or keep yolov8n.pt
  helmet:   models/weights/helmet.pt              # was null
  seatbelt: models/weights/seatbelt.pt            # was null
  plate:    models/weights/plate.pt               # was null
  ocr_langs: [en]
```

The helmet/seatbelt/plate modules are skipped while their key is `null`; set
the path and they activate automatically — nothing else to change. Verify:

```bash
python -m trace_cv.cli detect path/to/traffic.jpg
# the "models:" line in the output shows which checkpoints loaded
```

---

## Notes
- These scripts never run training or downloads on their own — you invoke them.
- Weights are intentionally kept out of git (`.gitignore` ignores `*.pt` and
  `models/weights/` except `.gitkeep`).
- Tune `thresholds.helmet_conf` / `thresholds.seatbelt_conf` in
  `config/default.yaml` to trade precision vs recall once a model is wired in.
