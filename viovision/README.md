# VioVision — boilerplate

Implements the architecture from the training guide: one PyTorch component
(YOLOv11 detector, non-negotiable per the guide), three scikit-learn
classifiers (helmet, seatbelt, signal-state), pure-logic geometry checks
for the four no-model violations, an OCR ensemble adapter, and a
prompt-only VLM adjudication adapter.

## Why sklearn here and torch there

YOLOv11 is a real-time multi-class object detector — there's no scikit-learn
equivalent at that accuracy/speed point, and the guide treats it as the
one thing worth fine-tuning a deep model for. Everything downstream of it
(helmet, seatbelt, signal-state) is a small binary/multiclass classifier on
a single cropped region, which is exactly the regime where HOG + color
histogram features feeding an SVM work fine and train in seconds on a CPU,
no GPU needed. `requirements.txt` reflects this split: torch/ultralytics
and paddleocr are commented out as optional, the core sklearn stack is
required.

## Layout

```
src/adapters/schema.py        - the contract: Detection, ClassifierResult,
                                  OCRResult, VLMVerdict + Protocol definitions
src/adapters/mocks.py         - mock implementation of every adapter
src/adapters/ocr_adapter.py   - PaddleOCR + Indian-plate-regex ensemble
src/adapters/vlm_adapter.py   - Claude-based prompt-only adjudicator
src/models/base_crop_classifier.py   - shared sklearn train/predict/save/load
src/models/helmet_classifier.py      - subclass: class names + threshold
src/models/seatbelt_classifier.py    - subclass: CLAHE on, higher threshold
src/models/signal_state_classifier.py - sklearn path + HSV heuristic fallback
src/models/yolo_detector.py   - the one torch-backed adapter, isolated
src/utils/features.py         - shared HOG + color-hist feature extraction
src/utils/geometry_violations.py - triple-riding/stop-line/parking/wrong-side,
                                     zero trained models
src/pipeline_factory.py       - builds real-or-mock adapters from config
configs/pipeline.yaml         - per-component USE_MOCKS toggle + thresholds
configs/traffic.yaml          - YOLO dataset config (class order matters!)
scripts/smoke_test_pipeline.py    - run this first, proves Day-1 flow
scripts/train_crop_classifier.py  - trains helmet/seatbelt/signal from
                                      class-named folders of crops
scripts/train_yolo.py             - wraps the Ultralytics fine-tune recipe
```

## Quickstart

```bash
pip install -r requirements.txt   # core sklearn stack only, ~seconds

# 1. Prove the pipeline wires together end to end, all mocked
python scripts/smoke_test_pipeline.py

# download all the helmet, seatbelt and windshield datasets

python scripts/prepare_helmet_dataset.py --api-key 6saSQEi01HXen4FmER89 --skip-download

python scripts/prepare_seatbelt_dataset.py --api-key 6saSQEi01HXen4FmER89

python scripts/prepare_seatbelt_and_windshield_dataset.py --api-key 6saSQEi01HXen4FmER89

huggingface-cli download iisc-aim/UVH-26 --repo-type dataset --local-dir ./uvh26/

python scripts/prepare_uvh26_detector.py --uvh-dir ./uvh26/ --partial-ok --dry-run

python scripts/prepare_uvh26_detector.py --uvh-dir ./uvh26/ --partial-ok

# 2. Once you have labeled crops in data/annotations/<model>/<class>/*.jpg:
python scripts/train_crop_classifier.py \ 
    --model helmet \
    --data-dir data/annotations/helmet \
    --out models/weights/helmet_svm.pkl

python scripts/train_crop_classifier.py \
    --model seatbelt \
    --data-dir data/annotations/seatbelt \
    --out models/weights/seatbelt_svm.pkl

# 3. Flip configs/pipeline.yaml's use_mocks.helmet to false, rerun the
#    smoke test — same code path, now backed by your trained model.

# 4. Once enough real images are annotated (configs/traffic.yaml schema),
#    uncomment torch/ultralytics in requirements.txt and run:
python scripts/train_yolo.py --epochs 80 --imgsz 640 --batch 8 --freeze-backbone-epochs 15

python scripts/train_yolo.py \
    --data configs/traffic.yaml \
    --epochs 65 \
    --imgsz 640 \
    --batch 8 \
    --base-weights runs/detect/runs/train/viovision_yolo11n_phase1/weights/best.pt \
    --name viovision_yolo11n \
    --skip-preflight

## After everything works, set config/pipeline.yaml as 
use_mocks:
  detector: false
  helmet: false
  seatbelt: false
  signal: true
  ocr: false
  vlm: true

## and then run

python3 ./scripts/smoke_test_pipeline.py

```

## Things you still need to fill in

- `data/annotations/{helmet,seatbelt,signal}/<class>/` folders with real crops.
- `configs/calibration` polygons in `pipeline.yaml` — currently placeholders,
  these are per-camera and must be set during deployment, not training.
- The orchestrator that ties violation-checks + classifiers + OCR + VLM
  together into a per-frame decision loop. `pipeline_factory.py` gives you
  the adapters; wiring them into the actual seven-violation decision logic
  is the next layer up, deliberately left out here since it's where your
  team's specific business logic (thresholds, which violations need OCR,
  when to call VLM) lives.

## Tracking (ByteTrack)

`illegal_parking` and `wrong_side_driving` are stateful across frames and
need a consistent `track_id` per object. Use `pipeline.detector.track(frame,
persist=True)` instead of `.predict(frame)` for these — it runs Ultralytics'
built-in ByteTrack (`tracker="bytetrack.yaml"`) and populates
`Detection.track_id`. `predict()` always leaves `track_id=None`; use it for
the stateless checks (triple_riding, stop_line, helmet, seatbelt, signal,
OCR) where no per-object history is needed.

The mock detector simulates a drifting car and a stationary two-wheeler
across calls so `scripts/smoke_test_pipeline.py` exercises `ParkingTracker`
and `check_wrong_side` end-to-end without needing torch/ultralytics
installed — see that script for the wiring pattern (one `ParkingTracker`
instance and one `TrackHistory` per camera, fed frame-by-frame).

`track()` is unavailable when `use_sahi: true` — ByteTrack's internal state
doesn't integrate with SAHI's tiled single-frame inference. If you need
both small-object recall and tracking, run `predict()` with SAHI for
detection-only frames and switch to `track()` without SAHI for the camera
feeds where the stateful violations apply.
