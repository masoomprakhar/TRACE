# TRACE — Requirements Coverage

How TRACE maps to every task in the problem statement. Status legend:
**✅ Implemented & verified** · **🔌 Pluggable** (works once a model/config is
supplied; never fabricated when absent).

| # | Problem-statement task | Status | Where |
|---|---|---|---|
| 1 | **Image preprocessing** — enhance & normalize | ✅ | `preprocessing/pipeline.py` |
| 1 | Handle low-light / rain / shadow / motion-blur | ✅ | adaptive router: CLAHE+gamma, Dark-Channel-Prior dehaze, unsharp deblur |
| 2 | Detect vehicles, riders, drivers, pedestrians | ✅ | `detection/detector.py` (YOLO, verified on real photo) |
| 2 | Classify vehicle categories | ✅ | `core/types.py::VehicleClass` (car/moto/bus/truck/bicycle/person) |
| 3 | Helmet non-compliance | 🔌 | `violation/helmet.py` (+ `training/train_helmet.py`) |
| 3 | Seatbelt non-compliance | 🔌 | `violation/seatbelt.py` (+ `training/train_seatbelt.py`) |
| 3 | Triple riding | ✅ | `violation/triple_riding.py` |
| 3 | Wrong-side driving | ✅ | `violation/wrong_side.py` (needs lane calibration) |
| 3 | Stop-line violation | ✅ | `violation/stop_line.py` (needs stop-line config) |
| 3 | Red-light violation | ✅ | `violation/red_light.py` (HSV signal + stop line) |
| 3 | Illegal parking | ✅ | `violation/parking.py` (zones + dwell time) |
| 4 | Categorize into predefined classes | ✅ | `core/types.py::ViolationType` |
| 4 | Assign confidence scores | ✅ | per-`Violation.confidence` |
| 4 | Calibrated confidence | ✅ | `evaluation/calibration.py` (temperature scaling, ECE) |
| 5 | Detect number plates | 🔌 | `detection` plate model, else lower-ROI fallback |
| 5 | OCR registration details | ✅ | `ocr/plate_ocr.py` (EasyOCR, verified live) |
| 5 | India plate correction | ✅ | `ocr/corrector.py` (format + confusion-aware) |
| 6 | Annotated evidence images | ✅ | `evidence/annotator.py`, `builder.py` |
| 6 | Store metadata + timestamps | ✅ | `storage/` (SQLite/Postgres) + JSON sidecar |
| 7 | Violation statistics & trends | ✅ | `GET /api/analytics/summary` (by type/hour/vehicle) |
| 7 | Searchable records | ✅ | `GET /api/violations`, `GET /api/plates/search` |
| 7 | Summary reports / export | ✅ | `GET /api/report/summary`, `GET /api/violations.csv` |
| 8 | Accuracy, Precision, Recall, F1 | ✅ | `evaluation/metrics.py` |
| 8 | mAP (@0.5 and @0.5:0.95) | ✅ | `evaluation/metrics.py::detection_map` |
| 8 | OCR CER / exact-match | ✅ | `evaluation/metrics.py` |
| 8 | Computational efficiency | ✅ | `scripts/benchmark.py` (per-stage latency + FPS) |
| 8 | Scalability | ✅ | stateless pipeline, Docker, ONNX/edge export path |
| — | Robust to conditions / density / quality | ✅ | adaptive preprocessing + tracking + confirm-frames |

## Notes on the 🔌 items

Helmet, seatbelt, and dedicated plate **detection** need a trained checkpoint —
COCO (the default detector) has no such classes. This is deliberate: the
modules report `available == False` and are skipped rather than guessing, so
the system never issues a false citation. `training/` provides runnable
scripts and dataset instructions to produce these models; once trained, point
`config/default.yaml` (`models.helmet` / `models.seatbelt` / `models.plate`)
at the weights and the modules activate automatically.

Everything else runs out of the box on CPU with auto-downloaded COCO weights.
