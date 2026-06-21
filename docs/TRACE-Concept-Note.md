# TRACE
### Traffic Rule Analysis & Compliance Engine

#### Automated Photo Identification and Classification for Traffic Violations Using Computer Vision

**Concept Note / Solution Framework**

| | |
|---|---|
| **Team** | _‹Your team name›_ |
| **Members** | _‹Member 1, Member 2, Member 3, Member 4›_ |
| **Institution** | _‹Your institution›_ |
| **Track** | Computer Vision · Intelligent Transportation Systems |
| **Date** | June 2026 |

---

## 1. Executive Summary

Traffic surveillance cameras across Indian cities generate millions of images every day, yet the vast majority are never reviewed. Manual inspection is slow, expensive, inconsistent, and impossible to scale to the volume of footage now being captured.

**TRACE** is a computer-vision system that automatically processes traffic imagery to detect road users, identify and classify seven categories of traffic violations, read license plates, and generate court-ready annotated evidence — all with confidence scores and full audit metadata. It is engineered to remain accurate under the real-world conditions that defeat naïve detectors: low light, rain, shadows, motion blur, and dense traffic.

Three design decisions set TRACE apart from a generic "run YOLO on an image" approach:

1. **Quality-adaptive preprocessing** that diagnoses each frame and applies only the corrections it needs.
2. **Temporal reasoning** through multi-object tracking, so violations that are fundamentally sequential (triple riding, illegal parking, red-light running) are judged across frames rather than guessed from one.
3. **Domain-adapted plate recognition** tuned to the Indian number-plate format, materially improving OCR accuracy over off-the-shelf engines.

The result is a scalable, auditable enforcement assistant that reduces manual review effort while improving both the consistency and the defensibility of traffic-law enforcement.

---

## 2. Problem Statement & Motivation

India records over **150,000 road-traffic fatalities every year** — among the highest in the world (Ministry of Road Transport & Highways, *Road Accidents in India*). A large share is attributable to preventable, enforceable behaviours: riding without a helmet, driving without a seatbelt, overloading two-wheelers, jumping red lights, and driving on the wrong side.

Enforcement has not kept pace with surveillance. Cities have deployed thousands of cameras, but the bottleneck has simply moved downstream:

- **Volume.** A single junction camera can produce tens of thousands of frames per day. No human team can review them all.
- **Inconsistency.** Manual adjudication varies between reviewers and across shifts; identical violations are treated differently.
- **Latency.** By the time footage is reviewed, the evidentiary and deterrent value has decayed.
- **Cost.** Skilled reviewer time is expensive and poorly utilised on routine, repetitive screening.

The opportunity is clear: an intelligent system that performs the first-pass screening automatically — flagging probable violations with evidence and confidence scores, and escalating only ambiguous cases to a human — turns an unmanageable firehose into a prioritised, reviewable queue.

---

## 3. Solution Overview

TRACE ingests still images or video frames and returns, for every detected violation, a structured record containing: the violation type(s), a confidence score per violation, the offending vehicle's class and license-plate number, an annotated evidence image, and a complete metadata sidecar (timestamp, location, track ID, processing time).

**Capabilities delivered against the problem statement:**

- **Image Preprocessing** — adaptive enhancement for low light, rain/fog, shadows, and motion blur.
- **Vehicle & Road-User Detection** — localises and classifies cars, motorcycles, trucks, buses, bicycles, pedestrians, riders, and drivers.
- **Violation Detection** — seven classes: helmet non-compliance, seatbelt non-compliance, triple riding, wrong-side driving, stop-line violation, red-light violation, and illegal parking.
- **Violation Classification** — each detection is categorised and assigned a calibrated confidence score.
- **License Plate Recognition** — plate localisation followed by OCR with India-specific format correction.
- **Evidence Generation** — annotated images plus persisted metadata and timestamps.
- **Analytics & Reporting** — violation statistics, trends, hotspots, and searchable records.
- **Performance Evaluation** — Accuracy, Precision, Recall, F1-score, and mAP, plus throughput benchmarks.

---

## 4. System Architecture

TRACE is a modular pipeline. Each stage is independently testable and replaceable, which keeps the system maintainable and lets us swap models without disturbing the rest of the flow.

```
                    ┌─────────────────────────────┐
   Image / Video →  │  1. Adaptive Preprocessing  │  low-light · deblur · dehaze
                    └──────────────┬──────────────┘
                                   ▼
                    ┌─────────────────────────────┐
                    │  2. Detection (YOLOv10)      │  vehicles · riders · pedestrians
                    └──────────────┬──────────────┘
                                   ▼
                    ┌─────────────────────────────┐
                    │  3. Tracking (ByteTrack)     │  persistent IDs across frames
                    └──────────────┬──────────────┘
                                   ▼
                    ┌─────────────────────────────┐
                    │  4. Violation Engine         │  7 parallel violation modules
                    └──────────────┬──────────────┘
                                   ▼
                    ┌─────────────────────────────┐
                    │  5. License Plate + OCR      │  detect → read → correct
                    └──────────────┬──────────────┘
                                   ▼
                    ┌─────────────────────────────┐
                    │  6. Evidence + Metadata      │  annotated image + JSON record
                    └──────────────┬──────────────┘
                                   ▼
              ┌──────────────────────────────────────────┐
              │  7. API · Database · Analytics Dashboard  │
              └──────────────────────────────────────────┘
```

---

## 5. Technical Approach

### 5.1 Image Preprocessing — *adaptive, not blanket*

Most pipelines apply a fixed enhancement to every image, which wastes compute and can degrade already-good frames. TRACE first **diagnoses** each frame, then routes it only to the corrections it needs:

| Condition | Detector (cheap heuristic) | Corrective model |
|---|---|---|
| Low light | mean luminance below threshold | Zero-DCE++ (no paired data required) |
| Motion blur | Laplacian variance below threshold | DeblurGAN-v2 (MobileNet backbone) |
| Rain / fog | high-frequency / haze estimate | Dark Channel Prior / MSBDN |
| Shadows / contrast | histogram spread | CLAHE on the luminance channel |

This keeps the average frame fast (most frames need little correction) while rescuing the hard frames that would otherwise produce misses or false alarms.

### 5.2 Vehicle & Road-User Detection

We use **YOLOv10-M** as the primary detector. Its NMS-free design yields 20–30% lower latency than YOLOv8 at comparable accuracy, and a single multi-class head detects vehicles and people together. The base model is fine-tuned on the **India Driving Dataset (IDD)** so it is robust to Indian road scenes, mixed traffic, and occlusion. Detected classes: `motorcycle, car, truck, bus, bicycle, person`.

### 5.3 Traffic Violation Detection

| Violation | Detection strategy |
|---|---|
| **Helmet non-compliance** | Crop rider head region → EfficientNet-B2 classifier (`helmet` / `no_helmet`) |
| **Seatbelt non-compliance** | Crop driver region → MobileNetV3, 3-class (`belt` / `no_belt` / `occluded`) |
| **Triple riding** | Count distinct tracked persons whose centroids fall within a motorcycle's box |
| **Wrong-side driving** | Movement vector (from tracking) compared against the calibrated lane direction |
| **Stop-line violation** | Vehicle box crosses the configured stop-line band while the signal is red |
| **Red-light violation** | Signal-ROI colour state + stop-line crossing, evaluated as a two-event sequence |
| **Illegal parking** | Vehicle stationary beyond a duration threshold inside a no-parking zone polygon |

The **`occluded` seatbelt class** is a deliberate choice: a naïve binary classifier reports false violations whenever the driver is partly hidden by window glare or A-pillars. Explicitly modelling occlusion removes that failure mode.

### 5.4 Violation Classification & Confidence

Each module emits a softmax probability. Because raw softmax scores are poorly calibrated, we apply **temperature scaling** fitted on the validation set so that a reported "90% confidence" genuinely corresponds to ~90% empirical correctness. Violations below a configurable confidence threshold are routed to a **human-review queue** rather than auto-penalised — the system augments officers, it does not replace their judgement.

### 5.5 License Plate Recognition

A lightweight **YOLOv8n** localises the plate within each vehicle crop; **PaddleOCR (PP-OCRv4)** reads the characters. We then apply an **India-specific correction layer**:

- Format validation against the standard pattern `[A-Z]{2} [0-9]{2} [A-Z]{1,2} [0-9]{4}` (e.g. `MH 01 AB 1234`).
- Confusion-aware correction of common OCR errors (`0↔O`, `1↔I`, `8↔B`, `5↔S`) using the known position of letters vs digits in the plate.

This domain adaptation recovers plates that generic OCR returns malformed.

### 5.6 Evidence Generation

For every confirmed violation the system produces a tamper-evident evidence package:

- **Annotated image** — colour-coded boxes (vehicle, violation region, plate), violation labels with confidence, and a metadata banner.
- **Metadata sidecar (JSON)** — timestamp, location/camera ID, vehicle type, violation types, plate number and confidence, track ID, and pipeline latency.
- Both are persisted to the database and object store for retrieval, export, and audit.

### 5.7 Analytics & Reporting

A web dashboard surfaces:

- Live annotated feed and real-time per-type violation counters.
- A searchable, filterable violations table (by date, type, plate, confidence) with one-click evidence view and CSV export.
- Trend charts (violations over time, by type, by vehicle class) and a temporal heatmap of hotspots.
- **Fuzzy plate search** that ranks partial matches, returning every incident linked to a plate.

---

## 6. Key Innovations

What distinguishes TRACE from a standard detector demo:

1. **Quality-adaptive preprocessing.** Per-frame diagnosis routes only the frames that need help to the right restoration model — faster on average and more robust on the hard cases that decide accuracy in the field.
2. **Temporal reasoning via tracking.** Persistent IDs turn single-frame guesses into multi-frame decisions. A 5-frame confirmation window cuts false positives substantially, and inherently sequential violations (triple riding, parking duration, red-light + stop-line) become tractable.
3. **Calibrated confidence + human-in-the-loop.** Temperature-scaled scores and a review queue make the system's output trustworthy and deployable, not just a leaderboard number.
4. **Domain-adapted Indian-plate OCR.** Format-aware, confusion-aware correction meaningfully lifts real plate-read accuracy.
5. **Edge-ready.** Models export to ONNX/TensorRT, enabling deployment on a Jetson-class device at the camera rather than a costly central GPU farm.

---

## 7. Technology Stack

| Layer | Choice |
|---|---|
| Detection | YOLOv10 (vehicles/people), YOLOv8n (plates) |
| Tracking | ByteTrack |
| Classifiers | EfficientNet-B2 (helmet), MobileNetV3 (seatbelt) |
| Preprocessing | Zero-DCE++, DeblurGAN-v2, Dark Channel Prior, OpenCV/CLAHE |
| OCR | PaddleOCR PP-OCRv4 + custom Indian-plate corrector |
| Training | PyTorch, Ultralytics, Albumentations, Weights & Biases |
| Backend | FastAPI, SQLAlchemy, WebSocket |
| Storage | PostgreSQL / SQLite, object store for evidence |
| Dashboard | React + Vite + Tailwind, Recharts, Leaflet |
| Packaging | Docker Compose, ONNX / TensorRT export |

---

## 8. Evaluation & Expected Performance

We evaluate every stage on a held-out test split, plus an end-to-end test on ~200 fully annotated images where predicted violations are matched to ground truth by IoU > 0.5.

| Task | Metric | Target |
|---|---|---|
| Vehicle detection | mAP@0.5 | > 0.80 |
| Helmet compliance | F1 | > 0.88 |
| Seatbelt compliance | F1 | > 0.82 |
| Triple riding | F1 | > 0.85 |
| Plate detection | mAP@0.5 | > 0.90 |
| Plate OCR | Exact-match | > 0.75 |
| Overall violations | F1 | > 0.82 |
| Throughput | FPS | 15 (GPU) / 5 (CPU) |

Reported metrics: **Accuracy, Precision, Recall, F1-score, mAP@0.5, mAP@0.5:0.95**, OCR character-error-rate and exact-match, plus per-class confusion matrices and computational-efficiency benchmarks.

---

## 9. Implementation Roadmap

**Data strategy.** Public datasets (IDD, COCO, Roboflow helmet/seatbelt/plate sets) augmented with synthetic data — weather/blur augmentation via Albumentations and copy-paste / generative augmentation for rare classes (e.g. night-time triple riding) — to overcome class scarcity.

| Phase | Focus | Outcome |
|---|---|---|
| **1 — Foundation** | Repo, preprocessing, YOLOv10 inference | Detections on real footage |
| **2 — Core slice** | Tracking + helmet + plate OCR end-to-end | First demoable pipeline |
| **3 — Breadth** | Remaining violation modules | All 7 violations |
| **4 — Product** | FastAPI + dashboard + database | Searchable evidence & analytics |
| **5 — Evaluation** | Metrics, calibration, benchmarks | Numbers for the report |

Wrong-side and parking detection require one-time per-camera calibration and are sequenced last as enhancements, so they never block the core demo.

---

## 10. Impact, Scalability & Deployment

- **Efficiency.** Automated first-pass screening converts an unreviewable volume of footage into a prioritised, evidence-backed queue, freeing officers for adjudication and field work.
- **Consistency & fairness.** Uniform, calibrated criteria applied to every frame remove reviewer-to-reviewer variability.
- **Scalability.** A stateless, containerised pipeline scales horizontally; edge export pushes inference to the camera, reducing bandwidth and central compute.
- **Auditability.** Every decision ships with annotated evidence, confidence, and metadata — defensible if challenged.
- **Safety outcome.** Faster, more consistent enforcement of helmet, seatbelt, and signal compliance targets exactly the behaviours most strongly linked to fatalities.

---

## 11. Conclusion

TRACE reframes traffic enforcement from a manual review bottleneck into an automated, auditable, and scalable screening pipeline. By pairing strong detection with the features that matter in the real world — adaptive preprocessing, temporal reasoning, calibrated confidence, and domain-adapted plate recognition — it delivers accuracy that survives outside the lab and evidence that stands up to scrutiny. The architecture is modular, edge-deployable, and measurable against the exact metrics the challenge specifies, making it both a compelling prototype and a credible path to real-world deployment.

---

### Appendix — Primary Data Sources

- **IDD** — India Driving Dataset (Indian road scenes)
- **COCO** — base vehicle/person classes
- **Roboflow Universe** — helmet, seatbelt, and license-plate datasets
- **CCPD / UFPR-ALPR** — license-plate detection and recognition benchmarks
- Synthetic augmentation — Albumentations (weather/blur), copy-paste and generative methods for rare classes
