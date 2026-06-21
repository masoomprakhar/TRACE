"""Run TRACE on a labeled holdout manifest and compute metrics."""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2

from trace_cv.core.config import load_settings
from trace_cv.core.types import Detection
from trace_cv.evaluation.dataset import load_manifest, manifest_samples
from trace_cv.evaluation.metrics import detection_map, multilabel_report, ocr_cer, ocr_exact_match
from trace_cv.pipeline import TracePipeline

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_image(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (_REPO_ROOT / p).resolve()


def run_eval(
    config: str | None,
    out: Path | None,
    *,
    persist: bool = False,
    use_gt_detections: bool = False,
) -> dict:
    manifest = load_manifest()
    settings = load_settings(config)
    pipe = TracePipeline(settings)

    y_true: list[set] = []
    y_pred: list[set] = []
    det_gts = []
    det_preds = []
    det_kinds: list[str] = []
    ocr_preds: list[str] = []
    ocr_gts: list[str] = []
    per_sample: list[dict] = []

    t0 = time.perf_counter()
    det_labels = manifest.get("detection_labels", [])
    samples_list = manifest_samples(manifest)
    has_violation_gt = any(sample.get("violations") for sample in samples_list)
    is_idd_only = str(manifest.get("dataset", "")).startswith("idd-lite") and not has_violation_gt

    for sample in samples_list:
        img_path = _resolve_image(sample["image"])
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        enhanced, quality = pipe.pre.process(img)
        eval_kind = sample.get("eval_kind", "")

        # Plate-localization eval uses the plate detector on plate samples.
        if eval_kind == "plate_detection" and pipe.plate_detector and pipe.plate_detector.available:
            plate_dets = pipe.plate_detector.detect(enhanced)
            live_det_list = [d.to_dict() for d in plate_dets]
        else:
            live_dets = pipe.detector.detect(enhanced)
            live_det_list = [d.to_dict() for d in live_dets]

        use_gt = use_gt_detections or eval_kind == "helmet_violation"
        if use_gt:
            detections = [
                Detection(
                    cls=d["cls"],
                    bbox=tuple(d["bbox"]),
                    confidence=float(d.get("confidence", 0.9)),
                )
                for d in sample.get("detections_gt", [])
            ]
            violations = pipe.engine.run(
                enhanced, detections, frame_index=0, fps=pipe.settings.scene.fps
            )
            processing_ms = 0.0
            det_list = [d.to_dict() for d in detections]
            viol_list = [v.to_dict() for v in violations]
        else:
            res = pipe.process_image(
                img, location="Eval", persist=persist, use_tracking=False
            )
            processing_ms = res["processing_ms"]
            det_list = res["detections"]
            viol_list = res["violations"]

        gt_types = set(sample.get("violations", []))
        pred_types = {v["type"] for v in viol_list}
        y_true.append(gt_types)
        y_pred.append(pred_types)

        gt_boxes = [
            (d["cls"], tuple(d["bbox"])) for d in sample.get("detections_gt", [])
        ]
        if eval_kind == "plate_detection":
            gt_boxes = [(cls, bb) for cls, bb in gt_boxes if cls == "license_plate"]
            pred_boxes = [
                (d["cls"], tuple(d["bbox"]), d["confidence"])
                for d in live_det_list
                if d["cls"] == "license_plate"
            ]
        elif eval_kind == "helmet_violation":
            gt_boxes = [(cls, bb) for cls, bb in gt_boxes if cls in ("motorcycle", "person")]
            pred_boxes = [
                (d["cls"], tuple(d["bbox"]), d["confidence"])
                for d in live_det_list
                if d["cls"] in ("motorcycle", "person", "bicycle")
            ]
        else:
            pred_boxes = [
                (d["cls"], tuple(d["bbox"]), d["confidence"]) for d in live_det_list
            ]
        det_gts.append(gt_boxes)
        det_preds.append(pred_boxes)
        det_kinds.append(eval_kind)

        # OCR eval when plate_text is annotated in manifest.
        plate_gt = (sample.get("detail") or {}).get("plate_text")
        if plate_gt and pipe.ocr.available and pipe.plate_detector and pipe.plate_detector.available:
            from trace_cv.detection.roi import crop

            plate_dets = pipe.plate_detector.detect(enhanced)
            if plate_dets:
                best = max(plate_dets, key=lambda d: d.confidence)
                region = crop(enhanced, best.bbox)
                plate = pipe.ocr.read(region, bbox=best.bbox)
                if plate.text:
                    ocr_preds.append(plate.text)
                    ocr_gts.append(plate_gt)

        per_sample.append(
            {
                "id": sample["id"],
                "gt_violations": sorted(gt_types),
                "pred_violations": sorted(pred_types),
                "match": gt_types == pred_types,
                "processing_ms": processing_ms,
                "n_detections": len(det_list),
            }
        )

    labels = manifest.get("violation_labels") or []
    violation_report = (
        multilabel_report(y_true, y_pred, labels)
        if labels and not is_idd_only
        else {
            "note": "No violation ground truth in manifest — add Roboflow helmet images."
        }
    )

    # Detection mAP: split by eval kind when mixed manifest.
    helmet_gts = [g for g, k in zip(det_gts, det_kinds) if k != "plate_detection"]
    helmet_preds = [p for p, k in zip(det_preds, det_kinds) if k != "plate_detection"]
    plate_gts = [g for g, k in zip(det_gts, det_kinds) if k == "plate_detection"]
    plate_preds = [p for p, k in zip(det_preds, det_kinds) if k == "plate_detection"]

    det_metrics = detection_map(helmet_preds, helmet_gts, det_labels)
    plate_det_metrics = (
        detection_map(plate_preds, plate_gts, ["license_plate"])
        if plate_gts
        else {}
    )

    elapsed = round(time.perf_counter() - t0, 2)
    matched = sum(1 for s in per_sample if s["match"])
    accuracy = matched / len(per_sample) if per_sample else 0.0

    results = {
        "config": config or "default",
        "use_gt_detections": use_gt_detections,
        "model_status": pipe.model_status(),
        "n_samples": len(per_sample),
        "exact_match_accuracy": round(accuracy, 4),
        "violation_classification": violation_report,
        "detection": det_metrics,
        "plate_detection": plate_det_metrics,
        "ocr": {
            "exact_match": round(ocr_exact_match(ocr_preds, ocr_gts), 4) if ocr_gts else None,
            "mean_cer": round(
                sum(ocr_cer(p, g) for p, g in zip(ocr_preds, ocr_gts)) / len(ocr_gts), 4
            )
            if ocr_gts
            else None,
            "note": "OCR metrics require plate ground truth in manifest",
        },
        "efficiency": {
            "total_seconds": elapsed,
            "mean_ms_per_frame": round(
                sum(s["processing_ms"] for s in per_sample) / max(len(per_sample), 1), 2
            ),
        },
        "samples": per_sample,
    }

    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2))
    return results
