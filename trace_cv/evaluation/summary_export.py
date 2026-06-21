"""Write dashboard-friendly eval summary from evaluation JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_eval_summary(results: dict[str, Any]) -> dict[str, Any]:
    det = results.get("detection") or {}
    vc = results.get("violation_classification") or {}
    macro = vc.get("macro") or {}
    micro = vc.get("micro") or {}
    ocr = results.get("ocr") or {}
    eff = results.get("efficiency") or {}
    plate_det = results.get("plate_detection") or {}
    per_viol = vc.get("per_label") or {}

    return {
        "source": results.get("config") or "evaluation",
        "n_samples": results.get("n_samples"),
        "model_status": results.get("model_status") or {},
        "metrics": {
            "detection_map50": det.get("map50"),
            "detection_map5095": det.get("map5095"),
            "motorcycle_ap50": (det.get("per_class_ap50") or {}).get("motorcycle"),
            "plate_ap50": (det.get("per_class_ap50") or {}).get("license_plate")
            or plate_det.get("map50"),
            "violation_micro_f1": micro.get("f1"),
            "violation_micro_precision": micro.get("precision"),
            "violation_micro_recall": micro.get("recall"),
            "violation_macro_f1": macro.get("f1"),
            "no_helmet_f1": (per_viol.get("no_helmet") or {}).get("f1"),
            "no_seatbelt_f1": (per_viol.get("no_seatbelt") or {}).get("f1"),
            "ocr_exact_match": ocr.get("exact_match"),
            "ocr_mean_cer": ocr.get("mean_cer"),
            "latency_ms": eff.get("mean_ms_per_frame"),
            "throughput_fps": eff.get("fps"),
        },
        "updated_at": results.get("evaluated_at"),
    }


def write_eval_summary(results: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = build_eval_summary(results)
    path.write_text(json.dumps(summary, indent=2))
    return path
