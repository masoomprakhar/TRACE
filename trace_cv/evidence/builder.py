"""Compose evidence packages.

For each event (one image/frame) we save one annotated image and emit one
record per offending vehicle (violations sharing a track / box are grouped),
so a vehicle that both runs a red light and carries no helmet is a single
reviewable record listing both.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from trace_cv.core.logging import get_logger
from trace_cv.core.types import Detection, Violation, ViolationType, utcnow
from trace_cv.evidence.annotator import add_banner, annotate

log = get_logger("evidence")


def _group_key(v: Violation):
    if v.track_id is not None:
        return ("track", v.track_id)
    return ("box", tuple(round(c) for c in v.bbox))


def build_records(
    violations: list[Violation],
    *,
    event_id: str,
    timestamp: datetime,
    location: str,
    evidence_path: str,
    processing_ms: float = 0.0,
) -> list[dict]:
    """Group violations per vehicle into reviewable records."""
    groups: dict = {}
    for v in violations:
        groups.setdefault(_group_key(v), []).append(v)

    records: list[dict] = []
    for i, (_, group) in enumerate(groups.items()):
        types = []
        for v in group:  # preserve order, dedupe
            if v.type.value not in types:
                types.append(v.type.value)
        plate = next((v.plate for v in group if v.plate and v.plate.text), None)
        record = {
            "id": f"{event_id}_{i}",
            "event_id": event_id,
            "timestamp": timestamp,
            "location": location,
            "vehicle_type": group[0].vehicle_class or "unknown",
            "track_id": group[0].track_id,
            "violation_types": types,
            "violation_label": ", ".join(ViolationType(t).label for t in types),
            "confidence": round(max(v.confidence for v in group), 4),
            "plate_number": plate.text if plate else None,
            "plate_confidence": round(plate.confidence, 4) if plate else 0.0,
            "evidence_path": evidence_path,
            "bbox": [round(c, 1) for c in group[0].bbox],
            "detail": {v.type.value: v.detail for v in group},
            "processing_ms": round(processing_ms, 2),
        }
        records.append(record)
    return records


class EvidenceBuilder:
    def __init__(self, storage_dir: str = "data/output"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def build(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        violations: list[Violation],
        *,
        location: str = "Camera-01",
        event_id: Optional[str] = None,
        processing_ms: float = 0.0,
        timestamp: Optional[datetime] = None,
        quality: Optional[dict] = None,
    ) -> dict:
        event_id = event_id or f"evt_{uuid.uuid4().hex[:12]}"
        timestamp = timestamp or utcnow()

        annotated = annotate(frame, detections, violations)
        footer = [
            event_id,
            timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
            location,
            f"{len(violations)} violation(s)",
        ]
        if quality and quality.get("applied"):
            footer.append("pre: " + ",".join(quality["applied"]))
        annotated = add_banner(annotated, "TRACE - Traffic Violation Evidence", footer)

        img_name = f"{event_id}.jpg"
        img_path = self.storage_dir / img_name
        cv2.imwrite(str(img_path), annotated)

        records = build_records(
            violations,
            event_id=event_id,
            timestamp=timestamp,
            location=location,
            evidence_path=str(img_path),
            processing_ms=processing_ms,
        )

        sidecar = {
            "event_id": event_id,
            "timestamp": timestamp.isoformat(),
            "location": location,
            "processing_ms": round(processing_ms, 2),
            "quality": quality,
            "detections": [d.to_dict() for d in detections],
            "violations": [v.to_dict() for v in violations],
            "records": [{**r, "timestamp": r["timestamp"].isoformat()} for r in records],
        }
        (self.storage_dir / f"{event_id}.json").write_text(json.dumps(sidecar, indent=2))

        return {
            "event_id": event_id,
            "timestamp": timestamp,
            "evidence_path": str(img_path),
            "annotated": annotated,
            "records": records,
        }
