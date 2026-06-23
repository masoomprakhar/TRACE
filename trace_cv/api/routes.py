"""REST routes under /api.

Contract (consumed by the dashboard):
  GET  /api/health
  POST /api/analyze                      (multipart 'file')
  GET  /api/violations                   (?type=&plate=&limit=&offset=)
  GET  /api/violations/{id}
  GET  /api/violations/{id}/evidence     -> image
  GET  /api/events/{event_id}/evidence   -> image
  GET  /api/analytics/summary
  GET  /api/plates/search                (?q=)
  POST /api/live/reset
  POST /api/live/frame                   (multipart, ?tracking=&persist=)
  GET  /api/live/stream                  (?source=0 or ?url=rtsp://...)
"""

from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response

from trace_cv import __version__
from trace_cv.api.deps import get_pipeline, get_repo, get_settings

router = APIRouter(prefix="/api")

_ID_RE = re.compile(r"^[A-Za-z0-9_\-]+$")


@router.get("/health")
def health(full: bool = Query(False)) -> dict:
    """Liveness probe — fast by default for Render/load balancers.

    Pass ``?full=1`` (Settings page) to include lazy-loaded model status.
    """
    out = {"status": "ok", "version": __version__}
    if full:
        out["models"] = get_pipeline().model_status()
    return out


@router.post("/analyze")
async def analyze(file: UploadFile = File(...)) -> dict:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty upload")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="could not decode image")

    try:
        result = get_pipeline().process_image(img, location="Upload")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"analysis failed: {exc}") from exc
    result.pop("annotated", None)
    result.pop("records", None)
    result.pop("evidence_path", None)
    return result


@router.get("/violations")
def list_violations(
    type: Optional[str] = Query(default=None),
    plate: Optional[str] = Query(default=None),
    limit: int = Query(default=20, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> dict:
    return get_repo().list(vtype=type, plate=plate, limit=limit, offset=offset)


@router.get("/violations.csv")
def violations_csv(
    type: Optional[str] = Query(default=None),
    plate: Optional[str] = Query(default=None),
):
    """Export violation records as CSV (downloadable report). Honors the same
    type/plate filters as the table; exports everything when unfiltered."""
    if type or plate:
        records = get_repo().list(vtype=type, plate=plate, limit=100000)["items"]
    else:
        records = get_repo().all_records()
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        ["id", "timestamp", "location", "vehicle_type", "violation_types",
         "confidence", "plate_number", "plate_confidence"]
    )
    for r in records:
        writer.writerow(
            [r["id"], r["timestamp"], r["location"], r["vehicle_type"],
             "|".join(r["violation_types"]), r["confidence"],
             r["plate_number"] or "", r["plate_confidence"]]
        )
    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=trace_violations.csv"},
    )


@router.get("/report/summary")
def report_summary() -> dict:
    """A human-readable summary report plus the raw analytics."""
    s = get_repo().summary()
    lines = [
        "TRACE — Violation Summary Report",
        "=" * 34,
        f"Total violations : {s['total']}",
        f"Avg confidence   : {s['avg_confidence']}",
        f"Processing speed : {s['processing_fps']} FPS",
        "",
        "By violation type:",
    ]
    for k, v in sorted(s["by_type"].items(), key=lambda x: -x[1]):
        lines.append(f"  {k:<18} {v}")
    lines.append("")
    lines.append("Top offending plates:")
    for p in s["top_plates"]:
        lines.append(f"  {p['plate']:<16} {p['count']}")
    return {"report": "\n".join(lines), "summary": s}


@router.get("/violations/{record_id}")
def get_violation(record_id: str) -> dict:
    row = get_repo().get(record_id)
    if row is None:
        raise HTTPException(status_code=404, detail="violation not found")
    return row.to_record()


@router.get("/violations/{record_id}/evidence")
def violation_evidence(record_id: str):
    row = get_repo().get(record_id)
    if row is None or not row.evidence_path or not Path(row.evidence_path).exists():
        raise HTTPException(status_code=404, detail="evidence not found")
    return FileResponse(row.evidence_path, media_type="image/jpeg")


@router.get("/events/{event_id}/evidence")
def event_evidence(event_id: str):
    if not _ID_RE.match(event_id):
        raise HTTPException(status_code=400, detail="invalid event id")
    path = Path(get_settings().storage_dir) / f"{event_id}.jpg"
    if not path.exists():
        raise HTTPException(status_code=404, detail="evidence not found")
    return FileResponse(str(path), media_type="image/jpeg")


@router.get("/analytics/summary")
def analytics_summary() -> dict:
    return get_repo().summary()


@router.get("/eval/summary")
def eval_summary() -> dict:
    """Latest offline evaluation metrics (mAP, F1, OCR) for the dashboard."""
    path = Path(__file__).resolve().parents[2] / "data" / "eval" / "eval-summary.json"
    if not path.exists():
        # Fall back to full results.json if summary not generated yet.
        full = path.parent / "results.json"
        if full.exists():
            from trace_cv.evaluation.summary_export import build_eval_summary

            return build_eval_summary(json.loads(full.read_text()))
        return {"metrics": {}, "note": "Run scripts/run_full_eval.py to generate metrics."}
    return json.loads(path.read_text())


@router.get("/plates/search")
def plates_search(q: str = Query(default="")) -> dict:
    return get_repo().plate_search(q)
