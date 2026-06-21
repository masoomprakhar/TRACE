"""Repository over the violations table: insert, query, analytics, search.

Aggregations expand the per-record `violation_types` list in Python, which is
simple and plenty fast at hackathon/demo scale.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker

from trace_cv.core.logging import get_logger
from trace_cv.ocr.corrector import normalize_plate, plate_similarity
from trace_cv.storage.models import Base, ViolationRow

log = get_logger("storage")


class Repository:
    def __init__(self, db_url: str = "sqlite:///data/trace.db"):
        if db_url.startswith("sqlite:///"):
            Path(db_url.replace("sqlite:///", "")).parent.mkdir(
                parents=True, exist_ok=True
            )
        self.engine = create_engine(db_url, future=True)
        self._Session = sessionmaker(bind=self.engine, future=True)
        Base.metadata.create_all(self.engine)

    # -- writes -------------------------------------------------------------
    def add_records(self, records: list[dict]) -> int:
        if not records:
            return 0
        with self._Session() as s:
            for r in records:
                s.merge(
                    ViolationRow(
                        id=r["id"],
                        event_id=r["event_id"],
                        timestamp=r["timestamp"],
                        location=r.get("location", "Camera-01"),
                        vehicle_type=r.get("vehicle_type", "unknown"),
                        track_id=r.get("track_id"),
                        violation_types=r.get("violation_types", []),
                        confidence=r.get("confidence", 0.0),
                        plate_number=r.get("plate_number"),
                        plate_confidence=r.get("plate_confidence", 0.0),
                        evidence_path=r.get("evidence_path", ""),
                        processing_ms=r.get("processing_ms", 0.0),
                        detail=r.get("detail", {}),
                    )
                )
            s.commit()
        return len(records)

    # -- reads --------------------------------------------------------------
    def list(
        self,
        vtype: Optional[str] = None,
        plate: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> dict:
        with self._Session() as s:
            stmt = select(ViolationRow)
            if plate:
                stmt = stmt.where(
                    ViolationRow.plate_number.ilike(f"%{plate.upper()}%")
                )
            rows = s.scalars(stmt.order_by(ViolationRow.timestamp.desc())).all()

        # type filter expands the JSON list, so apply in Python.
        if vtype:
            rows = [r for r in rows if vtype in (r.violation_types or [])]

        total = len(rows)
        page = rows[offset : offset + limit]
        return {"total": total, "items": [r.to_record() for r in page]}

    def all_records(self) -> list[dict]:
        """Every record (newest first) — used for CSV export / reports."""
        with self._Session() as s:
            rows = s.scalars(
                select(ViolationRow).order_by(ViolationRow.timestamp.desc())
            ).all()
        return [r.to_record() for r in rows]

    def get(self, record_id: str) -> Optional[ViolationRow]:
        with self._Session() as s:
            return s.get(ViolationRow, record_id)

    def count(self) -> int:
        with self._Session() as s:
            return int(s.scalar(select(func.count()).select_from(ViolationRow)) or 0)

    # -- analytics ----------------------------------------------------------
    def summary(self) -> dict:
        with self._Session() as s:
            rows = s.scalars(select(ViolationRow)).all()

        by_type: Counter = Counter()
        by_hour: Counter = Counter()
        by_vehicle: Counter = Counter()
        plates: Counter = Counter()
        confs: list[float] = []
        proc: list[float] = []

        for r in rows:
            for t in r.violation_types or []:
                by_type[t] += 1
            if r.timestamp:
                by_hour[r.timestamp.hour] += 1
            by_vehicle[r.vehicle_type or "unknown"] += 1
            if r.plate_number:
                plates[r.plate_number] += 1
            confs.append(r.confidence)
            if r.processing_ms:
                proc.append(r.processing_ms)

        avg_ms = sum(proc) / len(proc) if proc else 0.0
        fps = (1000.0 / avg_ms) if avg_ms > 0 else 0.0
        return {
            "total": len(rows),
            "by_type": dict(by_type),
            "by_hour": {str(h): by_hour.get(h, 0) for h in range(24)},
            "by_vehicle": dict(by_vehicle),
            "top_plates": [
                {"plate": p, "count": c} for p, c in plates.most_common(10)
            ],
            "avg_confidence": round(sum(confs) / len(confs), 4) if confs else 0.0,
            "processing_fps": round(fps, 2),
        }

    def plate_search(self, query: str, limit: int = 20) -> dict:
        q = normalize_plate(query)
        with self._Session() as s:
            rows = s.scalars(
                select(ViolationRow).where(ViolationRow.plate_number.isnot(None))
            ).all()

        agg: dict[str, dict] = {}
        for r in rows:
            plate = r.plate_number
            sim = plate_similarity(q, plate) if q else 1.0
            contains = q in normalize_plate(plate) if q else True
            if not (contains or sim >= 0.6):
                continue
            entry = agg.setdefault(
                plate,
                {
                    "plate": plate,
                    "count": 0,
                    "last_seen": None,
                    "violations": set(),
                    "score": 0.0,
                },
            )
            entry["count"] += 1
            entry["score"] = max(entry["score"], 1.0 if contains else sim)
            for t in r.violation_types or []:
                entry["violations"].add(t)
            ts = r.timestamp.isoformat() if r.timestamp else None
            if ts and (entry["last_seen"] is None or ts > entry["last_seen"]):
                entry["last_seen"] = ts

        items = sorted(agg.values(), key=lambda e: (-e["score"], -e["count"]))[:limit]
        for e in items:
            e["violations"] = sorted(e["violations"])
            e.pop("score", None)
        return {"items": items}
