"""ORM model for a stored violation record (one per offending vehicle)."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import JSON, DateTime, Float, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from trace_cv.core.types import ViolationType


class Base(DeclarativeBase):
    pass


class ViolationRow(Base):
    __tablename__ = "violations"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    event_id: Mapped[str] = mapped_column(String, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True)
    location: Mapped[str] = mapped_column(String, default="Camera-01")
    vehicle_type: Mapped[str] = mapped_column(String, default="unknown")
    track_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    violation_types: Mapped[list] = mapped_column(JSON, default=list)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    plate_number: Mapped[Optional[str]] = mapped_column(String, nullable=True, index=True)
    plate_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    evidence_path: Mapped[str] = mapped_column(String, default="")
    processing_ms: Mapped[float] = mapped_column(Float, default=0.0)
    detail: Mapped[dict] = mapped_column(JSON, default=dict)

    def to_record(self) -> dict:
        types = self.violation_types or []
        return {
            "id": self.id,
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "location": self.location,
            "vehicle_type": self.vehicle_type,
            "track_id": self.track_id,
            "violation_types": types,
            "violation_label": ", ".join(
                _safe_label(t) for t in types
            ),
            "confidence": round(self.confidence, 4),
            "plate_number": self.plate_number,
            "plate_confidence": round(self.plate_confidence, 4),
            "evidence_url": f"/api/violations/{self.id}/evidence",
            "detail": self.detail or {},
        }


def _safe_label(t: str) -> str:
    try:
        return ViolationType(t).label
    except ValueError:
        return t
