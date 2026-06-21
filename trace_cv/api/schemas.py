"""Pydantic response schemas (used where the shape is simple; richer
endpoints return plain dicts already shaped to the documented contract)."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class ModelStatus(BaseModel):
    detector: bool
    ocr: bool
    helmet: bool
    seatbelt: bool
    plate: bool


class HealthResponse(BaseModel):
    status: str
    version: str
    models: ModelStatus


class TopPlate(BaseModel):
    plate: str
    count: int


class AnalyticsSummary(BaseModel):
    total: int
    by_type: dict[str, int]
    by_hour: dict[str, int]
    by_vehicle: dict[str, int]
    top_plates: list[TopPlate]
    avg_confidence: float
    processing_fps: float


class ViolationRecord(BaseModel):
    id: str
    event_id: str
    timestamp: Optional[str]
    location: str
    vehicle_type: str
    track_id: Optional[int] = None
    violation_types: list[str]
    violation_label: str
    confidence: float
    plate_number: Optional[str] = None
    plate_confidence: float
    evidence_url: str
    detail: dict = {}


class ViolationList(BaseModel):
    total: int
    items: list[ViolationRecord]
