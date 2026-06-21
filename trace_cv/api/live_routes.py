"""Live camera routes — MJPEG CCTV proxy + per-frame analysis with tracking."""

from __future__ import annotations

import base64
import threading
import time
from typing import Iterator

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse

from trace_cv.api.deps import get_pipeline
from trace_cv.core.logging import get_logger

log = get_logger("live")

router = APIRouter(prefix="/api/live")

# One OpenCV capture per stream key (url or device index).
_CAPTURES: dict[str, cv2.VideoCapture] = {}
_CAPTURE_LOCK = threading.Lock()


def _stream_key(source: str | None, url: str | None) -> str:
    if url:
        return f"url:{url.strip()}"
    return f"dev:{(source or '0').strip()}"


def _open_capture(source: str | None, url: str | None) -> cv2.VideoCapture:
    key = _stream_key(source, url)
    with _CAPTURE_LOCK:
        cap = _CAPTURES.get(key)
        if cap is not None and cap.isOpened():
            return cap
        if url:
            cap = cv2.VideoCapture(url.strip())
        else:
            src = (source or "0").strip()
            cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
        if not cap.isOpened():
            raise HTTPException(
                status_code=400,
                detail="Could not open camera stream. Check the URL or device index.",
            )
        _CAPTURES[key] = cap
        return cap


def _close_all_captures() -> None:
    with _CAPTURE_LOCK:
        for cap in _CAPTURES.values():
            try:
                cap.release()
            except Exception:
                pass
        _CAPTURES.clear()


def _mjpeg_frames(source: str | None, url: str | None) -> Iterator[bytes]:
    cap = _open_capture(source, url)
    boundary = b"--frame"
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok:
                continue
            chunk = buf.tobytes()
            yield (
                boundary
                + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
                + str(len(chunk)).encode()
                + b"\r\n\r\n"
                + chunk
                + b"\r\n"
            )
            time.sleep(0.04)  # ~25 fps cap for bandwidth
    except GeneratorExit:
        pass


@router.get("/stream")
def live_stream(
    source: str = Query(default="0", description="Local device index (0 = default webcam)"),
    url: str | None = Query(default=None, description="RTSP or HTTP CCTV URL"),
):
    """MJPEG proxy for CCTV feeds browsers cannot play natively (RTSP, etc.)."""
    return StreamingResponse(
        _mjpeg_frames(source, url),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store", "Pragma": "no-cache"},
    )


@router.post("/reset")
def live_reset() -> dict:
    """Reset tracker + violation temporal state before a new live session."""
    get_pipeline().reset_live_session()
    return {"status": "ok"}


@router.post("/frame")
async def live_frame(
    file: UploadFile = File(...),
    location: str = Query(default="Live-Camera"),
    tracking: bool = Query(default=True),
    persist: bool = Query(default=False),
    preview: bool = Query(default=True, description="Include base64 annotated JPEG"),
) -> dict:
    """Analyze one live frame with optional multi-frame tracking."""
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty frame")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="could not decode frame")

    result = get_pipeline().process_image(
        img,
        location=location,
        persist=persist,
        use_tracking=tracking,
    )
    annotated = result.pop("annotated", None)
    result.pop("records", None)
    result.pop("evidence_path", None)
    if preview and annotated is not None:
        ok, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if ok:
            b64 = base64.b64encode(buf.tobytes()).decode("ascii")
            result["annotated_preview"] = f"data:image/jpeg;base64,{b64}"
    return result
