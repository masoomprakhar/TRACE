"""Draw annotated evidence: faint boxes for all detections, bold colour-coded
boxes + labels for violations, and a metadata banner. Colours match the
dashboard's per-violation palette."""

from __future__ import annotations

import cv2
import numpy as np

from trace_cv.core.types import Detection, Violation, ViolationType

# Hex palette (shared with the dashboard), keyed by violation type.
_HEX = {
    ViolationType.NO_HELMET: "f5b301",
    ViolationType.NO_SEATBELT: "f59e0b",
    ViolationType.TRIPLE_RIDING: "ef4444",
    ViolationType.WRONG_SIDE: "a855f7",
    ViolationType.STOP_LINE: "06b6d4",
    ViolationType.RED_LIGHT: "dc2626",
    ViolationType.ILLEGAL_PARKING: "3b82f6",
}


def _hex_to_bgr(h: str) -> tuple[int, int, int]:
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def violation_bgr(vtype: ViolationType) -> tuple[int, int, int]:
    return _hex_to_bgr(_HEX.get(vtype, "ef4444"))


def _label(img, text, org, color, scale=0.6, thickness=1):
    """Text with a filled background box for legibility."""
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = int(org[0]), int(org[1])
    y = max(y, th + 4)
    cv2.rectangle(img, (x, y - th - base - 2), (x + tw + 6, y + 2), color, -1)
    cv2.putText(
        img, text, (x + 3, y - 2),
        cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA,
    )


def annotate(
    frame: np.ndarray,
    detections: list[Detection],
    violations: list[Violation],
    draw_detections: bool = True,
) -> np.ndarray:
    out = frame.copy()

    if draw_detections:
        for d in detections:
            x1, y1, x2, y2 = (int(v) for v in d.bbox)
            cv2.rectangle(out, (x1, y1), (x2, y2), (150, 150, 150), 1)

    for v in violations:
        color = violation_bgr(v.type)
        x1, y1, x2, y2 = (int(c) for c in v.bbox)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
        text = f"{v.type.label} {int(round(v.confidence * 100))}%"
        if v.plate and v.plate.text:
            text += f" | {v.plate.text}"
        _label(out, text, (x1, y1 - 4), color)

    return out


def add_banner(img: np.ndarray, title: str, lines: list[str]) -> np.ndarray:
    """Semi-transparent header (title) + footer (metadata) banners."""
    out = img.copy()
    h, w = out.shape[:2]
    navy = (63, 36, 22)  # #16243f in BGR

    # Header
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, 34), navy, -1)
    cv2.addWeighted(overlay, 0.75, out, 0.25, 0, out)
    cv2.putText(out, title, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (1, 179, 245), 2, cv2.LINE_AA)  # amber #f5b301

    # Footer
    footer = " | ".join(lines)
    overlay = out.copy()
    cv2.rectangle(overlay, (0, h - 26), (w, h), navy, -1)
    cv2.addWeighted(overlay, 0.75, out, 0.25, 0, out)
    cv2.putText(out, footer, (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (231, 237, 247), 1, cv2.LINE_AA)
    return out
