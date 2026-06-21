"""Metrics: Precision, Recall, F1, Accuracy, multi-label report, detection
mAP@0.5 and mAP@0.5:0.95, and OCR CER / exact-match.

Pure Python (uses the geometry helpers) so it runs without the ML stack.
"""

from __future__ import annotations

from dataclasses import dataclass

from trace_cv.core.types import BBox, bbox_iou
from trace_cv.ocr.corrector import normalize_plate


@dataclass
class PRF:
    precision: float
    recall: float
    f1: float
    accuracy: float
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def to_dict(self) -> dict:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "accuracy": round(self.accuracy, 4),
            "tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn,
        }


def binary_prf(tp: int, fp: int, fn: int, tn: int = 0) -> PRF:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total else 0.0
    return PRF(precision, recall, f1, accuracy, tp, fp, fn, tn)


def multilabel_report(
    y_true: list[set], y_pred: list[set], labels: list[str]
) -> dict:
    """Per-label PRF plus micro/macro averages for multi-label predictions
    (each sample is a set of active labels)."""
    per_label: dict[str, PRF] = {}
    micro_tp = micro_fp = micro_fn = 0
    for label in labels:
        tp = fp = fn = tn = 0
        for gt, pr in zip(y_true, y_pred):
            in_gt, in_pr = label in gt, label in pr
            if in_gt and in_pr:
                tp += 1
            elif in_pr and not in_gt:
                fp += 1
            elif in_gt and not in_pr:
                fn += 1
            else:
                tn += 1
        per_label[label] = binary_prf(tp, fp, fn, tn)
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn

    macro_f1 = sum(p.f1 for p in per_label.values()) / len(labels) if labels else 0.0
    macro_p = sum(p.precision for p in per_label.values()) / len(labels) if labels else 0.0
    macro_r = sum(p.recall for p in per_label.values()) / len(labels) if labels else 0.0
    micro = binary_prf(micro_tp, micro_fp, micro_fn)
    return {
        "per_label": {k: v.to_dict() for k, v in per_label.items()},
        "macro": {
            "precision": round(macro_p, 4),
            "recall": round(macro_r, 4),
            "f1": round(macro_f1, 4),
        },
        "micro": micro.to_dict(),
    }


def confusion_matrix(
    y_true: list[str], y_pred: list[str], labels: list[str]
) -> dict:
    index = {l: i for i, l in enumerate(labels)}
    matrix = [[0] * len(labels) for _ in labels]
    for t, p in zip(y_true, y_pred):
        if t in index and p in index:
            matrix[index[t]][index[p]] += 1
    return {"labels": labels, "matrix": matrix}


# --------------------------------------------------------------------------- #
# Detection mAP
# --------------------------------------------------------------------------- #
def _average_precision(recalls: list[float], precisions: list[float]) -> float:
    """All-points (VOC2010+/COCO-style) AP: area under the monotonic
    precision envelope."""
    rec = [0.0] + recalls + [1.0]
    prec = [0.0] + precisions + [0.0]
    for i in range(len(prec) - 2, -1, -1):
        prec[i] = max(prec[i], prec[i + 1])
    ap = 0.0
    for i in range(1, len(rec)):
        if rec[i] != rec[i - 1]:
            ap += (rec[i] - rec[i - 1]) * prec[i]
    return ap


def _ap_at_iou(
    preds: list[tuple[str, BBox, float]],
    gts: list[tuple[str, BBox]],
    cls: str,
    iou_thr: float,
) -> float | None:
    """AP for one class at one IoU threshold. `preds` is flattened across
    images but each prediction/gt is tagged with an image id via closure;
    here we assume single-image matching is pre-grouped by the caller.

    Returns ``None`` when the class is absent from BOTH ground truth and
    predictions — such a class carries no signal and must be *excluded* from
    the mean rather than scored a free 1.0 (which silently inflates mAP).
    A class that has predictions but no ground truth scores 0.0, because every
    one of those predictions is a false positive (hallucinated detection)."""
    cls_preds = sorted(
        [p for p in preds if p[0] == cls], key=lambda x: -x[2]
    )
    cls_gts = [g for g in gts if g[0] == cls]
    n_gt = len(cls_gts)
    if n_gt == 0:
        # No GT, no preds -> class not present in this split: exclude it.
        # No GT but preds exist -> all preds are false positives: AP = 0.
        return 0.0 if cls_preds else None

    matched = [False] * n_gt
    tp = [0] * len(cls_preds)
    fp = [0] * len(cls_preds)
    for i, (_, box, _) in enumerate(cls_preds):
        best_iou, best_j = iou_thr, -1
        for j, (_, gbox) in enumerate(cls_gts):
            if matched[j]:
                continue
            iou = bbox_iou(box, gbox)
            if iou >= best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0:
            matched[best_j] = True
            tp[i] = 1
        else:
            fp[i] = 1

    cum_tp = cum_fp = 0
    recalls, precisions = [], []
    for i in range(len(cls_preds)):
        cum_tp += tp[i]
        cum_fp += fp[i]
        recalls.append(cum_tp / n_gt)
        precisions.append(cum_tp / (cum_tp + cum_fp))
    return _average_precision(recalls, precisions)


def detection_map(
    preds_per_image: list[list[tuple[str, BBox, float]]],
    gts_per_image: list[list[tuple[str, BBox]]],
    labels: list[str],
    iou_thresholds: list[float] | None = None,
) -> dict:
    """mAP@0.5 and mAP@0.5:0.95 over a dataset.

    Matching is done per image (a prediction can only match a gt in the same
    image) by offsetting boxes into disjoint coordinate bands per image.
    """
    if iou_thresholds is None:
        iou_thresholds = [round(0.5 + 0.05 * i, 2) for i in range(10)]

    # Offset each image's boxes so cross-image matches are impossible.
    OFF = 100000.0
    flat_preds: list[tuple[str, BBox, float]] = []
    flat_gts: list[tuple[str, BBox]] = []
    for idx, preds in enumerate(preds_per_image):
        shift = idx * OFF
        for c, b, s in preds:
            flat_preds.append((c, (b[0] + shift, b[1], b[2] + shift, b[3]), s))
    for idx, gts in enumerate(gts_per_image):
        shift = idx * OFF
        for c, b in gts:
            flat_gts.append((c, (b[0] + shift, b[1], b[2] + shift, b[3])))

    # Per-class support (image-level counts) for transparency in reports. A
    # class that never appears in GT cannot legitimately contribute to mAP.
    per_class_support: dict[str, dict] = {}
    for cls in labels:
        n_gt = sum(1 for c, _ in flat_gts if c == cls)
        n_pred = sum(1 for c, _, _ in flat_preds if c == cls)
        per_class_support[cls] = {"n_gt": n_gt, "n_pred": n_pred}

    per_thr: dict[float, float | None] = {}
    per_class_50: dict[str, float] = {}
    for thr in iou_thresholds:
        aps = []
        for cls in labels:
            ap = _ap_at_iou(flat_preds, flat_gts, cls, thr)
            if ap is None:
                continue  # class absent from GT and preds -> excluded
            aps.append(ap)
            if abs(thr - 0.5) < 1e-9:
                per_class_50[cls] = round(ap, 4)
        per_thr[thr] = sum(aps) / len(aps) if aps else None

    map50 = per_thr.get(0.5)
    valid = [v for v in per_thr.values() if v is not None]
    map5095 = sum(valid) / len(valid) if valid else None
    classes_evaluated = sorted(per_class_50.keys())
    return {
        # None (not 0.0) signals "no evaluable class" so it can never be
        # mistaken for a measured-but-poor detector.
        "map50": round(map50, 4) if map50 is not None else None,
        "map5095": round(map5095, 4) if map5095 is not None else None,
        "per_class_ap50": per_class_50,
        "per_class_support": per_class_support,
        "classes_evaluated": classes_evaluated,
        "n_classes_evaluated": len(classes_evaluated),
    }


# --------------------------------------------------------------------------- #
# OCR
# --------------------------------------------------------------------------- #
def _edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[len(b)]


def ocr_cer(pred: str, gt: str) -> float:
    """Character Error Rate on the normalized plate strings.

    Plate CER must ignore spacing/case — otherwise a perfectly read plate
    ``"MH 01 AB 1234"`` scores errors against an unspaced ground truth
    ``"MH01AB1234"``. This normalization mirrors :func:`ocr_exact_match`
    (previously CER did NOT normalize, which silently inflated the metric)."""
    p = normalize_plate(pred)
    g = normalize_plate(gt)
    if not g:
        return 0.0 if not p else 1.0
    return _edit_distance(p, g) / len(g)


def ocr_exact_match(preds: list[str], gts: list[str]) -> float:
    """Fraction of plates read exactly right (ignoring spacing/case)."""
    if not gts:
        return 0.0
    hits = sum(
        1 for p, g in zip(preds, gts) if normalize_plate(p) == normalize_plate(g)
    )
    return hits / len(gts)
