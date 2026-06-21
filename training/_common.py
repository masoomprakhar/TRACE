"""Shared helpers for the TRACE training scripts.

Kept deliberately small: lazy Ultralytics import with a friendly error,
repo-path resolution, and a "copy best.pt into models/weights/" helper so
every train_*.py lands its checkpoint where the runtime expects it
(see config/default.yaml -> models.*).
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

# repo root = .../Flipkart-grid (this file lives in training/).
REPO_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = REPO_ROOT / "models" / "weights"

_INSTALL_HINT = (
    "Ultralytics is not installed. Install the ML extras first:\n"
    "    pip install -r requirements-ml.txt\n"
    "(or: pip install -e '.[ml]')"
)


def load_yolo():
    """Import and return the Ultralytics ``YOLO`` class (lazy).

    Raised here rather than at module import so the scripts stay importable
    for --help / tooling without the heavy ML stack present.
    """
    try:
        from ultralytics import YOLO  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on env
        print(_INSTALL_HINT, file=sys.stderr)
        raise SystemExit(2) from exc
    return YOLO


def ensure_weights_dir() -> Path:
    """Create models/weights/ if missing and return it."""
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    return WEIGHTS_DIR


def publish_best(run_dir: Path, dest_name: str) -> Path | None:
    """Copy a finished run's ``weights/best.pt`` to models/weights/<dest_name>.

    ``run_dir`` is what Ultralytics returns as ``results.save_dir``. Returns
    the destination path, or None if best.pt could not be found.
    """
    ensure_weights_dir()
    best = Path(run_dir) / "weights" / "best.pt"
    if not best.exists():
        print(f"warning: {best} not found; nothing copied.", file=sys.stderr)
        return None
    dest = WEIGHTS_DIR / dest_name
    shutil.copy2(best, dest)
    return dest


def rel(path: Path) -> str:
    """Path relative to the repo root when possible (nicer log lines)."""
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)
