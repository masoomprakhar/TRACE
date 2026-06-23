import os

import cv2
import numpy as np
import pytest

pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402


def _client(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "device: cpu\n"
        f"storage_dir: {tmp_path}/out\n"
        f"db_url: sqlite:///{tmp_path}/t.db\n"
        "models:\n  detector: yolov8n.pt\n  ocr_langs: [en]\n"
    )
    os.environ["TRACE_CONFIG"] = str(cfg)
    import trace_cv.api.deps as deps

    deps._pipeline = None
    deps._settings = None
    from trace_cv.api.main import create_app

    return TestClient(create_app()), deps


def test_health(tmp_path):
    client, _ = _client(tmp_path)
    r = client.get("/api/health?full=1")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "detector" in body["models"]


def test_violations_and_analytics(tmp_path):
    client, deps = _client(tmp_path)
    from trace_cv.demo import seed_demo

    seed_demo(deps.get_settings(), n=6)

    r = client.get("/api/violations?limit=3")
    body = r.json()
    assert body["total"] >= 1
    assert len(body["items"]) <= 3

    r = client.get("/api/analytics/summary")
    assert "by_type" in r.json()


def test_analyze_empty_image(tmp_path):
    client, _ = _client(tmp_path)
    img = np.zeros((64, 64, 3), np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    r = client.post(
        "/api/analyze",
        files={"file": ("x.jpg", buf.tobytes(), "image/jpeg")},
    )
    assert r.status_code == 200
    body = r.json()
    assert "violations" in body and "detections" in body


def test_csv_export(tmp_path):
    client, deps = _client(tmp_path)
    from trace_cv.demo import seed_demo

    seed_demo(deps.get_settings(), n=4)
    r = client.get("/api/violations.csv")
    assert r.status_code == 200
    assert "text/csv" in r.headers["content-type"]
    assert r.text.splitlines()[0].startswith("id,timestamp")
    assert len(r.text.splitlines()) >= 2  # header + at least one row


def test_csv_export_filtered(tmp_path):
    client, deps = _client(tmp_path)
    from trace_cv.demo import seed_demo

    seed_demo(deps.get_settings(), n=10)
    r = client.get("/api/violations.csv?type=red_light")
    assert r.status_code == 200
    rows = r.text.splitlines()[1:]  # drop header
    # filtered export: every returned row carries the requested type
    assert all("red_light" in row for row in rows)


def test_report_summary(tmp_path):
    client, deps = _client(tmp_path)
    from trace_cv.demo import seed_demo

    seed_demo(deps.get_settings(), n=4)
    r = client.get("/api/report/summary")
    assert r.status_code == 200
    body = r.json()
    assert "report" in body and "summary" in body
    assert "Violation Summary Report" in body["report"]


def test_live_reset_and_frame(tmp_path):
    client, _ = _client(tmp_path)
    r = client.post("/api/live/reset")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

    img = np.zeros((64, 64, 3), np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    r = client.post(
        "/api/live/frame?tracking=true&persist=false&preview=true",
        files={"file": ("frame.jpg", buf.tobytes(), "image/jpeg")},
    )
    assert r.status_code == 200
    body = r.json()
    assert "violations" in body and "detections" in body
