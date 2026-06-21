from datetime import datetime, timezone

from trace_cv.storage.db import Repository


def _rec(rid, types, plate, ts, vt="car"):
    return {
        "id": rid,
        "event_id": rid.split("_")[0],
        "timestamp": ts,
        "location": "Cam-01",
        "vehicle_type": vt,
        "track_id": None,
        "violation_types": types,
        "confidence": 0.9,
        "plate_number": plate,
        "plate_confidence": 0.8,
        "evidence_path": "",
        "processing_ms": 50.0,
        "detail": {},
    }


def _repo(tmp_path):
    return Repository(f"sqlite:///{tmp_path}/t.db")


def test_add_list_and_filter(tmp_path):
    repo = _repo(tmp_path)
    ts = datetime(2024, 1, 1, 9, 0, tzinfo=timezone.utc)
    repo.add_records(
        [
            _rec("evt1_0", ["red_light"], "MH01AB1234", ts),
            _rec("evt2_0", ["no_helmet", "triple_riding"], "DL05CA0001", ts, "motorcycle"),
        ]
    )
    assert repo.list(limit=10)["total"] == 2
    assert repo.list(vtype="no_helmet")["total"] == 1
    item = repo.list(vtype="red_light")["items"][0]
    assert item["violation_label"] == "Red-Light Violation"


def test_summary_counts(tmp_path):
    repo = _repo(tmp_path)
    ts = datetime(2024, 1, 1, 9, 0, tzinfo=timezone.utc)
    repo.add_records(
        [
            _rec("evt1_0", ["red_light"], "MH01AB1234", ts),
            _rec("evt2_0", ["no_helmet", "triple_riding"], "DL05CA0001", ts, "motorcycle"),
        ]
    )
    s = repo.summary()
    assert s["total"] == 2
    assert s["by_type"]["red_light"] == 1
    assert s["by_type"]["triple_riding"] == 1
    assert s["by_vehicle"]["motorcycle"] == 1
    assert s["processing_fps"] > 0


def test_plate_search(tmp_path):
    repo = _repo(tmp_path)
    ts = datetime(2024, 1, 1, 9, 0, tzinfo=timezone.utc)
    repo.add_records([_rec("evt1_0", ["red_light"], "MH01AB1234", ts)])
    res = repo.plate_search("MH01")
    assert len(res["items"]) == 1
    assert res["items"][0]["plate"] == "MH01AB1234"
    assert "red_light" in res["items"][0]["violations"]
