"""TRACE command-line interface.

  trace detect IMAGE [--config C] [--location L] [--no-persist]
  trace video VIDEO  [--config C] [--sample-every N] [--max-frames M]
  trace serve [--host H] [--port P] [--reload]
  trace seed-demo [-n N] [--config C]
  trace eval
  trace eval-real [--config C] [--rebuild] [--out PATH]
  trace calibrate [--image PATH] [--width W --height H] [--out PATH]
  trace import-idd [--tar PATH] [--split val|train] [--max-images N]
  trace sample [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _cmd_detect(args) -> int:
    import cv2

    from trace_cv.core.config import load_settings
    from trace_cv.pipeline import TracePipeline

    img = cv2.imread(args.image)
    if img is None:
        print(f"error: could not read image: {args.image}", file=sys.stderr)
        return 1
    pipe = TracePipeline(load_settings(args.config))
    print(f"models: {pipe.model_status()}")
    res = pipe.process_image(
        img, location=args.location, persist=not args.no_persist
    )
    print(f"\nEvent {res['id']}  ({res['processing_ms']} ms)")
    print(f"preprocessing applied: {res['quality']['applied'] or 'none'}")
    print(f"detections: {len(res['detections'])}")
    if not res["violations"]:
        print("violations: none")
    for v in res["violations"]:
        plate = f"  plate={v['plate']['text']}" if v.get("plate") else ""
        print(f"  - {v['label']:<22} {int(v['confidence']*100)}%{plate}")
    print(f"\nevidence image: {res['evidence_path']}")
    return 0


def _cmd_video(args) -> int:
    from trace_cv.core.config import load_settings
    from trace_cv.pipeline import TracePipeline

    pipe = TracePipeline(load_settings(args.config))
    summary = pipe.process_video(
        args.video,
        sample_every=args.sample_every,
        max_frames=args.max_frames,
    )
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_serve(args) -> int:
    import uvicorn

    uvicorn.run(
        "trace_cv.api.main:app", host=args.host, port=args.port, reload=args.reload
    )
    return 0


def _cmd_seed(args) -> int:
    from trace_cv.core.config import load_settings
    from trace_cv.demo import seed_demo

    n = seed_demo(load_settings(args.config), n=args.n)
    print(f"seeded {n} violation records.")
    return 0


def _cmd_eval(args) -> int:
    from trace_cv.demo import run_demo_eval

    print(json.dumps(run_demo_eval(), indent=2))
    return 0


def _cmd_eval_real(args) -> int:
    from trace_cv.evaluation.dataset import build_eval_set
    from trace_cv.evaluation.runner import run_eval

    root = Path(__file__).resolve().parents[1]
    if args.rebuild or not (root / "data" / "eval" / "manifest.json").exists():
        build_eval_set(n_per_template=args.per_template)
    out = root / args.out
    results = run_eval(
        args.config, out, persist=args.persist, use_gt_detections=args.gt_detections
    )
    print(json.dumps({k: v for k, v in results.items() if k != "samples"}, indent=2))
    print(f"\nWrote per-sample details to {out}")
    return 0


def _cmd_calibrate(args) -> int:
    import cv2
    import yaml

    from scripts.calibrate_camera import geometry_for_frame

    root = Path(__file__).resolve().parents[1]
    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            print(f"error: cannot read {args.image}", file=sys.stderr)
            return 1
        h, w = img.shape[:2]
    elif args.width and args.height:
        w, h = args.width, args.height
    else:
        from trace_cv.demo import make_synthetic_scene

        ref = root / "data" / "samples" / "junction-01-reference.jpg"
        ref.parent.mkdir(parents=True, exist_ok=True)
        scene = make_synthetic_scene()
        cv2.imwrite(str(ref), scene)
        h, w = scene.shape[:2]
        print(f"wrote reference frame: {ref} ({w}×{h})")
    cfg = geometry_for_frame(w, h, camera_name=args.camera)
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False))
    print(f"calibrated config written: {out}")
    print(f"  stop_line.y      = {cfg['scene']['stop_line']['y']}")
    print(f"  lane.divider_x   = {cfg['scene']['lane']['divider_x']}")
    print(f"  signal.bbox      = {cfg['scene']['signal']['bbox']}")
    return 0


def _cmd_import_idd(args) -> int:
    from trace_cv.evaluation.idd_lite import DEFAULT_ROOT, build_manifest, extract_archive

    root = DEFAULT_ROOT
    tar = Path(args.tar)
    if not root.exists():
        alt = Path.home() / "Downloads" / "idd-lite (1).tar.gz"
        if not tar.exists() and alt.exists():
            tar = alt
        if not tar.exists():
            print(f"error: provide --tar (missing {tar})", file=sys.stderr)
            return 1
        print(f"Extracting {tar} ...")
        extract_archive(tar)
    manifest = build_manifest(root, split=args.split, max_images=args.max_images)
    print(f"imported {manifest['n_samples']} IDD Lite samples -> data/eval/manifest.json")
    return 0


def _cmd_sample(args) -> int:
    import cv2

    from trace_cv.demo import make_synthetic_scene

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), make_synthetic_scene())
    print(f"wrote sample scene: {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="trace", description="TRACE CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("detect", help="analyze a single image")
    d.add_argument("image")
    d.add_argument("--config")
    d.add_argument("--location", default="CLI")
    d.add_argument("--no-persist", action="store_true")
    d.set_defaults(func=_cmd_detect)

    v = sub.add_parser("video", help="analyze a video")
    v.add_argument("video")
    v.add_argument("--config")
    v.add_argument("--sample-every", type=int, default=1)
    v.add_argument("--max-frames", type=int, default=None)
    v.set_defaults(func=_cmd_video)

    s = sub.add_parser("serve", help="run the API + dashboard")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--reload", action="store_true")
    s.set_defaults(func=_cmd_serve)

    sd = sub.add_parser("seed-demo", help="seed the DB with illustrative records")
    sd.add_argument("-n", type=int, default=40)
    sd.add_argument("--config")
    sd.set_defaults(func=_cmd_seed)

    e = sub.add_parser("eval", help="run the evaluation showcase")
    e.set_defaults(func=_cmd_eval)

    er = sub.add_parser("eval-real", help="evaluate pipeline on labeled holdout set")
    er.add_argument("--config")
    er.add_argument("--out", default="data/eval/results.json")
    er.add_argument("--rebuild", action="store_true")
    er.add_argument("--per-template", type=int, default=4)
    er.add_argument("--gt-detections", action="store_true",
                    help="use manifest GT boxes for violation eval (isolates geometry/classifier logic)")
    er.add_argument("--persist", action="store_true")
    er.set_defaults(func=_cmd_eval_real)

    cal = sub.add_parser("calibrate", help="generate calibrated camera config YAML")
    cal.add_argument("--image", help="reference frame path")
    cal.add_argument("--width", type=int)
    cal.add_argument("--height", type=int)
    cal.add_argument("--out", default="config/camera-junction-01.yaml")
    cal.add_argument("--camera", default="Junction-01")
    cal.set_defaults(func=_cmd_calibrate)

    idd = sub.add_parser("import-idd", help="import IDD Lite dataset into eval manifest")
    idd.add_argument("--tar", default=str(Path.home() / "Downloads" / "idd-lite.tar.gz"))
    idd.add_argument("--split", default="val", choices=["train", "val"])
    idd.add_argument("--max-images", type=int, default=None)
    idd.set_defaults(func=_cmd_import_idd)

    sm = sub.add_parser("sample", help="write a synthetic sample scene")
    sm.add_argument("--out", default="data/samples/sample.jpg")
    sm.set_defaults(func=_cmd_sample)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
