"""
export_onnx.py -- export trained YOLO weights to ONNX for cloud hosting, and VERIFY the lightweight decoder.

    pip install onnx onnxruntime            # on your laptop, next to ultralytics
    python -m vision.export_onnx --weights runs/detect/ecg_yolo/weights/best.pt --out models/ecg_yolo_best.onnx --check 40

Writes models/ecg_yolo_best.onnx and models/ecg_yolo_meta.json (class names + the train/val/test records and test metrics
of this run, so the hosted app can say whether a record was a test record). --check compares Ultralytics (.pt) with the
onnxruntime decoder on real test images and exits non-zero if they disagree: do NOT deploy if it fails.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def check_agreement(weights: Path, onnx_path: Path, images: list[Path], conf=0.25, iou=0.5, px_tol=4.0) -> dict:
    from PIL import Image
    from ultralytics import YOLO
    from vision.onnx_infer import OnnxYolo

    pt, ox = YOLO(str(weights)), OnnxYolo(onnx_path)
    n_pt = n_ox = n_match = 0
    for f in images:
        img = Image.open(f).convert("RGB")
        a = []
        for b in pt.predict(img, imgsz=640, conf=conf, iou=iou, verbose=False)[0].boxes:
            x0, _, x1, _ = (float(v) for v in b.xyxy[0].tolist())
            a.append((0.5 * (x0 + x1), int(b.cls[0])))
        o = [(0.5 * (r[0] + r[2]), r[4]) for r in ox.predict_xyxy(img, conf=conf, iou=iou)]
        n_pt += len(a); n_ox += len(o)
        used = [False] * len(o)
        for cx, c in a:
            for j, (ocx, oc) in enumerate(o):
                if not used[j] and oc == c and abs(ocx - cx) <= px_tol:
                    used[j] = True; n_match += 1; break
    return {"images": len(images), "ultralytics_boxes": n_pt, "onnx_boxes": n_ox, "matched": n_match,
            "agreement": n_match / max(n_pt, n_ox, 1)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default="runs/detect/ecg_yolo/weights/best.pt")
    p.add_argument("--out", default="models/ecg_yolo_best.onnx")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--check", type=int, default=0, help="verify on N test images (needs the dataset)")
    p.add_argument("--dataset", default="datasets/ecg_yolo")
    args = p.parse_args(argv)

    from ultralytics import YOLO
    weights, out = Path(args.weights), Path(args.out)
    model = YOLO(str(weights))
    exported = Path(model.export(format="onnx", imgsz=args.imgsz, dynamic=False, simplify=False))
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(exported, out)

    meta = {"names": {int(k): v for k, v in model.names.items()}, "imgsz": args.imgsz, "weights": str(weights)}
    ds = Path(args.dataset) / "dataset_summary.json"
    if ds.exists():
        meta["splits"] = json.loads(ds.read_text()).get("splits")
    tm = weights.resolve().parent.parent / "test_metrics.json"
    if tm.exists():
        meta["test_metrics"] = json.loads(tm.read_text())
    (out.parent / "ecg_yolo_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB) and {out.parent / 'ecg_yolo_meta.json'}")

    if args.check:
        imgs = sorted((Path(args.dataset) / "images" / "test").glob("*.png"))
        step = max(len(imgs) // args.check, 1)
        res = check_agreement(weights, out, imgs[::step][: args.check], px_tol=4.0)
        print("agreement check:", res)
        if res["agreement"] < 0.95:
            sys.exit("ONNX decoder disagrees with Ultralytics (<95 % of boxes match): do not deploy; send me this output.")
        print("OK: the ONNX path reproduces the Ultralytics detections.")


if __name__ == "__main__":
    main()
