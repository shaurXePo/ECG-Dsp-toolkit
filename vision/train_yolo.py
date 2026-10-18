"""
train_yolo.py -- train a YOLOv8 model on the dataset from vision.make_dataset.

    pip install ultralytics
    python -m vision.train_yolo --data datasets/ecg_yolo/dataset.yaml --epochs 30

Notes
  * ECG images are NOT natural images: flips and colour jitter are switched off (a flipped ECG is a
    different waveform) and mosaic is off by default (it shrinks the traces). Change them and report it.
  * Ultralytics is AGPL-3.0: fine for an open-source repo, read the licence before commercial use.
  * Report test-set (DS2) numbers only; the val split is for choosing checkpoints.
  * CPU works for yolov8n with a few thousand images (slow); a free Colab GPU is much faster.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="datasets/ecg_yolo/dataset.yaml")
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default=None, help="'cpu', '0', ... (default: auto)")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mosaic", type=float, default=0.0)
    p.add_argument("--project", default="runs/detect")
    p.add_argument("--name", default="ecg_yolo")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.device,
                workers=args.workers, seed=args.seed, project=args.project, name=args.name, exist_ok=True,
                fliplr=0.0, flipud=0.0, mosaic=args.mosaic, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
                degrees=0.0, shear=0.0, perspective=0.0, translate=0.05, scale=0.1)

    best = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"\nbest weights: {best}")
    metrics = YOLO(str(best)).val(data=args.data, split="test", imgsz=args.imgsz, device=args.device)
    res = {"map50": float(metrics.box.map50), "map50_95": float(metrics.box.map)}
    try:
        names = metrics.names
        res["ap50_per_class"] = {names[int(i)]: float(a) for i, a in zip(metrics.box.ap_class_index, metrics.box.ap50)}
    except Exception:                                    # noqa: BLE001
        pass
    out = Path(args.project) / args.name / "test_metrics.json"
    out.write_text(json.dumps(res, indent=2))
    print("TEST (DS2):", res, "->", out)


if __name__ == "__main__":
    main()
