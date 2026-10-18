"""
train_yolo.py -- train a YOLOv8 model on the dataset from vision.make_dataset, then evaluate on the TEST split (DS2).

    pip install ultralytics
    python -m vision.train_yolo --data datasets/ecg_yolo/dataset.yaml --epochs 30 --patience 8

    # evaluate weights you already have (no training):
    python -m vision.train_yolo --eval-only --weights runs/detect/ecg_yolo/weights/best.pt

Notes
  * Ultralytics prepends its own runs directory to a RELATIVE --project, which hid the weights in
    runs/detect/runs/detect/... in an earlier version. The project path is now made absolute.
  * --patience stops training when validation fitness has not improved for N epochs (the default of 100
    in Ultralytics means "never" for a 30-epoch run).
  * ECG images are NOT natural images: flips and colour jitter are off and mosaic is off by default.
  * Ultralytics is AGPL-3.0: fine for an open-source repo, read the licence before commercial use.
  * Report test-set (DS2) numbers only; the val split is for choosing checkpoints.
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
    p.add_argument("--patience", type=int, default=8, help="early stopping (epochs without improvement)")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default=None, help="'cpu', '0', ... (default: auto)")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mosaic", type=float, default=0.0)
    p.add_argument("--project", default="runs/detect")
    p.add_argument("--name", default="ecg_yolo")
    p.add_argument("--eval-only", action="store_true", help="skip training; evaluate --weights on the test split")
    p.add_argument("--weights", default=None, help="weights for --eval-only")
    return p.parse_args(argv)


def evaluate(weights: Path, args) -> dict:
    from ultralytics import YOLO

    metrics = YOLO(str(weights)).val(data=args.data, split="test", imgsz=args.imgsz, device=args.device)
    res = {"weights": str(weights), "map50": float(metrics.box.map50), "map50_95": float(metrics.box.map)}
    try:
        names = metrics.names
        res["ap50_per_class"] = {names[int(i)]: float(a) for i, a in zip(metrics.box.ap_class_index, metrics.box.ap50)}
    except Exception:                                    # noqa: BLE001
        pass
    out = weights.resolve().parent.parent / "test_metrics.json"
    out.write_text(json.dumps(res, indent=2))
    print("TEST (DS2):", res, "->", out)
    return res


def main(argv=None):
    args = parse_args(argv)
    if args.eval_only:
        if not args.weights or not Path(args.weights).exists():
            raise SystemExit("--eval-only needs --weights pointing to an existing .pt file")
        return evaluate(Path(args.weights), args)

    from ultralytics import YOLO

    project = Path(args.project).resolve()                # absolute: Ultralytics must not prepend its own runs dir
    model = YOLO(args.model)
    model.train(data=args.data, epochs=args.epochs, patience=args.patience, imgsz=args.imgsz, batch=args.batch,
                device=args.device, workers=args.workers, seed=args.seed, project=str(project), name=args.name,
                exist_ok=True, fliplr=0.0, flipud=0.0, mosaic=args.mosaic, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
                degrees=0.0, shear=0.0, perspective=0.0, translate=0.05, scale=0.1)

    trainer = getattr(model, "trainer", None)
    save_dir = Path(getattr(trainer, "save_dir", project / args.name))
    best = save_dir / "weights" / "best.pt"
    print(f"\nbest weights: {best}")
    return evaluate(best, args)


if __name__ == "__main__":
    main()
