"""
08_yolo_results.py -- fill section 9.5 of README.md from the REAL training/evaluation outputs.

    python scripts/08_yolo_results.py                  # print the markdown
    python scripts/08_yolo_results.py --update-readme  # write it between <!-- YOLO:START --> and <!-- YOLO:END -->

Reads
    datasets/ecg_yolo/dataset_summary.json      written by  python -m vision.make_dataset
    runs/detect/ecg_yolo/test_metrics.json      written by  python -m vision.train_yolo

Nothing is typed by hand. Output from --synthetic datasets (fake labels) is refused.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build_markdown(summary: dict, metrics: dict) -> str:
    a = summary.get("args", {})
    classes = summary["classes"]
    sp = summary["splits"]
    lines = []
    noisy = a.get("noise_snrs") or []
    lines.append(
        f"_Filter before rendering: **{a.get('method', '?')}** (mains {a.get('mains_hz', '?')} Hz); "
        f"windows {a.get('window_s', '?')} s / stride {a.get('stride_s', '?')} s; "
        + (f"noisy copies: {', '.join(a.get('noise_kinds', []))} at {', '.join(f'{s:g}' for s in noisy)} dB; " if noisy else "clean excerpts only; ")
        + f"seed {a.get('seed', '?')}._\n")
    lines.append("**Inter-patient split (de Chazal DS1/DS2)**\n")
    lines.append("| Split | Records | Images | " + " | ".join(classes) + " |")
    lines.append("|---|---|---:|" + "---:|" * len(classes))
    for name in ("train", "val", "test"):
        boxes = summary["boxes"].get(name, {})
        lines.append(f"| {name} | {', '.join(sp.get(name, []))} | {summary['images'].get(name, 0)} | "
                     + " | ".join(str(boxes.get(c, 0)) for c in classes) + " |")
    lines.append("\n**Test-split detection metrics (DS2 records only)**\n")
    lines.append("| Metric | Value |\n|---|---:|")
    lines.append(f"| mAP@50 | {metrics['map50']:.3f} |")
    lines.append(f"| mAP@50-95 | {metrics['map50_95']:.3f} |")
    per = metrics.get("ap50_per_class")
    if per:
        lines.append("\n**AP@50 per class**\n")
        lines.append("| Class | AP@50 | Test boxes |\n|---|---:|---:|")
        tb = summary["boxes"].get("test", {})
        for c in classes:
            lines.append(f"| {c} | {per[c]:.3f} | {tb.get(c, 0)} |" if c in per else f"| {c} | n/a | {tb.get(c, 0)} |")
    lines.append("\n_Read per-class AP together with the box counts: a class with few test boxes gives an unreliable AP. "
                 "Numbers come from the files above; they are small-data results unless many DS1/DS2 records were used._")
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--summary", default=str(ROOT / "datasets/ecg_yolo/dataset_summary.json"))
    p.add_argument("--metrics", default=str(ROOT / "runs/detect/ecg_yolo/test_metrics.json"))
    p.add_argument("--readme", default=str(ROOT / "README.md"))
    p.add_argument("--update-readme", action="store_true")
    args = p.parse_args(argv)

    for f in (args.summary, args.metrics):
        if not Path(f).exists():
            sys.exit(f"missing {f}\n  run `python -m vision.make_dataset` and `python -m vision.train_yolo` first.")
    summary = json.loads(Path(args.summary).read_text())
    metrics = json.loads(Path(args.metrics).read_text())
    if summary.get("args", {}).get("synthetic"):
        sys.exit("dataset was built with --synthetic (fake labels): refusing to write it into the README.")

    md = build_markdown(summary, metrics)
    print(md)
    if args.update_readme:
        readme = Path(args.readme)
        t = readme.read_text(encoding="utf-8")
        a, b = "<!-- YOLO:START -->", "<!-- YOLO:END -->"
        if a not in t or b not in t:
            sys.exit(f"markers {a} / {b} not found in {readme}")
        i, j = t.index(a) + len(a), t.index(b)
        readme.write_text(t[:i] + "\n" + md + t[j:], encoding="utf-8")
        print(f"README updated: {readme}")


if __name__ == "__main__":
    main()
