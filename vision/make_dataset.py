"""
make_dataset.py -- build a YOLO dataset from MIT-BIH (optionally with NSTDB noise).

Order of operations (this matters):
    1. take a whole record excerpt (default 300 s),
    2. optionally add NSTDB noise to the WHOLE excerpt at an exact SNR,
    3. filter the WHOLE excerpt with your toolkit filters (so the filters see seconds of context),
    4. only then cut windows, render them as images and write YOLO labels.

Splits are INTER-PATIENT (de Chazal DS1 / DS2). DS1 -> train + val (by record), DS2 -> test.
Noise for train/val comes from the first half of each NSTDB record, test from the second half.

    python -m vision.make_dataset                       # real data in data/mitdb
    python -m vision.make_dataset --synthetic           # offline smoke test (synthetic ECG, fake labels)
    python -m vision.make_dataset --noise-snrs 12 6     # add noisy copies (all 3 noise types x each SNR)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import io as ecgio                      # noqa: E402
from src import noise as ecgnoise                # noqa: E402
from src import pipeline                         # noqa: E402
from vision.labels import CLASSES, CLASSES_ALL, beat_boxes, to_yolo_lines   # noqa: E402
from vision.render import render_strip           # noqa: E402

# de Chazal et al. (2004) inter-patient protocol (the four paced records are excluded)
DS1 = ["101", "106", "108", "109", "112", "114", "115", "116", "118", "119", "122", "124",
       "201", "203", "205", "207", "208", "209", "215", "220", "223", "230"]
DS2 = ["100", "103", "105", "111", "113", "117", "121", "123", "200", "202", "210", "212",
       "213", "214", "219", "221", "222", "228", "231", "232", "233", "234"]
NOISE_KINDS = ["bw", "em", "ma"]
PREFERRED_VAL = ["230", "124"]


def plan_splits(available, val_records=None) -> dict[str, list[str]]:
    """Inter-patient split restricted to the records that are actually available."""
    avail = set(available)
    ds1 = [r for r in DS1 if r in avail]
    ds2 = [r for r in DS2 if r in avail]
    if val_records is None:
        pref = [r for r in PREFERRED_VAL if r in ds1]
        val = pref[:1] if pref else (ds1[-1:] if len(ds1) >= 2 else [])
    else:
        val = [r for r in val_records if r in ds1]
    train = [r for r in ds1 if r not in val]
    splits = {"train": train, "val": val, "test": ds2}
    flat = train + val + ds2
    assert len(flat) == len(set(flat)), "a record appears in more than one split"
    return splits


def _synthetic_records(duration_s: float, seed: int):
    rng = np.random.default_rng(seed)
    recs = {}
    for i, name in enumerate(["101", "119", "207", "100", "200"]):
        r = ecgio.synthetic_record(name, duration_s=duration_s, seed=seed + i)
        sym = r["ann_symbol"].copy()
        u = rng.random(len(sym))
        sym[u < 0.10] = "V"                       # fake labels: smoke-testing only
        sym[(u >= 0.10) & (u < 0.15)] = "S"
        r["ann_symbol"] = sym
        recs[name] = r
    return recs


def _noise_pool(args, n_needed: int):
    pool = {}
    for k, kind in enumerate(NOISE_KINDS):
        if args.synthetic:
            arr, _ = ecgio.synthetic_noise_record(kind, max(4 * n_needed, 20000), ecgio.FS_MITDB, seed=k + 1)
        else:
            arr, _ = ecgio.load_noise(kind, Path(args.data_dir))
        pool[kind] = np.asarray(arr, dtype=float)
    return pool


def _noise_segment(pool, kind, split, n, rng):
    arr = pool[kind]
    half = len(arr) // 2
    seg = arr[:half] if split in ("train", "val") else arr[half:]
    if len(seg) < n:
        raise ValueError("noise record too short for this excerpt")
    off = int(rng.integers(0, len(seg) - n + 1))
    return seg[off:off + n]


def build(args) -> dict:
    out = Path(args.out)
    classes = CLASSES_ALL if args.with_q else CLASSES
    duration = None if args.full else args.duration_s
    rng = np.random.default_rng(args.seed)

    # ---- records -------------------------------------------------------------------------
    if args.synthetic:
        records = _synthetic_records(args.duration_s, args.seed)
    else:
        mit = Path(args.data_dir) / "mitdb"
        present = sorted(p.stem for p in mit.glob("*.dat"))
        wanted = args.records or present
        missing = [r for r in wanted if r not in present]
        if missing:
            print(f"  not found in {mit}: {missing}")
        records = {r: ecgio.load_record(r, Path(args.data_dir), duration_s=duration)
                   for r in wanted if r in present}
    splits = plan_splits(list(records), args.val_records)
    ignored = [r for r in records if all(r not in v for v in splits.values())]
    if ignored:
        print(f"  ignored (not in DS1/DS2, e.g. paced records): {ignored}")
    if not splits["train"] or not splits["test"]:
        raise SystemExit("Need at least one DS1 record (train) and one DS2 record (test). "
                         "Download more records: the five defaults cover 101,119,207 (DS1) and 100,200 (DS2).")
    print("splits:", {k: v for k, v in splits.items()})

    # ---- variants (clean + optional noisy copies) -------------------------------------------
    variants = [("clean", None, None)]
    for kind in args.noise_kinds:
        for snr in args.noise_snrs:
            variants.append((kind, float(snr), f"{kind}{snr:g}dB"))
    need_noise = len(variants) > 1
    longest = max(len(r["signal"]) for r in records.values())
    pool = _noise_pool(args, longest) if need_noise else {}

    banks, rows = {}, []
    summary = {"splits": splits, "classes": classes, "images": {}, "boxes": {}, "args": vars(args)}
    for split, names in splits.items():
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        summary["images"][split] = 0
        summary["boxes"][split] = {c: 0 for c in classes}
        for name in names:
            rec = records[name]
            fs, x = rec["fs"], np.asarray(rec["signal"], dtype=float)
            ann, sym = np.asarray(rec["ann_sample"]), np.asarray(rec["ann_symbol"])
            if fs not in banks:
                banks[fs] = pipeline.build_filter_bank(fs, args.mains_hz)
            bank = banks[fs]
            win, stride, trim = (int(round(s * fs)) for s in (args.window_s, args.stride_s, args.trim_s))

            for vkind, snr, vname in variants:
                xn = x
                if vkind != "clean":
                    nz = _noise_segment(pool, vkind, split, len(x), rng)
                    xn = ecgnoise.mix_at_snr(x, nz, snr)[0]
                y = (xn - xn.mean()) if args.method == "none" else bank.apply(args.method, xn)
                tag = vname or "clean"
                for start in range(trim, len(y) - trim - win + 1, stride):
                    strip = render_strip(y[start:start + win], fs)
                    m = (ann >= start) & (ann < start + win)
                    boxes = beat_boxes(strip, start, ann[m], sym[m], args.box_half_s, classes)
                    if not boxes:
                        continue
                    stem = f"{name}_{tag}_{start}"
                    strip.image.save(out / "images" / split / f"{stem}.png")
                    (out / "labels" / split / f"{stem}.txt").write_text(to_yolo_lines(boxes))
                    summary["images"][split] += 1
                    for b in boxes:
                        summary["boxes"][split][classes[b[0]]] += 1
                    rows.append((split, name, tag, start, len(boxes)))
            print(f"  {split:5s} {name}: done")

    yaml = [f"path: {out.resolve()}", "train: images/train", "val: images/val", "test: images/test", "names:"]
    yaml += [f"  {i}: {c}" for i, c in enumerate(classes)]
    (out / "dataset.yaml").write_text("\n".join(yaml) + "\n")
    (out / "index.csv").write_text("split,record,variant,start_sample,n_boxes\n" +
                                   "\n".join(",".join(map(str, r)) for r in rows) + "\n")
    (out / "dataset_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="datasets/ecg_yolo")
    p.add_argument("--data-dir", default=str(ecgio.DATA_DIR))
    p.add_argument("--records", nargs="+", default=None, help="default: every .dat in data/mitdb")
    p.add_argument("--val-records", nargs="+", default=None, help="DS1 records held out for validation")
    p.add_argument("--method", default="IIR (filtfilt)", choices=list(pipeline.METHODS) + ["none"],
                   help="filter applied to the whole excerpt before rendering")
    p.add_argument("--mains-hz", type=float, default=60.0, help="60 for MIT-BIH, 50 for Indian recordings")
    p.add_argument("--duration-s", type=float, default=300.0)
    p.add_argument("--full", action="store_true", help="use whole 30-min records (many images!)")
    p.add_argument("--window-s", type=float, default=4.0)
    p.add_argument("--stride-s", type=float, default=2.0)
    p.add_argument("--trim-s", type=float, default=5.0, help="seconds dropped at both ends (filter transients)")
    p.add_argument("--box-half-s", type=float, default=0.20)
    p.add_argument("--with-q", action="store_true", help="include the (very rare) Q class")
    p.add_argument("--noise-snrs", nargs="*", type=float, default=[], help="e.g. 12 6 adds noisy copies")
    p.add_argument("--noise-kinds", nargs="*", default=NOISE_KINDS, choices=NOISE_KINDS)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--synthetic", action="store_true", help="offline smoke test: synthetic ECG, FAKE labels")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    s = build(args)
    print("\nimages :", s["images"])
    print("boxes  :", s["boxes"])
    print(f"dataset.yaml -> {Path(args.out) / 'dataset.yaml'}")
    if args.synthetic:
        print("NOTE: synthetic data with fake labels; never report results from it.")
    return s


if __name__ == "__main__":
    main()
