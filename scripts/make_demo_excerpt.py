"""
make_demo_excerpt.py -- save a short REAL MIT-BIH excerpt (signal + beat annotations) for the hosted app.

    python scripts/make_demo_excerpt.py --record 100 --record 200 --seconds 60

Writes app/sample_data/mitdb_<record>_<seconds>s.npz (a few hundred kB) and SOURCE.txt. Use TEST records (DS2: 100, 200)
so the app's model check is fair. Check the PhysioNet page for the database's licence terms before redistributing.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import io as ecgio                      # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--record", action="append", default=None)
    p.add_argument("--seconds", type=float, default=60.0)
    p.add_argument("--start-s", type=float, default=10.0)
    p.add_argument("--data-dir", default=str(ecgio.DATA_DIR))
    p.add_argument("--out", default=str(ROOT / "app" / "sample_data"))
    p.add_argument("--synthetic", action="store_true", help="tests only")
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in a.record or ["100", "200"]:
        if a.synthetic:
            r = ecgio.synthetic_record(name, duration_s=a.seconds + a.start_s, seed=1)
            s0 = int(a.start_s * r["fs"])
            keep = (r["ann_sample"] >= s0) & (r["ann_sample"] < s0 + int(a.seconds * r["fs"]))
            r = {"signal": r["signal"][s0:s0 + int(a.seconds * r["fs"])], "fs": r["fs"],
                 "ann_sample": r["ann_sample"][keep] - s0, "ann_symbol": r["ann_symbol"][keep]}
        else:
            r = ecgio.load_record(name, Path(a.data_dir), duration_s=a.seconds, start_s=a.start_s)
        f = out / f"mitdb_{name}_{a.seconds:g}s.npz"
        np.savez_compressed(f, signal=np.asarray(r["signal"], dtype=np.float32), fs=float(r["fs"]),
                            ann_sample=np.asarray(r["ann_sample"], dtype=np.int64),
                            ann_symbol=np.asarray(r["ann_symbol"]).astype("<U1"), record=str(name))
        print(f"wrote {f} ({f.stat().st_size / 1e3:.0f} kB, {len(r['ann_sample'])} beats)")
    (out / "SOURCE.txt").write_text(
        "Excerpts of the MIT-BIH Arrhythmia Database (PhysioNet).\n"
        "Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. IEEE Eng Med Biol 20(3):45-50 (2001).\n"
        "Goldberger et al. PhysioBank, PhysioToolkit, and PhysioNet. Circulation 101(23):e215-e220 (2000).\n"
        "https://physionet.org/content/mitdb/1.0.0/  (see the page for licence terms)\n")


if __name__ == "__main__":
    main()
