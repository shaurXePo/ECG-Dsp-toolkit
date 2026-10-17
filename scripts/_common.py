"""
Shared helpers for the numbered scripts (not meant to be run directly).

* every script accepts   --data-dir  --mains-hz  --seed  --synthetic
* --synthetic swaps real MIT-BIH/NSTDB data for the synthetic stand-ins in src/io.py and
  writes to results_synthetic/ instead of results/, so a smoke run can never overwrite
  or be mistaken for real results.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")                      # scripts write PNGs; no display needed
import matplotlib.pyplot as plt            # noqa: E402
import numpy as np                         # noqa: E402

from src import io as ecgio                # noqa: E402

SEED = 42
SNRS = [24, 18, 12, 6, 0]
NOISE_TYPES = ["bw", "em", "ma"]
NOISE_NAMES = {"bw": "baseline wander", "em": "electrode motion", "ma": "muscle artefact",
               "mains": "mains hum (synthetic)"}

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "axes.grid": True,
                     "grid.alpha": 0.3, "font.size": 10})


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--data-dir", type=Path, default=ecgio.DATA_DIR,
                   help="folder containing mitdb/ and nstdb/ (default: ./data)")
    p.add_argument("--mains-hz", type=float, default=60.0,
                   help="mains / notch frequency: 60 for MIT-BIH, 50 for Indian recordings")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--synthetic", action="store_true",
                   help="offline smoke run with synthetic ECG/noise (outputs go to results_synthetic/)")
    return p


def out_dirs(args) -> tuple[Path, Path]:
    base = ROOT / ("results_synthetic" if args.synthetic else "results")
    fig = base / "figures"
    fig.mkdir(parents=True, exist_ok=True)
    return base, fig


def tag(args) -> str:
    return "[SYNTHETIC DATA] " if args.synthetic else ""


def get_record(args, name: str, duration_s: float | None = None, start_s: float = 0.0) -> dict:
    """Real MIT-BIH excerpt, or a synthetic one under --synthetic."""
    if args.synthetic:
        idx = ecgio.MITDB_RECORDS.index(name) if name in ecgio.MITDB_RECORDS else 0
        hr = (62, 75, 68, 80, 90)[idx]
        return ecgio.synthetic_record(name, duration_s or 300.0, ecgio.FS_MITDB, hr, args.seed + idx)
    return ecgio.load_record(name, args.data_dir, duration_s=duration_s, start_s=start_s)


def get_noise(args, kind: str, n: int, fs: float, key: int = 0) -> np.ndarray:
    """
    A length-n segment of the NSTDB noise record `kind` (random but seeded offset), or a
    synthetic look-alike under --synthetic.
    """
    if args.synthetic:
        return ecgio.synthetic_noise_record(kind, n, fs, seed=args.seed + 17 * key)[0]
    # the NSTDB records are 650 000 samples long
    rng = np.random.default_rng(args.seed + 1000 * key)
    full, fs_n = ecgio.load_noise(kind, args.data_dir)
    if abs(fs_n - fs) > 1e-9:
        raise ValueError(f"NSTDB fs {fs_n} != record fs {fs}")
    if n > len(full):
        raise ValueError("requested more noise than the NSTDB record holds")
    off = int(rng.integers(0, len(full) - n + 1))
    return full[off: off + n]


def savefig(fig, fig_dir: Path, name: str) -> Path:
    path = fig_dir / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT)}")
    return path
