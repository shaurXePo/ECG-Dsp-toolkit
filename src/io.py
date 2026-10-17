"""
io.py -- loading MIT-BIH Arrhythmia and MIT-BIH Noise Stress Test (NSTDB) records
through the `wfdb` package, plus a SYNTHETIC ECG generator used only for unit tests
and offline smoke runs (scripts accept --synthetic; their outputs go to a separate
results_synthetic/ folder so they can never be confused with real results).

Conventions
-----------
* Signals are in PHYSICAL units (mV), channel 0 (lead MLII in all five records used).
* Annotation sample indices are returned RELATIVE to the start of the excerpt.
* Only BEAT annotations are kept as reference R-peaks (rhythm / noise / signal-quality
  markers such as '+', '~', '|', '!' are dropped).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = [
    "REPO_ROOT",
    "DATA_DIR",
    "MITDB_RECORDS",
    "NSTDB_NOISE",
    "FS_MITDB",
    "BEAT_SYMBOLS",
    "data_available",
    "load_record",
    "load_noise",
    "synthetic_record",
    "synthetic_noise_record",
]

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
MITDB_RECORDS = ["100", "101", "119", "200", "207"]
NSTDB_NOISE = {"bw": "bw", "em": "em", "ma": "ma"}
FS_MITDB = 360.0

# MIT-BIH beat-annotation codes (everything else is a non-beat marker)
BEAT_SYMBOLS = set("NLRBAaJSVrFejnE/fQ?")


def _paths(data_dir: Path, db: str, name: str) -> str:
    return str(Path(data_dir) / db / name)


def data_available(data_dir: Path = DATA_DIR) -> bool:
    """True if every MIT-BIH record and every NSTDB noise record is on disk."""
    d = Path(data_dir)
    ok_m = all((d / "mitdb" / f"{r}.{e}").exists() for r in MITDB_RECORDS for e in ("dat", "hea", "atr"))
    ok_n = all((d / "nstdb" / f"{r}.{e}").exists() for r in NSTDB_NOISE.values() for e in ("dat", "hea"))
    return ok_m and ok_n


# --------------------------------------------------------------------------- #
# real data
# --------------------------------------------------------------------------- #
def load_record(name: str, data_dir: Path = DATA_DIR, channel: int = 0,
                duration_s: float | None = None, start_s: float = 0.0) -> dict:
    """
    Load one MIT-BIH record excerpt.

    Returns dict(name, fs, signal [mV], ann_sample, ann_symbol, synthetic=False).
    ann_sample is relative to the excerpt start and restricted to beat annotations.
    """
    import wfdb  # imported lazily so the rest of the package works without it

    path = _paths(data_dir, "mitdb", name)
    if not Path(path + ".dat").exists():
        raise FileNotFoundError(
            f"{path}.dat not found. Run `python data/download_data.py` "
            "(or pass --synthetic for an offline smoke run)."
        )
    rec = wfdb.rdrecord(path, channels=[channel])
    fs = float(rec.fs)
    x = np.asarray(rec.p_signal[:, 0], dtype=float)
    ann = wfdb.rdann(path, "atr")
    samples = np.asarray(ann.sample, dtype=np.int64)
    symbols = np.asarray(ann.symbol)

    i0 = int(round(start_s * fs))
    i1 = len(x) if duration_s is None else min(len(x), i0 + int(round(duration_s * fs)))
    x = x[i0:i1]
    is_beat = np.array([s in BEAT_SYMBOLS for s in symbols], dtype=bool)
    sel = is_beat & (samples >= i0) & (samples < i1)
    return {
        "name": name,
        "fs": fs,
        "signal": x,
        "ann_sample": samples[sel] - i0,
        "ann_symbol": symbols[sel],
        "synthetic": False,
    }


def load_noise(kind: str, data_dir: Path = DATA_DIR, channel: int = 0,
               n_samples: int | None = None, offset: int = 0):
    """
    Load an NSTDB noise record ('bw', 'em' or 'ma'), channel 0.
    Returns (noise[n_samples], fs).  The records are 30 min long, so n_samples + offset
    must not exceed ~650 000.
    """
    import wfdb

    path = _paths(data_dir, "nstdb", NSTDB_NOISE[kind])
    if not Path(path + ".dat").exists():
        raise FileNotFoundError(f"{path}.dat not found. Run `python data/download_data.py`.")
    rec = wfdb.rdrecord(path, channels=[channel])
    x = np.asarray(rec.p_signal[:, 0], dtype=float)
    if n_samples is not None:
        if offset + n_samples > len(x):
            raise ValueError("requested noise segment runs past the end of the record")
        x = x[offset: offset + n_samples]
    return x, float(rec.fs)


# --------------------------------------------------------------------------- #
# synthetic stand-in
# --------------------------------------------------------------------------- #
# (amplitude [mV], centre relative to R [s], width sigma [s]) for P, Q, R, S, T
_WAVES = [(0.15, -0.20, 0.025), (-0.12, -0.040, 0.010), (1.00, 0.0, 0.012),
          (-0.20, 0.040, 0.012), (0.30, 0.25, 0.045)]


def synthetic_record(name: str = "synthetic", duration_s: float = 120.0, fs: float = FS_MITDB,
                     heart_rate_bpm: float = 72.0, seed: int = 0) -> dict:
    """
    Sum-of-Gaussians ECG (P, Q, R, S, T waves) with ~5 % RR variability.
    The R-peak truth is exact, so detectors can be unit-tested.
    NOT a substitute for real data in reported results.
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s * fs)
    t = np.arange(n) / fs
    x = np.zeros(n)
    beat_times, tb = [], 0.6
    while tb < duration_s - 0.6:
        beat_times.append(tb)
        tb += (60.0 / heart_rate_bpm) * (1.0 + 0.05 * rng.standard_normal())
    for tb in beat_times:
        lo, hi = max(0, int((tb - 0.5) * fs)), min(n, int((tb + 0.7) * fs))
        tt = t[lo:hi] - tb
        for a, mu, sg in _WAVES:
            x[lo:hi] += a * np.exp(-0.5 * ((tt - mu) / sg) ** 2)
    x += 0.005 * rng.standard_normal(n)
    return {
        "name": name,
        "fs": float(fs),
        "signal": x,
        "ann_sample": np.round(np.array(beat_times) * fs).astype(np.int64),
        "ann_symbol": np.array(["N"] * len(beat_times)),
        "synthetic": True,
    }


def synthetic_noise_record(kind: str, n_samples: int, fs: float = FS_MITDB, seed: int = 0):
    """Synthetic look-alike of an NSTDB noise record (see noise.synthetic_noise)."""
    from .noise import synthetic_noise

    return synthetic_noise(kind, n_samples, fs, seed), float(fs)
