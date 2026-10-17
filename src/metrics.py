"""
metrics.py -- output SNR, SNR improvement, RMSE and beat-detection scores.

DEFINITIONS (all computed on DC-removed signals over the same evaluation window)
-----------------------------------------------------------------------------
    err[n]   = test0[n] - ref0[n]            (x0 = x - mean(x))
    SNR_out  = 10 log10( sum ref0^2 / sum err^2 )
    dSNR     = SNR_out - SNR_in              SNR_in = same formula applied to the NOISY signal
    RMSE     = sqrt( mean err^2 )            (same units as the signal, mV here)

The reference `ref` is the ORIGINAL clean record.  Consequence worth remembering:
the filter also removes whatever baseline drift / HF noise the "clean" record
itself contains, and that counts as error.  So at high input SNR a filter can
have dSNR < 0: the metric includes the filter's own distortion, which is honest.

Beat detection is scored against the expert annotations with one-to-one
matching inside a tolerance window (ANSI/AAMI EC57 uses 150 ms):

    Se  (sensitivity) = TP / (TP + FN)       fraction of true beats found
    PPV (precision)   = TP / (TP + FP)       fraction of detections that are real
    F1  = 2 Se PPV / (Se + PPV)
"""
from __future__ import annotations

import numpy as np

__all__ = ["snr_db", "snr_improvement_db", "rmse", "trim_slice", "match_peaks", "peak_scores"]


def _center(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x - x.mean()


def snr_db(ref, test) -> float:
    """10 log10( energy(ref0) / energy(test0 - ref0) )."""
    ref0, test0 = _center(ref), _center(test)
    err = test0 - ref0
    e_err = np.sum(err ** 2)
    if e_err == 0.0:                       # perfect reconstruction
        return float("inf")
    return float(10.0 * np.log10(np.sum(ref0 ** 2) / e_err))


def snr_improvement_db(ref, noisy, filtered) -> float:
    """SNR_out - SNR_in, both measured against `ref`."""
    return snr_db(ref, filtered) - snr_db(ref, noisy)


def rmse(ref, test) -> float:
    """Root-mean-square error between DC-removed signals."""
    err = _center(test) - _center(ref)
    return float(np.sqrt(np.mean(err ** 2)))


def trim_slice(n: int, fs: float, trim_s: float) -> slice:
    """Slice that drops `trim_s` seconds at both ends (filter edge transients)."""
    k = int(round(trim_s * fs))
    if 2 * k >= n:
        raise ValueError("trim is longer than the signal")
    return slice(k, n - k)


def match_peaks(ref_peaks, det_peaks, tol_samples: int):
    """
    Greedy one-to-one matching.  For each reference beat (in time order) take the
    nearest still-unmatched detection within +/- tol_samples.
    Returns (tp, fp, fn).
    """
    ref = np.sort(np.asarray(ref_peaks, dtype=np.int64))
    det = np.sort(np.asarray(det_peaks, dtype=np.int64))
    used = np.zeros(len(det), dtype=bool)
    tp = 0
    for r in ref:
        if len(det) == 0:
            break
        j = np.searchsorted(det, r)
        best, best_d = None, tol_samples + 1
        for c in (j - 1, j, j + 1):                     # neighbours of the insertion point
            if 0 <= c < len(det) and not used[c]:
                d = abs(det[c] - r)
                if d < best_d:
                    best, best_d = c, d
        if best is not None and best_d <= tol_samples:
            used[best] = True
            tp += 1
    return tp, int(len(det) - tp), int(len(ref) - tp)


def peak_scores(tp: int, fp: int, fn: int) -> dict:
    """Sensitivity, positive predictive value and F1 from match counts."""
    se = tp / (tp + fn) if (tp + fn) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    f1 = 2 * se * ppv / (se + ppv) if (se + ppv) and not np.isnan(se + ppv) else float("nan")
    return {"se": se, "ppv": ppv, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
