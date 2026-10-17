"""
peaks.py -- a compact Pan-Tompkins-style QRS detector.

Pipeline (Pan & Tompkins, 1985), adapted for offline use:

  1. Band-pass 5-15 Hz   -> keeps QRS energy, rejects P/T waves, baseline, mains, most EMG.
                            Applied ZERO-PHASE so the detector adds no delay of its own:
                            any timing shift seen in the results comes from the filter
                            under test, not from the detector.
  2. 5-point derivative  -> emphasises the steep QRS slopes:
                            y[n] = (fs/8)(-x[n-2] - 2x[n-1] + 2x[n+1] + x[n+2])
  3. Squaring            -> makes everything positive, emphasises large slopes.
  4. Moving-window integration (150 ms, centred) -> one smooth hump per QRS.
  5. Adaptive thresholds on the hump heights:
         SPKI = 0.125 peak + 0.875 SPKI   (running estimate of signal-peak level)
         NPKI = 0.125 peak + 0.875 NPKI   (running estimate of noise-peak level)
         THR  = NPKI + 0.25 (SPKI - NPKI)
     plus a 200 ms refractory period (physiologically a ventricle cannot re-fire sooner)
     and a simple search-back: if no beat has been accepted for 1.66 x the mean of the
     recent RR intervals, accept the strongest rejected hump above THR/2.
  6. R-peak location = argmax |band-passed signal| within +/-100 ms of the hump.

What this is NOT: a clinically validated detector (no T-wave discrimination, no
second search-back threshold set, no RR-interval plausibility checks).  It is
good enough to compare filters on equal footing.
"""
from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d

__all__ = ["detect_r_peaks"]


def detect_r_peaks(x, fs: float, band=(5.0, 15.0), mwi_ms: float = 150.0,
                   refractory_ms: float = 200.0) -> np.ndarray:
    """Return sample indices of detected R peaks in x (1-D ECG)."""
    x = np.asarray(x, dtype=float)
    x = x - np.median(x)

    sos = signal.butter(2, band, btype="bandpass", fs=fs, output="sos")
    xb = signal.sosfiltfilt(sos, x)

    # derivative kernel; sign is irrelevant because of the squaring that follows
    d = np.convolve(xb, np.array([1.0, 2.0, 0.0, -2.0, -1.0]) * fs / 8.0, mode="same")
    sq = d ** 2
    mwi = uniform_filter1d(sq, size=max(1, int(round(mwi_ms * 1e-3 * fs))), mode="nearest")

    refractory = int(round(refractory_ms * 1e-3 * fs))
    cand, _ = signal.find_peaks(mwi, distance=refractory)
    if len(cand) == 0:
        return np.array([], dtype=np.int64)

    init = mwi[: int(2 * fs)]
    spki = 0.5 * init.max()
    npki = 0.5 * init.mean()

    accepted: list[int] = []
    rejected: list[tuple[int, float]] = []
    rr: list[int] = []

    def _accept(idx: int, val: float, weight: float) -> None:
        nonlocal spki, rejected
        if accepted:
            rr.append(idx - accepted[-1])
        accepted.append(idx)
        spki = weight * val + (1.0 - weight) * spki
        rejected = [r for r in rejected if r[0] > idx]

    for p in cand:
        v = mwi[p]
        thr = npki + 0.25 * (spki - npki)

        # search-back for a missed beat
        if len(rr) >= 2 and rejected and (p - accepted[-1]) > 1.66 * np.mean(rr[-8:]):
            best = max(rejected, key=lambda t: t[1])
            if best[1] > 0.5 * thr:
                _accept(best[0], best[1], 0.25)

        if v > thr:
            _accept(int(p), float(v), 0.125)
        else:
            npki = 0.125 * v + 0.875 * npki
            rejected.append((int(p), float(v)))

    # refine: location of the largest |band-passed| sample around each hump
    half = int(round(0.100 * fs))
    absxb = np.abs(xb)
    peaks = []
    for c in accepted:
        lo, hi = max(0, c - half), min(len(x), c + half + 1)
        peaks.append(lo + int(np.argmax(absxb[lo:hi])))
    peaks = np.unique(np.array(peaks, dtype=np.int64))

    # enforce the refractory period again after refinement
    keep = [peaks[0]] if len(peaks) else []
    for q in peaks[1:]:
        if q - keep[-1] >= refractory:
            keep.append(q)
    return np.array(keep, dtype=np.int64)
