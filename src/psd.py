"""
psd.py -- periodogram, Welch PSD and spectrogram, implemented by hand with the
FFT from fft.py, written so the results match scipy.signal (see tests).

THEORY
------
Periodogram of a windowed segment x_w[n] = w[n] x[n]  (N samples):

    P[k] = |X_w[k]|^2 / (fs * sum(w^2))          [units: V^2/Hz, "density" scaling]

The sum(w^2) in the denominator compensates for the energy the window removes,
so that  integral(P df) = signal power  regardless of the window.

A single periodogram is an INCONSISTENT estimator: its variance does not shrink
as N grows (each bin has ~100 % relative std).  Welch's method trades
frequency resolution for variance: cut the signal into overlapping segments,
window each, compute one periodogram per segment, and AVERAGE.  With K roughly
independent segments the variance drops ~ 1/K.  With a Hann window, 50 %
overlap recovers most of the data lost to the window taper.

One-sided spectrum of a real signal: bins 1 .. N/2-1 appear twice (positive and
negative frequency) so they are doubled; DC and Nyquist are not.

The SPECTROGRAM is the same per-segment periodogram kept as a function of time
(no averaging): rows = frequency, columns = segment centre times.  Segment
length sets the time/frequency resolution trade-off (Heisenberg-Gabor).
"""
from __future__ import annotations

import numpy as np

from . import windows as _win
from .fft import fft_iterative, is_power_of_two, next_pow2, zero_pad_pow2

__all__ = ["periodogram", "welch", "spectrogram"]


def _detrend(seg: np.ndarray, kind) -> np.ndarray:
    if kind in (None, False, "none"):
        return seg
    if kind == "constant":
        return seg - seg.mean(axis=-1, keepdims=True)
    if kind == "linear":
        n = seg.shape[-1]
        t = np.arange(n) - (n - 1) / 2.0
        slope = (seg * t).sum(axis=-1, keepdims=True) / np.sum(t ** 2)
        return seg - seg.mean(axis=-1, keepdims=True) - slope * t
    raise ValueError("detrend must be None, 'constant' or 'linear'")


def _segment_spectra(x, fs, window, nperseg, noverlap, nfft, detrend, scaling):
    """Return (freqs, P[n_seg, n_freq], segment_start_indices), one-sided."""
    x = np.asarray(x, dtype=float)
    if nperseg > len(x):
        raise ValueError("nperseg longer than the signal")
    if noverlap is None:
        noverlap = nperseg // 2
    if not 0 <= noverlap < nperseg:
        raise ValueError("need 0 <= noverlap < nperseg")
    if nfft is None:
        nfft = next_pow2(nperseg)
    if not is_power_of_two(nfft) or nfft < nperseg:
        raise ValueError("nfft must be a power of two >= nperseg")

    w = _win.get_window(window, nperseg, sym=False)       # periodic window
    step = nperseg - noverlap
    n_seg = (len(x) - noverlap) // step                    # drop incomplete tail
    starts = step * np.arange(n_seg)
    seg = x[starts[:, None] + np.arange(nperseg)[None, :]]  # (n_seg, nperseg)

    seg = _detrend(seg, detrend) * w
    X = fft_iterative(zero_pad_pow2(seg, nfft))             # (n_seg, nfft)

    if scaling == "density":
        scale = 1.0 / (fs * np.sum(w ** 2))
    elif scaling == "spectrum":
        scale = 1.0 / np.sum(w) ** 2
    else:
        raise ValueError("scaling must be 'density' or 'spectrum'")

    P = (np.abs(X) ** 2) * scale
    P = P[:, : nfft // 2 + 1]
    P[:, 1:-1] *= 2.0                                       # fold negative freqs
    f = np.arange(nfft // 2 + 1) * fs / nfft
    return f, P, starts


def periodogram(x, fs=1.0, window="hann", nfft=None, detrend="constant", scaling="density"):
    """Single windowed periodogram of the whole signal."""
    x = np.asarray(x, dtype=float)
    f, P, _ = _segment_spectra(x, fs, window, len(x), 0, nfft, detrend, scaling)
    return f, P[0]


def welch(x, fs=1.0, window="hann", nperseg=1024, noverlap=None, nfft=None,
          detrend="constant", scaling="density"):
    """
    Welch PSD estimate (mean of the per-segment periodograms).
    Defaults mirror scipy.signal.welch: periodic Hann, 50 % overlap, per-segment
    mean removal, density scaling.  nfft defaults to next_pow2(nperseg).
    """
    f, P, _ = _segment_spectra(x, fs, window, nperseg, noverlap, nfft, detrend, scaling)
    return f, P.mean(axis=0)


def spectrogram(x, fs=1.0, window="hann", nperseg=256, noverlap=None, nfft=None,
                detrend="constant", scaling="density"):
    """
    Short-time PSD.  Returns (f, t, Sxx) with Sxx.shape == (len(f), n_segments);
    t are the segment centre times in seconds.
    """
    f, P, starts = _segment_spectra(x, fs, window, nperseg, noverlap, nfft, detrend, scaling)
    t = (starts + nperseg / 2.0) / fs
    return f, t, P.T
