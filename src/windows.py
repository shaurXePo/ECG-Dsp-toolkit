"""
windows.py -- rectangular, Hann, Hamming, Blackman windows written from the
cosine-sum definition, plus the numbers that matter when choosing one.

THEORY
------
Analysing N samples means multiplying the (infinite) signal by a rectangular
window.  Multiplication in time is convolution in frequency, so every sinusoid
is smeared by the window's spectrum W(f):

  * main-lobe width  -> frequency RESOLUTION (can two close tones be separated?)
  * sidelobe level   -> DYNAMIC RANGE (can a weak tone be seen next to a strong one?)

You cannot improve one without hurting the other:

    window        peak sidelobe   main-lobe (null-to-null)   sidelobe roll-off
    rectangular     -13 dB            2 bins                   6 dB/oct
    Hann            -31 dB            4 bins                  18 dB/oct
    Hamming         -43 dB            4 bins                   6 dB/oct
    Blackman        -58 dB            6 bins                  18 dB/oct

All four are "cosine-sum" windows:

    w[n] = sum_k (-1)^k a_k cos(2 pi k n / M),    n = 0..N-1

with  M = N-1  (SYMMETRIC window, used for FIR design: w[n] = w[N-1-n], which
keeps the FIR linear-phase)  or  M = N  (PERIODIC window, used for spectral
analysis: it is one period of a sequence that tiles seamlessly, which is what
the DFT implicitly assumes.  scipy.signal.get_window / welch use this form.)
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "rectangular",
    "hann",
    "hamming",
    "blackman",
    "get_window",
    "window_properties",
    "WINDOWS",
]


def _cosine_sum(N: int, coeffs, sym: bool) -> np.ndarray:
    if N < 1:
        raise ValueError("N must be >= 1")
    if N == 1:
        return np.ones(1)
    M = N - 1 if sym else N
    n = np.arange(N)
    w = np.zeros(N)
    for k, a in enumerate(coeffs):
        w += ((-1) ** k) * a * np.cos(2 * np.pi * k * n / M)
    return w


def rectangular(N: int, sym: bool = True) -> np.ndarray:
    return np.ones(N)


def hann(N: int, sym: bool = True) -> np.ndarray:
    """0.5 - 0.5 cos(2 pi n / M)  (touches zero at the ends when sym=True)."""
    return _cosine_sum(N, (0.5, 0.5), sym)


def hamming(N: int, sym: bool = True) -> np.ndarray:
    """0.54 - 0.46 cos(...) : coefficients chosen to cancel the first sidelobe."""
    return _cosine_sum(N, (0.54, 0.46), sym)


def blackman(N: int, sym: bool = True) -> np.ndarray:
    """0.42 - 0.5 cos(...) + 0.08 cos(2 ...) : exact-Blackman coefficients."""
    return _cosine_sum(N, (0.42, 0.5, 0.08), sym)


WINDOWS = {
    "rectangular": rectangular,
    "boxcar": rectangular,
    "hann": hann,
    "hanning": hann,
    "hamming": hamming,
    "blackman": blackman,
}


def get_window(name: str, N: int, sym: bool = True) -> np.ndarray:
    """Look up a window by name.  sym=True for filter design, False for analysis."""
    try:
        return WINDOWS[name.lower()](N, sym=sym)
    except KeyError:
        raise ValueError(f"unknown window '{name}'. Choose from {sorted(set(WINDOWS))}")


def window_properties(w: np.ndarray, nfft: int = 2 ** 16) -> dict:
    """
    Numerically measure a window's figures of merit.  (Uses numpy's FFT only
    for the very long zero-padded transform; the quantities are standard.)

    Returns
    -------
    coherent_gain      mean(w): amplitude of a sinusoid's peak relative to rectangular
    enbw_bins          equivalent noise bandwidth in DFT bins = N sum(w^2) / sum(w)^2
    mainlobe_null_bins full null-to-null main-lobe width, in bins of the N-point DFT
    bw3db_bins         -3 dB width, in bins
    peak_sidelobe_db   highest sidelobe relative to the main-lobe peak
    """
    w = np.asarray(w, dtype=float)
    N = len(w)
    A = np.abs(np.fft.rfft(w, nfft))
    A = A / A[0]
    dB = 20 * np.log10(np.maximum(A, 1e-300))
    bins = np.arange(len(A)) * N / nfft            # frequency axis in units of fs/N

    # first null = first index where the magnitude starts rising again
    rising = np.where(np.diff(A) > 0)[0]
    first_null = int(rising[0]) if len(rising) else len(A) - 1
    below3 = np.where(dB < -3.0103)[0]
    k3 = int(below3[0]) if len(below3) else len(A) - 1

    return {
        "coherent_gain": float(np.mean(w)),
        "enbw_bins": float(N * np.sum(w ** 2) / np.sum(w) ** 2),
        "mainlobe_null_bins": float(2 * bins[first_null]),
        "bw3db_bins": float(2 * bins[k3]),
        "peak_sidelobe_db": float(np.max(dB[first_null:])),
    }
