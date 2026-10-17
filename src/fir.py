"""
fir.py -- windowed-sinc FIR band-pass designed from the ideal impulse response.

THEORY
------
1. IDEAL response.  A brick-wall low-pass with cut-off fc has the impulse
   response (inverse DTFT of a rectangle):

       h_lp[m] = sin(wc m) / (pi m) = (2 fc/fs) sinc(2 fc m / fs),   wc = 2 pi fc/fs

   (np.sinc(x) = sin(pi x)/(pi x)).  A band-pass [f1, f2] is the difference of two
   low-passes:   h_bp[m] = h_lp[m; f2] - h_lp[m; f1].
   It is infinitely long and non-causal (m runs from -inf to +inf).

2. TRUNCATE + SHIFT.  Keep N = M+1 taps and delay by M/2 samples so the response is
   causal:  m = n - M/2,  n = 0..M.   Plain truncation = multiplying by a
   rectangular window -> Gibbs ripple (~ -21 dB stop-band at best).

3. WINDOW.  Multiply by a tapered window (Hamming here).  Time-domain product =
   frequency-domain convolution, so the brick-wall edge is smoothed into a
   transition band of width

       delta_f  ~  K * fs / M        K = 0.9 (rect), 3.1 (Hann), 3.3 (Hamming), 5.5 (Blackman)

   and the stop-band attenuation is set by the window's sidelobes
   (~21, 44, 53, 74 dB for rect, Hann, Hamming, Blackman).  Hence the ORDER RULE

       M = ceil( K * fs / delta_f )

   Narrow transition bands are expensive: a 0.5 Hz transition at fs = 360 Hz with
   Hamming needs M ~ 2376 taps (6.6 s long!).  This is the price of FIR near DC.

4. LINEAR PHASE.  If h[n] = h[M-n] (symmetric, guaranteed by a symmetric window
   on a symmetric ideal response), then

       H(e^{jw}) = A(w) e^{-j w M/2},      A(w) real

   so the phase is exactly linear and the group delay is the constant M/2 samples
   at all frequencies: every frequency component is delayed equally, so the
   waveform shape (e.g. QRS) is preserved.  For ECG we remove the known delay
   by shifting the output (offline) -- see apply_fir_zero_delay.

5. GAIN NORMALISATION.  Scale so the amplitude response is exactly 1 at the
   centre of the pass-band (same convention as scipy.signal.firwin(scale=True)).

The cut-off frequencies are the -6 dB points (mid-transition), NOT -3 dB points.
"""
from __future__ import annotations

import numpy as np
from scipy import signal

from . import windows as _win

__all__ = [
    "WINDOW_TRANSITION_K",
    "estimate_numtaps",
    "ideal_lowpass_impulse",
    "ideal_bandpass_impulse",
    "design_bandpass_fir",
    "apply_fir_zero_delay",
    "apply_fir_causal",
]

# transition width = K * fs / M   (Oppenheim & Schafer, window design table)
WINDOW_TRANSITION_K = {"rectangular": 0.9, "hann": 3.1, "hamming": 3.3, "blackman": 5.5}


def estimate_numtaps(transition_hz: float, fs: float, window: str = "hamming") -> int:
    """
    Number of taps N = M + 1 from the transition-width rule M = K fs / delta_f.
    M is forced even so N is odd (Type-I linear phase: integer delay M/2 samples,
    and no forced zero at Nyquist).
    """
    if transition_hz <= 0:
        raise ValueError("transition_hz must be > 0")
    K = WINDOW_TRANSITION_K[window.lower()]
    M = int(np.ceil(K * fs / transition_hz))
    if M % 2:
        M += 1
    return M + 1


def ideal_lowpass_impulse(numtaps: int, fc: float, fs: float) -> np.ndarray:
    """Truncated, delayed ideal low-pass:  (2fc/fs) sinc(2 fc m / fs),  m = n - M/2."""
    m = np.arange(numtaps) - (numtaps - 1) / 2.0
    return (2.0 * fc / fs) * np.sinc(2.0 * fc * m / fs)


def ideal_bandpass_impulse(numtaps: int, f_low: float, f_high: float, fs: float) -> np.ndarray:
    """Ideal band-pass = LP(f_high) - LP(f_low)."""
    return ideal_lowpass_impulse(numtaps, f_high, fs) - ideal_lowpass_impulse(numtaps, f_low, fs)


def design_bandpass_fir(
    f_low: float,
    f_high: float,
    fs: float,
    numtaps: int | None = None,
    transition_hz: float = 0.5,
    window: str = "hamming",
    scale: bool = True,
) -> np.ndarray:
    """
    Windowed-sinc band-pass FIR (type-I, linear phase).

    numtaps=None -> derive from `transition_hz` with estimate_numtaps().
    """
    if not 0 < f_low < f_high < fs / 2:
        raise ValueError("need 0 < f_low < f_high < fs/2")
    if numtaps is None:
        numtaps = estimate_numtaps(transition_hz, fs, window)
    if numtaps % 2 == 0:
        raise ValueError("numtaps must be odd (type-I linear phase)")

    h = ideal_bandpass_impulse(numtaps, f_low, f_high, fs)
    h = h * _win.get_window(window, numtaps, sym=True)       # symmetric window!

    if scale:
        # amplitude response A(f) = sum h[n] cos(2 pi f (n - M/2)/fs)  (h symmetric)
        fc = 0.5 * (f_low + f_high)
        n = np.arange(numtaps)
        A = np.sum(h * np.cos(2 * np.pi * fc * (n - (numtaps - 1) / 2.0) / fs))
        h = h / A
    return h


def apply_fir_zero_delay(h: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Offline filtering with the filter's constant delay M/2 removed:
    out[n] = sum_k h[k] x[n + M/2 - k].  (fftconvolve 'same' centres the output.)
    Samples within M/2 of either end see implicit zero-padding (edge transient).
    """
    return signal.fftconvolve(x, h, mode="same")


def apply_fir_causal(h: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Real-time (causal) filtering: output is delayed by M/2 samples."""
    return signal.fftconvolve(x, h, mode="full")[: len(x)]
