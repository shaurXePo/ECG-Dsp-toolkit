"""
iir.py -- Butterworth band-pass (second-order sections), 50/60 Hz notch, and the
two ways of applying an IIR filter: causal (sosfilt) and zero-phase (sosfiltfilt).

DESIGN RECIPE (what scipy.signal.butter does internally; implemented here by hand)
---------------------------------------------------------------------------------
1. Analogue low-pass PROTOTYPE with cut-off 1 rad/s.  Butterworth = maximally
   flat magnitude  |H(jW)|^2 = 1 / (1 + W^(2N)).  Its N poles sit on the left half
   of the unit circle in the s-plane:

       p_k = exp( j pi (2k + N - 1) / (2N) ),   k = 1..N       (no zeros)

2. PRE-WARP the digital band edges.  The bilinear transform
       s = 2 fs (z - 1) / (z + 1)
   maps the whole analogue axis W in (-inf, inf) onto the digital circle once,
   at the price of a non-linear frequency map
       W = 2 fs tan(w/2) = 2 fs tan(pi f / fs)
   (f small -> W ~ 2 pi f: almost linear; f -> fs/2 -> W -> infinity: strong
   compression).  Without pre-warping the -3 dB edges would land at the wrong
   digital frequencies, increasingly so near Nyquist.  We therefore compute the
   ANALOGUE edges W1, W2 from the wanted DIGITAL edges f1, f2 with the formula
   above, so that after the bilinear map they come back exactly to f1, f2.

3. LOW-PASS -> BAND-PASS transformation in the analogue domain:
       s -> (s^2 + W0^2) / (BW s),    W0 = sqrt(W1 W2),   BW = W2 - W1
   Each prototype pole p becomes TWO poles (roots of s^2 - BW p s + W0^2 = 0) and
   N zeros appear at s = 0.  An N-th order prototype -> a 2N-th order band-pass.

4. BILINEAR TRANSFORM of every pole/zero:  z = (2 fs + s) / (2 fs - s).
   Zeros at s = 0 land at z = +1 (kills DC); the N "missing" zeros at s = infinity
   land at z = -1 (kills Nyquist).  Gain is fixed so the pass-band gain is 1.

5. SECOND-ORDER SECTIONS (SOS).  A band-pass with edges at 0.5 Hz while fs = 360 Hz
   has poles within ~0.009 rad of z = 1.  As ONE high-order polynomial H = B(z)/A(z)
   the coefficients are so ill-conditioned that rounding error can move poles
   outside the unit circle.  Factoring into biquads (pairs of poles + pairs of
   zeros) keeps each polynomial tiny and well-conditioned.

NOTCH.  H(z) = b0 (1 - 2cos(w0) z^-1 + z^-2) / (1 - 2 r cos(w0) z^-1 + r^2 z^-2):
   zeros ON the unit circle at +/- w0 (exact null), poles just inside at radius r.
   The -3 dB width is BW = f0 / Q; the pole radius r is set by BW.

ZERO-PHASE FILTERING.  filtfilt runs the filter forward, reverses the output, runs
   it again, reverses again.  Net response = |H(w)|^2 with ZERO phase, but:
   * the filter is NON-CAUSAL (output at time t depends on the future) -> offline only;
   * the -3 dB edges become -6 dB edges (magnitude is squared);
   * effective order doubles; the impulse response is symmetric (rings both ways).
   Causal filtering delays different frequencies by different amounts
   (non-constant group delay) -> the QRS shape is distorted.
"""
from __future__ import annotations

import numpy as np
from scipy import signal

__all__ = [
    "prewarp",
    "butter_lowpass_prototype_zpk",
    "lp2bp_zpk",
    "bilinear_zpk",
    "butter_bandpass_zpk",
    "butter_bandpass_sos",
    "notch_sos",
    "design_iir_chain",
    "sosfilt_causal",
    "sosfilt_zero_phase",
]


# --------------------------------------------------------------------------- #
# from-scratch design pieces
# --------------------------------------------------------------------------- #
def prewarp(f_hz, fs: float):
    """Analogue frequency (rad/s) that the bilinear transform maps to f_hz."""
    return 2.0 * fs * np.tan(np.pi * np.asarray(f_hz, dtype=float) / fs)


def butter_lowpass_prototype_zpk(order: int):
    """Normalised (1 rad/s) analogue Butterworth low-pass: zeros, poles, gain."""
    k = np.arange(1, order + 1)
    theta = np.pi * (2 * k + order - 1) / (2 * order)
    p = np.exp(1j * theta)                         # all have Re < 0
    return np.array([], dtype=complex), p, 1.0


def lp2bp_zpk(z, p, k, w0: float, bw: float):
    """Analogue low-pass -> band-pass: s -> (s^2 + w0^2) / (bw s)."""
    z = np.asarray(z, dtype=complex)
    p = np.asarray(p, dtype=complex)
    degree = len(p) - len(z)
    z_lp = z * bw / 2.0
    p_lp = p * bw / 2.0
    # each LP root r gives two BP roots  r +/- sqrt(r^2 - w0^2)
    z_bp = np.concatenate([z_lp + np.sqrt(z_lp ** 2 - w0 ** 2),
                           z_lp - np.sqrt(z_lp ** 2 - w0 ** 2)])
    p_bp = np.concatenate([p_lp + np.sqrt(p_lp ** 2 - w0 ** 2),
                           p_lp - np.sqrt(p_lp ** 2 - w0 ** 2)])
    z_bp = np.append(z_bp, np.zeros(degree))       # zeros at s = 0 (DC)
    return z_bp, p_bp, k * bw ** degree


def bilinear_zpk(z, p, k, fs: float):
    """Bilinear transform of zeros/poles/gain: s = 2 fs (z-1)/(z+1)."""
    z = np.asarray(z, dtype=complex)
    p = np.asarray(p, dtype=complex)
    fs2 = 2.0 * fs
    degree = len(p) - len(z)
    z_z = (fs2 + z) / (fs2 - z)
    p_z = (fs2 + p) / (fs2 - p)
    z_z = np.append(z_z, -np.ones(degree))         # zeros at s = inf  ->  z = -1
    k_z = k * np.real(np.prod(fs2 - z) / np.prod(fs2 - p))
    return z_z, p_z, k_z


def butter_bandpass_zpk(f_low: float, f_high: float, fs: float, order: int):
    """Digital Butterworth band-pass (order = prototype order; filter order 2*order)."""
    if not 0 < f_low < f_high < fs / 2:
        raise ValueError("need 0 < f_low < f_high < fs/2")
    z, p, k = butter_lowpass_prototype_zpk(order)
    W1, W2 = prewarp([f_low, f_high], fs)          # <- pre-warping
    w0, bw = np.sqrt(W1 * W2), W2 - W1
    z, p, k = lp2bp_zpk(z, p, k, w0, bw)
    return bilinear_zpk(z, p, k, fs)


def butter_bandpass_sos(f_low: float, f_high: float, fs: float, order: int = 3,
                        from_scratch: bool = True) -> np.ndarray:
    """
    Butterworth band-pass as an (order x 6) SOS array.
    from_scratch=True  : our zpk design above, then scipy's zpk2sos (pairing only)
    from_scratch=False : scipy.signal.butter (reference)
    The two give the same frequency response (tested).
    """
    if from_scratch:
        z, p, k = butter_bandpass_zpk(f_low, f_high, fs, order)
        return signal.zpk2sos(z, p, k)
    return signal.butter(order, [f_low, f_high], btype="bandpass", fs=fs, output="sos")


def notch_sos(f0: float, fs: float, Q: float = 30.0) -> np.ndarray:
    """
    Second-order IIR notch at f0 Hz (50 or 60 Hz mains).  Q = f0 / (-3 dB width);
    Q = 30 -> 2 Hz wide at 60 Hz, a compromise between attenuating mains drift
    (mains frequency wanders ~ +/-0.2 Hz) and distorting ECG content near f0.
    """
    if not 0 < f0 < fs / 2:
        raise ValueError("notch frequency must be inside (0, fs/2)")
    b, a = signal.iirnotch(f0, Q, fs=fs)
    return signal.tf2sos(b, a)


def design_iir_chain(fs: float, f_low: float = 0.5, f_high: float = 40.0, order: int = 3,
                     notch_hz: float | None = 60.0, notch_q: float = 30.0) -> np.ndarray:
    """Band-pass biquads followed (optionally) by a notch biquad, as one SOS array."""
    sos = butter_bandpass_sos(f_low, f_high, fs, order)
    if notch_hz is not None:
        sos = np.vstack([sos, notch_sos(notch_hz, fs, notch_q)])
    return sos


# --------------------------------------------------------------------------- #
# applying filters
# --------------------------------------------------------------------------- #
def sosfilt_causal(sos: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Causal (real-time) filtering.  Initial state is set to the steady state for a
    constant input x[0], which suppresses the start-up transient when x[0] != 0.
    """
    zi = signal.sosfilt_zi(sos) * x[0]
    y, _ = signal.sosfilt(sos, x, zi=zi)
    return y


def sosfilt_zero_phase(sos: np.ndarray, x: np.ndarray, padtype: str | None = "odd") -> np.ndarray:
    """Forward-backward filtering (zero phase, squared magnitude, offline only)."""
    return signal.sosfiltfilt(sos, x, padtype=padtype)
