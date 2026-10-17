"""
analysis.py -- frequency-domain and time-domain analysis of the designed filters.

A filter is described by a small dict ("spec"):

    {"name": str,
     "kind": "fir" | "sos",
     "h":   FIR taps            (kind == "fir")
     "sos": second-order sections (kind == "sos")
     "zero_phase": bool         (kind == "sos": analyse forward-backward filtering)
     "t_max": float             (seconds of impulse/step response to plot)}

Quantities
----------
* Magnitude (dB) = 20 log10 |H(f)|
* Phase  = unwrap(angle H)
* Group delay tau(w) = -d phi / d w   [samples]  -> the delay seen by a narrow-band
  component at w.  Constant tau  <=>  linear phase  <=>  no waveform distortion.
  For a cascade, group delays ADD.
* Pole-zero plot: poles inside the unit circle <=> stable.  Zeros ON the circle
  <=> exact nulls (DC, Nyquist, notch).
* Impulse response h[n] = response to a unit sample; step response s[n] = cumsum h[n].
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy import signal

from .fft import fft, rfftfreq, zero_pad_pow2

__all__ = [
    "freqz_fir",
    "freqz_sos",
    "mag_db",
    "group_delay_fir",
    "group_delay_sos",
    "zeros_poles",
    "impulse_step",
    "plot_pole_zero",
    "plot_filter_report",
]


def mag_db(H, floor: float = 1e-10) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.abs(H), floor))


def freqz_fir(h, fs: float, nfft: int = 2 ** 15):
    """
    Frequency response of an FIR = DTFT sampled on nfft points = FFT of the
    zero-padded taps (our own FFT).  Zero-padding does not change the filter; it
    just evaluates its response on a finer grid.
    """
    X = fft(zero_pad_pow2(np.asarray(h, dtype=float), nfft), pad=False)
    return rfftfreq(nfft, fs), X[: nfft // 2 + 1]


def freqz_sos(sos, fs: float, n: int = 8192):
    w, H = signal.sosfreqz(sos, worN=n, fs=fs)
    return w, H


def group_delay_fir(h, fs: float, n: int = 8192):
    """(f [Hz], group delay [samples]) of an FIR."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w, gd = signal.group_delay((np.asarray(h, float), [1.0]), w=n, fs=fs)
    return w, gd


def group_delay_sos(sos, fs: float, n: int = 8192):
    """Group delay of a cascade = sum of the sections' group delays (samples)."""
    total = np.zeros(n)
    f = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")       # scipy warns at exact notch nulls
        for sec in np.atleast_2d(sos):
            f, gd = signal.group_delay((sec[:3], sec[3:]), w=n, fs=fs)
            total += gd
    return f, total


def zeros_poles(spec: dict):
    """Zeros and poles.  FIR zeros are only meaningful for short filters."""
    if spec["kind"] == "sos":
        z, p, _ = signal.sos2zpk(spec["sos"])
        return z, p
    z = np.roots(spec["h"])
    return z, np.zeros(0)


def impulse_step(spec: dict, fs: float):
    """(t [s], impulse, step) responses of the spec as it would be applied."""
    t_max = spec.get("t_max", 10.0)
    if spec["kind"] == "fir":
        h = np.asarray(spec["h"], float)
        t = np.arange(len(h)) / fs
        return t, h, np.cumsum(h)
    sos = spec["sos"]
    n = int(t_max * fs)
    if spec.get("zero_phase", False):
        # put the impulse in the middle so the NON-CAUSAL part is visible
        x = np.zeros(2 * n + 1)
        x[n] = 1.0
        imp = signal.sosfiltfilt(sos, x, padtype=None)
        stp = signal.sosfiltfilt(sos, np.concatenate([np.zeros(n), np.ones(n + 1)]), padtype=None)
        return (np.arange(2 * n + 1) - n) / fs, imp, stp
    x = np.zeros(n)
    x[0] = 1.0
    imp = signal.sosfilt(sos, x)
    stp = signal.sosfilt(sos, np.ones(n))
    return np.arange(n) / fs, imp, stp


def plot_pole_zero(ax, z, p, title: str = "Pole-zero plot") -> None:
    """Pole-zero diagram; repeated roots are annotated with their multiplicity."""
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), "k--", lw=0.8)
    ax.axhline(0, color="0.7", lw=0.5)
    ax.axvline(0, color="0.7", lw=0.5)
    for roots, marker, col, lab in ((z, "o", "tab:blue", "zeros"), (p, "x", "tab:red", "poles")):
        if len(roots):
            ax.plot(roots.real, roots.imag, marker, mfc="none", color=col, ms=7, label=lab)
            keys = np.round(roots, 3)
            uniq, counts = np.unique(keys, return_counts=True)
            for u, c in zip(uniq, counts):
                if c > 1:
                    ax.annotate(f"x{c}", (u.real, u.imag), textcoords="offset points",
                                xytext=(6, 6), fontsize=8, color=col)
    ax.set_aspect("equal")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel("Re z")
    ax.set_ylabel("Im z")
    ax.set_title(title)
    ax.legend(loc="lower left", fontsize=8)


def plot_filter_report(spec: dict, fs: float, fmax: float = 100.0, show_zeros: bool = True,
                       zero_override: tuple | None = None):
    """
    2x3 report: magnitude, phase, group delay, pole-zero, impulse, step.
    `zero_override=(z, p, note)` lets the caller supply roots from a shorter surrogate
    (used for the 2377-tap FIR whose degree-2376 polynomial cannot be root-found reliably).
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    name = spec["name"]

    # ---- frequency response ----
    if spec["kind"] == "fir":
        f, H = freqz_fir(spec["h"], fs)
        fg, gd = group_delay_fir(spec["h"], fs)
    else:
        f, H = freqz_sos(spec["sos"], fs, n=16384)
        fg, gd = group_delay_sos(spec["sos"], fs)
        if spec.get("zero_phase", False):
            H = np.abs(H) ** 2                  # |H|^2, zero phase
            gd = np.zeros_like(gd)

    m = mag_db(H)
    ax[0, 0].plot(f, m)
    ax[0, 0].set_xlim(0, fmax)
    ax[0, 0].set_ylim(-100, 5)
    ax[0, 0].axhline(-3, color="0.6", ls=":", lw=0.8)
    ax[0, 0].set(title="Magnitude response", xlabel="Frequency [Hz]", ylabel="|H| [dB]")
    ax[0, 0].grid(alpha=0.3)

    ph = np.unwrap(np.angle(H))
    ph = np.where(m > -40, ph, np.nan)          # phase is meaningless in deep stop-bands
    ax[0, 1].plot(f, ph)
    ax[0, 1].set_xlim(0, fmax)
    ax[0, 1].set(title="Phase (unwrapped, where |H| > -40 dB)", xlabel="Frequency [Hz]",
                 ylabel="rad")
    ax[0, 1].grid(alpha=0.3)

    ax[0, 2].plot(fg, gd * 1000.0 / fs)
    ax[0, 2].set_xlim(0, fmax)
    ax[0, 2].set(title="Group delay", xlabel="Frequency [Hz]", ylabel="ms")
    ax[0, 2].grid(alpha=0.3)

    # ---- pole-zero ----
    if zero_override is not None:
        z, p, note = zero_override
        plot_pole_zero(ax[1, 0], z, p, f"Pole-zero ({note})")
    elif spec["kind"] == "fir" and len(spec["h"]) > 600:
        ax[1, 0].text(0.5, 0.5, "root-finding on a degree-%d polynomial\nis ill-conditioned: not plotted"
                      % (len(spec["h"]) - 1), ha="center", va="center", transform=ax[1, 0].transAxes)
        ax[1, 0].set_axis_off()
    elif show_zeros:
        z, p = zeros_poles(spec)
        plot_pole_zero(ax[1, 0], z, p)

    # ---- time domain ----
    t, imp, stp = impulse_step(spec, fs)
    ax[1, 1].plot(t, imp, lw=0.8)
    ax[1, 1].set(title="Impulse response", xlabel="Time [s]", ylabel="h")
    ax[1, 1].grid(alpha=0.3)
    ax[1, 2].plot(t, stp, lw=0.8)
    ax[1, 2].set(title="Step response", xlabel="Time [s]", ylabel="s")
    ax[1, 2].grid(alpha=0.3)

    fig.suptitle(name, fontsize=14)
    fig.tight_layout()
    return fig
