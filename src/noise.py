"""
noise.py -- mixing noise into a clean signal at an exact target SNR.

THEORY
------
Define power as the mean square of the DC-removed signal (= variance):

    P(x) = (1/N) sum (x[n] - mean(x))^2

(We remove the mean because MIT-BIH records carry an arbitrary DC offset that
says nothing about signal content, and the band-pass removes DC anyway.)

    SNR_dB = 10 log10( P_signal / P_noise )

To hit a target SNR we scale the noise by alpha:

    P(alpha n) = alpha^2 P(n)    =>    alpha = sqrt( P_s / ( P_n * 10^(SNR/10) ) )

so that   10 log10( P_s / (alpha^2 P_n) ) = SNR  exactly (up to floating point).
The noisy signal is  y = s + alpha n.
"""
from __future__ import annotations

import numpy as np
from scipy import signal

__all__ = [
    "signal_power",
    "noise_gain_for_snr",
    "mix_at_snr",
    "measured_snr_db",
    "synthetic_noise",
    "mains_interference",
]


def signal_power(x) -> float:
    """Mean-square of the DC-removed signal."""
    x = np.asarray(x, dtype=float)
    return float(np.mean((x - x.mean()) ** 2))


def noise_gain_for_snr(clean, noise, snr_db: float) -> float:
    """alpha such that clean + alpha*noise has exactly `snr_db` dB SNR."""
    clean = np.asarray(clean, dtype=float)
    noise = np.asarray(noise, dtype=float)
    if clean.shape != noise.shape:
        raise ValueError(f"shape mismatch: clean {clean.shape} vs noise {noise.shape}")
    p_s, p_n = signal_power(clean), signal_power(noise)
    if p_s <= 0:
        raise ValueError("clean signal has zero power")
    if p_n <= 0:
        raise ValueError("noise has zero power")
    return float(np.sqrt(p_s / (p_n * 10.0 ** (snr_db / 10.0))))


def mix_at_snr(clean, noise, snr_db: float):
    """Return (noisy, scaled_noise, alpha)."""
    alpha = noise_gain_for_snr(clean, noise, snr_db)
    scaled = alpha * np.asarray(noise, dtype=float)
    return np.asarray(clean, dtype=float) + scaled, scaled, alpha


def measured_snr_db(clean, scaled_noise) -> float:
    """SNR actually present in (clean, scaled_noise), same definition as above."""
    return 10.0 * np.log10(signal_power(clean) / signal_power(scaled_noise))


# --------------------------------------------------------------------------- #
# synthetic stand-ins (tests / offline smoke runs ONLY -- never used for the
# reported results, which use the real NSTDB records)
# --------------------------------------------------------------------------- #
def synthetic_noise(kind: str, n: int, fs: float, seed: int = 0) -> np.ndarray:
    """
    Crude look-alikes of the three NSTDB noise types:
      bw : baseline wander  - a few sinusoids 0.1-0.4 Hz + slow random drift
      em : electrode motion - low-pass (<~8 Hz) noise with bursty amplitude
      ma : muscle artefact  - 20-120 Hz band-limited noise, slowly varying envelope
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    if kind == "bw":
        x = sum(rng.uniform(0.5, 1.0) * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
                for f in (0.12, 0.2, 0.33, 0.4))
        walk = np.cumsum(rng.standard_normal(n))
        sos = signal.butter(2, 0.3, "low", fs=fs, output="sos")
        return x + 0.5 * signal.sosfilt(sos, walk) / (np.std(walk) + 1e-12) * 5
    if kind == "em":
        sos = signal.butter(2, 8.0, "low", fs=fs, output="sos")
        base = signal.sosfilt(sos, rng.standard_normal(n))
        env_sos = signal.butter(1, 0.2, "low", fs=fs, output="sos")
        env = np.exp(2.0 * signal.sosfilt(env_sos, rng.standard_normal(n)) / 0.1)
        return base * env / env.mean()
    if kind == "ma":
        sos = signal.butter(2, [20.0, min(120.0, 0.45 * fs)], "bandpass", fs=fs, output="sos")
        base = signal.sosfilt(sos, rng.standard_normal(n))
        env_sos = signal.butter(1, 0.3, "low", fs=fs, output="sos")
        env = 1.0 + 0.5 * np.tanh(signal.sosfilt(env_sos, rng.standard_normal(n)) / 0.15)
        return base * env
    raise ValueError("kind must be 'bw', 'em' or 'ma'")


def mains_interference(n: int, fs: float, f0: float = 60.0, seed: int = 0) -> np.ndarray:
    """Unit-amplitude sinusoid at the mains frequency with a random phase."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    return np.sin(2 * np.pi * f0 * t + rng.uniform(0, 2 * np.pi))
