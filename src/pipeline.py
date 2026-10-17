"""
pipeline.py -- the three filtering strategies compared in the experiments.

    "FIR"             windowed-sinc band-pass, linear phase, constant delay M/2 removed
    "IIR (causal)"    Butterworth band-pass + notch, sosfilt (real-time, phase distortion)
    "IIR (causal, lag-corrected)"
                      DIAGNOSTIC: the causal output shifted earlier by the filter's median
                      pass-band group delay (5-30 Hz).  A constant shift cannot undo the NON-constant
                      group delay, so what remains is pure waveform (shape) distortion.
    "IIR (filtfilt)"  same SOS applied forward-backward (zero phase, offline)

All three start by subtracting the signal mean: the pass-bands exclude DC anyway,
and removing the (large) MIT-BIH offset avoids a giant step at the record edges,
where FIR convolution implicitly zero-pads.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import analysis as _an
from . import fir as _fir
from . import iir as _iir

__all__ = ["FilterBank", "build_filter_bank", "METHODS"]

METHODS = ("FIR", "IIR (causal)", "IIR (causal, lag-corrected)", "IIR (filtfilt)")


@dataclass
class FilterBank:
    fs: float
    h: np.ndarray                 # FIR taps
    sos: np.ndarray               # IIR band-pass (+ notch) sections
    params: dict = field(default_factory=dict)
    iir_lag: int = 0              # median pass-band group delay of the IIR chain [samples]

    def apply(self, method: str, x) -> np.ndarray:
        x0 = np.asarray(x, dtype=float)
        x0 = x0 - x0.mean()
        if method == "FIR":
            return _fir.apply_fir_zero_delay(self.h, x0)
        if method == "IIR (causal)":
            return _iir.sosfilt_causal(self.sos, x0)
        if method == "IIR (causal, lag-corrected)":
            y = _iir.sosfilt_causal(self.sos, x0)
            return np.concatenate([y[self.iir_lag:], np.full(self.iir_lag, y[-1])])
        if method == "IIR (filtfilt)":
            return _iir.sosfilt_zero_phase(self.sos, x0)
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")

    def specs(self) -> list[dict]:
        """Filter descriptions for analysis.plot_filter_report."""
        return [
            {"name": "FIR band-pass (windowed sinc)", "kind": "fir", "h": self.h, "slug": "fir_bandpass"},
            {"name": "IIR Butterworth band-pass + notch (causal)", "kind": "sos", "sos": self.sos,
             "zero_phase": False, "t_max": 8.0, "slug": "iir_causal"},
            {"name": "IIR Butterworth band-pass + notch (filtfilt, zero phase)", "kind": "sos",
             "sos": self.sos, "zero_phase": True, "t_max": 8.0, "slug": "iir_filtfilt"},
        ]


def build_filter_bank(fs: float, mains_hz: float | None = 60.0, f_low: float = 0.5,
                      f_high: float = 40.0, fir_transition_hz: float = 0.5,
                      fir_window: str = "hamming", iir_order: int = 3,
                      notch_q: float = 30.0) -> FilterBank:
    """
    mains_hz : notch frequency of the IIR chain.  60.0 for MIT-BIH (US recordings);
               50.0 for Indian / European mains; None to disable the notch.
    """
    h = _fir.design_bandpass_fir(f_low, f_high, fs, transition_hz=fir_transition_hz,
                                 window=fir_window)
    sos = _iir.design_iir_chain(fs, f_low, f_high, iir_order, mains_hz, notch_q)
    params = dict(fs=fs, mains_hz=mains_hz, f_low=f_low, f_high=f_high,
                  fir_transition_hz=fir_transition_hz, fir_window=fir_window,
                  numtaps=len(h), iir_order=iir_order, notch_q=notch_q)
    f, gd = _an.group_delay_sos(sos, fs)
    lag = int(round(np.median(gd[(f >= 5.0) & (f <= 30.0)])))
    params["iir_lag_samples"] = lag
    return FilterBank(fs=fs, h=h, sos=sos, params=params, iir_lag=lag)
