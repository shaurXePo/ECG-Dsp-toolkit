"""Filter-response tests: FIR (windowed sinc), IIR (Butterworth SOS), notch."""
import numpy as np
import pytest
from scipy import signal

from src import analysis, fir, iir

FS = 360.0


# ------------------------------------------------------------------ FIR ---- #
@pytest.fixture(scope="module")
def h():
    return fir.design_bandpass_fir(0.5, 40.0, FS, transition_hz=0.5, window="hamming")


def test_order_from_transition_width():
    # Hamming: M = ceil(3.3 * 360 / 0.5) = 2376 -> 2377 taps (odd)
    assert fir.estimate_numtaps(0.5, FS, "hamming") == 2377
    assert fir.estimate_numtaps(0.5, FS, "blackman") > fir.estimate_numtaps(0.5, FS, "hamming")
    assert fir.estimate_numtaps(1.0, FS, "hamming") < fir.estimate_numtaps(0.5, FS, "hamming")
    assert fir.estimate_numtaps(2.0, 250.0, "hann") % 2 == 1


def test_fir_matches_scipy_firwin(h):
    ref = signal.firwin(len(h), [0.5, 40.0], pass_zero=False, window="hamming", fs=FS)
    assert np.allclose(h, ref, atol=1e-12)


@pytest.mark.parametrize("window", ["hann", "hamming", "blackman"])
def test_fir_matches_firwin_other_windows(window):
    mine = fir.design_bandpass_fir(1.0, 30.0, 250.0, numtaps=401, window=window)
    ref = signal.firwin(401, [1.0, 30.0], pass_zero=False, window=window, fs=250.0)
    assert np.allclose(mine, ref, atol=1e-12)


def test_fir_is_symmetric_hence_linear_phase(h):
    assert np.allclose(h, h[::-1], atol=1e-15)


def test_fir_constant_group_delay(h):
    f, gd = analysis.group_delay_fir(h, FS, n=4096)
    band = (f > 2) & (f < 38)
    assert np.allclose(gd[band], (len(h) - 1) / 2, atol=1e-3)       # M/2 samples, i.e. 3.3 s


def test_fir_phase_is_linear(h):
    f, H = analysis.freqz_fir(h, FS)
    band = (f > 2) & (f < 38)
    expected = -2 * np.pi * f * ((len(h) - 1) / 2) / FS                 # -w M/2
    resid = np.angle(H[band] * np.exp(-1j * expected[band]))           # should be 0 or pi
    resid = np.abs(np.angle(np.exp(2j * resid))) / 2                   # fold +-pi ambiguity
    assert np.max(resid) < 1e-6


def test_fir_passband_stopband(h):
    f, H = analysis.freqz_fir(h, FS)
    m = analysis.mag_db(H)
    passband = (f >= 1.5) & (f <= 38.5)
    assert np.max(np.abs(m[passband])) < 0.25            # Hamming ripple ~ +/-0.17 dB
    # unity gain at the pass-band centre: evaluate the DTFT exactly at 20.25 Hz
    # (the FFT grid has no bin exactly there, so don't read it off the grid)
    n = np.arange(len(h))
    assert abs(np.sum(h * np.exp(-2j * np.pi * 20.25 * n / FS))) == pytest.approx(1.0, abs=1e-9)
    for f_stop in (0.0, 0.1, 50.0, 60.0, 100.0, 170.0):
        assert m[np.argmin(np.abs(f - f_stop))] < -45.0  # Hamming ~ 53 dB stop-band


def test_fir_6db_at_cutoffs(h):
    f, H = analysis.freqz_fir(h, FS)
    m = analysis.mag_db(H)
    assert m[np.argmin(np.abs(f - 40.0))] == pytest.approx(-6.0, abs=0.6)   # mid-transition


def test_fir_rejects_bad_args():
    with pytest.raises(ValueError):
        fir.design_bandpass_fir(40, 0.5, FS)
    with pytest.raises(ValueError):
        fir.design_bandpass_fir(0.5, 40, FS, numtaps=100)      # even length


def test_fir_application_delay_conventions(h):
    x = np.zeros(8000)
    x[4000] = 1.0
    zero_delay = fir.apply_fir_zero_delay(h, x)
    causal = fir.apply_fir_causal(h, x)
    assert np.argmax(zero_delay) == pytest.approx(4000, abs=3)       # peak where the impulse was
    M2 = (len(h) - 1) // 2
    # causal[n + M/2] == zero_delay[n]: the same output, just delayed by M/2 samples
    assert np.allclose(causal[M2:], zero_delay[: len(x) - M2], atol=1e-12)


# ------------------------------------------------------------------ IIR ---- #
@pytest.fixture(scope="module")
def sos():
    return iir.butter_bandpass_sos(0.5, 40.0, FS, order=3)


def test_scratch_design_matches_scipy_response():
    for order in (1, 2, 3, 4):
        mine = iir.butter_bandpass_sos(0.5, 40.0, FS, order, from_scratch=True)
        ref = iir.butter_bandpass_sos(0.5, 40.0, FS, order, from_scratch=False)
        w = np.concatenate([np.linspace(0.05, 179.9, 4000), [0.5, 40.0]])
        _, Hm = signal.sosfreqz(mine, worN=w, fs=FS)
        _, Hr = signal.sosfreqz(ref, worN=w, fs=FS)
        assert np.allclose(Hm, Hr, rtol=1e-6, atol=1e-9), f"order {order}"


def test_scratch_zpk_matches_scipy_zpk():
    z, p, k = iir.butter_bandpass_zpk(5.0, 45.0, 250.0, 3)
    zr, pr, kr = signal.butter(3, [5.0, 45.0], "bandpass", fs=250.0, output="zpk")
    assert np.allclose(np.sort_complex(np.round(p, 9)), np.sort_complex(np.round(pr, 9)))
    assert np.allclose(np.sort_complex(np.round(z, 6)), np.sort_complex(np.round(zr, 6)))
    assert k == pytest.approx(kr, rel=1e-9)


def test_prewarp_formula():
    # low frequencies: warping is nearly the identity (W ~ 2 pi f)
    assert iir.prewarp(0.5, FS) == pytest.approx(2 * np.pi * 0.5, rel=1e-4)
    # near Nyquist it blows up
    assert iir.prewarp(179.0, FS) > 20 * 2 * np.pi * 179.0


def test_bandpass_3db_edges_and_flat_passband(sos):
    f, H = analysis.freqz_sos(sos, FS, n=2 ** 16)
    m = analysis.mag_db(H)
    for edge in (0.5, 40.0):
        assert m[np.argmin(np.abs(f - edge))] == pytest.approx(-3.0103, abs=0.02)   # pre-warping works
    assert np.max(m) == pytest.approx(0.0, abs=0.01)                                 # Butterworth: max = 1
    assert m[np.argmin(np.abs(f - 0.0))] < -100                                      # DC null
    assert abs(m[np.argmin(np.abs(f - 10.0))]) < 0.1


def test_bandpass_stable(sos):
    _, p, _ = signal.sos2zpk(sos)
    assert np.all(np.abs(p) < 1.0)


def test_sos_beats_ba_form_numerically():
    """Single high-order polynomial form is much less accurate than SOS."""
    sos6 = iir.butter_bandpass_sos(0.5, 40.0, FS, order=6, from_scratch=False)
    b, a = signal.butter(6, [0.5, 40.0], "bandpass", fs=FS)
    w = np.linspace(0.1, 179, 2000)
    _, Hs = signal.sosfreqz(sos6, worN=w, fs=FS)
    _, Hba = signal.freqz(b, a, worN=w, fs=FS)
    # the SOS response is the trustworthy one: flat to 0 dB in mid-band
    assert abs(analysis.mag_db(Hs)[np.argmin(np.abs(w - 10))]) < 0.1
    # (we only assert the SOS is sane; the b,a deviation is printed by scripts/04)
    assert np.all(np.isfinite(Hba))


@pytest.mark.parametrize("f0,fs", [(60.0, 360.0), (50.0, 360.0), (50.0, 500.0), (50.0, 250.0)])
def test_notch_depth_and_passband(f0, fs):
    sos_n = iir.notch_sos(f0, fs, Q=30.0)
    f, H = analysis.freqz_sos(sos_n, fs, n=2 ** 16)
    m = analysis.mag_db(H)
    assert m[np.argmin(np.abs(f - f0))] < -50                       # deep null at f0
    assert m[np.argmin(np.abs(f - (f0 - 5.0)))] > -0.5               # barely touches neighbours
    assert m[np.argmin(np.abs(f - 10.0))] > -0.01
    # -3 dB bandwidth = f0 / Q
    below = f[m < -3.0103]
    assert (below.max() - below.min()) == pytest.approx(f0 / 30.0, rel=0.1)


def test_notch_frequency_validation():
    with pytest.raises(ValueError):
        iir.notch_sos(200.0, FS)


def test_chain_attenuates_mains_and_keeps_qrs_band():
    for mains in (50.0, 60.0):
        chain = iir.design_iir_chain(FS, 0.5, 40.0, 3, notch_hz=mains)
        f, H = analysis.freqz_sos(chain, FS, n=2 ** 16)
        m = analysis.mag_db(H)
        assert m[np.argmin(np.abs(f - mains))] < -60
        assert abs(m[np.argmin(np.abs(f - 15.0))]) < 0.2


def test_filtfilt_is_zero_phase_causal_is_not(sos):
    n = 6000
    x = np.zeros(n)
    x[n // 2] = 1.0
    y_zp = iir.sosfilt_zero_phase(sos, x, padtype=None)
    y_c = iir.sosfilt_causal(sos, x)
    assert np.argmax(np.abs(y_zp)) == n // 2                                  # no delay
    assert np.allclose(y_zp[n // 2 + 1: n // 2 + 400], y_zp[n // 2 - 1: n // 2 - 400: -1])   # symmetric
    assert np.argmax(np.abs(y_c)) >= n // 2                                   # causal: not before the impulse
    assert np.allclose(y_c[: n // 2], 0.0, atol=1e-12)


def test_filtfilt_squares_magnitude(sos):
    """Pass a sinusoid at the -3 dB edge: causal gain 0.707, filtfilt gain 0.5."""
    t = np.arange(int(60 * FS)) / FS
    x = np.sin(2 * np.pi * 40.0 * t)
    sl = slice(int(10 * FS), int(50 * FS))
    g_c = np.std(iir.sosfilt_causal(sos, x)[sl]) / np.std(x[sl])
    g_z = np.std(iir.sosfilt_zero_phase(sos, x)[sl]) / np.std(x[sl])
    assert g_c == pytest.approx(0.7071, abs=0.01)
    assert g_z == pytest.approx(0.5, abs=0.01)


def test_causal_group_delay_not_constant(sos):
    f, gd = analysis.group_delay_sos(sos, FS)
    band = (f > 1) & (f < 38)
    assert (gd[band].max() - gd[band].min()) / FS * 1000 > 10.0       # > 10 ms spread across the band


def test_analysis_impulse_step_shapes(sos, h):
    spec_c = {"kind": "sos", "sos": sos, "zero_phase": False, "t_max": 2.0}
    t, imp, stp = analysis.impulse_step(spec_c, FS)
    assert len(t) == len(imp) == len(stp) == int(2.0 * FS)
    assert np.allclose(stp, np.cumsum(imp), atol=1e-8)
    spec_z = {"kind": "sos", "sos": sos, "zero_phase": True, "t_max": 2.0}
    t, imp, stp = analysis.impulse_step(spec_z, FS)
    assert t[0] < 0 < t[-1]                                            # non-causal response visible
