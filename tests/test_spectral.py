"""Windows, periodogram, Welch and spectrogram vs scipy.signal."""
import numpy as np
import pytest
from scipy import signal
from scipy.signal import windows as sw

from src import psd, windows

RNG = np.random.default_rng(7)
_trapz = getattr(np, "trapezoid", None) or np.trapz   # numpy 2.x / 1.x


@pytest.mark.parametrize("name,ref", [("hann", sw.hann), ("hamming", sw.hamming),
                                      ("blackman", sw.blackman), ("rectangular", sw.boxcar)])
@pytest.mark.parametrize("N", [8, 33, 128])
@pytest.mark.parametrize("sym", [True, False])
def test_windows_match_scipy(name, ref, N, sym):
    assert np.allclose(windows.get_window(name, N, sym=sym), ref(N, sym=sym))


def test_symmetric_window_is_symmetric():
    for name in ("hann", "hamming", "blackman"):
        w = windows.get_window(name, 101, sym=True)
        assert np.allclose(w, w[::-1])


def test_window_sidelobes_ordering():
    """Textbook numbers: peak sidelobe ~ -13, -31, -43, -58 dB (loose tolerances)."""
    props = {n: windows.window_properties(windows.get_window(n, 128, sym=False))
             for n in ("rectangular", "hann", "hamming", "blackman")}
    assert props["rectangular"]["peak_sidelobe_db"] == pytest.approx(-13.3, abs=0.5)
    assert props["hann"]["peak_sidelobe_db"] == pytest.approx(-31.5, abs=1.0)
    assert props["hamming"]["peak_sidelobe_db"] == pytest.approx(-42.7, abs=1.5)
    assert props["blackman"]["peak_sidelobe_db"] == pytest.approx(-58.0, abs=2.0)
    # main lobe widens as sidelobes drop
    assert (props["rectangular"]["mainlobe_null_bins"] < props["hann"]["mainlobe_null_bins"]
            < props["blackman"]["mainlobe_null_bins"])


@pytest.mark.parametrize("nperseg", [128, 256, 1024])
@pytest.mark.parametrize("window", ["hann", "hamming", "blackman"])
@pytest.mark.parametrize("detrend", ["constant", None])
def test_welch_matches_scipy(nperseg, window, detrend):
    fs = 360.0
    x = RNG.standard_normal(20000) + 3.0
    f1, P1 = psd.welch(x, fs, window=window, nperseg=nperseg, detrend=detrend)
    f2, P2 = signal.welch(x, fs, window=window, nperseg=nperseg, nfft=nperseg,
                          detrend=detrend if detrend else False)
    assert np.allclose(f1, f2)
    assert np.allclose(P1, P2, rtol=1e-9, atol=1e-14)


def test_welch_spectrum_scaling_and_overlap():
    fs = 360.0
    x = RNG.standard_normal(10000)
    f1, P1 = psd.welch(x, fs, nperseg=512, noverlap=384, scaling="spectrum")
    f2, P2 = signal.welch(x, fs, nperseg=512, noverlap=384, nfft=512, scaling="spectrum")
    assert np.allclose(P1, P2, rtol=1e-9, atol=1e-14)


def test_welch_zero_padded_nfft():
    x = RNG.standard_normal(5000)
    f1, P1 = psd.welch(x, 100.0, nperseg=200, nfft=512)
    f2, P2 = signal.welch(x, 100.0, nperseg=200, nfft=512)
    assert np.allclose(P1, P2, rtol=1e-9, atol=1e-14)


def test_welch_integrates_to_variance():
    """PSD density scaling: integral of PSD over frequency ~ signal power."""
    fs = 360.0
    x = RNG.standard_normal(100000)
    f, P = psd.welch(x, fs, nperseg=1024)
    assert _trapz(P, f) == pytest.approx(np.var(x), rel=0.03)


def test_periodogram_matches_scipy():
    x = RNG.standard_normal(1024)
    f1, P1 = psd.periodogram(x, 360.0, window="hann")
    f2, P2 = signal.periodogram(x, 360.0, window="hann", nfft=1024)
    assert np.allclose(P1, P2, rtol=1e-9, atol=1e-14)


def test_spectrogram_matches_scipy():
    fs = 360.0
    x = RNG.standard_normal(6000)
    f1, t1, S1 = psd.spectrogram(x, fs, nperseg=256, noverlap=192)
    f2, t2, S2 = signal.spectrogram(x, fs, window="hann", nperseg=256, noverlap=192, nfft=256)
    assert np.allclose(f1, f2) and np.allclose(t1, t2)
    assert np.allclose(S1, S2, rtol=1e-9, atol=1e-14)


def test_welch_finds_tone():
    fs = 360.0
    t = np.arange(36000) / fs
    x = np.sin(2 * np.pi * 45.0 * t) + 0.1 * RNG.standard_normal(len(t))
    f, P = psd.welch(x, fs, nperseg=1024)
    assert abs(f[np.argmax(P)] - 45.0) <= fs / 1024
