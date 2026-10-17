"""Exact SNR scaling for noise mixing."""
import numpy as np
import pytest

from src import io as ecgio
from src import noise

RNG = np.random.default_rng(99)


@pytest.mark.parametrize("snr", [24, 18, 12, 6, 0, -6, 3.5])
def test_mix_hits_target_snr_exactly(snr):
    s = RNG.standard_normal(5000) * 2.0 + 10.0      # with a big DC offset on purpose
    n = RNG.standard_normal(5000) * 0.3 - 4.0
    noisy, scaled, alpha = noise.mix_at_snr(s, n, snr)
    assert noise.measured_snr_db(s, scaled) == pytest.approx(snr, abs=1e-9)
    assert np.allclose(noisy, s + scaled)
    assert alpha > 0


def test_alpha_closed_form():
    s = RNG.standard_normal(1000)
    n = RNG.standard_normal(1000)
    a = noise.noise_gain_for_snr(s, n, 12.0)
    expected = np.sqrt(noise.signal_power(s) / (noise.signal_power(n) * 10 ** 1.2))
    assert a == pytest.approx(expected, rel=1e-12)


def test_equal_power_at_0_db():
    s = RNG.standard_normal(1000)
    n = 7.0 * RNG.standard_normal(1000)
    _, scaled, _ = noise.mix_at_snr(s, n, 0.0)
    assert noise.signal_power(scaled) == pytest.approx(noise.signal_power(s), rel=1e-12)


def test_snr_is_independent_of_noise_amplitude_and_dc():
    s = RNG.standard_normal(2000)
    n = RNG.standard_normal(2000)
    a1 = noise.noise_gain_for_snr(s, n, 6.0)
    a2 = noise.noise_gain_for_snr(s, 100.0 * n + 50.0, 6.0)
    assert 100.0 * a2 == pytest.approx(a1, rel=1e-12)


def test_errors():
    with pytest.raises(ValueError):
        noise.noise_gain_for_snr(np.ones(10), np.arange(10.0), 6)         # zero-power signal
    with pytest.raises(ValueError):
        noise.noise_gain_for_snr(np.arange(10.0), np.ones(10), 6)         # zero-power noise
    with pytest.raises(ValueError):
        noise.noise_gain_for_snr(np.arange(10.0), np.arange(11.0), 6)     # length mismatch


@pytest.mark.parametrize("kind", ["bw", "em", "ma"])
def test_synthetic_noise_is_seeded(kind):
    a = noise.synthetic_noise(kind, 4000, 360.0, seed=3)
    b = noise.synthetic_noise(kind, 4000, 360.0, seed=3)
    c = noise.synthetic_noise(kind, 4000, 360.0, seed=4)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_mixing_on_synthetic_ecg_all_snrs():
    rec = ecgio.synthetic_record(duration_s=60, seed=1)
    for kind in ("bw", "em", "ma"):
        nz = noise.synthetic_noise(kind, len(rec["signal"]), rec["fs"], seed=2)
        for snr in (24, 18, 12, 6, 0):
            _, scaled, _ = noise.mix_at_snr(rec["signal"], nz, snr)
            assert noise.measured_snr_db(rec["signal"], scaled) == pytest.approx(snr, abs=1e-9)


def test_mains_interference_frequency():
    x = noise.mains_interference(3600, 360.0, f0=60.0, seed=0)
    X = np.abs(np.fft.rfft(x))
    assert np.fft.rfftfreq(3600, 1 / 360.0)[np.argmax(X)] == pytest.approx(60.0)
