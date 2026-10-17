"""FFT correctness: against numpy.fft, round trip, padding, batching, real ECG."""
import numpy as np
import pytest

from src import fft as myfft
from src import io as ecgio

RNG = np.random.default_rng(1234)
POW2 = [1, 2, 4, 8, 16, 64, 256, 1024, 4096]


@pytest.mark.parametrize("N", POW2)
@pytest.mark.parametrize("method", ["recursive", "iterative"])
def test_matches_numpy_complex(N, method):
    x = RNG.standard_normal(N) + 1j * RNG.standard_normal(N)
    assert np.allclose(myfft.fft(x, method=method), np.fft.fft(x))


@pytest.mark.parametrize("N", POW2)
@pytest.mark.parametrize("method", ["recursive", "iterative"])
def test_matches_numpy_real(N, method):
    x = RNG.standard_normal(N)
    assert np.allclose(myfft.fft(x, method=method), np.fft.fft(x))


@pytest.mark.parametrize("N", POW2)
def test_inverse_roundtrip(N):
    x = RNG.standard_normal(N) + 1j * RNG.standard_normal(N)
    assert np.allclose(myfft.ifft(myfft.fft(x)), x)
    assert np.allclose(myfft.ifft(np.fft.fft(x)), x)       # inverse of numpy's spectrum too


@pytest.mark.parametrize("N", [3, 5, 100, 1000, 1500])
def test_zero_padding_to_next_power_of_two(N):
    x = RNG.standard_normal(N)
    M = myfft.next_pow2(N)
    X = myfft.fft(x)
    assert X.shape[-1] == M
    assert np.allclose(X, np.fft.fft(x, n=M))              # numpy pads the same way


def test_pad_false_rejects_non_power_of_two():
    with pytest.raises(ValueError):
        myfft.fft(np.ones(12), pad=False)
    with pytest.raises(ValueError):
        myfft.ifft(np.ones(12))


def test_next_pow2_and_is_power_of_two():
    assert [myfft.next_pow2(n) for n in (1, 2, 3, 4, 5, 1023, 1024, 1025)] == [1, 2, 4, 4, 8, 1024, 1024, 2048]
    assert myfft.is_power_of_two(1024) and not myfft.is_power_of_two(1000) and not myfft.is_power_of_two(0)


def test_bit_reversal_n8():
    assert myfft.bit_reverse_indices(8).tolist() == [0, 4, 2, 6, 1, 5, 3, 7]


def test_naive_dft_matches_numpy_any_length():
    for N in (7, 12, 100):
        x = RNG.standard_normal(N)
        assert np.allclose(myfft.dft_naive(x), np.fft.fft(x))


def test_batched_last_axis():
    x = RNG.standard_normal((5, 3, 64))
    assert np.allclose(myfft.fft(x), np.fft.fft(x, axis=-1))


def test_parseval():
    x = RNG.standard_normal(512)
    X = myfft.fft(x)
    assert np.isclose(np.sum(np.abs(x) ** 2), np.sum(np.abs(X) ** 2) / len(X))


def test_rfft_and_freqs():
    x = RNG.standard_normal(256)
    assert np.allclose(myfft.rfft(x), np.fft.rfft(x))
    assert np.allclose(myfft.rfftfreq(256, 360.0), np.fft.rfftfreq(256, 1 / 360.0))


def test_matches_numpy_on_ecg():
    """Real MIT-BIH excerpt if downloaded, else the synthetic ECG (labelled)."""
    if ecgio.data_available():
        x = ecgio.load_record("100", duration_s=30)["signal"]
    else:
        x = ecgio.synthetic_record(duration_s=30)["signal"]
    assert not np.isnan(x).any()
    assert np.allclose(myfft.fft(x), np.fft.fft(x, n=myfft.next_pow2(len(x))))
    assert np.allclose(myfft.fft(x, "recursive"), np.fft.fft(x, n=myfft.next_pow2(len(x))))
