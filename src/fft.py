"""
fft.py -- hand-written radix-2 decimation-in-time (DIT) FFT.

THEORY
------
The DFT of an N-point sequence x[n] is

    X[k] = sum_{n=0}^{N-1} x[n] * W_N^{k n},     W_N = exp(-j 2 pi / N)

Computed directly this costs N complex multiplies per output bin -> O(N^2).

Radix-2 DIT splits x into its even-indexed and odd-indexed samples
(each of length N/2) and uses the fact that W_N^2 = W_{N/2}:

    X[k]       = E[k] + W_N^k * O[k]          k = 0 .. N/2-1
    X[k + N/2] = E[k] - W_N^k * O[k]

where E = DFT_{N/2}(x[0::2]) and O = DFT_{N/2}(x[1::2]).  The second line
follows from W_N^{k+N/2} = -W_N^k and the N/2-periodicity of E and O.
Applying this recursively gives log2(N) levels of N/2 "butterflies", i.e.
O(N log N) work.

Two implementations are provided:

* fft_recursive : a direct transcription of the two equations above.
* fft_iterative : the in-place version used in practice.  Unrolling the
  recursion shows that the leaves are the input in *bit-reversed* order, so we
  permute once, then run log2(N) passes ("stages") of butterflies of growing
  size 2, 4, 8, ..., N.

Both accept arrays of shape (..., N) and transform the LAST axis, which lets
psd.py compute all Welch segments in one call.

Zero padding: radix-2 needs N = 2^m.  `fft()` zero-pads to the next power of
two.  This does NOT change the underlying signal's spectrum; it samples the
same DTFT on a denser grid (interpolation), it adds no new information.

The inverse uses  x = conj( FFT( conj(X) ) ) / N,  so no new butterfly code is
needed.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "is_power_of_two",
    "next_pow2",
    "zero_pad_pow2",
    "bit_reverse_indices",
    "fft_recursive",
    "fft_iterative",
    "fft",
    "ifft",
    "rfft",
    "rfftfreq",
    "dft_naive",
]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def is_power_of_two(n: int) -> bool:
    """True iff n = 1, 2, 4, 8, ...  (a power of two has exactly one set bit)."""
    return n >= 1 and (n & (n - 1)) == 0


def next_pow2(n: int) -> int:
    """Smallest power of two >= n.  (n-1).bit_length() = ceil(log2 n) for n >= 2."""
    if n < 1:
        raise ValueError("n must be >= 1")
    return 1 << (int(n) - 1).bit_length()


def zero_pad_pow2(x, n: int | None = None) -> np.ndarray:
    """Zero-pad the last axis of x to length n (default: next power of two)."""
    x = np.asarray(x)
    N = x.shape[-1]
    M = next_pow2(N) if n is None else int(n)
    if not is_power_of_two(M):
        raise ValueError(f"target length {M} is not a power of two")
    if M < N:
        raise ValueError(f"target length {M} < signal length {N}")
    if M == N:
        return x
    pad_width = [(0, 0)] * (x.ndim - 1) + [(0, M - N)]
    return np.pad(x, pad_width)  # constant zeros by default


def _check_pow2(N: int) -> None:
    if not is_power_of_two(N):
        raise ValueError(
            f"length {N} is not a power of two; use fft(x) which zero-pads, "
            "or call zero_pad_pow2 first"
        )


def bit_reverse_indices(N: int) -> np.ndarray:
    """
    Index permutation i -> bit-reversal of i using log2(N) bits.
    Example N=8: 0,4,2,6,1,5,3,7.  This is the order in which the recursion's
    leaves visit the input samples.
    """
    _check_pow2(N)
    bits = N.bit_length() - 1          # log2(N)
    idx = np.arange(N)
    rev = np.zeros(N, dtype=np.int64)
    for b in range(bits):
        # take bit b of idx and place it at position (bits-1-b)
        rev |= ((idx >> b) & 1) << (bits - 1 - b)
    return rev


# --------------------------------------------------------------------------- #
# FFT implementations
# --------------------------------------------------------------------------- #
def fft_recursive(x) -> np.ndarray:
    """Recursive radix-2 DIT FFT of the last axis (length must be 2^m)."""
    x = np.asarray(x, dtype=np.complex128)
    N = x.shape[-1]
    _check_pow2(N)
    if N == 1:                                   # DFT of one sample is itself
        return x.copy()
    even = fft_recursive(x[..., 0::2])           # E[k], length N/2
    odd = fft_recursive(x[..., 1::2])            # O[k], length N/2
    w = np.exp(-2j * np.pi * np.arange(N // 2) / N)   # twiddles W_N^k
    t = w * odd
    return np.concatenate([even + t, even - t], axis=-1)   # the butterfly


def fft_iterative(x) -> np.ndarray:
    """
    Iterative in-place radix-2 DIT FFT of the last axis (length must be 2^m).

    Stage s (size = 2^s) combines pairs of already-computed half-size DFTs:
    each row of the reshaped array is one block of `size` samples whose first
    half is E and second half is O (thanks to the bit-reversal permutation).
    The textbook triple loop is

        for size in 2,4,...,N:
          for block_start in range(0, N, size):
            for k in range(size//2):
              t = W_size^k * a[block_start+k+size//2]
              a[block_start+k+size//2] = a[block_start+k] - t
              a[block_start+k]         = a[block_start+k] + t

    Here the two inner loops are vectorised with NumPy (same arithmetic).
    """
    x = np.asarray(x, dtype=np.complex128)
    N = x.shape[-1]
    _check_pow2(N)
    lead = x.shape[:-1]
    a = x[..., bit_reverse_indices(N)].copy()    # bit-reversed copy
    size = 2
    while size <= N:
        half = size // 2
        tw = np.exp(-2j * np.pi * np.arange(half) / size)       # W_size^k
        a = a.reshape(*lead, N // size, size)                    # blocks of `size`
        top = a[..., :half].copy()                               # E part
        bot = a[..., half:] * tw                                 # W^k * O part
        a[..., :half] = top + bot
        a[..., half:] = top - bot
        size *= 2
    return a.reshape(*lead, N)


def fft(x, method: str = "iterative", pad: bool = True) -> np.ndarray:
    """
    FFT along the last axis.

    pad=True  : zero-pad to the next power of two first (output length is the
                padded length).
    pad=False : length must already be a power of two.
    method    : "iterative" (default) or "recursive".
    """
    x = np.asarray(x)
    if pad:
        x = zero_pad_pow2(x)
    if method == "iterative":
        return fft_iterative(x)
    if method == "recursive":
        return fft_recursive(x)
    raise ValueError("method must be 'iterative' or 'recursive'")


def ifft(X, method: str = "iterative") -> np.ndarray:
    """
    Inverse FFT along the last axis; length must be a power of two.
    (Padding a *spectrum* with zeros is not a valid operation, so no pad option.)
    Uses  x = conj(FFT(conj(X))) / N.
    """
    X = np.asarray(X, dtype=np.complex128)
    N = X.shape[-1]
    return np.conj(fft(np.conj(X), method=method, pad=False)) / N


def rfft(x, method: str = "iterative") -> np.ndarray:
    """One-sided spectrum (bins 0 .. N/2) of a real signal, after zero-padding."""
    X = fft(np.asarray(x, dtype=float), method=method, pad=True)
    N = X.shape[-1]
    return X[..., : N // 2 + 1]


def rfftfreq(n: int, fs: float) -> np.ndarray:
    """Frequencies in Hz of the one-sided spectrum of an n-point FFT."""
    return np.arange(n // 2 + 1) * fs / n


# --------------------------------------------------------------------------- #
# O(N^2) reference
# --------------------------------------------------------------------------- #
def dft_naive(x) -> np.ndarray:
    """
    Direct O(N^2) DFT, valid for ANY length N (no padding).
    One output bin per loop iteration (O(N) memory).  NOTE: because each
    iteration is NumPy-vectorised this flatters the naive method's constant
    factor; a pure-Python double loop would be ~100x slower still.
    """
    x = np.asarray(x, dtype=np.complex128)
    N = x.shape[-1]
    n = np.arange(N)
    X = np.empty(x.shape, dtype=np.complex128)
    for k in range(N):
        X[..., k] = x @ np.exp(-2j * np.pi * k * n / N)
    return X
