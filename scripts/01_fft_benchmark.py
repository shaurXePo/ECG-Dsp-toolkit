"""
Step 1 -- own FFT: correctness against numpy.fft and speed benchmark.

Outputs
  results/figures/01_fft_vs_numpy_ecg.png
  results/figures/01_fft_benchmark.png
  results/fft_benchmark.csv
"""
import csv
import time

from _common import base_parser, get_record, out_dirs, plt, savefig, tag
import numpy as np

from src import fft as myfft


def best_time(fn, x, min_total=0.05, repeats=3) -> float:
    """Seconds per call: run until `min_total` s have elapsed, repeat, keep the fastest."""
    best = np.inf
    for _ in range(repeats):
        n_calls, t0 = 0, time.perf_counter()
        while True:
            fn(x)
            n_calls += 1
            dt = time.perf_counter() - t0
            if dt >= min_total:
                break
        best = min(best, dt / n_calls)
    return best


def main():
    args = base_parser(__doc__).parse_args()
    res_dir, fig_dir = out_dirs(args)
    rng = np.random.default_rng(args.seed)

    # ------------------------------------------------------------------ #
    # 1. correctness on random inputs
    # ------------------------------------------------------------------ #
    print("Correctness vs numpy.fft (max |error|, random complex input):")
    for k in range(1, 15):
        N = 2 ** k
        x = rng.standard_normal(N) + 1j * rng.standard_normal(N)
        ref = np.fft.fft(x)
        e_it = np.max(np.abs(myfft.fft(x, "iterative") - ref))
        e_re = np.max(np.abs(myfft.fft(x, "recursive") - ref)) if N <= 2 ** 12 else np.nan
        e_inv = np.max(np.abs(myfft.ifft(myfft.fft(x)) - x))
        assert np.allclose(myfft.fft(x), ref)
        print(f"  N=2^{k:<2d} iterative {e_it:.2e}   recursive {e_re:.2e}   ifft(fft(x))-x {e_inv:.2e}")

    # ------------------------------------------------------------------ #
    # 2. real ECG (non power of two -> zero padding)
    # ------------------------------------------------------------------ #
    rec = get_record(args, "100", duration_s=10.0)
    x, fs = rec["signal"], rec["fs"]            # 3600 samples, not a power of two
    N, M = len(x), myfft.next_pow2(len(x))
    X_mine = myfft.fft(x)
    X_np = np.fft.fft(x, n=M)
    ok = np.allclose(X_mine, X_np)
    print(f"ECG record {rec['name']}: N={N} -> padded to {M}; allclose(numpy) = {ok}; "
          f"max err = {np.max(np.abs(X_mine - X_np)):.2e}")
    assert ok

    f = myfft.rfftfreq(M, fs)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    ax[0].plot(np.arange(N) / fs, x, lw=0.8)
    ax[0].set(title=f"{tag(args)}ECG record {rec['name']} (10 s, N={N})", xlabel="Time [s]", ylabel="mV")
    mag_m = 20 * np.log10(np.abs(X_mine[: M // 2 + 1]) + 1e-12)
    mag_n = 20 * np.log10(np.abs(X_np[: M // 2 + 1]) + 1e-12)
    ax[1].plot(f, mag_n, lw=2.5, color="0.7", label="numpy.fft")
    ax[1].plot(f, mag_m, lw=0.8, color="tab:red", label="own FFT")
    ax[1].set(title=f"Magnitude spectrum (zero-padded to {M})", xlabel="Frequency [Hz]", ylabel="|X| [dB]")
    ax[1].legend()
    ax[2].semilogy(f, np.abs(X_mine - X_np)[: M // 2 + 1] + 1e-20, lw=0.7)
    ax[2].set(title="Absolute error |own - numpy|", xlabel="Frequency [Hz]", ylabel="error")
    fig.tight_layout()
    savefig(fig, fig_dir, "01_fft_vs_numpy_ecg.png")

    # ------------------------------------------------------------------ #
    # 3. benchmark
    # ------------------------------------------------------------------ #
    sizes = [2 ** k for k in range(3, 17)]          # 8 .. 65536
    algos = {                                       # (function, largest N we are willing to time)
        "naive DFT  O(N^2)": (myfft.dft_naive, 2 ** 11),
        "recursive radix-2": (myfft.fft_recursive, 2 ** 14),
        "iterative radix-2": (myfft.fft_iterative, 2 ** 16),
        "numpy.fft (C)": (np.fft.fft, 2 ** 16),
    }
    table = {name: {} for name in algos}
    print("\nBenchmark (seconds per transform):")
    for N in sizes:
        x = rng.standard_normal(N) + 1j * rng.standard_normal(N)
        row = []
        for name, (fn, nmax) in algos.items():
            if N <= nmax:
                table[name][N] = best_time(fn, x, min_total=0.03 if N < 2 ** 10 else 0.08)
                row.append(f"{table[name][N] * 1e3:9.3f} ms")
            else:
                row.append(f"{'-':>12}")
        print(f"  N={N:<6d}", "  ".join(row))

    with open(res_dir / "fft_benchmark.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["N"] + list(algos))
        for N in sizes:
            w.writerow([N] + [f"{table[n][N]:.6e}" if N in table[n] else "" for n in algos])

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    styles = {"naive DFT  O(N^2)": "tab:red", "recursive radix-2": "tab:orange",
              "iterative radix-2": "tab:blue", "numpy.fft (C)": "tab:green"}
    for name in algos:
        Ns = sorted(table[name])
        ax[0].loglog(Ns, [table[name][n] for n in Ns], "o-", color=styles[name], label=name)
        ax[1].semilogx(Ns, [table[name][n] / (n * np.log2(n)) * 1e9 for n in Ns], "o-",
                       color=styles[name], label=name)
    # Reference growth laws, anchored at the LARGEST N measured for that algorithm.  (Anchoring at a
    # small N would be misleading: there, Python/NumPy call overhead dominates the arithmetic.)
    Ns = np.array(sizes, dtype=float)
    n_a = max(table["naive DFT  O(N^2)"])
    ax[0].loglog(Ns, table["naive DFT  O(N^2)"][n_a] * (Ns / n_a) ** 2, "k:", lw=0.8,
                 label="N^2 law (anchored at naive's largest N)")
    n_b = max(table["iterative radix-2"])
    ax[0].loglog(Ns, table["iterative radix-2"][n_b] * (Ns * np.log2(Ns)) / (n_b * np.log2(n_b)),
                 "k--", lw=0.8, label="N log2 N law (anchored at iterative's largest N)")
    ax[0].set(title=f"{tag(args)}FFT time vs N", xlabel="N (log)", ylabel="seconds (log)")
    ax[0].legend(fontsize=8)
    ax[1].set(title="Time / (N log2 N)  [ns]  -- flat means O(N log N)", xlabel="N (log)", ylabel="ns")
    ax[1].set_yscale("log")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, fig_dir, "01_fft_benchmark.png")


if __name__ == "__main__":
    main()
