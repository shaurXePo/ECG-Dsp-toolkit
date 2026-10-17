"""
Step 2 -- spectral analysis: windows, leakage demo, hand-written Welch PSD vs scipy,
noise-vs-ECG spectra, spectrogram.

Outputs (results/figures/)
  02_windows.png          window shapes + their spectra
  02_leakage_demo.png     resolution vs dynamic range with two close sinusoids + noise
  02_welch_vs_scipy.png   own Welch vs scipy.signal.welch (and a raw periodogram)
  02_noise_psd.png        PSD of ECG vs the three NSTDB noise types, with the 0.5-40 Hz band
  02_spectrogram.png      clean vs noisy ECG spectrogram
results/window_properties.csv
"""
import csv

from _common import (NOISE_NAMES, base_parser, get_noise, get_record, out_dirs, plt, savefig, tag)
import numpy as np
from scipy import signal

from src import fft as myfft
from src import noise, psd, windows


def fig_windows(args, res_dir, fig_dir):
    N = 64
    names = ["rectangular", "hann", "hamming", "blackman"]
    colors = dict(zip(names, ["tab:gray", "tab:blue", "tab:orange", "tab:green"]))
    nfft = 4096
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    rows = []
    for nm in names:
        w = windows.get_window(nm, N, sym=False)
        ax[0].plot(np.arange(N), w, "-", color=colors[nm], label=nm)
        W = myfft.fft(np.pad(w, (0, nfft - N)), pad=False)      # zero-padded -> smooth curve
        mag = 20 * np.log10(np.abs(W[: nfft // 2]) / np.abs(W[0]) + 1e-12)
        bins = np.arange(nfft // 2) * N / nfft                 # axis in DFT bins of the N-point DFT
        ax[1].plot(bins, mag, color=colors[nm], label=nm, lw=1.0)
        ax[2].plot(bins, mag, color=colors[nm], label=nm, lw=1.0)
        pr = windows.window_properties(w)
        rows.append({"window": nm, **{k: round(v, 3) for k, v in pr.items()}})
    ax[0].set(title="Window shapes (N = 64, periodic)", xlabel="n", ylabel="w[n]")
    ax[0].legend()
    ax[1].set(title="Window spectra (dB, zero-padded FFT)", xlabel="Frequency offset [bins]",
              ylabel="dB", xlim=(0, 16), ylim=(-100, 3))
    ax[2].set(title="Zoom on the main lobe", xlabel="Frequency offset [bins]", ylabel="dB",
              xlim=(0, 4), ylim=(-70, 3))
    ax[1].legend()
    fig.suptitle("Window trade-off: narrow main lobe (resolution) vs low sidelobes (dynamic range)")
    fig.tight_layout()
    savefig(fig, fig_dir, "02_windows.png")

    with open(res_dir / "window_properties.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print("  window properties:")
    for r in rows:
        print("   ", r)


def fig_leakage(args, fig_dir):
    fs, N = 360.0, 512                                   # bin spacing fs/N = 0.703 Hz
    t = np.arange(N) / fs
    rng = np.random.default_rng(args.seed)
    nz = 1e-4 * rng.standard_normal(N)                   # noise floor
    cases = [
        ("(a) RESOLUTION: two equal tones 2.3 bins apart\nf = 60.3 and 61.9 Hz",
         [(60.3, 1.0), (61.9, 1.0)]),
        ("(b) DYNAMIC RANGE: strong tone + tone 50 dB weaker, 11.7 bins apart\nf = 60.3 (A=1) and 68.5 Hz (A=0.003)",
         [(60.3, 1.0), (68.5, 0.003)]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, (title, tones) in zip(axes, cases):
        x = sum(a * np.sin(2 * np.pi * f * t + 0.7) for f, a in tones) + nz
        for nm, col in zip(["rectangular", "hann", "blackman"], ["tab:red", "tab:blue", "tab:green"]):
            w = windows.get_window(nm, N, sym=False)
            nfft = 8192                                   # heavy zero-padding: smooth curve, not new info
            X = myfft.fft(np.pad(x * w, (0, nfft - N)), pad=False)[: nfft // 2 + 1]
            mag = 20 * np.log10(np.abs(X) / (np.sum(w) / 2) + 1e-12)   # 0 dB = unit-amplitude sinusoid
            f = np.arange(nfft // 2 + 1) * fs / nfft
            ax.plot(f, mag, color=col, lw=1.0, label=nm)
        for f0, a in tones:
            ax.axvline(f0, color="k", ls=":", lw=0.7)
        ax.set(title=title, xlabel="Frequency [Hz]", ylabel="dB re. unit-amplitude tone",
               xlim=(50, 75), ylim=(-120, 5))
        ax.legend(loc="upper right")
    fig.suptitle(f"{tag(args)}Spectral leakage: N = {N}, fs = {fs:.0f} Hz, white noise sigma = 1e-4 "
                 "(dotted lines = true tone frequencies)")
    fig.tight_layout()
    savefig(fig, fig_dir, "02_leakage_demo.png")


def fig_welch(args, fig_dir, rec):
    x, fs = rec["signal"], rec["fs"]
    nperseg = 1024                                        # 2.84 s -> 0.35 Hz resolution
    f1, P1 = psd.welch(x, fs, nperseg=nperseg)
    f2, P2 = signal.welch(x, fs, nperseg=nperseg, nfft=nperseg)
    rel = np.max(np.abs(P1 - P2) / (np.abs(P2) + 1e-30))
    print(f"  Welch own vs scipy: max relative difference = {rel:.2e}")
    fp, Pp = psd.periodogram(x[: 8 * nperseg], fs)        # one raw periodogram on 8 segments' worth of data

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].semilogy(fp, Pp, color="0.75", lw=0.6, label="single periodogram (high variance)")
    ax[0].semilogy(f2, P2, lw=3, color="tab:orange", alpha=0.6, label="scipy.signal.welch")
    ax[0].semilogy(f1, P1, lw=1.0, color="tab:blue", label="own Welch")
    ax[0].set(title=f"{tag(args)}PSD of ECG record {rec['name']} (Hann, nperseg=1024, 50 % overlap)",
              xlabel="Frequency [Hz]", ylabel="PSD [mV$^2$/Hz]", xlim=(0, fs / 2))
    ax[0].legend()
    ax[1].semilogy(f1, np.abs(P1 - P2) / np.abs(P2) + 1e-18, lw=0.8)
    ax[1].set(title="Relative difference own vs scipy", xlabel="Frequency [Hz]", ylabel="|own-scipy|/scipy")
    fig.tight_layout()
    savefig(fig, fig_dir, "02_welch_vs_scipy.png")


def fig_noise_psd(args, fig_dir, rec):
    x, fs = rec["signal"], rec["fs"]
    n = len(x)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    f, P = psd.welch(x, fs, nperseg=1024)
    ax.semilogy(f, P, color="k", lw=1.5, label=f"ECG record {rec['name']}")
    for key, col in zip(["bw", "em", "ma"], ["tab:blue", "tab:orange", "tab:green"]):
        nz = get_noise(args, key, n, fs, key=1)
        _, scaled, _ = noise.mix_at_snr(x, nz, 6.0)     # compare at equal (6 dB) power
        f, P = psd.welch(scaled, fs, nperseg=1024)
        ax.semilogy(f, P, color=col, lw=1.0, label=f"{NOISE_NAMES[key]} ({key}), scaled to 6 dB SNR")
    ax.axvspan(0.5, 40.0, color="tab:red", alpha=0.08, label="pass-band 0.5-40 Hz")
    ax.axvline(args.mains_hz, color="tab:red", ls="--", lw=0.8, label=f"mains {args.mains_hz:g} Hz")
    ax.set(title=f"{tag(args)}Where does the noise live?", xlabel="Frequency [Hz]",
           ylabel="PSD [mV$^2$/Hz]", xlim=(0, fs / 2))
    ax.legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, fig_dir, "02_noise_psd.png")


def fig_spectrogram(args, fig_dir):
    rec = get_record(args, "100", duration_s=30.0)
    x, fs = rec["signal"], rec["fs"]
    n = len(x)
    # noisy version: baseline wander 12 dB + muscle 12 dB + mains hum 18 dB (each relative to the clean ECG)
    noisy = x.copy()
    for key, snr, k in (("bw", 12.0, 1), ("ma", 12.0, 2)):
        noisy = noisy + noise.mix_at_snr(x, get_noise(args, key, n, fs, key=k), snr)[1]
    noisy = noisy + noise.mix_at_snr(x, noise.mains_interference(n, fs, args.mains_hz), 18.0)[1]

    fig, ax = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
    for a, sig, title in ((ax[0], x, "clean"), (ax[1], noisy, f"noisy: bw 12 dB + ma 12 dB + {args.mains_hz:g} Hz hum 18 dB")):
        f, t, S = psd.spectrogram(sig, fs, nperseg=256, noverlap=192)     # 0.71 s windows, 75 % overlap
        Sdb = 10 * np.log10(S + 1e-20)
        im = a.pcolormesh(t, f, Sdb, shading="auto", vmin=Sdb.max() - 80, vmax=Sdb.max(), cmap="magma")
        a.set(title=f"{tag(args)}{title}", xlabel="Time [s]", ylim=(0, 100))
        a.grid(False)
        fig.colorbar(im, ax=a, label="dB")
    ax[0].set_ylabel("Frequency [Hz]")
    fig.suptitle("Spectrogram (own implementation, Hann 256 samples, 75 % overlap)")
    fig.tight_layout()
    savefig(fig, fig_dir, "02_spectrogram.png")


def main():
    args = base_parser(__doc__).parse_args()
    res_dir, fig_dir = out_dirs(args)
    rec = get_record(args, "100", duration_s=None if not args.synthetic else 300.0)
    print("Windows");           fig_windows(args, res_dir, fig_dir)
    print("Leakage demo");      fig_leakage(args, fig_dir)
    print("Welch");             fig_welch(args, fig_dir, rec)
    print("Noise PSD");         fig_noise_psd(args, fig_dir, get_record(args, "100", duration_s=300.0))
    print("Spectrogram");       fig_spectrogram(args, fig_dir)


if __name__ == "__main__":
    main()
