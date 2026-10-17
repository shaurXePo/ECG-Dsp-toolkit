"""
Step 4 -- Butterworth band-pass (SOS), bilinear transform + pre-warping, notch,
causal vs zero-phase filtering and the phase distortion it causes on the QRS.

Outputs (results/figures/)
  04_iir_bilinear_prewarp.png   frequency-warping curve; designs with and without pre-warping
  04_iir_design.png             effect of order; SOS vs single (b,a) polynomial
  04_phase_distortion_qrs.png   one beat through FIR / IIR causal / IIR filtfilt
  04_notch_demo.png             mains hum removed by the notch (PSD before/after)
"""
from _common import base_parser, get_record, out_dirs, plt, savefig, tag
import numpy as np
from scipy import signal

from src import analysis, iir, noise, pipeline, psd

FS = 360.0


def naive_bandpass_sos(f1, f2, fs, order):
    """Same recipe as iir.butter_bandpass_zpk but WITHOUT pre-warping (W = 2 pi f)."""
    z, p, k = iir.butter_lowpass_prototype_zpk(order)
    W1, W2 = 2 * np.pi * f1, 2 * np.pi * f2
    z, p, k = iir.lp2bp_zpk(z, p, k, np.sqrt(W1 * W2), W2 - W1)
    z, p, k = iir.bilinear_zpk(z, p, k, fs)
    return signal.zpk2sos(z, p, k)


def edges_3db(sos, fs):
    f, H = analysis.freqz_sos(sos, fs, n=2 ** 16)
    m = analysis.mag_db(H)
    inside = np.where(m >= -3.0103)[0]
    return f[inside[0]], f[inside[-1]]


def fig_prewarp(fig_dir):
    f = np.linspace(0.01, FS / 2 - 0.01, 2000)
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    ax[0].plot(f, iir.prewarp(f, FS), label=r"pre-warped  $\Omega = 2 f_s \tan(\pi f/f_s)$")
    ax[0].plot(f, 2 * np.pi * f, "k--", label=r"linear  $\Omega = 2\pi f$")
    ax[0].set(title="Bilinear transform warps frequency", xlabel="digital f [Hz]", ylabel=r"analogue $\Omega$ [rad/s]",
              ylim=(0, 3000))
    ax[0].legend()

    # a wide band so the warping is visible: intended -3 dB edges 20 and 150 Hz
    f1, f2, order = 20.0, 150.0, 3
    good = iir.butter_bandpass_sos(f1, f2, FS, order)
    bad = naive_bandpass_sos(f1, f2, FS, order)
    for sos, lab, col in ((good, "with pre-warping", "tab:blue"), (bad, "WITHOUT pre-warping", "tab:red")):
        fr, H = analysis.freqz_sos(sos, FS, n=8192)
        ax[1].plot(fr, analysis.mag_db(H), color=col, label=lab)
        lo, hi = edges_3db(sos, FS)
        print(f"  {lab:>20s}: intended edges {f1:g}/{f2:g} Hz -> measured -3 dB edges {lo:.2f}/{hi:.2f} Hz")
    ax[1].axvline(f1, color="k", ls=":")
    ax[1].axvline(f2, color="k", ls=":")
    ax[1].axhline(-3.01, color="0.6", ls=":")
    ax[1].set(title=f"Intended -3 dB edges {f1:g}/{f2:g} Hz (dotted)", xlabel="Frequency [Hz]", ylabel="dB",
              ylim=(-40, 3))
    ax[1].legend()

    sos = iir.butter_bandpass_sos(0.5, 40.0, FS, 3)
    fr, H = analysis.freqz_sos(sos, FS, n=2 ** 16)
    ax[2].semilogx(fr[1:], analysis.mag_db(H)[1:])
    ax[2].axvline(0.5, color="k", ls=":")
    ax[2].axvline(40.0, color="k", ls=":")
    ax[2].axhline(-3.01, color="0.6", ls=":")
    ax[2].set(title="Our ECG design (0.5-40 Hz, order 3 -> 6th-order BP): -3 dB at both edges",
              xlabel="Frequency [Hz] (log)", ylabel="dB", ylim=(-60, 3))
    fig.tight_layout()
    savefig(fig, fig_dir, "04_iir_bilinear_prewarp.png")


def fig_order_and_sos(fig_dir):
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    for order in (1, 2, 3, 4, 5):
        sos = iir.butter_bandpass_sos(0.5, 40.0, FS, order)
        f, H = analysis.freqz_sos(sos, FS, n=16384)
        fg, gd = analysis.group_delay_sos(sos, FS, n=16384)
        ax[0].semilogx(f[1:], analysis.mag_db(H)[1:], label=f"order {order} ({2 * order}th-order BP)")
        ax[1].semilogx(fg[1:], gd[1:] * 1000 / FS, label=f"order {order}")
    ax[0].set(title="Higher order: steeper skirts", xlabel="Frequency [Hz]", ylabel="dB", ylim=(-80, 3))
    ax[0].legend(fontsize=8)
    ax[1].set(title="...but more group delay (and ringing)", xlabel="Frequency [Hz]", ylabel="ms", ylim=(0, 1500))
    ax[1].legend(fontsize=8)

    # SOS vs single polynomial
    w = np.linspace(0.05, 179, 20000)
    print("  SOS vs (b,a) form, max deviation of the response in dB (0.5-40 Hz band-pass):")
    for order in (2, 3, 4, 6, 8):
        sos = signal.butter(order, [0.5, 40.0], "bandpass", fs=FS, output="sos")
        b, a = signal.butter(order, [0.5, 40.0], "bandpass", fs=FS)
        _, Hs = signal.sosfreqz(sos, worN=w, fs=FS)
        _, Hb = signal.freqz(b, a, worN=w, fs=FS)
        dev = np.abs(analysis.mag_db(Hb, 1e-12) - analysis.mag_db(Hs, 1e-12))
        mid = (w > 1) & (w < 100)
        print(f"    order {order}: max |dB(b,a) - dB(sos)| for 1-100 Hz = {dev[mid].max():.3e} dB;"
              f"  max |pole| of ba form = {np.max(np.abs(np.roots(a))):.9f}")
        ax[2].semilogy(w[mid], dev[mid] + 1e-16, lw=0.8, label=f"order {order}")
    ax[2].set(title="Deviation of the single-polynomial (b,a) form from SOS", xlabel="Frequency [Hz]", ylabel="dB")
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, fig_dir, "04_iir_design.png")


def fig_phase_distortion(args, fig_dir):
    rec = get_record(args, "100", duration_s=60.0)
    x, fs = rec["signal"], rec["fs"]
    bank = pipeline.build_filter_bank(fs, args.mains_hz)
    outs = {m: bank.apply(m, x) for m in pipeline.METHODS}
    x0 = x - x.mean()

    # pick a beat away from the edges (annotation or detected)
    beats = rec["ann_sample"]
    beats = beats[(beats > 15 * fs) & (beats < 45 * fs)]
    r = int(beats[len(beats) // 2])
    lo, hi = r - int(0.35 * fs), r + int(0.55 * fs)
    t = (np.arange(lo, hi) - r) / fs * 1000

    fig, ax = plt.subplots(1, 2, figsize=(16, 5))
    ax[0].plot(t, x0[lo:hi], color="0.6", lw=3, label="input (DC removed)")
    cols = {"FIR": "tab:green", "IIR (causal)": "tab:red", "IIR (causal, lag-corrected)": "tab:orange",
            "IIR (filtfilt)": "tab:blue"}
    print("  R-peak position shift and shape error (one beat; positive = later):")
    for m, y in outs.items():
        ax[0].plot(t, y[lo:hi], color=cols[m], lw=1.2, label=m)
        seg_in, seg_out = x0[lo:hi], y[lo:hi]
        shift_ms = (np.argmax(seg_out) - np.argmax(seg_in)) / fs * 1000
        print(f"    {m:<15s} peak shift {shift_ms:+6.1f} ms")
    ax[0].set(title=f"{tag(args)}Record {rec['name']}, one beat: waveform after each filter",
              xlabel="Time re. R peak [ms]", ylabel="mV")
    ax[0].legend()

    # difference to the linear-phase FIR output: where the causal IIR bends the waveform
    ref = outs["FIR"][lo:hi]
    for m in ("IIR (causal)", "IIR (causal, lag-corrected)", "IIR (filtfilt)"):
        ax[1].plot(t, outs[m][lo:hi] - ref, color=cols[m], label=f"{m} - FIR")
    ax[1].set(title="Difference to the linear-phase FIR output (same beat)",
              xlabel="Time re. R peak [ms]", ylabel="mV")
    ax[1].legend()
    fig.tight_layout()
    savefig(fig, fig_dir, "04_phase_distortion_qrs.png")


def fig_notch(args, fig_dir):
    rec = get_record(args, "100", duration_s=120.0)
    x, fs = rec["signal"], rec["fs"]
    hum = noise.mains_interference(len(x), fs, args.mains_hz, seed=args.seed)
    noisy, _, _ = noise.mix_at_snr(x, hum, 12.0)

    sos_bp = iir.butter_bandpass_sos(0.5, 40.0, fs, 3)
    sos_all = iir.design_iir_chain(fs, 0.5, 40.0, 3, args.mains_hz)
    y_bp = iir.sosfilt_causal(sos_bp, noisy - noisy.mean())
    y_all = iir.sosfilt_causal(sos_all, noisy - noisy.mean())

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    for sig, lab, col in ((noisy, "noisy (hum at 12 dB SNR)", "0.6"), (y_bp, "band-pass only", "tab:orange"),
                          (y_all, "band-pass + notch", "tab:blue")):
        f, P = psd.welch(sig, fs, nperseg=2048)
        ax[0].semilogy(f, P, color=col, label=lab, lw=1.0)
    ax[0].axvline(args.mains_hz, color="r", ls=":", lw=0.8)
    ax[0].set(title=f"{tag(args)}PSD around mains ({args.mains_hz:g} Hz)", xlabel="Frequency [Hz]",
              ylabel="mV$^2$/Hz", xlim=(0, fs / 2))
    ax[0].legend()
    f, H = analysis.freqz_sos(iir.notch_sos(args.mains_hz, fs), fs, n=2 ** 15)
    ax[1].plot(f, analysis.mag_db(H))
    ax[1].set(title=f"Notch frequency response (f0 = {args.mains_hz:g} Hz, Q = 30)", xlabel="Frequency [Hz]",
              ylabel="dB", xlim=(0, fs / 2), ylim=(-80, 3))
    fig.tight_layout()
    savefig(fig, fig_dir, "04_notch_demo.png")


def main():
    args = base_parser(__doc__).parse_args()
    _, fig_dir = out_dirs(args)
    print("Pre-warping");        fig_prewarp(fig_dir)
    print("Order / SOS");        fig_order_and_sos(fig_dir)
    print("Phase distortion");   fig_phase_distortion(args, fig_dir)
    print("Notch");              fig_notch(args, fig_dir)


if __name__ == "__main__":
    main()
