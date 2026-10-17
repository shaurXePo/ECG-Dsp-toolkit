"""
Step 3 -- windowed-sinc FIR band-pass 0.5-40 Hz.

Outputs (results/figures/)
  03_fir_design.png        order vs transition width, impulse response, response vs scipy.firwin,
                           transition zoom, effect of the window (rectangular vs Hamming)
  03_fir_linear_phase.png  phase vs the straight line -w*M/2, constant group delay, symmetry
"""
from _common import base_parser, out_dirs, plt, savefig
import numpy as np
from scipy import signal

from src import analysis, fir, windows

FS = 360.0
F_LOW, F_HIGH = 0.5, 40.0
TRANSITION_HZ = 0.5


def main():
    args = base_parser(__doc__).parse_args()
    _, fig_dir = out_dirs(args)

    # ------------------------------------------------------------------ #
    # design + comparison with scipy
    # ------------------------------------------------------------------ #
    numtaps = fir.estimate_numtaps(TRANSITION_HZ, FS, "hamming")
    M = numtaps - 1
    h = fir.design_bandpass_fir(F_LOW, F_HIGH, FS, transition_hz=TRANSITION_HZ, window="hamming")
    h_sp = signal.firwin(numtaps, [F_LOW, F_HIGH], pass_zero=False, window="hamming", fs=FS)
    print(f"Design: Hamming, transition {TRANSITION_HZ} Hz at fs={FS:g} Hz -> M = ceil(3.3*{FS:g}/{TRANSITION_HZ}) "
          f"= {M} (order), {numtaps} taps")
    print(f"  filter length = {numtaps / FS:.2f} s, constant group delay = M/2 = {M // 2} samples = {M / 2 / FS:.3f} s")
    print(f"  max |h_own - h_firwin| = {np.max(np.abs(h - h_sp)):.2e}")
    print(f"  symmetry max |h[n]-h[M-n]| = {np.max(np.abs(h - h[::-1])):.2e}")
    for w in ("rectangular", "hann", "hamming", "blackman"):
        for tw in (0.5, 1.0, 2.0):
            print(f"  {w:<11s} transition {tw:>3.1f} Hz -> {fir.estimate_numtaps(tw, FS, w):>5d} taps")

    f, H = analysis.freqz_fir(h, FS)
    _, H_sp = analysis.freqz_fir(h_sp, FS)
    m, m_sp = analysis.mag_db(H), analysis.mag_db(H_sp)

    # a rectangular-window (plain truncation) version of the same length, to show Gibbs ripple
    h_rect = fir.ideal_bandpass_impulse(numtaps, F_LOW, F_HIGH, FS)
    _, H_rect = analysis.freqz_fir(h_rect, FS)

    fig, ax = plt.subplots(2, 3, figsize=(17, 9))

    # (a) order vs transition width
    tws = np.logspace(-0.5, 1.3, 50)
    for w, col in zip(("rectangular", "hann", "hamming", "blackman"),
                      ("tab:gray", "tab:blue", "tab:orange", "tab:green")):
        ax[0, 0].loglog(tws, [fir.estimate_numtaps(t, FS, w) for t in tws], color=col, label=w)
    ax[0, 0].axvline(TRANSITION_HZ, color="k", ls=":")
    ax[0, 0].plot([TRANSITION_HZ], [numtaps], "ko")
    ax[0, 0].annotate(f"chosen: {numtaps} taps", (TRANSITION_HZ, numtaps), (0.9, numtaps * 1.6), fontsize=9,
                      arrowprops=dict(arrowstyle="->"))
    ax[0, 0].set(title=f"Order from transition width: M = K fs/$\\Delta$f  (fs={FS:g} Hz)",
                 xlabel="Transition width $\\Delta$f [Hz]", ylabel="number of taps")
    ax[0, 0].legend(fontsize=8)

    # (b) impulse response
    t = (np.arange(numtaps) - M / 2) / FS
    ax[0, 1].plot(t, h_rect, color="0.7", lw=0.8, label="ideal truncated (rectangular)")
    ax[0, 1].plot(t, h, color="tab:blue", lw=1.0, label="x Hamming window")
    ax[0, 1].set(title="Impulse response (time re. centre tap)", xlabel="Time [s]", ylabel="h")
    ax[0, 1].legend(fontsize=8)

    # (c) magnitude, own vs firwin
    ax[0, 2].plot(f, m_sp, lw=3.5, color="tab:orange", alpha=0.5, label="scipy.signal.firwin")
    ax[0, 2].plot(f, m, lw=1.0, color="tab:blue", label="own windowed-sinc")
    ax[0, 2].set(title="Magnitude response (dB)", xlabel="Frequency [Hz]", ylabel="dB", xlim=(0, 100),
                 ylim=(-100, 5))
    ax[0, 2].legend(fontsize=8)

    # (d) transition zoom at the low edge
    ax[1, 0].plot(f, analysis.mag_db(H_rect), color="0.6", lw=1.0, label="rectangular (truncation)")
    ax[1, 0].plot(f, m, color="tab:blue", lw=1.2, label="Hamming")
    ax[1, 0].axvline(F_LOW - TRANSITION_HZ / 2, color="k", ls=":", lw=0.8)
    ax[1, 0].axvline(F_LOW + TRANSITION_HZ / 2, color="k", ls=":", lw=0.8)
    ax[1, 0].axvline(F_LOW, color="tab:red", ls="--", lw=0.8)
    ax[1, 0].axhline(-6.02, color="0.5", ls=":", lw=0.8)
    ax[1, 0].set(title=f"Low edge: transition {F_LOW - TRANSITION_HZ / 2:g}-{F_LOW + TRANSITION_HZ / 2:g} Hz "
                       f"(red = cut-off {F_LOW} Hz, at -6 dB)", xlabel="Frequency [Hz]", ylabel="dB",
                 xlim=(0, 1.5), ylim=(-90, 3))
    ax[1, 0].legend(fontsize=8)

    # (e) high edge zoom, linear scale -> passband ripple
    ax[1, 1].plot(f, np.abs(H), color="tab:blue")
    ax[1, 1].set(title="Pass-band in linear scale (ripple $\\approx \\pm 0.002$ for Hamming)",
                 xlabel="Frequency [Hz]", ylabel="|H|", xlim=(0, 50), ylim=(-0.05, 1.1))

    # (f) window comparison at fixed length: stop-band attenuation
    for wn, col in zip(("rectangular", "hann", "hamming", "blackman"), ("tab:gray", "tab:blue", "tab:orange", "tab:green")):
        hw = fir.design_bandpass_fir(F_LOW, F_HIGH, FS, numtaps=numtaps, window=wn)
        _, Hw = analysis.freqz_fir(hw, FS)
        ax[1, 2].plot(f, analysis.mag_db(Hw), color=col, lw=0.9, label=wn)
    ax[1, 2].set(title=f"Same length ({numtaps} taps), different window", xlabel="Frequency [Hz]",
                 ylabel="dB", xlim=(35, 100), ylim=(-120, 5))
    ax[1, 2].legend(fontsize=8)
    fig.suptitle(f"Windowed-sinc FIR band-pass {F_LOW}-{F_HIGH} Hz, fs = {FS:g} Hz")
    fig.tight_layout()
    savefig(fig, fig_dir, "03_fir_design.png")

    # ------------------------------------------------------------------ #
    # linear phase
    # ------------------------------------------------------------------ #
    band = (f > 1.5) & (f < 38.5)
    phase = np.unwrap(np.angle(H))
    line = -2 * np.pi * f * (M / 2) / FS                       # -w M/2 with w = 2 pi f/fs
    # remove the +-pi jumps of the sign of A(w): compare modulo pi
    dev = np.angle(np.exp(2j * (phase - line))) / 2

    fg, gd = analysis.group_delay_fir(h, FS, n=4096)

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    ax[0].plot(f[band], phase[band], lw=3.5, color="tab:orange", alpha=0.5, label="unwrapped phase of H")
    ax[0].plot(f[band], line[band], "k--", lw=1.0, label=r"$-\omega M/2$")
    ax[0].set(title="Phase is a straight line in the pass-band", xlabel="Frequency [Hz]", ylabel="rad")
    ax[0].legend()
    ax[1].plot(fg, gd / FS, color="tab:blue")
    ax[1].set(title=f"Group delay = M/2 = {M // 2} samples = {M / 2 / FS:.2f} s at every frequency",
              xlabel="Frequency [Hz]", ylabel="s", ylim=(0, 4))
    ax[2].plot(f[band], np.degrees(dev[band]), color="tab:red")
    ax[2].set(title="Deviation from exact linear phase (deg)", xlabel="Frequency [Hz]", ylabel="degrees")
    fig.suptitle("Symmetric taps  =>  linear phase  =>  constant delay  =>  waveform shape preserved")
    fig.tight_layout()
    savefig(fig, fig_dir, "03_fir_linear_phase.png")
    print(f"  max phase deviation from linear in the pass-band: {np.degrees(np.max(np.abs(dev[band]))):.3e} deg")


if __name__ == "__main__":
    main()
