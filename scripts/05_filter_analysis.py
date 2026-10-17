"""
Step 5 -- analysis plots for EVERY filter in the project.

One 2x3 report per filter (magnitude dB, phase, group delay, pole-zero, impulse, step):
  05_report_fir_bandpass.png     windowed-sinc FIR (2377 taps)
  05_report_butter_bandpass.png  Butterworth band-pass alone (causal)
  05_report_notch.png            mains notch alone
  05_report_iir_causal.png       band-pass + notch chain, causal
  05_report_iir_filtfilt.png     same chain, forward-backward (zero phase)

FIR pole-zero note: the 2377-tap FIR has a degree-2376 polynomial; computing its roots is
numerically meaningless (tiny coefficient perturbations move roots a lot), so the zero plot
uses a 41-tap surrogate (same design code, 5-40 Hz, wide transition) to show the STRUCTURE
of linear-phase zeros: they come in mirror pairs z, 1/z*, and on-circle zeros give nulls.
"""
from _common import base_parser, out_dirs, plt, savefig
import numpy as np
from scipy import signal

from src import analysis, fir, iir, pipeline

FS = 360.0


def main():
    args = base_parser(__doc__).parse_args()
    _, fig_dir = out_dirs(args)
    bank = pipeline.build_filter_bank(FS, args.mains_hz)

    # surrogate short FIR for the zero plot
    h_short = fir.design_bandpass_fir(5.0, 40.0, FS, numtaps=41, window="hamming")
    z_short = np.roots(h_short)
    override = (z_short, np.zeros(0), "41-tap surrogate, 5-40 Hz")

    specs = bank.specs()
    extra = [
        {"name": f"Butterworth band-pass 0.5-40 Hz, order {bank.params['iir_order']} (causal)", "kind": "sos",
         "sos": iir.butter_bandpass_sos(0.5, 40.0, FS, bank.params["iir_order"]), "zero_phase": False,
         "t_max": 8.0, "slug": "butter_bandpass"},
        {"name": f"IIR notch {args.mains_hz:g} Hz, Q = 30 (causal)", "kind": "sos",
         "sos": iir.notch_sos(args.mains_hz, FS, 30.0), "zero_phase": False, "t_max": 2.0, "slug": "notch"},
    ]
    for spec in specs + extra:
        print(spec["name"])
        kw = {"zero_override": override} if spec["kind"] == "fir" else {}
        fig = analysis.plot_filter_report(spec, FS, **kw)
        savefig(fig, fig_dir, f"05_report_{spec['slug']}.png")

        if spec["kind"] == "sos":
            _, p, _ = signal.sos2zpk(spec["sos"])
            print(f"   stable (max |pole| = {np.max(np.abs(p)):.6f} < 1): {bool(np.all(np.abs(p) < 1))}")
        else:
            print(f"   {len(spec['h'])} taps, symmetric: {bool(np.allclose(spec['h'], spec['h'][::-1]))}")


if __name__ == "__main__":
    main()
