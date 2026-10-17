"""
Step 6 -- ECG denoising experiments.

For every record x noise type x input SNR:
    noisy = clean + alpha * noise            (alpha from noise.noise_gain_for_snr, exact SNR)
    for each method in {FIR, IIR (causal), IIR (filtfilt)}:  filtered = method(noisy)
    metrics on the evaluation window (edges trimmed by --trim-s seconds to drop filter transients):
        SNR_in, SNR_out, dSNR, RMSE      vs the ORIGINAL record (DC removed)
        R-peak sensitivity / PPV / F1    vs the expert annotations, +/-150 ms tolerance
    plus detection on the unfiltered noisy signal ("none") and on the clean record ("clean").

Outputs
  results/metrics_per_record.csv        one row per (record, noise, SNR, method)
  results/figures/06_before_after_<noise>.png
  results/figures/06_snr_improvement.png
  results/figures/06_peak_f1.png
(then run 07_results_table.py to aggregate into the table.)

Notes
  * --duration: seconds of each record to use (default 300; --full uses the whole 30 min).
  * --with-mains adds a 4th, synthetic noise type: a sinusoid at --mains-hz (NSTDB has no mains hum),
    which is what the notch is for.
  * The "clean" record is MIT-BIH itself and already contains its own noise; SNR is relative to it.
"""
import csv

from _common import (NOISE_NAMES, NOISE_TYPES, SNRS, ROOT, base_parser, get_noise, get_record, out_dirs, plt,
                     savefig, tag)
import numpy as np

from src import io as ecgio
from src import metrics, noise, peaks, pipeline

TOL_S = 0.150                       # beat-matching tolerance (ANSI/AAMI EC57 uses 150 ms)
FIELDS = ["record", "noise", "snr_target", "snr_in_eval", "method", "snr_out", "d_snr", "rmse_mv",
          "tp", "fp", "fn", "se", "ppv", "f1", "n_ref"]


def detection_row(sig, fs, ref_peaks, cut):
    det = peaks.detect_r_peaks(sig, fs)
    det = det[(det >= cut.start) & (det < cut.stop)]
    tp, fp, fn = metrics.match_peaks(ref_peaks, det, int(round(TOL_S * fs)))
    return metrics.peak_scores(tp, fp, fn)


def main():
    p = base_parser(__doc__)
    p.add_argument("--records", nargs="+", default=ecgio.MITDB_RECORDS)
    p.add_argument("--snrs", nargs="+", type=float, default=SNRS)
    p.add_argument("--duration", type=float, default=300.0, help="seconds per record (default 300)")
    p.add_argument("--full", action="store_true", help="use the whole record (30 min)")
    p.add_argument("--trim-s", type=float, default=5.0, help="seconds dropped at both ends before scoring")
    p.add_argument("--with-mains", action="store_true", help="add synthetic mains hum as a 4th noise type")
    p.add_argument("--fig-record", default="100", help="record shown in the before/after figures")
    p.add_argument("--fig-snr", type=float, default=6.0, help="SNR shown in the before/after figures")
    args = p.parse_args()
    res_dir, fig_dir = out_dirs(args)
    duration = None if args.full else args.duration
    noise_types = NOISE_TYPES + (["mains"] if args.with_mains else [])

    rows, example = [], {}
    for ri, name in enumerate(args.records):
        rec = get_record(args, name, duration_s=duration)
        x, fs, ann = rec["signal"], rec["fs"], rec["ann_sample"]
        n = len(x)
        bank = pipeline.build_filter_bank(fs, args.mains_hz)
        cut = metrics.trim_slice(n, fs, args.trim_s)
        ref = ann[(ann >= cut.start) & (ann < cut.stop)]
        print(f"record {name}: {n / fs:.0f} s, {len(ref)} reference beats in the scored window")

        # ceiling: the detector on the clean record itself
        sc = detection_row(x, fs, ref, cut)
        rows.append(dict(record=name, noise="none", snr_target="", snr_in_eval="", method="clean",
                         snr_out="", d_snr="", rmse_mv=0.0, n_ref=len(ref), **sc))

        for ni, kind in enumerate(noise_types):
            if kind == "mains":
                nz = noise.mains_interference(n, fs, args.mains_hz, seed=args.seed + ri)
            else:
                nz = get_noise(args, kind, n, fs, key=100 * ri + ni + 1)
            for snr in args.snrs:
                noisy, _, _ = noise.mix_at_snr(x, nz, snr)
                snr_in = metrics.snr_db(x[cut], noisy[cut])
                base = dict(record=name, noise=kind, snr_target=snr, snr_in_eval=snr_in, n_ref=len(ref))

                sc = detection_row(noisy, fs, ref, cut)
                rows.append(dict(base, method="none", snr_out=snr_in, d_snr=0.0,
                                 rmse_mv=metrics.rmse(x[cut], noisy[cut]), **sc))
                outs = {}
                for method in pipeline.METHODS:
                    y = bank.apply(method, noisy)
                    outs[method] = y
                    sc = detection_row(y, fs, ref, cut)
                    s_out = metrics.snr_db(x[cut], y[cut])
                    rows.append(dict(base, method=method, snr_out=s_out, d_snr=s_out - snr_in,
                                     rmse_mv=metrics.rmse(x[cut], y[cut]), **sc))
                if name == args.fig_record and snr == args.fig_snr:
                    example[kind] = dict(x=x, noisy=noisy, outs=outs, fs=fs, cut=cut, ref=ref)
            print(f"   {kind}: done")

    with open(res_dir / "metrics_per_record.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {(res_dir / 'metrics_per_record.csv').relative_to(ROOT)}  ({len(rows)} rows)")

    # ------------------------------------------------------------------ figures
    cols = {"none": "0.5", "FIR": "tab:green", "IIR (causal)": "tab:red",
            "IIR (causal, lag-corrected)": "tab:orange", "IIR (filtfilt)": "tab:blue"}

    for kind, ex in example.items():
        fs, x = ex["fs"], ex["x"]
        c0 = ex["cut"].start + int(10 * fs)
        sl = slice(c0, c0 + int(8 * fs))
        t = np.arange(sl.start, sl.stop) / fs
        x0 = x - x.mean()
        panels = [("noisy (no filter)", ex["noisy"] - ex["noisy"].mean(), "none")] + \
                 [(m, ex["outs"][m], m) for m in pipeline.METHODS]
        fig, axes = plt.subplots(len(panels), 1, figsize=(14, 10), sharex=True)
        for ax, (title, y, key) in zip(axes, panels):
            ax.plot(t, x0[sl], color="0.75", lw=2.5, label="clean record (DC removed)")
            ax.plot(t, y[sl], color=cols[key], lw=0.9, label=title)
            det = peaks.detect_r_peaks(y, fs)
            det = det[(det >= sl.start) & (det < sl.stop)]
            ax.plot(det / fs, y[det], "kv", ms=5, label="detected R peaks")
            ax.set_ylabel("mV")
            ax.legend(loc="upper right", fontsize=8, ncol=3)
        axes[-1].set_xlabel("Time [s]")
        fig.suptitle(f"{tag(args)}Record {args.fig_record}, {NOISE_NAMES[kind]} at {args.fig_snr:g} dB input SNR")
        fig.tight_layout()
        savefig(fig, fig_dir, f"06_before_after_{kind}.png")

    # aggregate quickly for the summary plots (mean over records; Se/PPV micro-averaged)
    def agg(kind, snr, method):
        sel = [r for r in rows if r["noise"] == kind and r["snr_target"] == snr and r["method"] == method]
        d = np.mean([r["d_snr"] for r in sel])
        tp, fp, fn = (sum(r[k] for r in sel) for k in ("tp", "fp", "fn"))
        return d, metrics.peak_scores(tp, fp, fn)["f1"]

    fig1, ax1 = plt.subplots(1, len(noise_types), figsize=(5 * len(noise_types), 4.5), squeeze=False)
    fig2, ax2 = plt.subplots(1, len(noise_types), figsize=(5 * len(noise_types), 4.5), squeeze=False)
    for j, kind in enumerate(noise_types):
        for method in ("none",) + pipeline.METHODS:
            vals = [agg(kind, s, method) for s in args.snrs]
            if method != "none":
                ax1[0, j].plot(args.snrs, [v[0] for v in vals], "o-", color=cols[method], label=method)
            ax2[0, j].plot(args.snrs, [100 * v[1] for v in vals], "o-", color=cols[method],
                           label="no filter" if method == "none" else method)
        ax1[0, j].set(title=NOISE_NAMES[kind], xlabel="input SNR [dB]", ylabel="SNR improvement [dB]")
        ax1[0, j].axhline(0, color="k", lw=0.6)
        ax1[0, j].invert_xaxis()
        ax2[0, j].set(title=NOISE_NAMES[kind], xlabel="input SNR [dB]", ylabel="R-peak F1 [%]")
        ax2[0, j].invert_xaxis()
        ax1[0, j].legend(fontsize=8)
        ax2[0, j].legend(fontsize=8)
    fig1.suptitle(f"{tag(args)}SNR improvement (mean over {len(args.records)} records)")
    fig2.suptitle(f"{tag(args)}R-peak detection F1 vs input SNR (pooled over {len(args.records)} records)")
    fig1.tight_layout()
    fig2.tight_layout()
    savefig(fig1, fig_dir, "06_snr_improvement.png")
    savefig(fig2, fig_dir, "06_peak_f1.png")


if __name__ == "__main__":
    main()
