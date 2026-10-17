"""
Step 7 -- results table: FIR vs IIR (causal) vs IIR (filtfilt) across noise types and SNRs.

Reads  results/metrics_per_record.csv  (written by 06_ecg_experiments.py)  and writes
  results/summary_by_condition.csv
  results/results_table.md          three tables: SNR improvement, RMSE, R-peak F1 (Se / PPV)
and, with --update-readme, replaces the block between
  <!-- RESULTS:START -->  and  <!-- RESULTS:END -->   in README.md.

Aggregation: SNR improvement and RMSE = mean over records; Se, PPV = pooled (micro-averaged)
over records (sum TP/FP/FN first), F1 from those.
"""
import csv

from _common import NOISE_NAMES, ROOT, base_parser, out_dirs
import numpy as np

from src import metrics
from src.pipeline import METHODS as METHODS_ALL

METHODS = list(METHODS_ALL)


def load(path):
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in ("tp", "fp", "fn", "n_ref"):
            r[k] = int(r[k])
        for k in ("snr_target", "snr_in_eval", "snr_out", "d_snr", "rmse_mv"):
            r[k] = float(r[k]) if r[k] != "" else np.nan
    return rows


def main():
    p = base_parser(__doc__)
    p.add_argument("--update-readme", action="store_true")
    args = p.parse_args()
    res_dir, _ = out_dirs(args)
    src = res_dir / "metrics_per_record.csv"
    if not src.exists():
        raise SystemExit(f"{src} not found: run scripts/06_ecg_experiments.py first")
    rows = load(src)

    kinds = [k for k in ("bw", "em", "ma", "mains") if any(r["noise"] == k for r in rows)]
    snrs = sorted({r["snr_target"] for r in rows if r["noise"] != "none"}, reverse=True)
    n_rec = len({r["record"] for r in rows})

    summary = []
    for kind in kinds:
        for snr in snrs:
            for method in ["none"] + METHODS:
                sel = [r for r in rows if r["noise"] == kind and r["snr_target"] == snr and r["method"] == method]
                if not sel:
                    continue
                tp, fp, fn = (sum(r[k] for r in sel) for k in ("tp", "fp", "fn"))
                sc = metrics.peak_scores(tp, fp, fn)
                summary.append(dict(noise=kind, snr_in=snr, method=method,
                                    snr_out=np.mean([r["snr_out"] for r in sel]),
                                    d_snr=np.mean([r["d_snr"] for r in sel]),
                                    rmse_mv=np.mean([r["rmse_mv"] for r in sel]),
                                    se=sc["se"], ppv=sc["ppv"], f1=sc["f1"], n_records=len(sel)))
    with open(res_dir / "summary_by_condition.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)

    def get(kind, snr, method):
        return next(s for s in summary if s["noise"] == kind and s["snr_in"] == snr and s["method"] == method)

    def table(title, header, cell, methods):
        out = [f"**{title}**", "", "| Noise | Input SNR [dB] | " + " | ".join(header) + " |",
               "|---|---:|" + "|".join(["---:"] * len(header)) + "|"]
        for kind in kinds:
            for snr in snrs:
                out.append(f"| {NOISE_NAMES[kind]} | {snr:g} | " + " | ".join(cell(get(kind, snr, m)) for m in methods) + " |")
        return "\n".join(out)

    md = [
        f"_Mean over {n_rec} record(s); beat-detection scores pooled over records; "
        f"edges trimmed before scoring._",
        "",
        table("SNR improvement [dB] (higher is better)", METHODS, lambda s: f"{s['d_snr']:+.2f}", METHODS),
        "",
        table("RMSE vs clean record [mV] (lower is better)", ["no filter"] + METHODS,
              lambda s: f"{s['rmse_mv']:.3f}", ["none"] + METHODS),
        "",
        table("R-peak detection: Se / PPV [%]", ["no filter"] + METHODS,
              lambda s: f"{100 * s['se']:.1f} / {100 * s['ppv']:.1f}", ["none"] + METHODS),
    ]
    text = "\n".join(md) + "\n"
    (res_dir / "results_table.md").write_text(text)
    print(text)
    print(f"wrote {(res_dir / 'results_table.md').relative_to(ROOT)}")

    if args.update_readme:
        if args.synthetic:
            raise SystemExit("refusing to put synthetic-data results into the README")
        readme = ROOT / "README.md"
        s = readme.read_text()
        a, b = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
        if a not in s or b not in s:
            raise SystemExit("README markers not found")
        pre, rest = s.split(a)
        _, post = rest.split(b)
        readme.write_text(pre + a + "\n" + text + b + post)
        print("README.md updated")


if __name__ == "__main__":
    main()
