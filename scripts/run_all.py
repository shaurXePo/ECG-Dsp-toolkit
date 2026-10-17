"""
Run the whole pipeline in order:  01 FFT -> 02 spectral -> 03 FIR -> 04 IIR -> 05 analysis ->
06 ECG experiments -> 07 results table.

    python scripts/run_all.py                       # real data (run data/download_data.py first)
    python scripts/run_all.py --synthetic           # offline smoke test, outputs in results_synthetic/
    python scripts/run_all.py --mains-hz 50         # 50 Hz notch (Indian recordings)
    python scripts/run_all.py --full                # 30-min records in step 6 (slower)

Extra flags are passed to the scripts that understand them.
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMMON = ("--data-dir", "--mains-hz", "--seed")
STEPS = [
    ("01_fft_benchmark.py", set(COMMON) | {"--synthetic"}),
    ("02_spectral_analysis.py", set(COMMON) | {"--synthetic"}),
    ("03_fir_design.py", set(COMMON) | {"--synthetic"}),
    ("04_iir_design.py", set(COMMON) | {"--synthetic"}),
    ("05_filter_analysis.py", set(COMMON) | {"--synthetic"}),
    ("06_ecg_experiments.py", set(COMMON) | {"--synthetic", "--records", "--snrs", "--duration", "--full",
                                              "--trim-s", "--with-mains", "--fig-record", "--fig-snr"}),
    ("07_results_table.py", set(COMMON) | {"--synthetic", "--update-readme"}),
]


def filter_args(argv, allowed):
    """Keep only flags this script knows, together with their values."""
    out, i = [], 0
    while i < len(argv):
        a = argv[i]
        if a in allowed:
            out.append(a)
            i += 1
            while i < len(argv) and not argv[i].startswith("--"):
                out.append(argv[i])
                i += 1
        else:
            i += 1
            while i < len(argv) and not argv[i].startswith("--"):
                i += 1
    return out


def main():
    argv = sys.argv[1:]
    for script, allowed in STEPS:
        print(f"\n=== {script} ===", flush=True)
        cmd = [sys.executable, str(HERE / script)] + filter_args(argv, allowed)
        r = subprocess.run(cmd)
        if r.returncode != 0:
            sys.exit(f"{script} failed (exit {r.returncode})")
    print("\nAll steps finished.")


if __name__ == "__main__":
    main()
