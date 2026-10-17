"""
Download exactly the PhysioNet files this project needs (about 25 MB total):

  MIT-BIH Arrhythmia Database (mitdb): records 100, 101, 119, 200, 207  (.dat .hea .atr)
  MIT-BIH Noise Stress Test Database (nstdb): noise records bw, em, ma   (.dat .hea)

Usage:   python data/download_data.py [--data-dir DIR]

Files land in  <data-dir>/mitdb/  and  <data-dir>/nstdb/.  Already-downloaded files are skipped.
If the automatic download fails (firewall etc.) download the same files by hand from
  https://physionet.org/content/mitdb/   and   https://physionet.org/content/nstdb/
and put them in those two folders.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import io as ecgio  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=ecgio.DATA_DIR)
    args = ap.parse_args()

    import wfdb

    jobs = [("mitdb", [f"{r}.{e}" for r in ecgio.MITDB_RECORDS for e in ("dat", "hea", "atr")]),
            ("nstdb", [f"{r}.{e}" for r in ecgio.NSTDB_NOISE.values() for e in ("dat", "hea")])]
    for db, files in jobs:
        dl_dir = args.data_dir / db
        dl_dir.mkdir(parents=True, exist_ok=True)
        todo = [f for f in files if not (dl_dir / f).exists()]
        print(f"{db}: {len(files) - len(todo)} present, {len(todo)} to download")
        if todo:
            try:
                wfdb.dl_files(db, str(dl_dir), todo)
            except Exception as exc:          # network / firewall / PhysioNet layout change
                sys.exit(f"download of {db} failed: {type(exc).__name__}: {str(exc)[:200]}\n"
                         f"Download these files by hand from https://physionet.org/content/{db}/ "
                         f"into {dl_dir}:\n  " + ", ".join(todo))

    print("data_available:", ecgio.data_available(args.data_dir))
    if not ecgio.data_available(args.data_dir):
        sys.exit("some files are missing; see the docstring for manual download")


if __name__ == "__main__":
    main()
