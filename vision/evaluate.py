"""
evaluate.py -- beat-level check of a detector against expert annotations (pure NumPy, no ML imports).

A detection counts as matching a reference beat if its centre is within tol_s (default 150 ms, the AAMI EC57 tolerance
the toolkit already uses), greedy one-to-one. Then:
    * location: sensitivity = matched / reference beats, PPV = matched / detections,
    * class (only if the detector outputs classes): confusion matrix with a 'missed' column.
"""
from __future__ import annotations

import numpy as np

MISSED = "missed"


def match_detections(ref_t, ref_cls, det_t, det_cls, tol_s: float = 0.15, classes=("N", "S", "V", "F")) -> dict:
    ref_t = np.asarray(ref_t, dtype=float)
    det_t = np.asarray(det_t, dtype=float)
    ref_cls, det_cls = list(ref_cls), list(det_cls)
    label_aware = bool(det_cls) and not all(c == "R" for c in det_cls)

    used = np.zeros(len(det_t), dtype=bool)
    matrix = {r: {c: 0 for c in list(classes) + [MISSED]} for r in classes}
    n_matched = 0
    for i in np.argsort(ref_t):
        pred = MISSED
        if len(det_t):
            d = np.abs(det_t - ref_t[i])
            d[used] = np.inf
            j = int(np.argmin(d))
            if d[j] <= tol_s:
                used[j] = True
                n_matched += 1
                pred = det_cls[j] if det_cls[j] in classes else MISSED
        if ref_cls[i] in matrix:
            matrix[ref_cls[i]][pred] += 1

    n_ref, n_det = len(ref_t), len(det_t)
    out = {
        "n_ref": n_ref, "n_det": n_det, "matched": n_matched,
        "false_positives": int((~used).sum()), "missed": n_ref - n_matched,
        "sens_loc": n_matched / n_ref if n_ref else float("nan"),
        "ppv_loc": n_matched / n_det if n_det else float("nan"),
        "label_aware": label_aware, "matrix": matrix, "per_class": {},
    }
    for c in classes:
        row = matrix[c]
        n = sum(row.values())
        right = row[c]
        out["per_class"][c] = {"reference": n, "correct": right,
                               "wrong_class": n - right - row[MISSED], "missed": row[MISSED],
                               "sensitivity": right / n if n else float("nan")}
    return out
