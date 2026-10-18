"""
ECG-DSP Lab -- filter an ECG with the toolkit's FIR/IIR filters, then detect beats with
YOLO (if trained weights exist) or the classical R-peak detector.

    streamlit run app/streamlit_app.py

Educational demo. NOT a medical device.
"""
from __future__ import annotations

import sys
from pathlib import Path

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import io as ecgio                       # noqa: E402
from src import noise as ecgnoise                 # noqa: E402
from src import pipeline                          # noqa: E402
from vision import evaluate as veval                # noqa: E402
from vision import infer                           # noqa: E402
from vision.labels import AAMI_MAP                 # noqa: E402
from vision.render import render_strip            # noqa: E402

st.set_page_config(page_title="ECG-DSP Lab", page_icon=":anatomical_heart:", layout="wide")

COLORS = {"N": "#2ECC71", "S": "#F39C12", "V": "#E74C3C", "F": "#9B59B6", "Q": "#95A5A6", "R": "#1ABC9C"}
NOISE_NAMES = {"bw": "baseline wander", "em": "electrode motion", "ma": "muscle artefact"}
DEFAULT_WEIGHTS = "runs/detect/ecg_yolo/weights/best.pt"


# --------------------------------------------------------------------------- cached helpers
@st.cache_resource
def get_bank(fs: float, mains: float | None):
    return pipeline.build_filter_bank(fs, mains)


@st.cache_resource
def get_yolo(weights: str):
    return infer.load_yolo(weights)


@st.cache_data
def load_source(kind: str, name: str, duration_s: float, upload_bytes: bytes | None, upload_fs: float):
    """Returns dict(signal, fs, ann_sample, ann_symbol, label)."""
    if kind == "Synthetic demo ECG":
        r = ecgio.synthetic_record("synthetic", duration_s=duration_s, seed=7)
        return dict(signal=r["signal"], fs=r["fs"], ann_sample=r["ann_sample"], ann_symbol=r["ann_symbol"],
                    label="synthetic")
    if kind == "MIT-BIH record":
        r = ecgio.load_record(name, duration_s=duration_s)
        return dict(signal=r["signal"], fs=r["fs"], ann_sample=r["ann_sample"], ann_symbol=r["ann_symbol"],
                    label=f"MIT-BIH {name}")
    arr = np.loadtxt(__import__("io").BytesIO(upload_bytes), delimiter=",", ndmin=2)
    x = arr[:, -1] if arr.shape[1] > 1 else arr[:, 0]
    n = min(len(x), int(duration_s * upload_fs))
    return dict(signal=x[:n].astype(float), fs=float(upload_fs), ann_sample=np.array([], dtype=int),
                ann_symbol=np.array([]), label="uploaded CSV")


@st.cache_data
def add_noise(x: np.ndarray, fs: float, kind: str, snr: float, use_real: bool) -> np.ndarray:
    n = len(x)
    nz = None
    if use_real:
        try:
            nz = ecgio.load_noise(kind, n_samples=n, offset=100_000)[0]
        except Exception:                          # noqa: BLE001
            nz = None
    if nz is None:
        nz = ecgio.synthetic_noise_record(kind, n, fs, seed=3)[0]
    return ecgnoise.mix_at_snr(x, nz, snr)[0]


@st.cache_data
def run_filter(x: np.ndarray, fs: float, mains: float | None, method: str) -> np.ndarray:
    if method == "None (mean removed)":
        return np.asarray(x, float) - np.mean(x)
    return get_bank(fs, mains).apply(method, x)


def mit_records() -> list[str]:
    try:
        return sorted(p.stem for p in (ecgio.DATA_DIR / "mitdb").glob("*.dat"))
    except Exception:                              # noqa: BLE001
        return []



@st.cache_data(show_spinner="Running the detector on the whole excerpt...")
def detect_all(y: np.ndarray, fs: float, use_yolo: bool, weights: str, conf: float):
    """Detections over the whole filtered excerpt as plain tuples (label, conf, t0, t1) + the detector actually used."""
    if use_yolo:
        model, _ = get_yolo(weights)
        if model is not None:
            return [(d.label, d.conf, d.t0, d.t1) for d in infer.yolo_detect_long(model, y, fs, conf=conf)], "YOLO"
    return [(d.label, d.conf, d.t0, d.t1) for d in infer.classical_detect(y, fs)], "classical"


def split_status(src_kind: str, rec: str) -> tuple[str, str]:
    """('ok' | 'warn' | 'info', message): was this record used to train / validate the loaded model?"""
    if src_kind != "MIT-BIH record":
        return "info", "Not an MIT-BIH record, so there is no train/test bookkeeping for it."
    try:
        sp = json.loads((ROOT / "datasets/ecg_yolo/dataset_summary.json").read_text())["splits"]
    except Exception:                              # noqa: BLE001
        return "info", "datasets/ecg_yolo/dataset_summary.json not found: cannot tell whether this record was used for training."
    if rec in sp.get("train", []):
        return "warn", f"Record {rec} was a TRAINING record of this model: the result below is optimistic and not a valid test."
    if rec in sp.get("val", []):
        return "warn", f"Record {rec} was the VALIDATION record (used to pick the checkpoint): not a clean test."
    if rec in sp.get("test", []):
        return "ok", f"Record {rec} is a TEST record (DS2) that the model never saw: this is a fair check."
    return "info", f"Record {rec} was not used in this run (neither training, validation nor test): unseen, but not part of the reported test set."

# --------------------------------------------------------------------------- sidebar
st.title("ECG-DSP Lab")
st.caption("FIR / IIR filtering from scratch + YOLO beat detection. Educational demo, not a medical device.")

with st.sidebar:
    st.header("1. Signal")
    recs = mit_records()
    sources = ["Synthetic demo ECG"] + (["MIT-BIH record"] if recs else []) + ["Upload CSV"]
    src_kind = st.radio("Source", sources)
    rec_name, up_bytes, up_fs = "", None, 360.0
    if src_kind == "MIT-BIH record":
        rec_name = st.selectbox("Record", recs)
    if src_kind == "Upload CSV":
        up = st.file_uploader("CSV: one column of mV (or time,mV)", type=["csv", "txt"])
        up_fs = st.number_input("Sampling rate [Hz]", 50.0, 2000.0, 360.0, 10.0)
        up_bytes = up.getvalue() if up else None
    dur = st.slider("Analyse first [s]", 30, 300, 60, 10)

    st.header("2. Noise (optional)")
    add_n = st.checkbox("Add noise")
    n_kind = st.selectbox("Type", list(NOISE_NAMES), format_func=NOISE_NAMES.get, disabled=not add_n)
    n_snr = st.slider("Input SNR [dB]", 0, 24, 6, 2, disabled=not add_n)

    st.header("3. Filter")
    method = st.selectbox("Method", ["None (mean removed)"] + list(pipeline.METHODS), index=4)
    mains_opt = st.radio("Mains notch (IIR)", ["60 Hz (US / MIT-BIH)", "50 Hz (India / EU)", "Off"])
    mains = {"60 Hz (US / MIT-BIH)": 60.0, "50 Hz (India / EU)": 50.0, "Off": None}[mains_opt]

    st.header("4. Detector")
    det_kind = st.radio("Detector", ["YOLO (trained weights)", "Classical R-peak (Pan-Tompkins style)"])
    weights = st.text_input("YOLO weights", DEFAULT_WEIGHTS, disabled=det_kind.startswith("Classical"))
    conf = st.slider("Confidence", 0.05, 0.95, 0.25, 0.05, disabled=det_kind.startswith("Classical"))

    st.header("5. View")
    view_s = st.select_slider("Window shown [s]", [6, 8, 10, 15, 20], value=10)

if src_kind == "Upload CSV" and up_bytes is None:
    st.info("Upload a CSV in the sidebar, or pick another source.")
    st.stop()

# --------------------------------------------------------------------------- pipeline
src = load_source(src_kind, rec_name, float(dur), up_bytes, float(up_fs))
fs, x_raw = src["fs"], src["signal"]
x_in = add_noise(x_raw, fs, n_kind, float(n_snr), bool(recs)) if add_n else x_raw
y = run_filter(x_in, fs, mains, method)

total_s = len(y) / fs
view_s = min(view_s, int(total_s))
t_start = st.slider("Start [s]", 0.0, max(total_s - view_s, 0.1), 0.0, 0.5)
i0, i1 = int(t_start * fs), int((t_start + view_s) * fs)
seg = y[i0:i1]
t = np.arange(len(seg)) / fs + t_start

# detection on the visible window (needs >= 4 s)
dets, det_msg = [], ""
if det_kind.startswith("YOLO"):
    model, msg = get_yolo(weights)
    if model is None:
        det_msg = f"YOLO unavailable ({msg}). Showing the classical detector instead."
        dets = infer.classical_detect(seg, fs)
    else:
        dets = infer.yolo_detect_long(model, seg, fs, conf=conf)
        det_msg = msg
else:
    dets = infer.classical_detect(seg, fs)

# --------------------------------------------------------------------------- plot
fig = go.Figure()
if add_n or method != "None (mean removed)":
    fig.add_trace(go.Scatter(x=t, y=(x_in[i0:i1] - np.mean(x_in)), name="input", line=dict(color="#6E7681", width=1),
                             opacity=0.6))
fig.add_trace(go.Scatter(x=t, y=seg, name=f"filtered: {method}", line=dict(color="#4C9AFF", width=1.6)))
lo, hi = float(np.min(seg)), float(np.max(seg))
pad = 0.08 * (hi - lo + 1e-9)
for d in dets:
    a, b = d.t0 + t_start, d.t1 + t_start
    sel = (t >= a) & (t <= b)
    ylo = float(seg[sel].min()) - pad if sel.any() else lo
    yhi = float(seg[sel].max()) + pad if sel.any() else hi
    c = COLORS.get(d.label, "#FFFFFF")
    fig.add_shape(type="rect", x0=a, x1=b, y0=ylo, y1=yhi, line=dict(color=c, width=2), fillcolor=c, opacity=0.15)
    fig.add_annotation(x=(a + b) / 2, y=yhi, text=f"{d.label} {d.conf:.2f}" if d.label != "R" else "R",
                       showarrow=False, yshift=10, font=dict(color=c, size=12))
ann, sym = src["ann_sample"], src["ann_symbol"]
if len(ann) and st.checkbox("Show reference annotations (MIT-BIH / synthetic)", value=False):
    m = (ann >= i0) & (ann < i1)
    fig.add_trace(go.Scatter(x=ann[m] / fs, y=[hi + 2.5 * pad] * int(m.sum()), mode="text",
                             text=[AAMI_MAP.get(str(s), "?") for s in sym[m]], name="reference",
                             textfont=dict(color="#C9D1D9")))
fig.update_layout(height=430, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Time [s]", yaxis_title="mV",
                  legend=dict(orientation="h", y=1.08), template="plotly_dark")
st.plotly_chart(fig, width="stretch")
if det_msg:
    st.caption(det_msg)

# --------------------------------------------------------------------------- numbers
c1, c2, c3 = st.columns(3)
c1.metric("Beats detected (window)", len(dets))
if len(dets) >= 3:
    rr = np.diff([d.t for d in dets])
    c2.metric("Heart rate (median RR)", f"{60.0 / float(np.median(rr)):.0f} bpm")
else:
    c2.metric("Heart rate", "n/a")
counts = {}
for d in dets:
    counts[d.label] = counts.get(d.label, 0) + 1
c3.metric("Classes", ", ".join(f"{k}:{v}" for k, v in sorted(counts.items())) or "-")

with st.expander("Model check against the reference annotations"):
    if not len(ann):
        st.write("This source has no reference annotations (uploaded CSV), so there is nothing to compare with.")
    else:
        kind, msg = split_status(src_kind, rec_name)
        {"ok": st.success, "warn": st.warning, "info": st.info}[kind](msg)
        if st.checkbox("Run the detector on the whole analysed excerpt and compare"):
            dets_all, used = detect_all(y, fs, det_kind.startswith("YOLO"), weights, float(conf))
            trim = 5.0 if total_s > 20 else 0.0                      # drop filter edge transients
            r_t = ann / fs
            keep = (r_t >= trim) & (r_t <= total_s - trim)
            ref_c = [AAMI_MAP.get(str(s)) for s in sym[keep]]
            ref_t = [t_ for t_, c in zip(r_t[keep], ref_c) if c]
            ref_c = [c for c in ref_c if c]
            det = [d for d in dets_all if trim <= 0.5 * (d[2] + d[3]) <= total_s - trim]
            res = veval.match_detections(ref_t, ref_c, [0.5 * (d[2] + d[3]) for d in det], [d[0] for d in det])
            st.caption(f"Detector used: {used}. Scored {len(ref_t)} reference beats in {total_s - 2 * trim:.0f} s "
                       f"(match tolerance 150 ms, edges trimmed {trim:.0f} s).")
            a, b, c_ = st.columns(3)
            a.metric("Beat detection sensitivity", f"{100 * res['sens_loc']:.1f} %")
            b.metric("Precision (PPV)", f"{100 * res['ppv_loc']:.1f} %")
            c_.metric("False alarms", res["false_positives"])
            if res["label_aware"]:
                rows = [{"class": k, "reference beats": v["reference"], "found, right class": v["correct"],
                         "found, wrong class": v["wrong_class"], "missed": v["missed"],
                         "class sensitivity": "n/a" if not v["reference"] else f"{100 * v['sensitivity']:.0f} %"}
                        for k, v in res["per_class"].items() if v["reference"]]
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
                cm = pd.DataFrame(res["matrix"]).T
                cm = cm.loc[[i for i in cm.index if cm.loc[i].sum()]]
                st.caption("Confusion matrix: rows = reference class, columns = what the detector said ('missed' = no box).")
                st.dataframe(cm, width="stretch")
            else:
                st.caption("The classical detector only locates beats (it labels all of them 'R'), so no class table. "
                           "Switch the detector to YOLO to see classes.")
            st.caption("Few distinct beats make per-class numbers noisy. Classes the model never saw in training "
                       "(S and F in the pilot run) cannot be found.")

with st.expander("What the detector sees (rendered strip, first 4 s of the window)"):
    if len(seg) >= int(4 * fs):
        st.image(render_strip(seg[: int(4 * fs)], fs).image, width="stretch")
    else:
        st.write("Window shorter than 4 s.")
with st.expander("About"):
    st.markdown(
        "Pipeline: raw ECG -> toolkit filters (whole excerpt, so filters see seconds of context) -> render 4 s "
        "strips -> YOLO boxes (class + confidence) -> drawn back on the time axis.  \n"
        "Without trained weights the classical Pan-Tompkins-style detector is used.  \n"
        "Classes follow AAMI: N normal, S supraventricular, V ventricular, F fusion.  \n"
        "**Educational demo. Not for diagnosis.**")
