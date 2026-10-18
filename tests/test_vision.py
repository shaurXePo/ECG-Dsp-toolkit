import numpy as np

from vision import infer
from vision.labels import AAMI_MAP, CLASSES, beat_boxes, yolo_to_pixels
from vision.make_dataset import DS1, DS2, main as make_main, plan_splits
from vision.render import render_strip
from src import io as ecgio


def _strip(n_s=4.0):
    r = ecgio.synthetic_record(duration_s=20, seed=1)
    fs = r["fs"]
    s0 = int(5 * fs)
    return r, render_strip(r["signal"][s0:s0 + int(n_s * fs)], fs), s0


def test_render_mapping_roundtrip():
    r, st, _ = _strip()
    assert st.image.size == (640, 256)
    assert st.xpix[0] == 0 and abs(st.xpix[-1] - 639) < 1e-9
    t = np.array([0.0, 1.0, 3.9])
    assert np.allclose(st.px_to_time(st.time_to_px(t)), t)


def test_box_centre_is_the_beat_time():
    r, st, s0 = _strip()
    m = (r["ann_sample"] >= s0 + 100) & (r["ann_sample"] < s0 + st.n - 100)
    boxes = beat_boxes(st, s0, r["ann_sample"][m], r["ann_symbol"][m])
    assert len(boxes) == int(m.sum()) > 0
    for (c, cx, cy, w, h), s in zip(boxes, r["ann_sample"][m]):
        assert 0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w < 1 and 0 < h <= 1
        t_box = st.px_to_time(cx * st.size[0])
        assert abs(t_box - (s - s0) / st.fs) <= 1.5 / st.fs


def test_aami_mapping_and_classes():
    assert AAMI_MAP["L"] == "N" and AAMI_MAP["V"] == "V" and AAMI_MAP["/"] == "Q"
    assert CLASSES == ["N", "S", "V", "F"]
    r, st, s0 = _strip()
    assert beat_boxes(st, s0, [s0 + 200], ["+"]) == []          # non-beat symbol ignored


def test_splits_are_inter_patient():
    assert not set(DS1) & set(DS2) and len(DS1) == 22 and len(DS2) == 22
    s = plan_splits(DS1 + DS2)
    assert not set(s["train"]) & set(s["val"]) and not set(s["train"]) & set(s["test"])
    assert not set(s["val"]) & set(s["test"])
    assert set(s["train"]) | set(s["val"]) == set(DS1) and set(s["test"]) == set(DS2)


def test_make_dataset_synthetic(tmp_path):
    s = make_main(["--synthetic", "--duration-s", "60", "--out", str(tmp_path), "--noise-snrs", "6",
                   "--noise-kinds", "bw"])
    assert all(s["images"][k] > 0 for k in ("train", "val", "test"))
    f = next((tmp_path / "labels" / "test").glob("*.txt"))
    for line in f.read_text().strip().splitlines():
        parts = line.split()
        assert len(parts) == 5 and int(parts[0]) < 4 and all(0 <= float(v) <= 1 for v in parts[1:])
    assert (tmp_path / "dataset.yaml").exists()
    # clean and noisy variants both exist
    names = [p.name for p in (tmp_path / "images" / "train").glob("*.png")]
    assert any("_clean_" in n for n in names) and any("_bw6dB_" in n for n in names)


def test_classical_detector_and_merge():
    r = ecgio.synthetic_record(duration_s=30, seed=2)
    d = infer.classical_detect(r["signal"], r["fs"])
    assert abs(len(d) - len(r["ann_sample"])) <= 2
    dd = [infer.Detection("N", 0.9, 1.0, 1.4), infer.Detection("V", 0.5, 1.02, 1.42), infer.Detection("N", 0.8, 2.0, 2.4)]
    m = infer.merge(dd)
    assert len(m) == 2 and m[0].label == "N"


class _T:                                     # tiny stand-in for a torch tensor row
    def __init__(self, v): self.v = v
    def tolist(self): return list(self.v) if hasattr(self.v, "__iter__") else self.v
    def __getitem__(self, i): return self.v[i] if hasattr(self.v, "__iter__") else self.v


class _Box:
    def __init__(self, x0, x1, cls, conf):
        self.xyxy, self.cls, self.conf = [_T([x0, 5.0, x1, 200.0])], [cls], [conf]


class _FakeYolo:
    """Pretends to be a trained model: 'detects' the QRS with the classical detector, in pixel coordinates."""
    names = {0: "N", 1: "S", 2: "V", 3: "F"}

    def predict(self, img, imgsz, conf, iou, verbose):
        from vision.render import IMG_W
        arr = np.asarray(img)[..., 0].astype(float)          # rendered strip, black trace on white
        cols = np.where(arr.min(axis=0) < 128)[0]
        # use the topmost trace pixel per column as a crude 'signal' and find R peaks (< 40 px from top)
        top = np.array([np.where(arr[:, c] < 128)[0].min() for c in cols])
        idx = cols[top < 40]
        groups = np.split(idx, np.where(np.diff(idx) > 5)[0] + 1) if len(idx) else []
        boxes = [_Box(g.mean() - 30, g.mean() + 30, 0, 0.9) for g in groups if len(g)]
        return [type("R", (), {"boxes": boxes})()]


def test_yolo_inference_path_with_fake_model():
    r = ecgio.synthetic_record(duration_s=30, seed=3)
    fs = r["fs"]
    y = r["signal"][: int(12 * fs)]
    dets = infer.yolo_detect_long(_FakeYolo(), y, fs)
    truth = r["ann_sample"][r["ann_sample"] < len(y)] / fs
    assert all(d.label == "N" for d in dets)
    # every truth beat (not at the very edge) has a detection centre within 40 ms
    for tb in truth[(truth > 0.3) & (truth < 11.7)]:
        assert min(abs(d.t - tb) for d in dets) < 0.04
    assert len(dets) <= len(truth) + 1                        # merging removed window duplicates
