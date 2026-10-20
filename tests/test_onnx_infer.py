import numpy as np

from vision import infer
from vision.onnx_infer import OnnxYolo
from vision.render import render_strip
from src import io as ecgio


class _FakeSession:
    """Returns a hand-made YOLOv8-style output [1, 4+nc, 8400] so the decoder can be tested without a model file."""
    def __init__(self, rows):
        self.rows = rows

    class _In:  name = "images"
    def get_inputs(self): return [self._In()]

    def run(self, _, feed):
        x = feed["images"]
        assert x.shape == (1, 3, 640, 640) and x.dtype == np.float32 and 0.0 <= x.min() and x.max() <= 1.0
        out = np.zeros((1, 8, 8400), dtype=np.float32)
        for i, (cx, cy, w, h, cls, sc) in enumerate(self.rows):
            out[0, :4, i] = (cx, cy, w, h)
            out[0, 4 + cls, i] = sc
        return [out]


# letterboxed coordinates: a 640x256 strip sits at y = 192..448 inside the 640x640 input, x is unchanged
ROWS = [(160, 320, 64, 100, 0, 0.90),     # N at x 128..192
        (162, 320, 64, 100, 0, 0.80),     # near-duplicate: removed by NMS
        (480, 320, 64, 100, 2, 0.60),     # V at x 448..512
        (320, 320, 64, 100, 1, 0.10)]     # below the confidence threshold


def test_decoder_boxes_classes_nms_and_coordinates():
    m = OnnxYolo(session=_FakeSession(ROWS))
    res = m.predict_xyxy(np.full((256, 640, 3), 255, np.uint8), conf=0.25, iou=0.5)
    assert [(r[4], round(r[5], 2)) for r in res] == [(0, 0.9), (2, 0.6)]
    assert abs(res[0][0] - 128) < 1 and abs(res[0][2] - 192) < 1
    assert abs(res[1][0] - 448) < 1 and abs(res[1][2] - 512) < 1
    assert 0 <= res[0][1] < res[0][3] <= 256                     # y mapped back out of the letterbox padding


def test_detect_window_converts_pixels_to_seconds():
    m = OnnxYolo(session=_FakeSession(ROWS))
    r = ecgio.synthetic_record(duration_s=20, seed=1)
    fs = r["fs"]
    y = r["signal"][: int(4 * fs)]
    dets = infer.yolo_detect_window(m, y, fs, conf=0.25)
    strip = render_strip(y, fs)
    assert [d.label for d in dets] == ["N", "V"] and abs(dets[0].t0 - float(strip.px_to_time(128))) < 1e-6


def test_load_yolo_is_graceful(tmp_path):
    assert infer.load_yolo(str(tmp_path / "missing.onnx"))[0] is None
    bad = tmp_path / "bad.onnx"
    bad.write_bytes(b"not a model")
    model, msg = infer.load_yolo(str(bad))
    assert model is None and "ONNX" in msg


def test_demo_excerpt_script_and_bundled_source(tmp_path):
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location("mde", "scripts/make_demo_excerpt.py")
    mde = importlib.util.module_from_spec(spec); spec.loader.exec_module(mde)
    mde.main(["--synthetic", "--record", "200", "--seconds", "30", "--out", str(tmp_path)])
    z = np.load(tmp_path / "mitdb_200_30s.npz", allow_pickle=False)
    assert z["signal"].dtype == np.float32 and len(z["signal"]) == int(30 * float(z["fs"]))
    assert len(z["ann_sample"]) == len(z["ann_symbol"]) > 10 and str(z["record"]) == "200"
    assert (tmp_path / "SOURCE.txt").exists()
