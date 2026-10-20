"""
onnx_infer.py -- run an exported YOLOv8 detector with onnxruntime only (no PyTorch, no Ultralytics).

Why: PyTorch/Ultralytics are too heavy for small free hosts (Streamlit Community Cloud has a few GB of RAM).
The model is exported once on your laptop (python -m vision.export_onnx) and the ~12 MB .onnx file is committed.

Assumes the standard YOLOv8 detect head: output [1, 4 + n_classes, n_anchors] (or its transpose), boxes as
(cx, cy, w, h) in input pixels, class scores already sigmoid-ed, no objectness. The export script's --check option
compares this decoder with Ultralytics on real test images, so verify before deploying.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_NAMES = {0: "N", 1: "S", 2: "V", 3: "F"}


def _letterbox(img: np.ndarray, size: int):
    """Ultralytics-style letterbox: scale to fit, pad with 114 to size x size. Returns (array, ratio, left, top)."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    if (nw, nh) != (w, h):
        img = np.asarray(Image.fromarray(img).resize((nw, nh), Image.BILINEAR))
    dw, dh = (size - nw) / 2, (size - nh) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    out = np.full((size, size, 3), 114, dtype=np.uint8)
    out[top:top + nh, left:left + nw] = img
    return out, r, left, top


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    order = np.argsort(-scores)
    keep = []
    while len(order):
        i = int(order[0])
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        xx0 = np.maximum(boxes[i, 0], boxes[rest, 0]); yy0 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx1 = np.minimum(boxes[i, 2], boxes[rest, 2]); yy1 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx1 - xx0, 0, None) * np.clip(yy1 - yy0, 0, None)
        a_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        a_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        order = rest[inter / (a_i + a_r - inter + 1e-9) <= iou_thr]
    return keep


class OnnxYolo:
    def __init__(self, path=None, names=None, session=None, imgsz: int = 640):
        if session is None:
            import onnxruntime as ort                       # lazy: only needed when a real file is loaded
            session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.session = session
        self.input_name = session.get_inputs()[0].name
        self.imgsz = imgsz
        self.names = names or self._names_from_meta(session) or dict(DEFAULT_NAMES)

    @staticmethod
    def _names_from_meta(session):
        try:
            raw = session.get_modelmeta().custom_metadata_map.get("names")
            d = ast.literal_eval(raw) if raw else None
            return {int(k): str(v) for k, v in d.items()} if d else None
        except Exception:                                    # noqa: BLE001
            return None

    def predict_xyxy(self, image, conf: float = 0.25, iou: float = 0.5, max_det: int = 300):
        """image: PIL image or HxWx3 uint8 RGB array. Returns [(x0, y0, x1, y1, class_index, confidence)] in image pixels."""
        arr = np.asarray(image.convert("RGB") if hasattr(image, "convert") else image, dtype=np.uint8)
        h0, w0 = arr.shape[:2]
        lb, r, left, top = _letterbox(arr, self.imgsz)
        x = (lb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        out = np.asarray(self.session.run(None, {self.input_name: x})[0])[0]
        if out.shape[0] < out.shape[1]:                      # [4+nc, N] -> [N, 4+nc]
            out = out.T
        boxes_cxcywh, scores = out[:, :4], out[:, 4:]
        cls = scores.argmax(1)
        cf = scores.max(1)
        m = cf >= conf
        if not m.any():
            return []
        b, cls, cf = boxes_cxcywh[m], cls[m], cf[m]
        xyxy = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1)
        keep = _nms(xyxy + cls[:, None] * 4096.0, cf, iou)[:max_det]       # class offset = per-class NMS
        res = []
        for i in keep:
            x0, y0, x1, y1 = xyxy[i]
            x0, x1 = np.clip([(x0 - left) / r, (x1 - left) / r], 0, w0)
            y0, y1 = np.clip([(y0 - top) / r, (y1 - top) / r], 0, h0)
            res.append((float(x0), float(y0), float(x1), float(y1), int(cls[i]), float(cf[i])))
        return res
