"""
render.py -- turn a (filtered) ECG window into an image a YOLO model can read.

The mapping between samples and pixels is kept (Strip.xpix / Strip.ypix), so labels can be
generated exactly and detected boxes can be converted back to seconds.

Design choices (change them in one place):
  * white background, black trace, no grid: nothing on the image except the waveform;
  * per-window amplitude normalisation (robust, 99.9th percentile of |x - median|), so the
    trace always fills the strip. Absolute mV is NOT preserved; pass fixed_half_range_mv to keep it;
  * 640 x 256 px for a 4 s window -> 160 px per second (a QRS of ~0.1 s is ~16 px wide).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

IMG_W, IMG_H = 640, 256
MARGIN_PX = 14


@dataclass
class Strip:
    image: Image.Image          # RGB, IMG_W x IMG_H
    xpix: np.ndarray            # pixel column of every sample
    ypix: np.ndarray            # pixel row of every sample
    fs: float
    n: int

    @property
    def size(self) -> tuple[int, int]:
        return self.image.size

    def px_to_time(self, px) -> np.ndarray:
        """Pixel column -> seconds from the start of the window."""
        w = self.image.size[0]
        return np.asarray(px, dtype=float) * (self.n - 1) / (self.fs * (w - 1))

    def time_to_px(self, t) -> np.ndarray:
        w = self.image.size[0]
        return np.asarray(t, dtype=float) * self.fs * (w - 1) / (self.n - 1)


def render_strip(x, fs: float, width: int = IMG_W, height: int = IMG_H, line_px: int = 2,
                 fixed_half_range_mv: float | None = None) -> Strip:
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        raise ValueError("need at least 2 samples")
    dev = x - np.median(x)
    if fixed_half_range_mv is None:
        half = float(np.percentile(np.abs(dev), 99.9)) * 1.1
    else:
        half = float(fixed_half_range_mv)
    half = max(half, 1e-9)
    yn = np.clip(dev / half, -1.0, 1.0)
    ypix = height / 2.0 - yn * (height / 2.0 - MARGIN_PX)
    xpix = np.arange(n) * (width - 1) / (n - 1)

    img = Image.new("RGB", (width, height), (255, 255, 255))
    ImageDraw.Draw(img).line(list(zip(xpix.tolist(), ypix.tolist())), fill=(0, 0, 0),
                             width=line_px, joint="curve")
    return Strip(img, xpix, ypix, float(fs), n)
