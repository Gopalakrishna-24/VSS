# services/tamper-detect/tamper_detect.py
"""CPU-only camera-tampering detector (SC-2 production path).

Non-semantic signal processing: no GPU, no VLM. Flags sudden occlusion /
defocus / spray by watching three per-frame signals collapse together.
"""
from collections import deque
import numpy as np
import cv2


def laplacian_variance(gray: np.ndarray) -> float:
    """Higher = sharper/more detail. Collapses toward 0 on a covered lens."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def edge_density(gray: np.ndarray) -> float:
    """Fraction of pixels on a Canny edge, 0..1. Drops to ~0 on a flat scene."""
    edges = cv2.Canny(gray, 100, 200)
    return float(np.count_nonzero(edges)) / float(edges.size)


def histogram_entropy(gray: np.ndarray) -> float:
    """Shannon entropy (bits, 0..8) of the brightness histogram.
    High when brightness values are spread out; ~0 when the frame is uniform."""
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    p = hist / max(hist.sum(), 1.0)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())

class TamperDetector:
    """Flags tampering when >= min_signals of the 3 signals stay below
    drop_ratio * baseline for `window` consecutive frames.

    The first `baseline_frames` frames are assumed clean and used to learn
    what 'normal' looks like (the median of each signal). After that, every
    frame is compared against that baseline.
    """

    def __init__(self, window: int = 8, baseline_frames: int = 30,
                 drop_ratio: float = 0.4, min_signals: int = 2):
        self.window = window
        self.baseline_frames = baseline_frames
        self.drop_ratio = drop_ratio
        self.min_signals = min_signals
        self._baseline_buf = {k: [] for k in ("lap", "edge", "ent")}
        self._baseline = None
        self._below = deque(maxlen=window)  # 1 if this frame looks tampered, else 0

    def signals(self, gray: np.ndarray) -> dict:
        return {
            "lap": laplacian_variance(gray),
            "edge": edge_density(gray),
            "ent": histogram_entropy(gray),
        }

    def update(self, gray: np.ndarray) -> bool:
        s = self.signals(gray)
        # Phase 1: still learning the baseline -> never flag.
        if self._baseline is None:
            for k in s:
                self._baseline_buf[k].append(s[k])
            if len(self._baseline_buf["lap"]) >= self.baseline_frames:
                self._baseline = {k: float(np.median(v))
                                  for k, v in self._baseline_buf.items()}
            return False
        # Phase 2: compare each signal to its baseline; count how many collapsed.
        tripped = sum(
            1 for k in s
            if s[k] < self.drop_ratio * max(self._baseline[k], 1e-6)
        )
        self._below.append(1 if tripped >= self.min_signals else 0)
        # Flag only if the whole window is tampered (sustained, not a blip).
        return len(self._below) == self.window and all(self._below)
