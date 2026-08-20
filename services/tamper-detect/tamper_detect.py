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
