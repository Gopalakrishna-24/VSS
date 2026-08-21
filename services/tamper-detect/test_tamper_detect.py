# services/tamper-detect/test_tamper_detect.py
import numpy as np
from tamper_detect import (
    laplacian_variance, edge_density, histogram_entropy #TamperDetector
)

def _textured(seed=0):
    """A random, detailed grayscale image = 'normal camera view'."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(120, 160), dtype=np.uint8)

def _black():
    """A solid black image = 'lens fully covered'."""
    return np.zeros((120, 160), dtype=np.uint8)

def test_laplacian_variance_drops_on_black():
    assert laplacian_variance(_textured()) > laplacian_variance(_black())

def test_edge_density_zero_on_black():
    assert edge_density(_black()) == 0.0
    assert edge_density(_textured()) > 0.0

def test_histogram_entropy_low_on_uniform():
    assert histogram_entropy(_black()) < 0.1
    assert histogram_entropy(_textured()) > 4.0
