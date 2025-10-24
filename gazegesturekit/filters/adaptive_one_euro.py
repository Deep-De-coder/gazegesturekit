from __future__ import annotations
from .one_euro import OneEuro

class AdaptiveOneEuro:
    """
    Confidence-aware wrapper: decreases smoothing when confidence is high,
    increases smoothing when low.
    """
    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.base_min = min_cutoff; self.base_beta = beta; self.d_cutoff=d_cutoff
        self.fx = OneEuro(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
        self.fy = OneEuro(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)

    def __call__(self, x, y, conf: float):
        # conf in [0..1]; lower conf -> higher min_cutoff (more smoothing)
        k = max(0.4, 1.2 - conf)  # 0.4..1.2
        self.fx.min_cutoff = self.base_min * k
        self.fy.min_cutoff = self.base_min * k
        self.fx.beta = self.base_beta
        self.fy.beta = self.base_beta
        return self.fx(x), self.fy(y)
