from __future__ import annotations
import math, time
from collections import deque
from typing import Deque, Tuple, Optional

class FixationDetector:
    """
    Hybrid I-VT (velocity threshold) + I-DT (dispersion threshold).
    """
    def __init__(self, vt_thresh: float=120.0, dt_ms:int=120, disp_px:int=45, max_hist:int=60):
        self.vt_thresh = vt_thresh
        self.dt_ms = dt_ms
        self.disp_px = disp_px
        self.buf: Deque[Tuple[float,int,int]] = deque(maxlen=max_hist)
        self._state = "free"
        self._start = None

    def update(self, x:int, y:int, t: Optional[float]=None):
        t = t or time.time()
        self.buf.append((t,x,y))

    def _velocity(self) -> float:
        if len(self.buf) < 2: return 0.0
        t0,x0,y0 = self.buf[0]
        t1,x1,y1 = self.buf[-1]
        dt = max(t1 - t0, 1e-6)
        dist = math.hypot(x1-x0, y1-y0)
        return dist / dt

    def _dispersion(self) -> float:
        xs = [x for _,x,_ in self.buf]
        ys = [y for _,_,y in self.buf]
        return (max(xs)-min(xs) + max(ys)-min(ys)) / 2.0

    def state(self) -> str:
        v = self._velocity()
        d = self._dispersion()
        t_now = time.time()
        if v < self.vt_thresh and d < self.disp_px:
            if self._state != "fixation":
                self._state = "fixation"; self._start = t_now
        else:
            self._state = "free"; self._start = None
        return self._state

    def fixation_ms(self) -> int:
        return int((time.time() - self._start)*1000) if self._start else 0

    def centroid(self) -> tuple[int,int]:
        if not self.buf: return (0,0)
        xs = [x for _,x,_ in self.buf]; ys = [y for _,_,y in self.buf]
        return int(sum(xs)/len(xs)), int(sum(ys)/len(ys))
