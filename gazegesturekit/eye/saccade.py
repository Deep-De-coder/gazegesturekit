from __future__ import annotations
import math, time
from collections import deque
from typing import Deque, Tuple, Optional

class EyeKinetics:
    """
    Tracks gaze kinematics to detect saccades and fixations in real-time.
    """
    def __init__(self, max_ms: int = 600, saccade_vel: float = 500.0, fixation_vel: float = 80.0):
        self.window: Deque[Tuple[float,int,int]] = deque(maxlen=90)  # (t, x, y)
        self.saccade_vel = saccade_vel
        self.fixation_vel = fixation_vel
        self._last_state = "idle"
        self._last_change = time.time()

    def update(self, x:int, y:int, t: Optional[float]=None):
        t = t or time.time()
        self.window.append((t, x, y))

    def velocity(self) -> float:
        if len(self.window) < 2: return 0.0
        (t0,x0,y0) = self.window[0]
        (t1,x1,y1) = self.window[-1]
        dt = max(t1 - t0, 1e-6)
        dist = math.hypot(x1-x0, y1-y0)
        return dist / dt  # px/sec

    def state(self) -> str:
        v = self.velocity()
        if v >= self.saccade_vel:
            st = "saccade"
        elif v <= self.fixation_vel:
            st = "fixation"
        else:
            st = "pursuit"
        if st != self._last_state:
            self._last_state = st
            self._last_change = time.time()
        return st

    def fixation_ms(self) -> int:
        if self._last_state != "fixation": return 0
        return int((time.time() - self._last_change) * 1000)

    def centroid(self) -> Tuple[int,int]:
        if not self.window: return (0,0)
        xs = [x for _,x,_ in self.window]
        ys = [y for _,_,y in self.window]
        return int(sum(xs)/len(xs)), int(sum(ys)/len(ys))
