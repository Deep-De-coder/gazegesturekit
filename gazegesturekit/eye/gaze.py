from __future__ import annotations
import numpy as np, time
from .landmarks import FaceLandmarks
from .blink import blink_state
from ..filters.one_euro import OneEuro

# Simple normalized gaze proxy: vector from eye center to iris area using FaceMesh indices
LEFT_IRIS=[468,469,470,471]; RIGHT_IRIS=[473,474,475,476]
LEFT_EYE_CORNERS=[33,133]; RIGHT_EYE_CORNERS=[362,263]

class GazeEstimator:
    def __init__(self, smooth=0.6, calibration=None, screen=(1280,720)):
        self.face = FaceLandmarks()
        self.fx = OneEuro(min_cutoff=1.0,beta=smooth); self.fy = OneEuro(min_cutoff=1.0,beta=smooth)
        self.cal = calibration; self.screen=screen
        self._last = None

    def _eye_center(self, pts, left=True):
        idx = LEFT_EYE_CORNERS if left else RIGHT_EYE_CORNERS
        return pts[idx].mean(axis=0)

    def _iris_center(self, pts, left=True):
        idx = LEFT_IRIS if left else RIGHT_IRIS
        return pts[idx].mean(axis=0)

    def _raw_gaze(self, pts):
        lc, li = self._eye_center(pts, True), self._iris_center(pts, True)
        rc, ri = self._eye_center(pts, False), self._iris_center(pts, False)
        c = (lc + rc) / 2.0; i = (li + ri) / 2.0  # normalized [0..1]
        v = i - c
        gx = self.fx(c[0] - v[0]); gy = self.fy(c[1] - v[1])
        return float(gx), float(gy)

    def _apply_cal(self, gx, gy):
        if not self.cal:  # map normalized to screen directly
            sx, sy = int(gx*self.screen[0]), int(gy*self.screen[1]); return sx, sy, 0.6
        import numpy as np
        wx=np.array(self.cal["wx"]); wy=np.array(self.cal["wy"]); w=self.cal["w"]; h=self.cal["h"]
        f=lambda W: W[0]*gx + W[1]*gy + W[2]*gx*gx + W[3]*gy*gy + W[4]*gx*gy + W[5]
        x,y = f(wx), f(wy)
        return int(max(0,min(w-1,x))), int(max(0,min(h-1,y))), 0.85

    def __call__(self, frame_bgr):
        pts = self.face(frame_bgr)
        if pts is None: return None
        gx, gy = self._raw_gaze(pts)
        sx, sy, conf = self._apply_cal(gx, gy)
        blink, wink, ear = blink_state(pts)
        t=time.time()
        if self._last is None: self._last=(sx,sy,t,0,0)
        lx,ly,lt,_,_ = self._last
        dt = max(t-lt,1e-6); dx=(sx-lx)/dt; dy=(sy-ly)/dt
        fixation_ms = int(1000*dt) if abs(dx)<80 and abs(dy)<80 else 0
        self._last=(sx,sy,t,dx,dy)
        return {"screen_xy":(sx,sy), "conf":conf, "blink":blink, "wink":wink, "fixation_ms":fixation_ms, "dx":dx, "dy":dy}
