from __future__ import annotations
import numpy as np, time
from .landmarks import FaceLandmarks
from .blink import blink_state
from ..filters.one_euro import OneEuro
from .saccade import EyeKinetics
from .headpose import estimate_head_pose

LEFT_IRIS=[468,469,470,471]; RIGHT_IRIS=[473,474,475,476]
LEFT_EYE_CORNERS=[33,133]; RIGHT_EYE_CORNERS=[362,263]

class GazeEstimator:
    def __init__(self, smooth=0.6, calibration=None, screen=(1280,720), head_comp=True, multiscreen=None):
        self.face = FaceLandmarks()
        self.fx = OneEuro(min_cutoff=1.0,beta=smooth); self.fy = OneEuro(min_cutoff=1.0,beta=smooth)
        self.cal = calibration; self.screen=screen
        self.kin = EyeKinetics()
        self._last = None
        self.head_comp = head_comp
        self.bias = np.array([0.0,0.0])  # drift correction (slow)
        self.multiscreen = multiscreen  # {"primary": {"w":..., "h":...}, "ext": {...}, "active":"primary"}

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
        # active screen
        scr = self.multiscreen[self.multiscreen["active"]] if self.multiscreen else {"w":self.screen[0],"h":self.screen[1]}
        w,h = scr["w"], scr["h"]
        if not self.cal:
            sx, sy = int(gx*w), int(gy*h)
            return sx, sy, 0.6, w, h
        import numpy as np
        wx=np.array(self.cal["wx"]); wy=np.array(self.cal["wy"])
        f=lambda W: W[0]*gx + W[1]*gy + W[2]*gx*gx + W[3]*gy*gy + W[4]*gx*gy + W[5]
        x,y = f(wx), f(wy)
        return int(max(0,min(w-1,x))), int(max(0,min(h-1,y))), 0.85, w, h

    def __call__(self, frame_bgr):
        h0,w0 = frame_bgr.shape[:2]
        pts = self.face(frame_bgr)
        if pts is None: return None

        # Head pose compensation (yaw/pitch)
        yaw, pitch = estimate_head_pose(pts, w0, h0) if self.head_comp else (0.0,0.0)

        gx, gy = self._raw_gaze(pts)
        sx, sy, conf, w, h = self._apply_cal(gx, gy)

        # Adjust by head pose (simple proportional)
        sx += int(yaw * 2.0); sy += int(-pitch * 2.0)

        # Apply slow drift bias
        sx = int(sx + self.bias[0]); sy = int(sy + self.bias[1])

        blink, wink, ear = blink_state(pts)
        t=time.time()
        if self._last is None: self._last=(sx,sy,t,0,0)
        lx,ly,lt,_,_ = self._last
        dt = max(t-lt,1e-6); dx=(sx-lx)/dt; dy=(sy-ly)/dt
        self._last=(sx,sy,t,dx,dy)

        # Kinetics
        self.kin.update(sx, sy, t)
        state = self.kin.state()
        fix_ms = self.kin.fixation_ms()

        return {"screen_xy":(sx,sy), "conf":conf, "blink":blink, "wink":wink,
                "fixation_ms":fix_ms, "dx":dx, "dy":dy, "state":state, "yaw":yaw, "pitch":pitch}

    def apply_click_bias(self, target_xy, observed_xy, rate=0.02):
        """
        Slowly learn bias after reliable click (move prediction toward actual click).
        """
        tgt = np.array(target_xy, dtype=float)
        obs = np.array(observed_xy, dtype=float)
        delta = obs - tgt
        self.bias = 0.98*self.bias + rate*delta