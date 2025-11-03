from __future__ import annotations
import numpy as np, time, cv2
from .landmarks import FaceLandmarks
from .blink import blink_state
from ..filters.adaptive_one_euro import AdaptiveOneEuro
from .fixation import FixationDetector
from .headpose import estimate_head_pose
from .pupil import estimate_pupil_center
from .camera_model import load_intrinsics, pixel_to_camera_ray

LEFT_EYE_IDX=[33,133]; RIGHT_EYE_IDX=[362,263]  # corners for rough ROI

def _roi_from_landmarks(frame, pts, eye_idx, scale=1.8):
    h,w = frame.shape[:2]
    a = (pts[eye_idx[0]] * [w,h]).astype(int)
    b = (pts[eye_idx[1]] * [w,h]).astype(int)
    cx,cy = ((a+b)//2)
    rad = int(np.linalg.norm(a-b) * scale * 0.5)
    x0,y0 = max(0,cx-rad), max(0,cy-rad); x1,y1 = min(w,cx+rad), min(h,cy+rad)
    roi = frame[y0:y1, x0:x1].copy()
    return roi, (x0,y0,cx,cy)

class GazeEstimator:
    def __init__(self, smooth=0.6, calibration=None, screen=(1280,720), intrinsics_path:str|None=None, face_lock:bool=True):
        self.faces = FaceLandmarks(max_num_faces=2)
        self.smoother = AdaptiveOneEuro(min_cutoff=1.0, beta=smooth)
        self.fix = FixationDetector()
        self.cal = calibration  # dict of screens or one-screen
        self.screen=screen
        self.bias = np.array([0.0,0.0], dtype=np.float32)
        self.face_lock = face_lock
        self.lock_idx = 0
        self._last = None
        self._K = None; self._Kinv = None; self._screen_id = "primary"
        self._intrinsics_path = intrinsics_path

    def _ensure_intrinsics(self, frame_w, frame_h):
        if self._K is not None: return
        from .camera_model import load_intrinsics
        K = load_intrinsics(self._intrinsics_path, frame_w, frame_h)
        self._K = K.K; self._Kinv = K.Kinv

    def _active_screen(self):
        # choose by provided calibration else fallback
        if isinstance(self.cal, dict) and self._screen_id in self.cal:
            meta = self.cal[self._screen_id]["screen"]
            return meta["w"], meta["h"]
        return self.screen

    def get_raw_feature(self, frame_bgr):
        """
        Extract raw gaze feature coordinates (ray_x, ray_y) before calibration mapping.
        Returns (fx, fy) or None if face/pupils not detected.
        """
        h0,w0 = frame_bgr.shape[:2]
        self._ensure_intrinsics(w0, h0)

        faces = self.faces(frame_bgr)
        if not faces: return None
        face = faces[min(self.lock_idx, len(faces)-1)] if self.face_lock else faces[0]
        pts = face["pts"]

        # Eye ROIs and pupil centers
        left_roi, (lx0,ly0,lcx,lcy) = _roi_from_landmarks(frame_bgr, pts, LEFT_EYE_IDX)
        right_roi,(rx0,ry0,rcx,rcy) = _roi_from_landmarks(frame_bgr, pts, RIGHT_EYE_IDX)

        lp, lc = estimate_pupil_center(left_roi, None)
        rp, rc = estimate_pupil_center(right_roi, None)

        if lp is None or rp is None:
            return None

        # Convert local pupil coords to image pixel coords
        lux, luy = lx0 + lp[0], ly0 + lp[1]
        rux, ruy = rx0 + rp[0], ry0 + rp[1]
        u = float((lux + rux) / 2.0); v = float((luy + ruy) / 2.0)

        # Ray in camera coords
        ray = pixel_to_camera_ray(u, v, self._Kinv)

        # Feature space for calibration: take (x/z, y/z) from ray direction
        feat_xy = (float(ray[0]/max(1e-6,ray[2])), float(ray[1]/max(1e-6,ray[2])))
        return feat_xy

    def _map_feature_to_screen(self, feat_xy):
        # feat_xy := (ray_x, ray_y)
        if isinstance(self.cal, dict) and self._screen_id in self.cal:
            mapping = self.cal[self._screen_id]["mapping"]
        else:
            mapping = self.cal
        w,h = self._active_screen()
        if mapping and mapping.get("type") == "tps":
            from ..calibrate.tps import apply_tps
            model = {k:np.array(v, dtype=np.float32) if isinstance(v, list) else v for k,v in mapping["model"].items()}
            xy = apply_tps(model, np.array([feat_xy], dtype=np.float32))[0]
            x,y = int(np.clip(xy[0], 0, w-1)), int(np.clip(xy[1], 0, h-1))
            return x,y,0.9
        elif mapping and mapping.get("type") == "poly2":
            wx = np.array(mapping["wx"], dtype=np.float32); wy = np.array(mapping["wy"], dtype=np.float32)
            gx,gy = feat_xy
            f=lambda W: W[0]*gx + W[1]*gy + W[2]*gx*gx + W[3]*gy*gy + W[4]*gx*gy + W[5]
            x,y = f(wx), f(wy)
            return int(np.clip(x,0,w-1)), int(np.clip(y,0,h-1)), 0.85
        else:
            gx,gy = feat_xy
            return int(np.clip(gx*w,0,w-1)), int(np.clip(gy*h,0,h-1)), 0.6

    def __call__(self, frame_bgr):
        h0,w0 = frame_bgr.shape[:2]
        self._ensure_intrinsics(w0, h0)

        faces = self.faces(frame_bgr)
        if not faces: return None
        face = faces[min(self.lock_idx, len(faces)-1)] if self.face_lock else faces[0]
        pts = face["pts"]

        # Head pose (for diagnostics; TPS handles mapping)
        yaw, pitch, R, tvec = estimate_head_pose(pts, w0, h0)

        # Eye ROIs and pupil centers
        left_roi, (lx0,ly0,lcx,lcy) = _roi_from_landmarks(frame_bgr, pts, LEFT_EYE_IDX)
        right_roi,(rx0,ry0,rcx,rcy) = _roi_from_landmarks(frame_bgr, pts, RIGHT_EYE_IDX)

        lp, lc = estimate_pupil_center(left_roi, None)
        rp, rc = estimate_pupil_center(right_roi, None)
        conf = float((lc + rc) / 2.0)

        if lp is None or rp is None:
            return None

        # Convert local pupil coords to image pixel coords
        lux, luy = lx0 + lp[0], ly0 + lp[1]
        rux, ruy = rx0 + rp[0], ry0 + rp[1]
        u = float((lux + rux) / 2.0); v = float((luy + ruy) / 2.0)

        # Ray in camera coords
        ray = pixel_to_camera_ray(u, v, self._Kinv)

        # Feature space for calibration: take (x/z, y/z) from ray direction
        feat_xy = (float(ray[0]/max(1e-6,ray[2])), float(ray[1]/max(1e-6,ray[2])))

        # Map to screen via calibration model
        sx, sy, mconf = self._map_feature_to_screen(feat_xy)

        # Apply drift bias (EMA-updated elsewhere)
        sx = int(sx + self.bias[0]); sy = int(sy + self.bias[1])

        # Blink/wink & fixation
        blink, wink, ear = blink_state(pts)
        self.fix.update(sx, sy)
        _ = self.fix.state()
        fix_ms = self.fix.fixation_ms()

        # Adaptive smoothing
        sx, sy = self.smoother(sx, sy, conf=min(1.0, (conf+mconf)/2.0))

        # Velocities
        t=time.time()
        if self._last is None: self._last=(sx,sy,t,0,0)
        lx,ly,lt,_,_ = self._last
        dt = max(t-lt,1e-6); dx=(sx-lx)/dt; dy=(sy-ly)/dt
        self._last=(sx,sy,t,dx,dy)

        return {"screen_xy":(int(sx),int(sy)), "conf":float(min(1.0,(conf+mconf)/2.0)),
                "blink":blink, "wink":wink, "fixation_ms":fix_ms, "dx":dx, "dy":dy,
                "yaw":yaw, "pitch":pitch}