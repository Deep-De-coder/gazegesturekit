from __future__ import annotations
import numpy as np
import cv2

# Minimal 3D face model: select a few stable landmarks (nose tip, eye corners)
# Using MediaPipe FaceMesh indices (approx):
LM = {
  "nose_tip": 1,
  "left_eye_outer": 33,
  "right_eye_outer": 263,
  "left_eye_inner": 133,
  "right_eye_inner": 362,
  "chin": 152,
}

# Fake 3D template (arbitrary units) roughly in a frontal face coord system
MODEL_3D = np.array([
  [0.0,   0.0,   0.0],   # nose_tip
  [-30.0,  35.0, -30.0], # left_eye_outer
  [ 30.0,  35.0, -30.0], # right_eye_outer
  [-10.0,  35.0, -30.0], # left_eye_inner
  [ 10.0,  35.0, -30.0], # right_eye_inner
  [ 0.0,  -55.0, -15.0], # chin
], dtype=np.float32)

def estimate_head_pose(pts_norm: np.ndarray, frame_w: int, frame_h: int):
    """pts_norm: (468,2) normalized [0,1] FaceMesh points -> returns yaw, pitch in degrees."""
    img_pts = np.array([
        pts_norm[LM["nose_tip"]],
        pts_norm[LM["left_eye_outer"]],
        pts_norm[LM["right_eye_outer"]],
        pts_norm[LM["left_eye_inner"]],
        pts_norm[LM["right_eye_inner"]],
        pts_norm[LM["chin"]],
    ], dtype=np.float32)
    img_pts[:,0] *= frame_w
    img_pts[:,1] *= frame_h

    # Camera intrinsics heuristic
    f = 0.9 * frame_w
    cam = np.array([[f,0,frame_w/2],[0,f,frame_h/2],[0,0,1]], dtype=np.float32)
    dist = np.zeros((4,1), dtype=np.float32)

    ok, rvec, tvec = cv2.solvePnP(MODEL_3D, img_pts, cam, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0.0, 0.0
    R,_ = cv2.Rodrigues(rvec)
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    pitch = np.degrees(np.arctan2(-R[2,0], sy))
    yaw   = np.degrees(np.arctan2(R[1,0], R[0,0]))
    return float(yaw), float(pitch)
