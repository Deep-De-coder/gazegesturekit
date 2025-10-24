from __future__ import annotations
import numpy as np
import cv2

LM = {"nose_tip": 1, "left_eye_outer": 33, "right_eye_outer": 263, "left_eye_inner": 133, "right_eye_inner": 362, "chin": 152}
MODEL_3D = np.array([[0.0,0.0,0.0], [-30.0,35.0,-30.0], [30.0,35.0,-30.0], [-10.0,35.0,-30.0], [10.0,35.0,-30.0], [0.0,-55.0,-15.0]], dtype=np.float32)

def estimate_head_pose(pts_norm: np.ndarray, frame_w: int, frame_h: int):
    img_pts = np.array([pts_norm[LM[k]] for k in ["nose_tip","left_eye_outer","right_eye_outer","left_eye_inner","right_eye_inner","chin"]], dtype=np.float32)
    img_pts[:,0] *= frame_w; img_pts[:,1] *= frame_h
    f = 0.9 * frame_w
    cam = np.array([[f,0,frame_w/2],[0,f,frame_h/2],[0,0,1]], dtype=np.float32)
    dist = np.zeros((4,1), dtype=np.float32)
    ok, rvec, tvec = cv2.solvePnP(MODEL_3D, img_pts, cam, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        R = np.eye(3, dtype=np.float32); yaw=pitch=0.0
    else:
        R,_ = cv2.Rodrigues(rvec)
        sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
        pitch = float(np.degrees(np.arctan2(-R[2,0], sy)))
        yaw   = float(np.degrees(np.arctan2(R[1,0], R[0,0])))
    return yaw, pitch, R.astype(np.float32), tvec.astype(np.float32) if ok else np.zeros((3,1), np.float32)
