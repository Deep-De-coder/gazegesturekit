import numpy as np

# Indices approximate eye contour in FaceMesh (left eye example):
LEFT_EYE = [33, 160, 158, 133, 153, 144]  # (outer corners + top/bottom)
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def ear(eye_pts):
    A = np.linalg.norm(eye_pts[1] - eye_pts[5])
    B = np.linalg.norm(eye_pts[2] - eye_pts[4])
    C = np.linalg.norm(eye_pts[0] - eye_pts[3])
    return (A + B) / (2.0 * C)

def blink_state(pts, thr=0.22):
    le = pts[LEFT_EYE]; re = pts[RIGHT_EYE]
    ear_l, ear_r = ear(le), ear(re)
    blink = ear_l<thr and ear_r<thr
    wink = "left" if ear_l<thr and ear_r>=thr else ("right" if ear_r<thr and ear_l>=thr else None)
    return blink, wink, (ear_l+ear_r)/2.0
