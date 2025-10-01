from __future__ import annotations
import numpy as np

THUMB_TIP=4; INDEX_TIP=8; MIDDLE_TIP=12; RING_TIP=16; PINKY_TIP=20; WRIST=0

def pinch(pts, thr=0.06):
    d = np.linalg.norm(pts[THUMB_TIP,:2] - pts[INDEX_TIP,:2])
    return d < thr, float(max(0.0, 1.0 - d/thr))

def palm_open(pts, thr=0.12):
    # open palm has larger average distance of fingertips to wrist
    wrist = pts[WRIST,:2]
    tips = pts[[INDEX_TIP,MIDDLE_TIP,RING_TIP,PINKY_TIP],:2]
    s = np.mean(np.linalg.norm(tips - wrist, axis=1))
    return s > thr, float(min(1.0, s/thr))

def pointing(pts, curl_thr=0.04):
    # simplistic: index extended vs middle ring curled (tip closer to wrist)
    wrist=pts[WRIST,:2]
    idx = np.linalg.norm(pts[INDEX_TIP,:2]-wrist)
    mid = np.linalg.norm(pts[MIDDLE_TIP,:2]-wrist)
    ring= np.linalg.norm(pts[RING_TIP,:2]-wrist)
    return (idx > mid+curl_thr and idx > ring+curl_thr), 0.7

def classify(hand):
    pts=hand["pts"]; handed=hand["handedness"]
    for fn, name in [(pinch,"pinch"), (palm_open,"palm"), (pointing,"point")]:
        ok,conf = fn(pts)
        if ok: return {"gesture":name, "conf":conf, "handedness":handed}
    return {"gesture":None, "conf":0.0, "handedness":handed}
