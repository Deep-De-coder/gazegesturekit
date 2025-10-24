from __future__ import annotations
import numpy as np

WRIST=0; THUMB_TIP=4; INDEX_TIP=8; MIDDLE_TIP=12; RING_TIP=16; PINKY_TIP=20

def fist(pts, thr=0.045):
    wrist = pts[WRIST,:2]
    tips = pts[[INDEX_TIP,MIDDLE_TIP,RING_TIP,PINKY_TIP],:2]
    d = np.mean(np.linalg.norm(tips - wrist, axis=1))
    return d < thr, float(min(1.0, (thr - d)/thr))

def thumbs_up(pts, dir_thr=0.03):
    # thumb extended, other fingers curled
    wrist = pts[WRIST,:2]
    thumb_d = np.linalg.norm(pts[THUMB_TIP,:2] - wrist)
    idx_d = np.linalg.norm(pts[INDEX_TIP,:2] - wrist)
    mid_d = np.linalg.norm(pts[MIDDLE_TIP,:2] - wrist)
    ring_d= np.linalg.norm(pts[RING_TIP,:2] - wrist)
    pink_d= np.linalg.norm(pts[PINKY_TIP,:2] - wrist)
    extended = thumb_d > (idx_d + mid_d + ring_d + pink_d)/4 + dir_thr
    curled   = idx_d < thumb_d and mid_d < thumb_d and ring_d < thumb_d
    return (extended and curled), 0.8

def twohand_zoom(l_pts, r_pts, prev_dist=None):
    # returns (is_zooming, scale (>1 zoom out/in), new_dist)
    l = l_pts[INDEX_TIP,:2]; r = r_pts[INDEX_TIP,:2]
    dist = float(np.linalg.norm(l - r))
    if prev_dist is None:
        return False, 1.0, dist
    if dist == 0: return False, 1.0, dist
    scale = dist/prev_dist
    is_zoom = abs(scale - 1.0) > 0.08
    return is_zoom, scale, dist
