import numpy as np
from gazegesturekit.hand.gestures import pinch, palm_open, pointing

def fake_pts(scale=1.0):
    pts = np.zeros((21,3),dtype=float)
    pts[0,:2] = [0.5,0.5]   # wrist
    pts[4,:2] = [0.52,0.5]  # thumb_tip
    pts[8,:2] = [0.53,0.5]  # index_tip
    pts[12,:2] = [0.51,0.5] # middle_tip
    pts[16,:2] = [0.505,0.5]
    pts[20,:2] = [0.50,0.5]
    return pts

def test_pinch_detects_close():
    ok, conf = pinch(fake_pts(), thr=0.06)
    assert ok and conf>0

def test_pointing_rule():
    ok, conf = pointing(fake_pts())
    assert isinstance(ok, (bool, np.bool_))
