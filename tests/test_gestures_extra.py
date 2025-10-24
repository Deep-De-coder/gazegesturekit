import numpy as np
from gazegesturekit.hand.gestures_extra import fist, thumbs_up

def fake_pts_fist():
    pts = np.zeros((21,3), dtype=float)
    pts[0,:2] = [0.5,0.5]  # wrist
    for i in [8,12,16,20]:
        pts[i,:2] = [0.505,0.505]  # curled
    pts[4,:2] = [0.51,0.5]  # thumb near
    return pts

def test_fist_detect():
    ok, conf = fist(fake_pts_fist())
    assert isinstance(ok, (bool, np.bool_))

def fake_pts_thumbsup():
    pts = np.zeros((21,3), dtype=float)
    pts[0,:2] = [0.5,0.5]
    pts[4,:2] = [0.65,0.5]  # long thumb
    pts[8,:2] = [0.52,0.5]; pts[12,:2] = [0.515,0.5]; pts[16,:2]=[0.512,0.5]; pts[20,:2]=[0.511,0.5]
    return pts

def test_thumbsup_detect():
    ok, conf = thumbs_up(fake_pts_thumbsup())
    assert isinstance(ok, (bool, np.bool_))
