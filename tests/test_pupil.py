import numpy as np, cv2
from gazegesturekit.eye.pupil import estimate_pupil_center

def test_pupil_simple():
    img = np.full((80,120,3), 200, np.uint8)
    cv2.circle(img, (60,40), 12, (20,20,20), -1)
    c, conf = estimate_pupil_center(img, None)
    assert c is not None and conf >= 0.1
