from gazegesturekit.eye.saccade import EyeKinetics
import time
def test_fixation_and_saccade():
    kin = EyeKinetics()
    t = time.time()
    for i in range(5):
        kin.update(100,100,t+i*0.05)
    assert kin.state() in ("fixation","pursuit")
    for i in range(5):
        kin.update(100+100*i,100,t+0.3+i*0.01)
    assert kin.state() in ("saccade","pursuit")
