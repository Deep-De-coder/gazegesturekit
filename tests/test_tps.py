import numpy as np
from gazegesturekit.calibrate.tps import fit_tps, apply_tps

def test_tps_roundtrip():
    src = np.array([[0,0],[1,0],[0,1],[1,1],[0.5,0.5]], dtype=np.float32)
    dst = np.array([[10,10],[110,10],[10,110],[110,110],[60,60]], dtype=np.float32)
    model = fit_tps(src, dst, reg=1e-3)
    out = apply_tps(model, src)
    assert np.allclose(out, dst, atol=1.5)
