from __future__ import annotations
import numpy as np

def _U(r):
    r = np.where(r==0, 1e-12, r)
    return (r**2) * np.log(r)

def fit_tps(src: np.ndarray, dst: np.ndarray, reg: float=1e-3):
    """
    src: (N,2) features (e.g., ray_x, ray_y)
    dst: (N,2) targets (screen x,y)
    returns dict with weights for x and y
    """
    N = src.shape[0]
    K = _U(np.sqrt(((src[:,None,:]-src[None,:,:])**2).sum(-1)))
    P = np.concatenate([np.ones((N,1)), src], axis=1)  # [1, x, y]
    O = np.zeros((3,3))
    L = np.block([[K + reg*np.eye(N), P],
                  [P.T, O]])
    Vx = np.concatenate([dst[:,0], np.zeros(3)])
    Vy = np.concatenate([dst[:,1], np.zeros(3)])
    w_x = np.linalg.solve(L, Vx)
    w_y = np.linalg.solve(L, Vy)
    return {"ctrl": src, "w_x": w_x, "w_y": w_y}

def apply_tps(model, pts: np.ndarray):
    ctrl = model["ctrl"]
    K = _U(np.sqrt(((pts[:,None,:]-ctrl[None,:,:])**2).sum(-1)))
    P = np.concatenate([np.ones((pts.shape[0],1)), pts], axis=1)
    wx, wy = model["w_x"], model["w_y"]
    N = ctrl.shape[0]
    a_x, a_y = wx[N:], wy[N:]  # affine terms (3,)
    yx = K @ wx[:N] + P @ a_x
    yy = K @ wy[:N] + P @ a_y
    return np.stack([yx, yy], axis=1)
