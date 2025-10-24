from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import numpy as np

@dataclass
class CameraIntrinsics:
    fx: float; fy: float; cx: float; cy: float

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx],
                         [0, self.fy, self.cy],
                         [0,      0,      1]], dtype=np.float32)

    @property
    def Kinv(self) -> np.ndarray:
        return np.linalg.inv(self.K)

    @staticmethod
    def heuristic(width:int, height:int) -> "CameraIntrinsics":
        f = 0.9 * width
        return CameraIntrinsics(fx=f, fy=f, cx=width/2, cy=height/2)

def load_intrinsics(path: str|Path|None, width:int, height:int) -> CameraIntrinsics:
    if path and Path(path).exists():
        data = json.loads(Path(path).read_text())
        return CameraIntrinsics(**data)
    return CameraIntrinsics.heuristic(width, height)

def save_intrinsics(path: str|Path, K: CameraIntrinsics):
    Path(path).write_text(json.dumps(asdict(K), indent=2))

def pixel_to_camera_ray(u: float, v: float, Kinv: np.ndarray) -> np.ndarray:
    """Return normalized 3D ray dir in camera coords given pixel (u,v)."""
    p = np.array([u, v, 1.0], dtype=np.float32)
    ray = Kinv @ p
    ray = ray / np.linalg.norm(ray)
    return ray
