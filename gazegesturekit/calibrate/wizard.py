from __future__ import annotations
import cv2, json, time, numpy as np
from pathlib import Path
from .tps import fit_tps, apply_tps

class Calibrator:
    def __init__(self, points:int=5, save_path: str|Path=None, mapping_type: str="tps", screen_id:str="primary"):
        self.points=points; self.save_path=Path(save_path or ".ggk_calibration.json")
        self.samples=[]  # (feat_x, feat_y) -> (screen_x, screen_y)
        self.mapping_type = mapping_type
        self.screen_id = screen_id

    def _targets(self, w:int, h:int):
        if self.points==5:
            return [(w//2,h//2),(50,50),(w-50,50),(50,h-50),(w-50,h-50)]
        xs=[50,w//2,w-50]; ys=[50,h//2,h-50]; return [(x,y) for y in ys for x in xs]

    def fit(self):
        src = np.array([[a,b] for (a,b,_,_) in self.samples], dtype=np.float32)
        dst = np.array([[sx,sy] for (_,_,sx,sy) in self.samples], dtype=np.float32)
        if self.mapping_type=="tps":
            model = fit_tps(src, dst, reg=1e-3)
            return {"type":"tps", "model": {k:(v.tolist() if hasattr(v, "tolist") else v) for k,v in model.items()}}
        # fallback: poly2
        A=[]; bx=[]; by=[]
        for (gx,gy,_,_),(_,_,sx,sy) in zip(self.samples, self.samples):
            A.append([gx,gy,gx*gx,gy*gy,gx*gy,1.0]); bx.append(sx); by.append(sy)
        A=np.asarray(A); bx=np.asarray(bx); by=np.asarray(by)
        wx,_ ,_, _= np.linalg.lstsq(A,bx,rcond=None)
        wy,_ ,_, _= np.linalg.lstsq(A,by,rcond=None)
        return {"type":"poly2", "wx":wx.tolist(), "wy":wy.tolist()}

    def run(self, get_feature_xy, screen_size=(1280,720)):
        w,h=screen_size
        for (tx,ty) in self._targets(w,h):
            for _ in range(20):
                raw = get_feature_xy()  # returns (fx,fy) feature space (e.g., ray_x, ray_y)
                if raw is None: continue
                fx,fy = raw
                self.samples.append((fx,fy,tx,ty))
                time.sleep(0.02)
        mapping = self.fit()
        params={"screen": {"id": self.screen_id, "w":w,"h":h}, "mapping": mapping}
        # load existing to support multi-screen
        cfg={}
        if self.save_path.exists():
            try: cfg=json.loads(self.save_path.read_text())
            except: cfg={}
        cfg[self.screen_id] = params
        self.save_path.write_text(json.dumps(cfg,indent=2))
        return params
