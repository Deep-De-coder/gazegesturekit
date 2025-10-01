from __future__ import annotations
import cv2, json, time, numpy as np
from pathlib import Path

class Calibrator:
    def __init__(self, points:int=5, save_path: str|Path=None):
        self.points=points; self.save_path=Path(save_path or ".ggk_calibration.json")
        self.samples=[]  # tuples of (raw_gaze_x, raw_gaze_y, screen_x, screen_y)

    def _targets(self, w:int, h:int):
        if self.points==5:
            return [(w//2,h//2),(50,50),(w-50,50),(50,h-50),(w-50,h-50)]
        else:
            xs=[50,w//2,w-50]; ys=[50,h//2,h-50]; return [(x,y) for y in ys for x in xs]

    def fit(self):
        # 2D polynomial (x,y, x^2, y^2, xy) -> screen x,y
        A=[]; bx=[]; by=[]
        for gx,gy,sx,sy in self.samples:
            A.append([gx,gy,gx*gx,gy*gy,gx*gy,1.0]); bx.append(sx); by.append(sy)
        A=np.asarray(A); bx=np.asarray(bx); by=np.asarray(by)
        wx,_ ,_, _= np.linalg.lstsq(A,bx,rcond=None)
        wy,_ ,_, _= np.linalg.lstsq(A,by,rcond=None)
        return wx, wy

    def map(self, raw_xy, w, h, wx, wy):
        gx,gy=raw_xy; f=lambda W: W[0]*gx + W[1]*gy + W[2]*gx*gx + W[3]*gy*gy + W[4]*gx*gy + W[5]
        x,y = f(wx), f(wy)
        return max(0,min(w-1,int(x))), max(0,min(h-1,int(y)))

    def run(self, get_raw_gaze, screen_size=(1280,720)):
        w,h=screen_size
        for (tx,ty) in self._targets(w,h):
            for _ in range(20):  # brief dwell
                raw = get_raw_gaze()  # returns (gx,gy) normalized ~[0,1]
                if raw is None: continue
                gx,gy = raw
                self.samples.append((gx,gy,tx,ty))
                time.sleep(0.02)
        wx,wy=self.fit()
        params={"w":w,"h":h,"wx":wx.tolist(),"wy":wy.tolist()}
        self.save_path.write_text(json.dumps(params,indent=2))
        return params
