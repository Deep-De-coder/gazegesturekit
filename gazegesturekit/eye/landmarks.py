from __future__ import annotations
import mediapipe as mp
import numpy as np
import cv2

class FaceLandmarks:
    def __init__(self, static_image_mode=False, max_num_faces=2):
        self.mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=static_image_mode,
                                                    refine_landmarks=True,
                                                    max_num_faces=max_num_faces)

    def __call__(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.mesh.process(rgb)
        if not res.multi_face_landmarks: return []
        faces=[]
        h,w = frame_bgr.shape[:2]
        for lms in res.multi_face_landmarks:
            pts = np.array([(lm.x, lm.y) for lm in lms.landmark], dtype=np.float32)
            # bbox center & pseudo-conf (area)
            xs = pts[:,0]*w; ys = pts[:,1]*h
            x0,x1 = xs.min(), xs.max(); y0,y1 = ys.min(), ys.max()
            area = (x1-x0) * (y1-y0)
            faces.append({"pts": pts, "bbox": (x0,y0,x1,y1), "score": float(area)})
        # sort: largest area (closest face) first
        faces.sort(key=lambda f: f["score"], reverse=True)
        return faces
