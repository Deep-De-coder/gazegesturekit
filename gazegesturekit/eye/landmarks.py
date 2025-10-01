from __future__ import annotations
import mediapipe as mp
import numpy as np

class FaceLandmarks:
    def __init__(self, static_image_mode=False, max_num_faces=1):
        self.mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=static_image_mode,
                                                    refine_landmarks=True,
                                                    max_num_faces=max_num_faces)
    def __call__(self, frame_bgr):
        import cv2
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.mesh.process(rgb)
        if not res.multi_face_landmarks: return None
        lms = res.multi_face_landmarks[0]
        pts = np.array([(lm.x, lm.y) for lm in lms.landmark], dtype=np.float32)
        return pts  # normalized [0..1]
